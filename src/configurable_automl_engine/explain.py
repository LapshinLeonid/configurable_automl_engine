"""Explainability of the winner pipeline (permutation importance, optional
SHAP).

``explain_permutation_importance`` measures how much the model's score
degrades when one input feature is shuffled; ``explain_shap`` computes
per-row Shapley values when the optional ``shap`` extra is installed. Both
work on the *raw* (pre-pipeline) columns of the training data through the
trainer's ``predict``, so one score per original feature comes out without
knowing the pipeline internals. The post-preprocessing view is available
when the caller asks for it: the pipeline matrix columns are named by
``preprocessor.get_feature_names_out()`` (``cat__<col>`` / ``num__<col>`` /
``remainder__<col>``), and one-hot / hashing components keep the encoded
source column in the name.

Score semantics: the scorer is resolved from the result's
``metric_user`` (custom metrics must be registered in this process via
``register_custom_metrics``) and returns raw optimizer-semantic values;
``importances_mean`` is the mean score decrease under permutation, and
the ``baseline_score`` is converted to user semantics with
``to_user_value`` (positive RMSE/MAE, plain R²).

The importance values may be negative on held-out rows (a permutation
can *improve* the metric by chance) — they are reported as computed,
rounded to six decimal places for readability.
"""

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from configurable_automl_engine.common.datetime_features import (
    expand_datetime_columns,
)
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.metrics import (
    get_scorer_object,
    to_user_value,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from configurable_automl_engine.experiment.result import ExperimentResult
    from configurable_automl_engine.trainer import ModelTrainer

__all__ = [
    "DEFAULT_N_REPEATS",
    "DEFAULT_SAMPLE_ROWS",
    "SHAP_PACKAGE",
    "ExplainNotAvailableError",
    "FeatureImportance",
    "aggregate_to_original_features",
    "explain_permutation_importance",
    "explain_shap",
    "render_importance_markdown",
]

_LOG = logging.getLogger(__name__)

#: Default number of shuffles per feature.
DEFAULT_N_REPEATS = 5

#: Default row budget of one importance computation (the OOM guard).
DEFAULT_SAMPLE_ROWS = 5000

#: The optional SHAP package (the ``[shap]`` extra).
SHAP_PACKAGE = "shap"

#: The number of decimal places of the rendered / stored importance values.
_IMPORTANCE_DECIMALS = 6


class ExplainNotAvailableError(RuntimeError):
    """The requested explainability computation cannot be served.

    Raised when the model has no pipeline (not fitted), the data is
    empty or has no feature columns, or SHAP is requested but not
    installed (the message names the ``[shap]`` extra).
    """


@dataclass(frozen=True)
class FeatureImportance:
    """One feature's importance of the winner pipeline.

    Attributes:
        feature: The feature name (an original data column for
            permutation importance; a pipeline column name for the
            post-preprocessing SHAP view).
        importance_mean: The mean importance over the repeats (rounded
            to :data:`_IMPORTANCE_DECIMALS` decimal places; a non-finite
            value is sanitized to ``0.0`` with a warning).
        importance_std: The standard deviation over the repeats (same
            rounding; ``None`` when not computed, e.g. for SHAP).
    """

    feature: str
    importance_mean: float
    importance_std: float | None = None


@dataclass(frozen=True)
class ShapFeatureImportance:
    """The structured SHAP result of one computation.

    Attributes:
        importances: The per-feature SHAP importance, sorted by the
            descending mean absolute value.
        post_preprocessing: ``True`` — the features are the pipeline
            matrix columns (``get_feature_names_out`` names); ``False`` —
            they are the original data columns (the SHAP values were
            aggregated over the pipeline's encoding expansion).
        baseline_prediction: The mean model prediction on the scored
            rows (the base value the SHAP contributions explain).
    """

    importances: tuple[FeatureImportance, ...]
    post_preprocessing: bool
    baseline_prediction: float


@dataclass(frozen=True)
class PermutationImportanceResult:
    """The structured result of one permutation-importance computation.

    Attributes:
        importances: The per-feature importance, sorted by the
            descending mean.
        baseline_score: The unshuffled scorer value, in user semantics
            (positive RMSE/MAE, plain R²).
        metric_user: The metric name the importance was measured with.
        n_rows: The number of data rows actually used (after the
            sampling of :data:`DEFAULT_SAMPLE_ROWS`).
        n_features: The number of features scored.
        data_semantics: What the scored rows are: ``"train"`` (the full
            training data the pipeline was fit on) or ``"holdout"``
            (the held-out rows; the pipeline never saw them during
            training). The final artifact is always fit on all data,
            so ``"train"`` rows are in-sample for it.
    """

    importances: tuple[FeatureImportance, ...]
    baseline_score: float
    metric_user: str
    n_rows: int
    n_features: int
    data_semantics: str


# --------------------------------------------------------------------------- #
#  Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _require_pipeline(trainer: ModelTrainer) -> Any:
    """The fitted pipeline of ``trainer`` or the not-available error."""
    pipeline = trainer.pipeline
    if pipeline is None:
        raise ExplainNotAvailableError(
            "The model has no fitted pipeline: explain() requires a "
            "trained model (call fit() or run the experiment first)."
        )
    return pipeline


def _sanitized(value: float, label: str) -> float:
    """Sanitize one importance value: non-finite values become ``0.0``.

    A NaN/±inf importance carries no information (it would break the
    sorting and the consumers); it is replaced with ``0.0`` and reported
    with a WARNING instead of failing the whole computation.
    """
    number = float(value)
    if not math.isfinite(number):
        _LOG.warning(
            "Feature importance of %r is non-finite (%r) — replaced with 0.0",
            label,
            value,
        )
        return 0.0
    return round(number, _IMPORTANCE_DECIMALS)


def _sample_rows(X: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Sample ``X`` to ``max_rows`` rows (deterministically, seeded).

    The row budget keeps ``n_repeats × n_features`` prediction calls
    tractable on large data; the subsample is drawn without replacement
    with a fixed seed so repeated calls are reproducible.
    """
    if len(X) <= max_rows:
        return X
    _LOG.info(
        "Importance computed on a random subsample: %d of %d rows",
        max_rows,
        len(X),
    )
    return X.sample(n=max_rows, random_state=0)


def _shap_feature_names(trainer: ModelTrainer, matrix: np.ndarray) -> list[str]:
    """The names of the pipeline-matrix columns (best effort).

    ``preprocessor.get_feature_names_out()`` names the expanded columns
    (``cat__<col>`` / ``num__<col>`` / ``remainder__<col>``); when it is
    unavailable the positional fallback ``x0 ... xN`` is used. The number
    of names must match the matrix width — a disagreement falls back to
    positional names too. When the matrix is narrower than the
    preprocessor output (an active column reducer), the caller applies
    :func:`_selected_component_names` BEFORE this width check — the
    selector's ``get_support`` reduces the preprocessor's names to the
    surviving components first.
    """
    n_columns = int(matrix.shape[1])
    try:
        preprocessor = trainer.pipeline.named_steps["preprocessor"]  # type: ignore[union-attr]
        names = [str(name) for name in preprocessor.get_feature_names_out()]
        if len(names) != n_columns:
            raise ValueError(
                f"get_feature_names_out returned {len(names)} names for "
                f"{n_columns} columns"
            )
        return names
    except Exception as err:  # noqa: BLE001 — names are cosmetic, never fatal
        _LOG.debug("Post-preprocessing feature names unavailable: %s", err)
        return [f"x{i}" for i in range(n_columns)]


def _preprocessor_feature_names(trainer: ModelTrainer) -> list[str]:
    """The preprocessor's component names (best effort).

    ``get_feature_names_out()`` names the expanded columns
    (``cat__<col>`` / ``num__<col>`` / ``remainder__<col>``); when it is
    unavailable the positional fallback ``x0 ... xN`` sized by the
    preprocessor's own output width is used. The caller reduces this
    list through the fitted column reducers (``_selected_component_names``)
    before matching it against the explained matrix width.
    """
    try:
        preprocessor = trainer.pipeline.named_steps["preprocessor"]  # type: ignore[union-attr]
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception as err:  # noqa: BLE001 — names are cosmetic, never fatal
        _LOG.debug("Post-preprocessing feature names unavailable: %s", err)
        return []


def _selected_component_names(pipeline: Any, names: list[str]) -> list[str]:
    """The pipeline column names actually reaching the model.

    With an active column reducer (``feature_selector`` /
    ``isotonic_selector``) the transformed matrix is narrower than the
    preprocessor's ``get_feature_names_out``: the fitted selector's
    ``get_support`` maps the matrix columns back to the preprocessor
    component positions, so every explained column keeps its real name
    (no positional ``xN`` fallback, no all-zero aggregation). Without a
    reducer the names pass through unchanged.

    Args:
        pipeline: The fitted winner pipeline.
        names: The preprocessor's component names (the full, unselected
            list from ``get_feature_names_out``).

    Returns:
        list[str]: The names of the matrix columns the model actually
            receives, in the matrix's column order.
    """
    for name, step in list(pipeline.steps)[:-1]:
        if name in {"feature_selector", "isotonic_selector"} and hasattr(
            step, "get_support"
        ):
            support = np.asarray(step.get_support(), dtype=bool)
            if len(support) != len(names):
                # A defensive guard: the selector's input width disagrees
                # with the preprocessor's names — keep the full list (the
                # caller falls back to positional names on a mismatch).
                return names
            return [label for label, keep in zip(names, support, strict=True) if keep]
    return names


def _explainer_importance(
    shap_values: Any, labels: list[str]
) -> tuple[FeatureImportance, ...]:
    """Aggregate raw SHAP values into per-feature importances.

    ``shap_values`` is normalized to a 2D matrix (rows × features) by
    ``_shap_matrix``; the importance of one feature is the mean absolute
    SHAP value over the rows, sorted descending.
    """
    matrix = _shap_matrix(shap_values)
    means = np.mean(np.abs(matrix), axis=0)
    order = np.argsort(-means, kind="stable")
    return tuple(
        FeatureImportance(
            feature=labels[int(idx)],
            importance_mean=_sanitized(means[int(idx)], labels[int(idx)]),
        )
        for idx in order
    )


def _shap_matrix(shap_values: Any) -> np.ndarray:
    """Normalize the various SHAP value shapes into a 2D matrix.

    Explainers return ``ndarray (n, m)`` for regression, an ``Explanation``
    (array-like, ``.values`` carries the matrix) and, for multi-output
    wrappers, a list of per-output matrices. A list is collapsed by summing
    the absolute contributions of the outputs (a target-transformed model
    may report several outputs); any other shape is an error.
    """
    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        values = np.sum([np.abs(np.asarray(v)) for v in values], axis=0)
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 3 and matrix.shape[2] == 1:
        # Some explainers append a trailing singleton output axis.
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ExplainNotAvailableError(
            f"Unexpected SHAP values shape {matrix.shape}: a 2D "
            "(rows × features) matrix is required."
        )
    return matrix


# --------------------------------------------------------------------------- #
#  Permutation importance                                                      #
# --------------------------------------------------------------------------- #


def _resolve_scorer_for_explain(metric_user: str, y: Any) -> Callable[..., Any]:
    """Build the scorer of ``metric_user`` for the importance computation.

    The engine's own scorer resolution is reused: custom metrics resolve
    against the process registry (they must be registered first — the
    trainer does it in ``fit()``), built-in names fall back to sklearn.
    ``global_nrmse`` needs the target vector to build its scorer.
    """
    scorer = get_scorer_object(metric_user, global_y=np.asarray(y))
    return cast("Callable[..., Any]", scorer)


def _validate_target(X: pd.DataFrame, y: Any) -> tuple[pd.Series | np.ndarray, str]:
    """Validate the caller's target against the (unsampled) frame ``X``.

    A ``pd.Series`` target is checked for the frame's labels (a missing
    label raises immediately — the caller's alignment problem, surfaced
    here instead of a far-away sklearn ``Input contains NaN``); any other
    target (numpy array, list) is checked for the length (a silent
    truncation of surplus rows would quietly score a different sample
    than the caller built).

    Args:
        X: The full, unsampled feature frame.
        y: The target aligned with the caller's frame.

    Returns:
        tuple: The validated target (kept in its original form — the row
            selection is applied later, against the same structure) and
            its kind (``"series"`` or ``"positional"``).

    Raises:
        ExplainNotAvailableError: The Series is missing labels of ``X``,
                or the positional target length differs from ``len(X)``.
    """
    if isinstance(y, pd.Series):
        missing = [label for label in X.index if label not in y.index]
        if missing:
            raise ExplainNotAvailableError(
                "The target Series is missing labels required by the "
                f"feature frame: {missing[:5]} "
                f"({len(missing)} total). Align y with X (same index) "
                "and pass it again."
            )
        return y, "series"
    values = np.asarray(list(y) if not isinstance(y, np.ndarray) else y)
    if len(values) != len(X):
        raise ExplainNotAvailableError(
            f"The target length ({len(values)}) does not match the "
            f"feature frame rows ({len(X)}): y and X must be aligned "
            "row-by-row."
        )
    return values, "positional"


def _sample_rows_with_target(
    X: pd.DataFrame, y: pd.Series | np.ndarray, kind: str, max_rows: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sample the frame and take the target of the SAME rows.

    The row selection is drawn ONCE (``DataFrame.sample`` with a fixed
    seed) and both structures are reduced by it — a numpy/list target is
    sliced by the sampled ROW POSITIONS (labels of the sampled frame are
    not positions), a Series is reindexed by the sampled labels. This is
    the only sampling point: reducing ``X`` and ``y`` independently (by
    labels for one and positions for the other) silently desynchronizes
    the pairs (row, target) and produces meaningless importances.

    Args:
        X: The full, validated feature frame.
        y: The validated target (the output of :func:`_validate_target`).
        kind: ``"series"`` or ``"positional"`` (the target kind).
        max_rows: The row budget of the computation.

    Returns:
        tuple: The sampled frame and the target values of exactly its
            rows, in the frame's row order.
    """
    if len(X) <= max_rows:
        if isinstance(y, pd.Series):
            return X, y.reindex(X.index).to_numpy()
        return X, np.asarray(y)
    _LOG.info(
        "Importance computed on a random subsample: %d of %d rows",
        max_rows,
        len(X),
    )
    X_sampled = X.sample(n=max_rows, random_state=0)
    if kind == "series":
        series = cast("pd.Series[Any]", y)
        return X_sampled, series.reindex(X_sampled.index).to_numpy()
    positions = np.asarray(
        [X.index.get_loc(label) for label in X_sampled.index], dtype=int
    )
    values = cast("np.ndarray[Any, Any]", y)
    return X_sampled, values[positions]


def _expand_frame_to_trainer_schema(
    trainer: ModelTrainer, X: pd.DataFrame
) -> pd.DataFrame:
    """The frame with the trainer's recorded datetime expansion applied.

    The saved pipeline was fit on the datetime-expanded columns
    (``ts_month`` …), which do not exist in a raw input frame; without
    the expansion the alignment to ``trainer.feature_names`` fails. A
    frame that already carries the components is left untouched (the
    helper skips a source column when all of its components exist).
    """
    expansion = trainer.datetime_expansion_
    if not expansion:
        return X
    try:
        return expand_datetime_columns(X, expansion)
    except ValueError as err:
        raise ExplainNotAvailableError(
            f"The datetime expansion recorded at training time cannot be "
            f"applied to the explanation frame: {err}"
        ) from err


def _baseline_score(
    metric_user: str,
    trainer: ModelTrainer,
    X: pd.DataFrame,
    y: np.ndarray,
    scorer: Callable[..., Any],
) -> float:
    """The unshuffled scorer value of ``trainer`` on ``(X, y)``, in user
    semantics.

    ``permutation_importance`` reports only the *differences* to the
    baseline, so the baseline itself is computed once here with the same
    scorer and converted by ``to_user_value`` (positive RMSE/MAE, plain
    R²). A non-finite baseline is kept as computed: the training result
    model rejects non-finite scores, but a raw frame passed by the caller
    may still produce one (e.g. NRMSE on a constant target) and the
    importance table stays usable.
    """
    value = float(scorer(trainer, X, y))
    return round(to_user_value(metric_user, value), _IMPORTANCE_DECIMALS)


def explain_permutation_importance(
    trainer: ModelTrainer,
    X: pd.DataFrame,
    y: Any,
    result: ExperimentResult,
    *,
    n_repeats: int = DEFAULT_N_REPEATS,
    max_rows: int = DEFAULT_SAMPLE_ROWS,
) -> PermutationImportanceResult:
    """Permutation importance of every original input column.

    The importance is computed on the *raw* pre-pipeline columns through
    ``trainer.predict``: one number per original data column, no matter
    how the pipeline expands, encodes or selects them. The features
    dropped by the pipeline (uninformative cleaning, ID-like dropping,
    datetime ``'drop'`` policy) are reported with a zero importance.

    Rows are subsampled to ``max_rows`` (the tractability budget of the
    ``n_repeats × n_features`` prediction calls); the whole scored block
    is reported as ``data_semantics='train'`` — the saved artifact is fit
    on all data, so these rows are in-sample for it (the caller passes a
    held-out frame explicitly to measure out-of-sample sensitivity).

    Args:
        trainer: The trained winner (``ModelTrainer``).
        X: The feature columns (the raw training frame without the
            target).
        y: The target vector aligned with ``X``.
        result: The finished experiment result (the metric source).
        n_repeats: Shuffles per feature (``>= 1``).
        max_rows: The row budget of the computation (``>= 1``).

    Returns:
        PermutationImportanceResult: The per-feature importance, sorted
            by the descending mean.

    Raises:
        ExplainNotAvailableError: The model is not fitted, the data is
            empty or has no feature columns.
        TrainingError: The pipeline rejects the data (a raw schema
            mismatch); the message names the problem.
    """
    _require_pipeline(trainer)
    X = X.copy()
    if y is None or len(X) == 0:
        raise ExplainNotAvailableError(
            "Permutation importance requires a non-empty feature frame "
            "and an aligned target vector."
        )
    for col in trainer.dropped_features_:
        if col not in X.columns:
            X[col] = np.nan
    # Apply the recorded datetime expansion BEFORE the schema check: a raw
    # frame (with the datetime source column) is a legitimate input —
    # trainer.predict expands it the same way, so the components the
    # pipeline was fit on appear here too.
    X = _expand_frame_to_trainer_schema(trainer, X)
    # The caller's alignment is validated BEFORE the row sampling: a
    # length mismatch must be reported against the original frame, not
    # against the subsample. The sampling then reduces X and y TOGETHER
    # (one row selection for both structures) — sampling X by its own
    # RNG and y by positions afterwards silently desynchronizes the
    # (row, target) pairs.
    y_validated, target_kind = _validate_target(X, y)
    X, y_sampled = _sample_rows_with_target(X, y_validated, target_kind, max_rows)
    if trainer.feature_names:
        missing = [name for name in trainer.feature_names if name not in X.columns]
        if missing:
            raise ExplainNotAvailableError(
                "The feature frame does not match the training schema: "
                f"missing columns {missing} (the recorded datetime "
                "expansion was applied; expand the components upstream "
                "if the source columns are absent)."
            )
        X = X[trainer.feature_names]
    # The number of features actually scored (after the schema filtering,
    # which drops the caller's extra columns) — the docstring of
    # ``n_features`` describes the scored features, not the raw width.
    n_features = X.shape[1]
    if n_features == 0:
        raise ExplainNotAvailableError(
            "The data has no feature columns: permutation importance "
            "cannot be computed."
        )
    metric_user = result.metric_user
    scorer = _resolve_scorer_for_explain(metric_user, y_sampled)
    baseline = _baseline_score(metric_user, trainer, X, y_sampled, scorer)
    raw = permutation_importance(
        trainer,
        X,
        y_sampled,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=trainer.random_state,
    )
    importances = tuple(
        FeatureImportance(
            feature=str(col),
            importance_mean=_sanitized(mean, str(col)),
            importance_std=_sanitized(std, str(col)) if std is not None else None,
        )
        for col, mean, std in zip(
            X.columns,
            raw.importances_mean,
            raw.importances_std,
            strict=True,
        )
    )
    importances = tuple(sorted(importances, key=lambda item: -item.importance_mean))
    return PermutationImportanceResult(
        importances=importances,
        baseline_score=round(baseline, _IMPORTANCE_DECIMALS),
        metric_user=metric_user,
        n_rows=int(X.shape[0]),
        n_features=n_features,
        data_semantics="train",
    )


# --------------------------------------------------------------------------- #
#  SHAP (optional extra)                                                       #
# --------------------------------------------------------------------------- #


def _transform_prefix(pipeline: Any) -> Any:
    """The leading ``Pipeline`` of the steps that actually transform rows.

    The prefix ends right before the first sampler step (a step with
    ``fit_resample`` and no ``transform`` — ``DataOversampler``):
    resampling happens only inside ``fit`` (it grows the training rows),
    it has no meaning at explanation time and its presence breaks
    ``Pipeline.transform``. The column reducers (``feature_selector`` /
    ``isotonic_selector``) stay in the chain: the matrix width then
    matches the model's expectation exactly, so the raw model step can be
    explained without any adapter; the component names of the selected
    columns are resolved by :func:`_selected_component_names`.

    The last step (the model) is always excluded.
    """
    steps: list[tuple[str, Any]] = []
    for name, step in list(pipeline.steps)[:-1]:
        is_sampler = hasattr(step, "fit_resample") and not hasattr(step, "transform")
        if is_sampler:
            break
        steps.append((name, step))
    return type(pipeline)(steps=steps)


def explain_shap(
    trainer: ModelTrainer,
    X: pd.DataFrame,
    result: ExperimentResult,
    *,
    max_rows: int = DEFAULT_SAMPLE_ROWS,
    post_preprocessing: bool = False,
) -> ShapFeatureImportance:
    """SHAP importance of the winner pipeline (the optional ``[shap]`` extra).

    The importances are mean absolute SHAP contributions per feature.
    With ``post_preprocessing=False`` (default) the values are computed
    on the model-input matrix columns (post-encoder and post-selection —
    the matrix width matches the model exactly) and aggregated back to
    the original input columns — a one-hot / hashing component
    contributes to its source column, and the components surviving the
    feature selection keep their real preprocessor names through the
    fitted selector's ``get_support``. With ``post_preprocessing=True``
    the per-component values are reported as-is under those (selected)
    pipeline names — components dropped by the selector are not explained
    (the model never sees them).

    SHAP explains the model's output directly, not the score: the
    baseline is the mean prediction on the scored rows. Computed on the
    training rows, the values are in-sample for the saved artifact (it
    is fit on all data).

    Args:
        trainer: The trained winner.
        X: The feature columns (the raw frame, without the target).
        result: The finished experiment result (the run record; SHAP
            does not use the metric, but the call is kept symmetric with
            :func:`explain_permutation_importance`).
        max_rows: The row budget (SHAP cost grows with both rows and
            features).
        post_preprocessing: Report per-component values under their
            pipeline names instead of aggregating to the original
            columns.

    Returns:
        ShapFeatureImportance: The per-feature importance, sorted by the
            descending mean absolute contribution.

    Raises:
        ExplainNotAvailableError: The model is not fitted, the data is
            empty, does not match the training schema, has no feature
            columns, or the ``shap`` package is not installed (the
            message names the extra).
    """
    del result  # symmetry with the permutation call; SHAP needs no metric
    _require_pipeline(trainer)
    if not is_installed(SHAP_PACKAGE):
        raise ExplainNotAvailableError(
            "SHAP explainability requires the optional dependency 'shap': "
            "install the package with the [shap] extra "
            "(pip install 'configurable-automl-engine[shap]')"
        )
    X = X.copy()
    if len(X) == 0:
        raise ExplainNotAvailableError(
            "SHAP importance requires a non-empty feature frame."
        )
    # The recorded datetime expansion BEFORE the schema check (symmetric
    # with the permutation path): the saved pipeline was fit on the
    # generated components, which do not exist in a raw input frame.
    X = _expand_frame_to_trainer_schema(trainer, X)
    if trainer.feature_names:
        missing = [name for name in trainer.feature_names if name not in X.columns]
        if missing:
            raise ExplainNotAvailableError(
                "The feature frame does not match the training schema: "
                f"missing columns {missing} (the recorded datetime "
                "expansion was applied; expand the components upstream "
                "if the source columns are absent)."
            )
        X = X[trainer.feature_names]
    X = _sample_rows(X, max_rows)
    shap = importlib.import_module(SHAP_PACKAGE)
    pipeline = trainer.pipeline
    assert pipeline is not None  # guarded by _require_pipeline
    # The SHAP matrix is computed on the pipeline's internal features:
    # the preprocessing (and any active column reducer) transform the raw
    # frame, the last step (the wrapped regressor) is the explained model.
    # The transform prefix stops before the first sampler step
    # (DataOversampler implements only fit_resample — Pipeline.transform
    # on a prefix that ends with it fails with "no attribute 'transform'"):
    # SHAP explains the model, not the row resampling.
    transform_prefix = _transform_prefix(pipeline)
    transformed = transform_prefix.transform(X)
    # The column reducers stay in the chain: the matrix width matches the
    # model's expectation exactly (no double reduction, no adapter). The
    # preprocessor's component names are reduced by the fitted selectors'
    # ``get_support`` mapping BEFORE the width check, so every explained
    # column keeps its real name (no positional ``xN`` fallback, no
    # all-zero aggregation).
    model_step = pipeline.steps[-1][1]
    preprocessor_names = _preprocessor_feature_names(trainer)
    feature_names = _selected_component_names(pipeline, preprocessor_names)
    if len(feature_names) != transformed.shape[1]:
        feature_names = _shap_feature_names(trainer, transformed)
    try:
        explainer = shap.Explainer(model_step, transformed)
        shap_values = explainer(transformed)
    except Exception as err:  # noqa: BLE001 — a model-specific explainer is tried next
        _LOG.debug("The generic shap.Explainer failed (%s); trying Permutation.", err)
        try:
            explainer = shap.Explainer(
                model_step.predict, transformed, algorithm="permutation"
            )
            shap_values = explainer(transformed)
        except Exception as perm_err:  # noqa: BLE001 — reported, not raised
            raise ExplainNotAvailableError(
                "SHAP could not explain this pipeline: neither the generic "
                f"Explainer nor the Permutation explainer succeeded ({perm_err})"
            ) from err
    matrix = _shap_matrix(shap_values)
    baseline = round(float(np.mean(pipeline.predict(X))), _IMPORTANCE_DECIMALS)
    if post_preprocessing:
        importances = _explainer_importance(shap_values, feature_names)
        return ShapFeatureImportance(
            importances=importances,
            post_preprocessing=True,
            baseline_prediction=baseline,
        )
    aggregated = _aggregate_shap_to_original(trainer, matrix, feature_names)
    return ShapFeatureImportance(
        importances=aggregated,
        post_preprocessing=False,
        baseline_prediction=baseline,
    )


def _aggregate_shap_to_original(
    trainer: ModelTrainer,
    matrix: np.ndarray,
    feature_names: list[str],
) -> tuple[FeatureImportance, ...]:
    """Map per-component SHAP values back to the original columns.

    ``feature_names`` carry the source column in the preprocessor's
    ``<block>__<name>`` convention; the name is resolved against the
    training columns (``_source_column_of``), and a component without a
    recognizable source (``x3`` fallback names) keeps its own label.
    Missing-value indicators and encoders without a source column cannot
    be mapped and are dropped from the aggregated view (their total mass
    is reported with a WARNING when it is non-negligible).
    """
    original = trainer.feature_names or []
    totals: dict[str, float] = {col: 0.0 for col in original}
    unmapped_mass = 0.0
    for pos, name in enumerate(feature_names):
        contribution = float(np.mean(np.abs(matrix[:, pos])))
        source = _source_column_of(name, original)
        if source is None:
            unmapped_mass += contribution
            continue
        totals[source] = totals.get(source, 0.0) + contribution
    if unmapped_mass > 0.0:
        _LOG.warning(
            "SHAP contributions of unmapped pipeline components "
            "(total %.6f of mean |SHAP| mass) are not attributed to any "
            "original column and are omitted from the aggregated view.",
            unmapped_mass,
        )
    importances = [
        FeatureImportance(feature=col, importance_mean=_sanitized(value, col))
        for col, value in totals.items()
    ]
    return tuple(sorted(importances, key=lambda item: -item.importance_mean))


def _source_column_of(name: str, original: list[str]) -> str | None:
    """The original column behind one pipeline column name (``None`` if none).

    ``get_feature_names_out`` names look like ``cat__col`` /
    ``num__col`` / ``remainder__col`` (one-hot appends the category:
    ``cat__col_value``; hashing nests a second separator:
    ``cat__col__h0``); the name is stripped of the block prefix and then
    matched against the original columns by an exact name, by a
    ``<col>__`` (hashing) or by a ``<col>_`` / ``<col>-`` (one-hot
    category) prefix.
    """
    body = name.split("__", 1)[1] if "__" in name else name
    if body in original:
        return body
    for col in original:
        if body.startswith((f"{col}__", f"{col}_", f"{col}-")):
            return col
    return None


# --------------------------------------------------------------------------- #
#  Cross-view aggregation and rendering                                        #
# --------------------------------------------------------------------------- #


def aggregate_to_original_features(
    importances: Iterable[FeatureImportance],
    original: list[str] | None = None,
) -> tuple[FeatureImportance, ...]:
    """Aggregate post-preprocessing importances into original columns.

    The pipeline column names (``get_feature_names_out``:
    ``cat__<col>`` / ``num__<col>`` / ``remainder__<col>``) carry the
    source column before the encoding expansion; every component whose
    name resolves to an original column is summed into it (the mean
    absolute contributions of the one-hot / hashing components of one
    categorical column add up), and components without a recognizable
    source keep their own name. The result is sorted by the descending
    mean.

    Args:
        importances: The per-component importances (e.g. the SHAP view
            with ``post_preprocessing=True``).
        original: The training feature names used for the source-column
            resolution (``None`` — the name-based heuristic of
            :func:`_source_column_of` decides without the training
            schema: the block prefix is stripped, a hashing component
            keeps ``<col>__hN``, a one-hot component keeps the prefix
            before the category separator).

    Returns:
        tuple[FeatureImportance, ...]: The aggregated importances (the
            std values are dropped — a sum of component stds is not the
            std of the sum).
    """
    totals: dict[str, float] = {}
    for item in importances:
        if original is not None:
            column = _source_column_of(item.feature, original) or item.feature
        else:
            column = _heuristic_source_column(item.feature)
        totals[column] = totals.get(column, 0.0) + item.importance_mean
    aggregated = [
        FeatureImportance(feature=col, importance_mean=_sanitized(value, col))
        for col, value in totals.items()
    ]
    return tuple(sorted(aggregated, key=lambda item: -item.importance_mean))


def _heuristic_source_column(name: str) -> str:
    """The source column of one pipeline column name without the schema.

    The block prefix (``cat__`` / ``num__``) is stripped and the rest is
    reduced to its source column: a hashing component keeps
    ``<col>__hN`` → ``<col>``, a one-hot component keeps the encoded
    column before the category separator (``<col>_<category>`` →
    ``<col>``).
    """
    stripped = name.split("__", 1)[1] if "__" in name else name
    column = stripped.split("__", 1)[0]
    if column == stripped and "_" in stripped and not stripped.endswith("_"):
        # One-hot: ``<col>_<category>`` — the source column is the
        # prefix before the first underscore separator.
        column = stripped.split("_", 1)[0]
    return column


def render_importance_markdown(
    title: str,
    importances: Iterable[FeatureImportance],
    *,
    baseline: float | None = None,
    metric_user: str | None = None,
    max_rows: int = 20,
) -> str:
    """Render an importance table as a Markdown section.

    The table is sorted (the input order is kept), shows the importance
    mean and std columns and, when given, the baseline metric score it
    was measured against. The output starts with a ``##`` heading and
    ends with a blank line, so it can be appended to a report body.

    Args:
        title: The section heading text (escaped for Markdown).
        importances: The importances to render.
        baseline: The unshuffled score (user semantics) the importances
            were measured against (``None`` — the line is omitted).
        metric_user: The metric name for the baseline line (``None`` —
            the line is omitted).
        max_rows: The row budget of the table; the rest is summarized in
            a trailing line (``0`` renders no table).

    Returns:
        str: The Markdown section.
    """
    items = list(importances)
    lines = [f"## {_escape_inline(title)}", ""]
    if metric_user is not None and baseline is not None:
        lines.append(
            f"Baseline {metric_user}: **{baseline:.3f}** "
            "(positive RMSE/MAE, plain R²; importances are the mean score "
            "change under permutation)."
        )
        lines.append("")
    if not items:
        lines.append("_No feature importances available._")
        lines.append("")
        return "\n".join(lines)
    if max_rows <= 0:
        lines.append(
            f"_The table is omitted ({len(items)} features; the largest "
            f"importance is {items[0].importance_mean:.6f})._"
        )
        lines.append("")
        return "\n".join(lines)
    lines.append("| # | Feature | Importance mean | Importance std |")
    lines.append("|--:|:--------|----------------:|---------------:|")
    for position, item in enumerate(items[:max_rows], start=1):
        std = "—" if item.importance_std is None else f"{item.importance_std:.6f}"
        lines.append(
            f"| {position} "
            f"| {_escape_cell(item.feature)} "
            f"| {item.importance_mean:.6f} "
            f"| {std} |"
        )
    if len(items) > max_rows:
        lines.append(
            f"_… and {len(items) - max_rows} more features (importance "
            f"≤ {items[max_rows].importance_mean:.6f})._"
        )
    lines.append("")
    return "\n".join(lines)


def _escape_cell(value: object) -> str:
    """Escape one Markdown table cell (pipes, line breaks)."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def _escape_inline(value: object) -> str:
    """Escape one Markdown line (headings)."""
    return str(value).replace("\r", " ").replace("\n", " ").strip()
