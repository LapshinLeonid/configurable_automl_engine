"""Trial pipeline assembly of the HPO runner.

Builds the preprocessor of an HPO phase (with the uninformative-feature
cleaning and the post-preprocessing feature counts used by the complexity
guard), resolves the per-trial feature selection decision and assembles the
trial estimator in the order preprocessor → [isotonic adapter] → [feature
selector] → [oversampler] → model, consistent with the final
``ModelTrainer``. It may import ``hpo.types`` only and must not import
``hpo.runner`` at runtime (type checking only) or
``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.trial import Trial
from sklearn.base import clone

from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    requires_dense_input,
    resolve_algorithm_name,
)
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
    _clean_uninformative_features,
    build_preprocessor,
    detect_feature_types,
    estimate_n_features_after_preprocessing,
)
from configurable_automl_engine.preprocessing_presets import (
    requires_missing_indicator,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.trainer import IsotonicDataTransformer
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionMode,
)

if TYPE_CHECKING:
    from .runner import _OptimizeRunner

# The records keep the logger name of the pre-split ``tuner`` module: the
# logger name is part of the observable behavior (handlers, filters and
# ``caplog`` assertions select records by it).
log = _logging.getLogger("configurable_automl_engine.tuner")


def _missing_indicator_enabled(algorithm: str | None) -> bool:
    """Safely determine whether missing-value indicators are needed.

    Strictly univariate algorithms (``isotonic_regression``) are incompatible
    with missing-indicator columns (their input contract is exactly one
    feature), so ``False`` is returned for them. An empty algorithm name or an
    invalid name (unknown algorithm) must not crash the HPO phase — in that
    case the safe default ``True`` is returned.

    Args:
        algorithm: Regression algorithm name; it may be
            ``None``/an empty string before name validation.

    Returns:
        ``True`` — missing indicators are enabled; ``False`` — the algorithm
        is strictly univariate.
    """
    if not algorithm:
        return True
    try:
        return requires_missing_indicator(algorithm)
    except ValueError:
        # Unknown algorithm name: do not mask the config error, but do not
        # let it surface as an unhandled exception during preprocessor
        # assembly — name validation is a separate step.
        return True


def build_trial_pipeline(runner: _OptimizeRunner) -> None:
    """Build the preprocessor for the HPO phase (step 3).

    A single point of preprocessor assembly: categories -> one-hot (or
    ordinal if encoding is set), numerics -> the preprocessing preset
    selected automatically for the algorithm. The same logic is used in
    the final training (trainer.ModelTrainer), so HPO and the final fit
    stay consistent. The ready pipeline factory is ``assemble_estimator``.

    Cleaning of uninformative features is applied here too — at the point
    where ``detect_feature_types`` runs, before the HPO folds and before
    the final training, so that both phases use a consistent feature set.
    Cleaning runs only in the standalone branch (``runner.preprocessor is
    None``): if the user passed a ready preprocessor, it has already fixed
    its feature set, and ``runner.X`` must not be silently changed. For
    ``np.ndarray`` without column names, cleaning is skipped (a
    limitation shared with ``detect_feature_types``); the list of dropped
    columns is stored in ``runner.dropped_features_``.
    """
    runner.dropped_features_ = []
    if runner.preprocessor is None and isinstance(runner.X, pd.DataFrame):
        cleaned = _clean_uninformative_features(runner.X)
        runner.dropped_features_ = [
            col for col in runner.X.columns if col not in cleaned.columns
        ]
        if runner.dropped_features_:
            runner.X = cleaned
            # Explicitly passed column lists are reconciled with the
            # cleaned feature set: otherwise build_preprocessor would fail
            # on missing columns (Unknown feature name).
            if runner.categorical_features is not None:
                runner.categorical_features = [
                    col
                    for col in runner.categorical_features
                    if col in runner.X.columns
                ]
            if runner.numerical_features is not None:
                runner.numerical_features = [
                    col for col in runner.numerical_features if col in runner.X.columns
                ]
    if runner.preprocessor is None:
        # Guard for an unknown algorithm: the preprocessor cannot be
        # assembled (``resolve_preprocessing_preset`` would raise). The
        # search-space resolution raises the proper ``HyperoptError``
        # afterwards; skipping the preprocessor here preserves the exact
        # error contract for unknown algorithms.
        if resolve_algorithm_name(runner.algo) not in AVAILABLE_ALGORITHMS:
            runner.n_features_eff = max(1, int(runner.X.shape[1]))
            runner.n_features_size = max(1, int(runner.X.shape[1]))
            return
        encoding_strategy: EncodingStrategy = runner.encoding or "one_hot"
        # Automatic preset selection + application of an explicit override.
        # The preset is resolved once before training (performance) and
        # logged for observability.
        preset = resolve_preprocessing_preset(
            runner.algo, runner.preprocessing_override
        )
        log.info(
            "Resolved preprocessing preset for algorithm '%s': %s",
            runner.algo,
            preset,
        )
        if (
            runner.categorical_features is not None
            or runner.numerical_features is not None
        ):
            # Explicitly passed column lists (the main path from
            # training_engine)
            if isinstance(runner.X, pd.DataFrame):
                runner.preprocessor = build_preprocessor(
                    list(runner.X.columns),
                    runner.categorical_features or [],
                    runner.numerical_features or [],
                    encoding=encoding_strategy,
                    imputation_strategy=preset.imputation_strategy,
                    scaling=preset.scaling,
                    high_cardinality_threshold=runner.high_cardinality_threshold,
                    high_cardinality_encoding=runner.high_cardinality_encoding,
                    hashing_n_components=runner.hashing_n_components,
                    target_encoding_smoothing=runner.target_encoding_smoothing,
                    target_encoding_fallback=runner.target_encoding_fallback,
                    random_state=runner.random_state,
                    force_dense_output=requires_dense_input(runner.algo),
                    add_indicator=_missing_indicator_enabled(runner.algo),
                )
            else:
                log.warning(
                    "categorical/numerical_features provided, but X is not a "
                    "pandas.DataFrame — skipping automatic preprocessor."
                )
        elif isinstance(runner.X, pd.DataFrame):
            # Autodetect categories (default behavior when called outside
            # the engine). The preprocessor is built when ANY features
            # (categorical or numerical) are present: this guarantees that
            # on a purely numerical DataFrame the HPO phase applies
            # scaling just like the final ModelTrainer.
            cats, nums = detect_feature_types(runner.X)
            # Store the detected lists so that the isotonic numeric block
            # (``n_numeric_eff``) is determinable from the same feature
            # set that built the preprocessor.
            runner.categorical_features = cats
            runner.numerical_features = nums
            if cats or nums:
                runner.preprocessor = build_preprocessor(
                    list(runner.X.columns),
                    cats,
                    nums,
                    encoding=encoding_strategy,
                    imputation_strategy=preset.imputation_strategy,
                    scaling=preset.scaling,
                    high_cardinality_threshold=runner.high_cardinality_threshold,
                    high_cardinality_encoding=runner.high_cardinality_encoding,
                    hashing_n_components=runner.hashing_n_components,
                    target_encoding_smoothing=runner.target_encoding_smoothing,
                    target_encoding_fallback=runner.target_encoding_fallback,
                    random_state=runner.random_state,
                    force_dense_output=requires_dense_input(runner.algo),
                    add_indicator=_missing_indicator_enabled(runner.algo),
                )
        else:
            log.warning(
                "X is not a pandas.DataFrame and categorical_features/"
                "numerical_features were not provided — categorical columns "
                "(if any) will be treated as numeric."
            )

    # The number of features the model actually sees after preprocessing
    # (including one-hot and missing-value indicators). Used by the
    # adaptive complexity guard as ``p`` for the rules that depend on
    # the feature count (``max_features``, ``colsample_bytree``).
    # Computed deterministically once per phase — the same value in every
    # HPO phase and in the subprocesses.
    runner.n_features_eff = _estimate_n_features_after_preprocessing(runner)
    # The number of ORIGINAL columns (a categorical column counts as 1,
    # before the one-hot expansion). Used by the adaptive complexity
    # guard for the size-mode ratio ``n / p`` only. Without a preprocessor
    # (np.ndarray input without explicit column lists) the model sees
    # exactly the original columns, so the size count is aligned with the
    # post-preprocessing count — this preserves the legacy behavior of
    # direct optimize() calls (no one-hot expansion, the guard sees the
    # same p it always did).
    runner.n_features_size = max(
        1,
        int(runner.X.shape[1])
        if runner.preprocessor is not None
        else runner.n_features_eff,
    )
    # The number of NUMERICAL features after preprocessing: the isotonic
    # ``feature_index`` is restricted to this range (numeric columns only;
    # one-hot dummies are not meaningful inputs). Deterministic — one
    # column per numerical feature (isotonic disables missing indicators
    # and never expands numeric columns).
    if runner.preprocessor is not None and not (
        runner.categorical_features or runner.numerical_features
    ):
        # An external preprocessor without explicit column lists: the
        # numerical block is not determinable.
        runner.n_numeric_eff = None
    elif runner.preprocessor is None and not isinstance(runner.X, pd.DataFrame):
        # ndarray without column lists: no preprocessor is built and every
        # column is treated as numeric, so the feature_index is positional
        # over the whole matrix.
        runner.n_numeric_eff = None
    else:
        nums = list(runner.numerical_features or [])
        if not nums and isinstance(runner.X, pd.DataFrame):
            _, nums = detect_feature_types(runner.X)
        runner.n_numeric_eff = max(0, len(nums))


def _estimate_n_features_after_preprocessing(runner: _OptimizeRunner) -> int:
    """Estimate the number of features after the preprocessor.

    The number of columns the model actually sees: numerical features
    plus their missing-value indicators (``add_indicator=True``) plus the
    encoded categorical features. The estimate is computed
    deterministically from the data and the encoding settings — the same
    value in every HPO phase and in the subprocesses (no fitting, no
    randomness). It is used by the adaptive complexity guard as ``p`` for
    the rules that depend on the feature count (``max_features``,
    ``colsample_bytree``); the size-mode ratio ``n / p`` uses the
    original column count instead (``n_features_size``).

    One-hot encoding expands each categorical column into its number of
    unique categories; ordinal/target/frequency encoding keeps one column
    per categorical column; hashing encoding expands into
    ``n_components`` columns. In the automatic high-cardinality mode
    (``high_cardinality_threshold`` + ``high_cardinality_encoding``), each
    categorical column is counted with the strategy it actually receives
    at fit time: columns whose cardinality is strictly greater than the
    threshold use the high-cardinality strategy, the rest — the main
    ``encoding`` strategy (matches
    :class:`~configurable_automl_engine.preprocessing.SplitCategoricalEncoder`).

    Returns:
        int: The estimated feature count (>= 1).
    """
    if runner.preprocessor is not None:
        n_out = getattr(runner.preprocessor, "n_features_out_", None)
        if n_out is not None and int(n_out) >= 1:
            return int(n_out)

    # Explicitly passed column lists (the engine path) — the accurate
    # source; otherwise fall back to auto-detected types.
    cats = list(runner.categorical_features or [])
    nums = list(runner.numerical_features or [])
    if not cats and not nums and isinstance(runner.X, pd.DataFrame):
        cats, nums = detect_feature_types(runner.X)

    return estimate_n_features_after_preprocessing(
        runner.X,
        categorical_features=cats,
        numerical_features=nums,
        encoding=cast(EncodingStrategy, runner.encoding or "one_hot"),
        high_cardinality_threshold=runner.high_cardinality_threshold,
        high_cardinality_encoding=runner.high_cardinality_encoding,
        hashing_n_components=int(getattr(runner, "hashing_n_components", 16)),
        add_indicator=_missing_indicator_added(runner),
    )


def _missing_indicator_added(runner: _OptimizeRunner) -> bool:
    """Whether missing-value indicators are added for numerical features."""
    return _missing_indicator_enabled(runner.algo)


def assemble_estimator(
    runner: _OptimizeRunner,
    model_impl: Any,
    apply_fs: bool = False,
    feature_index: int | None = None,
) -> Any:
    """Assemble the pipeline in the order preprocessor -> [selector] -> [sampler] -> model.

    Oversampling receives already-encoded numerical features (one-hot), so
    SMOTE/ADASYN always work with numerical data. Feature selection is
    activated strictly between the preprocessor and the oversampler:
    synthetic rows are generated only for informative features (the order
    is consistent with ModelTrainer).

    For ``isotonic_regression`` an ``IsotonicDataTransformer`` step is
    inserted right after the preprocessor (before selection/oversampling):
    the algorithm is strictly univariate and the adapter reduces the
    preprocessed matrix to the single column selected via
    ``feature_index`` (0 by default).

    Args:
        model_impl: The base regressor.
        apply_fs: Flag to enable the ``feature_selector`` step.
        feature_index: The ordinal feature index for the isotonic data
            adapter (``None`` — 0).
    """
    steps: list[tuple[str, Any]] = []
    if runner.preprocessor is not None:
        steps.append(("preprocessor", runner.preprocessor))
    if runner.is_isotonic:
        steps.append(
            (
                "isotonic_selector",
                IsotonicDataTransformer(
                    feature_index=feature_index if feature_index is not None else 0,
                    n_numeric=runner.n_numeric_eff,
                ),
            )
        )
    # New step: feature selection is activated strictly between the
    # preprocessor and the oversampler.
    if apply_fs:
        steps.append(("feature_selector", _fs_transformer_factory(runner)))
    if runner.oversampling_config["active"]:
        steps.append(
            ("sampler", DataOversampler(**runner.oversampling_config["params"]))
        )
    steps.append(("model", model_impl))
    if len(steps) == 1:
        return model_impl
    return ImbPipeline(steps)


def _resolve_apply_fs(runner: _OptimizeRunner, trial: Trial) -> bool:
    """Determine whether feature selection is needed for the current trial.

    In 'auto' mode Optuna itself explores the decision space (whether
    reducing features is beneficial); in 'always'/'disabled' the decision
    strictly follows the configuration. For isotonic_regression the
    use_feature_selection category is not offered even in 'auto' mode:
    selection is forcibly disabled (a univariate algorithm with its own
    IsotonicDataTransformer), so the suggestion would be meaningless and
    the auxiliary key must not leak into best_params.

    Args:
        trial (Trial): The current Optuna trial object.
    Returns:
        bool: Whether the ``feature_selector`` step should be added to the pipeline.
    """
    if runner.fs_mode == FeatureSelectionMode.auto and not runner.is_isotonic:
        apply_fs = trial.suggest_categorical("use_feature_selection", [True, False])
    elif runner.fs_mode == FeatureSelectionMode.always:
        apply_fs = True
    else:
        apply_fs = False
    # Isotonic regression requires exactly one feature and uses its own
    # IsotonicDataTransformer — selection is forcibly disabled (including
    # when 'always' or enqueued initial_params carried the flag).
    if runner.is_isotonic:
        apply_fs = False
    return apply_fs


def _resolve_isotonic_feature_name(
    runner: _OptimizeRunner, feature_index: int
) -> str | None:
    """Resolve the name of the numerical feature selected by isotonic.

    The preprocessor emits the categorical block first and the numerical
    block last, so the selected numerical feature occupies the absolute
    position ``(n_columns - n_numeric) + feature_index`` in the
    preprocessed matrix. The name is resolved from the fitted
    preprocessor's ``get_feature_names_out`` (the preprocessor is fitted
    once on the full data); ``None`` is returned when the names are not
    available (np.ndarray input without column lists) or the resolution
    fails (the estimate does not match a fitted output — non-fatal).

    Args:
        feature_index: The numeric-block-relative index of the best trial.

    Returns:
        The resolved feature name, or ``None``.
    """
    if runner.preprocessor is None or runner.n_numeric_eff is None:
        return None
    try:
        # The preprocessor used by the trials is only ever fitted through
        # clones (per fold), so fit a copy once on the full data to obtain
        # the stable output names (the numerical block width and the
        # ``num__<col>`` names are data-independent for the numeric block).
        names = runner.preprocessor.get_feature_names_out()
    except Exception:  # noqa: BLE001 — best-effort name resolution
        try:
            probe = clone(runner.preprocessor).fit(runner.X, runner.y)
            names = probe.get_feature_names_out()
        except Exception:  # noqa: BLE001 — best-effort name resolution
            return None
    names_list = list(names)
    n_cols = len(names_list)
    if n_cols < runner.n_numeric_eff or feature_index >= runner.n_numeric_eff:
        return None
    absolute = (n_cols - int(runner.n_numeric_eff)) + feature_index
    return str(names_list[absolute])


def _fs_transformer_factory(runner: _OptimizeRunner) -> FeatureSelector:
    """Create a fresh feature selector instance for a pipeline step.

    Each call returns a new ``FeatureSelector`` with a deterministic
    ``random_state`` (consistent with the final ModelTrainer training),
    which guarantees reproducible selection across trials and threads.
    With ``random_state=None`` (a fully non-deterministic Optuna run),
    selection still uses the fixed seed 42: otherwise each factory call
    would create a selector with a new random seed and selection would be
    irreproducible even with all other pipeline stages deterministic.
    """
    return FeatureSelector(
        method=runner.fs_cfg.method.value,
        percentile=runner.fs_cfg.percentile,
        min_features=runner.fs_cfg.min_features,
        variance_threshold=runner.fs_cfg.variance_threshold,
        n_estimators=runner.fs_cfg.n_estimators,
        random_state=runner.random_state if runner.random_state is not None else 42,
    )
