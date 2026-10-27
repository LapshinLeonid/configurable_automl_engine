"""Trial pipeline assembly of the HPO runner.

Plans the preprocessor of an HPO phase (``plan_preprocessing``: the
uninformative-feature cleaning and the post-preprocessing feature counts used
by the complexity guard), resolves the per-trial feature selection decision
and assembles the trial estimator (``assemble_estimator``) in the order
preprocessor → [isotonic adapter] → [feature selector] → [oversampler] →
model, consistent with the final ``ModelTrainer``. Every function is pure: it
takes explicit inputs and returns its result. The module may import
``hpo.types``, ``hpo.plans`` and ``hpo.request`` only and must not import
``hpo.runner`` or ``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.trial import Trial
from sklearn.base import clone

from configurable_automl_engine.feature_selection import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
    FeatureSelector,
)
from configurable_automl_engine.isotonic import IsotonicDataTransformer
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

from .plans import PreprocessingPlan
from .request import HpoRequest, OversamplingSettings

if TYPE_CHECKING:
    from .request import EncodingSettings

log = _logging.getLogger(__name__)


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


def plan_preprocessing(X: Any, request: HpoRequest) -> PreprocessingPlan:
    """Plan the preprocessor of the HPO phase.

    A single point of preprocessor assembly: categories -> one-hot (or
    ordinal if encoding is set), numerics -> the preprocessing preset
    selected automatically for the algorithm. The same logic is used in
    the final training (trainer.ModelTrainer), so HPO and the final fit
    stay consistent. The trial pipelines are assembled by
    ``assemble_estimator``.

    Cleaning of uninformative features is applied here too — at the point
    where ``detect_feature_types`` runs, before the HPO folds and before
    the final training, so that both phases use a consistent feature set.
    Cleaning runs only in the standalone branch (``request.preprocessor is
    None``): if the user passed a ready preprocessor, it has already fixed
    its feature set, and ``X`` must not be changed. For ``np.ndarray``
    without column names, cleaning is skipped (a limitation shared with
    ``detect_feature_types``). The cleaned matrix is returned in the plan;
    the input ``X`` is not mutated.

    Args:
        X (Any): The (validated) feature matrix.
        request (HpoRequest): The run settings.

    Returns:
        PreprocessingPlan: The cleaned matrix, the column lists, the
            preprocessor and the feature counts.
    """
    algo = request.algo_name.lower()
    categorical = _copy(request.categorical_features)
    numerical = _copy(request.numerical_features)
    preprocessor = request.preprocessor
    dropped: list[str] = []
    if preprocessor is None and isinstance(X, pd.DataFrame):
        cleaned = _clean_uninformative_features(X)
        dropped = [col for col in X.columns if col not in cleaned.columns]
        if dropped:
            X = cleaned
            # Explicitly passed column lists are reconciled with the cleaned
            # feature set: otherwise build_preprocessor would fail on missing
            # columns (Unknown feature name).
            if categorical is not None:
                categorical = [col for col in categorical if col in X.columns]
            if numerical is not None:
                numerical = [col for col in numerical if col in X.columns]
    if preprocessor is None:
        # Guard for an unknown algorithm: the preprocessor cannot be
        # assembled (``resolve_preprocessing_preset`` would raise). The
        # search-space resolution raises the proper ``HyperoptError``
        # afterwards; skipping the preprocessor here preserves the exact
        # error contract for unknown algorithms.
        if resolve_algorithm_name(algo) not in AVAILABLE_ALGORITHMS:
            n_columns = max(1, int(X.shape[1]))
            return PreprocessingPlan(
                X=X,
                dropped_features=dropped,
                categorical_features=categorical,
                numerical_features=numerical,
                preprocessor=None,
                n_features_eff=n_columns,
                n_numeric_eff=None,
                n_features_size=n_columns,
                isotonic_max_feature_index=_isotonic_max_feature_index(
                    None, False, n_columns, int(X.shape[1])
                ),
            )
        preprocessor, categorical, numerical = _build_trial_preprocessor(
            X, algo, request, categorical, numerical
        )

    # The number of features the model actually sees after preprocessing
    # (including one-hot and missing-value indicators). Used by the
    # adaptive complexity guard as ``p`` for the rules that depend on
    # the feature count (``max_features``, ``colsample_bytree``).
    # Computed deterministically once per phase — the same value in every
    # HPO phase and in the subprocesses.
    n_features_eff = _estimate_n_features_after_preprocessing(
        X,
        preprocessor=preprocessor,
        categorical_features=categorical,
        numerical_features=numerical,
        encoding=request.encoding,
        algo=algo,
    )
    # The number of ORIGINAL columns (a categorical column counts as 1,
    # before the one-hot expansion). Used by the adaptive complexity
    # guard for the size-mode ratio ``n / p`` only. Without a preprocessor
    # (np.ndarray input without explicit column lists) the model sees
    # exactly the original columns, so the size count is aligned with the
    # post-preprocessing count — this preserves the legacy behavior of
    # direct optimize() calls (no one-hot expansion, the guard sees the
    # same p it always did).
    n_features_size = max(
        1, int(X.shape[1]) if preprocessor is not None else n_features_eff
    )
    n_numeric_eff = _numeric_feature_count(X, preprocessor, categorical, numerical)
    return PreprocessingPlan(
        X=X,
        dropped_features=dropped,
        categorical_features=categorical,
        numerical_features=numerical,
        preprocessor=preprocessor,
        n_features_eff=n_features_eff,
        n_numeric_eff=n_numeric_eff,
        n_features_size=n_features_size,
        isotonic_max_feature_index=_isotonic_max_feature_index(
            n_numeric_eff,
            preprocessor is not None,
            n_features_eff,
            int(getattr(X, "shape", (0, 1))[1]),
        ),
    )


def _copy(columns: list[str] | None) -> list[str] | None:
    """A copy of a column list (``None`` stays ``None``)."""
    return None if columns is None else list(columns)


def _build_trial_preprocessor(
    X: Any,
    algo: str,
    request: HpoRequest,
    categorical: list[str] | None,
    numerical: list[str] | None,
) -> tuple[Any, list[str] | None, list[str] | None]:
    """Build the trial preprocessor from the column lists or the detected types.

    Args:
        X (Any): The cleaned feature matrix.
        algo (str): The lower-cased algorithm name.
        request (HpoRequest): The run settings.
        categorical (list[str] | None): The explicit categorical columns.
        numerical (list[str] | None): The explicit numerical columns.

    Returns:
        tuple[Any, list[str] | None, list[str] | None]: The preprocessor (or
            ``None``) and the column lists it was built from (the detected
            lists when none were passed).
    """
    encoding_strategy: EncodingStrategy = request.encoding.encoding or "one_hot"
    # Automatic preset selection + application of an explicit override.
    # The preset is resolved once before training (performance) and
    # logged for observability.
    preset = resolve_preprocessing_preset(algo, request.preprocessing_override)
    log.info(
        "Resolved preprocessing preset for algorithm '%s': %s",
        algo,
        preset,
    )
    if categorical is not None or numerical is not None:
        # Explicitly passed column lists (the main path from
        # training_engine)
        if not isinstance(X, pd.DataFrame):
            log.warning(
                "categorical/numerical_features provided, but X is not a "
                "pandas.DataFrame — skipping automatic preprocessor."
            )
            return None, categorical, numerical
        cats, nums = categorical or [], numerical or []
    elif isinstance(X, pd.DataFrame):
        # Autodetect categories (default behavior when called outside
        # the engine). The preprocessor is built when ANY features
        # (categorical or numerical) are present: this guarantees that
        # on a purely numerical DataFrame the HPO phase applies
        # scaling just like the final ModelTrainer. The detected lists are
        # kept so that the isotonic numeric block (``n_numeric_eff``) is
        # determinable from the same feature set that built the
        # preprocessor.
        cats, nums = detect_feature_types(X)
        categorical, numerical = cats, nums
        if not (cats or nums):
            return None, categorical, numerical
    else:
        log.warning(
            "X is not a pandas.DataFrame and categorical_features/"
            "numerical_features were not provided — categorical columns "
            "(if any) will be treated as numeric."
        )
        return None, categorical, numerical
    encoding = request.encoding
    preprocessor = build_preprocessor(
        list(X.columns),
        cats,
        nums,
        encoding=encoding_strategy,
        imputation_strategy=preset.imputation_strategy,
        scaling=preset.scaling,
        high_cardinality_threshold=encoding.high_cardinality_threshold,
        high_cardinality_encoding=encoding.high_cardinality_encoding,
        hashing_n_components=encoding.hashing_n_components,
        target_encoding_smoothing=encoding.target_encoding_smoothing,
        target_encoding_fallback=encoding.target_encoding_fallback,
        random_state=request.random_state,
        force_dense_output=requires_dense_input(algo),
        add_indicator=_missing_indicator_enabled(algo),
    )
    return preprocessor, categorical, numerical


def _numeric_feature_count(
    X: Any,
    preprocessor: Any,
    categorical: list[str] | None,
    numerical: list[str] | None,
) -> int | None:
    """The number of NUMERICAL features after preprocessing.

    The isotonic ``feature_index`` is restricted to this range (numeric
    columns only; one-hot dummies are not meaningful inputs). Deterministic —
    one column per numerical feature (isotonic disables missing indicators
    and never expands numeric columns).

    Args:
        X (Any): The cleaned feature matrix.
        preprocessor (Any): The trial preprocessor, or ``None``.
        categorical (list[str] | None): The categorical columns.
        numerical (list[str] | None): The numerical columns.

    Returns:
        int | None: The count, or ``None`` when not determinable.
    """
    if preprocessor is not None and not (categorical or numerical):
        # An external preprocessor without explicit column lists: the
        # numerical block is not determinable.
        return None
    if preprocessor is None and not isinstance(X, pd.DataFrame):
        # ndarray without column lists: no preprocessor is built and every
        # column is treated as numeric, so the feature_index is positional
        # over the whole matrix.
        return None
    nums = list(numerical or [])
    if not nums and isinstance(X, pd.DataFrame):
        _, nums = detect_feature_types(X)
    return max(0, len(nums))


def _isotonic_max_feature_index(
    n_numeric_eff: int | None,
    has_preprocessor: bool,
    n_features_eff: int,
    n_columns: int,
) -> int:
    """The upper bound of the isotonic ``feature_index`` search range.

    Candidates are restricted to numerical features (one-hot dummies are
    not meaningful inputs): the count comes from the numerical feature
    list (``n_numeric_eff``). When the numeric block is not determinable
    (``n_numeric_eff is None``):

    * an already-built preprocessor — fall back to its estimated output
      width (``n_features_eff``);
    * no preprocessor (ndarray without column lists) — every column is
      treated as numeric and the index is positional over the raw matrix.

    Args:
        n_numeric_eff (int | None): The number of numerical features after
            preprocessing.
        has_preprocessor (bool): Whether a preprocessor is built.
        n_features_eff (int): The number of features after preprocessing.
        n_columns (int): The number of columns of the raw matrix.

    Returns:
        int: The inclusive upper bound of the index range (>= 0).
    """
    if n_numeric_eff is not None:
        n_features = int(n_numeric_eff)
    elif has_preprocessor:
        n_features = int(n_features_eff)
    else:
        n_features = int(n_columns)
    return max(0, n_features - 1)


def _estimate_n_features_after_preprocessing(
    X: Any,
    *,
    preprocessor: Any,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    encoding: EncodingSettings,
    algo: str,
) -> int:
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

    Args:
        X (Any): The cleaned feature matrix.
        preprocessor (Any): The trial preprocessor, or ``None``.
        categorical_features (list[str] | None): The categorical columns.
        numerical_features (list[str] | None): The numerical columns.
        encoding (EncodingSettings): The categorical encoding settings.
        algo (str): The lower-cased algorithm name.

    Returns:
        int: The estimated feature count (>= 1).
    """
    if preprocessor is not None:
        n_out = getattr(preprocessor, "n_features_out_", None)
        if n_out is not None and int(n_out) >= 1:
            return int(n_out)

    # Explicitly passed column lists (the engine path) — the accurate
    # source; otherwise fall back to auto-detected types.
    cats = list(categorical_features or [])
    nums = list(numerical_features or [])
    if not cats and not nums and isinstance(X, pd.DataFrame):
        cats, nums = detect_feature_types(X)

    return estimate_n_features_after_preprocessing(
        X,
        categorical_features=cats,
        numerical_features=nums,
        encoding=cast(EncodingStrategy, encoding.encoding or "one_hot"),
        high_cardinality_threshold=encoding.high_cardinality_threshold,
        high_cardinality_encoding=encoding.high_cardinality_encoding,
        hashing_n_components=int(encoding.hashing_n_components),
        add_indicator=_missing_indicator_added(algo),
    )


def _missing_indicator_added(algo: str) -> bool:
    """Whether missing-value indicators are added for numerical features."""
    return _missing_indicator_enabled(algo)


def assemble_estimator(
    model_impl: Any,
    preprocessing: PreprocessingPlan,
    *,
    is_isotonic: bool,
    fs_cfg: FeatureSelectionCfg,
    oversampling: OversamplingSettings,
    random_state: int | None,
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
        preprocessing: The preprocessing plan (the preprocessor and the
            numerical block width).
        is_isotonic: Whether the algorithm is ``isotonic_regression``.
        fs_cfg: The feature selection configuration.
        oversampling: The oversampling settings.
        random_state: The run seed (the feature selector seed).
        apply_fs: Flag to enable the ``feature_selector`` step.
        feature_index: The ordinal feature index for the isotonic data
            adapter (``None`` — 0).

    Returns:
        Any: The trial pipeline (or the bare model when it has no other
            step).
    """
    steps: list[tuple[str, Any]] = []
    if preprocessing.preprocessor is not None:
        steps.append(("preprocessor", preprocessing.preprocessor))
    if is_isotonic:
        steps.append(
            (
                "isotonic_selector",
                IsotonicDataTransformer(
                    feature_index=feature_index if feature_index is not None else 0,
                    n_numeric=preprocessing.n_numeric_eff,
                ),
            )
        )
    # New step: feature selection is activated strictly between the
    # preprocessor and the oversampler.
    if apply_fs:
        steps.append(
            ("feature_selector", _fs_transformer_factory(fs_cfg, random_state))
        )
    if oversampling.enable:
        steps.append(("sampler", DataOversampler(**oversampling.sampler_params())))
    steps.append(("model", model_impl))
    if len(steps) == 1:
        return model_impl
    return ImbPipeline(steps)


def _resolve_apply_fs(
    trial: Trial, fs_mode: FeatureSelectionMode, is_isotonic: bool
) -> bool:
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
        fs_mode (FeatureSelectionMode): The feature selection mode.
        is_isotonic (bool): Whether the algorithm is ``isotonic_regression``.
    Returns:
        bool: Whether the ``feature_selector`` step should be added to the pipeline.
    """
    if fs_mode == FeatureSelectionMode.auto and not is_isotonic:
        apply_fs = trial.suggest_categorical("use_feature_selection", [True, False])
    elif fs_mode == FeatureSelectionMode.always:
        apply_fs = True
    else:
        apply_fs = False
    # Isotonic regression requires exactly one feature and uses its own
    # IsotonicDataTransformer — selection is forcibly disabled (including
    # when 'always' or enqueued initial_params carried the flag).
    if is_isotonic:
        apply_fs = False
    return apply_fs


def _resolve_isotonic_feature_name(
    preprocessor: Any,
    n_numeric_eff: int | None,
    X: Any,
    y: Any,
    feature_index: int,
) -> str | None:
    """Resolve the name of the numerical feature selected by isotonic.

    The preprocessor emits the categorical block first and the numerical
    block last, so the selected numerical feature occupies the absolute
    position ``(n_columns - n_numeric) + feature_index`` in the
    preprocessed matrix. The name is resolved from the preprocessor's
    ``get_feature_names_out`` (a clone is fitted once on the full data when
    the preprocessor itself is not fitted; the preprocessor is not
    modified); ``None`` is returned when the names are not available
    (np.ndarray input without column lists) or the resolution fails (the
    estimate does not match a fitted output — non-fatal).

    Args:
        preprocessor: The trial preprocessor, or ``None``.
        n_numeric_eff: The number of numerical features after preprocessing.
        X: The feature matrix of the run.
        y: The target vector of the run.
        feature_index: The numeric-block-relative index of the best trial.

    Returns:
        The resolved feature name, or ``None``.
    """
    if preprocessor is None or n_numeric_eff is None:
        return None
    try:
        # The preprocessor used by the trials is only ever fitted through
        # clones (per fold), so fit a copy once on the full data to obtain
        # the stable output names (the numerical block width and the
        # ``num__<col>`` names are data-independent for the numeric block).
        names = preprocessor.get_feature_names_out()
    except Exception:  # noqa: BLE001 — best-effort name resolution
        try:
            probe = clone(preprocessor).fit(X, y)
            names = probe.get_feature_names_out()
        except Exception:  # noqa: BLE001 — best-effort name resolution
            return None
    names_list = list(names)
    n_cols = len(names_list)
    if n_cols < n_numeric_eff or feature_index >= n_numeric_eff:
        return None
    absolute = (n_cols - int(n_numeric_eff)) + feature_index
    return str(names_list[absolute])


def _fs_transformer_factory(
    fs_cfg: FeatureSelectionCfg, random_state: int | None
) -> FeatureSelector:
    """Create a fresh feature selector instance for a pipeline step.

    Each call returns a new ``FeatureSelector`` with a deterministic
    ``random_state`` (consistent with the final ModelTrainer training),
    which guarantees reproducible selection across trials and threads.
    With ``random_state=None`` (a fully non-deterministic Optuna run),
    selection still uses the fixed seed 42: otherwise each factory call
    would create a selector with a new random seed and selection would be
    irreproducible even with all other pipeline stages deterministic.

    Args:
        fs_cfg: The feature selection configuration.
        random_state: The run seed.

    Returns:
        FeatureSelector: A new, unfitted selector.
    """
    return FeatureSelector(
        method=fs_cfg.method.value,
        percentile=fs_cfg.percentile,
        min_features=fs_cfg.min_features,
        variance_threshold=fs_cfg.variance_threshold,
        n_estimators=fs_cfg.n_estimators,
        random_state=random_state if random_state is not None else 42,
    )
