"""Input data handling of the single-model trainer.

The data-side steps of ``ModelTrainer.fit``, all pure functions over explicit
inputs: separating and validating X/y (``load_training_data``), the input
diagnostics, the uninformative-feature cleaning and the column typing
(``prepare_features``), the preprocessor assembly through the shared
``preprocessing`` module and the preset table
(``build_trainer_preprocessor``), the isotonic adapter sizing, the assembly
of the pipeline steps (``assemble_pipeline_steps``) and the preparation of
the prediction input (``prepare_predict_input``). The module must not import
``configurable_automl_engine.trainer``.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.compose import ColumnTransformer

from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.common.settings import EncodingSettings, InputPolicies
from configurable_automl_engine.common.shared_frame import SharedDataFrame
from configurable_automl_engine.common.validation_utils import (
    apply_inf_policy,
    check_target_constant,
    check_target_numeric,
    datetime_expansion_map,
    expand_datetime_columns,
    validate_features,
)
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.isotonic import IsotonicDataTransformer
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
    _clean_uninformative_features,
    build_preprocessor,
    estimate_n_features_after_preprocessing,
)
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    requires_missing_indicator,
    resolve_preprocessing_preset,
)

from .models import requires_dense_input, resolve_algorithm_name
from .trainer_state import (
    PreparedFeatures,
    TrainerPreprocessor,
    TrainerSettings,
    TrainingData,
)

_LOG = logging.getLogger(__name__)


def extract_feature_names(X: Any) -> list[str] | None:
    """Extract column names from various data structures without copying.

    Args:
        X: The feature matrix (``SharedDataFrame``, ``pd.DataFrame``, a 2-D
            array or anything else).

    Returns:
        list[str] | None: The column names (``col_<i>`` for a 2-D array),
            or ``None`` when ``X`` has no columns.
    """
    if hasattr(X, "get_data_info"):
        return cast(list[str], X.get_data_info()["columns"])
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    if hasattr(X, "shape") and len(X.shape) > 1:
        return [str(f"col_{i}") for i in range(X.shape[1])]
    return None


def load_training_data(X: Any, y: Any) -> TrainingData:
    """Validate the input data types and separate features from the target.

    ``y`` may be a target column name of ``X`` (a ``pd.DataFrame``), a
    ``pd.Series`` / array, a single-column ``pd.DataFrame`` or a
    ``SharedDataFrame``. The target is validated like at the pipeline entry
    point (``prepare_dataset`` / ``hpo.runner._validate_data``): a
    non-numeric or non-finite y would otherwise fail late (inside the
    validation splits) with an unclear sklearn error, and a constant y makes
    training meaningless (every model trivially reports a perfect score).

    Args:
        X: The feature matrix.
        y: The target (vector or column name).

    Returns:
        TrainingData: The features, the target, the feature names and the
            dtypes of a data frame input.

    Raises:
        TrainingError: On an unsupported type, empty or mismatched data,
            fewer than 2 rows, or a non-numeric, non-finite or constant
            target.
    """
    try:
        feature_names = extract_feature_names(X)
        detected_dtypes = X.dtypes if isinstance(X, pd.DataFrame) else None
        # 1. Type validation
        valid_types = (pd.DataFrame, pd.Series, np.ndarray, SharedDataFrame)
        if not isinstance(X, valid_types):
            raise TrainingError(f"Unsupported data type: {type(X)}")

        X_obj: pd.DataFrame | pd.Series | np.ndarray
        y_obj: pd.Series | np.ndarray

        if isinstance(y, str):
            if not isinstance(X, pd.DataFrame):
                raise TrainingError(
                    f"Target column '{y}' specified, but X is not a DataFrame"
                )
            feature_names = [col for col in X.columns if col != y]
            X_obj = X[feature_names]
            y_obj = X[y]
        else:
            X_obj = X.get_view() if isinstance(X, SharedDataFrame) else X
            if isinstance(y, (pd.Series, np.ndarray)):
                y_obj = y
            elif isinstance(y, pd.DataFrame):
                y_obj = y.iloc[:, 0]
            elif hasattr(y, "get_view"):  # Support SharedDataFrame for y
                y_obj = y.get_view().iloc[:, 0]
            else:
                y_obj = np.asarray(y)
        # Consolidated validation
        n_samples = X_obj.shape[0] if hasattr(X_obj, "shape") else len(X_obj)
        if n_samples == 0:
            raise TrainingError("Data is empty")
        nx = X_obj.shape[0] if hasattr(X_obj, "shape") else len(X_obj)
        ny = y_obj.shape[0] if hasattr(y_obj, "shape") else len(y_obj)

        if nx != ny:
            raise TrainingError(f"Mismatched samples: X has {nx}, y has {ny}")
    except (ValueError, TypeError, IndexError) as e:
        if str(e) == "Data is empty":
            raise
        raise TrainingError(f"Data transformation error: {e}")

    _check_training_data(X_obj, y_obj)
    return TrainingData(
        X=X_obj, y=y_obj, feature_names=feature_names, detected_dtypes=detected_dtypes
    )


def _check_training_data(X: Any, y: Any) -> None:
    """Reject too few rows and a non-numeric or constant target.

    Raises:
        TrainingError: If there are fewer than 2 rows or the target is
            non-numeric, non-finite or constant.
    """
    n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
    if n_samples < 2:
        raise TrainingError("Insufficient records for training")
    try:
        check_target_numeric(y)
        check_target_constant(y)
    except ValueError as err:
        raise TrainingError(str(err)) from err


def prepare_features(
    X: Any,
    *,
    feature_names: list[str] | None,
    detected_dtypes: Any = None,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    id_column: str | None,
    policies: InputPolicies,
    warn_id_like: bool = True,
) -> PreparedFeatures:
    """Diagnose, repair and clean the training features; detect their types.

    For a ``pd.DataFrame``: the input-data diagnostics (duplicate columns,
    ±inf, datetime and ID-like columns, following ``policies``), the
    cleaning of uninformative features (fully empty and strictly constant
    columns, removed before type detection so that HPO and the final
    training use a consistent feature set) and the column typing (explicit
    lists are validated and reconciled; missing lists are auto-detected,
    excluding ``id_column``). For other inputs the columns are positional:
    every column is numerical. The input ``X`` and the lists are not
    mutated.

    Args:
        X: The training feature matrix.
        feature_names: The feature names of ``X`` (``TrainingData``).
        detected_dtypes: The dtypes of the input frame (passed through).
        categorical_features: The explicit categorical columns (``None`` —
            auto-detect).
        numerical_features: The explicit numerical columns (``None`` —
            auto-detect).
        id_column: An identifier column excluded from the type detection.
        policies: The input-data diagnostics policies.
        warn_id_like: Whether the ID-like WARNING is emitted.

    Returns:
        PreparedFeatures: The prepared matrix and its column lists.

    Raises:
        TrainingError: If a policy rejects the data or an explicit column is
            missing.
    """
    names = None if feature_names is None else list(feature_names)
    cats = None if categorical_features is None else list(categorical_features)
    nums = None if numerical_features is None else list(numerical_features)
    expansion: dict[str, list[str]] = {}
    dropped: list[str] = []
    if isinstance(X, pd.DataFrame):
        X, names, cats, nums, expansion = _apply_input_diagnostics(
            X, names, cats, nums, policies, warn_id_like
        )
        X, names, cats, nums, dropped = _drop_uninformative_features(
            X, names, cats, nums
        )
        cats, nums = _detect_feature_types(X, cats, nums, id_column)
    else:
        if cats is None:
            cats = []
        if nums is None:
            n_cols = X.shape[1] if len(X.shape) > 1 else 1
            nums = [f"col_{i}" for i in range(n_cols)]
        if names is None:
            names = nums
    if names is None:
        # If there are no names, try to extract them again or create an
        # empty list.
        names = extract_feature_names(X) or []
    return PreparedFeatures(
        X=X,
        feature_names=names,
        categorical_features=cats,
        numerical_features=nums,
        dropped_features=dropped,
        datetime_expansion=expansion,
        detected_dtypes=detected_dtypes,
    )


def _apply_input_diagnostics(
    X: pd.DataFrame,
    names: list[str] | None,
    cats: list[str] | None,
    nums: list[str] | None,
    policies: InputPolicies,
    warn_id_like: bool,
) -> tuple[
    pd.DataFrame,
    list[str] | None,
    list[str] | None,
    list[str] | None,
    dict[str, list[str]],
]:
    """Apply the input-data diagnostics policies to the training frame.

    The same checks as the pipeline entry point ``prepare_dataset``:
    duplicate columns, ±inf in numeric features, datetime/timedelta columns
    and ID-like columns are detected before cleaning/type detection.
    Duplicate names would surface later as an unclear pandas error, ±inf
    breaks distance/linear models, datetime columns are dropped by the
    preprocessor (remainder="drop"), and ID-like columns are usually
    uninformative.

    Returns:
        The repaired frame, the reconciled feature names and column lists,
        and the datetime expansion map.

    Raises:
        TrainingError: If a policy rejects the data.
    """
    try:
        X_checked, generated = validate_features(
            X,
            inf_policy=policies.inf_policy,
            datetime_policy=policies.datetime_policy,
            drop_id_like_columns=policies.drop_id_like_columns,
            warn_id_like=warn_id_like,
        )
    except ValueError as err:
        raise TrainingError(str(err)) from err
    expansion = datetime_expansion_map(X, generated)
    columns = X_checked.columns
    if names is not None:
        names = [name for name in names if name in columns]
    if cats is not None:
        cats = [f for f in cats if f in columns]
    if nums is not None:
        nums = [f for f in nums if f in columns]
    if generated:
        # The expansion happens BEFORE the feature names were fixed (they are
        # extracted from the original X), so the generated columns must be
        # appended to the feature list for the preprocessor.
        names = list(names or []) + generated
        # The generated components are numeric: they must join the
        # numerical feature list (or be added to it when the user provided
        # explicit lists), otherwise the preprocessor would not know how to
        # handle them.
        if nums is not None:
            nums = list(nums) + [name for name in generated if name not in nums]
    return X_checked, names, cats, nums, expansion


def _drop_uninformative_features(
    X: pd.DataFrame,
    names: list[str] | None,
    cats: list[str] | None,
    nums: list[str] | None,
) -> tuple[
    pd.DataFrame, list[str] | None, list[str] | None, list[str] | None, list[str]
]:
    """Remove fully empty and strictly constant columns from the training frame.

    Cleaning runs BEFORE feature type detection and preprocessor assembly,
    so that HPO and final training work with a consistent feature set (a
    single point shared with ``tuner.optimize``). The decision is made only
    on the training data (no leakage). The feature names are updated to the
    cleaned list: ``predict`` aligns columns by them, so removed columns are
    dropped automatically at inference, while missing ones produce an
    explicit error. User-specified column lists are reconciled too.

    Returns:
        The cleaned frame, the reconciled feature names and column lists,
        and the dropped columns.
    """
    cleaned = _clean_uninformative_features(X)
    dropped = [col for col in X.columns if col not in cleaned.columns]
    if not dropped:
        return X, names, cats, nums, dropped
    if names is not None:
        names = [name for name in names if name not in dropped]
    if cats is not None:
        cats = [f for f in cats if f not in dropped]
    if nums is not None:
        nums = [f for f in nums if f not in dropped]
    return cleaned, names, cats, nums, dropped


def _detect_feature_types(
    X: pd.DataFrame,
    cats: list[str] | None,
    nums: list[str] | None,
    id_column: str | None,
) -> tuple[list[str], list[str]]:
    """Automatically classify columns into numerical and categorical ones.

    Explicit lists given for both kinds are only validated against ``X``;
    a missing list is detected from the dtypes (excluding ``id_column``).

    Returns:
        The categorical and the numerical columns.

    Raises:
        TrainingError: If an explicitly specified column is missing.
    """
    if cats is not None and nums is not None:
        missing = [col for col in cats + nums if col not in X.columns]
        if missing:
            raise TrainingError(f"Specified columns not found in data: {missing}")
        return cats, nums
    # Determine the column list for analysis (excluding the target and ID)
    exclude = {""}
    if id_column:
        exclude.add(id_column)
    df_to_analyze = X.drop(columns=list(exclude & set(X.columns)))
    # Auto-detect categorical features (object, category, bool)
    if cats is None:
        cats = df_to_analyze.select_dtypes(
            include=["object", "str", "category", "bool"]
        ).columns.tolist()
    # Auto-detect numerical features (all remaining 'number' dtypes)
    if nums is None:
        nums = df_to_analyze.select_dtypes(include=["number"]).columns.tolist()
    _LOG.info(f"Auto-detected features: {len(cats)} cat, {len(nums)} num.")
    return cats, nums


def build_trainer_preprocessor(
    algorithm: str,
    categorical_features: list[str],
    numerical_features: list[str],
    *,
    feature_names: list[str],
    encoding: EncodingSettings,
    override: PreprocessingOverride | None,
    random_state: int | None,
) -> TrainerPreprocessor:
    """Build the ColumnTransformer of the trainer and resolve its preset.

    Delegates assembly to the shared :mod:`preprocessing` module, which is
    also used by the HPO phase in ``tuner.optimize`` (DRY, single logic).
    The numerical feature handling strategy (imputation and scaling) is
    selected automatically based on the algorithm via an adaptive
    preprocessing preset (plus the explicit override) and is logged.

    Args:
        algorithm: The algorithm name.
        categorical_features: The categorical columns.
        numerical_features: The numerical columns.
        feature_names: The feature names in training order.
        encoding: The categorical encoding settings.
        override: The explicit preprocessing preset override.
        random_state: The seed of the randomized encoders.

    Returns:
        TrainerPreprocessor: The unfitted preprocessor and the preset.
    """
    matched = [col for col in categorical_features if col in feature_names] + [
        col for col in numerical_features if col in feature_names
    ]
    if not matched:
        _LOG.warning(
            "No features matched for preprocessing. Defaulting to passthrough."
        )
    preset = resolve_preprocessing_preset(algorithm, override)
    _LOG.info(
        "Resolved preprocessing preset for algorithm '%s': %s",
        algorithm,
        preset,
    )
    # GPR/Isotonic/ARD algorithms reject sparse matrices: with hashing
    # encoding (sparse output) the preprocessor must return a dense matrix
    # via force_dense_output. Strictly univariate algorithms (isotonic) are
    # incompatible with missing-value indicator columns; an empty algorithm
    # name uses the safe default (True).
    preprocessor = build_preprocessor(
        feature_names,
        categorical_features,
        numerical_features,
        encoding=cast(EncodingStrategy, encoding.encoding),
        imputation_strategy=preset.imputation_strategy,
        scaling=preset.scaling,
        high_cardinality_threshold=encoding.high_cardinality_threshold,
        high_cardinality_encoding=encoding.high_cardinality_encoding,
        hashing_n_components=encoding.hashing_n_components,
        target_encoding_smoothing=encoding.target_encoding_smoothing,
        target_encoding_fallback=encoding.target_encoding_fallback,
        random_state=random_state,
        force_dense_output=requires_dense_input(algorithm),
        add_indicator=requires_missing_indicator(algorithm) if algorithm else True,
    )
    return TrainerPreprocessor(preprocessor=preprocessor, preset=preset)


def _resolve_isotonic_feature_index(hyperparams: dict[str, Any]) -> int:
    """Resolve the numerical feature selected by the isotonic data adapter.

    The ``feature_index`` value is a user/HPO hyperparameter (not an
    ``IsotonicRegression`` constructor parameter): it is popped from the
    model kwargs before ``create_model`` and consumed here by
    ``IsotonicDataTransformer``. The index is relative to the numerical
    feature block of the preprocessed matrix (after imputation/scaling);
    one-hot dummies of categorical columns are not meaningful inputs for
    the strictly univariate isotonic model.

    Args:
        hyperparams: The trainer hyperparameters.

    Returns:
        The ordinal index of the numerical feature to feed to
        ``IsotonicRegression``.

    Raises:
        TrainingError: If ``feature_index`` is not a non-negative integer.
    """
    raw = hyperparams.get("feature_index", 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise TrainingError(
            "feature_index must be a non-negative integer for "
            f"IsotonicRegression, got {raw!r}"
        )
    return raw


def _isotonic_n_numeric(numerical_features: list[str] | None) -> int:
    """Number of numerical features after preprocessing for isotonic.

    For the strictly univariate algorithms the preprocessor is built with
    ``add_indicator=False`` (``requires_missing_indicator`` returns
    ``False``) and numeric columns are never expanded, so the numerical
    block of the preprocessed matrix has exactly one column per original
    numerical feature.

    Returns:
        The number of numerical features (>= 0).
    """
    return len(numerical_features) if numerical_features else 0


def _isotonic_n_total_estimate(
    X: Any,
    *,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    encoding: EncodingSettings,
) -> int:
    """Estimate the total post-preprocessing feature count for the guard.

    The same deterministic logic the tuner uses to size the isotonic
    search space, applied to the trainer's own feature lists and data so
    that the adapter can reject a divergence between the estimate and the
    fitted preprocessor output with a non-fatal ``ValueError``.

    Args:
        X: The prepared feature matrix.
        categorical_features: The categorical columns.
        numerical_features: The numerical columns.
        encoding: The categorical encoding settings.

    Returns:
        The estimated total column count after preprocessing (>= 1).
    """
    return estimate_n_features_after_preprocessing(
        X,
        categorical_features=list(categorical_features or []),
        numerical_features=list(numerical_features or []),
        encoding=cast(EncodingStrategy, encoding.encoding),
        high_cardinality_threshold=encoding.high_cardinality_threshold,
        high_cardinality_encoding=encoding.high_cardinality_encoding,
        hashing_n_components=int(encoding.hashing_n_components),
        add_indicator=False,
    )


def log_preprocessed_matrix(preprocessor: Any, X: Any) -> None:
    """Log the shape and the storage format of the preprocessed matrix.

    Emits the INFO line ``preprocessed matrix: n × p (sparse/dense)`` for
    the training data after the final fit: ``n`` — rows, ``p`` — the
    number of features the model saw (after one-hot expansion /
    indicators), and the storage representation (``sparse`` for a
    ``scipy.sparse`` output — e.g. sparse one-hot/hashing — and
    ``dense`` otherwise). Used to make the categorical dimensionality
    visible in the logs (dense one-hot on high-cardinality
    columns inflated the matrix without any log signal).

    Args:
        preprocessor: The fitted ``ColumnTransformer``.
        X: The training feature matrix.
    """
    try:
        matrix = preprocessor.transform(X)
        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        fmt = "sparse" if sp.issparse(matrix) else "dense"
        _LOG.info("preprocessed matrix: %d × %d (%s)", n_rows, n_cols, fmt)
    except Exception as err:  # noqa: BLE001 — observability only
        _LOG.debug("Could not log the preprocessed matrix dimensions: %s", err)


def _align_predict_columns(
    feature_names: list[str] | None, X: pd.DataFrame
) -> pd.DataFrame:
    """Align the DataFrame columns to the training order.

    A defensive layer for ``predict``: validates the presence of all
    ``feature_names``, drops extra columns, and returns a DataFrame with
    columns in the training order. This also fixes legacy serialized models
    whose preprocessor uses positional indices: after alignment, the
    positional slice again points to the same columns as during training.

    Args:
        feature_names: The training feature names.
        X: Input DataFrame with the features to predict on.

    Returns:
        A DataFrame with columns in the training order.

    Raises:
        TrainingError: If some of ``feature_names`` are missing in ``X``
            or appear in ``X`` more than once.
    """
    if not feature_names:
        return X
    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise TrainingError(
            f"Missing columns in prediction data: {missing}. "
            f"Expected columns: {feature_names}."
        )
    duplicated = [col for col in feature_names if list(X.columns).count(col) > 1]
    if duplicated:
        raise TrainingError(
            f"Duplicated columns in prediction data: {duplicated}. "
            "Column selection by name is ambiguous."
        )
    return X[feature_names]


def _align_predict_shared_dataframe(
    feature_names: list[str] | None, X: SharedDataFrame
) -> Any:
    """Prepare a SharedDataFrame for prediction.

    If the container stores column names, the data is restored into a
    DataFrame and aligned to the training order (the same guarantees as for
    a regular ``pd.DataFrame``); otherwise a positional numpy fallback in
    the training order is used.

    Args:
        feature_names: The training feature names.
        X: Data container in shared memory.

    Returns:
        A ``pd.DataFrame`` with columns in the training order, or a
        ``np.ndarray`` from shared memory if column names are unavailable.

    Raises:
        TrainingError: If some of ``feature_names`` are missing from the
            container columns.
    """
    if X.columns is None or not feature_names:
        return X.shared_array
    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise TrainingError(
            f"Missing columns in prediction data: {missing}. "
            f"Expected columns: {feature_names}."
        )
    return X.get_view(feature_names)


def prepare_predict_input(
    X: Any,
    feature_names: list[str] | None,
    datetime_expansion: Any,
    inf_policy: str,
) -> Any:
    """Bring the input of ``ModelTrainer.predict`` to the training schema.

    A ``pd.DataFrame`` gets its datetime columns expanded as at training
    time and its columns aligned to the training order; a
    ``SharedDataFrame`` is restored and aligned the same way; arrays pass
    as they are. The training ``inf_policy`` is applied to frames and
    numeric arrays, so inference is symmetric with ``fit``.

    Args:
        X: The feature matrix to predict on.
        feature_names: The training feature order (``None`` if unknown).
        datetime_expansion: The datetime expansion recorded at fit time
            (``ModelTrainer.datetime_expansion_``).
        inf_policy: The ±inf policy of the trainer.

    Returns:
        Any: The input for ``pipeline.predict``.

    Raises:
        TrainingError: If ``inf_policy`` is ``"error"`` and ``X`` contains
            ±inf (the message names the columns).
    """
    # Align columns for inputs with names; for the rest, avoid
    # conversion if X is already an array or DataFrame
    X_input: Any
    if isinstance(X, pd.DataFrame):
        X_input = _align_predict_columns(
            feature_names,
            expand_datetime_columns(X, datetime_expansion),
        )
    elif isinstance(X, SharedDataFrame):
        X_input = _align_predict_shared_dataframe(feature_names, X)
    elif isinstance(X, (pd.Series, np.ndarray)):
        X_input = X
    else:
        X_input = np.asarray(X)

    # The same ±inf policy as in fit(): without it a frame that
    # trained fine (inf → NaN) would fail at prediction time with
    # an sklearn "Input X contains infinity" error.
    if isinstance(X_input, (pd.DataFrame, np.ndarray)):
        try:
            X_input = apply_inf_policy(X_input, inf_policy)
        except ValueError as err:
            raise TrainingError(str(err)) from err
    return X_input


def assemble_pipeline_steps(
    settings: TrainerSettings,
    preprocessor: ColumnTransformer,
    base_model: Any,
    *,
    categorical_features: list[str] | None,
    numerical_features: list[str] | None,
    feature_selection_active: bool,
    X_for_estimate: Any = None,
) -> list[tuple[str, Any]]:
    """Assemble the list of training pipeline steps.

    Strict step order:
    ``preprocessor → [isotonic_selector] → [feature_selector] → [oversampler] → model``.

    For ``isotonic_regression`` a data adapter step
    (:class:`IsotonicDataTransformer`) is inserted right after the
    preprocessor: the algorithm is strictly univariate and accepts exactly
    one feature, so even a multi-column preprocessed matrix is reduced to
    the single numerical column selected via the ``feature_index``
    hyperparameter. The adapter receives the number of numerical features
    (``n_numeric``) to resolve the index against the numerical block of
    the preprocessed matrix and the estimated total feature count to
    validate against the actual preprocessor output (a disagreement
    raises ``ValueError`` — non-fatal for the HPO circuit breaker).
    The feature selector is inserted strictly between the preprocessor and
    the oversampler so that synthetic rows are generated only for
    informative features.

    Args:
        settings: The trainer settings.
        preprocessor: Feature preprocessing ``ColumnTransformer``.
        base_model: Final regressor.
        categorical_features: The categorical columns of the trainer.
        numerical_features: The numerical columns of the trainer.
        feature_selection_active: Flag enabling the feature selection step.
        X_for_estimate: The feature matrix used to estimate the total
            post-preprocessing feature count for the isotonic guard
            (``None`` — the count is disabled).

    Returns:
        A list of ``(name, transformer)`` tuples for building the Pipeline.
    """
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if resolve_algorithm_name(settings.algorithm) == "isotonic_regression":
        steps.append(
            (
                "isotonic_selector",
                IsotonicDataTransformer(
                    feature_index=_resolve_isotonic_feature_index(settings.hyperparams),
                    n_numeric=_isotonic_n_numeric(numerical_features),
                    n_total=(
                        _isotonic_n_total_estimate(
                            X_for_estimate,
                            categorical_features=categorical_features,
                            numerical_features=numerical_features,
                            encoding=settings.encoding,
                        )
                        if X_for_estimate is not None
                        else None
                    ),
                ),
            )
        )
    if feature_selection_active:
        cfg = settings.feature_selection_cfg
        selector = FeatureSelector(
            method=cfg.method.value,
            percentile=cfg.percentile,
            min_features=cfg.min_features,
            variance_threshold=cfg.variance_threshold,
            n_estimators=cfg.n_estimators,
            # Fixed fallback seed for selection (consistent with the tuner
            # and the auto-check): with random_state=None in the main
            # training, the selector must still use the fixed seed 42,
            # otherwise selection is not reproducible between fit() calls
            # and diverges from the HPO phase.
            random_state=(
                settings.random_state if settings.random_state is not None else 42
            ),
        )
        steps.append(("feature_selector", selector))
    oversampling = settings.oversampling
    if oversampling.enable:
        oversampler = DataOversampler(
            algorithm=oversampling.algorithm,
            multiplier=oversampling.multiplier,
            target_bins=oversampling.target_bins,
            add_noise=oversampling.add_noise,
            noise_level=float(oversampling.noise_level),
        )
        steps.append(("oversampler", oversampler))
    steps.append(("model", base_model))
    return steps
