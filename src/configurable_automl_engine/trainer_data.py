"""Input data handling of the single-model trainer.

The data-side steps of ``ModelTrainer``: separating and validating X/y,
detecting the column types, assembling the preprocessor (through the shared
``preprocessing`` module and the preset table), the input diagnostics and
uninformative-feature cleaning of ``fit()``, the isotonic adapter sizing,
and the prediction-time column alignment. ``ModelTrainer`` keeps thin
methods for the steps that are part of its private interface. The module
must not import ``configurable_automl_engine.trainer`` at runtime (the
``ModelTrainer`` type is imported for type checking only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.compose import ColumnTransformer

from configurable_automl_engine.common.validation_utils import (
    check_target_constant,
    check_target_numeric,
    datetime_expansion_map,
    validate_features,
)
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
    _clean_uninformative_features,
    build_preprocessor,
    estimate_n_features_after_preprocessing,
)
from configurable_automl_engine.preprocessing_presets import (
    requires_missing_indicator,
)
from configurable_automl_engine.training_engine.thread_pool import SharedDataFrame

from .models import requires_dense_input
from .trainer_errors import TrainingError

if TYPE_CHECKING:
    from configurable_automl_engine.trainer import ModelTrainer


def _detect_feature_types(
    trainer: ModelTrainer, X: pd.DataFrame, target_column: str
) -> None:
    """Automatically classify columns into numerical and categorical ones."""
    with trainer.lock:
        # If both lists are already filled by the user, just validate them
        if (
            trainer.categorical_features is not None
            and trainer.numerical_features is not None
        ):
            trainer._validate_features(X)
            return
        # Determine the column list for analysis (excluding the target and ID)
        exclude = {target_column}
        if trainer.id_column:
            exclude.add(trainer.id_column)

        df_to_analyze = X.drop(columns=list(exclude & set(X.columns)))
        # Auto-detect categorical features (object, category, bool)
        if trainer.categorical_features is None:
            trainer.categorical_features = df_to_analyze.select_dtypes(
                include=["object", "str", "category", "bool"]
            ).columns.tolist()
        # Auto-detect numerical features (all remaining 'number' dtypes)
        if trainer.numerical_features is None:
            trainer.numerical_features = df_to_analyze.select_dtypes(
                include=["number"]
            ).columns.tolist()
        trainer.logger.info(
            f"Auto-detected features: {len(trainer.categorical_features)} cat, "
            f"{len(trainer.numerical_features)} num."
        )


def _build_preprocessor(
    trainer: ModelTrainer, feature_names: list[str]
) -> ColumnTransformer:
    """Build a ColumnTransformer for separate handling of data types.

    Delegates assembly to the shared :mod:`preprocessing` module, which is
    also used by the HPO phase in ``tuner.optimize`` (DRY, single logic).
    The numerical feature handling strategy (imputation and scaling) is
    selected automatically based on the algorithm via an adaptive
    preprocessing preset and is logged.
    """
    # Log a warning if no column matched
    cat_indices = [
        feature_names.index(col)
        for col in (trainer.categorical_features or [])
        if col in feature_names
    ]
    num_indices = [
        feature_names.index(col)
        for col in (trainer.numerical_features or [])
        if col in feature_names
    ]
    if not cat_indices and not num_indices:
        trainer.logger.warning(
            "No features matched for preprocessing. Defaulting to passthrough."
        )
    # Backward compatibility: models serialized before the encoding_strategy
    # parameter existed (the attribute is absent on unpickled objects) are
    # treated as one_hot by default. The new encoding parameters also get
    # their default values via getattr so that old saved models continue to
    # work without behavior changes.
    encoding: EncodingStrategy = cast(
        EncodingStrategy, getattr(trainer, "encoding_strategy", "one_hot")
    )
    hc_threshold: int | None = getattr(trainer, "high_cardinality_threshold", None)
    hc_encoding_raw: str | None = getattr(trainer, "high_cardinality_encoding", None)
    hc_encoding: EncodingStrategy | None = (
        cast(EncodingStrategy, hc_encoding_raw) if hc_encoding_raw is not None else None
    )
    hashing_n_components: int = getattr(trainer, "hashing_n_components", 16)
    target_smoothing: float = getattr(trainer, "target_encoding_smoothing", 20.0)
    target_fallback: float | None = getattr(trainer, "target_encoding_fallback", None)
    # Automatic preprocessing preset selection by algorithm, taking into
    # account the explicit override. The preset is resolved once before
    # training (performance) and recorded for observability.
    preset = trainer._resolve_preprocessing_preset()
    trainer.preprocessing_preset = preset
    trainer.logger.info(
        "Resolved preprocessing preset for algorithm '%s': %s",
        trainer.algorithm,
        preset,
    )
    # GPR/Isotonic/ARD algorithms reject sparse matrices: with hashing
    # encoding (sparse output) the preprocessor must return a dense matrix
    # via force_dense_output.
    force_dense = requires_dense_input(trainer.algorithm)
    # Strictly univariate algorithms (isotonic) are incompatible with
    # missing-value indicator columns. If the algorithm name is empty or
    # not set, a safe default value (True) is used.
    use_missing_indicator = (
        requires_missing_indicator(trainer.algorithm) if trainer.algorithm else True
    )
    return build_preprocessor(
        feature_names,
        trainer.categorical_features or [],
        trainer.numerical_features or [],
        encoding=encoding,
        imputation_strategy=preset.imputation_strategy,
        scaling=preset.scaling,
        high_cardinality_threshold=hc_threshold,
        high_cardinality_encoding=hc_encoding,
        hashing_n_components=hashing_n_components,
        target_encoding_smoothing=target_smoothing,
        target_encoding_fallback=target_fallback,
        random_state=trainer.random_state,
        force_dense_output=force_dense,
        add_indicator=use_missing_indicator,
    )


def _prepare_data(trainer: ModelTrainer, X: Any, y: Any) -> tuple[Any, Any]:
    """Validate the input data types
    and separate features from the target variable."""
    try:
        trainer.feature_names = trainer._extract_metadata(X)

        trainer._detected_dtypes = X.dtypes if isinstance(X, pd.DataFrame) else None
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
            trainer.feature_names = [col for col in X.columns if col != y]
            X_obj = X[trainer.feature_names]
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

        return X_obj, y_obj
    except (ValueError, TypeError, IndexError) as e:
        if str(e) == "Data is empty":
            raise
        raise TrainingError(f"Data transformation error: {e}")


def _resolve_isotonic_feature_index(trainer: ModelTrainer) -> int:
    """Resolve the numerical feature selected by the isotonic data adapter.

    The ``feature_index`` value is a user/HPO hyperparameter (not an
    ``IsotonicRegression`` constructor parameter): it is popped from the
    model kwargs before ``create_model`` and consumed here by
    ``IsotonicDataTransformer``. The index is relative to the numerical
    feature block of the preprocessed matrix (after imputation/scaling);
    one-hot dummies of categorical columns are not meaningful inputs for
    the strictly univariate isotonic model.

    Returns:
        The ordinal index of the numerical feature to feed to
        ``IsotonicRegression``.

    Raises:
        TrainingError: If ``feature_index`` is not a non-negative integer.
    """
    raw = trainer.hyperparams.get("feature_index", 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise TrainingError(
            "feature_index must be a non-negative integer for "
            f"IsotonicRegression, got {raw!r}"
        )
    return raw


def _isotonic_n_numeric(trainer: ModelTrainer) -> int:
    """Number of numerical features after preprocessing for isotonic.

    For the strictly univariate algorithms the preprocessor is built with
    ``add_indicator=False`` (``requires_missing_indicator`` returns
    ``False``) and numeric columns are never expanded, so the numerical
    block of the preprocessed matrix has exactly one column per original
    numerical feature.

    Returns:
        The number of numerical features (>= 0).
    """
    nums = getattr(trainer, "numerical_features", None)
    return len(nums) if nums else 0


def _isotonic_n_total_estimate(trainer: ModelTrainer, X: Any) -> int:
    """Estimate the total post-preprocessing feature count for the guard.

    The same deterministic logic the tuner uses to size the isotonic
    search space, applied to the trainer's own feature lists and data so
    that the adapter can reject a divergence between the estimate and the
    fitted preprocessor output with a non-fatal ``ValueError``.

    Args:
        X: The prepared feature matrix (full data).

    Returns:
        The estimated total column count after preprocessing (>= 1).
    """
    return estimate_n_features_after_preprocessing(
        X,
        categorical_features=list(getattr(trainer, "categorical_features", None) or []),
        numerical_features=list(getattr(trainer, "numerical_features", None) or []),
        encoding=cast(
            EncodingStrategy, getattr(trainer, "encoding_strategy", "one_hot")
        ),
        high_cardinality_threshold=getattr(trainer, "high_cardinality_threshold", None),
        high_cardinality_encoding=getattr(trainer, "high_cardinality_encoding", None),
        hashing_n_components=int(getattr(trainer, "hashing_n_components", 16)),
        add_indicator=False,
    )


def _log_preprocessed_matrix(trainer: ModelTrainer, preprocessor: Any, X: Any) -> None:
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
        trainer.logger.info("preprocessed matrix: %d × %d (%s)", n_rows, n_cols, fmt)
    except Exception as err:  # noqa: BLE001 — observability only
        trainer.logger.debug(
            "Could not log the preprocessed matrix dimensions: %s", err
        )


def _align_predict_columns(trainer: ModelTrainer, X: pd.DataFrame) -> pd.DataFrame:
    """Align the DataFrame columns to the training order.

    A defensive layer for ``predict``: validates the presence of all
    ``trainer.feature_names``, drops extra columns, and returns a DataFrame
    with columns in the training order. This also fixes legacy serialized
    models whose preprocessor uses positional indices: after alignment, the
    positional slice again points to the same columns as during training.

    Args:
        X: Input DataFrame with the features to predict on.

    Returns:
        A DataFrame with columns in the training order (``trainer.feature_names``).

    Raises:
        TrainingError: If some of ``trainer.feature_names`` are missing in ``X``
            or appear in ``X`` more than once.
    """
    feature_names = trainer.feature_names
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


def _align_predict_shared_dataframe(trainer: ModelTrainer, X: SharedDataFrame) -> Any:
    """Prepare a SharedDataFrame for prediction.

    If the container stores column names, the data is restored into a
    DataFrame and aligned to the training order (the same guarantees as for
    a regular ``pd.DataFrame``); otherwise a positional numpy fallback in
    the training order is used.

    Args:
        X: Data container in shared memory.

    Returns:
        A ``pd.DataFrame`` with columns in the training order, or a
        ``np.ndarray`` from shared memory if column names are unavailable.

    Raises:
        TrainingError: If some of ``trainer.feature_names`` are missing from
            the container columns.
    """
    if X.columns is None or not trainer.feature_names:
        return X.shared_array
    missing = [col for col in trainer.feature_names if col not in X.columns]
    if missing:
        raise TrainingError(
            f"Missing columns in prediction data: {missing}. "
            f"Expected columns: {trainer.feature_names}."
        )
    return X.get_view(trainer.feature_names)


def _reset_fit_state(trainer: ModelTrainer) -> None:
    """Reset the training state of the trainer at the start of ``fit()``."""
    # Stage 0: Reset the training state at the start of fit(), before
    # any data processing. If training fails at any stage (data
    # preparation, activity resolution, validation scoring, pipeline.fit),
    # the trainer must not retain values from the previous training
    # (the negative scenario with every fold failing). The duplicate
    # reset in _fit_internal is kept for direct calls to that method.
    trainer.pipeline = None
    trainer.val_score = None
    trainer.additional_scores = {}
    # OOF channel: reset together with the rest of the state so that a
    # failed fit() does not leave values from the previous training.
    trainer.oof_score_ = None
    trainer.oof_predictions_ = None
    trainer.oof_coverage_ = None
    trainer.feature_selection_active_ = False
    trainer.selected_features_mask_ = None
    trainer.dropped_features_ = []
    trainer.datetime_expansion_ = {}


def _check_training_data(X_prepared: Any, y_s: Any) -> None:
    """Reject too few rows and a non-numeric or constant target.

    Raises:
        TrainingError: If there are fewer than 2 rows or the target is
            non-numeric, non-finite or constant.
    """
    n_samples = X_prepared.shape[0] if hasattr(X_prepared, "shape") else len(X_prepared)
    if n_samples < 2:
        raise TrainingError("Insufficient records for training")

    # Target guards — the same checks as at the pipeline entry point
    # (prepare_dataset / tuner._validate_data): a non-numeric or
    # non-finite y would otherwise fail late (inside the validation
    # splits) with an unclear sklearn error, and a constant y makes
    # training meaningless (every model trivially reports a perfect
    # score).
    try:
        check_target_numeric(y_s)
        check_target_constant(y_s)
    except ValueError as err:
        raise TrainingError(str(err)) from err


def _prepare_training_features(trainer: ModelTrainer, X_prepared: Any) -> Any:
    """Diagnose, repair and clean the training features; detect their types.

    Args:
        trainer: The trainer being fitted (its feature lists are updated).
        X_prepared: The prepared feature matrix.

    Returns:
        The feature matrix the pipeline is trained on.
    """
    if isinstance(X_prepared, pd.DataFrame):
        X_prepared = _apply_input_diagnostics(trainer, X_prepared)
        X_prepared = _drop_uninformative_features(trainer, X_prepared)
        trainer._detect_feature_types(X_prepared, target_column="")
    else:
        if trainer.categorical_features is None:
            trainer.categorical_features = []
        if trainer.numerical_features is None:
            n_cols = X_prepared.shape[1] if len(X_prepared.shape) > 1 else 1
            trainer.numerical_features = [f"col_{i}" for i in range(n_cols)]
        if trainer.feature_names is None:
            trainer.feature_names = trainer.numerical_features
    return X_prepared


def _apply_input_diagnostics(
    trainer: ModelTrainer, X_prepared: pd.DataFrame
) -> pd.DataFrame:
    """Apply the input-data diagnostics policies to the training frame.

    Args:
        trainer: The trainer being fitted (its feature lists are updated).
        X_prepared: The training frame.

    Returns:
        The repaired training frame.

    Raises:
        TrainingError: If a policy rejects the data.
    """
    # Input-data diagnostics (the same checks as the pipeline entry
    # point prepare_dataset): duplicate columns, ±inf in numeric
    # features, datetime/timedelta columns and ID-like columns are
    # detected before cleaning/type detection. Duplicate names
    # would surface later as an unclear pandas error, ±inf breaks
    # distance/linear models, datetime columns are dropped by the
    # preprocessor (remainder="drop"), and ID-like columns are
    # usually uninformative.
    try:
        X_checked, generated = validate_features(
            X_prepared,
            inf_policy=str(getattr(trainer, "inf_policy", "as_missing")),
            datetime_policy=str(getattr(trainer, "datetime_policy", "expand")),
            drop_id_like_columns=bool(getattr(trainer, "drop_id_like_columns", False)),
            warn_id_like=bool(getattr(trainer, "warn_id_like", True)),
        )
    except ValueError as err:
        raise TrainingError(str(err)) from err
    trainer.datetime_expansion_ = datetime_expansion_map(X_prepared, generated)
    X_prepared = X_checked
    if trainer.feature_names is not None:
        trainer.feature_names = [
            name for name in trainer.feature_names if name in X_prepared.columns
        ]
    for attr in ("categorical_features", "numerical_features"):
        features = getattr(trainer, attr)
        if features is not None:
            setattr(
                trainer,
                attr,
                [f for f in features if f in X_prepared.columns],
            )
    if generated:
        # The expansion happens BEFORE trainer.feature_names was
        # fixed (it is extracted in _prepare_data from the
        # original X), so the generated columns must be appended
        # to the feature list for the preprocessor.
        trainer.feature_names = list(trainer.feature_names or []) + generated
        # The generated components are numeric: they must join the
        # numerical feature list (or be added to it when the user
        # provided explicit lists), otherwise the preprocessor
        # would not know how to handle them.
        if trainer.numerical_features is not None:
            trainer.numerical_features = list(trainer.numerical_features) + [
                name for name in generated if name not in trainer.numerical_features
            ]
    return X_prepared


def _drop_uninformative_features(
    trainer: ModelTrainer, X_prepared: pd.DataFrame
) -> pd.DataFrame:
    """Remove fully empty and strictly constant columns from the training frame.

    Args:
        trainer: The trainer being fitted (its feature lists are updated).
        X_prepared: The training frame.

    Returns:
        The cleaned training frame.
    """
    # Cleaning uninformative features: fully empty and strictly
    # constant columns are removed BEFORE feature type detection and
    # preprocessor assembly, so that HPO and final training work
    # with a consistent feature set (a single point shared with
    # tuner.optimize). The decision is made only on the training
    # data (no leakage); for np.ndarray without column names no
    # cleaning is performed (a documented limitation, as in
    # detect_feature_types).
    cleaned = _clean_uninformative_features(X_prepared)
    trainer.dropped_features_ = [
        col for col in X_prepared.columns if col not in cleaned.columns
    ]
    if trainer.dropped_features_:
        X_prepared = cleaned
        # Update feature_names to the cleaned list: predict()
        # aligns columns by trainer.feature_names, so removed columns
        # are dropped automatically at inference, while missing ones
        # produce an explicit error.
        if trainer.feature_names is not None:
            trainer.feature_names = [
                name
                for name in trainer.feature_names
                if name not in trainer.dropped_features_
            ]
        # User-specified column lists must also be consistent with
        # the cleaned feature set: otherwise
        # _detect_feature_types/build_preprocessor would fail on
        # missing columns.
        for attr in ("categorical_features", "numerical_features"):
            features = getattr(trainer, attr)
            if features is not None:
                setattr(
                    trainer,
                    attr,
                    [
                        feature
                        for feature in features
                        if feature not in trainer.dropped_features_
                    ],
                )
    return X_prepared
