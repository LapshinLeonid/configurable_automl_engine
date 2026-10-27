"""Model Trainer: A universal orchestrator of the regression model lifecycle.

This module encapsulates the logic for building training pipelines,
combining data preprocessing, sample balancing, and training of Scikit-learn algorithms.

A key feature is the full automation of feature preparation:
the system classifies data types on its own,
handles missing values and scales features,
guaranteeing correct data delivery to the final estimator.

Main components:

ModelTrainer: The main container class for model training, validation, and deployment.
train_model: A functional interface (Facade) that provides
    backward compatibility with the legacy API.
TrainingError: A dedicated exception
    for unified handling of training failures.

Implementation highlights:

Pipeline Integration: Using imblearn.pipeline allows seamless integration
    of oversampling mechanisms (SMOTE, ADASYN) directly into the training process.
Feature Selection: An optional feature selection step (FeatureSelector) is
    inserted strictly between the preprocessor and the oversampler; its activity
    is controlled by an explicit flag (HPO result) or by the configuration mode
    (always/disabled/auto). Mode ``auto`` performs a standalone feasibility check
    on each fit() (two additional pipeline fits, ~2-3x time), so within an HPO
    loop an explicit ``feature_selection_active`` flag is recommended.
Automated Feature Engineering: The built-in ColumnTransformer automatically applies
    One-Hot encoding for categorical features and StandardScaler for numerical ones.
Thread Safety: Recursive locks (threading.RLock) guarantee safety
    when accessing the model state from multiple threads.
Isotonic Support: A dedicated IsotonicDataTransformer adapter transforms
    multidimensional data to meet the strict requirements of isotonic regression algorithms.
Persistence: Built-in save and load methods support various serialization
    formats (Pickle, Joblib) through the artifact system."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from configurable_automl_engine.common.definitions import (
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.serialization_utils import (
    load_artifact,
    resolve_load_formats,
    save_artifact,
)
from configurable_automl_engine.common.validation_utils import (
    apply_inf_policy,
    expand_datetime_columns,
)
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
)
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    PreprocessingPreset,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
)
from configurable_automl_engine.training_engine.logger import setup_logging
from configurable_automl_engine.training_engine.metrics import (
    is_error_metric,
)
from configurable_automl_engine.training_engine.thread_pool import SharedDataFrame

from .common.artifact_metadata import (
    _collect_artifact_metadata,
    _warn_on_version_mismatch,
)
from .models import (
    create_model,
    resolve_algorithm_name,
)
from .trainer_data import (
    _align_predict_columns,
    _align_predict_shared_dataframe,
    _build_preprocessor,
    _check_training_data,
    _detect_feature_types,
    _isotonic_n_numeric,
    _isotonic_n_total_estimate,
    _log_preprocessed_matrix,
    _prepare_data,
    _prepare_training_features,
    _reset_fit_state,
    _resolve_isotonic_feature_index,
)
from .trainer_errors import TrainingError
from .trainer_feature_selection import (
    _resolve_feature_selection_active,
    _run_feature_selection_auto_check,
    _validate_feature_selection_cfg,
)
from .trainer_params import (
    _normalize_additional_metrics,
    _validate_algorithm,
    _validate_encoding_strategy,
    _validate_feature_lists,
    _validate_feature_selection_active,
    _validate_high_cardinality,
    _validate_hyperparams,
    _validate_input_policies,
    _validate_oversampling,
    _validate_preprocessing_override,
    _validate_test_size,
    _validate_validation_strategy,
)
from .trainer_scoring import (
    _align_oof_vector,
    _finalize_oof,
    _score_on_validation_splits,
)

__all__ = ["ModelTrainer", "TrainingError", "train_model"]


class IsotonicDataTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Transformer for preparing data for IsotonicRegression.
    Selects the specified numerical feature and ensures correct handling of
    missing values to meet the requirements of the isotonic regression
    algorithm.

    The preprocessor (``build_preprocessor``) always emits the categorical
    block (one-hot dummies etc.) first and the numerical block
    (imputation/scaling) last. Only the numerical features are meaningful
    inputs for the strictly univariate isotonic model, so the adapter treats
    ``feature_index`` as an index RELATIVE to the numerical block of the
    input matrix and maps it to the absolute column via
    ``(n_columns - n_numeric) + feature_index``. The number of numerical
    features (``n_numeric``) is passed by the caller; the estimated total
    feature count (``n_total``) guards against a divergence between the
    estimate and the fitted preprocessor output.

    Attributes:
        feature_index (int): Ordinal index of the numerical feature to
            extract (relative to the numerical block).
        n_numeric (int | None): Number of numerical features in the input
            matrix (used to find the numerical block). ``None`` — the index
            is treated as positional into the whole matrix (standalone use).
        n_total (int | None): Estimated total feature count after
            preprocessing, used for the guard. ``None`` — no guard.
        median_ (float): Computed median used to fill missing values.
    """

    def __init__(
        self,
        feature_index: int = 0,
        n_numeric: int | None = None,
        n_total: int | None = None,
    ):
        """Initialize the transformer with the index of the target feature.

        Args:
            feature_index: Ordinal index of the numerical feature to extract
                (relative to the numerical block of the input matrix; with
                ``n_numeric=None`` — positional into the whole matrix).
            n_numeric: Expected number of numerical features in the input
                matrix (the numerical block width). ``None`` — standalone
                positional mode.
            n_total: Estimated total feature count after preprocessing
                (guard against an estimate/preprocessor divergence).
                ``None`` — no guard.
        """
        self.feature_index = feature_index
        self.n_numeric = n_numeric
        self.n_total = n_total

    def _get_dimensions(self, X: Any) -> tuple[int, int]:
        """Return the number of rows and columns of the input data
        in a universal format."""
        if hasattr(X, "shape"):
            return X.shape[0], (X.shape[1] if len(X.shape) > 1 else 1)
        n_rows = len(X)
        n_cols = len(X[0]) if n_rows > 0 and isinstance(X[0], (list, np.ndarray)) else 1
        return n_rows, n_cols

    def _resolve_absolute_index(self, n_cols: int) -> int:
        """Map the (possibly numerical-block-relative) index to an absolute column.

        Args:
            n_cols: Number of columns of the actual input matrix.

        Returns:
            The absolute column index to extract.

        Raises:
            ValueError: If the estimated total feature count disagrees with
                the actual column count, ``n_numeric`` is invalid, or
                ``feature_index`` lies outside the numerical block.
                ValueError is non-fatal for the HPO circuit breaker (unlike
                ``TrainingError`` — a ``RuntimeError`` — which is fatal).
        """
        if self.n_numeric is None:
            # Standalone mode: positional index into the whole matrix.
            if self.feature_index >= n_cols:
                raise ValueError(
                    f"IsotonicDataTransformer: feature_index {self.feature_index} "
                    f"out of bounds. Dataset has only {n_cols} columns."
                )
            return self.feature_index

        # Engine mode: feature_index is relative to the numerical block.
        if self.n_total is not None and int(self.n_total) != int(n_cols):
            raise ValueError(
                f"IsotonicDataTransformer: the estimated feature count "
                f"{self.n_total} disagrees with the fitted preprocessor "
                f"output ({n_cols} columns)."
            )
        if int(self.n_numeric) <= 0:
            raise ValueError(
                "IsotonicDataTransformer: no numerical features to select "
                "from (n_numeric=0). Isotonic regression requires at least one "
                "numerical feature."
            )
        if self.feature_index >= int(self.n_numeric):
            raise ValueError(
                f"IsotonicDataTransformer: feature_index {self.feature_index} "
                f"out of bounds: only {self.n_numeric} numerical features "
                "after preprocessing."
            )
        offset = int(n_cols) - int(self.n_numeric)
        if offset < 0:
            raise ValueError(
                f"IsotonicDataTransformer: the preprocessed matrix has fewer "
                f"columns ({n_cols}) than the estimated numerical features "
                f"({self.n_numeric})."
            )
        return offset + self.feature_index

    def _extract_column(self, X: Any, idx: int) -> pd.Series | np.ndarray:
        """Extract the ``idx``-th column of ``X`` without copying."""
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, idx]
        if isinstance(X, np.ndarray):
            return X[:, idx] if len(X.shape) > 1 else X
        return np.asarray(X)[:, idx]

    def fit(self, X: Any, y: Any = None) -> IsotonicDataTransformer:
        """Compute the median of the selected feature for subsequent missing-value imputation."""
        _, n_cols = self._get_dimensions(X)
        abs_idx = self._resolve_absolute_index(n_cols)

        X_col = self._extract_column(X, abs_idx)
        s_col = X_col if isinstance(X_col, pd.Series) else pd.Series(X_col)
        self.median_ = float(s_col.median()) if not s_col.isna().all() else 0.0
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Extract the feature, fill missing values,
        and convert to a 2D array (n_samples, 1)."""
        try:
            # 1. Get metadata without forcing a copy into a DataFrame
            _n_rows, n_cols = self._get_dimensions(X)

            # 2. Validate the estimated feature count / resolve the absolute
            #    column (numeric-block-relative feature_index).
            abs_idx = self._resolve_absolute_index(n_cols)

            # 3. Extract the column (zero-copy for numpy and pandas)
            X_col = self._extract_column(X, abs_idx)
            # 4. NaN handling logic
            # Check whether all values in the column are NaN
            # Use pd.Series for convenient median computation,
            # if it is not already a Series
            s_col = X_col if isinstance(X_col, pd.Series) else pd.Series(X_col)

            if s_col.isna().all():
                raise ValueError(
                    f"Column with index {abs_idx} contains only NaN values."
                    " Training IsotonicRegression is impossible."
                )
            # Compute the median and fill missing values
            fill_value = getattr(self, "median_", 0.0)
            X_imputed = s_col.fillna(fill_value)
            # Return the result as a 2D array, as scikit-learn expects
            result = X_imputed.to_numpy().reshape(-1, 1)
            return cast(np.ndarray, result)
        except ValueError:
            # ValueError is non-fatal for the circuit breaker in HPO (see the
            # tuner), while TrainingError (a RuntimeError) is fatal — so
            # validation/selection errors must not be wrapped.
            raise
        except Exception as e:  # noqa: BLE001
            # Unify the error message according to the tests
            raise TrainingError(f"Data transformation error: {e}")


class ModelTrainer:
    """Orchestrator of the training and validation process for regression models.

    Automates the construction of a pipeline that includes feature preprocessing
    (imputation, scaling, encoding), synthetic sample enlargement
    (oversampling), and training of the selected algorithm with mandatory validation.
    Attributes:
        algorithm (str): Algorithm name
            (e.g., 'elasticnet', 'xgboosting').
        hyperparams (dict): Hyperparameter configuration for model initialization..
        metric (str): Metric key (r2, mse, mae) used to assess quality on validation.
        random_state (int | None): Random seed for reproducible splits and training.
        serialization_format (SerializationFormat): Storage format (pickle/joblib).
        categorical_features (list[str] | None): List of columns for One-Hot encoding.
        numerical_features (list[str] | None): List of columns to scale.
        id_column (str | None): Identifier excluded from the training process.
        encoding_strategy (str): Categorical encoding strategy
            ('one_hot', 'ordinal', 'target', 'frequency' or 'hashing').
            Defaults to 'one_hot'.
        additional_metrics (Iterable[str] | None): Additional quality metrics
            computed for the trained model using the same validation method as
            the primary metric (``train_test_split`` — hold-out scoring,
            ``k_fold``/``loo`` — mean across folds for additive metrics, the
            pooled OOF value for non-additive ones (RMSE/MSE/R²/nRMSE/RMSLE/
            D²/explained_variance/max error/median absolute error and their
            aliases)). Any iterable sequence of strings is accepted
            (list, tuple, set, generator); the metric name is normalized to
            lowercase. These metrics are informational only and do not affect
            training. Values are stored in ``additional_scores`` after calling
            fit().
        preprocessing_override (PreprocessingOverride | dict | None): Explicit
            override of the feature preprocessing preset. It may be specified
            partially or fully and takes precedence over automatic selection.
        os_enable (bool): Flag enabling dataset balancing/enlargement.
        os_multiplier (float): Coefficient for generating synthetic data.
        os_algorithm (str): Oversampling algorithm ('random', 'smote', 'adasyn').
        pipeline (Pipeline | None): Final pipeline object after calling fit().
        val_score (float | None): Value of the primary metric computed on the
            hold-out part of the data (``train_test_split``) or — for
            ``k_fold``/``loo`` — as the mean across folds for additive
            metrics and as the pooled value over the OOF vector for
            non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D²,
            explained_variance, max error, median absolute error and their
            aliases) —
            using the same validation method that HPO uses to compare models
            (``validation.iter_splits``). It is not a train score: the
            estimate is always computed on data not used to train the pipeline
            of that fold. For error metrics the value is positive
            (sign-corrected).
        oof_score_ (float | None): ``RMSE_oof`` — an out-of-fold estimate on
            all rows: predictions from validation folds are concatenated into
            a single vector and the metric is computed on the whole vector at
            once (rather than averaged over small folds). Every row is
            predicted by a fold-model that has not been trained on it, but the
            splits are the same as in HPO (``validation.iter_splits`` with the
            same seed): when the hyperparameters were selected on these
            splits, the estimate inherits the selection bias and is NOT an
            independent estimate of generalization (for RMSE it equals the
            best-trial ``val_score``). ``None`` if an OOF estimate is
            impossible (no valid pairs at all). Separate channel: the
            ``val_score`` of HPO phases is not affected.
        oof_predictions_ (pd.Series | np.ndarray | None): The ``y_pred_oof``
            vector, aligned to the original row order (``pd.Series`` with the
            preserved pandas index for data with a unique index, including
            MultiIndex; ``np.ndarray`` by position for numpy inputs and data
            with duplicated indexes). Uncovered rows (e.g., the train part of
            ``train_test_split``) and non-finite predictions are NaN.
            ``None`` until the first ``fit()``.
        oof_coverage_ (float | None): Fraction of the original data rows covered
            by OOF predictions (1.0 for ``k_fold``/``loo``; < 1.0 for
            ``train_test_split`` — the limitation is documented in the log).
        additional_scores (dict[str, float]): Values of the additional metrics
            computed using the same validation method as the primary metric
            (fold mean for additive metrics, the pooled OOF value for
            non-additive ones; empty dict if no metrics are configured).
        validation_strategy (str): Validation strategy for the final model:
            ``'auto'`` (default, as ``general.validation_strategy``),
            ``'train_test_split'``, ``'k_fold'`` or ``'loo'``. ``'auto'`` is resolved inside fit() via
            ``validation.make_cv``; the engine path passes an already resolved
            method. The list of allowed methods is in
            ``validation.SUPPORTED_VALIDATION_METHODS`` (a single extension
            point: new strategies, e.g. order-aware ``TimeSeriesSplit``, are
            registered there and in ``make_cv``, after which the OOF channel
            works with them without rewriting).
        n_folds (int): Number of folds for the ``'k_fold'`` strategy (default 5).
        test_size (float | int): Size of the hold-out part: a fraction in (0, 1)
            or an integer number of rows (as in sklearn ``train_test_split``).
            Defaults to 0.2.
        feature_names (list[str] | None): List of feature names identified
            during training.
        dropped_features_ (list[str]): Names of columns removed by cleaning
            uninformative features (empty and strictly constant ones) in the
            last ``fit()`` call. Empty list if cleaning was not applied (input
            without column names) or nothing was removed. Populated only for
            ``pd.DataFrame`` inputs: ``np.ndarray`` inputs have no column names,
            so cleaning is not performed.
        preprocessing_preset (PreprocessingPreset | None): The resolved
            preprocessing preset (populated during fit()).
        feature_selection_cfg (FeatureSelectionCfg): Validated feature
            selection configuration (method, percentile, min_features, etc.).
            Defaults to ``FeatureSelectionCfg()`` (mode='disabled').
        feature_selection_active (bool | None): Explicit flag of feature
            selection activity passed from outside (result of an HPO trial).
            Takes precedence over the mode from ``feature_selection_cfg``;
            ``None`` — the decision is made based on the configuration.
        feature_selection_active_ (bool): Actual feature selection status in
            the final training (populated in fit(); False before the first fit).
        selected_features_mask_ (np.ndarray | None): Boolean mask of the
            selected columns after fit() (None if selection was not applied).
            Note: the mask has the dimensionality of the post-preprocessor
            matrix (after one-hot expansion/scaling), not of the original
            columns; no mapping back to original names is performed — the
            filtering is transparent through the pipeline.
        lock (threading.RLock): Recursive lock for thread-safe training.
    """

    # Set by ``_prepare_data`` during fit() (the dtypes of the training frame).
    _detected_dtypes: Any

    def __init__(
        self,
        algorithm: str = "elasticnet",
        hyperparams: dict[str, Any] | None = None,
        metric: str = "r2",
        random_state: int | None = 42,
        data_oversampling: bool = False,
        data_oversampling_multiplier: float = 1.0,
        data_oversampling_algorithm: str = "random",
        data_oversampling_target_bins: int = 8,
        data_oversampling_add_noise: bool = False,
        data_oversampling_noise_level: float = 0.01,
        serialization_format: SerializationFormat = SerializationFormat.pickle,
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
        id_column: str | None = None,
        encoding_strategy: str = "one_hot",
        additional_metrics: Iterable[str] | None = None,
        preprocessing_override: PreprocessingOverride | dict[str, Any] | None = None,
        high_cardinality_threshold: int | None = 50,
        high_cardinality_encoding: str | None = "frequency",
        hashing_n_components: int = 16,
        target_encoding_smoothing: float = 20.0,
        target_encoding_fallback: float | None = None,
        feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
        feature_selection_active: bool | None = None,
        validation_strategy: ValidationStrategy | str = "auto",
        n_folds: int = 5,
        test_size: float = 0.2,
        inf_policy: str = "as_missing",
        datetime_policy: str = "expand",
        drop_id_like_columns: bool = False,
        warn_id_like: bool = True,
    ):
        """Initialize the trainer with model parameters and preprocessing settings.

        Args:
            feature_selection_cfg: Object or dict of feature selection settings
                (method, percentile, min_features, etc.). ``None`` —
                ``FeatureSelectionCfg()`` (mode='disabled', selection off).
            feature_selection_active: Explicit boolean flag of selection activity
                (takes precedence over the mode from ``feature_selection_cfg``).
                It is passed by an external HPO loop that has already decided
                whether selection is needed. ``None`` — the decision is made
                based on the configuration.
            validation_strategy: Validation strategy for the final model
                ('train_test_split', 'k_fold', 'loo' or 'auto'). Strings are
                normalized via ``validation.norm_val_method`` (a single source
                shared with HPO). Defaults to 'auto' — the same default as
                ``general.validation_strategy`` and ``tuner.optimize``.
            n_folds: Number of folds for the 'k_fold' strategy (>= 2).
            test_size: Size of the hold-out part: a fraction in (0, 1) or an
                integer number of rows (as in sklearn ``train_test_split``).
                Defaults to 0.2.
            inf_policy: Policy for ±inf values in numeric features:
                ``"as_missing"`` (default) — replace with NaN and warn;
                ``"error"`` — reject with ValueError.
            datetime_policy: Policy for datetime/timedelta columns:
                ``"expand"`` (default) — replace with derived components and
                warn; ``"drop"`` — drop with a warning; ``"error"`` — reject.
            drop_id_like_columns: Whether ID-like columns are dropped (default
                ``False`` — they are only reported with a WARNING).
            warn_id_like: Whether the ID-like WARNING is emitted (default
                ``True``). Set to ``False`` when the diagnostics have already
                been reported by the pipeline entry point (``prepare_dataset``)
                to avoid duplicate warnings on the engine path.
        """

        self.logger = logging.getLogger(__name__)

        self.lock = threading.RLock()

        # Validate the algorithm
        self.algorithm = _validate_algorithm(algorithm)

        # hyperparams
        self.hyperparams = _validate_hyperparams(hyperparams)

        # Remaining parameters
        self.metric = metric.lower()
        self.random_state = random_state
        self.serialization_format = serialization_format
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.id_column = id_column
        # ---------- input-data diagnostics policies ----------
        # Validated immediately: invalid policy values are rejected at
        # initialization time, before training starts.
        _validate_input_policies(
            inf_policy=inf_policy,
            datetime_policy=datetime_policy,
            drop_id_like_columns=drop_id_like_columns,
            warn_id_like=warn_id_like,
        )
        self.inf_policy: str = inf_policy
        self.datetime_policy: str = datetime_policy
        self.drop_id_like_columns: bool = drop_id_like_columns
        self.warn_id_like: bool = warn_id_like
        # Categorical encoding strategy: 'one_hot' (default), 'ordinal',
        # 'target', 'frequency' or 'hashing'. Stored as a string primitive
        # to remain pickle-compatible.
        self.encoding_strategy: EncodingStrategy = _validate_encoding_strategy(
            encoding_strategy
        )

        # High-cardinality encoding parameters and new strategies.
        # Validated immediately so that invalid values are rejected at
        # initialization time, before training starts.
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.hashing_n_components = hashing_n_components
        self.target_encoding_smoothing = target_encoding_smoothing
        self.target_encoding_fallback = target_encoding_fallback
        _validate_high_cardinality(
            high_cardinality_threshold=high_cardinality_threshold,
            high_cardinality_encoding=high_cardinality_encoding,
            hashing_n_components=hashing_n_components,
            target_encoding_smoothing=target_encoding_smoothing,
        )

        # ---------- additional (informational) metrics ----------
        self.additional_metrics: list[str] = _normalize_additional_metrics(
            additional_metrics
        )
        self.additional_scores: dict[str, float] = {}

        # Explicit override of the preprocessing preset: validate it immediately
        # so that invalid values are rejected at initialization time.
        self.preprocessing_override: PreprocessingOverride | None = (
            _validate_preprocessing_override(preprocessing_override)
        )
        # The resolved preset is populated in _build_preprocessor during fit().
        self.preprocessing_preset: PreprocessingPreset | None = None

        # ---------- feature selection ----------
        # Validate the feature selection configuration immediately so that
        # invalid values (percentile=150, unknown method/mode, extra keys)
        # are rejected at initialization time, before expensive training.
        self.feature_selection_cfg: FeatureSelectionCfg = (
            _validate_feature_selection_cfg(feature_selection_cfg)
        )
        # Explicit flag from HPO/component.py (takes precedence over the config);
        # None — the decision is made in fit() based on the configuration mode.
        self.feature_selection_active: bool | None = _validate_feature_selection_active(
            feature_selection_active
        )
        # Actual selection status in the final training (populated in fit()).
        self.feature_selection_active_: bool = False
        # Boolean mask of the selected columns in post-preprocessor dimensionality.
        self.selected_features_mask_: np.ndarray | None = None

        # ---------- validation ----------
        # Quality assessment strategy for the final model. Stored as a string
        # primitive (pickle compatibility); strings are normalized via
        # validation.norm_val_method — a single mapping source shared with HPO
        # (validation.make_cv / iter_splits).
        self.validation_strategy: str = _validate_validation_strategy(
            validation_strategy, n_folds
        )
        self.n_folds: int = n_folds
        self.test_size: float = _validate_test_size(test_size)

        # ---------- oversampling ----------
        self.os_enable = data_oversampling
        self.os_multiplier = data_oversampling_multiplier
        self.os_algorithm = data_oversampling_algorithm
        self.os_target_bins = data_oversampling_target_bins
        self.os_add_noise = data_oversampling_add_noise
        self.os_noise_level = data_oversampling_noise_level
        _validate_oversampling(
            multiplier=self.os_multiplier,
            algorithm=self.os_algorithm,
            target_bins=self.os_target_bins,
            noise_level=self.os_noise_level,
        )

        # Fields populated after fit(...)
        self.pipeline: Pipeline | None = None
        self.base_model: Any = None
        self.val_score: float | None = None
        # OOF candidate evaluation channel: a metric on all rows without
        # training leakage (for hyperparameters selected on the same splits it
        # still carries the selection bias). Each row is predicted by a model that has not seen it
        # during training; predictions from validation folds are collected into
        # a single y_pred_oof vector aligned to the original row order, and
        # RMSE_oof is computed on the whole vector at once.
        self.oof_score_: float | None = None
        self.oof_predictions_: pd.Series | np.ndarray | None = None
        self.oof_coverage_: float | None = None
        self.feature_names: list[str] | None = None
        # Names of columns removed by cleaning uninformative features.
        # Populated in fit() for pd.DataFrame inputs; empty list by default
        # (pickle-compatible attribute).
        self.dropped_features_: list[str] = []
        # Datetime expansion applied to the training data: source column →
        # generated component columns (``datetime_policy="expand"``). predict()
        # applies the same expansion to raw input data, so a model trained on
        # a frame with datetime columns can predict on the same raw schema.
        # Populated in fit() and by register_datetime_expansion() (the engine
        # expands the data in prepare_dataset, before the trainer sees it).
        self.datetime_expansion_: dict[str, list[str]] = {}
        self._last_train_y: pd.Series | None = None
        self._last_val_y: pd.Series | None = None
        # Artifact metadata, populated by :meth:`save` (and read back by
        # :meth:`load` when the saved object carries it). Contains the engine
        # version, dependency versions, the creation timestamp, the algorithm,
        # hyperparameters and a hash of the artifact configuration.
        self.artifact_metadata_: dict[str, Any] | None = None

        _validate_feature_lists(categorical_features, numerical_features)

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Verify that all specified feature names are present in the given DataFrame."""

        specified_features = (self.categorical_features or []) + (
            self.numerical_features or []
        )
        missing = [col for col in specified_features if col not in X.columns]
        if missing:
            raise TrainingError(f"Specified columns not found in data: {missing}")

    def _detect_feature_types(self, X: pd.DataFrame, target_column: str) -> None:
        """Automatically classify columns into numerical and categorical ones.

        See :func:`trainer_data._detect_feature_types`.
        """
        _detect_feature_types(self, X, target_column)

    def _resolve_preprocessing_preset(self) -> PreprocessingPreset:
        """Resolve the preprocessing preset for the current algorithm.

        Automatic selection based on the lookup table
        (:func:`resolve_preprocessing_preset`) plus application of an explicit
        user override.

        Returns:
            The final preprocessing preset.
        """
        override = getattr(self, "preprocessing_override", None)
        return resolve_preprocessing_preset(self.algorithm, override)

    def _extract_metadata(self, X: Any) -> list[str] | None:
        """Extract column names from various data structures
        without copying the contents."""
        if hasattr(X, "get_data_info"):
            return cast(list[str], X.get_data_info()["columns"])
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        if hasattr(X, "shape") and len(X.shape) > 1:
            return [str(f"col_{i}") for i in range(X.shape[1])]
        return None

    def __getstate__(self) -> dict[str, Any]:
        """Control the object state for correct serialization
        (excluding lock and logger)."""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        if "lock" in state:
            del state["lock"]
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the object state after deserialization,
        initializing a thread-safe lock and a logger."""
        self.__dict__.update(state)
        # Re-initialize the lock after unpickling
        self.lock = threading.RLock()
        # Re-initialize logger if necessary
        self.logger = logging.getLogger(__name__)

    def _build_preprocessor(self, feature_names: list[str]) -> ColumnTransformer:
        """Build a ColumnTransformer for separate handling of data types.

        See :func:`trainer_data._build_preprocessor`.
        """
        return _build_preprocessor(self, feature_names)

    def _prepare_data(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Validate the input data types and separate features from the target.

        See :func:`trainer_data._prepare_data`.
        """
        return _prepare_data(self, X, y)

    def _resolve_feature_selection_active(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
    ) -> bool:
        """Resolve the actual feature selection activity for fit().

        See :func:`trainer_feature_selection._resolve_feature_selection_active`.
        """
        return _resolve_feature_selection_active(
            self, X_train, y_train, preprocessor, base_model
        )

    def _run_feature_selection_auto_check(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
    ) -> bool:
        """Run the Standalone Auto-Check of feature selection usefulness.

        See :func:`trainer_feature_selection._run_feature_selection_auto_check`.
        """
        return _run_feature_selection_auto_check(
            self, X_train, y_train, preprocessor, base_model
        )

    def _assemble_steps(
        self,
        preprocessor: ColumnTransformer,
        base_model: Any,
        *,
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
            preprocessor: Feature preprocessing ``ColumnTransformer``.
            base_model: Final regressor.
            feature_selection_active: Flag enabling the feature selection step.
            X_for_estimate: The feature matrix used to estimate the total
                post-preprocessing feature count for the isotonic guard
                (``None`` — the count is disabled).

        Returns:
            A list of ``(name, transformer)`` tuples for building the Pipeline.
        """
        steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
        if resolve_algorithm_name(self.algorithm) == "isotonic_regression":
            steps.append(
                (
                    "isotonic_selector",
                    IsotonicDataTransformer(
                        feature_index=_resolve_isotonic_feature_index(self),
                        n_numeric=_isotonic_n_numeric(self),
                        n_total=(
                            _isotonic_n_total_estimate(self, X_for_estimate)
                            if X_for_estimate is not None
                            else None
                        ),
                    ),
                )
            )
        if feature_selection_active:
            # Backward compatibility with old pickles: the configuration is read
            # via getattr (similar to encoding_strategy in _build_preprocessor);
            # a missing attribute yields the FeatureSelectionCfg() default.
            cfg = getattr(self, "feature_selection_cfg", None)
            if not isinstance(cfg, FeatureSelectionCfg):
                cfg = FeatureSelectionCfg()
            selector = FeatureSelector(
                method=cfg.method.value,
                percentile=cfg.percentile,
                min_features=cfg.min_features,
                variance_threshold=cfg.variance_threshold,
                n_estimators=cfg.n_estimators,
                # Fixed fallback seed for selection (consistent with the tuner
                # and _check_auto_feature_selection): with random_state=None in
                # the main training, the selector must still use the fixed seed
                # 42, otherwise selection is not reproducible between fit() calls
                # and diverges from the HPO phase.
                random_state=self.random_state if self.random_state is not None else 42,
            )
            steps.append(("feature_selector", selector))
        if self.os_enable:
            oversampler = DataOversampler(
                algorithm=self.os_algorithm,
                multiplier=self.os_multiplier,
                target_bins=self.os_target_bins,
                add_noise=self.os_add_noise,
                noise_level=float(self.os_noise_level),
            )
            steps.append(("oversampler", oversampler))
        steps.append(("model", base_model))
        return steps

    def _fit_internal(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
        *,
        feature_selection_active: bool = False,
    ) -> Pipeline:
        """Assemble the final pipeline and run
        the training process on the prepared data.

        Args:
            X_train: Training feature matrix.
            y_train: Target variable.
            preprocessor: Feature preprocessing ``ColumnTransformer``.
            base_model: Final regressor.
            feature_selection_active: Flag enabling the feature selection step
                (keyword-only; defaults to ``False`` — no-selection behavior).
        """
        # 0) Reset the actual selection status and the selected-column mask.
        # If pipeline.fit fails, the trainer must not retain values from the
        # previous training. After a successful fit the values are overwritten
        # in step 4.
        self.feature_selection_active_ = False
        self.selected_features_mask_ = None
        # 1) Assemble the pipeline steps
        steps = self._assemble_steps(
            preprocessor,
            base_model,
            feature_selection_active=feature_selection_active,
            X_for_estimate=X_train,
        )
        # 2) Initialize the pipeline
        # Use imblearn's Pipeline so that resampling steps work correctly
        self.pipeline = Pipeline(steps=steps)
        # 3) Run the training
        try:
            self.pipeline.fit(X_train, y_train)
        except (ValueError, TypeError):
            # Re-raise validation errors directly so that tests can catch them
            # This is critical for tests checking invalid hyperparameters
            raise
        except TrainingError:
            raise
        except Exception as e:
            # Wrap all other system errors in TrainingError
            self.logger.debug("Unexpected pipeline fit failure trace:", exc_info=True)
            self.logger.error(f"Unexpected error in pipeline fit: {e}")
            raise TrainingError(f"Internal training failure: {e}")
        # 4) Record the actual selection status and the selected-column mask.
        # The mask is taken from the selector step inside the pipeline and has
        # the dimensionality of the post-preprocessor matrix (after one-hot
        # expansion/scaling).
        self.feature_selection_active_ = feature_selection_active
        selector_step = self.pipeline.named_steps.get("feature_selector")
        if selector_step is not None and hasattr(selector_step, "support_"):
            self.selected_features_mask_ = np.asarray(
                selector_step.support_, dtype=bool
            )
        else:
            self.selected_features_mask_ = None
        return self.pipeline

    def fit(self, X: Any, y: Any) -> ModelTrainer:
        """Run the full cycle of data preparation, model training, and metric evaluation.

        Quality evaluation (``val_score`` / ``additional_scores``) is performed
        on held-out data using the same validation method that HPO uses to
        compare models (``validation.iter_splits``): ``train_test_split`` — a
        single hold-out split with scoring on the held-out part;
        ``k_fold``/``loo`` — mean across folds for additive metrics, the
        pooled value over the OOF vector for non-additive metrics (RMSE, MSE,
        R², nRMSE, RMSLE, D², explained_variance, max error, median absolute
        error and their aliases). In addition, an OOF channel is
        collected: predictions from validation folds are concatenated into
        ``y_pred_oof``, and ``RMSE_oof`` (``oof_score_``) is computed on the
        whole vector at once — an estimate on all rows without training
        leakage, separate from ``val_score``. Both use the HPO splits, so for
        hyperparameters selected on those splits neither is independent of
        the selection. After validation scoring, the final
        pipeline is trained on the full data: the artifact stores a training on
        100% of the sample, while ``val_score`` remains an honest quality
        estimate (not a train score).
        """

        with self.lock:
            _reset_fit_state(self)

            # Stage 1: Validation and data preparation
            X_prepared, y_s = self._prepare_data(X, y)
            _check_training_data(X_prepared, y_s)
            X_prepared = _prepare_training_features(self, X_prepared)

            # Stage 3: Create the model via the factory.
            # Done BEFORE preprocessor assembly so that an unknown algorithm
            # (a typo in the name) is rejected here with a clear message rather
            # than being masked by the default preprocessing preset.
            # The model seed follows the trainer's ``random_state`` (the engine
            # seed from ``general.random_state``), unless the user set
            # ``random_state`` explicitly in the hyperparameters — an explicit
            # value always wins. ``random_state=None`` (trainer or explicit
            # hyperparameter) produces a truly random fit.
            try:
                # Auxiliary keys of the HPO result ``params`` are not model
                # hyperparameters: ``use_feature_selection`` (the 'auto' mode
                # decision) and the informational ``isotonic_feature`` are
                # popped like in the engine and the tuner, so that
                # ``ModelTrainer(hyperparams=res["params"])`` does not fail on
                # the strict ``create_model`` name check.
                model_kwargs = self.hyperparams.copy()
                model_kwargs.pop("feature_index", None)
                model_kwargs.pop("use_feature_selection", None)
                model_kwargs.pop("isotonic_feature", None)
                if "random_state" not in model_kwargs:
                    model_kwargs["random_state"] = self.random_state
                base_model = create_model(self.algorithm, **model_kwargs)
            except (ValueError, ImportError) as e:
                raise TrainingError(f"Error creating model: {e}")

            # Stage 4: Build the preprocessor via delegation
            if self.feature_names is None:
                # If there are no names, try to extract them again or create an empty list
                self.feature_names = self._extract_metadata(X_prepared) or []
            preprocessor = self._build_preprocessor(self.feature_names)

            # Stage 4.5: Resolve feature selection activity.
            # Priority: isotonic bypass -> explicit flag -> configuration mode
            # (always/disabled) -> standalone auto-check for mode='auto'.
            feature_selection_active = self._resolve_feature_selection_active(
                X_prepared, y_s, preprocessor, base_model
            )

            # Stage 5: Validation scoring. Quality evaluation is performed with
            # the same split method used for model comparison in HPO
            # (validation.iter_splits), so that val_score/additional_scores are
            # honest hold-out/CV estimates rather than train scores.
            self._score_on_validation_splits(
                X_prepared,
                y_s,
                preprocessor,
                base_model,
                feature_selection_active=feature_selection_active,
            )

            # Stage 6: Final training on the full data (the artifact as before):
            # the pipeline is trained on 100% of the sample, while val_score
            # remains the estimate computed on held-out data (stage 5).
            self.pipeline = self._fit_internal(
                X_prepared,
                y_s,
                preprocessor,
                base_model,
                feature_selection_active=feature_selection_active,
            )

            # Observability: the dimensionality and the representation of the
            # preprocessed matrix that the model actually received (one-hot
            # expansion, missing-value indicators, sparse/dense storage).
            _log_preprocessed_matrix(self, preprocessor, X_prepared)

            return self

    def _score_on_validation_splits(
        self,
        X: Any,
        y: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
        *,
        feature_selection_active: bool,
    ) -> None:
        """Evaluate model quality on held-out data.

        See :func:`trainer_scoring._score_on_validation_splits`.
        """
        _score_on_validation_splits(
            self,
            X,
            y,
            preprocessor,
            base_model,
            feature_selection_active=feature_selection_active,
        )

    def _finalize_oof(
        self,
        oof_pred: np.ndarray,
        oof_y: np.ndarray,
        X: Any,
        n_samples: int,
        *,
        method: str,
    ) -> None:
        """Finalize the OOF evaluation channel.

        See :func:`trainer_scoring._finalize_oof`.
        """
        _finalize_oof(self, oof_pred, oof_y, X, n_samples, method=method)

    def _align_oof_vector(
        self,
        oof_pred: np.ndarray,
        X: Any,
    ) -> pd.Series | np.ndarray:
        """Align the OOF vector ``y_pred_oof`` to the original row order.

        See :func:`trainer_scoring._align_oof_vector`.
        """
        return _align_oof_vector(oof_pred, X)

    def register_datetime_expansion(self, expansion: dict[str, list[str]]) -> None:
        """Record a datetime expansion applied to the data before ``fit``.

        Used when the expansion happened upstream of the trainer (the engine
        expands datetime columns in ``prepare_dataset``): ``predict`` then
        applies the same expansion to raw input data. Entries recorded by
        ``fit`` itself take precedence for the same source column.

        Args:
            expansion: Mapping ``source column → generated component
                columns`` (see ``datetime_expansion_map``).

        Raises:
            TrainingError: If ``expansion`` is not a mapping of column names
                to lists of column names.
        """
        if not isinstance(expansion, dict) or not all(
            isinstance(src, str)
            and isinstance(names, list)
            and all(isinstance(name, str) for name in names)
            for src, names in expansion.items()
        ):
            raise TrainingError(
                "expansion must be a dict[str, list[str]] "
                "(source column → generated component columns)"
            )
        with self.lock:
            current = getattr(self, "datetime_expansion_", None) or {}
            self.datetime_expansion_ = {
                **{src: list(names) for src, names in expansion.items()},
                **current,
            }

    def predict(self, X: Any) -> np.ndarray:
        """Get model predictions for new data using the trained pipeline.

        For ``pd.DataFrame`` inputs, a defensive validation and alignment of
        the column order to the training order (``self.feature_names``) is
        performed: missing columns raise an explicit error, extra columns are
        dropped, and duplicated names are rejected. Datetime/timedelta columns
        that were expanded into components at training time
        (``datetime_expansion_``) are expanded the same way first, so the
        model accepts the raw schema it was trained on. ``SharedDataFrame`` is
        restored into a DataFrame and aligned the same way when column names
        are available. For ``np.ndarray`` and other inputs without names, the
        positional column order fixed at training time is used.

        The ±inf policy used at training time (``inf_policy``) is applied to
        the input as well, so inference is symmetric with :meth:`fit`:
        ``"as_missing"`` replaces ±inf in numeric features with NaN (the
        preprocessor imputes them) and logs a WARNING listing the columns and
        counts; ``"error"`` rejects the input. Both ``pd.DataFrame`` and
        numeric ``np.ndarray`` inputs are handled.

        Args:
            X: Feature matrix to predict on.

        Returns:
            The model predictions as a ``np.ndarray``.

        Raises:
            TrainingError: If the model is not trained, if ``inf_policy`` is
                ``"error"`` and ``X`` contains ±inf (the message names the
                columns), or if the prediction itself fails.
        """

        with self.lock:
            # 1) Check whether the model is trained
            if self.pipeline is None:
                raise TrainingError(
                    "The predict method called for an untrained model."
                    " Perform fit() first."
                )
            try:
                # Align columns for inputs with names; for the rest, avoid
                # conversion if X is already an array or DataFrame
                X_input: Any
                if isinstance(X, pd.DataFrame):
                    X_input = _align_predict_columns(
                        self,
                        expand_datetime_columns(
                            X, getattr(self, "datetime_expansion_", None) or {}
                        ),
                    )
                elif isinstance(X, SharedDataFrame):
                    X_input = _align_predict_shared_dataframe(self, X)
                elif isinstance(X, (pd.Series, np.ndarray)):
                    X_input = X
                else:
                    X_input = np.asarray(X)

                # The same ±inf policy as in fit(): without it a frame that
                # trained fine (inf → NaN) would fail at prediction time with
                # an sklearn "Input X contains infinity" error.
                if isinstance(X_input, (pd.DataFrame, np.ndarray)):
                    try:
                        X_input = apply_inf_policy(
                            X_input,
                            str(getattr(self, "inf_policy", "as_missing")),
                        )
                    except ValueError as err:
                        raise TrainingError(str(err)) from err

                preds = self.pipeline.predict(X_input)
                return np.asarray(preds)

            except TrainingError:
                raise
            except Exception as e:  # noqa: BLE001
                raise TrainingError(f"Error during prediction: {e}")

    def save(self, path: str | Path) -> None:
        """Serialize and save the current trainer instance to disk.

        The artifact is written in ``self.serialization_format`` (pickle or
        joblib). Before writing, the instance is enriched with
        ``artifact_metadata_`` — a dict with the engine version, the versions
        of Python and the ML stack, the creation timestamp (UTC), the
        algorithm, hyperparameters and a hash of the configuration — so that
        :meth:`load` can warn about potentially incompatible artifacts.

        Args:
            path: Destination path of the artifact. The directory is created
                if it does not exist.
        """
        with self.lock:
            if self.pipeline is None and self.base_model is None:
                raise TrainingError("Nothing to save: model is not trained")

            path_obj = Path(path)
            # Create the directories if they do not exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Enrich the artifact with environment and configuration metadata
            # so that load() can detect version mismatches and the engine can
            # audit the provenance of the artifact.
            self.artifact_metadata_ = _collect_artifact_metadata(self)

            # Call the new utility instead of pickle.dump
            save_artifact(obj=self, path=path_obj, fmt=self.serialization_format)

    @classmethod
    def load(
        cls, path: str | Path, fmt: SerializationFormat | None = None
    ) -> ModelTrainer:
        """Load a previously saved ModelTrainer object from a file.

        The serialization format is auto-detected when ``fmt`` is ``None``:
        the file extension decides (``.joblib`` → joblib, ``.pkl``/``.pickle``
        → pickle), and the other format is tried as a fallback so that a
        wrongly named file still loads. For an unknown extension, joblib is
        tried first, then pickle. When an explicit ``fmt`` is passed, only
        that format is tried (as before).

        After a successful load, a WARNING is logged when the artifact's
        engine or scikit-learn major/minor version differs from the one in the
        current environment.

        Args:
            path: Path to the saved artifact.
            fmt: Explicit serialization format. ``None`` (default) — the
                format is auto-detected from the extension with a joblib-then-
                pickle fallback.

        Returns:
            The deserialized :class:`ModelTrainer` instance.

        Raises:
            TrainingError: If the file does not exist, cannot be deserialized
                with any candidate format, or does not contain a
                ``ModelTrainer`` instance.
        """
        path_obj = Path(path)

        formats = resolve_load_formats(path_obj, fmt)

        obj: Any = None
        last_error: Exception | None = None
        for candidate in formats:
            try:
                obj = load_artifact(path=path_obj, fmt=candidate)
                break
            except FileNotFoundError:
                raise TrainingError(f"File not found: {path}")
            except Exception as e:  # noqa: BLE001
                last_error = e
        if obj is None:
            raise TrainingError(f"Error loading artifact: {last_error}") from last_error

        # Verify the type of the loaded object
        if not isinstance(obj, cls):
            raise TrainingError(f"Loaded object is not a ModelTrainer: {path}")

        # Warn about a major/minor version mismatch of the engine or
        # scikit-learn against the current environment (best effort: only
        # artifacts saved by a recent version carry artifact_metadata_).
        metadata = getattr(obj, "artifact_metadata_", None)
        if metadata is not None:
            _warn_on_version_mismatch(metadata, obj.logger)

        return obj


def train_model(
    cfg_or_algo: dict[str, Any] | str,
    metric_or_testsize: str | float,
    params_or_metric: dict[str, Any] | str,
    X: Any = None,
    y: Any = None,
    enable_logging: bool = False,
    random_state: int | None = 42,
    log_path: str | Path | None = None,
    *,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
    feature_selection_active: bool | None = None,
) -> float:
    """Provide backward compatibility with the legacy model training API.
    A facade function that accepts a configuration or a set of positional
    arguments, instantiates a ModelTrainer, and returns the metric value
    computed on held-out data (hold-out or the CV estimate — the fold mean
    for additive metrics, the pooled OOF value for non-additive metrics such
    as RMSE/MSE/R²/nRMSE) — using the
    same validation method used to assess quality in ``ModelTrainer.fit()``
    (not a train score).

    The feature selection parameters (``feature_selection_cfg`` /
    ``feature_selection_active``) are keyword-only: in the "config dict"
    branch, the same-named dict keys take precedence over the explicit
    arguments (similarly to ``data_oversampling``). A key present in the config
    with the value ``None`` is treated as "value not set": in that case, as
    when the key is absent, the explicit function argument is used (fallback).
    In the simple API branch, the function arguments are used directly.

    Attributes:
        cfg_or_algo (dict | str): Configuration dict or algorithm name.
        metric_or_testsize (str | float): Metric or test sample size.
        params_or_metric (dict | str): Hyperparameters or metric name.
        X (Any): Feature matrix.
        y (Any): Target variable vector.
        enable_logging (bool): Flag enabling logging.
        random_state (int | None): Random seed.
        log_path (str | Path | None): Path to the log file. Used only when
            ``enable_logging`` is True: file logging is initialized with
            ``setup_logging(logfile=...)`` so the "Training finished" message
            and the library logs are written to this file. ``None`` — no file
            is created here and messages go to the already configured handlers
            (e.g. the console). The file handler is attached once per process:
            if one is already registered, the first path wins and later
            ``log_path`` values do not re-route the output. In the "config
            dict" branch, the ``log_path`` key takes precedence over this
            argument, except when the key is present with the value ``None`` —
            then this argument is used (fallback).
        feature_selection_cfg (FeatureSelectionCfg | dict | None):
            Feature selection configuration (method, percentile, min_features,
            etc.). In the "config dict" branch, the ``feature_selection_cfg``
            key takes precedence over this argument, except when the key is
            present with the value ``None`` — then this argument is used
            (fallback). ``None`` — ``FeatureSelectionCfg()`` (mode='disabled').
        feature_selection_active (bool | None): Explicit boolean flag of
            feature selection activity (takes precedence over the mode from
            ``feature_selection_cfg``). In the "config dict" branch, the
            ``feature_selection_active`` key takes precedence over this
            argument, except when the key is present with the value ``None`` —
            then this argument is used (fallback). Non-boolean values are
            rejected by the ``ModelTrainer`` constructor (strict typing).
            ``None`` — the decision is based on the configuration.
    """
    algo: str = ""
    metric: str = ""
    hyperparams: dict[str, Any] = {}
    rs: int | None = random_state

    fs_cfg: FeatureSelectionCfg | dict[str, Any] | None
    fs_active: bool | None

    # The "config dict" case
    if isinstance(cfg_or_algo, dict):
        cfg: dict[str, Any] = cfg_or_algo
        algo = str(cfg.get("algorithm", ""))
        metric = str(cfg.get("metric", ""))
        hyperparams = cast(dict[str, Any], cfg.get("hyperparams", {}))
        rs = cast(int | None, cfg.get("random_state", 42))
        enable_logging = bool(cfg.get("enable_logging", False))
        # ``log_path`` follows the same precedence rule as
        # ``feature_selection_cfg``: the config key takes precedence over the
        # explicit argument, except when the key is present with the value
        # None ("value not set") — then the explicit argument is used
        # (fallback).
        if "log_path" in cfg and cfg["log_path"] is not None:
            log_path = cast(str | Path | None, cfg["log_path"])
        data_os = bool(cfg.get("data_oversampling", False))
        data_os_mult = float(cfg.get("data_oversampling_multiplier", 1.0))
        data_os_alg = str(cfg.get("data_oversampling_algorithm", "random"))
        # Config keys take precedence over the explicit arguments (similarly to
        # ``data_oversampling``); a key with the value None is treated as
        # "value not set" and falls back to the function argument (see docstring).
        if "feature_selection_cfg" in cfg and cfg["feature_selection_cfg"] is not None:
            fs_cfg = cast(
                FeatureSelectionCfg | dict[str, Any] | None,
                cfg["feature_selection_cfg"],
            )
        else:
            fs_cfg = feature_selection_cfg
        if (
            "feature_selection_active" in cfg
            and cfg["feature_selection_active"] is not None
        ):
            fs_active = cast(bool | None, cfg["feature_selection_active"])
        else:
            fs_active = feature_selection_active
    else:
        # Simple API
        # If None is passed, we do NOT turn it into the string "None" right
        # away to preserve the original validation logic for the tests.
        algo = cfg_or_algo if cfg_or_algo is not None else ""
        metric = str(metric_or_testsize)
        hyperparams = cast(dict[str, Any], params_or_metric)
        rs = random_state
        data_os = False
        data_os_mult = 1.0
        data_os_alg = "random"
        fs_cfg = feature_selection_cfg
        fs_active = feature_selection_active

    # Validate the algorithm
    if not isinstance(cfg_or_algo, (str, dict)):
        raise TrainingError("Invalid algorithm")
    algo_key = algo.lower()

    if not hyperparams and algo_key not in ["isotonic", "isotonic_regression"]:
        raise TrainingError("Model parameters are not specified")

    # Create and train the ModelTrainer
    trainer = ModelTrainer(
        algorithm=algo,
        hyperparams=hyperparams,
        metric=metric,
        random_state=rs,
        data_oversampling=data_os,
        data_oversampling_multiplier=data_os_mult,
        data_oversampling_algorithm=data_os_alg,
        feature_selection_cfg=fs_cfg,
        feature_selection_active=fs_active,
    )
    trainer.fit(X, y)
    val_score = trainer.val_score
    if val_score is None:
        raise TrainingError("Model did not return a metric value")

    # Logging (if enable_logging=True)
    if enable_logging:
        # Initialize file logging when an explicit log path is provided.
        # ``setup_logging`` attaches a file handler only once per process:
        # when one is already registered, the first path wins and this
        # argument does not re-route existing output (see the docstring).
        if log_path is not None:
            setup_logging(logfile=Path(log_path))
        # Get the logger and write the message
        logger = logging.getLogger(__name__)
        # Metric type follows the user-facing representation: errors (RMSE/MAE,
        # including neg_ metrics whose value is inverted back) are labeled as
        # "Error (Natural)", while score metrics (R² and similar) are labeled
        # as "Score".
        metric_type = "Error (Natural)" if is_error_metric(metric) else "Score"

        logger.info(
            f"Training finished: Algorithm={algo_key}, "
            f"Metric={metric.upper()} ({metric_type}), "
            f"Value={val_score:.4f} "
            f"(validation={trainer.validation_strategy})"
        )

    return float(val_score)
