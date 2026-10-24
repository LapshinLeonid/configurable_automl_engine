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
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from configurable_automl_engine.common.definitions import (
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.serialization_utils import (
    load_artifact,
    save_artifact,
)
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
    _clean_uninformative_features,
    build_preprocessor,
)
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    PreprocessingPreset,
    requires_missing_indicator,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from configurable_automl_engine.training_engine.logger import setup_logging
from configurable_automl_engine.training_engine.metrics import (
    direction_label,
    get_scorer_object,
    is_error_metric,
    is_non_additive_metric,
    oof_rmse,
    pooled_scorer_value,
    to_user_value,
    warn_pooled_fallback_once,
)
from configurable_automl_engine.training_engine.thread_pool import SharedDataFrame
from configurable_automl_engine.validation import (
    SUPPORTED_VALIDATION_METHODS,
    iter_splits,
    make_cv,
    norm_val_method,
)

from .models import (
    create_model,
    requires_dense_input,
    resolve_algorithm_name,
)

__all__ = ["ModelTrainer", "TrainingError", "train_model"]

#: Minimum number of rows for the standalone feature selection check (mode
#: ``auto``). The threshold ensures that both parts of the hold-out split (80/20)
#: contain at least 2 samples: 10 * 0.8 = 8 and 10 * 0.2 = 2.
_MIN_AUTO_CHECK_SAMPLES = 10


class TrainingError(RuntimeError):
    """Exception raised for errors related to data or training parameters."""


class IsotonicDataTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Transformer for preparing data for IsotonicRegression.
    Selects the specified column and ensures correct handling of missing values
    to meet the requirements of the isotonic regression algorithm.
    Attributes:
        feature_index (int): Ordinal index of the feature to extract.
        median_ (float): Computed median used to fill missing values.
    """

    def __init__(self, feature_index: int = 0):
        """Initialize the transformer with the index of the target feature."""
        self.feature_index = feature_index

    def _get_dimensions(self, X: Any) -> tuple[int, int]:
        """Return the number of rows and columns of the input data
        in a universal format."""
        if hasattr(X, "shape"):
            return X.shape[0], (X.shape[1] if len(X.shape) > 1 else 1)
        n_rows = len(X)
        n_cols = len(X[0]) if n_rows > 0 and isinstance(X[0], (list, np.ndarray)) else 1
        return n_rows, n_cols

    def fit(self, X: Any, y: Any = None) -> IsotonicDataTransformer:
        """Compute the median of the selected feature for subsequent missing-value imputation."""
        _, n_cols = self._get_dimensions(X)
        idx = self.feature_index if self.feature_index < n_cols else 0

        X_col: pd.Series | np.ndarray

        if isinstance(X, pd.DataFrame):
            X_col = X.iloc[:, idx]
        elif isinstance(X, np.ndarray):
            X_col = X[:, idx] if len(X.shape) > 1 else X
        else:
            X_col = np.asarray(X)[:, idx]

        s_col = pd.Series(X_col)
        self.median_ = float(s_col.median()) if not s_col.isna().all() else 0.0
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Extract the feature, fill missing values,
        and convert to a 2D array (n_samples, 1)."""
        try:
            # 1. Get metadata without forcing a copy into a DataFrame
            # We use the logic from _extract_metadata discussed earlier
            _n_rows, n_cols = self._get_dimensions(X)

            # 2. Validate the feature index
            if self.feature_index >= n_cols:
                raise TrainingError(
                    f"Data transformation error: feature_index {self.feature_index} "
                    f"out of bounds. Dataset has only {n_cols} columns."
                )
            # 3. Extract the column (zero-copy for numpy and pandas)
            # If X is a DataFrame, use iloc. If numpy, use regular slicing.
            X_col: pd.Series | np.ndarray
            if isinstance(X, pd.DataFrame):
                X_col = X.iloc[:, self.feature_index]
            elif isinstance(X, np.ndarray):
                X_col = X[:, self.feature_index]
            else:
                # For lists or other types, convert to numpy (minimal copy)
                X_col = np.asarray(X)[:, self.feature_index]
            # 4. NaN handling logic
            # Check whether all values in the column are NaN
            # Use pd.Series for convenient median computation,
            # if it is not already a Series
            s_col = X_col if isinstance(X_col, pd.Series) else pd.Series(X_col)

            if s_col.isna().all():
                raise TrainingError(
                    f"Column with index {self.feature_index} contains only NaN values."
                    " Training IsotonicRegression is impossible."
                )
            # Compute the median and fill missing values
            fill_value = getattr(self, "median_", 0.0)
            X_imputed = s_col.fillna(fill_value)
            # Return the result as a 2D array, as scikit-learn expects
            result = X_imputed.to_numpy().reshape(-1, 1)
            return cast(np.ndarray, result)
        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            # Unify the error message according to the tests
            raise TrainingError(f"Data transformation error: {e}")


class ModelTrainer:
    """Orchestrator of the training and validation process for regression models.

    Automates the construction of a pipeline that includes feature preprocessing
    (imputation, scaling, encoding), synthetic sample enlargement
    (oversampling), and training of the selected algorithm with mandatory validation.
    Attributes:
        algorithm (str): Algorithm name or alias
            (e.g., 'elasticnet', 'xgboost').
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
        oof_score_ (float | None): ``RMSE_oof`` — an honest quality estimate on
            all rows without leakage: predictions from validation folds are
            concatenated into a single vector and the metric is computed on the
            whole vector at once (rather than averaged over small folds).
            ``None`` if an OOF estimate is impossible (no valid pairs at all).
            Separate channel: the ``val_score`` of HPO phases is not affected.
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
            ``'train_test_split'`` (default), ``'k_fold'``, ``'loo'`` or
            ``'auto'``. ``'auto'`` is resolved inside fit() via
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

    def __init__(
        self,
        algorithm: str = "elasticnet",
        hyperparams: dict[str, Any] | None = None,
        metric: str = "r2",
        random_state: int | None = 42,
        data_oversampling: bool = False,
        data_oversampling_multiplier: float = 1.0,
        data_oversampling_algorithm: str = "random",
        serialization_format: SerializationFormat = SerializationFormat.pickle,
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
        id_column: str | None = None,
        encoding_strategy: str = "one_hot",
        additional_metrics: Iterable[str] | None = None,
        preprocessing_override: PreprocessingOverride | dict[str, Any] | None = None,
        high_cardinality_threshold: int | None = None,
        high_cardinality_encoding: str | None = None,
        hashing_n_components: int = 16,
        target_encoding_smoothing: float = 20.0,
        target_encoding_fallback: float | None = None,
        feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
        feature_selection_active: bool | None = None,
        validation_strategy: ValidationStrategy | str = "train_test_split",
        n_folds: int = 5,
        test_size: float = 0.2,
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
                shared with HPO). Defaults to 'train_test_split' — a single
                hold-out split (a cheap "honest" quality estimate).
            n_folds: Number of folds for the 'k_fold' strategy (>= 2).
            test_size: Size of the hold-out part: a fraction in (0, 1) or an
                integer number of rows (as in sklearn ``train_test_split``).
                Defaults to 0.2.
        """

        self.logger = logging.getLogger(__name__)

        self.lock = threading.RLock()

        # Validate the algorithm
        if not isinstance(algorithm, str):
            raise TrainingError(f"Invalid algorithm: {algorithm!r}")
        self.algorithm = algorithm.lower()

        # hyperparams
        if hyperparams is not None:
            if not isinstance(hyperparams, dict):
                raise TrainingError("hyperparams must be a dictionary")
            self.hyperparams = hyperparams
        else:
            self.hyperparams = {}

        # Remaining parameters
        self.metric = metric.lower()
        self.random_state = random_state
        self.serialization_format = serialization_format
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.id_column = id_column
        # Categorical encoding strategy: 'one_hot' (default), 'ordinal',
        # 'target', 'frequency' or 'hashing'. Stored as a string primitive
        # to remain pickle-compatible.
        if encoding_strategy not in (
            "one_hot",
            "ordinal",
            "target",
            "frequency",
            "hashing",
        ):
            raise TrainingError(
                f"Unknown encoding_strategy: {encoding_strategy!r}. "
                "Expected one of ('one_hot', 'ordinal', 'target', 'frequency', "
                "'hashing')."
            )
        self.encoding_strategy: EncodingStrategy = cast(
            EncodingStrategy, encoding_strategy
        )

        # High-cardinality encoding parameters and new strategies.
        # Validated immediately so that invalid values are rejected at
        # initialization time, before training starts.
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.hashing_n_components = hashing_n_components
        self.target_encoding_smoothing = target_encoding_smoothing
        self.target_encoding_fallback = target_encoding_fallback
        if (high_cardinality_threshold is None) != (high_cardinality_encoding is None):
            raise TrainingError(
                "high_cardinality_threshold and high_cardinality_encoding must "
                "be set together (both provided or both None)."
            )
        if high_cardinality_threshold is not None and high_cardinality_threshold < 0:
            raise TrainingError(
                "high_cardinality_threshold must be >= 0, got "
                f"{high_cardinality_threshold}."
            )
        if high_cardinality_encoding is not None and high_cardinality_encoding not in (
            "one_hot",
            "ordinal",
            "target",
            "frequency",
            "hashing",
        ):
            raise TrainingError(
                f"Unknown high_cardinality_encoding: {high_cardinality_encoding!r}."
            )
        if hashing_n_components < 1:
            raise TrainingError(
                f"hashing_n_components must be >= 1, got {hashing_n_components}."
            )
        if target_encoding_smoothing < 0:
            raise TrainingError(
                f"target_encoding_smoothing must be >= 0, got "
                f"{target_encoding_smoothing}."
            )

        # ---------- additional (informational) metrics ----------
        # Any iterable sequence of strings is accepted (list, tuple,
        # set, generator, etc.). Scalar strings and mappings (dict) are rejected:
        # a string cannot be iterated character-by-character into metric names,
        # and iterating a dict over keys would be implicit and surprising behavior.
        # The iterable is materialized exactly once so that generators
        # and other one-shot iterators are validated correctly.
        if additional_metrics is not None and isinstance(
            additional_metrics, (str, bytes, Mapping)
        ):
            raise TrainingError("additional_metrics must be an iterable of strings")
        normalized_metrics = list(additional_metrics or [])
        if not all(isinstance(m, str) for m in normalized_metrics):
            raise TrainingError("additional_metrics must be an iterable of strings")
        self.additional_metrics: list[str] = [m.lower() for m in normalized_metrics]
        self.additional_scores: dict[str, float] = {}

        # Explicit override of the preprocessing preset: validate it immediately
        # so that invalid values are rejected at initialization time.
        self.preprocessing_override: PreprocessingOverride | None = (
            self._validate_preprocessing_override(preprocessing_override)
        )
        # The resolved preset is populated in _build_preprocessor during fit().
        self.preprocessing_preset: PreprocessingPreset | None = None

        # ---------- feature selection ----------
        # Validate the feature selection configuration immediately so that
        # invalid values (percentile=150, unknown method/mode, extra keys)
        # are rejected at initialization time, before expensive training.
        self.feature_selection_cfg: FeatureSelectionCfg = (
            self._validate_feature_selection_cfg(feature_selection_cfg)
        )
        # Explicit flag from HPO/component.py (takes precedence over the config);
        # None — the decision is made in fit() based on the configuration mode.
        # The type is strictly bool | None: the string "false" from the
        # configuration/YAML must not silently become True via bool("false") in
        # _resolve_feature_selection_active, so non-boolean values are rejected
        # at initialization time, like the other constructor parameters.
        if feature_selection_active is not None and not isinstance(
            feature_selection_active, bool
        ):
            raise TrainingError(
                "feature_selection_active must be a bool or None, got "
                f"{type(feature_selection_active).__name__}"
            )
        self.feature_selection_active: bool | None = feature_selection_active
        # Actual selection status in the final training (populated in fit()).
        self.feature_selection_active_: bool = False
        # Boolean mask of the selected columns in post-preprocessor dimensionality.
        self.selected_features_mask_: np.ndarray | None = None

        # ---------- validation ----------
        # Quality assessment strategy for the final model. Stored as a string
        # primitive (pickle compatibility); strings are normalized via
        # validation.norm_val_method — a single mapping source shared with HPO
        # (validation.make_cv / iter_splits). Validated immediately so that an
        # unknown strategy is rejected at initialization time.
        # The list of allowed methods is taken from
        # validation.SUPPORTED_VALIDATION_METHODS — the single extension point
        # of the strategy interface: adding a new strategy (e.g.,
        # TimeSeriesSplit) does not require changes here.
        normalized_strategy = norm_val_method(validation_strategy)
        if normalized_strategy not in SUPPORTED_VALIDATION_METHODS:
            raise TrainingError(
                f"Unknown validation_strategy: {validation_strategy!r}. "
                f"Expected one of {SUPPORTED_VALIDATION_METHODS}."
            )
        self.validation_strategy: str = normalized_strategy
        if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 1:
            raise TrainingError(f"n_folds must be a positive integer, got {n_folds!r}")
        if normalized_strategy == "k_fold" and n_folds < 2:
            raise TrainingError(
                f"n_folds must be >= 2 for k_fold validation, got {n_folds}"
            )
        self.n_folds: int = n_folds
        # test_size accepts a fraction in (0, 1) or an integer number of rows
        # (as in sklearn train_test_split): an integer test_size comes from the
        # engine path after 'auto' resolution (choose_validation_method returns
        # a row count).
        if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
            raise TrainingError(f"test_size must be a number, got {test_size!r}")
        if isinstance(test_size, float) and not (0.0 < test_size < 1.0):
            raise TrainingError(
                f"test_size must be a fraction in (0, 1), got {test_size!r}"
            )
        if isinstance(test_size, int) and test_size < 1:
            raise TrainingError(
                f"test_size must be a positive integer (row count), got {test_size!r}"
            )
        self.test_size: float = test_size

        # ---------- oversampling ----------
        self.os_enable = data_oversampling
        self.os_multiplier = data_oversampling_multiplier
        self.os_algorithm = data_oversampling_algorithm
        if self.os_multiplier < 1:
            raise TrainingError("data_oversampling_multiplier must be ≥ 1")
        if self.os_algorithm not in {"random", "smote", "adasyn"}:
            raise TrainingError("Unknown data_oversampling_algorithm")

        # Fields populated after fit(...)
        self.pipeline: Pipeline | None = None
        self.base_model: Any = None
        self.val_score: float | None = None
        # OOF candidate evaluation channel: an honest metric on all rows
        # without leakage. Each row is predicted by a model that has not seen it
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
        self._last_train_y: pd.Series | None = None
        self._last_val_y: pd.Series | None = None

        for attr_name, features in [
            ("categorical_features", categorical_features),
            ("numerical_features", numerical_features),
        ]:
            if features is not None and (
                not isinstance(features, list)
                or not all(isinstance(f, str) for f in features)
            ):
                raise TrainingError(
                    f"Parameter {attr_name} must be a list of strings (column names)"
                )

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Verify that all specified feature names are present in the given DataFrame."""

        specified_features = (self.categorical_features or []) + (
            self.numerical_features or []
        )
        missing = [col for col in specified_features if col not in X.columns]
        if missing:
            raise TrainingError(f"Specified columns not found in data: {missing}")

    def _detect_feature_types(self, X: pd.DataFrame, target_column: str) -> None:
        """Automatically classify columns into numerical and categorical ones."""
        with self.lock:
            # If both lists are already filled by the user, just validate them
            if (
                self.categorical_features is not None
                and self.numerical_features is not None
            ):
                self._validate_features(X)
                return
            # Determine the column list for analysis (excluding the target and ID)
            exclude = {target_column}
            if self.id_column:
                exclude.add(self.id_column)

            df_to_analyze = X.drop(columns=list(exclude & set(X.columns)))
            # Auto-detect categorical features (object, category, bool)
            if self.categorical_features is None:
                self.categorical_features = df_to_analyze.select_dtypes(
                    include=["object", "str", "category", "bool"]
                ).columns.tolist()
            # Auto-detect numerical features (all remaining 'number' dtypes)
            if self.numerical_features is None:
                self.numerical_features = df_to_analyze.select_dtypes(
                    include=["number"]
                ).columns.tolist()
            self.logger.info(
                f"Auto-detected features: {len(self.categorical_features)} cat, "
                f"{len(self.numerical_features)} num."
            )

    def _validate_preprocessing_override(
        self,
        override: PreprocessingOverride | dict[str, Any] | None,
    ) -> PreprocessingOverride | None:
        """Validate and normalize a user-provided preprocessing preset override.

        Args:
            override: A dict or a :class:`PreprocessingOverride` model.

        Returns:
            The normalized override model, or ``None``.

        Raises:
            TrainingError: If an object of an unsupported type or a dict with
                invalid values is passed.
        """
        if override is None:
            return None
        if isinstance(override, PreprocessingOverride):
            return override
        try:
            if isinstance(override, dict):
                return PreprocessingOverride.model_validate(override)
        except Exception as e:
            raise TrainingError(f"Invalid preprocessing_override: {e}") from e
        raise TrainingError(
            "preprocessing_override must be a dict or PreprocessingOverride, "
            f"got {type(override).__name__}"
        )

    def _validate_feature_selection_cfg(
        self,
        cfg: FeatureSelectionCfg | dict[str, Any] | None,
    ) -> FeatureSelectionCfg:
        """Validate and normalize the feature selection configuration.

        Logic follows :meth:`_validate_preprocessing_override`:
        ``None`` → default ``FeatureSelectionCfg()`` (mode='disabled');
        ``dict`` → ``FeatureSelectionCfg.model_validate(...)``, error → TrainingError;
        ``FeatureSelectionCfg`` → used as is;
        any other type → TrainingError.

        Args:
            cfg: A dict or a :class:`FeatureSelectionCfg` model.

        Returns:
            The normalized feature selection configuration model.

        Raises:
            TrainingError: If an object of an unsupported type or a dict with
                invalid values is passed.
        """
        if cfg is None:
            return FeatureSelectionCfg()
        if isinstance(cfg, FeatureSelectionCfg):
            return cfg
        try:
            if isinstance(cfg, dict):
                return FeatureSelectionCfg.model_validate(cfg)
        except Exception as e:
            raise TrainingError(f"Invalid feature_selection_cfg: {e}") from e
        raise TrainingError(
            "feature_selection_cfg must be a dict or FeatureSelectionCfg, "
            f"got {type(cfg).__name__}"
        )

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

        Delegates assembly to the shared :mod:`preprocessing` module, which is
        also used by the HPO phase in ``tuner.optimize`` (DRY, single logic).
        The numerical feature handling strategy (imputation and scaling) is
        selected automatically based on the algorithm via an adaptive
        preprocessing preset and is logged.
        """
        # Log a warning if no column matched
        cat_indices = [
            feature_names.index(col)
            for col in (self.categorical_features or [])
            if col in feature_names
        ]
        num_indices = [
            feature_names.index(col)
            for col in (self.numerical_features or [])
            if col in feature_names
        ]
        if not cat_indices and not num_indices:
            self.logger.warning(
                "No features matched for preprocessing. Defaulting to passthrough."
            )
        # Backward compatibility: models serialized before the encoding_strategy
        # parameter existed (the attribute is absent on unpickled objects) are
        # treated as one_hot by default. The new encoding parameters also get
        # their default values via getattr so that old saved models continue to
        # work without behavior changes.
        encoding: EncodingStrategy = cast(
            EncodingStrategy, getattr(self, "encoding_strategy", "one_hot")
        )
        hc_threshold: int | None = getattr(self, "high_cardinality_threshold", None)
        hc_encoding_raw: str | None = getattr(self, "high_cardinality_encoding", None)
        hc_encoding: EncodingStrategy | None = (
            cast(EncodingStrategy, hc_encoding_raw)
            if hc_encoding_raw is not None
            else None
        )
        hashing_n_components: int = getattr(self, "hashing_n_components", 16)
        target_smoothing: float = getattr(self, "target_encoding_smoothing", 20.0)
        target_fallback: float | None = getattr(self, "target_encoding_fallback", None)
        # Automatic preprocessing preset selection by algorithm, taking into
        # account the explicit override. The preset is resolved once before
        # training (performance) and recorded for observability.
        preset = self._resolve_preprocessing_preset()
        self.preprocessing_preset = preset
        self.logger.info(
            "Resolved preprocessing preset for algorithm '%s': %s",
            self.algorithm,
            preset,
        )
        # GPR/Isotonic/ARD algorithms reject sparse matrices: with hashing
        # encoding (sparse output) the preprocessor must return a dense matrix
        # via force_dense_output.
        force_dense = requires_dense_input(self.algorithm)
        # Strictly univariate algorithms (isotonic) are incompatible with
        # missing-value indicator columns. If the algorithm name is empty or
        # not set, a safe default value (True) is used.
        use_missing_indicator = (
            requires_missing_indicator(self.algorithm) if self.algorithm else True
        )
        return build_preprocessor(
            feature_names,
            self.categorical_features or [],
            self.numerical_features or [],
            encoding=encoding,
            imputation_strategy=preset.imputation_strategy,
            scaling=preset.scaling,
            high_cardinality_threshold=hc_threshold,
            high_cardinality_encoding=hc_encoding,
            hashing_n_components=hashing_n_components,
            target_encoding_smoothing=target_smoothing,
            target_encoding_fallback=target_fallback,
            random_state=self.random_state,
            force_dense_output=force_dense,
            add_indicator=use_missing_indicator,
        )

    def _prepare_data(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Validate the input data types
        and separate features from the target variable."""
        try:
            self.feature_names = self._extract_metadata(X)

            self._detected_dtypes = X.dtypes if isinstance(X, pd.DataFrame) else None
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
                self.feature_names = [col for col in X.columns if col != y]
                X_obj = X[self.feature_names]
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

    def _resolve_feature_selection_active(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
    ) -> bool:
        """Resolve the actual feature selection activity for fit().

        Priority (strictly in this order):
        1. IsotonicRegression — selection is forcibly disabled (the algorithm
           is strictly univariate and uses its own ``IsotonicDataTransformer``);
        2. the explicit ``feature_selection_active`` flag (from HPO/component.py)
           takes precedence over the configuration;
        3. the mode from ``feature_selection_cfg.mode``: ``always`` → True,
           ``disabled`` → False, ``auto`` → Standalone Auto-Check.

        Backward compatibility with old pickles: attributes are read via
        ``getattr`` (similar to ``encoding_strategy`` in ``_build_preprocessor``);
        missing attributes result in disabled behavior.

        Args:
            X_train: Training feature matrix (for the ``auto`` mode).
            y_train: Target variable (for the ``auto`` mode).
            preprocessor: Preprocessing ``ColumnTransformer`` (for the ``auto`` mode).
            base_model: Final regressor (for the ``auto`` mode).

        Returns:
            ``True`` if the selection step should be present in the pipeline.
        """
        # 1. Isotonic bypass: a univariate algorithm is incompatible with feature selection.
        algo_key = resolve_algorithm_name(self.algorithm)
        if algo_key == "isotonic_regression":
            self.logger.debug(
                "Feature selection is forcibly disabled for IsotonicRegression "
                "(univariate algorithm with its own data transformer)."
            )
            return False

        # 2. The explicit flag from HPO/component.py takes precedence over the config.
        explicit_flag = getattr(self, "feature_selection_active", None)
        if explicit_flag is not None:
            return bool(explicit_flag)

        # 3. Decision based on the configuration mode.
        cfg = getattr(self, "feature_selection_cfg", None)
        if not isinstance(cfg, FeatureSelectionCfg):
            cfg = FeatureSelectionCfg()
        mode = cfg.mode
        if mode == FeatureSelectionMode.always:
            return True
        if mode == FeatureSelectionMode.disabled:
            return False
        # mode == auto: standalone feasibility check for feature selection.
        return self._run_feature_selection_auto_check(
            X_train, y_train, preprocessor, base_model
        )

    def _run_feature_selection_auto_check(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
    ) -> bool:
        """Run the Standalone Auto-Check of feature selection usefulness.

        The ``auto`` mode is used when ``ModelTrainer.fit()`` is called
        directly without HPO. Logic:
        1. Hold-out split 80/20 with a fixed ``random_state``.
        2. Two control pipelines (without and with the selector) are trained
           on the train part; scoring is done on hold-out via
           ``get_scorer_object(self.metric, global_y=y_train)``.
        3. Directional comparison of "raw" scorer values: any scorer (from the
           registry or sklearn) is designed so that a larger value is better
           (neg_ metrics are already inverted for maximization), so no heuristic
           based on the metric name is needed and it does not break for
           neg_* metrics. Values are logged in user-facing semantics
           (``to_user_value``).
        4. Fail-safe: small data, split/scoring errors, or None/nan scores
           produce a WARNING and return ``False`` — the check does not crash fit().

        Cost: each ``fit()`` in ``auto`` mode additionally trains two full
        pipelines on 80% of the data — for heavy algorithms this is roughly
        2-3x the regular training time. The mode targets direct calls without
        HPO; within an HPO loop, selection activity should be passed via the
        explicit ``feature_selection_active`` flag so the check is not run for
        every trial.

        No side effects: the method does not touch ``self.pipeline``,
        ``self.val_score`` or ``self.feature_names``.

        Args:
            X_train: Training feature matrix.
            y_train: Target variable.
            preprocessor: Preprocessing ``ColumnTransformer``.
            base_model: Final regressor.

        Returns:
            ``True`` if the quality with selection is strictly higher, otherwise ``False``.
        """
        n_samples = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
        if n_samples < _MIN_AUTO_CHECK_SAMPLES:
            self.logger.warning(
                "Standalone feature selection auto-check skipped: only %d "
                "samples available, need at least %d. Feature selection "
                "disabled.",
                n_samples,
                _MIN_AUTO_CHECK_SAMPLES,
            )
            return False
        try:
            # Deterministic hold-out split: with trainer random_state=None
            # (non-fixed main training) the split must still be reproducible,
            # otherwise the selection decision would change between fit() calls.
            # Use a fixed fallback seed without touching the main training
            # configuration.
            check_random_state = (
                self.random_state if self.random_state is not None else 42
            )
            X_sub, X_hold, y_sub, y_hold = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=check_random_state,
            )
            # Control pipelines are isolated from each other and from the final
            # training: sklearn Pipeline does not clone the passed steps, so
            # shared preprocessor/base_model instances would mutate the state of
            # a neighboring pipeline on repeated fit.
            pipe_full = Pipeline(
                self._assemble_steps(
                    clone(preprocessor),
                    clone(base_model),
                    feature_selection_active=False,
                )
            )
            pipe_reduced = Pipeline(
                self._assemble_steps(
                    clone(preprocessor),
                    clone(base_model),
                    feature_selection_active=True,
                )
            )
            pipe_full.fit(X_sub, y_sub)
            pipe_reduced.fit(X_sub, y_sub)

            # global_y=y_train: the global NRMSE is normalized by the full
            # target (as in HPO and final validation scoring).
            scorer = cast(
                Callable[..., Any], get_scorer_object(self.metric, global_y=y_train)
            )
            raw_full = scorer(pipe_full, X_hold, y_hold)
            raw_reduced = scorer(pipe_reduced, X_hold, y_hold)
            if (
                raw_full is None
                or raw_reduced is None
                or not np.isfinite(float(raw_full))
                or not np.isfinite(float(raw_reduced))
            ):
                self.logger.warning(
                    "Standalone feature selection auto-check produced "
                    "non-finite scores (full=%r, reduced=%r). Feature "
                    "selection disabled.",
                    raw_full,
                    raw_reduced,
                )
                return False

            # The comparison direction is determined by the scorer object
            # itself: any scorer (from the registry or sklearn) is designed so
            # that a larger "raw" value is better — neg_ metrics are already
            # inverted for maximization. No heuristic based on the metric name
            # is used here, so the direction does not break for neg_* metrics.
            active = raw_reduced > raw_full
            # Log values in user-facing semantics (positive for errors), while
            # the decision is made on the "raw" values. The direction explicitly
            # accompanies the values ("min better" for errors) — the optimizer
            # greater_is_better flag is not used.
            score_full = to_user_value(self.metric, raw_full)
            score_reduced = to_user_value(self.metric, raw_reduced)
            self.logger.info(
                "Standalone feature selection auto-check: score_full=%.4f, "
                "score_reduced=%.4f (%s) -> active=%s",
                score_full,
                score_reduced,
                direction_label(self.metric),
                active,
            )
            return active
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                "Standalone feature selection auto-check failed: %s. "
                "Feature selection disabled.",
                e,
            )
            return False

    def _resolve_isotonic_feature_index(self) -> int:
        """Resolve the feature selected by the isotonic data adapter.

        The ``feature_index`` value is a user/HPO hyperparameter (not an
        ``IsotonicRegression`` constructor parameter): it is popped from the
        model kwargs before ``create_model`` and consumed here by
        ``IsotonicDataTransformer``. Although the preprocessor may output any
        number of columns (one-hot expansion, imputation), the adapter reduces
        them to a single feature — the input contract of the isotonic
        algorithm.

        Returns:
            The ordinal index of the feature to feed to ``IsotonicRegression``.

        Raises:
            TrainingError: If ``feature_index`` is not a non-negative integer.
        """
        raw = self.hyperparams.get("feature_index", 0)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise TrainingError(
                "feature_index must be a non-negative integer for "
                f"IsotonicRegression, got {raw!r}"
            )
        return raw

    def _assemble_steps(
        self,
        preprocessor: ColumnTransformer,
        base_model: Any,
        *,
        feature_selection_active: bool,
    ) -> list[tuple[str, Any]]:
        """Assemble the list of training pipeline steps.

        Strict step order:
        ``preprocessor → [isotonic_selector] → [feature_selector] → [oversampler] → model``.

        For ``isotonic_regression`` a data adapter step
        (:class:`IsotonicDataTransformer`) is inserted right after the
        preprocessor: the algorithm is strictly univariate and accepts exactly
        one feature, so even a multi-column preprocessed matrix is reduced to
        the single column selected via the ``feature_index`` hyperparameter.
        The feature selector is inserted strictly between the preprocessor and
        the oversampler so that synthetic rows are generated only for
        informative features.

        Args:
            preprocessor: Feature preprocessing ``ColumnTransformer``.
            base_model: Final regressor.
            feature_selection_active: Flag enabling the feature selection step.

        Returns:
            A list of ``(name, transformer)`` tuples for building the Pipeline.
        """
        steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
        if resolve_algorithm_name(self.algorithm) == "isotonic_regression":
            steps.append(
                (
                    "isotonic_selector",
                    IsotonicDataTransformer(
                        feature_index=self._resolve_isotonic_feature_index()
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
        whole vector at once — an honest estimate on all rows without leakage,
        separate from ``val_score``. After validation scoring, the final
        pipeline is trained on the full data: the artifact stores a training on
        100% of the sample, while ``val_score`` remains an honest quality
        estimate (not a train score).
        """

        with self.lock:
            # Stage 0: Reset the training state at the start of fit(), before
            # any data processing. If training fails at any stage (data
            # preparation, activity resolution, validation scoring, pipeline.fit),
            # the trainer must not retain values from the previous training
            # (the negative scenario with every fold failing). The duplicate
            # reset in _fit_internal is kept for direct calls to that method.
            self.pipeline = None
            self.val_score = None
            self.additional_scores = {}
            # OOF channel: reset together with the rest of the state so that a
            # failed fit() does not leave values from the previous training.
            self.oof_score_ = None
            self.oof_predictions_ = None
            self.oof_coverage_ = None
            self.feature_selection_active_ = False
            self.selected_features_mask_ = None
            self.dropped_features_ = []

            # Stage 1: Validation and data preparation
            X_prepared, y_s = self._prepare_data(X, y)

            n_samples = (
                X_prepared.shape[0] if hasattr(X_prepared, "shape") else len(X_prepared)
            )
            if n_samples < 2:
                raise TrainingError("Insufficient records for training")

            if isinstance(X_prepared, pd.DataFrame):
                # Cleaning uninformative features: fully empty and strictly
                # constant columns are removed BEFORE feature type detection and
                # preprocessor assembly, so that HPO and final training work
                # with a consistent feature set (a single point shared with
                # tuner.optimize). The decision is made only on the training
                # data (no leakage); for np.ndarray without column names no
                # cleaning is performed (a documented limitation, as in
                # detect_feature_types).
                cleaned = _clean_uninformative_features(X_prepared)
                self.dropped_features_ = [
                    col for col in X_prepared.columns if col not in cleaned.columns
                ]
                if self.dropped_features_:
                    X_prepared = cleaned
                    # Update feature_names to the cleaned list: predict()
                    # aligns columns by self.feature_names, so removed columns
                    # are dropped automatically at inference, while missing ones
                    # produce an explicit error.
                    if self.feature_names is not None:
                        self.feature_names = [
                            name
                            for name in self.feature_names
                            if name not in self.dropped_features_
                        ]
                    # User-specified column lists must also be consistent with
                    # the cleaned feature set: otherwise
                    # _detect_feature_types/build_preprocessor would fail on
                    # missing columns.
                    for attr in ("categorical_features", "numerical_features"):
                        features = getattr(self, attr)
                        if features is not None:
                            setattr(
                                self,
                                attr,
                                [
                                    feature
                                    for feature in features
                                    if feature not in self.dropped_features_
                                ],
                            )
                self._detect_feature_types(X_prepared, target_column="")
            else:
                if self.categorical_features is None:
                    self.categorical_features = []
                if self.numerical_features is None:
                    n_cols = X_prepared.shape[1] if len(X_prepared.shape) > 1 else 1
                    self.numerical_features = [f"col_{i}" for i in range(n_cols)]
                if self.feature_names is None:
                    self.feature_names = self.numerical_features

            # Stage 3: Create the model via the factory.
            # Done BEFORE preprocessor assembly so that an unknown algorithm
            # (a typo in the name) is rejected here with a clear message rather
            # than being masked by the default preprocessing preset.
            try:
                model_kwargs = self.hyperparams.copy()
                model_kwargs.pop("feature_index", None)
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
            self._log_preprocessed_matrix(preprocessor, X_prepared)

            return self

    def _log_preprocessed_matrix(self, preprocessor: Any, X: Any) -> None:
        """Log the shape and the storage format of the preprocessed matrix.

        Emits the INFO line ``preprocessed matrix: n × p (sparse/dense)`` for
        the training data after the final fit: ``n`` — rows, ``p`` — the
        number of features the model saw (after one-hot expansion /
        indicators), and the storage representation (``sparse`` for a
        ``scipy.sparse`` output — e.g. sparse one-hot/hashing — and
        ``dense`` otherwise). Used to make the categorical dimensionality
        visible in the logs (issue #157: dense one-hot on high-cardinality
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
            self.logger.info("preprocessed matrix: %d × %d (%s)", n_rows, n_cols, fmt)
        except Exception as err:  # noqa: BLE001 — observability only
            self.logger.debug(
                "Could not log the preprocessed matrix dimensions: %s", err
            )

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

        A single mechanism shared with HPO: splits are built via
        ``validation.iter_splits`` (the same factory used by ``tuner.optimize``),
        so ``val_score``/``additional_scores`` match the estimate used to
        compare models, up to sign correction.

        How it works:
        1. On each fold a fresh pipeline is assembled from clones of
           ``preprocessor``/``base_model`` (plus selector/oversampler per
           config) and trained on the fold's train part (a pattern taken from
           the feature selection auto-check and extracted into the shared
           ``_assemble_steps``).
        2. Raw values of the primary metric and all additional metrics are
           computed on the fold's validation part.
        3. ``val_score = to_user_value(metric, mean(raw))`` —
           ``additional_scores[m] = to_user_value(m, mean(raw_m))`` — for
           additive metrics; for non-additive metrics (RMSE, MSE, R², nRMSE,
           RMSLE, D², explained_variance, max error, median absolute error
           and their aliases) the value is the pooled estimate over the OOF
           vector (each row enters the calculation exactly once).
        4. In parallel, an OOF channel is collected: predictions from the
           validation parts of the folds are accumulated into a single
           ``y_pred_oof`` vector aligned to the original row order, and
           ``RMSE_oof`` is computed on the whole vector at once (see
           ``_finalize_oof``).

        A failure of a single additional metric (scorer error/None on a fold) →
        WARNING + the metric is skipped; failure of all folds of the primary
        metric → ``TrainingError`` (fail-fast, before final training).
        Non-finite fold scores (e.g., inf for NRMSE on a constant target) are
        returned as is. An OOF collection failure on a single fold does not
        crash the validation scoring: it only reduces the coverage of the OOF
        vector (WARNING).

        Note (strategy interface): building the OOF channel does not depend on
        a particular strategy — only the positional ``test_idx`` provided by
        ``iter_splits`` for any strategy is used. Order-aware strategies
        (``TimeSeriesSplit`` and similar) registered through
        ``validation.SUPPORTED_VALIDATION_METHODS``/``make_cv`` will work
        without rewriting this method.

        Args:
            X: Feature matrix (full data).
            y: Target variable (full data).
            preprocessor: Preprocessing ``ColumnTransformer``.
            base_model: Base regressor.
            feature_selection_active: Flag enabling the feature selection step.

        Raises:
            TrainingError: If no fold produced a valid score for the primary
                metric, or on a critical split-initialization error.
        """
        # Deterministic splits: with trainer random_state=None (non-fixed main
        # training) the validation splits must still be reproducible (a fixed
        # fallback seed, as in _run_feature_selection_auto_check).
        splits_seed = self.random_state if self.random_state is not None else 42
        # Backward compatibility with old pickles: the validation parameters
        # are read via getattr with defaults (the same pattern used for
        # feature_selection_cfg/encoding_strategy).
        method = getattr(self, "validation_strategy", "train_test_split")
        n_folds = getattr(self, "n_folds", 5)
        test_size = getattr(self, "test_size", 0.2)

        # global_y=y: the global NRMSE is normalized by the full target
        # (the range does not depend on a given fold).
        try:
            scorer = cast(
                Callable[..., Any], get_scorer_object(self.metric, global_y=y)
            )
        except Exception as err:  # noqa: BLE001
            raise TrainingError(f"Error calculating metrics on validation: {err}")
        add_scorers: dict[str, Callable[..., Any]] = {}
        for mname in self.additional_metrics:
            try:
                add_scorers[mname] = cast(
                    Callable[..., Any], get_scorer_object(mname, global_y=y)
                )
            except Exception as err:  # noqa: BLE001
                self.logger.warning(
                    "Additional metric '%s' could not be computed: %s. "
                    "Training result and artifact are unaffected.",
                    mname,
                    err,
                )

        # Materialize the splits before the fold loop: validation initialization
        # errors (insufficient data, unknown method, etc.) must abort training
        # immediately rather than being masked as a "failed fold".
        # include_indices=True: the splits additionally provide positional
        # indices of the validation rows for building an OOF vector aligned to
        # the original order.
        try:
            splits = list(
                iter_splits(
                    X,
                    y,
                    method=method,
                    n_folds=n_folds,
                    test_size=test_size,
                    random_state=splits_seed,
                    include_indices=True,
                )
            )
        except Exception as err:  # noqa: BLE001
            raise TrainingError(f"Error calculating metrics on validation: {err}")

        # The method actually applied after a possible fallback
        # (k_fold -> train_test_split for small N): OOF channel warnings must
        # rely on the splits actually used rather than the requested strategy.
        # Resolve via the same make_cv that iter_splits calls internally — the
        # values are guaranteed to match.
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        n_features = X.shape[1] if method == "auto" else None
        try:
            effective_method, _, _ = make_cv(
                n_samples,
                val_method=method,
                n_folds=n_folds,
                random_state=splits_seed,
                test_size=test_size,
                n_features=n_features,
            )
        except Exception:  # noqa: BLE001
            # Splits were already materialized above — we get here only in an
            # impossible/degenerate case; fall back to the requested method.
            effective_method = method

        fold_raw: list[float] = []
        add_raw: dict[str, list[float]] = {m: [] for m in add_scorers}
        fold_errors: list[str] = []
        # OOF channel buffers: filled by the positions of validation rows in
        # the original data; uncovered rows remain NaN and are discarded when
        # computing the pooled metrics (RMSE_oof and, for non-additive
        # metrics, val_score/additional_scores).
        oof_pred_buffer = np.full(n_samples, np.nan, dtype=float)
        oof_y_buffer = np.full(n_samples, np.nan, dtype=float)
        # Whether the primary/additional metric is non-additive and therefore
        # must be computed over the pooled OOF vector rather than averaged by
        # folds (RMSE, MSE, R², nRMSE). MAE and similar per-sample means stay
        # on the fold-average path (their fold mean equals the pooled value).
        pool_primary = is_non_additive_metric(self.metric)
        pool_additive = {mname: is_non_additive_metric(mname) for mname in add_scorers}
        for X_tr, X_te, y_tr, y_te, test_idx in splits:
            try:
                # A fresh pipeline per fold: clones of preprocessor/base_model
                # (sklearn Pipeline does not clone the passed steps, so shared
                # instances would mutate the state of a neighboring fold).
                fold_pipe = Pipeline(
                    self._assemble_steps(
                        clone(preprocessor),
                        clone(base_model),
                        feature_selection_active=feature_selection_active,
                    )
                )
                fold_pipe.fit(X_tr, y_tr)

                # For non-additive metrics the per-fold scorer is invoked only
                # until the first successful value (a fail-fast check that the
                # scorer works and does not return None): the final
                # val_score/additional_scores come from the pooled OOF vector,
                # so per-fold scoring noise (e.g. LOO+R², where every fold R²
                # is undefined) must not pollute the logs on the common path.
                # Additive metrics (MAE and similar per-sample means) keep
                # per-fold scoring on every fold — their fold mean IS the
                # estimate.
                if not pool_primary or not fold_raw:
                    raw = scorer(fold_pipe, X_te, y_te)
                    if raw is None:
                        raise TrainingError("Scorer returned None")
                    fold_raw.append(float(raw))

                for mname, m_scorer in add_scorers.items():
                    if pool_additive[mname] and add_raw[mname]:
                        continue
                    try:
                        m_raw = m_scorer(fold_pipe, X_te, y_te)
                        if m_raw is None:
                            raise TrainingError(
                                f"Scorer returned None for additional metric '{mname}'"
                            )
                        add_raw[mname].append(float(m_raw))
                    except Exception as err:  # noqa: BLE001
                        self.logger.warning(
                            "Additional metric '%s' could not be computed on a "
                            "validation fold: %s. Training result and artifact "
                            "are unaffected.",
                            mname,
                            err,
                        )

                # OOF channel: a prediction on the validation part of the fold
                # — the model was trained only on the train part, so the row is
                # guaranteed not to have been used in training (no leakage).
                # A separate try: an OOF collection failure does not crash the
                # fold scoring, it only reduces the OOF vector coverage (WARNING).
                try:
                    preds = np.asarray(fold_pipe.predict(X_te)).reshape(-1)
                    idx = np.asarray(test_idx)
                    if preds.shape[0] != idx.shape[0]:
                        raise TrainingError(
                            "OOF prediction length mismatch: got "
                            f"{preds.shape[0]} predictions for {idx.shape[0]} rows"
                        )
                    if y_te is not None:
                        oof_pred_buffer[idx] = preds
                        oof_y_buffer[idx] = np.asarray(y_te, dtype=float).reshape(-1)
                except Exception as err:  # noqa: BLE001
                    self.logger.warning(
                        "OOF predictions could not be collected on a "
                        "validation fold: %s. RMSE_oof coverage reduced.",
                        err,
                    )
            except Exception as err:  # noqa: BLE001
                fold_errors.append(str(err))

        if not fold_raw:
            detail = "; ".join(dict.fromkeys(fold_errors)) or "unknown failure"
            raise TrainingError(f"Validation scoring failed for all folds: {detail}")

        # Non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D²,
        # explained_variance, max error, median absolute error) are computed
        # over the pooled OOF vector at once, not averaged by folds: each row
        # enters the calculation exactly once. The pooled value is obtained
        # from the already-collected OOF buffer when the vector covers the
        # whole data (k_fold/loo); for train_test_split the buffer covers only
        # the hold-out part, so the metric is estimated on that subset — the
        # SAME subset as ``RMSE_oof`` (``_finalize_oof`` uses the identical
        # buffers), so val_score and oof_score_ stay mutually consistent by
        # construction. This is a documented limitation (identical to the OOF
        # channel: rows from the train part have no honest out-of-fold
        # prediction). The raw per-fold values still drive the fail-fast
        # check above (no fold produced a valid score).
        if pool_primary and np.any(
            np.isfinite(oof_pred_buffer) & np.isfinite(oof_y_buffer)
        ):
            try:
                raw_pooled = pooled_scorer_value(scorer, oof_y_buffer, oof_pred_buffer)
                raw_mean = raw_pooled
            except ValueError:
                # The pooled value is impossible (no valid pairs); fall back to
                # the fold mean (RMSE_oof handles its own None semantics). The
                # WARNING is emitted once per (algorithm, metric, 'final_fit').
                warn_pooled_fallback_once(
                    algo_name=self.algorithm,
                    phase_name=f"final_fit:{self.metric}",
                    reason="the pooled value over the OOF buffer cannot be "
                    "computed (no raw scorer metric function or no valid "
                    "pairs)",
                )
                raw_mean = float(np.mean(fold_raw))
        else:
            if pool_primary:
                warn_pooled_fallback_once(
                    algo_name=self.algorithm,
                    phase_name=f"final_fit:{self.metric}",
                    reason="the OOF buffer contains no finite (y_true, y_pred) pairs",
                )
            raw_mean = float(np.mean(fold_raw))
        self.val_score = to_user_value(self.metric, raw_mean)
        # User-facing semantics: val_score is a value in user representation,
        # so the direction in the log is user_direction ("min better" for
        # errors), not the optimizer greater_is_better flag (which is true for
        # neg_ metrics even though a smaller value is better for the user).
        self.logger.debug(
            f"Metric calculation: raw={raw_mean:.4f},"
            f" final val_score={self.val_score:.4f} "
            f"({direction_label(self.metric)})"
        )

        self.additional_scores = {}
        for mname, raw_values in add_raw.items():
            if not raw_values:
                # All folds failed for this metric — it was already skipped with a WARNING.
                continue
            if pool_additive[mname] and np.any(
                np.isfinite(oof_pred_buffer) & np.isfinite(oof_y_buffer)
            ):
                try:
                    pooled = pooled_scorer_value(
                        add_scorers[mname], oof_y_buffer, oof_pred_buffer
                    )
                    raw_values_agg = pooled
                except ValueError:
                    warn_pooled_fallback_once(
                        algo_name=self.algorithm,
                        phase_name=f"final_fit:{mname}",
                        reason=f"the pooled value for additional metric "
                        f"'{mname}' cannot be computed (no raw scorer metric "
                        f"function or no valid pairs)",
                    )
                    raw_values_agg = float(np.mean(raw_values))
            else:
                if pool_additive[mname]:
                    warn_pooled_fallback_once(
                        algo_name=self.algorithm,
                        phase_name=f"final_fit:{mname}",
                        reason=f"the OOF buffer contains no finite "
                        f"(y_true, y_pred) pairs for additional metric "
                        f"'{mname}'",
                    )
                raw_values_agg = float(np.mean(raw_values))
            self.additional_scores[mname] = to_user_value(mname, raw_values_agg)

        # OOF channel: RMSE_oof on the whole vector, warnings about
        # coverage/small N, and alignment of y_pred_oof to the original row order.
        self._finalize_oof(
            oof_pred_buffer,
            oof_y_buffer,
            X,
            n_samples,
            method=effective_method,
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

        Based on the prediction/target buffers collected on validation folds
        (``_score_on_validation_splits``), computes ``RMSE_oof`` over the whole
        vector at once via :func:`metrics.oof_rmse`, fills the
        ``oof_score_``/``oof_predictions_``/``oof_coverage_`` attributes, and
        logs warnings:
        - ``train_test_split`` covers only the validation part with OOF
          (a documented limitation);
        - very small N (2-5): a short OOF vector with high variance;
        - non-finite predictions/targets are excluded from RMSE_oof.

        Args:
            oof_pred: OOF prediction buffer (NaN for uncovered rows).
            oof_y: True value buffer (NaN for uncovered rows).
            X: Original feature matrix (to restore the pandas index).
            n_samples: Total number of rows in the original data.
            method: The validation method actually applied (after a possible
                fallback ``k_fold`` → ``train_test_split``), which drives the
                coverage warnings.
        """
        oof_vector = self._align_oof_vector(oof_pred, X)
        valid = np.isfinite(oof_pred) & np.isfinite(oof_y)
        n_valid = int(valid.sum())
        coverage = float(n_valid / n_samples) if n_samples > 0 else 0.0
        self.oof_coverage_ = coverage

        # train_test_split limitation: OOF covers only the validation part.
        # Explicitly documented in the log.
        if method == "train_test_split":
            self.logger.warning(
                "OOF evaluation with 'train_test_split' covers only the "
                "validation part (%d/%d rows, %.0f%%): rows from the training "
                "part have no honest out-of-fold prediction, so RMSE_oof is "
                "estimated on the covered subset only.",
                n_valid,
                n_samples,
                coverage * 100.0,
            )

        # Very small N: the OOF vector is short and the estimate has high variance.
        if n_samples <= 5:
            self.logger.warning(
                "OOF evaluation on very small data (N=%d): the OOF vector is "
                "short and RMSE_oof has high variance.",
                n_samples,
            )

        if n_valid == 0:
            self.oof_score_ = None
            self.oof_predictions_ = oof_vector
            self.logger.warning(
                "RMSE_oof could not be computed: no valid out-of-fold "
                "prediction/target pairs."
            )
            return

        dropped = n_samples - n_valid
        # For train_test_split the uncovered rows are the train part; the
        # limitation was already documented by a separate WARNING above, so the
        # generic row-exclusion warning is logged only for non-finite
        # predictions/targets.
        if dropped and method != "train_test_split":
            self.logger.warning(
                "%d row(s) excluded from RMSE_oof (non-finite prediction or "
                "target); coverage=%.3f.",
                dropped,
                coverage,
            )

        try:
            self.oof_score_ = oof_rmse(oof_y, oof_pred)
        except ValueError as err:
            self.oof_score_ = None
            self.logger.warning("RMSE_oof could not be computed: %s", err)
        self.oof_predictions_ = oof_vector
        # RMSE_oof is a user-facing error value ("min better"): the direction
        # is stated explicitly, and the optimizer flag is not applied.
        self.logger.info(
            "OOF evaluation: RMSE_oof=%.4f (min better) on %d/%d rows (coverage=%.3f).",
            self.oof_score_ if self.oof_score_ is not None else float("nan"),
            n_valid,
            n_samples,
            coverage,
        )

    def _align_oof_vector(
        self,
        oof_pred: np.ndarray,
        X: Any,
    ) -> pd.Series | np.ndarray:
        """Align the OOF vector ``y_pred_oof`` to the original row order.

        For pandas inputs (``DataFrame``/``Series``) with a unique index, a
        ``pd.Series`` with preserved row labels is returned, including
        MultiIndex. For numpy inputs and data with an unrecoverable index
        (duplicate labels), an ``np.ndarray`` aligned by position is returned:
        no sorting by index value is performed.

        Note: the ``Index.hasnans`` property is not defined for ``MultiIndex``
        (``NotImplementedError``), so the check is wrapped in try/except —
        otherwise ``fit()`` would fail on data with a MultiIndex even though
        such input worked on main.

        Args:
            oof_pred: OOF prediction buffer (NaN for uncovered rows), aligned
                by position to the original data.
            X: Original feature matrix.

        Returns:
            The ``y_pred_oof`` vector in the original row order.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)) and X.index.is_unique:
            try:
                has_nan_labels = bool(X.index.hasnans)
            except (NotImplementedError, TypeError):
                # MultiIndex and other indexes without hasnans support: NaN
                # labels cannot be checked, but uniqueness guarantees unambiguous
                # positional alignment — keep the row labels.
                has_nan_labels = False
            if not has_nan_labels:
                return pd.Series(oof_pred, index=X.index, name="y_pred_oof")
        return oof_pred

    def _align_predict_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align the DataFrame columns to the training order.

        A defensive layer for ``predict``: validates the presence of all
        ``self.feature_names``, drops extra columns, and returns a DataFrame
        with columns in the training order. This also fixes legacy serialized
        models whose preprocessor uses positional indices: after alignment, the
        positional slice again points to the same columns as during training.

        Args:
            X: Input DataFrame with the features to predict on.

        Returns:
            A DataFrame with columns in the training order (``self.feature_names``).

        Raises:
            TrainingError: If some of ``self.feature_names`` are missing in ``X``
                or appear in ``X`` more than once.
        """
        feature_names = self.feature_names
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

    def _align_predict_shared_dataframe(self, X: SharedDataFrame) -> Any:
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
            TrainingError: If some of ``self.feature_names`` are missing from
                the container columns.
        """
        if X.columns is None or not self.feature_names:
            return X.shared_array
        missing = [col for col in self.feature_names if col not in X.columns]
        if missing:
            raise TrainingError(
                f"Missing columns in prediction data: {missing}. "
                f"Expected columns: {self.feature_names}."
            )
        return X.get_view(self.feature_names)

    def predict(self, X: Any) -> np.ndarray:
        """Get model predictions for new data using the trained pipeline.

        For ``pd.DataFrame`` inputs, a defensive validation and alignment of
        the column order to the training order (``self.feature_names``) is
        performed: missing columns raise an explicit error, extra columns are
        dropped, and duplicated names are rejected. ``SharedDataFrame`` is
        restored into a DataFrame and aligned the same way when column names
        are available. For ``np.ndarray`` and other inputs without names, the
        positional column order fixed at training time is used.
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
                    X_input = self._align_predict_columns(X)
                elif isinstance(X, SharedDataFrame):
                    X_input = self._align_predict_shared_dataframe(X)
                elif isinstance(X, (pd.Series, np.ndarray)):
                    X_input = X
                else:
                    X_input = np.asarray(X)

                preds = self.pipeline.predict(X_input)
                return np.asarray(preds)

            except Exception as e:  # noqa: BLE001
                raise TrainingError(f"Error during prediction: {e}")

    def save(self, path: str | Path) -> None:
        """Serialize and save the current trainer instance to disk."""
        with self.lock:
            if self.pipeline is None and self.base_model is None:
                raise TrainingError("Nothing to save: model is not trained")

            path_obj = Path(path)
            # Create the directories if they do not exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Call the new utility instead of pickle.dump
            save_artifact(obj=self, path=path_obj, fmt=self.serialization_format)

    @classmethod
    def load(
        cls, path: str | Path, fmt: SerializationFormat = SerializationFormat.pickle
    ) -> ModelTrainer:
        """Load a previously saved ModelTrainer object from a file."""
        path_obj = Path(path)

        # Attempt to load via the utility
        try:
            obj = load_artifact(path=path_obj, fmt=fmt)
        except FileNotFoundError:
            raise TrainingError(f"File not found: {path}")
        except Exception as e:  # noqa: BLE001
            raise TrainingError(f"Error loading artifact: {e}")
        # Verify the type of the loaded object
        if not isinstance(obj, cls):
            raise TrainingError(f"Loaded object is not a ModelTrainer: {path}")

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
