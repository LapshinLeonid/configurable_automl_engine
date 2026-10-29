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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from configurable_automl_engine.common.definitions import (
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.common.serialization_utils import (
    save_artifact,
)
from configurable_automl_engine.common.settings import (
    InputPolicies,
    OversamplingSettings,
)
from configurable_automl_engine.feature_selection import (
    FeatureSelectionCfg,
)

# Re-exported: the documented import path of the isotonic data adapter.
from configurable_automl_engine.isotonic import IsotonicDataTransformer  # noqa: F401
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    PreprocessingPreset,
)

from .models import (
    create_model,
)
from .trainer_artifact import (
    _collect_artifact_metadata,
    load_trainer_artifact,
)
from .trainer_data import (
    assemble_pipeline_steps,
    build_trainer_preprocessor,
    load_training_data,
    log_preprocessed_matrix,
    prepare_features,
    prepare_predict_input,
)
from .trainer_feature_selection import _resolve_feature_selection_active
from .trainer_params import build_trainer_settings as _build_settings
from .trainer_quick import run_train_model
from .trainer_scoring import StepsFactory, score_on_validation_splits
from .trainer_state import (
    FeatureSelectionResult,
    PreparedFeatures,
    ValidationScores,
)

__all__ = ["ModelTrainer", "TrainingError", "train_model"]


class ModelTrainer:
    """Orchestrator of the training and validation process for regression models.

    Automates the construction of a pipeline that includes feature preprocessing
    (imputation, scaling, encoding), synthetic sample enlargement
    (oversampling), and training of the selected algorithm with mandatory validation.

    The state is grouped: the validated constructor settings
    (``settings``), the fit results of the pure steps (the prepared features,
    the validation estimate, the feature selection outcome) and the final
    ``pipeline``. The attributes below are read-only views of these groups,
    except ``pipeline``, ``preprocessing_preset``, ``datetime_expansion_``
    and ``artifact_metadata_``. The saved artifact records its format
    version (``trainer_state.ARTIFACT_FORMAT_VERSION``).
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
        settings (TrainerSettings): All validated constructor settings in
            one frozen object (the oversampling, encoding and input-policy
            groups included); the settings above are read-only views of it.
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
        # The constructor arguments are validated immediately (invalid values
        # are rejected at initialization time, before expensive training) and
        # kept as one frozen settings object.
        self.settings = _build_settings(
            algorithm=algorithm,
            hyperparams=hyperparams,
            metric=metric,
            random_state=random_state,
            serialization_format=serialization_format,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            id_column=id_column,
            encoding_strategy=encoding_strategy,
            high_cardinality_threshold=high_cardinality_threshold,
            high_cardinality_encoding=high_cardinality_encoding,
            hashing_n_components=hashing_n_components,
            target_encoding_smoothing=target_encoding_smoothing,
            target_encoding_fallback=target_encoding_fallback,
            additional_metrics=additional_metrics,
            preprocessing_override=preprocessing_override,
            feature_selection_cfg=feature_selection_cfg,
            feature_selection_active=feature_selection_active,
            validation_strategy=validation_strategy,
            n_folds=n_folds,
            test_size=test_size,
            oversampling=OversamplingSettings(
                enable=data_oversampling,
                multiplier=data_oversampling_multiplier,
                algorithm=data_oversampling_algorithm,
                target_bins=data_oversampling_target_bins,
                add_noise=data_oversampling_add_noise,
                noise_level=data_oversampling_noise_level,
            ),
            policies=InputPolicies(
                inf_policy=inf_policy,
                datetime_policy=datetime_policy,
                drop_id_like_columns=drop_id_like_columns,
            ),
            warn_id_like=warn_id_like,
        )
        # ── State populated by fit() (grouped results of the pure steps) ──
        # The final pipeline trained on 100% of the data.
        self.pipeline: Pipeline | None = None
        # The prepared features (names, column lists, dropped columns).
        self._features: PreparedFeatures | None = None
        # The validation estimate (val_score, additional scores, OOF channel).
        self._scores: ValidationScores | None = None
        # The feature selection outcome of the final fit.
        self._selection = FeatureSelectionResult()
        # The resolved preprocessing preset (documented attribute).
        self.preprocessing_preset: PreprocessingPreset | None = None
        # Datetime expansion applied to the training data: source column →
        # generated component columns (``datetime_policy="expand"``).
        # predict() applies the same expansion to raw input data, so a model
        # trained on a frame with datetime columns can predict on the same
        # raw schema. Populated in fit() and by register_datetime_expansion()
        # (the engine expands the data in prepare_dataset, before the trainer
        # sees it).
        self.datetime_expansion_: dict[str, list[str]] = {}
        # Artifact metadata, populated by :meth:`save` (and read back by
        # :meth:`load`). Contains the artifact format version, the engine
        # version, dependency versions, the creation timestamp, the
        # algorithm, hyperparameters and a hash of the artifact configuration.
        self.artifact_metadata_: dict[str, Any] | None = None

    # ── read-only views of the settings and of the fit results ────────────
    @property
    def algorithm(self) -> str:
        """The algorithm name."""
        return self.settings.algorithm

    @property
    def hyperparams(self) -> dict[str, Any]:
        """The model hyperparameters."""
        return self.settings.hyperparams

    @property
    def metric(self) -> str:
        """The primary metric (lower-cased)."""
        return self.settings.metric

    @property
    def random_state(self) -> int | None:
        """The seed of the splits and the model."""
        return self.settings.random_state

    @property
    def serialization_format(self) -> SerializationFormat:
        """The artifact format of :meth:`save`."""
        return self.settings.serialization_format

    @property
    def encoding_strategy(self) -> str:
        """The categorical encoding strategy."""
        return cast(str, self.settings.encoding.encoding)

    @property
    def additional_metrics(self) -> list[str]:
        """The additional (informational) metrics."""
        return self.settings.additional_metrics

    @property
    def feature_selection_cfg(self) -> FeatureSelectionCfg:
        """The validated feature selection configuration."""
        return self.settings.feature_selection_cfg

    @property
    def validation_strategy(self) -> str:
        """The validation strategy of the final model."""
        return self.settings.validation_strategy

    @property
    def n_folds(self) -> int:
        """The number of folds of ``k_fold``."""
        return self.settings.n_folds

    @property
    def test_size(self) -> float:
        """The hold-out size of ``train_test_split``."""
        return self.settings.test_size

    @property
    def categorical_features(self) -> list[str] | None:
        """The categorical columns (the detected ones after fit())."""
        if self._features is not None:
            return self._features.categorical_features
        return self.settings.categorical_features

    @property
    def numerical_features(self) -> list[str] | None:
        """The numerical columns (the detected ones after fit())."""
        if self._features is not None:
            return self._features.numerical_features
        return self.settings.numerical_features

    @property
    def feature_names(self) -> list[str] | None:
        """The feature names identified during training (``None`` before)."""
        return None if self._features is None else self._features.feature_names

    @property
    def dropped_features_(self) -> list[str]:
        """The columns removed as uninformative in the last fit()."""
        return [] if self._features is None else self._features.dropped_features

    @property
    def val_score(self) -> float | None:
        """The primary metric on held-out data (``None`` before fit())."""
        return None if self._scores is None else self._scores.val_score

    @property
    def additional_scores(self) -> dict[str, float]:
        """The additional metrics on held-out data (empty before fit())."""
        return {} if self._scores is None else self._scores.additional_scores

    @property
    def oof_predictions_(self) -> pd.Series | np.ndarray | None:
        """The out-of-fold prediction vector (``None`` before fit())."""
        return None if self._scores is None else self._scores.oof_predictions

    @property
    def oof_score_(self) -> float | None:
        """``RMSE_oof`` over the OOF vector (``None`` when impossible)."""
        return None if self._scores is None else self._scores.oof_score

    @property
    def oof_coverage_(self) -> float | None:
        """The fraction of rows covered by OOF predictions."""
        return None if self._scores is None else self._scores.oof_coverage

    @property
    def feature_selection_active_(self) -> bool:
        """Whether the final pipeline has the feature selection step."""
        return self._selection.active

    @property
    def selected_features_mask_(self) -> np.ndarray | None:
        """The selected-column mask of the post-preprocessor matrix."""
        return self._selection.mask

    # ── persistence of the object state ────────────────────────────────────
    def __getstate__(self) -> dict[str, Any]:
        """Control the object state for correct serialization
        (excluding lock and logger)."""
        state = self.__dict__.copy()
        state.pop("lock", None)
        state.pop("logger", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the object state after deserialization,
        initializing a thread-safe lock and a logger.

        The state is restored as is: ``load`` checks the artifact format
        version before using the object (``check_artifact_format``).
        """
        self.__dict__.update(state)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    # ── training ───────────────────────────────────────────────────────────
    def _create_base_model(self) -> Any:
        """Create the regressor via the model factory.

        The model seed follows the trainer's ``random_state`` (the engine
        seed from ``general.random_state``), unless the user set
        ``random_state`` explicitly in the hyperparameters — an explicit
        value always wins. ``random_state=None`` (trainer or explicit
        hyperparameter) produces a truly random fit. Auxiliary keys of the
        HPO result ``params`` are not model hyperparameters:
        ``use_feature_selection`` (the 'auto' mode decision), the data
        adapter hint ``feature_index`` and the informational
        ``isotonic_feature`` are popped, so that
        ``ModelTrainer(hyperparams=res["params"])`` does not fail on the
        strict ``create_model`` name check.

        Raises:
            TrainingError: If the model cannot be created (an unknown
                algorithm, an invalid hyperparameter or a missing package).
        """
        try:
            model_kwargs = self.hyperparams.copy()
            model_kwargs.pop("feature_index", None)
            model_kwargs.pop("use_feature_selection", None)
            model_kwargs.pop("isotonic_feature", None)
            if "random_state" not in model_kwargs:
                model_kwargs["random_state"] = self.random_state
            return create_model(self.algorithm, **model_kwargs)
        except (ValueError, ImportError) as e:
            raise TrainingError(f"Error creating model: {e}")

    def _assemble_steps(
        self,
        preprocessor: ColumnTransformer,
        base_model: Any,
        *,
        feature_selection_active: bool,
        X_for_estimate: Any = None,
    ) -> list[tuple[str, Any]]:
        """Assemble the training pipeline steps (see ``assemble_pipeline_steps``)."""
        return assemble_pipeline_steps(
            self.settings,
            preprocessor,
            base_model,
            categorical_features=self.categorical_features,
            numerical_features=self.numerical_features,
            feature_selection_active=feature_selection_active,
            X_for_estimate=X_for_estimate,
        )

    def _steps_factory(self, preprocessor: Any, base_model: Any) -> StepsFactory:
        """Build fresh pipeline steps from clones (the validation folds and
        the feature selection auto-check never share a fitted step)."""

        def make_steps(
            feature_selection_active: bool, X_for_estimate: Any
        ) -> list[tuple[str, Any]]:
            return self._assemble_steps(
                clone(preprocessor),
                clone(base_model),
                feature_selection_active=feature_selection_active,
                X_for_estimate=X_for_estimate,
            )

        return make_steps

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
        self._selection = FeatureSelectionResult()
        # 1) Assemble the pipeline steps and initialize the pipeline (imblearn's
        # Pipeline so that resampling steps work correctly)
        self.pipeline = Pipeline(
            steps=self._assemble_steps(
                preprocessor,
                base_model,
                feature_selection_active=feature_selection_active,
                X_for_estimate=X_train,
            )
        )
        # 2) Run the training
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
        # 3) Record the actual selection status and the selected-column mask.
        # The mask is taken from the selector step inside the pipeline and has
        # the dimensionality of the post-preprocessor matrix (after one-hot
        # expansion/scaling).
        selector_step = self.pipeline.named_steps.get("feature_selector")
        mask = (
            np.asarray(selector_step.support_, dtype=bool)
            if selector_step is not None and hasattr(selector_step, "support_")
            else None
        )
        self._selection = FeatureSelectionResult(
            active=feature_selection_active, mask=mask
        )
        return self.pipeline

    def _reset_fit_state(self) -> None:
        """Reset the training state at the start of ``fit()``.

        If training fails at any stage (data preparation, activity
        resolution, validation scoring, pipeline.fit), the trainer must not
        retain values from the previous training.
        """
        self.pipeline = None
        self._features = None
        self._scores = None
        self._selection = FeatureSelectionResult()
        self.datetime_expansion_ = {}

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

        The steps are pure functions (``load_training_data``,
        ``prepare_features``, ``build_trainer_preprocessor``,
        ``score_on_validation_splits``); the trainer keeps their results.
        """
        settings = self.settings
        with self.lock:
            self._reset_fit_state()

            # Stage 1: Validation and data preparation
            data = load_training_data(X, y)
            features = prepare_features(
                data.X,
                feature_names=data.feature_names,
                detected_dtypes=data.detected_dtypes,
                categorical_features=settings.categorical_features,
                numerical_features=settings.numerical_features,
                id_column=settings.id_column,
                policies=settings.policies,
                warn_id_like=settings.warn_id_like,
            )
            self._features = features
            self.datetime_expansion_ = dict(features.datetime_expansion)

            # Stage 2: Create the model via the factory. Done BEFORE
            # preprocessor assembly so that an unknown algorithm (a typo in
            # the name) is rejected here with a clear message rather than
            # being masked by the default preprocessing preset.
            base_model = self._create_base_model()

            # Stage 3: Build the preprocessor and resolve its preset.
            setup = build_trainer_preprocessor(
                settings.algorithm,
                features.categorical_features,
                features.numerical_features,
                feature_names=features.feature_names,
                encoding=settings.encoding,
                override=settings.preprocessing_override,
                random_state=settings.random_state,
            )
            self.preprocessing_preset = setup.preset
            make_steps = self._steps_factory(setup.preprocessor, base_model)

            # Stage 4: Resolve feature selection activity.
            # Priority: isotonic bypass -> explicit flag -> configuration mode
            # (always/disabled) -> standalone auto-check for mode='auto'.
            feature_selection_active = _resolve_feature_selection_active(
                features.X,
                data.y,
                make_steps,
                algorithm=settings.algorithm,
                explicit_flag=settings.feature_selection_active,
                fs_cfg=settings.feature_selection_cfg,
                metric=settings.metric,
                random_state=settings.random_state,
            )

            # Stage 5: Validation scoring. Quality evaluation is performed with
            # the same split method used for model comparison in HPO
            # (validation.iter_splits), so that val_score/additional_scores are
            # honest hold-out/CV estimates rather than train scores.
            self._scores = score_on_validation_splits(
                features.X,
                data.y,
                make_steps,
                feature_selection_active=feature_selection_active,
                metric=settings.metric,
                additional_metrics=settings.additional_metrics,
                validation_strategy=settings.validation_strategy,
                n_folds=settings.n_folds,
                test_size=settings.test_size,
                random_state=settings.random_state,
                algorithm=settings.algorithm,
            )

            # Stage 6: Final training on the full data (the artifact as before):
            # the pipeline is trained on 100% of the sample, while val_score
            # remains the estimate computed on held-out data (stage 5).
            self._fit_internal(
                features.X,
                data.y,
                setup.preprocessor,
                base_model,
                feature_selection_active=feature_selection_active,
            )

            # Observability: the dimensionality and the representation of the
            # preprocessed matrix that the model actually received (one-hot
            # expansion, missing-value indicators, sparse/dense storage).
            log_preprocessed_matrix(setup.preprocessor, features.X)

            return self

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
            self.datetime_expansion_ = {
                **{src: list(names) for src, names in expansion.items()},
                **self.datetime_expansion_,
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
                X_input = prepare_predict_input(
                    X,
                    self.feature_names,
                    self.datetime_expansion_,
                    self.settings.policies.inf_policy,
                )

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
            if self.pipeline is None:
                raise TrainingError("Nothing to save: model is not trained")

            path_obj = Path(path)
            # Create the directories if they do not exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Enrich the artifact with environment and configuration metadata
            # so that load() can detect version mismatches and the engine can
            # audit the provenance of the artifact.
            self.artifact_metadata_ = _collect_artifact_metadata(self.settings)

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

        The artifact format version is checked first: an artifact saved by
        another format (every engine up to 0.4.x) is refused with a clear
        error that says to retrain the model. After a successful load, a
        WARNING is logged when the artifact's engine or scikit-learn
        major/minor version differs from the one in the current environment.

        Args:
            path: Path to the saved artifact.
            fmt: Explicit serialization format. ``None`` (default) — the
                format is auto-detected from the extension with a joblib-then-
                pickle fallback.

        Returns:
            The deserialized :class:`ModelTrainer` instance.

        Raises:
            TrainingError: If the file does not exist, cannot be deserialized
                with any candidate format, does not contain a
                ``ModelTrainer`` instance, or was saved in another artifact
                format version.
        """
        return cast(ModelTrainer, load_trainer_artifact(path, fmt, cls))


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

    A facade that accepts a configuration dict or the positional
    ``(algorithm, metric, hyperparams, X, y)`` form, fits a
    :class:`ModelTrainer` and returns its validation score (the same
    estimate as ``ModelTrainer.fit()``, not a train score). The full
    description of the arguments and of the precedence of the config keys is
    in :func:`trainer_quick.run_train_model`.

    Args:
        cfg_or_algo: Configuration dict or algorithm name.
        metric_or_testsize: Metric (simple API) or ignored (config dict).
        params_or_metric: Hyperparameters (simple API) or ignored.
        X: Feature matrix.
        y: Target variable vector.
        enable_logging: Log the "Training finished" message.
        random_state: Random seed (simple API).
        log_path: Log file used when ``enable_logging`` is True.
        feature_selection_cfg: Feature selection configuration.
        feature_selection_active: Explicit feature selection flag.

    Returns:
        float: The validation score of the fitted trainer.

    Raises:
        TrainingError: For an invalid algorithm, missing hyperparameters or
            a trainer that returned no score.
    """
    return run_train_model(
        ModelTrainer,
        cfg_or_algo,
        metric_or_testsize,
        params_or_metric,
        X,
        y,
        enable_logging,
        random_state,
        log_path,
        feature_selection_cfg=feature_selection_cfg,
        feature_selection_active=feature_selection_active,
    )
