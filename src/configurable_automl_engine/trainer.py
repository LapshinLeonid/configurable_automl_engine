"""Model Trainer: A universal orchestrator of the regression model lifecycle.

Encapsulates the training pipeline: data preprocessing, sample balancing,
and training of Scikit-learn algorithms. Feature preparation is fully
automated — the system classifies data types on its own, handles missing
values and scales features.

Main components:

ModelTrainer: The main container class for model training, validation, and deployment.
train_model: A functional interface (Facade) over :class:`ModelTrainer`.
TrainingError: A dedicated exception for unified handling of training failures.

Key implementation points:

* imblearn.pipeline integrates oversampling (SMOTE, ADASYN) into the
  training process;
* the optional feature selection step (``FeatureSelector``) sits strictly
  between the preprocessor and the oversampler; its activity is controlled
  by an explicit flag (HPO result) or by the configuration mode
  (always/disabled/auto). Mode ``auto`` performs a standalone feasibility
  check on each fit() (two additional pipeline fits, ~2-3x time), so within
  an HPO loop an explicit ``feature_selection_active`` flag is recommended;
* recursive locks (``threading.RLock``) guarantee thread safety when
  accessing the model state;
* a dedicated ``IsotonicDataTransformer`` adapter transforms multidimensional
  data to meet the strict requirements of isotonic regression;
* save/load support several serialization formats (Pickle, Joblib)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.utils import Tags, TargetTags

from configurable_automl_engine.common.custom_metrics import CustomMetricSpec
from configurable_automl_engine.common.datetime_features import (
    datetime_expansion_columns,
)
from configurable_automl_engine.common.definitions import (
    SerializationFormat,
    TargetTransform,
    ValidationStrategy,
)
from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.common.metrics import register_custom_metrics
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
from configurable_automl_engine.preprocessing import validate_target_for_transform
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
    TrainerViewMixin,
    ValidationScores,
)

__all__ = ["ModelTrainer", "TrainingError", "train_model"]


class ModelTrainer(TrainerViewMixin):
    """Orchestrator of the training and validation process for regression models.

    Automates the construction of a pipeline that includes feature preprocessing
    (imputation, scaling, encoding), synthetic sample enlargement (oversampling)
    and training of the selected algorithm with mandatory validation. The state
    is grouped: the validated constructor settings (``settings``), the fit
    results of the pure steps (features, validation estimate, feature
    selection outcome) and the final ``pipeline``. The attributes below are
    read-only views of these groups, except ``pipeline``,
    ``preprocessing_preset``, ``datetime_expansion_`` and
    ``artifact_metadata_``. The artifact records its format version
    (``trainer_state.ARTIFACT_FORMAT_VERSION``).

    Attributes:
        algorithm (str): Algorithm name (e.g., 'elasticnet', 'xgboosting').
        hyperparams (dict): Hyperparameter configuration for model init.
        metric (str): Metric key (r2, mse, mae) used to assess quality on validation.
        random_state (int | None): Random seed for reproducible splits and training.
        serialization_format (SerializationFormat): Storage format (pickle/joblib).
        categorical_features (list[str] | None): List of columns for One-Hot encoding.
        numerical_features (list[str] | None): List of columns to scale.
        id_column (str | None): Identifier excluded from the training process.
        encoding_strategy (str): Categorical encoding strategy
            ('one_hot', 'ordinal', 'target', 'frequency' or 'hashing'; default 'one_hot').
        additional_metrics (Iterable[str] | None): Additional quality metrics
            computed with the same validation method as the primary metric
            (``train_test_split`` — hold-out scoring, ``k_fold``/``loo`` —
            mean across folds for additive metrics, the pooled OOF value for
            non-additive ones (RMSE/MSE/R²/nRMSE/RMSLE/D²/explained_variance/
            max error/median absolute error and their aliases)). Any iterable
            sequence of strings is accepted (list, tuple, set, generator);
            the metric name is normalized to lowercase. Informational only;
            stored in ``additional_scores`` after fit().
        custom_metrics (dict[str, CustomMetricSpec]): Custom metrics and
            scorers loaded by a dotted import path (name → declaration);
            ``metric`` and ``additional_metrics`` may refer to their names.
        preprocessing_override (PreprocessingOverride | dict | None):
            Explicit override of the preprocessing preset (partial or full;
            takes precedence over automatic selection).
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
            aliases) — using the same validation method that HPO uses to
            compare models (``validation.iter_splits``). It is not a train
            score: the estimate is always computed on data not used to train
            the pipeline of that fold; error metrics are positive
            (sign-corrected).
        oof_score_ (float | None): ``RMSE_oof`` — an out-of-fold estimate on
            all rows: predictions from validation folds are concatenated into
            a single vector and the metric is computed on the whole vector at
            once (rather than averaged over small folds). With
            ``n_repeats > 1`` it is the median (or mean) of the per-repeat
            values. Every row is predicted by a fold-model that has not been
            trained on it, but the splits are the same as in HPO
            (``validation.iter_splits`` with the same seed): when the
            hyperparameters were selected on these splits, the estimate
            inherits the selection bias and is NOT an independent estimate of
            generalization (for RMSE it equals the best-trial ``val_score``).
            ``None`` if an OOF estimate is impossible; the ``val_score`` of
            HPO phases is not affected.
        oof_predictions_ (pd.Series | np.ndarray | None): The ``y_pred_oof``
            vector, aligned to the original row order (``pd.Series`` with the
            preserved pandas index for data with a unique index, including
            MultiIndex; ``np.ndarray`` by position for numpy inputs and data
            with duplicated indexes). Uncovered rows (e.g., the train part of
            ``train_test_split``) and non-finite predictions are NaN;
            ``None`` until the first ``fit()``.
        oof_coverage_ (float | None): Fraction of OOF cells covered over the
            concatenation of all repeats (1.0 for ``k_fold``/``loo``; < 1.0
            for ``train_test_split`` — the limitation is documented in the
            log). Per-repeat scores and their spread are exposed by
            ``repeat_scores_``/``score_mean``/``score_median``/``score_std``
            and ``oof_repeat_scores_``/``oof_score_std_``.
        additional_scores (dict[str, float]): Values of the additional metrics
            computed using the same validation method as the primary metric
            (fold mean for additive metrics, the pooled OOF value for
            non-additive ones; empty dict if no metrics are configured).
        validation_strategy (str): Validation strategy for the final model:
            ``'auto'`` (default, as ``general.validation_strategy``),
            ``'train_test_split'``, ``'k_fold'`` or ``'loo'``. ``'auto'`` is
            resolved inside fit() via ``validation.make_cv``; the engine path
            passes an already resolved method. The allowed methods are listed
            in ``validation.SUPPORTED_VALIDATION_METHODS`` (a single
            extension point: new strategies, e.g. order-aware
            ``TimeSeriesSplit``, are registered there and in ``make_cv``,
            after which the OOF channel works with them without rewriting).
        n_folds (int): Number of folds for the ``'k_fold'`` strategy (default 5).
        test_size (float | int): Size of the hold-out part: a fraction in
            (0, 1) or an integer number of rows (as in sklearn
            ``train_test_split``). Defaults to 0.2.
        feature_names (list[str] | None): List of feature names identified
            during training.
        dropped_features_ (list[str]): Names of columns removed by cleaning
            uninformative features (empty and strictly constant ones) in the
            last ``fit()`` call. Empty list if cleaning was not applied (input
            without column names) or no columns qualified. Populated only for
            ``pd.DataFrame`` inputs: ``np.ndarray`` inputs have no column
            names, so cleaning is not performed.
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
            the final training (False before the first fit).
        selected_features_mask_ (np.ndarray | None): Boolean mask of the
            selected columns after fit() (None if selection was not applied).
            The mask has the dimensionality of the post-preprocessor matrix
            (after one-hot expansion/scaling), not of the original columns;
            no mapping back to original names is performed — the filtering
            is transparent through the pipeline.
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
        custom_metrics: Mapping[str, CustomMetricSpec | Mapping[str, Any]]
        | None = None,
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
        target_transform: TargetTransform | str = TargetTransform.none,
        target_transform_skew_threshold: float | None = None,
        n_repeats: int = 1,
        split_aggregation: str = "median",
        _allow_auto_target_transform: bool = False,
    ):
        """Initialize the trainer with model parameters and preprocessing
        settings.

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
            target_transform: Transformation of the target ``y`` inside the
                ``model`` step (``TransformedTargetRegressor``): the model
                is trained on the transformed ``y`` and ``predict`` returns
                values in the original scale. ``"none"`` (default) — off;
                ``"log1p"`` requires ``min(y) > -1``, ``"boxcox"`` requires
                ``min(y) > 0`` (a ``ValueError`` from ``fit()`` on a domain
                violation), ``"yeo-johnson"`` — any real ``y``. See
                "Target transformation" in API_REFERENCE.md. ``"auto"`` is
                accepted only from the engine's no-hold-out mode (the
                fold-local resolution).
            target_transform_skew_threshold: The ``|skew(y)|`` threshold of
                the ``'auto'`` decision (``None`` — the module default;
                must be strictly positive). Only used with ``'auto'``.
            n_repeats: Number of repeated validation splits (>= 1). With
                ``n_repeats > 1`` the base validation strategy is re-drawn
                with the deterministic seeds ``random_state + r`` and the
                per-repeat estimates are combined by ``split_aggregation``;
                ``1`` (default) keeps the classic single-split behavior.
            split_aggregation: How the per-repeat estimates are combined:
                ``'median'`` (default, robust to one degenerate repeat) or
                ``'mean'``. Used only when ``n_repeats > 1``.
            _allow_auto_target_transform: Engine-internal: enables the
                ``'auto'`` placeholder above (direct callers keep the strict
                contract).
        """
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        # The constructor arguments are validated immediately (invalid values
        # are rejected before expensive training) and kept as one frozen
        # settings object.
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
            custom_metrics=custom_metrics,
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
            target_transform=target_transform,
            target_transform_skew_threshold=target_transform_skew_threshold,
            n_repeats=n_repeats,
            split_aggregation=split_aggregation,
            allow_auto_target_transform=_allow_auto_target_transform,
        )
        register_custom_metrics(self.settings.custom_metrics)
        # ── State populated by fit() (results of the pure steps) ──
        self.pipeline: Pipeline | None = None
        self._features: PreparedFeatures | None = None
        self._scores: ValidationScores | None = None
        self._selection = FeatureSelectionResult()
        # The resolved preprocessing preset (documented attribute).
        self.preprocessing_preset: PreprocessingPreset | None = None
        # Datetime expansion applied to the training data: source column →
        # generated component columns (``datetime_policy="expand"``).
        # predict() applies the same expansion to raw input data, so a model
        # trained on a frame with datetime columns can predict on the same
        # raw schema. Populated in fit() and by register_datetime_expansion()
        # (the engine expands the data in prepare_dataset before the trainer
        # sees it).
        self.datetime_expansion_: dict[str, list[str]] = {}
        # The expansion registered BEFORE fit (``register_datetime_expansion``
        # called before the training data is seen). Consumed by the next
        # ``fit``: it drives the winsorizer exclusion of the datetime-derived
        # component columns, is merged into ``datetime_expansion_`` and is
        # cleared, so a later refit starts clean.
        self._pre_fit_expansion: dict[str, list[str]] = {}
        # Artifact metadata, populated by :meth:`save` (read back by
        # :meth:`load`): the format version, versions, timestamp, the
        # algorithm, hyperparameters and a config hash.
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
    def custom_metrics(self) -> dict[str, CustomMetricSpec]:
        """The custom metric declarations (empty for an artifact without any)."""
        # Settings unpickled from an artifact written without this field have
        # no such slot.
        return getattr(self.settings, "custom_metrics", None) or {}

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
    def target_transform(self) -> TargetTransform:
        """The target transformation applied inside the ``model`` step."""
        return self.settings.target_transform

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
    def feature_selection_active_(self) -> bool:
        """Whether the final pipeline has the feature selection step."""
        return self._selection.active

    @property
    def selected_features_mask_(self) -> np.ndarray | None:
        """The selected-column mask of the post-preprocessor matrix."""
        return self._selection.mask

    # ── persistence of the object state ───────────────────────────────────
    def __getstate__(self) -> dict[str, Any]:
        """Control the object state for serialization (no lock/logger)."""
        state = self.__dict__.copy()
        state.pop("lock", None)
        state.pop("logger", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the object state, reinitializing the lock and logger.

        The state is restored as is: ``load`` checks the artifact format
        version before using it."""
        self.__dict__.update(state)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    # ── training ───────────────────────────────────────────────────────────
    def _create_base_model(self) -> Any:
        """Create the regressor via the model factory.

        The model seed follows the trainer's ``random_state``, unless the
        user set ``random_state`` explicitly in the hyperparameters (an
        explicit value wins); ``None`` produces a truly random fit.
        Auxiliary keys of the HPO result ``params`` are not model
        hyperparameters: ``use_feature_selection`` (the 'auto' mode
        decision), the data adapter hint ``feature_index`` and the
        informational ``isotonic_feature`` are popped, so that
        ``ModelTrainer(hyperparams=res["params"])`` does not fail on the
        strict ``create_model`` name check.
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
            raise TrainingError(f"Error creating model: {e}") from e

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
        """Assemble the final pipeline and run the training on the prepared
        data."""
        # Reset first: if pipeline.fit fails, the trainer must not
        # retain values from the previous training.
        self._selection = FeatureSelectionResult()
        # imblearn's Pipeline so that resampling steps work correctly.
        self.pipeline = Pipeline(
            steps=self._assemble_steps(
                preprocessor,
                base_model,
                feature_selection_active=feature_selection_active,
                X_for_estimate=X_train,
            )
        )
        try:
            self.pipeline.fit(X_train, y_train)
        except (ValueError, TypeError):
            # Re-raise validation errors directly so that tests can catch
            # them (invalid hyperparameters).
            raise
        except TrainingError:
            raise
        except Exception as e:
            # Wrap all other system errors in TrainingError
            self.logger.debug("Unexpected pipeline fit failure trace:", exc_info=True)
            self.logger.error(f"Unexpected error in pipeline fit: {e}")
            raise TrainingError(f"Internal training failure: {e}")
        # Record the actual selection status and the selected-column mask.
        # The mask is taken from the selector step inside the pipeline and
        # has the dimensionality of the post-preprocessor matrix (after
        # one-hot expansion/scaling).
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
        """Reset the training state at the start of ``fit()``: a failed
        fit must not retain the previous training's values."""
        self.pipeline = None
        self._features = None
        self._scores = None
        self._selection = FeatureSelectionResult()
        self.datetime_expansion_ = {}

    def fit(self, X: Any, y: Any) -> ModelTrainer:
        """Run the full cycle of data preparation, training and evaluation.

        Quality evaluation (``val_score`` / ``additional_scores``) uses the
        same validation method that HPO uses to compare models
        (``validation.iter_splits``): ``train_test_split`` — a single
        hold-out split scored on the held-out part; ``k_fold``/``loo`` —
        mean across folds for additive metrics, the pooled value over the
        OOF vector for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE,
        D², explained_variance, max error, median absolute error and their
        aliases). An OOF channel is also collected: predictions from
        validation folds are concatenated into ``y_pred_oof``, and
        ``RMSE_oof`` (``oof_score_``) is computed on the whole vector at
        once — an estimate on all rows without training leakage, separate
        from ``val_score``. Both use the HPO splits, so for hyperparameters
        selected on those splits neither is independent of the selection.
        After validation scoring, the final pipeline is trained on the full
        data, while ``val_score`` remains an honest quality estimate (not a
        train score).
        """
        settings = self.settings
        # A restored-from-artifact trainer starts with this process's
        # registry: register its metrics again.
        register_custom_metrics(self.custom_metrics)
        # The expansion registered BEFORE fit (the engine registers the map
        # on the trainer before calling fit — the data is already expanded
        # in prepare_dataset). Consumed once: the next fit starts clean.
        pre_registered = self._pre_fit_expansion
        self._pre_fit_expansion = {}
        with self.lock:
            self._reset_fit_state()
            # Stage 1: validation and data preparation.
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
            # Merge the pre-registered expansion with the fit-time map
            # (entries recorded by fit itself win).
            self._features = features
            self.datetime_expansion_ = {
                **pre_registered,
                **dict(features.datetime_expansion),
            }

            # Domain validation of the target (log1p / boxcox): an
            # invalid y fails fast, before any training. 'auto' (the
            # engine's no-hold-out mode) is a no-op here: the fold-local
            # resolution validates the resolved domain per fit.
            validate_target_for_transform(
                data.y, settings.target_transform, allow_auto=True
            )

            # The model is created BEFORE the preprocessor assembly so that
            # an unknown algorithm (a typo in the name) is rejected here
            # with a clear message rather than being masked by the default
            # preprocessing preset.
            base_model = self._create_base_model()

            # The datetime-derived component columns (``datetime_policy=
            # "expand"``) are excluded from winsorization: year/month/day
            # features are bounded by construction and clipping them would
            # only lose information (e.g. the first and the last month of
            # the data range).
            setup = build_trainer_preprocessor(
                settings.algorithm,
                features.categorical_features,
                features.numerical_features,
                feature_names=features.feature_names,
                encoding=settings.encoding,
                override=settings.preprocessing_override,
                random_state=settings.random_state,
                winsorizer_exclude=datetime_expansion_columns(self.datetime_expansion_),
            )
            self.preprocessing_preset = setup.preset
            make_steps = self._steps_factory(setup.preprocessor, base_model)

            # Feature-selection priority: isotonic bypass → explicit flag →
            # configuration mode (always/disabled) → standalone auto-check
            # for mode='auto'.
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

            # Validation scoring with the same split method used
            # for model comparison in HPO (validation.iter_splits), so
            # val_score/additional_scores are honest estimates rather than
            # train scores.
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
                n_repeats=settings.n_repeats,
                split_aggregation=settings.split_aggregation,
            )

            # Final training on the full data (the artifact): the
            # pipeline sees 100% of the sample, val_score stays the
            # held-out estimate.
            self._fit_internal(
                features.X,
                data.y,
                setup.preprocessor,
                base_model,
                feature_selection_active=feature_selection_active,
            )

            # Observability: the shape and representation of the
            # preprocessed matrix the model actually received.
            log_preprocessed_matrix(setup.preprocessor, features.X)
            return self

    def register_datetime_expansion(self, expansion: dict[str, list[str]]) -> None:
        """Record a datetime expansion applied to the data before ``fit``.

        Used when the expansion happened upstream of the trainer (the engine
        expands datetime columns in ``prepare_dataset``): ``predict`` then
        applies the same expansion to raw input data. Entries recorded by
        ``fit`` itself take precedence for the same source column.

        A registration BEFORE the first ``fit`` is additionally kept as the
        pre-fit expansion: the next ``fit`` uses it to exclude the generated
        component columns from winsorization (they are bounded by
        construction), merges it into ``datetime_expansion_`` for
        ``predict`` and consumes it — a later refit on its own data starts
        with a clean expansion state.

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
            if self.pipeline is None:
                # Pre-fit registration: remembered for the next fit (the
                # winsorizer exclusion) and merged into the recorded map.
                self._pre_fit_expansion = {
                    **self._pre_fit_expansion,
                    **{src: list(names) for src, names in expansion.items()},
                }
            self.datetime_expansion_ = {
                **{src: list(names) for src, names in expansion.items()},
                **self.datetime_expansion_,
            }

    def __sklearn_tags__(self) -> Any:
        """Declare the estimator tags sklearn tools resolve on the trainer.

        sklearn introspection helpers (``permutation_importance`` and the
        other tools that call ``check_scoring``) resolve the tags through
        the MRO and fail on a plain wrapper class. The tags are taken from
        the wrapped pipeline, so the NaN tolerance of the preprocessing and
        the rest of the tags match what the pipeline itself declares.
        An unfitted trainer has no pipeline and no meaningful tags.
        """
        if self.pipeline is not None:
            return self.pipeline.__sklearn_tags__()
        # An unfitted trainer: the sklearn default regressor tags.
        return Tags(estimator_type="regressor", target_tags=TargetTags(required=True))

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
        positional column order fixed at training time is used. The ±inf
        policy used at training time (``inf_policy``) is applied to the
        input as well, so inference is symmetric with :meth:`fit`:
        ``"as_missing"`` replaces ±inf in numeric features with NaN (the
        preprocessor imputes them) and logs a WARNING listing the columns
        and counts; ``"error"`` rejects the input.

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

        The artifact is written in ``self.serialization_format`` (pickle
        or joblib). Before writing, the instance is enriched with
        ``artifact_metadata_`` (engine and ML-stack versions, the creation
        timestamp, the algorithm, hyperparameters, the target transform
        and a configuration hash) so that :meth:`load` can warn about
        incompatible artifacts.

        Args:
            path: Destination path of the artifact (the directory is
                created if it does not exist).
        """
        with self.lock:
            if self.pipeline is None:
                raise TrainingError("Nothing to save: model is not trained")

            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Enrich the artifact with environment and configuration
            # metadata (version-mismatch detection, provenance audit).
            self.artifact_metadata_ = _collect_artifact_metadata(self.settings)

            save_artifact(obj=self, path=path_obj, fmt=self.serialization_format)

    @classmethod
    def load(
        cls, path: str | Path, fmt: SerializationFormat | None = None
    ) -> ModelTrainer:
        """Load a saved ModelTrainer object from a file.

        The serialization format is auto-detected when ``fmt`` is ``None``:
        the file extension decides (``.joblib`` → joblib, ``.pkl``/``.pickle``
        → pickle), the other format is tried as a fallback, and for an
        unknown extension joblib is tried first. An explicit ``fmt`` is used
        exclusively. The artifact format version is checked first (an
        artifact saved by another format is refused with an error that says
        to retrain the model); after a successful load a WARNING is logged
        when the artifact's engine or scikit-learn ``major.minor`` version
        differs from the current one.
        Args:
            path: Path to the saved artifact.
            fmt: Explicit serialization format. ``None`` (default) — the
                format is auto-detected from the extension with a joblib-
                then-pickle fallback.
        Returns:
            The deserialized :class:`ModelTrainer` instance.
        Raises:
            TrainingError: If the file does not exist, cannot be
                deserialized with any candidate format, does not contain
                a ``ModelTrainer`` instance, or was saved in another
                artifact format version.
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
    """Functional facade over :class:`ModelTrainer`.

    A facade that accepts a configuration dict or the positional
    ``(algorithm, metric, hyperparams, X, y)`` form, fits a
    :class:`ModelTrainer` and returns its validation score (the same
    estimate as ``ModelTrainer.fit()``, not a train score). The full
    argument/precedence description is in
    :func:`trainer_quick.run_train_model`.

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
        feature_selection_active: Explicit selection flag.
    Returns:
        float: The fitted trainer's validation score.
    Raises:
        TrainingError: For an invalid algorithm, missing hyperparameters,
            or a trainer that returned no score.
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
