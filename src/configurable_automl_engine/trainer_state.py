"""The grouped state of the single-model trainer (``ModelTrainer``).

``ModelTrainer`` keeps a few frozen value objects instead of dozens of flat
attributes: the validated constructor settings (``TrainerSettings``) and the
results of the pure ``fit()`` steps — the split input (``TrainingData``), the
prepared features (``PreparedFeatures``), the preprocessor with its preset
(``TrainerPreprocessor``), the validation estimate (``ValidationScores``) and
the feature selection outcome (``FeatureSelectionResult``). Fields holding a
data frame, an estimator or an array are excluded from the equality. The
module holds definitions only (it is part of the saved artifact format, see
``ARTIFACT_FORMAT_VERSION``) and must not import ``trainer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from configurable_automl_engine.common.definitions import SerializationFormat
    from configurable_automl_engine.common.settings import (
        EncodingSettings,
        InputPolicies,
        OversamplingSettings,
    )
    from configurable_automl_engine.preprocessing_presets import (
        PreprocessingOverride,
        PreprocessingPreset,
    )
    from configurable_automl_engine.training_engine.config_parser import (
        FeatureSelectionCfg,
    )

#: The version of the saved ``ModelTrainer`` state. It is recorded in the
#: artifact metadata; ``ModelTrainer.load`` refuses any other version.
#: 1 — the flat attributes of the engine up to 0.4.x (no recorded version);
#: 2 — the grouped state of 0.5.0.
ARTIFACT_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    """The validated constructor settings of a ``ModelTrainer``.

    The fields are the ``ModelTrainer`` constructor arguments after
    validation and normalization; see the constructor for their meaning. The
    grouped settings are ``encoding`` (``encoding_strategy`` and the
    high-cardinality / hashing / target-encoding arguments), ``oversampling``
    (the ``data_oversampling*`` arguments) and ``policies``
    (``inf_policy`` / ``datetime_policy`` / ``drop_id_like_columns``).
    """

    algorithm: str
    hyperparams: dict[str, Any]
    metric: str
    random_state: int | None
    serialization_format: SerializationFormat
    categorical_features: list[str] | None
    numerical_features: list[str] | None
    id_column: str | None
    encoding: EncodingSettings
    policies: InputPolicies
    warn_id_like: bool
    additional_metrics: list[str]
    preprocessing_override: PreprocessingOverride | None
    feature_selection_cfg: FeatureSelectionCfg
    feature_selection_active: bool | None
    validation_strategy: str
    n_folds: int
    test_size: float
    oversampling: OversamplingSettings


@dataclass(frozen=True, slots=True)
class TrainingData:
    """The validated training input split into features and target.

    Attributes:
        X: The feature matrix (a ``SharedDataFrame`` is resolved to its view;
            a target column named by ``y`` is removed).
        y: The target vector.
        feature_names: The feature names of ``X`` (``None`` when ``X`` has no
            column names).
        detected_dtypes: The dtypes of a ``pd.DataFrame`` input (``None``
            otherwise).
    """

    X: Any = field(compare=False)
    y: Any = field(compare=False)
    feature_names: list[str] | None
    detected_dtypes: Any = field(compare=False)


@dataclass(frozen=True, slots=True)
class PreparedFeatures:
    """The training features after the diagnostics, cleaning and typing.

    Attributes:
        X: The feature matrix the pipeline is trained on.
        feature_names: The feature names in training order (``predict``
            aligns its input to them).
        categorical_features: The categorical columns.
        numerical_features: The numerical columns.
        dropped_features: The columns dropped as uninformative (empty and
            strictly constant ones).
        datetime_expansion: Source datetime column → generated components.
        detected_dtypes: The dtypes of the input frame (``None`` for arrays).
    """

    X: Any = field(compare=False)
    feature_names: list[str]
    categorical_features: list[str]
    numerical_features: list[str]
    dropped_features: list[str]
    datetime_expansion: dict[str, list[str]]
    detected_dtypes: Any = field(compare=False)


@dataclass(frozen=True, slots=True)
class TrainerPreprocessor:
    """The (unfitted) preprocessor of the trainer and its resolved preset."""

    preprocessor: Any = field(compare=False)
    preset: PreprocessingPreset


@dataclass(frozen=True, slots=True)
class ValidationScores:
    """The quality estimate on held-out data (see ``ModelTrainer.fit``).

    Attributes:
        val_score: The primary metric in user semantics.
        additional_scores: The additional metrics in user semantics.
        oof_predictions: The out-of-fold prediction vector in the original
            row order (NaN for rows without an honest prediction).
        oof_score: ``RMSE_oof`` over the OOF vector (``None`` when
            impossible).
        oof_coverage: The fraction of rows with a finite OOF pair.
    """

    val_score: float
    additional_scores: dict[str, float]
    oof_predictions: pd.Series | np.ndarray | None = field(compare=False)
    oof_score: float | None
    oof_coverage: float | None


@dataclass(frozen=True, slots=True)
class FeatureSelectionResult:
    """The feature selection outcome of the final fit.

    Attributes:
        active: Whether the ``feature_selector`` step is in the pipeline.
        mask: The selected-column mask of the post-preprocessor matrix
            (``None`` when selection is not applied).
    """

    active: bool = False
    mask: np.ndarray | None = field(default=None, compare=False)
