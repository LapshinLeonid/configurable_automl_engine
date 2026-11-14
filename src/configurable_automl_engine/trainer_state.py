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

from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from configurable_automl_engine.common.definitions import TargetTransform

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from configurable_automl_engine.common.custom_metrics import CustomMetricSpec
    from configurable_automl_engine.common.definitions import SerializationFormat
    from configurable_automl_engine.common.settings import (
        EncodingSettings,
        InputPolicies,
        OversamplingSettings,
    )
    from configurable_automl_engine.feature_selection import (
        FeatureSelectionCfg,
    )
    from configurable_automl_engine.preprocessing_presets import (
        PreprocessingOverride,
        PreprocessingPreset,
    )

#: The version of the saved ``ModelTrainer`` state. It is recorded in the
#: artifact metadata; ``ModelTrainer.load`` refuses any other version.
#: 1 — the flat attributes of the engine up to 0.4.x (no recorded version);
#: 2 — the grouped state of the 0.5.0 development builds;
#: 3 — 0.5.0: the preprocessing encoders, ``ColumnNameSelector`` and
#: ``DataOversampler`` moved into the ``preprocessing`` / ``oversampling``
#: packages (a pickle records the module path of every class).
ARTIFACT_FORMAT_VERSION = 3


def _frozen_slots_setstate(self, state: Any) -> None:
    """Restore a frozen slots-dataclass state, tolerating shorter states.

    A pickle of a slots dataclass stores the field values as a list zipped
    against the current fields. Artifacts of the current format (3) saved
    before an additive field was introduced carry a shorter list: the
    default ``zip`` restores them without an error but leaves the trailing
    slots unset, and a later access would fail with ``AttributeError``.
    Fill the missing tail fields with their dataclass defaults (additive
    fields are appended with a default, so the legacy behavior is
    preserved).

    Raises:
        TypeError: If the state is not a list/tuple (corrupt pickle).
    """
    if not isinstance(state, (list, tuple)):
        raise TypeError(
            f"Unexpected {type(self).__name__} pickle state: {type(state).__name__}"
        )
    # Fields with defaults (the tail of the field list) keep their default
    # when the legacy state is shorter; fill from the state first, then
    # patch the unset tail with the defaults.
    fields_ = fields(self)
    for i, field_ in enumerate(fields_):
        if i < len(state):
            object.__setattr__(self, field_.name, state[i])
    for field_ in fields_[len(state) :]:
        if field_.default is not MISSING:
            object.__setattr__(self, field_.name, field_.default)


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    """The validated constructor settings of a ``ModelTrainer``.

    The fields are the ``ModelTrainer`` constructor arguments after
    validation and normalization; see the constructor for their meaning. The
    grouped settings are ``encoding`` (``encoding_strategy`` and the
    high-cardinality / hashing / target-encoding arguments), ``oversampling``
    (the ``data_oversampling*`` arguments) and ``policies``
    (``inf_policy`` / ``datetime_policy`` / ``drop_id_like_columns``).
    ``custom_metrics`` holds the custom metric declarations (data only: the
    artifact never stores the metric callables).
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
    custom_metrics: dict[str, CustomMetricSpec] = field(default_factory=dict)
    target_transform: TargetTransform = TargetTransform.none
    #: The ``|skew(y)|`` threshold of the fold-local ``target_transform:
    #: 'auto'`` resolution (``general.target_transform_skew_threshold``).
    #: ``None`` — the ``AutoTargetTransformer`` module default. Additive
    #: field: legacy pickles restore the default via ``__setstate__``.
    target_transform_skew_threshold: float | None = None
    #: The number of repeated validation splits (``general.n_repeats``).
    #: 1 — the classic single-split behavior. Additive field: legacy pickles
    #: restore the default via ``__setstate__``.
    n_repeats: int = 1
    #: The per-repeat score aggregation (``general.split_aggregation``):
    #: 'median' or 'mean'. Additive field: legacy pickles restore the
    #: default via ``__setstate__``.
    split_aggregation: str = "median"


# ``@dataclass(slots=True)`` rebuilds the class and (for frozen dataclasses)
# unconditionally overwrites ``__setstate__`` with its own ``zip``-based
# helper, which silently leaves trailing additive slots unset for a legacy
# (shorter) state list. Re-assert the custom implementation AFTER the
# decorator so it survives on every supported Python version.
TrainerSettings.__setstate__ = _frozen_slots_setstate  # type: ignore[attr-defined]


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
        oof_coverage: The fraction of rows with a finite OOF pair over the
            concatenation of all repeats (``n_samples * R`` cells).
        repeat_scores: The primary metric per repeat in user semantics
            (``None`` when ``n_repeats = 1``); a failed repeat is NaN.
        score_mean: The mean of the finite ``repeat_scores`` (``None`` when
            ``n_repeats = 1`` or no repeat is valid).
        score_median: The median of the finite ``repeat_scores`` (``None``
            under the same conditions).
        score_std: The population standard deviation of the finite
            ``repeat_scores`` (``None`` under the same conditions).
        oof_repeat_scores: ``RMSE_oof`` per repeat (``None`` when
            ``n_repeats = 1``); a repeat with non-finite predictions is
            recorded as the worst-score sentinel ``inf``.
        oof_score_std: The population standard deviation of the finite
            ``oof_repeat_scores`` (``None`` when ``n_repeats = 1`` or no
            repeat is valid).
        oof_predictions_by_repeat: The OOF prediction vector of every repeat
            aligned to the original row order (``None`` when
            ``n_repeats = 1``). A public diagnostic kept in memory at a cost
            of O(R x N); it is excluded from the saved artifact.
    """

    val_score: float
    additional_scores: dict[str, float]
    oof_predictions: pd.Series | np.ndarray | None = field(compare=False)
    oof_score: float | None
    oof_coverage: float | None
    repeat_scores: list[float] | None = None
    score_mean: float | None = None
    score_median: float | None = None
    score_std: float | None = None
    oof_repeat_scores: list[float] | None = None
    oof_score_std: float | None = None
    oof_predictions_by_repeat: list[pd.Series | np.ndarray] | None = field(
        default=None, compare=False
    )


def _validation_scores_getstate(self) -> list[Any]:
    """Serialize the scores without the per-repeat OOF vectors.

    The vectors are a diagnostic and are also used by the report path to
    aggregate non-RMSE metrics; persisting R vectors of N predictions would
    grow the artifact by O(R x N), so the saved state keeps the scalar
    per-repeat statistics and drops the vectors.
    """
    all_fields = fields(self)
    state = [getattr(self, field_.name) for field_ in all_fields]
    vector_index = next(
        i
        for i, field_ in enumerate(all_fields)
        if field_.name == "oof_predictions_by_repeat"
    )
    state[vector_index] = None
    return state


# The additive fields above are appended with defaults; re-assert the custom
# state restoration after the decorator (see ``_frozen_slots_setstate``).
ValidationScores.__setstate__ = _frozen_slots_setstate  # type: ignore[attr-defined]
ValidationScores.__getstate__ = _validation_scores_getstate  # type: ignore[attr-defined,method-assign]


class TrainerViewMixin:
    """Read-only views of the trainer settings and validation results.

    The state itself is owned by ``ModelTrainer`` (``settings`` and the
    private ``_scores``); this mixin only exposes the derived values, so the
    trainer module stays within its structural size limit.
    """

    settings: TrainerSettings
    _scores: ValidationScores | None

    @property
    def n_repeats(self) -> int:
        """The number of repeated validation splits."""
        return self.settings.n_repeats

    @property
    def split_aggregation(self) -> str:
        """How the per-repeat validation scores are combined."""
        return self.settings.split_aggregation

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
        """The fraction of OOF cells covered over all repeats."""
        return None if self._scores is None else self._scores.oof_coverage

    @property
    def repeat_scores_(self) -> list[float] | None:
        """The primary metric per repeat (``None`` when ``n_repeats = 1``)."""
        return None if self._scores is None else self._scores.repeat_scores

    @property
    def score_mean(self) -> float | None:
        """The mean of the per-repeat primary metric (``None`` if absent)."""
        return None if self._scores is None else self._scores.score_mean

    @property
    def score_median(self) -> float | None:
        """The median of the per-repeat primary metric (``None`` if absent)."""
        return None if self._scores is None else self._scores.score_median

    @property
    def score_std(self) -> float | None:
        """The standard deviation of the per-repeat primary metric."""
        return None if self._scores is None else self._scores.score_std

    @property
    def oof_repeat_scores_(self) -> list[float] | None:
        """``RMSE_oof`` per repeat (``None`` when ``n_repeats = 1``)."""
        return None if self._scores is None else self._scores.oof_repeat_scores

    @property
    def oof_score_std_(self) -> float | None:
        """The standard deviation of the per-repeat ``RMSE_oof``."""
        return None if self._scores is None else self._scores.oof_score_std

    @property
    def oof_predictions_by_repeat_(self) -> list[pd.Series | np.ndarray] | None:
        """The OOF prediction vector of every repeat (``None`` for one)."""
        return None if self._scores is None else self._scores.oof_predictions_by_repeat


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
