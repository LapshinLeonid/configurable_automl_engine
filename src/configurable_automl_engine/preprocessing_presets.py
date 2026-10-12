"""Adaptive feature preprocessing presets.

This module implements automatic selection of the feature preprocessing strategy
depending on the selected regression algorithm.

Key concepts:

    * **Preprocessing preset** (:class:`PreprocessingPreset`) — a declaratively
      described set of rules for processing numeric features: the missing value
      imputation strategy (:data:`ImputationStrategy`) and the need for scaling
      together with its type (:data:`ScalingType`).
    * **Algorithm class** (:class:`AlgorithmClass`) — a group of regression
      algorithms with common preprocessing requirements.
    * **Mapping table** :data:`ALGORITHM_CLASS_MAPPING` — fixes
      the class membership of every supported algorithm. Adding a new algorithm
      does not require changing the core preprocessing logic —
      it is enough to register it in the table.

Selection rules:

    1. The algorithm name is normalized (lowercased and aliases are expanded) and
       checked against the registry of supported algorithms: an unknown name is
       rejected with a :class:`ValueError` (protection against typos in the config).
    2. The algorithm class is looked up in :data:`ALGORITHM_CLASS_MAPPING`; for a
       supported algorithm without explicit registration the ``default`` class is
       used (edge case — default behavior).
    3. A user override (:class:`PreprocessingOverride`)
       replaces preset fields fully or partially: an explicit user
       specification takes priority over automatic selection.

The single function :func:`resolve_preprocessing_preset` is used at every stage
of the lifecycle — the HPO phase, final training and prediction — which
guarantees consistent application of the preset.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    resolve_algorithm_name,
)

#: Missing value imputation strategies for numeric features.
ImputationStrategy = Literal["mean", "median"]

#: Types of scaling for numeric features.
ScalingType = Literal["standard", "robust", "none"]


class AlgorithmClass(str, Enum):
    """Classes of regression algorithms with common preprocessing requirements.

    Attributes:
        SCALE_SENSITIVE: Scale-sensitive models (linear, kernel-based,
            distance-based) — scaling is mandatory, missing values are filled
            with the standard strategy.
        TREES: Trees and tree ensembles — scaling is not required and should
            not be applied; missing value imputation is robust to outliers.
        GLM_SKEWED: Generalized linear models with skewed distributions —
            missing value imputation that accounts for data skewness; special
            scaling requirements.
        UNIVARIATE: Specialized univariate algorithms — non-standard
            preprocessing (no scaling).
        DEFAULT: Default behavior for algorithms without explicit registration.
    """

    SCALE_SENSITIVE = "scale_sensitive"
    TREES = "trees"
    GLM_SKEWED = "glm_skewed"
    UNIVARIATE = "univariate"
    DEFAULT = "default"


class PreprocessingPreset(BaseModel):
    """A declaratively described feature preprocessing preset.

    Attributes:
        algorithm_class: The algorithm class for which the preset is defined.
        imputation_strategy: The missing value imputation strategy for numeric
            features: ``'mean'`` (standard) or ``'median'``
            (robust to outliers, accounts for skewness).
        scaling: Whether scaling is needed and its type: ``'standard'``
            (StandardScaler), ``'robust'`` (RobustScaler, robust to outliers)
            or ``'none'`` (no scaling is applied).
        description: A human-readable description of the preset.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm_class: AlgorithmClass = Field(
        default=AlgorithmClass.DEFAULT,
        description="The algorithm class for which the preset is defined.",
    )
    imputation_strategy: ImputationStrategy = Field(
        default="mean",
        description=(
            "Missing value imputation strategy for numeric features: "
            "'mean' or 'median'."
        ),
    )
    scaling: ScalingType = Field(
        default="standard",
        description=("Scaling of numeric features: 'standard', 'robust' or 'none'."),
    )
    description: str = Field(
        default="",
        description="A human-readable description of the preset.",
    )

    def __str__(self) -> str:
        """Compact string for logging the selected preset."""
        return (
            f"PreprocessingPreset(class={self.algorithm_class.value!r}, "
            f"imputation_strategy={self.imputation_strategy!r}, "
            f"scaling={self.scaling!r})"
        )

    __repr__ = __str__


class PreprocessingOverride(BaseModel):
    """An explicit user override of the preprocessing preset.

    It may be specified partially or fully: ``None`` in a field means "keep the
    automatically selected value". Invalid values are rejected at the config
    validation stage (Pydantic) with a clear message.
    """

    model_config = ConfigDict(extra="forbid")

    imputation_strategy: ImputationStrategy | None = Field(
        default=None,
        description=(
            "Override of the missing value imputation strategy "
            "('mean' or 'median'). None — automatic selection."
        ),
    )
    scaling: ScalingType | None = Field(
        default=None,
        description=(
            "Scaling override ('standard', 'robust' or 'none'). "
            "None — automatic selection."
        ),
    )


#: Mapping table "algorithm -> algorithm class".
#: Keys are canonical algorithm names from ``AVAILABLE_ALGORITHMS``.
ALGORITHM_CLASS_MAPPING: dict[str, AlgorithmClass] = {
    # ── Scale-sensitive models (linear, kernel-based, distance-based) ──
    "elasticnet": AlgorithmClass.SCALE_SENSITIVE,
    "sgdregressor": AlgorithmClass.SCALE_SENSITIVE,
    "ridge": AlgorithmClass.SCALE_SENSITIVE,
    "lasso": AlgorithmClass.SCALE_SENSITIVE,
    "ardregression": AlgorithmClass.SCALE_SENSITIVE,
    "svr": AlgorithmClass.SCALE_SENSITIVE,
    "nearest_neighbors_regression": AlgorithmClass.SCALE_SENSITIVE,
    "gaussian_process_regression": AlgorithmClass.SCALE_SENSITIVE,
    # ── Trees and tree ensembles ──
    "decision_tree": AlgorithmClass.TREES,
    "random_forest": AlgorithmClass.TREES,
    "extra_trees": AlgorithmClass.TREES,
    "gradient_boosting": AlgorithmClass.TREES,
    "adaboost": AlgorithmClass.TREES,
    "xgboosting": AlgorithmClass.TREES,
    # ── GLM with skewed distributions ──
    "poissonregressor": AlgorithmClass.GLM_SKEWED,
    "gammaregressor": AlgorithmClass.GLM_SKEWED,
    "tweedieregressor": AlgorithmClass.GLM_SKEWED,
    "glm": AlgorithmClass.GLM_SKEWED,
    # ── Specialized univariate algorithms ──
    "isotonic_regression": AlgorithmClass.UNIVARIATE,
}

#: Preprocessing preset for each algorithm class.
PRESET_BY_CLASS: dict[AlgorithmClass, PreprocessingPreset] = {
    AlgorithmClass.SCALE_SENSITIVE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.SCALE_SENSITIVE,
        imputation_strategy="mean",
        scaling="standard",
        description=(
            "Scale-sensitive models: scaling is mandatory "
            "(StandardScaler), missing values are filled with the standard strategy (mean)."
        ),
    ),
    AlgorithmClass.TREES: PreprocessingPreset(
        algorithm_class=AlgorithmClass.TREES,
        imputation_strategy="median",
        scaling="none",
        description=(
            "Trees and tree ensembles: scaling is not required and is not "
            "applied; missing values are filled with the median (robust to outliers)."
        ),
    ),
    AlgorithmClass.GLM_SKEWED: PreprocessingPreset(
        algorithm_class=AlgorithmClass.GLM_SKEWED,
        imputation_strategy="median",
        scaling="robust",
        description=(
            "GLM with skewed distributions: missing values are filled with the median "
            "(robust to outliers, accounts for skewness); scaling with "
            "RobustScaler (robust to outliers)."
        ),
    ),
    AlgorithmClass.UNIVARIATE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.UNIVARIATE,
        imputation_strategy="median",
        scaling="none",
        description=(
            "Specialized univariate algorithms: scaling is not "
            "applied, missing values are filled with the median."
        ),
    ),
    AlgorithmClass.DEFAULT: PreprocessingPreset(
        algorithm_class=AlgorithmClass.DEFAULT,
        imputation_strategy="mean",
        scaling="standard",
        description=(
            "Default behavior for algorithms without explicit registration "
            "in the mapping table: mean + StandardScaler."
        ),
    ),
}


def normalize_algorithm(algorithm: str) -> str:
    """Normalize an algorithm name: lowercased + alias expansion.

    Args:
        algorithm: An algorithm name or alias (e.g. ``'RF'``, ``'xgb'``).

    Returns:
        The canonical algorithm name used for lookup in the mapping table.
    """
    return resolve_algorithm_name(algorithm)


def _coerce_override(
    override: PreprocessingOverride | dict[str, Any],
) -> PreprocessingOverride:
    """Coerce a user override into a :class:`PreprocessingOverride` model.

    Raises:
        TypeError: If an object of an unsupported type is passed.
    """
    if isinstance(override, PreprocessingOverride):
        return override
    if isinstance(override, dict):
        return PreprocessingOverride.model_validate(override)
    raise TypeError(
        f"preprocessing_override must be a dict or PreprocessingOverride, "
        f"got {type(override).__name__}"
    )


def resolve_preprocessing_preset(
    algorithm: str,
    override: PreprocessingOverride | dict[str, Any] | None = None,
) -> PreprocessingPreset:
    """Determine the preprocessing preset for a given algorithm.

    Selection logic:

    1. The algorithm name is normalized (lowercased + aliases) and checked
       against the registry of supported algorithms: an unknown name (typo)
       is rejected with an explicit error so that configuration mistakes are
       not hidden.
    2. The algorithm class is looked up in :data:`ALGORITHM_CLASS_MAPPING`; for a
       supported algorithm without explicit registration the ``default`` class
       is used (edge case — default behavior).
    3. If ``override`` is passed, its non-empty fields replace the corresponding
       fields of the selected preset (partial or full override).

    Args:
        algorithm: A regression algorithm name (or alias).
        override: An explicit preset override. ``None`` —
            only automatic selection.

    Returns:
        The final preprocessing preset for the algorithm.

    Raises:
        ValueError: If ``algorithm`` is not a supported algorithm.
        TypeError: If ``override`` has an unsupported type.
    """
    algo_key = normalize_algorithm(algorithm)
    if algo_key not in AVAILABLE_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            f"Available algorithms: {sorted(AVAILABLE_ALGORITHMS)}"
        )
    algo_class = ALGORITHM_CLASS_MAPPING.get(algo_key, AlgorithmClass.DEFAULT)
    preset = PRESET_BY_CLASS[algo_class]

    if override is not None:
        override_model = _coerce_override(override)
        update: dict[str, Any] = {}
        if override_model.imputation_strategy is not None:
            update["imputation_strategy"] = override_model.imputation_strategy
        if override_model.scaling is not None:
            update["scaling"] = override_model.scaling
        if update:
            preset = preset.model_copy(update=update)
    return preset


def requires_missing_indicator(algorithm: str) -> bool:
    """Whether missing value indicators are needed for the given algorithm.

    Missing value indicators are enabled for all algorithm classes except
    ``UNIVARIATE``: strictly univariate algorithms (``isotonic_regression``)
    accept exactly one feature, so an additional binary indicator column would
    violate their input contract. For the remaining algorithms the indicator
    lets the model distinguish "actually measured value" from "value recovered
    by imputation".

    Args:
        algorithm: A regression algorithm name (or alias).

    Returns:
        ``True`` — missing value indicators should be enabled; ``False`` —
        the algorithm is strictly univariate (the indicator is incompatible with
        its input).

    Raises:
        ValueError: If ``algorithm`` is not a supported
            algorithm (delegated to
            :func:`resolve_preprocessing_preset`). The caller must
            either guarantee the name is valid or handle the
            exception with a safe default value (``True``).
    """
    return (
        resolve_preprocessing_preset(algorithm).algorithm_class
        != AlgorithmClass.UNIVARIATE
    )
