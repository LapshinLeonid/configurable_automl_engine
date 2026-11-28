"""Adaptive feature preprocessing presets: engine tables and resolution.

The configuration models (scaling/outlier specs, ``PreprocessingPreset``,
``PreprocessingOverride`` and the merge helper) live in
:mod:`automl_contracts.preprocessing` and are re-exported here. This module
keeps the algorithm-to-class tables, the preset resolution and the runtime
helpers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from automl_contracts.preprocessing import (
    NO_OUTLIERS,
    AlgorithmClass,
    ImputationStrategy,
    IqrOutliers,
    NoOutliers,
    NoScaling,
    ObjectOutlierSpec,
    ObjectScalingSpec,
    OutlierSpec,
    PreprocessingOverride,
    PreprocessingPreset,
    QuantileOutliers,
    QuantileScaling,
    RobustScaling,
    ScalingSpec,
    ScalingType,
    StandardScaling,
    StringOutlierSpec,
    StringScaling,
    coerce_outliers,
    coerce_scaling,
    format_outliers,
    merge_preprocessing_overrides,
    scaling_type,
)

from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    resolve_algorithm_name,
)


def log_tree_class_scaling_ineffective(
    preset: PreprocessingPreset,
    logger: logging.Logger,
) -> None:
    """Log one INFO line when a tree-class preset applies a non-``none`` scaler.

    Args:
        preset: The resolved preprocessing preset.
        logger: The logger that receives the informational record.
    """
    if (
        preset.algorithm_class == AlgorithmClass.TREES
        and scaling_type(preset.scaling) != "none"
    ):
        logger.info(
            "Scaling (type=%s) has no effect on split-based models "
            "(algorithm class 'trees'): the splits are invariant to monotone "
            "per-feature transforms.",
            scaling_type(preset.scaling),
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
#: The ``scaling`` values are expressed in the object form of
#: :class:`ScalingSpec` (e.g. ``RobustScaling()``); they are equivalent to
#: the string shorthand and keep the table independent of the coercion.
PRESET_BY_CLASS: dict[AlgorithmClass, PreprocessingPreset] = {
    AlgorithmClass.SCALE_SENSITIVE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.SCALE_SENSITIVE,
        imputation_strategy="mean",
        scaling=StandardScaling(),
        outliers="none",
        description=(
            "Scale-sensitive models: scaling is mandatory "
            "(StandardScaler), missing values are filled with the standard strategy (mean)."
        ),
    ),
    AlgorithmClass.TREES: PreprocessingPreset(
        algorithm_class=AlgorithmClass.TREES,
        imputation_strategy="median",
        scaling=NoScaling(),
        outliers="none",
        description=(
            "Trees and tree ensembles: scaling is not required and is not "
            "applied; missing values are filled with the median (robust to outliers)."
        ),
    ),
    AlgorithmClass.GLM_SKEWED: PreprocessingPreset(
        algorithm_class=AlgorithmClass.GLM_SKEWED,
        imputation_strategy="median",
        scaling=RobustScaling(),
        outliers="none",
        description=(
            "GLM with skewed distributions: missing values are filled with the median "
            "(robust to outliers, accounts for skewness); scaling with "
            "RobustScaler (robust to outliers)."
        ),
    ),
    AlgorithmClass.UNIVARIATE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.UNIVARIATE,
        imputation_strategy="median",
        scaling=NoScaling(),
        outliers="none",
        description=(
            "Specialized univariate algorithms: scaling is not "
            "applied, missing values are filled with the median."
        ),
    ),
    AlgorithmClass.DEFAULT: PreprocessingPreset(
        algorithm_class=AlgorithmClass.DEFAULT,
        imputation_strategy="mean",
        scaling=StandardScaling(),
        outliers="none",
        description=(
            "Default behavior for algorithms without explicit registration "
            "in the mapping table: mean imputation and StandardScaler."
        ),
    ),
}


def _coerce_override(
    override: PreprocessingOverride | Mapping[str, object],
) -> PreprocessingOverride:
    """Coerce a user override into a :class:`PreprocessingOverride` model."""
    if isinstance(override, PreprocessingOverride):
        return override
    if isinstance(override, Mapping):
        return PreprocessingOverride.model_validate(override)
    raise TypeError(
        f"preprocessing_override must be a dict or PreprocessingOverride, "
        f"got {type(override).__name__}"
    )


def normalize_algorithm(algorithm: str) -> str:
    """Normalize an algorithm name: lowercased.

    Args:
        algorithm: An algorithm name (e.g. ``'Random_Forest'``).

    Returns:
        The canonical algorithm name used for lookup in the mapping table.
    """
    return resolve_algorithm_name(algorithm)


def resolve_preprocessing_preset(
    algorithm: str,
    override: PreprocessingOverride | Mapping[str, object] | None = None,
    *,
    default_override: PreprocessingOverride | Mapping[str, object] | None = None,
) -> PreprocessingPreset:
    """Determine the preprocessing preset for a given algorithm.

    Selection logic:

    1. The algorithm name is normalized (lowercased) and checked
       against the registry of supported algorithms: an unknown name (typo)
       is rejected with an explicit error so that configuration mistakes are
       not hidden.
    2. The algorithm class is looked up in :data:`ALGORITHM_CLASS_MAPPING`; for a
       supported algorithm without explicit registration the ``default`` class
       is used (edge case — default behavior).
    3. Every preset field is resolved with the precedence order (later items
       win): class preset < ``default_override`` < ``override``. A field set
       at a higher level replaces the whole field from the lower level;
       a field left as ``None`` falls through to the lower level. Nested
       values are never merged.

    Args:
        algorithm: A regression algorithm name.
        override: An explicit per-algorithm preset override. ``None`` —
            no algorithm-level override.
        default_override: The run-level (general) preset override applied
            for every algorithm. Keyword-only. Fields left as ``None``
            fall through to the automatic selection; fields set here take
            priority over the class preset but yield to ``override``.

    Returns:
        The final preprocessing preset for the algorithm.

    Raises:
        ValueError: If ``algorithm`` is not a supported algorithm.
        TypeError: If an override has an unsupported type.
    """
    algo_key = normalize_algorithm(algorithm)
    if algo_key not in AVAILABLE_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            f"Available algorithms: {sorted(AVAILABLE_ALGORITHMS)}"
        )
    algo_class = ALGORITHM_CLASS_MAPPING.get(algo_key, AlgorithmClass.DEFAULT)
    preset = PRESET_BY_CLASS[algo_class]

    update: dict[str, object] = {}
    if default_override is not None:
        default_model = _coerce_override(default_override)
        if default_model.imputation_strategy is not None:
            update["imputation_strategy"] = default_model.imputation_strategy
        if default_model.scaling is not None:
            update["scaling"] = default_model.scaling
        if default_model.outliers is not None:
            update["outliers"] = default_model.outliers
    if override is not None:
        override_model = _coerce_override(override)
        if override_model.imputation_strategy is not None:
            update["imputation_strategy"] = override_model.imputation_strategy
        if override_model.scaling is not None:
            # The override value arrives as a string shorthand or an object
            # dict: normalize it to the object form up front, so the copy
            # below shares a validated (frozen) spec instance instead of a
            # raw dict.
            update["scaling"] = coerce_scaling(override_model.scaling)
        if override_model.outliers is not None:
            # The same normalization as for ``scaling``: the string shorthand
            # or an object dict is coerced to the validated (frozen) object
            # form, so the copy below shares a spec instance instead of a
            # raw dict.
            update["outliers"] = coerce_outliers(override_model.outliers)
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
        algorithm: A regression algorithm name.

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


__all__ = [
    "ALGORITHM_CLASS_MAPPING",
    "NO_OUTLIERS",
    "PRESET_BY_CLASS",
    "AlgorithmClass",
    "ImputationStrategy",
    "IqrOutliers",
    "NoOutliers",
    "NoScaling",
    "ObjectOutlierSpec",
    "ObjectScalingSpec",
    "OutlierSpec",
    "PreprocessingOverride",
    "PreprocessingPreset",
    "QuantileOutliers",
    "QuantileScaling",
    "RobustScaling",
    "ScalingSpec",
    "ScalingType",
    "StandardScaling",
    "StringOutlierSpec",
    "StringScaling",
    "coerce_outliers",
    "coerce_scaling",
    "format_outliers",
    "log_tree_class_scaling_ineffective",
    "merge_preprocessing_overrides",
    "normalize_algorithm",
    "requires_missing_indicator",
    "resolve_preprocessing_preset",
    "scaling_type",
]
