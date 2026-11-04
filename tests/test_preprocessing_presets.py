"""Unit tests for adaptive preprocessing presets.

Cover automatic selection of the preprocessing preset by algorithm class, the
mapping table, preset override by the user, and the behavior for
unregistered algorithms (edge case). The object form of ``scaling``
(:class:`ScalingSpec`) and its string shorthand are covered by the
``ScalingSpec`` test section.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from configurable_automl_engine.models import AVAILABLE_ALGORITHMS
from configurable_automl_engine.preprocessing_presets import (
    ALGORITHM_CLASS_MAPPING,
    PRESET_BY_CLASS,
    AlgorithmClass,
    NoScaling,
    PreprocessingOverride,
    PreprocessingPreset,
    RobustScaling,
    StandardScaling,
    coerce_scaling,
    normalize_algorithm,
    requires_missing_indicator,
    resolve_preprocessing_preset,
    scaling_type,
)


# ──────────────────────────────────────────────────────────────────────────
#  Automatic selection of the correct preset by algorithm
# ──────────────────────────────────────────────────────────────────────────
def test_scale_sensitive_models_get_standard_scaling():
    """Scale-sensitive models: scaling='standard', imputation='mean'."""
    for algo in [
        "elasticnet",
        "sgdregressor",
        "ridge",
        "lasso",
        "ardregression",
        "svr",
        "nearest_neighbors_regression",
        "gaussian_process_regression",
    ]:
        preset = resolve_preprocessing_preset(algo)
        assert preset.algorithm_class == AlgorithmClass.SCALE_SENSITIVE
        assert preset.scaling.type == "standard"
        assert preset.imputation_strategy == "mean"


def test_trees_get_no_scaling():
    """Trees and ensembles: scaling is not applied."""
    for algo in [
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "adaboost",
        "xgboosting",
    ]:
        preset = resolve_preprocessing_preset(algo)
        assert preset.algorithm_class == AlgorithmClass.TREES
        assert preset.scaling.type == "none"
        assert preset.imputation_strategy == "median"


def test_glm_skewed_gets_robust_preprocessing():
    """GLM with skewed distributions: median + robust."""
    for algo in ["poissonregressor", "gammaregressor", "tweedieregressor", "glm"]:
        preset = resolve_preprocessing_preset(algo)
        assert preset.algorithm_class == AlgorithmClass.GLM_SKEWED
        assert preset.scaling.type == "robust"
        assert preset.imputation_strategy == "median"


def test_univariate_algorithms_get_no_scaling():
    """Univariate algorithms: no scaling (non-standard preprocessing)."""
    preset = resolve_preprocessing_preset("isotonic_regression")
    assert preset.algorithm_class == AlgorithmClass.UNIVARIATE
    assert preset.scaling.type == "none"
    assert preset.imputation_strategy == "median"


# ──────────────────────────────────────────────────────────────────────────
#  Missing-value indicators
# ──────────────────────────────────────────────────────────────────────────
def test_missing_indicators_disabled_for_univariate_algorithms():
    """Strictly univariate algorithms (isotonic) are incompatible with missing-value
    indicator columns (the input contract is exactly one feature)."""
    assert requires_missing_indicator("isotonic_regression") is False
    assert requires_missing_indicator("isotonic_regression") is False


def test_missing_indicators_enabled_for_all_other_algorithms():
    """Missing-value indicators are enabled for all other algorithms."""
    for algo in AVAILABLE_ALGORITHMS:
        if ALGORITHM_CLASS_MAPPING[normalize_algorithm(algo)] is AlgorithmClass.UNIVARIATE:
            continue
        assert requires_missing_indicator(algo) is True, algo


def test_missing_indicators_unknown_algorithm_rejected():
    """An unknown algorithm is rejected with the same explicit error as the preset."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        requires_missing_indicator("custom_model_2026")


def test_every_supported_algorithm_resolves_to_a_preset():
    """A correct preset is selected for every supported algorithm."""
    for algo in AVAILABLE_ALGORITHMS:
        preset = resolve_preprocessing_preset(algo)
        assert isinstance(preset, PreprocessingPreset)
        assert preset.imputation_strategy in ("mean", "median")
        assert preset.scaling.type in ("standard", "robust", "none")


def test_all_canonical_algorithms_registered_in_mapping():
    """Every canonical algorithm in the registry has a class in the mapping."""
    for algo in AVAILABLE_ALGORITHMS:
        assert normalize_algorithm(algo) in ALGORITHM_CLASS_MAPPING, algo


# ──────────────────────────────────────────────────────────────────────────
#  Same classes → same presets; different classes → different
# ──────────────────────────────────────────────────────────────────────────
def test_same_class_same_preset():
    assert resolve_preprocessing_preset("ridge") == resolve_preprocessing_preset(
        "lasso"
    )
    assert resolve_preprocessing_preset("random_forest") == resolve_preprocessing_preset(
        "gradient_boosting"
    )
    assert resolve_preprocessing_preset("poissonregressor") == (
        resolve_preprocessing_preset("tweedieregressor")
    )


def test_different_classes_different_presets():
    """Algorithms of different classes get different preprocessing."""
    trees = resolve_preprocessing_preset("random_forest")
    sensitive = resolve_preprocessing_preset("ridge")
    glm = resolve_preprocessing_preset("gammaregressor")

    assert trees.scaling != sensitive.scaling  # different specs
    assert glm.imputation_strategy == trees.imputation_strategy  # median for both
    assert glm.scaling != trees.scaling  # different specs
    assert glm.imputation_strategy != sensitive.imputation_strategy


# ──────────────────────────────────────────────────────────────────────────
#  Explicit preset override
# ──────────────────────────────────────────────────────────────────────────
def test_partial_override_keeps_automatic_values():
    """A partial override changes only the specified fields."""
    preset = resolve_preprocessing_preset("random_forest", {"scaling": "standard"})
    assert preset.algorithm_class == AlgorithmClass.TREES
    assert preset.scaling.type == "standard"  # overridden
    assert preset.imputation_strategy == "median"  # left as the automatic choice


def test_full_override_replaces_all_fields():
    preset = resolve_preprocessing_preset(
        "ridge",
        {"imputation_strategy": "median", "scaling": "none"},
    )
    assert preset.imputation_strategy == "median"
    assert preset.scaling.type == "none"


def test_override_with_model_instance():
    override = PreprocessingOverride(imputation_strategy="median")
    preset = resolve_preprocessing_preset("elasticnet", override)
    assert preset.imputation_strategy == "median"
    assert preset.scaling.type == "standard"


def test_override_invalid_value_rejected():
    with pytest.raises(ValidationError):
        PreprocessingOverride.model_validate({"scaling": "bogus"})

    with pytest.raises(ValidationError):
        PreprocessingOverride.model_validate({"imputation_strategy": 42})

    with pytest.raises(ValidationError):
        PreprocessingOverride.model_validate({"unknown_field": "x"})


def test_override_invalid_type_rejected():
    with pytest.raises(TypeError, match="preprocessing_override"):
        resolve_preprocessing_preset("ridge", override="standard")


# ──────────────────────────────────────────────────────────────────────────
#  Edge cases
# ──────────────────────────────────────────────────────────────────────────
def test_unknown_algorithm_rejected():
    """An unknown algorithm (typo) is rejected with an explicit error,
    instead of silently receiving the default preset."""
    with pytest.raises(ValueError, match="Unknown algorithm.*custom_model_2026"):
        resolve_preprocessing_preset("custom_model_2026")


def test_unknown_algorithm_error_lists_available():
    """The error message contains the list of available algorithms."""
    with pytest.raises(ValueError) as excinfo:
        resolve_preprocessing_preset("nonsense_algo")
    message = str(excinfo.value)
    assert "Available algorithms" in message
    assert "ridge" in message
    assert "random_forest" in message


def test_supported_but_unregistered_algorithm_uses_default(monkeypatch):
    """Edge case: a supported algorithm without an explicit
    registration in the mapping table receives the default preset."""
    from configurable_automl_engine import preprocessing_presets as pp

    mapping = dict(pp.ALGORITHM_CLASS_MAPPING)
    mapping.pop("adaboost", None)
    monkeypatch.setattr(pp, "ALGORITHM_CLASS_MAPPING", mapping)

    preset = resolve_preprocessing_preset("adaboost")
    assert preset.algorithm_class == AlgorithmClass.DEFAULT
    assert preset.imputation_strategy == "mean"
    assert preset.scaling.type == "standard"


@pytest.mark.parametrize("alias", ["rf", "xgboost", "knn", "isotonic"])
def test_short_aliases_are_rejected(alias):
    """Short aliases are not supported: they are rejected as unknown names."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        resolve_preprocessing_preset(alias)


def test_algorithm_name_is_case_insensitive():
    """The full algorithm name is matched case-insensitively."""
    assert resolve_preprocessing_preset("ISOTONIC_REGRESSION") == (
        resolve_preprocessing_preset("isotonic_regression")
    )


def test_preset_string_repr_for_logging():
    """The string representation of the preset contains all key fields."""
    text = str(resolve_preprocessing_preset("ridge"))
    assert "scale_sensitive" in text
    assert "imputation_strategy='mean'" in text
    assert "scaling='standard'" in text


def test_presets_are_immutable():
    """Each class has its own preset instance (no shared mutable state)."""
    classes = list(AlgorithmClass)
    assert len({id(PRESET_BY_CLASS[c]) for c in classes}) == len(classes)


# ──────────────────────────────────────────────────────────────────────────
#  ScalingSpec: the object form of `scaling` + the string shorthand
# ──────────────────────────────────────────────────────────────────────────
def test_preset_scaling_defaults_to_object_form():
    """The preset's scaling field is always the object form (ScalingSpec)."""
    assert resolve_preprocessing_preset("ridge").scaling == StandardScaling()
    assert resolve_preprocessing_preset("random_forest").scaling == NoScaling()
    assert resolve_preprocessing_preset("poissonregressor").scaling == (
        RobustScaling()
    )


def test_string_shorthand_normalized_to_object_form():
    """The string shorthand ('robust') is normalized to its object form."""
    assert coerce_scaling("robust") == RobustScaling()
    assert coerce_scaling("standard") == StandardScaling()
    assert coerce_scaling("none") == NoScaling()
    assert coerce_scaling({"type": "none"}) == NoScaling()
    assert coerce_scaling(NoScaling()) is not None
    # Already-object specs pass through unchanged.
    spec = RobustScaling(quantile_range=(10.0, 90.0))
    assert coerce_scaling(spec) == spec
    assert scaling_type("robust") == "robust"
    assert scaling_type(RobustScaling(quantile_range=(5.0, 95.0))) == "robust"
    assert scaling_type(StandardScaling()) == "standard"


def test_string_and_object_forms_produce_equal_presets():
    """The string and object forms produce identical presets."""
    from_str = resolve_preprocessing_preset("ridge", {"scaling": "robust"})
    from_obj = resolve_preprocessing_preset(
        "ridge", {"scaling": {"type": "robust", "quantile_range": [25.0, 75.0]}}
    )
    assert from_str == from_obj
    assert from_str.scaling == RobustScaling()


def test_robust_scaling_parameters_validated():
    """RobustScaling accepts custom parameters and defaults match sklearn."""
    spec = RobustScaling.model_validate(
        {"type": "robust", "quantile_range": [10, 90], "with_centering": False}
    )
    assert spec.quantile_range == (10.0, 90.0)
    assert spec.with_centering is False
    # A JSON roundtrip keeps the parameters.
    restored = RobustScaling.model_validate_json(spec.model_dump_json())
    assert restored == spec


@pytest.mark.parametrize(
    "quantile_range",
    [(90.0, 10.0), (75.0, 75.0), (-5.0, 50.0), (50.0, 150.0), (100.0, 101.0)],
)
def test_robust_scaling_invalid_quantile_range_rejected(quantile_range):
    """An invalid quantile_range (lo >= hi or outside [0, 100]) is rejected."""
    with pytest.raises(ValidationError, match="quantile_range"):
        RobustScaling.model_validate(
            {"type": "robust", "quantile_range": list(quantile_range)}
        )


def test_cross_type_parameter_rejected():
    """A parameter that belongs to another type is rejected (extra='forbid')."""
    with pytest.raises(ValidationError, match="quantile_range"):
        PreprocessingOverride.model_validate(
            {"scaling": {"type": "standard", "quantile_range": [10, 90]}}
        )
    with pytest.raises(ValidationError):
        PreprocessingOverride.model_validate(
            {"scaling": {"type": "none", "with_centering": True}}
        )


def test_unknown_scaling_type_rejected_in_override():
    """An unknown scaling type (string or object) is rejected with a clear error."""
    with pytest.raises(ValidationError):
        PreprocessingOverride.model_validate({"scaling": "quantile"})
    with pytest.raises(ValidationError):
        PreprocessingOverride.model_validate({"scaling": {"type": "quantile"}})


def test_object_override_with_robust_parameters_applied():
    """An object-form override with robust parameters reaches the preset."""
    preset = resolve_preprocessing_preset(
        "poissonregressor",
        {"scaling": {"type": "robust", "quantile_range": [10.0, 90.0]}},
    )
    assert preset.scaling == RobustScaling(quantile_range=(10.0, 90.0))


def test_preset_str_shows_scaling_parameters():
    """__str__ shows the scaling parameters (e.g. robust(q=(10.0, 90.0)))."""
    preset = resolve_preprocessing_preset(
        "poissonregressor",
        {"scaling": {"type": "robust", "quantile_range": [10.0, 90.0]}},
    )
    assert "robust(q=(10.0, 90.0))" in str(preset)
    # The default robust spec shows the default quantile range.
    assert "robust(q=(25.0, 75.0))" in str(resolve_preprocessing_preset("glm"))
    # String-typed specs stay plain.
    assert "scaling='standard'" in str(resolve_preprocessing_preset("ridge"))
    assert "scaling='none'" in str(resolve_preprocessing_preset("random_forest"))


def test_preset_str_shows_centering_false():
    """A non-default with_centering=False is visible in the log string."""
    preset = resolve_preprocessing_preset(
        "glm", {"scaling": {"type": "robust", "with_centering": False}}
    )
    assert "centering=False" in str(preset)


def test_scaling_spec_equivalence_after_dump_roundtrip():
    """A preset dump/validate roundtrip keeps the scaling spec (serialization)."""
    preset = resolve_preprocessing_preset(
        "glm", {"scaling": {"type": "robust", "quantile_range": [5.0, 95.0]}}
    )
    restored = PreprocessingPreset.model_validate(preset.model_dump(mode="json"))
    assert restored == preset
    assert restored.scaling == RobustScaling(quantile_range=(5.0, 95.0))


def test_scaling_specs_are_frozen():
    """The scaling spec models are frozen value objects.

    A module-level default (e.g. the ``build_preprocessor`` default) must not
    be mutable by a caller: attribute assignment is rejected.
    """
    for spec in (StandardScaling(), RobustScaling(), NoScaling()):
        with pytest.raises(ValidationError):
            spec.type = "none"
    with pytest.raises(ValidationError):
        RobustScaling().quantile_range = (10.0, 90.0)


def test_discriminator_union_gives_precise_errors():
    """The union is discriminated by 'type': errors are precise.

    A cross-type parameter is reported inside its own branch (no smart-union
    noise about every member), and an unknown string shorthand is reported as
    an invalid tag.
    """
    # Cross-type parameter: exactly the extra_forbidden error of the standard
    # branch (plus the harmless schema-only string alternative).
    with pytest.raises(ValidationError) as excinfo:
        PreprocessingOverride.model_validate(
            {"scaling": {"type": "standard", "quantile_range": [10, 90]}}
        )
    error_types = [e["type"] for e in excinfo.value.errors()]
    assert "extra_forbidden" in error_types
    assert not any(
        e["loc"][-1] == "quantile_range" and e["type"] == "missing"
        for e in excinfo.value.errors()
    )

    # Invalid quantile_range: the value_error of the robust branch.
    with pytest.raises(ValidationError) as excinfo:
        PreprocessingOverride.model_validate(
            {"scaling": {"type": "robust", "quantile_range": [90, 10]}}
        )
    assert "quantile_range must satisfy 0 <= low < high <= 100" in str(excinfo.value)


def test_coerce_scaling_error_is_validation_error():
    """coerce_scaling raises ValidationError (a ValueError subclass)."""
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        coerce_scaling("bogus")
    with pytest.raises(ValueError):
        coerce_scaling("bogus")
