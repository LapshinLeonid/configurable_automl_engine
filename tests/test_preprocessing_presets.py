"""Unit tests for adaptive preprocessing presets.

Cover automatic selection of the preprocessing preset by algorithm class, the
mapping table, preset override by the user, and the behavior for
unregistered algorithms (edge case).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from configurable_automl_engine.models import AVAILABLE_ALGORITHMS
from configurable_automl_engine.preprocessing_presets import (
    ALGORITHM_CLASS_MAPPING,
    PRESET_BY_CLASS,
    AlgorithmClass,
    PreprocessingOverride,
    PreprocessingPreset,
    normalize_algorithm,
    requires_missing_indicator,
    resolve_preprocessing_preset,
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
        assert preset.scaling == "standard"
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
        assert preset.scaling == "none"
        assert preset.imputation_strategy == "median"


def test_glm_skewed_gets_robust_preprocessing():
    """GLM with skewed distributions: median + robust."""
    for algo in ["poissonregressor", "gammaregressor", "tweedieregressor", "glm"]:
        preset = resolve_preprocessing_preset(algo)
        assert preset.algorithm_class == AlgorithmClass.GLM_SKEWED
        assert preset.scaling == "robust"
        assert preset.imputation_strategy == "median"


def test_univariate_algorithms_get_no_scaling():
    """Univariate algorithms: no scaling (non-standard preprocessing)."""
    preset = resolve_preprocessing_preset("isotonic_regression")
    assert preset.algorithm_class == AlgorithmClass.UNIVARIATE
    assert preset.scaling == "none"
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
        assert preset.scaling in ("standard", "robust", "none")


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

    assert trees.scaling != sensitive.scaling
    assert glm.imputation_strategy == trees.imputation_strategy  # median for both
    assert glm.scaling != trees.scaling
    assert glm.imputation_strategy != sensitive.imputation_strategy


# ──────────────────────────────────────────────────────────────────────────
#  Explicit preset override
# ──────────────────────────────────────────────────────────────────────────
def test_partial_override_keeps_automatic_values():
    """A partial override changes only the specified fields."""
    preset = resolve_preprocessing_preset("random_forest", {"scaling": "standard"})
    assert preset.algorithm_class == AlgorithmClass.TREES
    assert preset.scaling == "standard"  # overridden
    assert preset.imputation_strategy == "median"  # left as the automatic choice


def test_full_override_replaces_all_fields():
    preset = resolve_preprocessing_preset(
        "ridge",
        {"imputation_strategy": "median", "scaling": "none"},
    )
    assert preset.imputation_strategy == "median"
    assert preset.scaling == "none"


def test_override_with_model_instance():
    override = PreprocessingOverride(imputation_strategy="median")
    preset = resolve_preprocessing_preset("elasticnet", override)
    assert preset.imputation_strategy == "median"
    assert preset.scaling == "standard"


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
    assert preset.scaling == "standard"


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
