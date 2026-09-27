"""Unit tests for adaptive preprocessing presets (issue #18).

Покрывают автоматический выбор пресета предобработки по классу алгоритма,
таблицу соответствия, переопределение пресета пользователем (FR-5) и
поведение для незарегистрированных алгоритмов (граничный случай).
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
    resolve_preprocessing_preset,
)


# ──────────────────────────────────────────────────────────────────────────
#  FR-1 / AC-1: автоматический выбор корректного пресета по алгоритму
# ──────────────────────────────────────────────────────────────────────────
def test_scale_sensitive_models_get_standard_scaling():
    """Масштабо-чувствительные модели: scaling='standard', imputation='mean'."""
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
    """Деревья и ансамбли: масштабирование не применяется (AC-3)."""
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
    """GLM со скошенными распределениями: median + robust (AC-5)."""
    for algo in ["poissonregressor", "gammaregressor", "tweedieregressor", "glm"]:
        preset = resolve_preprocessing_preset(algo)
        assert preset.algorithm_class == AlgorithmClass.GLM_SKEWED
        assert preset.scaling == "robust"
        assert preset.imputation_strategy == "median"


def test_univariate_algorithms_get_no_scaling():
    """Одномерные алгоритмы: без масштабирования (нестандартная предобработка)."""
    preset = resolve_preprocessing_preset("isotonic_regression")
    assert preset.algorithm_class == AlgorithmClass.UNIVARIATE
    assert preset.scaling == "none"
    assert preset.imputation_strategy == "median"


def test_every_supported_algorithm_resolves_to_a_preset():
    """Для каждого поддерживаемого алгоритма выбирается корректный пресет (AC-1)."""
    for algo in AVAILABLE_ALGORITHMS:
        preset = resolve_preprocessing_preset(algo)
        assert isinstance(preset, PreprocessingPreset)
        assert preset.imputation_strategy in ("mean", "median")
        assert preset.scaling in ("standard", "robust", "none")


def test_all_canonical_algorithms_registered_in_mapping():
    """Каждый канонический алгоритм из реестра имеет класс в таблице (FR-2)."""
    for algo in AVAILABLE_ALGORITHMS:
        assert normalize_algorithm(algo) in ALGORITHM_CLASS_MAPPING, algo


# ──────────────────────────────────────────────────────────────────────────
#  AC-2: одинаковые классы → одинаковые пресеты; разные классы → разные
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
    """Алгоритмы разных классов получают различную предобработку (AC-2)."""
    trees = resolve_preprocessing_preset("random_forest")
    sensitive = resolve_preprocessing_preset("ridge")
    glm = resolve_preprocessing_preset("gammaregressor")

    assert trees.scaling != sensitive.scaling
    assert glm.imputation_strategy == trees.imputation_strategy  # median у обоих
    assert glm.scaling != trees.scaling
    assert glm.imputation_strategy != sensitive.imputation_strategy


# ──────────────────────────────────────────────────────────────────────────
#  FR-5 / AC-7: явное переопределение пресета
# ──────────────────────────────────────────────────────────────────────────
def test_partial_override_keeps_automatic_values():
    """Частичное переопределение меняет только указанные поля."""
    preset = resolve_preprocessing_preset("rf", {"scaling": "standard"})
    assert preset.algorithm_class == AlgorithmClass.TREES
    assert preset.scaling == "standard"  # переопределено
    assert preset.imputation_strategy == "median"  # осталось автовыбором


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
#  Граничные случаи
# ──────────────────────────────────────────────────────────────────────────
def test_unknown_algorithm_rejected():
    """Неизвестный алгоритм (опечатка) отклоняется явной ошибкой,
    а не молча получает default-пресет."""
    with pytest.raises(ValueError, match="Unknown algorithm.*custom_model_2026"):
        resolve_preprocessing_preset("custom_model_2026")


def test_unknown_algorithm_error_lists_available():
    """Сообщение об ошибке содержит список доступных алгоритмов."""
    with pytest.raises(ValueError) as excinfo:
        resolve_preprocessing_preset("nonsense_algo")
    message = str(excinfo.value)
    assert "Available algorithms" in message
    assert "ridge" in message
    assert "random_forest" in message


def test_supported_but_unregistered_algorithm_uses_default(monkeypatch):
    """Граничный случай issue #18: поддерживаемый алгоритм без явной
    регистрации в таблице соответствия получает пресет по умолчанию."""
    from configurable_automl_engine import preprocessing_presets as pp

    mapping = dict(pp.ALGORITHM_CLASS_MAPPING)
    mapping.pop("adaboost", None)
    monkeypatch.setattr(pp, "ALGORITHM_CLASS_MAPPING", mapping)

    preset = resolve_preprocessing_preset("adaboost")
    assert preset.algorithm_class == AlgorithmClass.DEFAULT
    assert preset.imputation_strategy == "mean"
    assert preset.scaling == "standard"


def test_aliases_are_resolved():
    """Алиасы алгоритмов раскрываются в канонические имена."""
    assert resolve_preprocessing_preset("rf") == resolve_preprocessing_preset(
        "random_forest"
    )
    assert resolve_preprocessing_preset("xgboost") == resolve_preprocessing_preset(
        "xgboosting"
    )
    assert resolve_preprocessing_preset("knn") == (
        resolve_preprocessing_preset("nearest_neighbors_regression")
    )
    assert resolve_preprocessing_preset("ISOTONIC") == (
        resolve_preprocessing_preset("isotonic_regression")
    )


def test_preset_string_repr_for_logging():
    """Строковое представление пресета содержит все ключевые поля (AC-9)."""
    text = str(resolve_preprocessing_preset("ridge"))
    assert "scale_sensitive" in text
    assert "imputation_strategy='mean'" in text
    assert "scaling='standard'" in text


def test_presets_are_immutable():
    """Каждый класс имеет собственный экземпляр пресета (без общего мутабельного состояния)."""
    classes = list(AlgorithmClass)
    assert len({id(PRESET_BY_CLASS[c]) for c in classes}) == len(classes)
