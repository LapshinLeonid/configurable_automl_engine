import pytest
import logging
from pathlib import Path
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import (
    GeneralCfg, OversamplingCfg, SearchSpaceEntry, AlgoCfg, Config, read_config, HPOPhaseCfg
)
from configurable_automl_engine.common.definitions import ValidationStrategy, SerializationFormat


BASE = {
    "general": {
        "comparison_metric": "nrmse",
        "validation_strategy": "k_fold",
        "n_folds": 5,
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            {"name": "refine", "n_trials": 1, "action": "refine_winner"}
        ]
    },
    "oversampling": {},
    "algorithms": {"lgbm": {"enable": True}},
}

def test_n_folds_ok():
    cfg = Config.model_validate(BASE)
    assert cfg.general.n_folds == 5

def test_n_folds_bad():
    bad = BASE | {"general": {**BASE["general"], "n_folds": 1}}
    with pytest.raises(ValidationError):
        Config.model_validate(bad)

def test_n_folds_ignored_for_loo():
    loo = BASE | {"general": {**BASE["general"], "validation_strategy": "loo", "n_folds": 1}}
    cfg = Config.model_validate(loo)  # не должно падать
    assert cfg.general.validation_strategy == ValidationStrategy.loo

# --- Тесты для OversamplingCfg (Строка 130 - логирование) ---
def test_oversampling_warn_useless_multiplier(caplog):
    # Покрытие строки 130: предупреждение при multiplier=1 и enable=True
    with caplog.at_level(logging.WARNING):
        OversamplingCfg(enable=True, multiplier=1.0)
    
    assert "Oversampling multiplier = 1 ➜ баланс классов не изменится" in caplog.text

# --- Тесты для AlgoCfg (Строки 203-205) ---
def test_algo_cfg_empty_paths():
    # Покрытие строк 203-205: пустые пути модулей
    with pytest.raises(ValidationError, match="module path must be non-empty"):
        AlgoCfg(tuner="")
    
    with pytest.raises(ValidationError, match="module path must be non-empty"):
        AlgoCfg(trainer_module="")
# --- Тесты для корневого Config и API ---
def test_config_no_enabled_algorithms():
    # Тест валидатора _must_have_enabled
    algo_disabled = AlgoCfg(enable=False)
    with pytest.raises(ValidationError, match="no algorithms enabled in config"):
        Config(
            general=GeneralCfg(phases=[]),
            algorithms={"test_algo": algo_disabled}
        )
def test_read_config_integration(tmp_path):
    # Тест функции read_config и корректной загрузки YAML
    yaml_content = """
    general:
      comparison_metric: "r2"
      phases:
        - name: "search"
          n_trials: 10
      validation_strategy: "k_fold"
      n_folds: 3
    algorithms:
      rf:
        enable: true
        hyperparameters:
          n_estimators: [[10, 50, 100], "categorical"]
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")
    
    config = read_config(config_file)
    assert config.general.n_folds == 3
    assert "rf" in config.algorithms
    assert config.algorithms["rf"].enable is True

# Тесты для (Успешная валидация GeneralCfg)
def test_general_cfg_valid_n_folds():
    """Покрывает строку 83: успешное завершение валидатора _check_n_folds."""
    cfg = GeneralCfg(
        phases=[HPOPhaseCfg(name="test", n_trials=1)],
        validation_strategy=ValidationStrategy.k_fold,
        n_folds=3
    )
    assert cfg.n_folds == 3

# Тесты для строк 159-173 (SearchSpaceEntry)
def test_search_space_unknown_type():
    """Покрывает строки 159-160: неизвестный тип распределения."""
    # Валидатор возвращает self, если тип не в ['int', 'float', 'float_log', 'categorical']
    entry = SearchSpaceEntry(bounds=[1, 10, "unknown_type"])
    assert entry.bounds[-1] == "unknown_type"

def test_search_space_categorical_invalid_structure():
    """Покрывает строки 161-167: ошибка, если для categorical первый элемент не list."""
    with pytest.raises(ValueError, match="For 'categorical' type, the first element must be a list"):
        # Первый элемент "option" (str) валиден для Union, но невалиден для логики categorical
        SearchSpaceEntry(bounds=["option", "categorical"])

def test_search_space_numerical_with_list_bounds():
    """
    Покрывает строки 168-173: ошибка, если в численном типе есть список.
    Используем model_construct, чтобы обойти предварительную проверку типов Pydantic.
    """
    # Создаем объект в обход валидации типов Union
    invalid_entry = SearchSpaceEntry.model_construct(
        bounds=[[1, 2], 10, "int"]
    )

    # Вручную вызываем валидатор, который теперь сработает и выбросит ValueError
    with pytest.raises(ValueError, match="Numerical distribution 'int' cannot have a list"):
        invalid_entry._validate_structure()

# 3. Тесты для строки 205 (AlgoCfg._must_not_be_empty)
def test_algo_cfg_empty_paths():
    """Покрывает строку 205: проверка на пустую строку в путях модулей."""
    with pytest.raises(ValueError, match="module path must be non-empty"):
        AlgoCfg(tuner="", hyperparameters={})
    
    with pytest.raises(ValueError, match="module path must be non-empty"):
        AlgoCfg(trainer_module="", hyperparameters={})

# Дополнительный тест для логики n_folds (граничные условия)
def test_general_cfg_invalid_n_folds_kfold():
    """Покрывает ошибку валидации при n_folds < 2 для k_fold."""
    with pytest.raises(ValueError, match="n_folds` must be ≥ 2"):
        GeneralCfg(
            phases=[HPOPhaseCfg(name="test", n_trials=1)],
            validation_strategy=ValidationStrategy.k_fold,
            n_folds=1
        )

# Покрытие строки 83: Вызов исключения ValueError
def test_general_cfg_coverage_line_83():
    """
    Покрывает строку 83.
    Мы передаем n_folds=0, что должно вызвать первое исключение в _check_n_folds
    вне зависимости от выбранной стратегии валидации.
    """
    # Используем первый доступный элемент из Enum, чтобы избежать AttributeError
    strategy = list(ValidationStrategy)[0] 
    
    with pytest.raises(ValueError, match="`n_folds` must be at least 1"):
        GeneralCfg(
            phases=[HPOPhaseCfg(name="test", n_trials=1)],
            validation_strategy=strategy,
            n_folds=0  # Это активирует raise на строке 83
        )

# Покрытие строки 173: Успешный возврат return self в SearchSpaceEntry
def test_search_space_coverage_line_173():
    """Покрывает строку 173: успешный проход валидатора распределения."""
    # Создаем корректную запись (например, для int), чтобы пройти все проверки
    # и достичь финального return self на строке 173
    entry = SearchSpaceEntry(bounds=[1, 10, "int"])
    assert entry.bounds[2] == "int"
    # Вызов метода напрямую для гарантии покрытия, если pydantic v2 оптимизирует вызовы
    result = entry._validate_structure()
    assert result == entry
# Покрытие строки 205: Успешный возврат return v в AlgoCfg
def test_algo_cfg_coverage_line_205():
    """Покрывает строку 205: успешный возврат значения пути модуля."""
    # При создании корректного AlgoCfg, валидатор _must_not_be_empty 
    # должен вернуть значение v (строка 205)
    algo = AlgoCfg(
        tuner="path.to.tuner",
        trainer_module="path.to.trainer"
    )
    assert algo.tuner == "path.to.tuner"
    assert algo.trainer_module == "path.to.trainer"