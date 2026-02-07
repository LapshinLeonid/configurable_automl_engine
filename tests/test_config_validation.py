import pytest
import logging
from pathlib import Path
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import (
    GeneralCfg, OversamplingCfg, SearchSpaceEntry, AlgoCfg, Config, read_config, HPOPhaseCfg
)
from configurable_automl_engine.common.definitions import ValidationStrategy, SerializationFormat

from unittest.mock import patch

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

# --- Тесты для OversamplingCfg ---
def test_oversampling_warn_useless_multiplier(caplog):
    # Предупреждение при multiplier=1 и enable=True
    with caplog.at_level(logging.WARNING):
        OversamplingCfg(enable=True, multiplier=1.0)
    
    assert "Oversampling multiplier = 1 ➜ баланс классов не изменится" in caplog.text

# --- Тесты для AlgoCfg ---
def test_algo_cfg_empty_paths():
    # Пустые пути модулей
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
    """Успешное завершение валидатора _check_n_folds."""
    cfg = GeneralCfg(
        phases=[HPOPhaseCfg(name="test", n_trials=1)],
        validation_strategy=ValidationStrategy.k_fold,
        n_folds=3
    )
    assert cfg.n_folds == 3

# Тесты для SearchSpaceEntry
def test_search_space_unknown_type():
    """Неизвестный тип распределения."""
    # Валидатор возвращает self, если тип не в ['int', 'float', 'float_log', 'categorical']
    entry = SearchSpaceEntry(bounds=[1, 10, "unknown_type"])
    assert entry.bounds[-1] == "unknown_type"

def test_search_space_categorical_invalid_structure():
    """Ошибка, если для categorical первый элемент не list."""
    with pytest.raises(ValueError, match="For 'categorical' type, the first element must be a list"):
        # Первый элемент "option" (str) валиден для Union, но невалиден для логики categorical
        SearchSpaceEntry(bounds=["option", "categorical"])

def test_search_space_numerical_with_list_bounds():
    """
    Ошибка, если в численном типе есть список.
    Используем model_construct, чтобы обойти предварительную проверку типов Pydantic.
    """
    # Создаем объект в обход валидации типов Union
    invalid_entry = SearchSpaceEntry.model_construct(
        bounds=[[1, 2], 10, "int"]
    )

    # Вручную вызываем валидатор, который теперь сработает и выбросит ValueError
    with pytest.raises(ValueError, match="Bounds for 'int' must be numerical"):
        invalid_entry._validate_structure()

# 3. Тесты для AlgoCfg._must_not_be_empty
def test_algo_cfg_empty_paths():
    """ Проверка на пустую строку в путях модулей."""
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

# Вызов исключения ValueError
def test_general_cfg_coverage_line_83():
    """
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

# Успешный возврат return self в SearchSpaceEntry
def test_search_space_coverage_line_173():
    """ Успешный проход валидатора распределения."""
    # Создаем корректную запись (например, для int), чтобы пройти все проверки
    # и достичь финального return self на строке 173
    entry = SearchSpaceEntry(bounds=[1, 10, "int"])
    assert entry.bounds[2] == "int"
    # Вызов метода напрямую для гарантии покрытия, если pydantic v2 оптимизирует вызовы
    result = entry._validate_structure()
    assert result == entry
# Успешный возврат return v в AlgoCfg
def test_algo_cfg_coverage_line_205():
    """Успешный возврат значения пути модуля."""
    # При создании корректного AlgoCfg, валидатор _must_not_be_empty 
    # должен вернуть значение v (строка 205)
    algo = AlgoCfg(
        tuner="path.to.tuner",
        trainer_module="path.to.trainer"
    )
    assert algo.tuner == "path.to.tuner"
    assert algo.trainer_module == "path.to.trainer"

@patch("configurable_automl_engine.training_engine.config_parser.is_installed")
def test_joblib_not_installed_raises_error(mock_is_installed):
    """Тест исключения при отсутствии joblib."""
    # Имитируем отсутствие пакета
    mock_is_installed.return_value = False
    
    data = {
        "general": {
            "phases": [{"name": "search", "n_trials": 10}],
            "serialization_format": "joblib",
            "validation_strategy": "k_fold",
            "n_folds": 5
        },
        "algorithms": {
            "any_algo": {"enable": True}
        }
    }
    
    with pytest.raises(ValueError, match="serialization_format='joblib' требует установленный пакет 'joblib'"):
        Config.model_validate(data)
@patch("configurable_automl_engine.training_engine.config_parser.is_installed")
def test_missing_algorithm_dependency_raises_error(mock_is_installed):
    """Тест исключения при отсутствии библиотеки алгоритма."""
    mock_is_installed.return_value = False
    
    data = {
        "general": {
            "phases": [{"name": "search", "n_trials": 10}],
            "validation_strategy": "train_test_split" 
        },
        "algorithms": {
            "xgboost": {"enable": True}
        }
    }
    
    expected_msg = "Алгоритм 'xgboost' включён, но пакет 'xgboost' не установлен"
    with pytest.raises(ValueError, match=expected_msg):
        Config.model_validate(data)

def test_search_space_entry_default_float_dist():
    """
    Тест проверяет ветку кода, когда len(bounds) == 2 
    и dist_type должен вернуть "float" по умолчанию.
    """
    
    # Создаем данные, где ровно 2 элемента и нет ключевого слова "categorical"
    # Это заставит dist_type вернуть "float" (строка 179)
    raw_data = {
        "bounds": [0.0, 10.0]
    }
    
    entry = SearchSpaceEntry(**raw_data)
    
    # 1. Проверяем, что dist_type определился как "float"
    assert entry.dist_type == "float"
    
    # 2. Проверяем, что свойства low и high работают корректно
    assert entry.low == 0.0
    assert entry.high == 10.0
    
    # 3. Проверяем, что step при этом None (так как элементов всего 2)
    assert entry.step is None
def test_search_space_entry_explicit_vs_implicit_float():
    """
    Сравнение явного указания типа и неявного (по умолчанию).
    """
    # Явное указание (срабатывает ветка elif len(self.bounds) >= 3)
    explicit_entry = SearchSpaceEntry(bounds=[0, 1, "float"])
    
    # Неявное указание (срабатывает целевая строка 179: return "float")
    implicit_entry = SearchSpaceEntry(bounds=[0, 1])
    
    assert explicit_entry.dist_type == "float"
    assert implicit_entry.dist_type == "float"
    assert explicit_entry.dist_type == implicit_entry.dist_type
def test_invalid_numerical_bounds_for_default_float():
    """
    Проверка валидации, если мы попали в "float" по умолчанию, 
    но передали не числа.
    """
    with pytest.raises(ValidationError) as excinfo:
        # Тип по умолчанию "float", но границы - строки
        SearchSpaceEntry(bounds=["min_val", "max_val"])
    
    assert "Bounds for 'float' must be numerical" in str(excinfo.value)

def test_search_space_step_return_value():
    """
    Тест проверяет корректную работу свойства step при наличии 4 элементов.
    Покрывает строку: return float(self.bounds[3])
    """
    # 1. Тест для распределения 'int' с шагом
    # Формат: [min, max, type, step]
    entry_int = SearchSpaceEntry(bounds=[1, 10, "int", 2])
    
    assert len(entry_int.bounds) == 4
    assert entry_int.step == 2.0  # float(2)
    assert isinstance(entry_int.step, float)
    # 2. Тест для распределения 'float' с шагом
    entry_float = SearchSpaceEntry(bounds=[0.0, 1.0, "float", 0.1])
    
    assert entry_float.step == 0.1
    assert entry_float.dist_type == "float"

def test_search_space_bounds_order_validation():
    """
    Проверка выбрасывания ошибки, если нижняя граница больше верхней.
    Покрывает: raise ValueError(f"Lower bound ({self.low}) must be less than or equal to...")
    """
    # Задаем границы, где low (10) > high (5)
    invalid_data = {"bounds": [10, 5, "int"]}
    
    with pytest.raises(ValidationError) as excinfo:
        SearchSpaceEntry(**invalid_data)
    
    # Проверяем наличие специфического сообщения об ошибке из кода
    assert "Lower bound (10) must be less than or equal to upper bound (5)" in str(excinfo.value)

def test_step_forbidden_for_float_log():
    """
    1. Покрывает: raise ValueError("The 'step' parameter is not supported for 'float_log'...")
    """
    # float_log не поддерживает 4-й элемент (step)
    invalid_bounds = [1, 100, "float_log", 1.0]
    
    with pytest.raises(ValidationError, match="The 'step' parameter is not supported for 'float_log'"):
        SearchSpaceEntry(bounds=invalid_bounds)
def test_step_must_be_integer_for_int_dist():
    """
    2. Покрывает: raise ValueError(f"Step for 'int' distribution must be an integer. Got {self.step}.")
    """
    # Для типа 'int' шаг 1.5 недопустим
    invalid_bounds = [1, 10, "int", 1.5]
    
    with pytest.raises(ValidationError, match="Step for 'int' distribution must be an integer"):
        SearchSpaceEntry(bounds=invalid_bounds)
def test_step_must_be_positive_for_int_dist():
    """
    3. Покрывает: raise ValueError(f"Step for 'int' distribution must be positive. Got {self.step}.")
    """
    # Для типа 'int' шаг 0 или отрицательный недопустим
    invalid_bounds = [1, 10, "int", 0]
    
    with pytest.raises(ValidationError, match="Step for 'int' distribution must be positive"):
        SearchSpaceEntry(bounds=invalid_bounds)
def test_step_must_be_positive_for_float_dist():
    """
    4. Покрывает: raise ValueError(f"Step for 'float' distribution must be positive. Got {self.step}.")
    """
    # Для типа 'float' шаг -0.1 недопустим
    invalid_bounds = [0.0, 1.0, "float", -0.1]
    
    with pytest.raises(ValidationError, match="Step for 'float' distribution must be positive"):
        SearchSpaceEntry(bounds=invalid_bounds)

def test_config_skips_disabled_algorithms_dependency_check():
    """
    Тест проверяет, что валидатор зависимостей игнорирует выключенные алгоритмы.
    Покрывает строку: if not algo_cfg.enable: continue
    """
    
    # Данные конфигурации: 
    # 1. 'some_custom_algo' включен (чтобы пройти валидатор _must_have_enabled)
    # 2. 'xgboost' выключен. Даже если библиотеки xgboost нет в системе, 
    #    ошибка не должна возникнуть благодаря 'continue'.
    config_data = {
        "general": {
            "phases": [{"name": "test", "n_trials": 1}]
        },
        "algorithms": {
            "some_custom_algo": {"enable": True},
            "xgboost": {"enable": False}
        }
    }
    # Имитируем отсутствие библиотеки xgboost в системе
    with patch("configurable_automl_engine.common.dependency_utils.is_installed", return_value=False):
        # Если continue работает, объект будет создан успешно.
        # Если continue не сработает, вылетит ValueError: "Алгоритм 'xgboost' включён..."
        config = Config(**config_data)
    
    assert config.algorithms["xgboost"].enable is False
    assert "xgboost" in config.algorithms