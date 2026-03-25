import pytest
import logging
import re
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import (
    GeneralCfg, OversamplingCfg, SearchSpaceEntry, AlgoCfg, Config, read_config, HPOPhaseCfg, 
    
)
from configurable_automl_engine.common.hyperopt_defaults import (
    NumericSpace, FloatSpace, IntSpace,
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
    "algorithms": {"elasticnet": {"enable": True}},
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
    
    assert "Oversampling multiplier = 1 ➜ class balance will not change" in caplog.text

# --- Тесты для AlgoCfg ---
def test_algo_cfg_empty_paths():
    # Пустые пути модулей
    with pytest.raises(ValidationError, match=".*not a valid dotted path.*"):
        AlgoCfg(tuner="")
    
    with pytest.raises(ValidationError, match=".*not a valid dotted path.*"):
        AlgoCfg(trainer_module="")
# --- Тесты для корневого Config и API ---
def test_config_no_enabled_algorithms():
    # Тест валидатора _must_have_enabled
    algo_disabled = AlgoCfg(enable=False)
    with pytest.raises(ValidationError, match=".*At least one algorithm must be enabled.*"):
        Config(
            general=GeneralCfg(phases=[]),
            algorithms={"elasticnet": algo_disabled}
        )
def test_read_config_integration(tmp_path):
    # Тест функции read_config и корректной загрузки YAML
    yaml_content = """
    general:
      comparison_metric: "r2"
      phases:
        - name: "search"
          n_trials: 10
          action: "all_algorithms"
      validation_strategy: "k_fold"
      n_folds: 3
    algorithms:
      random_forest:
        enable: true
        hyperparameters:
          n_estimators: [[10, 50, 100], "categorical"]
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")
    
    config = read_config(config_file)
    assert config.general.n_folds == 3
    assert hasattr(config.algorithms, 'random_forest')
    assert getattr(config.algorithms, "random_forest").enable is True

# Тесты для (Успешная валидация GeneralCfg)
def test_general_cfg_valid_n_folds():
    """Успешное завершение валидатора _check_n_folds."""
    cfg = GeneralCfg(
        phases=[HPOPhaseCfg(name="test", n_trials=1)],
        validation_strategy=ValidationStrategy.k_fold,
        n_folds=3
    )
    assert cfg.n_folds == 3

# 3. Тесты для AlgoCfg._must_not_be_empty
def test_algo_cfg_empty_paths():
    """ Проверка на пустую строку в путях модулей."""
    with pytest.raises(ValidationError) as exc_info:
        AlgoCfg(tuner="", hyperparameters={})
    
    error_msg = str(exc_info.value)
    
    # Assert specific parts of the message
    assert "tuner" in error_msg
    assert "not a valid dotted path" in error_msg
    assert "Value error" in error_msg

# Дополнительный тест для логики n_folds (граничные условия)
def test_general_cfg_invalid_n_folds_kfold():
    """Покрывает ошибку валидации при n_folds < 2 для k_fold."""
    with pytest.raises(ValidationError, match="(?s).*n_folds must be ≥ 2.*"):
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
    
    with pytest.raises(ValidationError, match="`n_folds` must be at least 1"):
        GeneralCfg(
            phases=[HPOPhaseCfg(name="test", n_trials=1)],
            validation_strategy=strategy,
            n_folds=0  # Это активирует raise на строке 83
        )

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
    
    with pytest.raises(ValidationError, match=".*serialization_format='joblib'.*"):
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
            "xgboosting": {"enable": True}
        }
    }
    
    expected_msg = "Algorithm 'xgboosting' is enabled, but the package 'xgboost' is not installed"
    with pytest.raises(ValueError, match=expected_msg):
        Config.model_validate(data)

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
            "elasticnet": {"enable": True},
            "xgboosting": {"enable": False}
        }
    }
    # Имитируем отсутствие библиотеки xgboost в системе
    with patch("configurable_automl_engine.common.dependency_utils.is_installed", return_value=False):
        # Если continue работает, объект будет создан успешно.
        # Если continue не сработает, вылетит ValueError: "Алгоритм 'xgboosting' включён..."
        config = Config.model_validate(config_data)
    
    assert getattr(config.algorithms, "xgboosting").enable is False
    assert hasattr(config.algorithms, "xgboosting")

# 1. Тест для: if self.low > self.high: raise ValueError(...)
def test_numeric_space_range_validation():
    # Ошибка: low > high
    # Используем re.escape, чтобы скобки (10.0) не воспринимались как группа в regex
    expected_msg = re.escape("low (10.0) must be <= high (5.0)")
    
    with pytest.raises(ValidationError, match=expected_msg):
        NumericSpace(type="base", low=10.0, high=5.0)
    
    # Успех: low == high (допустимо)
    n = NumericSpace(type="base", low=5.0, high=5.0)
    assert n.low == 5.0
# 2. Тест для: if self.type == "float_log" and self.step is not None:
def test_float_log_step_forbidden():
    with pytest.raises(ValidationError, match="The 'step' parameter is not supported for 'float_log'"):
        FloatSpace(type="float_log", low=1.0, high=10.0, step=0.1)
# 3. Тест для: if self.step is not None and self.step <= 0: (в FloatSpace и IntSpace)
def test_step_positive_validation():
    # Для FloatSpace
    with pytest.raises(ValidationError, match="Step must be positive. Got -1.0"):
        FloatSpace(type="float", low=0.0, high=1.0, step=-1.0)
    
    # Для IntSpace
    with pytest.raises(ValidationError, match="Step must be positive. Got 0"):
        IntSpace(type="int", low=1, high=10, step=0)
# 4. Тест для: _parse_list_to_dict (валидация различных форматов списков)
def test_parse_list_to_dict_logic():
    # Проверка len(data) >= 3 и payload["step"] = data[3]
    raw_data = [1, 10, "int", 2]
    entry = SearchSpaceEntry.model_validate(raw_data)
    assert entry.dist_type == "int"
    assert entry.step == 2
    
    # Проверка случая, если передали не список (должен вернуть как есть)
    # Pydantic выбросит ошибку валидации позже, если это не словарь, 
    # но сам метод _parse_list_to_dict должен пропустить данные.
    not_a_list = {"config": {"type": "int", "low": 1, "high": 5}}
    entry_from_dict = SearchSpaceEntry.model_validate(not_a_list)
    assert entry_from_dict.low == 1
# 5. Тест для: property bounds (формирование списка из объекта)
def test_search_space_bounds_property():
    # Для числового с шагом
    entry_int = SearchSpaceEntry.model_validate([1, 10, "int", 2])
    assert entry_int.bounds == [1, 10, "int", 2]
    
    # Для категориального
    cat_data = [["a", "b"], "categorical"]
    entry_cat = SearchSpaceEntry.model_validate(cat_data)
    assert entry_cat.bounds == [["a", "b"], "categorical"]
# 6. Тест для: _check_algorithm_dependencies (проверка установленных пакетов)
def test_algorithm_dependency_check():
    # Мокаем маппинг и функцию проверки установки
    with patch("configurable_automl_engine.training_engine.config_parser.ALGO_PACKAGE_MAPPING", {"xgboosting": "xgboost_pkg"}), \
         patch("configurable_automl_engine.training_engine.config_parser.is_installed") as mock_installed:
        
        # Ситуация: пакет НЕ установлен
        mock_installed.return_value = False
        
        config_data = {
            "general": {
                "phases": [{"name": "p1", "n_trials": 1}],
                "validation_strategy": "k_fold",
                "n_folds": 2
            },
            "algorithms": {
                "xgboosting": {"enable": True}
            }
        }
        
        expected_msg = "Algorithm 'xgboosting' is enabled, but the package 'xgboost_pkg' is not installed"
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            Config.model_validate(config_data)
        # Ситуация: пакет установлен
        mock_installed.return_value = True
        cfg = Config.model_validate(config_data)
        assert getattr(cfg.algorithms, "xgboosting").enable is True
# 7. Дополнительный тест на n_folds (общая логика GeneralCfg)
def test_general_cfg_n_folds():
    base_phases = [{"name": "test", "n_trials": 1}]
    
    # Ошибка: n_folds < 1
    with pytest.raises(ValidationError, match="`n_folds` must be at least 1"):
        GeneralCfg(phases=base_phases, n_folds=0)
        
    # Ошибка: k_fold требует n_folds >= 2
    with pytest.raises(ValidationError, match=".*n_folds must be ≥ 2 for k-fold.*"):
        GeneralCfg(
            phases=base_phases, 
            validation_strategy="k_fold", 
            n_folds=1
        )

def test_get_unknown_hyperparameters_none():
    cfg = AlgoCfg(hyperparameters=None)
    assert cfg.get_unknown_hyperparameters("xgboosting") == []


def test_get_unknown_hyperparameters_empty_allowed(monkeypatch):
    cfg = AlgoCfg(
        hyperparameters={"a": [1, 10]}  # ✅ как в YAML
    )

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {"xgboost": set()}
    )

    assert cfg.get_unknown_hyperparameters("xgboost") == []


def test_get_unknown_hyperparameters_valid(monkeypatch):
    cfg = AlgoCfg(
        hyperparameters={"lr": [0.0, 1.0]}  # ✅
    )

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {"xgboost": {"lr"}}
    )

    assert cfg.get_unknown_hyperparameters("xgboost") == []


def test_get_unknown_hyperparameters_unknown(monkeypatch):
    cfg = AlgoCfg(
        hyperparameters={"bad_param": [1, 10]}  # ✅
    )

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {"xgboost": {"lr"}}
    )

    assert cfg.get_unknown_hyperparameters("xgboost") == ["bad_param"]


def test_validator_allows_none():
    cfg = AlgoCfg(tuner=None, trainer_module=None)
    assert cfg.tuner is None
    assert cfg.trainer_module is None


def test_validator_valid_path():
    cfg = AlgoCfg(tuner="a.b", trainer_module="x.y.z")
    assert cfg.tuner == "a.b"


def test_validator_invalid_path():
    with pytest.raises(ValueError):
        AlgoCfg(tuner="invalid-path")

def test_hyperparameter_compatibility_error(monkeypatch):
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {algo_name: {"lr"}}
    )

    cfg_data = {
        "general": {"phases": [{"name": "p1", "n_trials": 1}]},
        "algorithms": {
            algo_name: {
                "enable": True,
                "hyperparameters": {
                    "bad": [1, 10]
                }
            }
        }
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)