import pytest
import logging
import re
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import (
    GeneralCfg,
    OversamplingCfg,
    SearchSpaceEntry,
    AlgoCfg,
    Config,
    read_config,
    HPOPhaseCfg,
    FeatureSelectionCfg,
    FeatureSelectionMode,
    FeatureSelectionMethod,
)
from configurable_automl_engine.common.hyperopt_defaults import (
    NumericSpace,
    FloatSpace,
    IntSpace,
)
from configurable_automl_engine.common.definitions import (
    ValidationStrategy,
    SerializationFormat,
)

from unittest.mock import patch

BASE = {
    "general": {
        "comparison_metric": "nrmse",
        "validation_strategy": "k_fold",
        "n_folds": 5,
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            {"name": "refine", "n_trials": 1, "action": "refine_winner"},
        ],
    },
    "algorithms": {"elasticnet": {"enable": True}},
}


def test_n_folds_ok():
    cfg = Config.model_validate(BASE)
    assert cfg.general.n_folds == 5


def test_categorical_encoding_ordinal_valid():
    """Конфиг с categorical_encoding='ordinal' проходит валидацию Config."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "categorical_encoding": "ordinal"}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.categorical_encoding == "ordinal"


def test_categorical_encoding_default_one_hot():
    """По умолчанию categorical_encoding='one_hot'."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.categorical_encoding == "one_hot"


def test_categorical_encoding_invalid_rejected():
    """Конфиг с categorical_encoding='binary' отклоняется (Literal)."""
    cfg_data = BASE | {"general": {**BASE["general"], "categorical_encoding": "binary"}}
    with pytest.raises(ValidationError):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize(
    "enc", ["one_hot", "ordinal", "target", "frequency", "hashing"]
)
def test_categorical_encoding_new_strategies_valid(enc):
    """Все поддерживаемые стратегии (включая новые) проходят валидацию."""
    cfg_data = BASE | {"general": {**BASE["general"], "categorical_encoding": enc}}
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.categorical_encoding == enc


def test_high_cardinality_encoding_pair_valid():
    """Согласованная пара threshold + high_cardinality_encoding проходит."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "categorical_encoding": "one_hot",
            "high_cardinality_threshold": 20,
            "high_cardinality_encoding": "target",
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.high_cardinality_threshold == 20
    assert cfg.general.high_cardinality_encoding == "target"


def test_high_cardinality_threshold_zero_allowed():
    """Порог, равный нулю, допустим (все непустые колонки — high-cardinality)."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": 0,
            "high_cardinality_encoding": "hashing",
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.high_cardinality_threshold == 0


def test_high_cardinality_threshold_negative_rejected():
    """Отрицательный порог отклоняется на этапе валидации конфигурации."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": -1,
            "high_cardinality_encoding": "target",
        }
    }
    with pytest.raises(ValidationError, match="high_cardinality_threshold"):
        Config.model_validate(cfg_data)


def test_high_cardinality_encoding_unsupported_rejected():
    """Неподдерживаемая HC-стратегия отклоняется."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": 10,
            "high_cardinality_encoding": "binary",
        }
    }
    with pytest.raises(ValidationError):
        Config.model_validate(cfg_data)


def test_high_cardinality_pair_must_be_set_together():
    """Задан только один из пары параметров -> понятная ошибка."""
    only_threshold = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_threshold": 10,
        }
    }
    with pytest.raises(ValidationError, match="must be set together"):
        Config.model_validate(only_threshold)

    only_encoding = BASE | {
        "general": {
            **BASE["general"],
            "high_cardinality_encoding": "target",
        }
    }
    with pytest.raises(ValidationError, match="must be set together"):
        Config.model_validate(only_encoding)


def test_hashing_n_components_zero_rejected():
    """hashing_n_components=0 отклоняется."""
    cfg_data = BASE | {"general": {**BASE["general"], "hashing_n_components": 0}}
    with pytest.raises(ValidationError, match="hashing_n_components"):
        Config.model_validate(cfg_data)


def test_target_encoding_smoothing_negative_rejected():
    """Отрицательное сглаживание target encoding отклоняется."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "target_encoding_smoothing": -0.5}
    }
    with pytest.raises(ValidationError, match="target_encoding_smoothing"):
        Config.model_validate(cfg_data)


def test_target_encoding_fallback_any_float():
    """target_encoding_fallback принимает float или None."""
    for fallback in (0.0, -5.5, 42.0):
        cfg_data = BASE | {
            "general": {**BASE["general"], "target_encoding_fallback": fallback}
        }
        cfg = Config.model_validate(cfg_data)
        assert cfg.general.target_encoding_fallback == fallback


def test_new_params_absent_backward_compat():
    """Конфигурация без новых параметров работает как раньше (обратная совместимость)."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.categorical_encoding == "one_hot"
    assert cfg.general.high_cardinality_threshold is None
    assert cfg.general.high_cardinality_encoding is None
    assert cfg.general.hashing_n_components == 16
    assert cfg.general.target_encoding_smoothing == 20.0
    assert cfg.general.target_encoding_fallback is None


def test_n_folds_bad():
    bad = BASE | {"general": {**BASE["general"], "n_folds": 1}}
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_n_folds_ignored_for_loo():
    loo = BASE | {
        "general": {**BASE["general"], "validation_strategy": "loo", "n_folds": 1}
    }
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
    with pytest.raises(
        ValidationError, match=".*At least one algorithm must be enabled.*"
    ):
        Config(general=GeneralCfg(phases=[]), algorithms={"elasticnet": algo_disabled})


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
    assert hasattr(config.algorithms, "random_forest")
    assert getattr(config.algorithms, "random_forest").enable is True


# Тесты для (Успешная валидация GeneralCfg)
def test_general_cfg_valid_n_folds():
    """Успешное завершение валидатора _check_n_folds."""
    cfg = GeneralCfg(
        phases=[HPOPhaseCfg(name="test", n_trials=1)],
        validation_strategy=ValidationStrategy.k_fold,
        n_folds=3,
    )
    assert cfg.n_folds == 3


# 3. Тесты для AlgoCfg._must_not_be_empty
def test_algo_cfg_empty_paths():
    """Проверка на пустую строку в путях модулей."""
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
            n_folds=1,
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
            n_folds=0,  # Это активирует raise на строке 83
        )


# Успешный возврат return v в AlgoCfg
def test_algo_cfg_coverage_line_205():
    """Успешный возврат значения пути модуля."""
    # При создании корректного AlgoCfg, валидатор _must_not_be_empty
    # должен вернуть значение v (строка 205)
    algo = AlgoCfg(tuner="path.to.tuner", trainer_module="path.to.trainer")
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
            "n_folds": 5,
        },
        "algorithms": {"any_algo": {"enable": True}},
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
            "validation_strategy": "train_test_split",
        },
        "algorithms": {"xgboosting": {"enable": True}},
    }

    expected_msg = (
        "Algorithm 'xgboosting' is enabled, but the package 'xgboost' is not installed"
    )
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
        "general": {"phases": [{"name": "test", "n_trials": 1}]},
        "algorithms": {"elasticnet": {"enable": True}, "xgboosting": {"enable": False}},
    }
    # Имитируем отсутствие библиотеки xgboost в системе
    with patch(
        "configurable_automl_engine.common.dependency_utils.is_installed",
        return_value=False,
    ):
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
    with pytest.raises(
        ValidationError, match="The 'step' parameter is not supported for 'float_log'"
    ):
        FloatSpace(type="float_log", low=1.0, high=10.0, step=0.1)


# 2a. Тест для: float_log требует low > 0 (issue #20)
@pytest.mark.parametrize("bad_low", [0, -1, -0.0001])
def test_float_log_non_positive_low_rejected(bad_low):
    """float_log c low <= 0 отклоняется на этапе валидации.

    Параметризовано по значению lower-границы: покрывает как краткую
    списочную форму ``[low, high, \"float_log\"]`` через SearchSpaceEntry
    (в т.ч. с целочисленным ``0``), так и словарную форму через FloatSpace.
    """
    with pytest.raises(
        ValidationError, match="low must be > 0 for log-scale distributions"
    ):
        SearchSpaceEntry.model_validate([bad_low, 1, "float_log"])

    with pytest.raises(
        ValidationError, match="low must be > 0 for log-scale distributions"
    ):
        FloatSpace(type="float_log", low=bad_low, high=1.0)


def test_float_log_positive_low_valid():
    """Позитивный сценарий: float_log со строго положительным low проходит."""
    entry = SearchSpaceEntry.model_validate([1e-6, 1.0, "float_log"])
    assert entry.dist_type == "float_log"
    assert entry.low == 1e-6
    assert entry.high == 1.0
    assert entry.step is None


def test_float_log_non_positive_low_rejected_in_config():
    """Конфиг с float_log и low <= 0 отклоняется на этапе Config.model_validate."""
    cfg_data = BASE | {
        "algorithms": {
            "ridge": {
                "enable": True,
                "hyperparameters": {"alpha": [0.0, 1.0, "float_log"]},
            }
        }
    }
    with pytest.raises(
        ValidationError, match="low must be > 0 for log-scale distributions"
    ):
        Config.model_validate(cfg_data)


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
    with (
        patch(
            "configurable_automl_engine.training_engine.config_parser.ALGO_PACKAGE_MAPPING",
            {"xgboosting": "xgboost_pkg"},
        ),
        patch(
            "configurable_automl_engine.training_engine.config_parser.is_installed"
        ) as mock_installed,
    ):
        # Ситуация: пакет НЕ установлен
        mock_installed.return_value = False

        config_data = {
            "general": {
                "phases": [{"name": "p1", "n_trials": 1}],
                "validation_strategy": "k_fold",
                "n_folds": 2,
            },
            "algorithms": {"xgboosting": {"enable": True}},
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
        GeneralCfg(phases=base_phases, validation_strategy="k_fold", n_folds=1)


def test_get_unknown_hyperparameters_none():
    cfg = AlgoCfg(hyperparameters=None)
    assert cfg.get_unknown_hyperparameters("xgboosting") == []


def test_get_unknown_hyperparameters_empty_allowed(monkeypatch):
    cfg = AlgoCfg(
        hyperparameters={"a": [1, 10]}  # ✅ как в YAML
    )

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {"xgboost": set()},
    )

    assert cfg.get_unknown_hyperparameters("xgboost") == []


def test_get_unknown_hyperparameters_valid(monkeypatch):
    cfg = AlgoCfg(
        hyperparameters={"lr": [0.0, 1.0]}  # ✅
    )

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {"xgboost": {"lr"}},
    )

    assert cfg.get_unknown_hyperparameters("xgboost") == []


def test_get_unknown_hyperparameters_unknown(monkeypatch):
    cfg = AlgoCfg(
        hyperparameters={"bad_param": [1, 10]}  # ✅
    )

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {"xgboost": {"lr"}},
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


# --- Тесты для preprocessing override (FR-5) ---
def test_algo_cfg_preprocessing_valid_override():
    """Валидный блок переопределения пресета принимается конфигом."""
    cfg = AlgoCfg(preprocessing={"imputation_strategy": "median", "scaling": "none"})
    assert cfg.preprocessing is not None
    assert cfg.preprocessing.imputation_strategy == "median"
    assert cfg.preprocessing.scaling == "none"


def test_algo_cfg_preprocessing_partial_override():
    """Частичное переопределение: незаданные поля остаются None."""
    cfg = AlgoCfg(preprocessing={"scaling": "robust"})
    assert cfg.preprocessing.scaling == "robust"
    assert cfg.preprocessing.imputation_strategy is None


def test_algo_cfg_preprocessing_default_none():
    """Без блока preprocessing автовыбор (None)."""
    assert AlgoCfg().preprocessing is None


def test_algo_cfg_preprocessing_invalid_scaling_rejected():
    """Некорректное значение scaling отклоняется на этапе валидации
    с понятным сообщением (негативный сценарий)."""
    with pytest.raises(ValidationError, match="scaling"):
        AlgoCfg(preprocessing={"scaling": "quantile"})


def test_algo_cfg_preprocessing_invalid_imputation_rejected():
    with pytest.raises(ValidationError, match="imputation_strategy"):
        AlgoCfg(preprocessing={"imputation_strategy": "mode"})


def test_algo_cfg_preprocessing_unknown_field_rejected():
    """Неизвестные поля в блоке переопределения отклоняются (extra='forbid')."""
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        AlgoCfg(preprocessing={"unknown_field": 1})


def test_config_preprocessing_override_end_to_end():
    """Блок preprocessing в конфиге алгоритма доходит до Config без потерь."""
    cfg_data = BASE | {
        "algorithms": {
            "elasticnet": {
                "enable": True,
                "preprocessing": {"imputation_strategy": "median"},
            }
        }
    }
    cfg = Config.model_validate(cfg_data)
    algo_cfg = getattr(cfg.algorithms, "elasticnet")
    assert algo_cfg.preprocessing.imputation_strategy == "median"
    assert algo_cfg.preprocessing.scaling is None


def test_hyperparameter_compatibility_error(monkeypatch):
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {algo_name: {"lr"}},
    )

    cfg_data = {
        "general": {"phases": [{"name": "p1", "n_trials": 1}]},
        "algorithms": {
            algo_name: {"enable": True, "hyperparameters": {"bad": [1, 10]}}
        },
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)


def test_hyperparameter_compatibility_error_lists_allowed(monkeypatch):
    """Сообщение об ошибке содержит реальный список допустимых гиперпараметров.

    Регрессионный тест: ранее в текст подставлялся литеральный ``{allowed}``
    (результат ``sorted(...)`` отбрасывался), что делало сообщение бесполезным.
    """
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {algo_name: {"lr", "alpha"}},
    )

    cfg_data = BASE | {
        "algorithms": {algo_name: {"enable": True, "hyperparameters": {"bad": [1, 10]}}}
    }

    with pytest.raises(ValueError) as exc_info:
        Config.model_validate(cfg_data)

    msg = str(exc_info.value)
    assert "unknown hyperparameters ['bad']" in msg
    assert "Allowed parameters: ['alpha', 'lr']" in msg
    assert "{allowed}" not in msg


def test_hyperparameter_compatibility_valid_hyperparameters_pass(monkeypatch):
    """Совместимые гиперпараметры не приводят к ошибке валидации."""
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {algo_name: {"lr"}},
    )

    cfg_data = BASE | {
        "algorithms": {
            algo_name: {"enable": True, "hyperparameters": {"lr": [0.0, 1.0]}}
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, algo_name).enable is True


def test_hyperparameter_compatibility_skips_disabled(monkeypatch):
    """Выключенные алгоритмы пропускаются проверкой совместимости.

    algo_b присутствует в подменённом реестре и содержит недопустимый
    гиперпараметр: тест проходит только благодаря guard'у ``enable=False``.
    """
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_a, algo_b = AVAILABLE_ALGORITHMS[0], AVAILABLE_ALGORITHMS[1]

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {algo_a: {"lr"}, algo_b: {"alpha"}},
    )

    cfg_data = BASE | {
        "algorithms": {
            algo_a: {"enable": True},
            algo_b: {"enable": False, "hyperparameters": {"bad": [1, 10]}},
        }
    }

    cfg = Config.model_validate(cfg_data)  # не должно падать
    assert getattr(cfg.algorithms, algo_a).enable is True
    assert getattr(cfg.algorithms, algo_b).enable is False


def test_hyperparameter_compatibility_aggregates_errors(monkeypatch):
    """Ошибки для нескольких алгоритмов собираются в одно исключение."""
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_a, algo_b = AVAILABLE_ALGORITHMS[0], AVAILABLE_ALGORITHMS[1]

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.config_parser.ALGO_HYPERPARAMETER_REGISTRY",
        {algo_a: {"lr"}, algo_b: {"alpha"}},
    )

    cfg_data = BASE | {
        "algorithms": {
            algo_a: {"enable": True, "hyperparameters": {"bad": [1, 10]}},
            algo_b: {"enable": True, "hyperparameters": {"wrong": [0, 1]}},
        }
    }

    with pytest.raises(ValueError) as exc_info:
        Config.model_validate(cfg_data)

    msg = str(exc_info.value)
    assert f"Algorithm '{algo_a}': unknown hyperparameters ['bad']" in msg
    assert f"Algorithm '{algo_b}': unknown hyperparameters ['wrong']" in msg


# ─────────────────── Tests for PruningCfg (early stopping) ───────────────────
def test_pruning_disabled_by_default():
    """Ранняя остановка выключена по умолчанию: конфиг без блока работает как раньше."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.pruning.enable is False
    assert cfg.general.pruning.strategy.value == "median"
    assert cfg.general.pruning.min_steps == 1
    assert cfg.general.pruning.n_startup_trials == 5
    assert cfg.general.pruning.reduction_factor == 3


def test_pruning_valid_median_config():
    """Корректная конфигурация median-прайнера проходит валидацию."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 5,
            "pruning": {
                "enable": True,
                "strategy": "median",
                "min_steps": 2,
                "n_startup_trials": 3,
            },
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.pruning.enable is True
    assert cfg.general.pruning.strategy.value == "median"
    assert cfg.general.pruning.min_steps == 2
    assert cfg.general.pruning.n_startup_trials == 3


def test_pruning_valid_hyperband_config():
    """Корректная конфигурация hyperband-прайнера проходит валидацию."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 5,
            "pruning": {
                "enable": True,
                "strategy": "hyperband",
                "min_steps": 1,
                "reduction_factor": 2,
            },
        }
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.pruning.strategy.value == "hyperband"
    assert cfg.general.pruning.reduction_factor == 2


def test_pruning_unknown_strategy_rejected():
    """Неизвестная стратегия отклоняется на этапе валидации конфигурации."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "strategy": "unknown_strategy"},
        }
    }
    with pytest.raises(ValidationError, match="unknown_strategy"):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("bad_min_steps", [0, -1, -100])
def test_pruning_invalid_min_steps_rejected(bad_min_steps):
    """Недопустимые значения min_steps (< 1) отклоняются."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "min_steps": bad_min_steps},
        }
    }
    with pytest.raises(ValidationError, match="min_steps"):
        Config.model_validate(cfg_data)


def test_pruning_invalid_n_startup_trials_rejected():
    """Недопустимое значение n_startup_trials (< 1) отклоняется."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "n_startup_trials": 0},
        }
    }
    with pytest.raises(ValidationError, match="n_startup_trials"):
        Config.model_validate(cfg_data)


def test_pruning_invalid_reduction_factor_rejected():
    """Недопустимое значение reduction_factor (< 2) отклоняется."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "pruning": {"enable": True, "strategy": "hyperband", "reduction_factor": 1},
        }
    }
    with pytest.raises(ValidationError, match="reduction_factor"):
        Config.model_validate(cfg_data)


def test_pruning_conflict_kfold_min_steps_exceeds_folds():
    """Конфликт с валидацией: min_steps > n_folds при k_fold отклоняется."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "pruning": {"enable": True, "min_steps": 3},
        }
    }
    with pytest.raises(
        ValidationError,
        match="pruning.min_steps .* cannot be greater than general.n_folds",
    ):
        Config.model_validate(cfg_data)


def test_pruning_disabled_skips_kfold_conflict_check():
    """При enable=False конфликт min_steps/n_folds не проверяется (обратная совместимость)."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "pruning": {"enable": False, "min_steps": 10},
        }
    }
    cfg = Config.model_validate(cfg_data)  # не должно падать
    assert cfg.general.pruning.enable is False


def test_pruning_train_test_split_warns_and_allows(caplog):
    """Прайнинг + train_test_split: конфиг принимается, выдаётся предупреждение.

    Для стратегий валидации без естественных шагов прайнер не применяется —
    поведение задокументировано и безопасно (AC-7).
    """
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "validation_strategy": "train_test_split",
            "pruning": {"enable": True, "min_steps": 1},
        }
    }
    with caplog.at_level(logging.WARNING):
        cfg = Config.model_validate(cfg_data)
    assert cfg.general.pruning.enable is True
    assert "early stopping will NOT be applied" in caplog.text


def test_pruning_allows_loo_and_auto():
    """Прайнинг разрешён для loo и auto (валидируется без ошибок)."""
    for strategy in ("loo", "auto"):
        cfg_data = BASE | {
            "general": {
                **BASE["general"],
                "validation_strategy": strategy,
                "pruning": {"enable": True, "min_steps": 1},
            }
        }
        cfg = Config.model_validate(cfg_data)
        assert cfg.general.pruning.enable is True


# ─────────────── Tests for feature_selection (issue #8) ───────────────
def test_feature_selection_defaults_backward_compat():
    """Конфиг без блока feature_selection парсится с дефолтным disabled-режимом.

    Обратная совместимость: отсутствие секции не меняет поведение библиотеки,
    все параметры принимают значения по умолчанию.
    """
    cfg = Config.model_validate(BASE)
    fs = cfg.general.feature_selection
    assert fs.mode == FeatureSelectionMode.disabled
    assert fs.method == FeatureSelectionMethod.importance
    assert fs.percentile == 50.0
    assert fs.min_features == 2
    assert fs.variance_threshold == 0.0
    assert fs.n_estimators == 50


@pytest.mark.parametrize("mode", ["disabled", "always", "auto"])
def test_feature_selection_explicit_modes_valid(mode):
    """Все поддерживаемые режимы отбора признаков проходят валидацию."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"mode": mode}}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.feature_selection.mode == FeatureSelectionMode(mode)


@pytest.mark.parametrize(
    "method", ["importance", "percentile", "mutual_info", "variance"]
)
def test_feature_selection_methods_valid(method):
    """Каждый из четырёх методов отбора признаков парсится корректно."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"method": method}}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.feature_selection.method == FeatureSelectionMethod(method)


def test_feature_selection_full_block_end_to_end():
    """Полный блок feature_selection доходит до Config без потерь."""
    fs_block = {
        "mode": "always",
        "method": "mutual_info",
        "percentile": 20.5,
        "min_features": 3,
        "variance_threshold": 0.1,
        "n_estimators": 100,
    }
    cfg_data = BASE | {"general": {**BASE["general"], "feature_selection": fs_block}}
    cfg = Config.model_validate(cfg_data)
    fs = cfg.general.feature_selection
    assert fs.mode == FeatureSelectionMode.always
    assert fs.method == FeatureSelectionMethod.mutual_info
    assert fs.percentile == 20.5
    assert fs.min_features == 3
    assert fs.variance_threshold == 0.1
    assert fs.n_estimators == 100


def test_feature_selection_percentile_int_coerced_to_float():
    """Целочисленный литерал percentile из YAML приводится к float без ошибок."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"percentile": 50}}
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.general.feature_selection.percentile == 50.0
    assert isinstance(cfg.general.feature_selection.percentile, float)


def test_feature_selection_model_standalone():
    """FeatureSelectionCfg можно использовать напрямую (без Config)."""
    fs = FeatureSelectionCfg(mode="auto", method="variance", n_estimators=25)
    assert fs.mode == FeatureSelectionMode.auto
    assert fs.method == FeatureSelectionMethod.variance
    assert fs.n_estimators == 25


@pytest.mark.parametrize(
    "field,value",
    [
        ("percentile", 100.0),
        ("percentile", 0.01),
        ("min_features", 1),
        ("variance_threshold", 0.0),
        ("n_estimators", 10),
    ],
)
def test_feature_selection_boundary_values_valid(field, value):
    """Граничные валидные значения параметров принимаются схемой."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {field: value}}
    }
    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.general.feature_selection, field) == value


@pytest.mark.parametrize("mode", ["magic", 123, "ALWAYS"])
def test_feature_selection_invalid_mode_rejected(mode):
    """Неизвестный или несовпадающий по регистру mode отклоняется.

    Сравнение значений enum строгое: 'ALWAYS' не равен 'always'.
    """
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"mode": mode}}
    }
    with pytest.raises(ValidationError, match="feature_selection"):
        Config.model_validate(cfg_data)
    with pytest.raises(ValidationError, match=str(mode)):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("method", ["pca", "rfe"])
def test_feature_selection_invalid_method_rejected(method):
    """Неизвестный метод отбора признаков отклоняется на этапе парсинга."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"method": method}}
    }
    with pytest.raises(ValidationError, match=method):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("percentile", [0.0, -10.0, 105.0])
def test_feature_selection_invalid_percentile_rejected(percentile):
    """percentile вне диапазона (0, 100] отклоняется."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"percentile": percentile}}
    }
    with pytest.raises(ValidationError, match="percentile"):
        Config.model_validate(cfg_data)


@pytest.mark.parametrize("min_features", [0, -1])
def test_feature_selection_invalid_min_features_rejected(min_features):
    """min_features < 1 отклоняется (защита от опустошения матрицы)."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"min_features": min_features}}
    }
    with pytest.raises(ValidationError, match="min_features"):
        Config.model_validate(cfg_data)


def test_feature_selection_negative_variance_threshold_rejected():
    """Отрицательный variance_threshold отклоняется."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "feature_selection": {"variance_threshold": -0.01},
        }
    }
    with pytest.raises(ValidationError, match="variance_threshold"):
        Config.model_validate(cfg_data)


def test_feature_selection_small_n_estimators_rejected():
    """n_estimators < 10 отклоняется (порог ge=10)."""
    cfg_data = BASE | {
        "general": {**BASE["general"], "feature_selection": {"n_estimators": 5}}
    }
    with pytest.raises(ValidationError, match="n_estimators"):
        Config.model_validate(cfg_data)


def test_feature_selection_extra_field_rejected():
    """Лишние ключи внутри блока отклоняются (extra='forbid')."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "feature_selection": {"unknown_param": 42},
        }
    }
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        Config.model_validate(cfg_data)


def test_feature_selection_invalid_rejected_even_when_disabled():
    """Строгая схема: невалидные параметры отклоняются даже при mode='disabled'."""
    cfg_data = BASE | {
        "general": {
            **BASE["general"],
            "feature_selection": {"mode": "disabled", "n_estimators": 3},
        }
    }
    with pytest.raises(ValidationError, match="n_estimators"):
        Config.model_validate(cfg_data)


def test_read_config_feature_selection_yaml(tmp_path):
    """Реальный YAML с секцией feature_selection парсится в корректный Config."""
    yaml_content = """
    general:
      comparison_metric: "r2"
      phases:
        - name: "search"
          n_trials: 5
          action: "all_algorithms"
      validation_strategy: "k_fold"
      n_folds: 3
      feature_selection:
        mode: "always"
        method: "percentile"
        percentile: 25
        min_features: 4
        variance_threshold: 0.0
        n_estimators: 30
    algorithms:
      elasticnet:
        enable: true
    """
    config_file = tmp_path / "test_config_fs.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = read_config(config_file)
    fs = config.general.feature_selection
    assert fs.mode == FeatureSelectionMode.always
    assert fs.method == FeatureSelectionMethod.percentile
    assert fs.percentile == 25.0
    assert fs.min_features == 4
    assert fs.variance_threshold == 0.0
    assert fs.n_estimators == 30


def test_read_config_feature_selection_absent_defaults(tmp_path):
    """YAML без секции feature_selection даёт disabled-режим по умолчанию."""
    yaml_content = """
    general:
      comparison_metric: "r2"
      phases:
        - name: "search"
          n_trials: 5
          action: "all_algorithms"
      validation_strategy: "k_fold"
      n_folds: 3
    algorithms:
      elasticnet:
        enable: true
    """
    config_file = tmp_path / "test_config_no_fs.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = read_config(config_file)
    fs = config.general.feature_selection
    assert fs.mode == FeatureSelectionMode.disabled
    assert fs.method == FeatureSelectionMethod.importance
    assert fs.percentile == 50.0
    assert fs.min_features == 2
    assert fs.variance_threshold == 0.0
    assert fs.n_estimators == 50
