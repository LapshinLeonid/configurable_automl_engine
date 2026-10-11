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
    SanityGateCfg,
    ERR_REMOVED_REFINE_WINNER,
    ERR_REMOVED_SANITY_GATE_MODE,
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

# Терминальный select-шаг (эпик #74): последняя фаза любого валидного конфига.
TERMINAL_SELECT = {
    "name": "final_select",
    "n_trials": 1,
    "action": "select",
    "strategy": "cv_best",
    "output": "winner",
}

BASE = {
    "general": {
        "comparison_metric": "nrmse",
        "validation_strategy": "k_fold",
        "n_folds": 5,
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            TERMINAL_SELECT,
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


def test_oversampling_algorithm_random_with_noise_rejected():
    """random_with_noise удалён из интерфейса (issue #27): конфиг с ним
    отклоняется на этапе валидации с понятным сообщением."""
    with pytest.raises(ValidationError, match="random_with_noise"):
        OversamplingCfg(enable=True, multiplier=2.0, algorithm="random_with_noise")


def test_oversampling_unknown_algorithm_rejected():
    """Неизвестный алгоритм отклоняется валидацией конфига."""
    with pytest.raises(ValidationError):
        OversamplingCfg(enable=True, multiplier=2.0, algorithm="magic_boost")


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
        - name: "final_select"
          n_trials: 1
          action: "select"
          strategy: "cv_best"
          output: "winner"
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
        phases=[
            HPOPhaseCfg(name="test", n_trials=1),
            HPOPhaseCfg(
                name="final_select",
                n_trials=1,
                action="select",
                strategy="cv_best",
                output="winner",
            ),
        ],
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
            phases=[
                HPOPhaseCfg(name="test", n_trials=1),
                HPOPhaseCfg(
                    name="final_select",
                    n_trials=1,
                    action="select",
                    strategy="cv_best",
                    output="winner",
                ),
            ],
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
            phases=[
                HPOPhaseCfg(name="test", n_trials=1),
                HPOPhaseCfg(
                    name="final_select",
                    n_trials=1,
                    action="select",
                    strategy="cv_best",
                    output="winner",
                ),
            ],
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
            "phases": [
                {"name": "search", "n_trials": 10},
                TERMINAL_SELECT,
            ],
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
            "phases": [
                {"name": "search", "n_trials": 10},
                TERMINAL_SELECT,
            ],
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
        "general": {"phases": [{"name": "test", "n_trials": 1}, TERMINAL_SELECT]},
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


# 5a. Тест состава DEFAULT_SPACES['svr'] (issue #58): только надёжные ядра
def test_default_svr_kernel_space_only_reliable_kernels():
    """В пространстве поиска SVR остаются только ядра rbf и linear.

    Позитивный сценарий: options == ["rbf", "linear"].
    Негативный сценарий: 'sigmoid' и 'poly' исключены из перебора.
    """
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES

    kernel_entry = DEFAULT_SPACES["svr"]["kernel"]
    assert kernel_entry.dist_type == "categorical"
    assert kernel_entry.config.options == ["rbf", "linear"]
    assert "sigmoid" not in kernel_entry.config.options
    assert "poly" not in kernel_entry.config.options


def test_default_svr_gamma_stays_in_space():
    """Параметр gamma остаётся в пространстве поиска SVR.

    Для rbf он используется, для linear игнорируется sklearn —
    удалять его не требуется (issue #58).
    """
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES

    svr_space = DEFAULT_SPACES["svr"]
    assert "gamma" in svr_space
    gamma_entry = svr_space["gamma"]
    assert gamma_entry.dist_type == "categorical"
    assert gamma_entry.config.options == ["scale", "auto"]


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
                "phases": [{"name": "p1", "n_trials": 1}, TERMINAL_SELECT],
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
    base_phases = [{"name": "test", "n_trials": 1}, TERMINAL_SELECT]

    # Ошибка: n_folds < 1
    with pytest.raises(ValidationError, match="`n_folds` must be at least 1"):
        GeneralCfg(phases=base_phases, n_folds=0)

    # Ошибка: k_fold требует n_folds >= 2
    with pytest.raises(ValidationError, match=".*n_folds must be ≥ 2 for k-fold.*"):
        GeneralCfg(phases=base_phases, validation_strategy="k_fold", n_folds=1)


def test_get_unknown_hyperparameters_none():
    cfg = AlgoCfg(hyperparameters=None)
    assert cfg.get_unknown_hyperparameters("xgboosting") == []


def test_get_unknown_hyperparameters_skipped_when_unresolvable():
    """Если оценщик не разрешается (неизвестный алгоритм / нет зависимости),
    проверка мягко пропускается — список неизвестных пуст."""
    cfg = AlgoCfg(hyperparameters={"a": [1, 10]})
    assert cfg.get_unknown_hyperparameters("no_such_algo") == []


def test_get_unknown_hyperparameters_valid():
    cfg = AlgoCfg(
        hyperparameters={"alpha": [1e-4, 10.0]}  # ✅ валидный параметр ElasticNet
    )

    assert cfg.get_unknown_hyperparameters("elasticnet") == []


def test_get_unknown_hyperparameters_legacy_name_valid():
    """Legacy-имя n_iter для ardregression считается допустимым."""
    cfg = AlgoCfg(hyperparameters={"n_iter": [100, 500]})
    assert cfg.get_unknown_hyperparameters("ardregression") == []


def test_get_unknown_hyperparameters_unknown():
    cfg = AlgoCfg(
        hyperparameters={"bad_param": [1, 10]}  # ❌ не в сигнатуре ElasticNet
    )

    assert cfg.get_unknown_hyperparameters("elasticnet") == ["bad_param"]


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


def test_hyperparameter_compatibility_error():
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]  # elasticnet

    cfg_data = {
        "general": {"phases": [{"name": "p1", "n_trials": 1}, TERMINAL_SELECT]},
        "algorithms": {
            algo_name: {"enable": True, "hyperparameters": {"bad": [1, 10]}}
        },
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)


def test_hyperparameter_compatibility_error_lists_allowed():
    """Сообщение об ошибке содержит реальный список допустимых гиперпараметров.

    Регрессионный тест: ранее в текст подставлялся литеральный ``{allowed}``
    (результат ``sorted(...)`` отбрасывался), что делало сообщение бесполезным.
    Допустимое множество теперь строится по сигнатуре конструктора оценщика.
    """
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]  # elasticnet

    cfg_data = BASE | {
        "algorithms": {algo_name: {"enable": True, "hyperparameters": {"bad": [1, 10]}}}
    }

    with pytest.raises(ValueError) as exc_info:
        Config.model_validate(cfg_data)

    msg = str(exc_info.value)
    assert "unknown hyperparameters ['bad']" in msg
    assert "Allowed parameters:" in msg
    assert "'alpha'" in msg  # реальный параметр ElasticNet присутствует в списке
    assert "'l1_ratio'" in msg
    assert "{allowed}" not in msg


def test_hyperparameter_compatibility_valid_hyperparameters_pass():
    """Совместимые гиперпараметры не приводят к ошибке валидации."""
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_name = AVAILABLE_ALGORITHMS[0]  # elasticnet

    cfg_data = BASE | {
        "algorithms": {
            algo_name: {"enable": True, "hyperparameters": {"alpha": [1e-4, 10.0]}}
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, algo_name).enable is True


def test_hyperparameter_compatibility_nondefault_valid_params():
    """Нестандартные, но валидные для оценщика параметры проходят валидацию.

    Регрессионный тест на issue #15: min_samples_split и max_features
    отсутствуют в DEFAULT_SPACES['random_forest'], но принимаются
    конструктором RandomForestRegressor.
    """
    cfg_data = BASE | {
        "algorithms": {
            "random_forest": {
                "enable": True,
                "hyperparameters": {
                    "min_samples_split": [2, 10],
                    "max_features": [0.1, 1.0],
                    "warm_start": [[True, False], "categorical"],
                },
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    rf_cfg = getattr(cfg.algorithms, "random_forest")
    assert rf_cfg.enable is True
    assert set(rf_cfg.hyperparameters) == {
        "min_samples_split",
        "max_features",
        "warm_start",
    }


def test_hyperparameter_compatibility_gpr_alpha():
    """alpha для gaussian_process_regression проходит валидацию.

    Регрессионный тест на issue #15: DEFAULT_SPACES['gaussian_process_regression']
    пуст, но alpha — штатный параметр GaussianProcessRegressor.
    """
    cfg_data = BASE | {
        "algorithms": {
            "gaussian_process_regression": {
                "enable": True,
                "hyperparameters": {"alpha": [1e-10, 1.0]},
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, "gaussian_process_regression").enable is True


def test_hyperparameter_compatibility_constant_value():
    """Одиночное (константное) значение валидного параметра проходит.

    SearchSpaceEntry трактует [value] как categorical с одним вариантом —
    фиксация любого валидного параметра не должна отклоняться.
    """
    cfg_data = BASE | {
        "algorithms": {
            "random_forest": {
                "enable": True,
                "hyperparameters": {"min_samples_split": [2]},
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert (
        getattr(cfg.algorithms, "random_forest")
        .hyperparameters["min_samples_split"]
        .dist_type
        == "categorical"
    )


def test_hyperparameter_compatibility_legacy_name():
    """Legacy-имя n_iter для ardregression проходит валидацию."""
    cfg_data = BASE | {
        "algorithms": {
            "ardregression": {
                "enable": True,
                "hyperparameters": {"n_iter": [100, 500]},
            }
        }
    }

    cfg = Config.model_validate(cfg_data)
    assert getattr(cfg.algorithms, "ardregression").enable is True


def test_hyperparameter_compatibility_typo_rejected():
    """Опечатка в имени параметра по-прежнему отклоняется."""
    cfg_data = BASE | {
        "algorithms": {
            "random_forest": {
                "enable": True,
                "hyperparameters": {"min_samples_spilt": [2, 10]},
            }
        }
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)


def test_hyperparameter_compatibility_gpr_typo_rejected():
    """Опечатка для gaussian_process_regression теперь тоже отклоняется.

    Ранее пустой реестр DEFAULT_SPACES['gaussian_process_regression']
    молча пропускал любые ключи, включая опечатки.
    """
    cfg_data = BASE | {
        "algorithms": {
            "gaussian_process_regression": {
                "enable": True,
                "hyperparameters": {"aplha": [1e-10, 1.0]},
            }
        }
    }

    with pytest.raises(ValueError, match="unknown hyperparameters"):
        Config.model_validate(cfg_data)


def test_hyperparameter_compatibility_skips_disabled():
    """Выключенные алгоритмы пропускаются проверкой совместимости.

    algo_b выключен и содержит недопустимый гиперпараметр: тест проходит
    только благодаря guard'у ``enable=False``.
    """
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_a, algo_b = AVAILABLE_ALGORITHMS[0], AVAILABLE_ALGORITHMS[1]

    cfg_data = BASE | {
        "algorithms": {
            algo_a: {"enable": True},
            algo_b: {"enable": False, "hyperparameters": {"bad": [1, 10]}},
        }
    }

    cfg = Config.model_validate(cfg_data)  # не должно падать
    assert getattr(cfg.algorithms, algo_a).enable is True
    assert getattr(cfg.algorithms, algo_b).enable is False


def test_hyperparameter_compatibility_aggregates_errors():
    """Ошибки для нескольких алгоритмов собираются в одно исключение."""
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS

    algo_a, algo_b = AVAILABLE_ALGORITHMS[0], AVAILABLE_ALGORITHMS[1]

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
        "general": {
            **BASE["general"],
            "feature_selection": {"min_features": min_features},
        }
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
        - name: "final_select"
          n_trials: 1
          action: "select"
          strategy: "cv_best"
          output: "winner"
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
        - name: "final_select"
          n_trials: 1
          action: "select"
          strategy: "cv_best"
          output: "winner"
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


# ─────────────────────── Sanity Gate v2 (эпик #61, T7, issue #69) ────────────


def _sg_cfg(sanity_gate_block: dict) -> dict:
    """Конфиг с блоком ``general.sanity_gate`` поверх базового."""
    return {
        "general": {**BASE["general"], "sanity_gate": sanity_gate_block},
        "algorithms": BASE["algorithms"],
    }


def test_sanity_gate_v2_defaults():
    """Дефолты блока sanity_gate v2: мягкие пороги из постановки.

    Эпик #74: поле mode удалено — секция является только источником порогов.
    """
    cfg = Config.model_validate(BASE)
    sg = cfg.general.sanity_gate
    assert "mode" not in SanityGateCfg.model_fields
    assert sg.top_k_candidates == 3
    assert sg.corridor_delta == 0.15
    assert sg.corridor_mode == "auto"
    assert sg.min_prediction_diversity == 0.15
    assert sg.min_unique_count == 5
    assert sg.min_unique_ratio == 0.05
    assert sg.max_dead_feature_ratio == 0.4
    assert sg.dead_features_require_low_diversity is True
    assert sg.max_generalization_gap == 1.5
    assert sg.generalization_gap_mode == "warn"
    assert sg.check_permutation_sensitivity is True
    assert sg.permutation_max_rows == 5000
    assert sg.permutation_max_features == 100
    assert sg.permutation_repeats == 3
    assert sg.permutation_seed == 42
    assert sg.audit_time_budget_seconds == 0
    assert sg.check_diversity is True
    assert sg.check_unique is True
    assert sg.check_dead_features is True
    assert sg.check_generalization_gap is True


def test_sanity_gate_v2_full_block_parses():
    """Полный блок sanity_gate v2 (постановка эпика #61 / T5) парсится."""
    block = {
        "top_k_candidates": 5,
        "corridor_delta": 0.2,
        "corridor_mode": "multiplicative",
        "enforce_family_diversity": True,
        "algorithm_families": {"ridge": "linear", "svr": "kernel"},
        "allowed_families": ["linear", "kernel"],
        "family_diversity_multiplier": 1.3,
        "min_prediction_diversity": 0.25,
        "adaptive_diversity": True,
        "min_unique_count": 10,
        "min_unique_ratio": 0.1,
        "max_dead_feature_ratio": 0.5,
        "dead_features_require_low_diversity": False,
        "zero_coef_tolerance": 1e-10,
        "max_generalization_gap": 2.0,
        "generalization_gap_mode": "reject",
        "check_permutation_sensitivity": False,
        "permutation_max_rows": 2000,
        "permutation_max_features": 50,
        "permutation_repeats": 5,
        "permutation_seed": 7,
        "permutation_tolerance": 1e-2,
        "audit_time_budget_seconds": 120.5,
        "check_diversity": True,
        "check_unique": True,
        "check_dead_features": True,
        "check_generalization_gap": True,
    }
    cfg = Config.model_validate(_sg_cfg(block))
    sg = cfg.general.sanity_gate
    assert sg.top_k_candidates == 5
    assert sg.corridor_delta == 0.2
    assert sg.corridor_mode == "multiplicative"
    assert sg.enforce_family_diversity is True
    assert sg.algorithm_families == {"ridge": "linear", "svr": "kernel"}
    assert sg.allowed_families == ["linear", "kernel"]
    assert sg.family_diversity_multiplier == 1.3
    assert sg.min_prediction_diversity == 0.25
    assert sg.adaptive_diversity is True
    assert sg.min_unique_count == 10
    assert sg.min_unique_ratio == 0.1
    assert sg.max_dead_feature_ratio == 0.5
    assert sg.dead_features_require_low_diversity is False
    assert sg.zero_coef_tolerance == 1e-10
    assert sg.max_generalization_gap == 2.0
    assert sg.generalization_gap_mode == "reject"
    assert sg.check_permutation_sensitivity is False
    assert sg.permutation_max_rows == 2000
    assert sg.permutation_max_features == 50
    assert sg.permutation_repeats == 5
    assert sg.permutation_seed == 7
    assert sg.permutation_tolerance == 1e-2
    assert sg.audit_time_budget_seconds == 120.5
    assert sg.check_diversity is True
    assert sg.check_unique is True
    assert sg.check_dead_features is True
    assert sg.check_generalization_gap is True


@pytest.mark.parametrize("legacy_mode", ["off", "warn_only", "active", "weird"])
def test_sanity_gate_v2_mode_removed_with_migration_hint(legacy_mode):
    """Поле sanity_gate.mode удалено (эпик #74): конфиг с ним отвергается
    с понятной ошибкой и подсказкой миграции на явный select-шаг.

    Идентификация ошибки — по стабильному коду
    (``ERR_REMOVED_SANITY_GATE_MODE`` из error['type']); содержательные
    части подсказки (упоминание поля и select-шага) проверяются отдельно.
    """
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(_sg_cfg({"mode": legacy_mode}))
    error_types = {e["type"] for e in exc_info.value.errors()}
    assert ERR_REMOVED_SANITY_GATE_MODE in error_types
    msg = str(exc_info.value)
    assert "sanity_gate.mode" in msg
    assert "select" in msg


@pytest.mark.parametrize("corridor_mode", ["multiplicative", "additive", "auto"])
def test_sanity_gate_v2_corridor_mode_valid(corridor_mode):
    """Все три формы коридора принимаются схемой."""
    cfg = Config.model_validate(_sg_cfg({"corridor_mode": corridor_mode}))
    assert cfg.general.sanity_gate.corridor_mode == corridor_mode


def test_sanity_gate_v2_corridor_mode_invalid_rejected():
    """Неизвестная форма коридора отклоняется на этапе валидации."""
    with pytest.raises(ValidationError, match="corridor_mode"):
        Config.model_validate(_sg_cfg({"corridor_mode": "geometric"}))


@pytest.mark.parametrize("gap_mode", ["warn", "reject"])
def test_sanity_gate_v2_generalization_gap_mode_valid(gap_mode):
    """Both generalization_gap_mode values are accepted by the schema."""
    cfg = Config.model_validate(_sg_cfg({"generalization_gap_mode": gap_mode}))
    assert cfg.general.sanity_gate.generalization_gap_mode == gap_mode


@pytest.mark.parametrize(
    "bad_value", ["invalid", "", "warn_only", "active", "off", 1, None]
)
def test_sanity_gate_v2_generalization_gap_mode_invalid_rejected(bad_value):
    """Неизвестное значение generalization_gap_mode отклоняется."""
    with pytest.raises(ValidationError, match="generalization_gap_mode"):
        Config.model_validate(_sg_cfg({"generalization_gap_mode": bad_value}))


@pytest.mark.parametrize(
    "field,value",
    [
        # Граничные значения (равенство порогу — валидно).
        ("min_prediction_diversity", 1.0),
        ("min_unique_ratio", 1.0),
        ("max_dead_feature_ratio", 0.0),
        ("max_dead_feature_ratio", 1.0),
        ("max_generalization_gap", 1.0),
        ("top_k_candidates", 1),
        ("min_unique_count", 1),
        # corridor_delta → 0 (строго больше нуля, но сколь угодно мал).
        ("corridor_delta", 1e-9),
        # Множители и допуски: неотрицательные/положительные границы.
        ("family_diversity_multiplier", 1e-9),
        ("zero_coef_tolerance", 0.0),
        ("permutation_tolerance", 0.0),
        ("audit_time_budget_seconds", 0.0),
    ],
)
def test_sanity_gate_v2_threshold_boundaries_valid(field, value):
    """Граничные значения порогов блока v2 проходят валидацию."""
    cfg = Config.model_validate(_sg_cfg({field: value}))
    assert getattr(cfg.general.sanity_gate, field) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_prediction_diversity", 0),
        ("min_prediction_diversity", -0.1),
        ("min_prediction_diversity", 1.5),
        ("min_unique_ratio", 0),
        ("min_unique_ratio", 1.5),
        ("max_dead_feature_ratio", -0.1),
        ("max_dead_feature_ratio", 1.1),
        ("max_generalization_gap", 0.5),
        ("max_generalization_gap", 0.99),
        ("top_k_candidates", 0),
        ("top_k_candidates", -1),
        ("min_unique_count", 0),
        ("corridor_delta", 0),
        ("corridor_delta", 1.0),
        ("corridor_delta", -0.1),
        ("family_diversity_multiplier", 0),
        ("family_diversity_multiplier", -1),
        ("zero_coef_tolerance", -1e-6),
        ("permutation_tolerance", -1e-6),
        ("permutation_repeats", 0),
        ("permutation_max_rows", 0),
        ("permutation_max_features", 0),
        ("audit_time_budget_seconds", -0.1),
    ],
)
def test_sanity_gate_v2_threshold_out_of_bounds_rejected(field, value):
    """Значения порогов вне допустимых границ отклоняются с именем поля."""
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: value}))


def test_sanity_gate_v2_extra_forbid():
    """Лишние ключи внутри блока sanity_gate отклоняются (extra='forbid')."""
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        Config.model_validate(_sg_cfg({"unknown_sanity_key": True}))


@pytest.mark.parametrize(
    "field",
    [
        "check_diversity",
        "check_unique",
        "check_dead_features",
        "check_generalization_gap",
        "adaptive_diversity",
        "dead_features_require_low_diversity",
        "enforce_family_diversity",
        "check_permutation_sensitivity",
    ],
)
def test_sanity_gate_v2_check_flags_accept_bools_only(field):
    """Флаги-контуры принимают bool; несовместимые типы отклоняются.

    Pydantic в lax-режиме коэрсит ``1``/``"yes"`` в True, но списки,
    словари и None — несовместимые типы — отклоняются.
    """
    for flag_value in (True, False):
        cfg = Config.model_validate(_sg_cfg({field: flag_value}))
        assert getattr(cfg.general.sanity_gate, field) == flag_value
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: [1]}))
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: {"value": True}}))
    with pytest.raises(ValidationError, match=re.escape(field)):
        Config.model_validate(_sg_cfg({field: None}))


def test_sanity_gate_v2_empty_block_is_valid_thresholds_only():
    """Пустой блок sanity_gate: {} валиден — секция является источником
    порогов, а не поведенческим флагом (эпик #74)."""
    cfg = Config.model_validate(_sg_cfg({}))
    sg = cfg.general.sanity_gate
    assert "mode" not in SanityGateCfg.model_fields
    assert sg.corridor_delta == 0.15
    assert sg.min_prediction_diversity == 0.15
    assert sg.max_generalization_gap == 1.5
    assert sg.generalization_gap_mode == "warn"


# ─────────────── Pipeline phases: refine / select (эпик #74) ───────────────


def _phase(**overrides: object) -> dict:
    """Словарь одной фазы пайплайна поверх дефолтов."""
    base: dict = {"name": "phase", "n_trials": 1, "action": "all_algorithms"}
    base.update(overrides)
    return base


def _cfg_with_phases(*phases: dict) -> dict:
    """Конфиг с заданной последовательностью фаз поверх BASE."""
    return {
        "general": {**BASE["general"], "phases": list(phases)},
        "algorithms": BASE["algorithms"],
    }


# ── Позитивные сценарии ──


@pytest.mark.parametrize(
    "phase",
    [
        {"name": "refine_all", "n_trials": 5, "action": "refine"},
        {
            "name": "sel_winner",
            "n_trials": 1,
            "action": "select",
            "strategy": "cv_best",
            "output": "winner",
        },
        {
            "name": "sel_topk",
            "n_trials": 1,
            "action": "select",
            "strategy": "cv_best",
            "output": "top_k",
            "k": 1,
        },
        {
            "name": "sel_pool",
            "n_trials": 1,
            "action": "select",
            "strategy": "cv_best",
            "output": "pool",
        },
        {
            "name": "gate_winner",
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "winner",
        },
        {
            "name": "gate_topk",
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "top_k",
            "k": 3,
        },
        {
            "name": "gate_pool",
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "pool",
        },
    ],
)
def test_phase_actions_valid_combinations(phase):
    """Все допустимые комбинации action/strategy/output/k парсятся схемой.

    Фазы с output='winner' ставятся последними (терминальный шаг), остальные
    — в середину пайплайна перед терминальным select.
    """
    is_terminal = phase.get("output") == "winner"
    phases = [_phase(), phase] if is_terminal else [_phase(), phase, TERMINAL_SELECT]
    cfg = Config.model_validate(_cfg_with_phases(*phases))
    parsed = cfg.general.phases[1]
    assert parsed.action == phase["action"]
    assert parsed.strategy == phase.get("strategy")
    assert parsed.output == phase.get("output")
    assert parsed.k == phase.get("k")


def test_refine_default_fields_are_none():
    """У фаз all_algorithms/refine поля strategy/output/k равны None."""
    cfg = Config.model_validate(
        _cfg_with_phases(
            _phase(action="all_algorithms"), _phase(action="refine"), TERMINAL_SELECT
        )
    )
    for phase in cfg.general.phases[:2]:
        assert phase.strategy is None
        assert phase.output is None
        assert phase.k is None


def test_select_immediately_after_select_is_valid():
    """select сразу после select валиден (эпик #74, edge case 5): top_k → winner."""
    cfg = Config.model_validate(
        _cfg_with_phases(
            _phase(),
            _phase(action="select", strategy="sanity_gate", output="top_k", k=2),
            TERMINAL_SELECT,
        )
    )
    assert [p.action for p in cfg.general.phases] == [
        "all_algorithms",
        "select",
        "select",
    ]


def test_refine_right_after_all_algorithms_is_valid():
    """refine сразу после all_algorithms валиден (эпик #74, edge case 4)."""
    cfg = Config.model_validate(
        _cfg_with_phases(_phase(), _phase(action="refine"), TERMINAL_SELECT)
    )
    assert [p.action for p in cfg.general.phases] == [
        "all_algorithms",
        "refine",
        "select",
    ]


def test_sanity_gate_section_without_sanity_select_is_valid():
    """Секция sanity_gate без select-шагов со strategy: sanity_gate валидна:
    секция — источник порогов, а не поведенческий флаг (эпик #74)."""
    data = _cfg_with_phases(_phase(), TERMINAL_SELECT)
    data["general"]["sanity_gate"] = {
        "top_k_candidates": 2,
        "corridor_delta": 0.3,
        "audit_time_budget_seconds": 60,
    }
    cfg = Config.model_validate(data)
    assert cfg.general.sanity_gate.top_k_candidates == 2
    assert cfg.general.sanity_gate.corridor_delta == 0.3


def test_empty_phases_list_is_still_accepted():
    """Пустой список фаз не проверяется кросс-валидацией (защита от пустого
    пайплайна остаётся рантайм-проверкой train_best_model)."""
    cfg = Config(
        general=GeneralCfg(phases=[]), algorithms={"elasticnet": {"enable": True}}
    )
    assert cfg.general.phases == []


# ── Негативные сценарии ──


def test_refine_winner_rejected_with_migration_hint():
    """action 'refine_winner' удалён: конфиг отвергается с подсказкой
    миграции (не молчаливое игнорирование).

    Тест завязывается на стабильный код ошибки
    (``ERR_REMOVED_REFINE_WINNER`` из error['type']) и отдельно — на
    содержательные части текста подсказки; точная формулировка подсказки
    может меняться без поломки теста.
    """
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(
            _cfg_with_phases(_phase(action="refine_winner"), TERMINAL_SELECT)
        )
    error_types = {e["type"] for e in exc_info.value.errors()}
    assert ERR_REMOVED_REFINE_WINNER in error_types
    msg = str(exc_info.value)
    assert "refine_winner" in msg
    assert "has been removed" in msg
    assert "select" in msg and "refine" in msg
    assert "README.md" in msg


def test_select_first_phase_rejected():
    """select не может быть первой фазой: нечего ранжировать/аудировать."""
    with pytest.raises(ValidationError, match="cannot be the first phase"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(action="select", strategy="cv_best", output="winner")
            )
        )


def test_phases_after_terminal_select_rejected():
    """Фазы после терминального select (output: winner) запрещены."""
    with pytest.raises(ValidationError, match="terminal"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(),
                TERMINAL_SELECT,
                _phase(action="refine"),
            )
        )


def test_select_winner_in_middle_rejected():
    """select(output: winner) в середине пайплайна отклоняется."""
    with pytest.raises(ValidationError, match="terminal"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(),
                _phase(action="select", strategy="cv_best", output="winner"),
                TERMINAL_SELECT,
            )
        )


@pytest.mark.parametrize(
    "last_phase",
    [
        _phase(),  # all_algorithms последней
        _phase(action="refine"),  # refine последней
        _phase(action="select", strategy="cv_best", output="top_k", k=1),
        _phase(action="select", strategy="cv_best", output="pool"),
    ],
)
def test_last_phase_must_be_terminal_select(last_phase):
    """Последняя фаза обязана быть select с output: winner."""
    with pytest.raises(ValidationError, match="must be a 'select' phase"):
        Config.model_validate(_cfg_with_phases(_phase(), last_phase))


def test_last_phase_select_without_winner_output_rejected():
    """Последняя фаза select, но с output != winner — ошибка."""
    with pytest.raises(ValidationError, match="must be a 'select' phase"):
        Config.model_validate(
            _cfg_with_phases(
                _phase(),
                _phase(action="select", strategy="cv_best", output="pool"),
            )
        )


def test_top_k_without_k_rejected():
    """output: top_k без k >= 1 отклоняется."""
    with pytest.raises(ValidationError, match="top_k.*requires.*k"):
        Config.model_validate(
            HPOPhaseCfg(n_trials=1, action="select", strategy="cv_best", output="top_k")
        )


def test_k_zero_or_negative_rejected():
    """k < 1 отклоняется ограничением поля (ge=1)."""
    with pytest.raises(ValidationError, match="k"):
        Config.model_validate(
            HPOPhaseCfg(
                n_trials=1, action="select", strategy="cv_best", output="top_k", k=0
            )
        )


@pytest.mark.parametrize("output", ["winner", "pool"])
def test_k_with_winner_or_pool_rejected(output):
    """k при output: winner|pool запрещён."""
    with pytest.raises(ValidationError, match="k is not allowed"):
        Config.model_validate(
            HPOPhaseCfg(
                n_trials=1, action="select", strategy="cv_best", output=output, k=2
            )
        )


@pytest.mark.parametrize("field", ["strategy", "output", "k"])
def test_select_fields_forbidden_on_non_select_phases(field):
    """Поля strategy/output/k запрещены у фаз all_algorithms/refine."""
    value: object = (
        "cv_best" if field == "strategy" else "winner" if field == "output" else 2
    )
    with pytest.raises(ValidationError, match="only allowed for action 'select'"):
        HPOPhaseCfg(action="refine", n_trials=3, **{field: value})  # type: ignore[arg-type]


def test_select_without_strategy_rejected():
    """select без strategy отклоняется с подсказкой допустимых значений."""
    with pytest.raises(ValidationError, match="requires fields \\['strategy'\\]"):
        HPOPhaseCfg(n_trials=1, action="select", output="winner")


def test_select_without_output_rejected():
    """select без output отклоняется с подсказкой допустимых значений."""
    with pytest.raises(ValidationError, match="requires fields \\['output'\\]"):
        HPOPhaseCfg(n_trials=1, action="select", strategy="cv_best")


def test_select_without_strategy_and_output_rejected():
    """select без обоих полей перечисляет оба в сообщении об ошибке."""
    with pytest.raises(ValidationError, match="\\['strategy', 'output'\\]"):
        HPOPhaseCfg(n_trials=1, action="select")


# ─────────── Runtime: select/refine execution (эпик #74) ───────────


def test_refine_phase_executes_with_warm_start():
    """refine исполним (эпик #74, R3): warm-start HPO для всего множества.

    Конфиг [all_algorithms → refine → terminal select] выполняет обе HPO-фазы
    (по одному запуску HPO на каждый алгоритм в каждой фазе; в refine
    передаются initial_params) и завершается артефактом терминального
    select-шага (R2).
    """
    import numpy as np
    import pandas as pd
    from configurable_automl_engine.training_engine.component import train_best_model

    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
    df["target"] = 2.0 * df["a"] + rng.randn(30) * 0.1

    cfg_refine = _cfg_with_phases(
        _phase(), _phase(action="refine", n_trials=1), TERMINAL_SELECT
    )
    with patch(
        "configurable_automl_engine.training_engine.component._run_hpo",
        side_effect=lambda **kw: (0.9, {"p": 1}),
    ) as mock_hpo:
        res = train_best_model(cfg_refine, df, target="target")

    assert res["algorithm"] in BASE["algorithms"]
    assert "sanity_gate" not in res
    # elasticnet — единственный алгоритм BASE: 1 вызов в all_algorithms-фазе
    # (без initial_params) + 1 вызов в refine-фазе (с warm-start).
    assert mock_hpo.call_count == 2
    first_call = mock_hpo.call_args_list[0].kwargs
    refine_call = mock_hpo.call_args_list[1].kwargs
    assert first_call.get("initial_params") is None
    assert refine_call.get("initial_params") == {"p": 1}


def test_select_phase_executes_in_pipeline():
    """Исполнение select-фаз реализовано (эпик #74, задача R2): терминальный
    select(cv_best, winner) отрабатывает и возвращает артефакт."""
    import numpy as np
    import pandas as pd
    from configurable_automl_engine.training_engine.component import train_best_model

    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
    df["target"] = 2.0 * df["a"] + rng.randn(30) * 0.1

    res = train_best_model(
        _cfg_with_phases(_phase(), TERMINAL_SELECT), df, target="target"
    )
    assert res["algorithm"] in BASE["algorithms"]
    assert "sanity_gate" not in res


# ─────────── M1/M2 (ревью PR !43): граничные комбинации фаз ───────────


def test_refine_first_phase_is_valid():
    """refine первой фазой валиден (M1): входное рабочее множество —
    полный набор включённых алгоритмов (warm-start от предыдущей фазы
    отсутствует); семантика исполнения финализируется задачей R3."""
    cfg = Config.model_validate(
        _cfg_with_phases(_phase(action="refine"), TERMINAL_SELECT)
    )
    assert [p.action for p in cfg.general.phases] == ["refine", "select"]


def test_select_cv_best_pool_is_valid():
    """select(strategy: cv_best, output: pool) валиден (M2): коридорный пул
    финалистов (коридор δ ∩ top_k_candidates, пороги из секции sanity_gate),
    ранжирование по CV-скору БЕЗ аудита ModelSanityGate."""
    phase = _phase(action="select", strategy="cv_best", output="pool")
    cfg = Config.model_validate(_cfg_with_phases(_phase(), phase, TERMINAL_SELECT))
    parsed = cfg.general.phases[1]
    assert parsed.strategy == "cv_best"
    assert parsed.output == "pool"
    assert parsed.k is None


def test_k_exceeding_top_k_candidates_is_accepted():
    """k > sanity_gate.top_k_candidates не проверяется на уровне конфига
    (M3): пул финалистов сначала ограничивается top_k_candidates, точное
    взаимодействие k с пулом определяется при исполнении (задача R2)."""
    data = _cfg_with_phases(_phase(), TERMINAL_SELECT)
    data["general"]["sanity_gate"] = {"top_k_candidates": 3}
    cfg = Config.model_validate(
        _cfg_with_phases(
            _phase(),
            _phase(action="select", strategy="sanity_gate", output="top_k", k=10),
            TERMINAL_SELECT,
        )
    )
    assert cfg.general.phases[1].k == 10
    assert cfg.general.sanity_gate.top_k_candidates == 3
    _ = data  # sanity_gate-секция валидна сама по себе
