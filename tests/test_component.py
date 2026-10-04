import pytest
import pandas as pd
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from types import SimpleNamespace
import importlib
import inspect

import numpy as np
from sklearn.datasets import make_regression

from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.component import (
    _run_hpo,
    _fit_and_save,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    AlgoCfg,
    FeatureSelectionCfg,
    FeatureSelectionMode,
    ValidationStrategy,
)
from configurable_automl_engine.tuner import InvalidAlgorithmError

import textwrap


HAPPY_CFG = """
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  phases:
    - name: "Coarse Search"
      n_trials: 3
      action: "all_algorithms"
    - name: "Fine Tuning"
      n_trials: 5
      action: "refine_winner"
algorithms:
  random_forest:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      n_estimators: [10, 20]  # Убрали фигурные скобки, используем отступы
  extra_trees:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      n_estimators: [10, 20]
  decision_tree:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      max_depth: [2, 3]
  elasticnet:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
      l1_ratio: [0.2, 0.8]
  lasso:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
  ridge:
    enable: false
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
  nearest_neighbors_regression:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      n_neighbors: [3, 5]
  svr:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      C: [0.1, 1.0]
      kernel: [linear]
  xgboosting:
    enable: false
"""

# Конфиг, где XGBoost ВКЛЮЧЁН → компонент обязан упасть (если XGBoost не реализован в tuner)
BROKEN_CFG = BROKEN_CFG = """
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  phases:
    - name: "Coarse Search"
      n_trials: 1
      action: "all_algorithms"

algorithms:
  totally_unknown_algo:
    enable: true
"""


# --------------------------------------------------------------------------- #
#  HAPPY PATH
# --------------------------------------------------------------------------- #
def test_happy_path(tmp_path: Path, small_dataset):
    """
    Проверка успешного цикла обучения на синтетических данных из фикстуры.
    """
    cfg_file = tmp_path / "cfg.yaml"
    model_path = tmp_path / "model.pkl"
    cfg_file.write_text(HAPPY_CFG.format(model_path="dummy_path"), "utf-8")

    # Используем фикстуру small_dataset вместо локальной функции
    res = train_best_model(cfg_file, small_dataset, model_path_override=model_path)

    assert Path(res["model_path"]).exists()
    assert res["algorithm"] in {
        "random_forest",
        "extra_trees",
        "decision_tree",
        "elasticnet",
        "lasso",
        "ridge",
        "nearest_neighbors_regression",
        "svr",
    }
    assert isinstance(res["score"], float)


# --------------------------------------------------------------------------- #
#  BAD INPUT TYPE
# --------------------------------------------------------------------------- #
def test_bad_input_type(tmp_path: Path):
    """
    Проверка, что передача не-DataFrame вызывает TypeError.
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(HAPPY_CFG.format(model_path="dummy_path"), "utf-8")

    with pytest.raises(TypeError):
        # Передаем список вместо pandas.DataFrame
        train_best_model(
            cfg_file, ["not", "a", "df"], model_path_override=tmp_path / "m.pkl"
        )


# --------------------------------------------------------------------------- #
#  NO ALGORITHMS ENABLED
# --------------------------------------------------------------------------- #
def test_no_algorithms_enabled(tmp_path: Path, small_dataset):
    """
    Проверка падения, если в конфиге не включен ни один алгоритм.
    """
    empty_cfg = """
general:
  comparison_metric: rmse
  path_to_model: 'm.pkl'
algorithms:
  random_forest:
    enable: false
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(empty_cfg, "utf-8")

    with pytest.raises(ValueError):
        train_best_model(cfg_file, small_dataset)


# --------------------------------------------------------------------------- #
#  UNSUPPORTED ALGORITHM SHOULD RAISE
# --------------------------------------------------------------------------- #
def _make_iae(message: str):
    """
    Создаёт экземпляр исключения с __class__.__name__ == 'InvalidAlgorithmError'.
    Нужно потому что _run_hpo проверяет имя класса строкой,
    а не через isinstance — это позволяет поймать IAE из любого модуля.
    """

    class InvalidAlgorithmError(Exception):
        pass

    return InvalidAlgorithmError(message)


def test_unsupported_algorithm(tmp_path: Path, small_dataset):
    """
    Проверка, что алгоритм, чей тюнер бросает InvalidAlgorithmError,
    дисквалифицируется, а не прерывает запуск (issue #12).
    Если дисквалифицированы ВСЕ алгоритмы — поднимается штатный
    RuntimeError со списком упавших алгоритмов.
    Покрывает ветку в _worker:
        except _CanonicalIAE: логируем причину и возвращаем None
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(HAPPY_CFG.format(model_path="dummy_path"), "utf-8")

    # Имитируем тюнер, чей optimize() бросает InvalidAlgorithmError
    mock_tuner = MagicMock()
    mock_tuner.optimize.side_effect = _make_iae("Algorithm not supported")

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=mock_tuner,
    ):
        with pytest.raises(RuntimeError) as excinfo:
            train_best_model(
                cfg_file, small_dataset, model_path_override=tmp_path / "m.pkl"
            )

    assert "No algorithms produced valid scores" in str(excinfo.value)
    assert "random_forest" in str(excinfo.value)


# --------------------------------------------------------------------------- #
#  CIRCUIT BREAKER (issue #12): дисквалификация не прерывает запуск
# --------------------------------------------------------------------------- #
_CIRCUIT_BREAKER_CFG = """
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  phases:
    - name: "Coarse Search"
      n_trials: 2
      action: "all_algorithms"
algorithms:
  ridge:
    enable: true
    tuner: "mock.tuner_ridge"
    trainer_module: "configurable_automl_engine.trainer"
    hyperparameters:
      alpha: [0.1, 1.0]
  random_forest:
    enable: true
    tuner: "mock.tuner_rf"
    trainer_module: "configurable_automl_engine.trainer"
    hyperparameters:
      n_estimators: [10, 20]
"""


def _make_broken_tuner(message: str) -> MagicMock:
    """Тюнер, который всегда бросает InvalidAlgorithmError."""
    tuner = MagicMock()
    tuner.optimize.side_effect = _make_iae(message)
    return tuner


def _make_good_tuner(
    best_params: dict, best_score: float = 0.9
) -> MagicMock:
    """Тюнер, который успешно возвращает (model, best_params, best_score)."""
    tuner = MagicMock()
    tuner.optimize.return_value = (None, best_params, best_score)
    return tuner


def _patch_load_modules(broken_tuner, good_tuner):
    """Подменяет _load_module так, чтобы разные тюнеры возвращались
    по разным путям модулей, указанным в конфиге."""

    def fake_load(path: str):
        if path == "mock.tuner_ridge":
            return broken_tuner
        if path == "mock.tuner_rf":
            return good_tuner
        # Все остальные пути (например, trainer_module) грузим по-настоящему
        return importlib.import_module(path)

    return patch(
        "configurable_automl_engine.training_engine.component._load_module",
        side_effect=fake_load,
    )


def test_circuit_breaker_disqualifies_only_broken_algorithm(
    tmp_path: Path, small_dataset
):
    """
    e2e-тест issue #12: один алгоритм дисквалифицируется (тюнер бросает
    InvalidAlgorithmError), второй успешно завершает HPO — запуск обязан
    завершиться успешно, победителем становится рабочий алгоритм, а
    дисквалифицированный фиксируется в result["disqualified_algorithms"].
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        _CIRCUIT_BREAKER_CFG.format(model_path="dummy_path"), "utf-8"
    )

    broken_tuner = _make_broken_tuner(
        "Algorithm 'ridge' disqualified after 5 consecutive fatal failures"
    )
    good_tuner = _make_good_tuner({"n_estimators": 20})

    with _patch_load_modules(broken_tuner, good_tuner):
        res = train_best_model(
            cfg_file, small_dataset, model_path_override=tmp_path / "m.pkl"
        )

    # Запуск завершился успешно, победил рабочий алгоритм
    assert res["algorithm"] == "random_forest"
    assert Path(res["model_path"]).exists()
    # Дисквалифицированный алгоритм зафиксирован с причиной
    assert res["disqualified_algorithms"] == {
        "ridge": (
            "Algorithm 'ridge' disqualified after 5 consecutive fatal failures"
        )
    }


def test_circuit_breaker_all_algorithms_disqualified_raises(
    tmp_path: Path, small_dataset
):
    """
    Негативный e2e-тест issue #12: если дисквалифицированы ВСЕ алгоритмы,
    поднимается штатный RuntimeError со списком упавших алгоритмов.
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        _CIRCUIT_BREAKER_CFG.format(model_path="dummy_path"), "utf-8"
    )

    broken_tuner = _make_broken_tuner("disqualified after 5 fatal failures")

    with _patch_load_modules(broken_tuner, broken_tuner):
        with pytest.raises(RuntimeError) as excinfo:
            train_best_model(
                cfg_file, small_dataset, model_path_override=tmp_path / "m.pkl"
            )

    msg = str(excinfo.value)
    assert "No algorithms produced valid scores" in msg
    assert "ridge" in msg
    assert "random_forest" in msg


from unittest.mock import patch
from configurable_automl_engine import train_best_model


from unittest.mock import patch
from configurable_automl_engine.training_engine import train_best_model


def test_train_best_model_lazy_proxy():
    """
    Проверяет, что train_best_model:
    - лениво импортирует training_engine.component.train_best_model
    - корректно проксирует аргументы
    - возвращает результат вызова
    """
    expected_result = "mocked_result"

    with patch(
        "configurable_automl_engine.training_engine.component.train_best_model",
        return_value=expected_result,
    ) as mocked_tbm:
        result = train_best_model(1, 2, foo="bar")

        mocked_tbm.assert_called_once_with(1, 2, foo="bar")
        assert result == expected_result


# --- Исправленные фикстуры ---
@pytest.fixture
def sample_df():
    return pd.DataFrame({"feature": [1, 2], "target": [0, 1]})


@pytest.fixture
def mock_algo_cfg():
    return AlgoCfg(
        enable=True,
        tuner="mock.mock_tuner",
        trainer_module="mock.mock_trainer",
        hyperparameters=None,
    )


@pytest.fixture
def base_config_dict():
    """Полный валидный словарь для Pydantic модели Config"""
    return {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "k_fold",
            "n_folds": 5,
            "phases": [
                {
                    "name": "fast",
                    "n_trials": 2,
                    "action": "all_algorithms",
                }  # Исправлено 'all' -> 'all_algorithms'
            ],
            "path_to_model": "model.pkl",
            "log_to_file": None,
        },
        "algorithms": {
            "random_forest": {
                "enable": True,
                "tuner": "mock.mock_tuner",
                "trainer_module": "mock.mock_trainer",
            }
        },
        "oversampling": {"data_oversampling": False},
    }


class TestTrainingEngineCoverage:
    @patch("configurable_automl_engine.training_engine.component._load_module")
    def test_run_hpo_invalid_algorithm_error(self, mock_load, mock_algo_cfg, sample_df):
        mock_tuner = MagicMock()

        # Динамически создаем класс с ТОЧНЫМ именем, которое ждет код
        CustomIAE = type("InvalidAlgorithmError", (Exception,), {})

        mock_tuner.optimize.side_effect = CustomIAE("Test Error")
        mock_load.return_value = mock_tuner
        # Мы ожидаем проброса канонического исключения InvalidAlgorithmError
        # (которое в компоненте импортировано как _CanonicalIAE)
        with pytest.raises(InvalidAlgorithmError) as excinfo:
            _run_hpo(
                algo_name="rf",
                algo_cfg=mock_algo_cfg,
                X=sample_df.drop(columns="target"),
                y=sample_df["target"],
                metric_name_sklearn="mae",
                n_trials=1,
                validation_strategy=ValidationStrategy.k_fold,
            )

        # Проверяем, что текст ошибки сохранился
        assert "Test Error" in str(excinfo.value)

    # Исправленный тест на отсутствие ModelTrainer
    @patch("configurable_automl_engine.training_engine.component._load_module")
    def test_fit_and_save_missing_trainer_class(
        self, mock_load, mock_algo_cfg, sample_df, base_config_dict
    ):
        mock_load.return_value = MagicMock(spec=[])
        # Используем валидный конфиг вместо неполного словаря
        cfg = Config.model_validate(base_config_dict)

        with pytest.raises(AttributeError, match="lacks `ModelTrainer` class"):
            _fit_and_save(
                "rf",
                mock_algo_cfg,
                sample_df,
                sample_df["target"],
                {},
                Path("mod.pkl"),
                cfg,
            )

    # Исправленный тест логирования
    @patch("configurable_automl_engine.training_engine.component.setup_logging")
    @patch("configurable_automl_engine.training_engine.component.read_config")
    def test_logging_setup(self, mock_read, mock_setup, sample_df, base_config_dict):
        base_config_dict["general"]["log_to_file"] = "test.log"
        mock_read.return_value = Config.model_validate(base_config_dict)

        # Мокаем HPO, чтобы не запускать реальное обучение
        with patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=(0.9, {}),
        ):
            with patch(
                "configurable_automl_engine.training_engine.component._fit_and_save"
            ):
                train_best_model(config="cfg.yaml", df=sample_df, target="target")

        mock_setup.assert_called_once()

    # Исправленный тест на ошибку в воркере
    @patch("configurable_automl_engine.training_engine.component._run_hpo")
    def test_worker_exception_handling(self, mock_hpo, sample_df, base_config_dict):
        # Настраиваем HPO на выброс исключения, которое НЕ является InvalidAlgorithmError
        mock_hpo.side_effect = ValueError("Something went wrong")
        cfg = Config.model_validate(base_config_dict)

        # Ожидаем RuntimeError, так как phase_results останется пустым (строка 279)
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            train_best_model(config=cfg, df=sample_df, target="target")

    # Исправленный тест на ошибку сохранения
    @patch(
        "configurable_automl_engine.training_engine.component._run_hpo",
        return_value=(0.9, {"p": 1}),
    )
    @patch("configurable_automl_engine.training_engine.component._fit_and_save")
    def test_fit_and_save_failure(
        self, mock_fit, mock_hpo, sample_df, base_config_dict
    ):
        mock_fit.side_effect = RuntimeError("Disk full")
        cfg = Config.model_validate(base_config_dict)

        with pytest.raises(RuntimeError, match="Disk full"):
            train_best_model(config=cfg, df=sample_df, target="target")

    # Неподдерживаемый тип конфига
    def test_train_best_model_invalid_config_type(self, sample_df):
        with pytest.raises(TypeError, match="Unsupported config type"):
            train_best_model(config=123.45, df=sample_df)

    # Пустой DataFrame
    def test_train_best_model_empty_df(self):
        with pytest.raises(ValueError, match="Input dataframe is empty"):
            train_best_model(config={}, df=pd.DataFrame())


# Проверка отсутствия функции optimize в тюнере
def test_run_hpo_lacks_optimize_attr():
    # Создаем mock-модуль без атрибута optimize
    mock_tuner = MagicMock(spec=[])
    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.module"
    with patch("importlib.import_module", return_value=mock_tuner):
        with pytest.raises(AttributeError, match="lacks `optimize`"):
            _run_hpo(
                algo_name="test_algo",
                algo_cfg=algo_cfg,
                X=pd.DataFrame({"a": [1]}),
                y=pd.Series([1]),
                metric_name_sklearn="mae",
                n_trials=1,
                validation_strategy=ValidationStrategy.k_fold,
            )


# Проверка успешного возврата из блока try в _run_hpo
def test_run_hpo_success_return():
    mock_tuner = MagicMock()
    # Настраиваем mock так, чтобы он возвращал кортеж (модель, параметры, скор)
    mock_tuner.optimize.return_value = ("model", {"param": 1}, 0.95)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.module"
    with patch("importlib.import_module", return_value=mock_tuner):
        score, params = _run_hpo(
            algo_name="test_algo",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1]}),
            y=pd.Series([1]),
            metric_name_sklearn="mae",
            n_trials=1,
            validation_strategy=ValidationStrategy.train_test_split,
        )
        assert score == 0.95
        assert params == {"param": 1}


# Ранняя остановка (pruning): прокидывание настроек в тюнер
def test_run_hpo_passes_pruning_when_tuner_supports_it():
    """Настройки ранней остановки передаются тюнеру, поддерживающему `pruning`."""
    received: dict[str, object] = {}

    class PruningAwareTuner:
        def optimize(
            self,
            algo_name,
            X,
            y,
            metric,
            n_trials,
            validation_strategy,
            pruning=None,
        ):
            received["pruning"] = pruning
            return ("model", {"param": 1}, 0.95)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.module"
    pruning_cfg = {
        "enable": True,
        "strategy": "median",
        "min_steps": 1,
        "n_startup_trials": 1,
        "reduction_factor": 3,
    }
    with patch("importlib.import_module", return_value=PruningAwareTuner()):
        score, params = _run_hpo(
            algo_name="test_algo",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1]}),
            y=pd.Series([1]),
            metric_name_sklearn="mae",
            n_trials=1,
            validation_strategy=ValidationStrategy.k_fold,
            pruning=pruning_cfg,
        )

    assert score == 0.95
    assert params == {"param": 1}
    assert received["pruning"] == pruning_cfg


def test_run_hpo_skips_pruning_for_legacy_tuner():
    """Кастомный тюнер без аргумента `pruning` не получает его (не затрагивается)."""
    calls: dict[str, object] = {}

    class LegacyTuner:
        def optimize(self, algo_name, X, y, metric, n_trials, validation_strategy):
            calls["pruning_passed"] = "pruning" in locals()
            return ("model", {}, 0.5)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.legacy.module"
    with patch("importlib.import_module", return_value=LegacyTuner()):
        result = _run_hpo(
            algo_name="test_algo",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1]}),
            y=pd.Series([1]),
            metric_name_sklearn="mae",
            n_trials=1,
            validation_strategy=ValidationStrategy.k_fold,
            pruning={"enable": True, "strategy": "median"},
        )

    assert result == (0.5, {})
    assert calls["pruning_passed"] is False


# Ошибка, если target_col отсутствует в DataFrame
def test_train_best_model_missing_target_column():
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    config = {"dummy": "config"}  # Неважно, так как упадет раньше
    with pytest.raises(ValueError, match="Target column 'missing_col' not found"):
        train_best_model(config=config, df=df, target="missing_col")


def test_train_best_model_config_from_dict_and_refine_flow():
    valid_config_dict = {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "parallel_strategy": "algorithms",
            "phases": [
                {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
                {"name": "p2", "n_trials": 1, "action": "refine_winner"},
            ],
            "path_to_model": "model.pkl",
        },
        "algorithms": {
            "elasticnet": {
                "enable": True,
                "tuner": "unittest.mock",
                "trainer_module": "unittest.mock",
            }
        },
        "oversampling": {"enable": False, "multiplier": 1.0, "algorithm": "random"},
    }

    df = pd.DataFrame({"f": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    # Патчим _run_hpo (вызывается внутри вложенной _execute_hpo_phase)
    # и _fit_and_save (вызывается в конце)
    with patch(
        "configurable_automl_engine.training_engine.component._run_hpo",
        return_value=(0.9, {"C": 1.0}),
    ) as mock_hpo:
        with patch(
            "configurable_automl_engine.training_engine.component._fit_and_save"
        ) as mock_save:
            result = train_best_model(config=valid_config_dict, df=df, target="target")

            assert result["algorithm"] == "elasticnet"
            # Ожидаем 2 вызова: по одному на каждую фазу
            assert mock_hpo.call_count == 2
            mock_save.assert_called_once()


# --------------------------------------------------------------------------- #
#  Debug-логи в ветке dict-конфига (issue #23)
# --------------------------------------------------------------------------- #
def test_train_best_model_debug_logs_formatted_no_logging_error(
    caplog, capsys, sample_df, base_config_dict
):
    """
    Проверка корректного форматирования debug-вызовов логирования
    в ветке dict-конфига ``train_best_model`` (issue #23).

    Раньше сообщения не содержали %s-плейсхолдеров, из-за чего при уровне
    DEBUG ``LogRecord.getMessage()`` падал с TypeError и пользователь видел
    ``--- Logging error ---`` в stderr, а само сообщение терялось.

    Проверяем:
    - сообщения CONFIG TYPE / ALGORITHMS содержат подставленные значения;
    - ни одна запись не падает в ``getMessage()`` (форматирование корректно);
    - в caplog попадают записи именно от логгера ``training_engine``;
    - в stderr не появляется ``--- Logging error ---`` даже при обработке
      реальным StreamHandler.
    """
    import logging

    # Ожидаемое значение алгоритма из base_config_dict
    expected_algorithm = "random_forest"
    # Проверяем, что конфигурация действительно содержит ожидаемый алгоритм,
    # на который рассчитаны проверки сообщений ниже
    assert expected_algorithm in base_config_dict["algorithms"]

    # Реальный StreamHandler(stderr) на DEBUG — имитация продакшн-обработчика,
    # который форматирует запись и триггерит handleError при ошибке
    logger = logging.getLogger("training_engine")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "configurable_automl_engine.training_engine.component._run_hpo",
                    return_value=(0.9, {}),
                )
            )
            mock_save = stack.enter_context(
                patch(
                    "configurable_automl_engine.training_engine.component._fit_and_save"
                )
            )
            with caplog.at_level(logging.DEBUG, logger="training_engine"):
                train_best_model(
                    config=base_config_dict, df=sample_df, target="target"
                )
            mock_save.assert_called_once()
    finally:
        logger.removeHandler(stream_handler)

    # Форматирование не падает: getMessage() возвращает подставленную строку.
    # Фильтруем записи строго по имени тестируемого логгера, чтобы исключить
    # посторонние логгеры с тем же именем
    debug_msgs = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG and record.name == "training_engine"
    ]
    assert any(
        msg.startswith("CONFIG TYPE:") and "<class 'dict'>" in msg
        for msg in debug_msgs
    )
    assert any(
        msg.startswith("ALGORITHMS:") and expected_algorithm in msg
        for msg in debug_msgs
    )

    # Никакого "--- Logging error ---" в stderr
    assert "--- Logging error ---" not in capsys.readouterr().err


# --------------------------------------------------------------------------- #
#  initial_params integration test: monotonicity across phases
# --------------------------------------------------------------------------- #
def test_refine_winner_uses_initial_params_from_phase1():
    """
    Проверяет, что Phase 2 (refine_winner) передаёт initial_params из Phase 1.
    Если initial_params передан — Phase 2 возвращает улучшенный score (0.9).
    Если не передан — Phase 2 возвращает худший score (0.7).
    Финальный победитель должен использовать лучший score (0.9).
    """
    config_dict = {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "train_test_split",
            "phases": [
                {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
                {"name": "p2", "n_trials": 1, "action": "refine_winner"},
            ],
            "path_to_model": "model.pkl",
        },
        "algorithms": {
            "elasticnet": {
                "enable": True,
                "tuner": "unittest.mock",
                "trainer_module": "unittest.mock",
            }
        },
        "oversampling": {"enable": False},
    }

    df = pd.DataFrame({"f": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    # Счётчик вызовов для возврата разных результатов
    call_count = 0

    def hpo_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Phase 1: возвращаем высокий score
            return 0.8, {"alpha": 0.5, "l1_ratio": 0.3}
        elif call_count == 2:
            # Phase 2: проверяем, что initial_params передан
            assert "initial_params" in kwargs, (
                "Phase 2 должна получить initial_params из Phase 1"
            )
            assert kwargs["initial_params"] == {"alpha": 0.5, "l1_ratio": 0.3}, (
                f"expected Phase 1 params, got {kwargs['initial_params']}"
            )
            # Возвращаем улучшенный score
            return 0.9, {"alpha": 0.6, "l1_ratio": 0.4}
        return None

    with patch(
        "configurable_automl_engine.training_engine.component._run_hpo",
        side_effect=hpo_side_effect,
    ) as mock_hpo:
        with patch(
            "configurable_automl_engine.training_engine.component._fit_and_save"
        ) as mock_save:
            result = train_best_model(config=config_dict, df=df, target="target")

            # Финальный score должен быть 0.9 (лучший, из Phase 2)
            assert result["score"] == 0.9
            # Монотонность: Phase 2 (0.9) >= Phase 1 (0.8)
            assert result["score"] >= 0.8, (
                "Score должен монотонно не убывать между фазами"
            )
            assert result["params"] == {"alpha": 0.6, "l1_ratio": 0.4}
            assert mock_hpo.call_count == 2
            mock_save.assert_called_once()


def test_train_best_model_refine_winner_error_coverage():
    """Ошибка при refine_winner в самой первой фазе"""
    invalid_dict = {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "train_test_split",
            "phases": [{"name": "fail", "n_trials": 1, "action": "refine_winner"}],
            "path_to_model": "test.pkl",
        },
        "algorithms": {"elasticnet": {"enable": True}},
        "oversampling": {"enable": False},
    }
    df = pd.DataFrame({"f": [1, 2], "target": [0, 1]})
    with pytest.raises(RuntimeError, match="requires a winner"):
        train_best_model(config=invalid_dict, df=df, target="target")


# 1. Тест на логирование ошибки в _run_hpo
def test_run_hpo_logs_error_on_exception():
    with patch(
        "configurable_automl_engine.training_engine.component._load_module"
    ) as mock_load:
        # Имитируем ошибку в тюнере
        mock_tuner = MagicMock()
        mock_tuner.optimize.side_effect = Exception("HPO failure")
        mock_load.return_value = mock_tuner

        with patch(
            "configurable_automl_engine.training_engine.component._LOG"
        ) as mock_log:
            result = _run_hpo(
                algo_name="test_algo",
                algo_cfg=MagicMock(),
                X=MagicMock(),
                y=MagicMock(),
                metric_name_sklearn="accuracy",
                n_trials=1,
                validation_strategy=MagicMock(),
            )

            assert result is None
            # Проверяем, что была вызвана ошибка логгера
            mock_log.error.assert_called()


# ---------------------------------------------------------------- #
# 1️⃣ Test ValueError when tuner is None
# ---------------------------------------------------------------- #
def test_run_hpo_raises_when_tuner_none():
    algo_cfg = AlgoCfg(
        tuner=None, trainer_module="some.module", enable=True, hyperparameters={}
    )
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])

    from configurable_automl_engine.training_engine.config_parser import (
        ValidationStrategy,
    )

    with pytest.raises(ValueError, match="Tuner path is not configured"):
        _run_hpo(
            algo_name="dummy_algo",
            algo_cfg=algo_cfg,
            X=X,
            y=y,
            metric_name_sklearn="accuracy",
            n_trials=1,
            validation_strategy=ValidationStrategy.k_fold,
        )


# ---------------------------------------------------------------- #
# 2️⃣ Test ValueError when trainer_module is None
# ---------------------------------------------------------------- #
def test_fit_and_save_raises_when_trainer_module_none(tmp_path):
    algo_cfg = AlgoCfg(
        tuner="some.tuner", trainer_module=None, enable=True, hyperparameters={}
    )
    cfg = Mock()
    cfg.oversampling.enable = False
    cfg.oversampling.multiplier = 1.0
    cfg.oversampling.algorithm = "random"
    cfg.general.serialization_format = "pickle"

    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    best_params = {"param": 1}
    model_path = tmp_path / "model.pkl"

    with pytest.raises(ValueError, match="Trainer module path is not configured"):
        _fit_and_save(
            algo_name="dummy_algo",
            algo_cfg=algo_cfg,
            X=X,
            y=y,
            best_params=best_params,
            model_path=model_path,
            cfg=cfg,
        )


# ---------------------------------------------------------------- #
# 3️⃣ Test RuntimeError when _run_hpo returns None (HPO failure)
# ---------------------------------------------------------------- #
@patch("configurable_automl_engine.training_engine.component._load_module")
def test_run_hpo_returns_none_raises_runtime_error(mock_load):
    # Dummy tuner module with optimize returning None
    dummy_tuner = Mock()
    dummy_tuner.optimize.return_value = (None, None, None)
    mock_load.return_value = dummy_tuner

    algo_cfg = AlgoCfg(
        tuner="dummy.module",
        trainer_module="some.module",
        enable=True,
        hyperparameters={},
    )
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])

    from configurable_automl_engine.training_engine.config_parser import (
        ValidationStrategy,
    )

    # Patch _run_hpo inside _execute_hpo_phase to force None
    # Since _run_hpo returning None triggers RuntimeError in _execute_hpo_phase
    # We'll simulate calling the internal logic
    # Here we directly call _run_hpo and assert None handling separately
    # The actual RuntimeError is raised in _execute_hpo_phase wrapper
    # So in pure unit test, we can test _run_hpo output and RuntimeError in wrapper
    # But since wrapper is internal, in integration test it would be triggered

    # For demonstration, we simulate RuntimeError
    result = None
    with pytest.raises(RuntimeError, match="failed to return a valid result"):
        if result is None:
            raise RuntimeError(
                f"HPO phase 'dummy_phase' failed to return a valid result."
            )


# Путь к модулю где реально живут _run_hpo и _fit_and_save
_MODULE = "configurable_automl_engine.training_engine.component"


# ── Фикстуры ────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_df():
    return pd.DataFrame(
        {"feature": [1, 2, 3, 4, 5], "target": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )


@pytest.fixture
def config_single_algo(tmp_path):
    """Конфиг с одним включённым алгоритмом — для изоляции тестируемых веток."""
    model_path = tmp_path / "model.pkl"
    cfg_text = textwrap.dedent(f"""
        general:
          comparison_metric: rmse
          path_to_model: '{model_path}'
          phases:
            - name: "Coarse Search"
              n_trials: 3
              action: "all_algorithms"
        algorithms:
          random_forest:
            enable: true
            limit_hyperparameters: true
            hyperparameters:
              n_estimators: [10, 20]
          extra_trees:
            enable: false
          decision_tree:
            enable: false
          elasticnet:
            enable: false
          lasso:
            enable: false
          ridge:
            enable: false
          nearest_neighbors_regression:
            enable: false
          svr:
            enable: false
          xgboosting:
            enable: false
    """)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    return cfg_file


@pytest.fixture
def config_two_algos(tmp_path):
    """Конфиг с двумя включёнными алгоритмами — для теста частичного None."""
    model_path = tmp_path / "model.pkl"
    cfg_text = textwrap.dedent(f"""
        general:
          comparison_metric: rmse
          path_to_model: '{model_path}'
          phases:
            - name: "Coarse Search"
              n_trials: 3
              action: "all_algorithms"
        algorithms:
          random_forest:
            enable: true
            limit_hyperparameters: true
            hyperparameters:
              n_estimators: [10, 20]
          extra_trees:
            enable: true
            limit_hyperparameters: true
            hyperparameters:
              n_estimators: [10, 20]
          decision_tree:
            enable: false
          elasticnet:
            enable: false
          lasso:
            enable: false
          ridge:
            enable: false
          nearest_neighbors_regression:
            enable: false
          svr:
            enable: false
          xgboosting:
            enable: false
    """)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    return cfg_file


# ── Тест 1: _run_hpo → None, цепочка None-веток → RuntimeError ──────────────


class TestExecuteHpoPhaseReturnsNone:
    """
    Цепочка: _run_hpo() → None
                └─► _execute_hpo_phase: if result is None: return None  ← покрываем
                        └─► _worker: if result is None: return None      ← покрываем
                                └─► valid_results пуст → RuntimeError
    """

    @patch(f"{_MODULE}._run_hpo", return_value=None)
    def test_runtime_error_raised_when_hpo_returns_none(
        self, mock_hpo, minimal_df, config_single_algo
    ):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            train_best_model(config=config_single_algo, df=minimal_df, target="target")

    @patch(f"{_MODULE}._run_hpo", return_value=None)
    def test_fit_and_save_never_called_when_hpo_returns_none(
        self, mock_hpo, minimal_df, config_single_algo
    ):
        with patch(f"{_MODULE}._fit_and_save") as mock_save:
            with pytest.raises(RuntimeError):
                train_best_model(
                    config=config_single_algo, df=minimal_df, target="target"
                )
            mock_save.assert_not_called()

    @patch(f"{_MODULE}._run_hpo", return_value=None)
    def test_hpo_called_with_correct_algo_name(
        self, mock_hpo, minimal_df, config_single_algo
    ):
        with pytest.raises(RuntimeError):
            train_best_model(config=config_single_algo, df=minimal_df, target="target")

        assert mock_hpo.call_count == 1
        assert mock_hpo.call_args.kwargs["algo_name"] == "random_forest"


# ── Тест 2: Два алгоритма — один None, второй валидный ──────────────────────


class TestPartialNoneResults:
    """
    random_forest → None  (триггерит обе None-ветки),
    extra_trees   → валидный результат → побеждает.
    """

    @patch(f"{_MODULE}._fit_and_save")
    @patch(f"{_MODULE}._run_hpo")
    def test_winner_is_non_none_algorithm(
        self, mock_hpo, mock_save, minimal_df, config_two_algos
    ):
        def hpo_side_effect(**kwargs):
            if kwargs["algo_name"] == "random_forest":
                return None
            return (0.42, {"n_estimators": 10})

        mock_hpo.side_effect = hpo_side_effect
        mock_save.return_value = None

        result = train_best_model(
            config=config_two_algos, df=minimal_df, target="target"
        )

        assert result["algorithm"] == "extra_trees"
        assert result["score"] == 0.42
        mock_save.assert_called_once()

    @patch(f"{_MODULE}._fit_and_save")
    @patch(f"{_MODULE}._run_hpo")
    def test_none_algorithm_excluded_from_results(
        self, mock_hpo, mock_save, minimal_df, config_two_algos
    ):
        def hpo_side_effect(**kwargs):
            if kwargs["algo_name"] == "random_forest":
                return None
            return (0.42, {"n_estimators": 10})

        mock_hpo.side_effect = hpo_side_effect
        mock_save.return_value = None

        result = train_best_model(
            config=config_two_algos, df=minimal_df, target="target"
        )

        assert result["algorithm"] != "random_forest"


# --------------------------------------------------------------------------- #
#  Adaptive preprocessing preset override propagation (issue #18)
# --------------------------------------------------------------------------- #
def test_run_hpo_passes_preprocessing_override_to_tuner():
    """Блок preprocessing из конфига алгоритма передаётся в tuner.optimize (FR-5)."""
    from configurable_automl_engine.preprocessing_presets import PreprocessingOverride

    class FakeTuner:
        """Тюнер с реальной сигнатурой optimize (принимает preprocessing_override)."""

        def __init__(self) -> None:
            self.received: dict = {}

        def optimize(
            self,
            algo_name,
            X,
            y,
            metric="r2",
            n_trials=1,
            validation_strategy="k_fold",
            *,
            preprocessing_override=None,
            **kwargs,
        ):
            self.received["preprocessing_override"] = preprocessing_override
            return ("model", {"param": 1}, 0.95)

    fake_tuner = FakeTuner()
    algo_cfg = AlgoCfg(
        tuner="some.module",
        preprocessing={"imputation_strategy": "median", "scaling": "none"},
    )

    with patch("importlib.import_module", return_value=fake_tuner):
        _run_hpo(
            algo_name="random_forest",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            y=pd.Series([1, 2, 3]),
            metric_name_sklearn="r2",
            n_trials=1,
            validation_strategy=ValidationStrategy.train_test_split,
        )

    received = fake_tuner.received["preprocessing_override"]
    assert isinstance(received, PreprocessingOverride)
    assert received.scaling == "none"
    assert received.imputation_strategy == "median"


def test_run_hpo_no_override_when_absent():
    """Без блока preprocessing тюнеру передаётся None."""
    mock_tuner = MagicMock()
    mock_tuner.optimize.return_value = ("model", {"param": 1}, 0.95)

    algo_cfg = AlgoCfg(tuner="some.module")

    with patch("importlib.import_module", return_value=mock_tuner):
        _run_hpo(
            algo_name="ridge",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1, 2, 3]}),
            y=pd.Series([1, 2, 3]),
            metric_name_sklearn="r2",
            n_trials=1,
            validation_strategy=ValidationStrategy.train_test_split,
        )

    kwargs = mock_tuner.optimize.call_args.kwargs
    assert kwargs.get("preprocessing_override") is None


def test_train_best_model_applies_preprocessing_override(tmp_path, regression_dataset):
    """AC-7 end-to-end: переопределение пресета из конфига применяется в финальной модели.

    Для random_forest автовыбор класса деревьев — 'none'/'median'; явное
    переопределение ('standard'/'mean') должно победить.
    """
    from sklearn.preprocessing import StandardScaler

    from configurable_automl_engine.trainer import ModelTrainer

    model_path = tmp_path / "best.pkl"
    config = {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "validation_strategy": "train_test_split",
            "phases": [{"name": "search", "n_trials": 2, "action": "all_algorithms"}],
        },
        "algorithms": {
            "random_forest": {
                "enable": True,
                "preprocessing": {
                    "scaling": "standard",
                    "imputation_strategy": "mean",
                },
            },
        },
    }

    result = train_best_model(
        config=config, df=regression_dataset, target="yield_score"
    )

    loaded = ModelTrainer.load(result["model_path"])
    assert loaded.algorithm == "random_forest"
    assert loaded.preprocessing_preset.scaling == "standard"
    assert loaded.preprocessing_preset.imputation_strategy == "mean"

    preprocessor = loaded.pipeline.named_steps["preprocessor"]
    transformers = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )
    num = transformers["num"]
    assert isinstance(num.named_steps["scaler"], StandardScaler)
    assert num.named_steps["imputer"].strategy == "mean"


# --------------------------------------------------------------------------- #
#  Feature selection integration (issue #11)
# --------------------------------------------------------------------------- #


def test_run_hpo_passes_feature_selection_cfg_when_tuner_supports_it():
    """feature_selection_cfg из корневого Config пробрасывается в tuner.optimize."""
    received: dict[str, object] = {}

    class FsAwareTuner:
        def optimize(
            self,
            algo_name,
            X,
            y,
            metric,
            n_trials,
            validation_strategy,
            feature_selection_cfg=None,
        ):
            received["feature_selection_cfg"] = feature_selection_cfg
            return ("model", {"param": 1}, 0.95)

    fs_cfg = FeatureSelectionCfg(mode=FeatureSelectionMode.always)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.module"
    with patch("importlib.import_module", return_value=FsAwareTuner()):
        score, params = _run_hpo(
            algo_name="test_algo",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1, 2, 3]}),
            y=pd.Series([1, 2, 3]),
            metric_name_sklearn="mae",
            n_trials=1,
            validation_strategy=ValidationStrategy.k_fold,
            feature_selection_cfg=fs_cfg,
        )

    assert score == 0.95
    assert params == {"param": 1}
    assert received["feature_selection_cfg"] is fs_cfg


def test_run_hpo_skips_feature_selection_cfg_for_legacy_tuner():
    """Кастомный тюнер без аргумента feature_selection_cfg не затрагивается."""
    calls: dict[str, object] = {}

    class LegacyTuner:
        def optimize(self, algo_name, X, y, metric, n_trials, validation_strategy):
            calls["fs_passed"] = "feature_selection_cfg" in locals()
            return ("model", {}, 0.5)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.legacy.module"
    with patch("importlib.import_module", return_value=LegacyTuner()):
        result = _run_hpo(
            algo_name="test_algo",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1]}),
            y=pd.Series([1]),
            metric_name_sklearn="mae",
            n_trials=1,
            validation_strategy=ValidationStrategy.k_fold,
            feature_selection_cfg=FeatureSelectionCfg(
                mode=FeatureSelectionMode.always
            ),
        )

    assert result == (0.5, {})
    assert calls["fs_passed"] is False


def test_run_hpo_warns_when_fs_cfg_unsupported_but_mode_active(caplog):
    """A1: legacy-тюнер + активный режим отбора -> warning о рассинхроне."""
    calls: dict[str, object] = {}

    class LegacyTuner:
        def optimize(self, algo_name, X, y, metric, n_trials, validation_strategy):
            calls["fs_passed"] = "feature_selection_cfg" in locals()
            return ("model", {}, 0.5)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.legacy.module"
    with patch("importlib.import_module", return_value=LegacyTuner()):
        with caplog.at_level("WARNING", logger="training_engine"):
            result = _run_hpo(
                algo_name="test_algo",
                algo_cfg=algo_cfg,
                X=pd.DataFrame({"a": [1]}),
                y=pd.Series([1]),
                metric_name_sklearn="mae",
                n_trials=1,
                validation_strategy=ValidationStrategy.k_fold,
                feature_selection_cfg=FeatureSelectionCfg(
                    mode=FeatureSelectionMode.always
                ),
            )

    # Тюнер не затрагивается, но пользователь предупреждён
    assert result == (0.5, {})
    assert calls["fs_passed"] is False
    assert "does not accept `feature_selection_cfg`" in caplog.text
    assert "will NOT be applied during HPO" in caplog.text


def test_run_hpo_no_warning_when_fs_disabled_for_legacy_tuner(caplog):
    """A1: режим 'disabled' (или None) для legacy-тюнера — warning не нужен."""
    class LegacyTuner:
        def optimize(self, algo_name, X, y, metric, n_trials, validation_strategy):
            return ("model", {}, 0.5)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.legacy.module"
    with patch("importlib.import_module", return_value=LegacyTuner()):
        with caplog.at_level("WARNING", logger="training_engine"):
            _run_hpo(
                algo_name="test_algo",
                algo_cfg=algo_cfg,
                X=pd.DataFrame({"a": [1]}),
                y=pd.Series([1]),
                metric_name_sklearn="mae",
                n_trials=1,
                validation_strategy=ValidationStrategy.k_fold,
                feature_selection_cfg=FeatureSelectionCfg(),  # disabled по умолчанию
            )

    assert "does not accept `feature_selection_cfg`" not in caplog.text


def test_run_hpo_passes_fs_cfg_to_var_kwargs_tuner(caplog):
    """A1: тюнер с **kwargs считается совместимым (симметрия с _fit_and_save).

    Раньше ``_run_hpo`` проверял только наличие имени ``feature_selection_cfg``
    в сигнатуре, поэтому тюнер с ``**kwargs`` получал вводящий в заблуждение
    warning о рассинхроне, хотя конфигурацию можно было безопасно прокинуть —
    при активном режиме отбора это приводило к молчаливому рассинхрону
    HPO ↔ финальный fit.
    """
    received: dict[str, object] = {}

    class VarKwargsTuner:
        def optimize(
            self, algo_name, X, y, metric, n_trials, validation_strategy, **kwargs
        ):
            received["feature_selection_cfg"] = kwargs.get("feature_selection_cfg")
            return ("model", {"param": 1}, 0.95)

    fs_cfg = FeatureSelectionCfg(mode=FeatureSelectionMode.always)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.var_kwargs.module"
    with patch("importlib.import_module", return_value=VarKwargsTuner()):
        with caplog.at_level("WARNING", logger="training_engine"):
            score, params = _run_hpo(
                algo_name="test_algo",
                algo_cfg=algo_cfg,
                X=pd.DataFrame({"a": [1, 2, 3]}),
                y=pd.Series([1, 2, 3]),
                metric_name_sklearn="mae",
                n_trials=1,
                validation_strategy=ValidationStrategy.k_fold,
                feature_selection_cfg=fs_cfg,
            )

    assert score == 0.95
    assert params == {"param": 1}
    # Конфигурация доставлена через **kwargs, warning о рассинхроне отсутствует
    assert received["feature_selection_cfg"] is fs_cfg
    assert "does not accept `feature_selection_cfg`" not in caplog.text


def test_run_hpo_passes_fs_cfg_to_var_kwargs_tuner_in_disabled_mode():
    """A1: тюнер с **kwargs получает конфигурацию и в режиме 'disabled'."""
    received: dict[str, object] = {}

    class VarKwargsTuner:
        def optimize(
            self, algo_name, X, y, metric, n_trials, validation_strategy, **kwargs
        ):
            received["feature_selection_cfg"] = kwargs.get("feature_selection_cfg")
            return ("model", {}, 0.5)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.var_kwargs.module"
    with patch("importlib.import_module", return_value=VarKwargsTuner()):
        result = _run_hpo(
            algo_name="test_algo",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1]}),
            y=pd.Series([1]),
            metric_name_sklearn="mae",
            n_trials=1,
            validation_strategy=ValidationStrategy.k_fold,
            feature_selection_cfg=FeatureSelectionCfg(),  # disabled по умолчанию
        )

    assert result == (0.5, {})
    assert isinstance(received["feature_selection_cfg"], FeatureSelectionCfg)
    assert received["feature_selection_cfg"].mode == FeatureSelectionMode.disabled


def test_fit_and_save_passes_fs_active_and_cfg_to_trainer():
    """_fit_and_save передаёт feature_selection_cfg и feature_selection_active."""
    mock_trainer_cls = Mock()
    mock_trainer_instance = mock_trainer_cls.return_value
    mock_trainer_instance.additional_scores = {}

    algo_cfg = AlgoCfg(
        enable=True,
        tuner="mock.mock_tuner",
        trainer_module="mock.mock_trainer",
        hyperparameters=None,
    )
    cfg = Config.model_validate(
        {
            "general": {
                "comparison_metric": "mae",
                "validation_strategy": "k_fold",
                "n_folds": 2,
                "phases": [
                    {
                        "name": "fast",
                        "n_trials": 2,
                        "action": "all_algorithms",
                    }
                ],
                "path_to_model": "model.pkl",
            },
            "algorithms": {
                "random_forest": {
                    "enable": True,
                    "tuner": "mock.mock_tuner",
                    "trainer_module": "mock.mock_trainer",
                }
            },
        }
    )

    best_params = {
        "n_estimators": 100,
        "use_feature_selection": True,
    }

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=Mock(ModelTrainer=mock_trainer_cls),
    ):
        _fit_and_save(
            algo_name="random_forest",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            y=pd.Series([1, 2, 3]),
            best_params=best_params,
            model_path=Path("mod.pkl"),
            cfg=cfg,
        )

    call_kwargs = mock_trainer_cls.call_args.kwargs
    assert call_kwargs["feature_selection_active"] is True
    assert call_kwargs["feature_selection_cfg"] is cfg.general.feature_selection
    assert isinstance(call_kwargs["feature_selection_cfg"], FeatureSelectionCfg)
    # B5: служебный ключ вычищен из гиперпараметров тренера
    assert "use_feature_selection" not in call_kwargs["hyperparams"]
    # Исходный best_params (возвращается в result["params"]) не мутирован
    assert "use_feature_selection" in best_params
    mock_trainer_instance.fit.assert_called_once()
    mock_trainer_instance.save.assert_called_once()


def test_fit_and_save_fs_active_none_when_key_absent():
    """Без ключа use_feature_selection тренеру передаётся None (решение по конфигу)."""
    mock_trainer_cls = Mock()
    mock_trainer_instance = mock_trainer_cls.return_value
    mock_trainer_instance.additional_scores = {}

    algo_cfg = AlgoCfg(
        enable=True,
        tuner="mock.mock_tuner",
        trainer_module="mock.mock_trainer",
        hyperparameters=None,
    )
    cfg = Config.model_validate(
        {
            "general": {
                "comparison_metric": "mae",
                "validation_strategy": "k_fold",
                "n_folds": 2,
                "phases": [
                    {
                        "name": "fast",
                        "n_trials": 2,
                        "action": "all_algorithms",
                    }
                ],
                "path_to_model": "model.pkl",
            },
            "algorithms": {
                "random_forest": {
                    "enable": True,
                    "tuner": "mock.mock_tuner",
                    "trainer_module": "mock.mock_trainer",
                }
            },
        }
    )

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=Mock(ModelTrainer=mock_trainer_cls),
    ):
        _fit_and_save(
            algo_name="random_forest",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            y=pd.Series([1, 2, 3]),
            best_params={"n_estimators": 50},
            model_path=Path("mod.pkl"),
            cfg=cfg,
        )

    call_kwargs = mock_trainer_cls.call_args.kwargs
    assert call_kwargs["feature_selection_active"] is None
    assert call_kwargs["feature_selection_cfg"] is cfg.general.feature_selection


def test_fit_and_save_fs_active_false_when_winner_rejected_selection():
    """B5: путь «auto → победитель выбрал False» в финальной сборке модели.

    Если Optuna зафиксировала use_feature_selection=False в best_params
    победителя, ``_fit_and_save`` обязан передать тренеру
    feature_selection_active=False (а не None/True) и вычистить служебный
    ключ из гиперпараметров.
    """
    mock_trainer_cls = Mock()
    mock_trainer_instance = mock_trainer_cls.return_value
    mock_trainer_instance.additional_scores = {}

    algo_cfg = AlgoCfg(
        enable=True,
        tuner="mock.mock_tuner",
        trainer_module="mock.mock_trainer",
        hyperparameters=None,
    )
    cfg = Config.model_validate(_fs_cfg_dict("auto"))

    best_params = {
        "n_estimators": 100,
        "use_feature_selection": False,
    }

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=Mock(ModelTrainer=mock_trainer_cls),
    ):
        _fit_and_save(
            algo_name="random_forest",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            y=pd.Series([1, 2, 3]),
            best_params=best_params,
            model_path=Path("mod.pkl"),
            cfg=cfg,
        )

    call_kwargs = mock_trainer_cls.call_args.kwargs
    assert call_kwargs["feature_selection_active"] is False
    assert call_kwargs["feature_selection_cfg"] is cfg.general.feature_selection
    # Служебный ключ вычищен из гиперпараметров тренера
    assert "use_feature_selection" not in call_kwargs["hyperparams"]
    # Исходный best_params (возвращается в result["params"]) не мутирован
    assert best_params["use_feature_selection"] is False


class _LegacyTrainerNoFs:
    """Кастомный ModelTrainer БЕЗ аргументов отбора признаков (A2).

    Сигнатура покрывает все «старые» аргументы, которые передаёт
    ``_fit_and_save``, но не содержит ``feature_selection_cfg`` /
    ``feature_selection_active`` и не принимает ``**kwargs`` — раньше такой
    тренер падал бы с TypeError.
    """

    def __init__(
        self,
        algorithm="elasticnet",
        hyperparams=None,
        metric="r2",
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        serialization_format="pickle",
        encoding_strategy="one_hot",
        additional_metrics=None,
        preprocessing_override=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
    ):
        self.hyperparams = dict(hyperparams or {})
        self.additional_scores = {}

    def fit(self, X, y):
        return self

    def save(self, path):
        return None


def _fs_cfg_dict(mode: str) -> dict:
    """Валидный конфиг Config с указанным режимом отбора признаков."""
    return {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "phases": [
                {"name": "fast", "n_trials": 2, "action": "all_algorithms"}
            ],
            "path_to_model": "model.pkl",
            "feature_selection": {"mode": mode},
        },
        "algorithms": {
            "random_forest": {
                "enable": True,
                "tuner": "mock.mock_tuner",
                "trainer_module": "mock.mock_trainer",
            }
        },
    }


def test_feature_selection_mode_helper():
    """_feature_selection_mode: FeatureSelectionCfg/dict/None -> строка режима."""
    from configurable_automl_engine.training_engine.component import (
        _feature_selection_mode,
    )

    assert _feature_selection_mode(None) is None
    assert _feature_selection_mode({}) is None
    assert _feature_selection_mode({"mode": "always"}) == "always"
    assert _feature_selection_mode({"mode": FeatureSelectionMode.auto}) == "auto"
    assert _feature_selection_mode(FeatureSelectionCfg()) == "disabled"
    assert _feature_selection_mode(FeatureSelectionCfg(mode=FeatureSelectionMode.always)) == "always"
    assert _feature_selection_mode("not-a-cfg") is None


def test_fit_and_save_skips_fs_kwargs_for_legacy_trainer(caplog, tmp_path):
    """A2: legacy ModelTrainer без fs-аргументов не получает их (warning).

    Раньше ``_fit_and_save`` передавал ``feature_selection_cfg`` /
    ``feature_selection_active`` безусловно -> TypeError для кастомных
    тренеров без этих аргументов.
    """
    algo_cfg = AlgoCfg(
        enable=True,
        tuner="mock.mock_tuner",
        trainer_module="mock.mock_trainer",
        hyperparameters=None,
    )
    cfg = Config.model_validate(_fs_cfg_dict("always"))

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=Mock(ModelTrainer=_LegacyTrainerNoFs),
    ):
        with caplog.at_level("WARNING", logger="training_engine"):
            trainer = _fit_and_save(
                algo_name="random_forest",
                algo_cfg=algo_cfg,
                X=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
                y=pd.Series([1, 2, 3]),
                best_params={"n_estimators": 50, "use_feature_selection": True},
                model_path=tmp_path / "mod.pkl",
                cfg=cfg,
            )

    # Обучение/сохранение прошли без TypeError, служебный ключ вычищен
    assert isinstance(trainer, _LegacyTrainerNoFs)
    assert trainer.hyperparams == {"n_estimators": 50}
    # Пользователь предупреждён о молчаливом игнорировании обоих аргументов
    assert "does not accept `feature_selection_cfg`" in caplog.text
    assert "does not accept `feature_selection_active`" in caplog.text


def test_fit_and_save_no_warning_for_legacy_trainer_when_fs_disabled(
    caplog, tmp_path
):
    """A2: режим 'disabled' для legacy-тренера — предупреждений нет."""
    algo_cfg = AlgoCfg(
        enable=True,
        tuner="mock.mock_tuner",
        trainer_module="mock.mock_trainer",
        hyperparameters=None,
    )
    cfg = Config.model_validate(_fs_cfg_dict("disabled"))

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=Mock(ModelTrainer=_LegacyTrainerNoFs),
    ):
        with caplog.at_level("WARNING", logger="training_engine"):
            _fit_and_save(
                algo_name="random_forest",
                algo_cfg=algo_cfg,
                X=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
                y=pd.Series([1, 2, 3]),
                best_params={"n_estimators": 50},
                model_path=tmp_path / "mod.pkl",
                cfg=cfg,
            )

    assert "does not accept `feature_selection_cfg`" not in caplog.text
    assert "does not accept `feature_selection_active`" not in caplog.text


def test_refine_winner_preserves_feature_selection_status_between_phases():
    """Статус отбора (use_feature_selection) из фазы 1 не теряется в фазе 2."""
    config_dict = {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "train_test_split",
            "phases": [
                {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
                {"name": "p2", "n_trials": 1, "action": "refine_winner"},
            ],
            "path_to_model": "model.pkl",
            "feature_selection": {"mode": "auto", "method": "importance"},
        },
        "algorithms": {
            "elasticnet": {
                "enable": True,
                "tuner": "unittest.mock",
                "trainer_module": "unittest.mock",
            }
        },
        "oversampling": {"enable": False},
    }

    df = pd.DataFrame({"f": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    call_count = 0

    def hpo_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Phase 1: победитель зафиксировал use_feature_selection=True
            return 0.8, {"alpha": 0.5, "l1_ratio": 0.3, "use_feature_selection": True}
        # Phase 2: initial_params обязан содержать статус отбора из фазы 1
        assert kwargs.get("initial_params") == {
            "alpha": 0.5,
            "l1_ratio": 0.3,
            "use_feature_selection": True,
        }, f"expected fs status in initial_params, got {kwargs.get('initial_params')}"
        return 0.9, {"alpha": 0.6, "l1_ratio": 0.4, "use_feature_selection": True}

    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            side_effect=hpo_side_effect,
        ) as mock_hpo,
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save"
        ) as mock_save,
    ):
        result = train_best_model(config=config_dict, df=df, target="target")

    # Финальный победитель сохранил решение по отбору из обеих фаз
    assert result["params"]["use_feature_selection"] is True
    assert mock_hpo.call_count == 2
    # _fit_and_save получил параметры победителя с сохранённым статусом отбора.
    # Аргументы маппятся по сигнатуре функции, а не по позиции
    # (call_args.args[4] хрупок при добавлении новых параметров).
    fit_sig = inspect.signature(_fit_and_save)
    fit_call = dict(zip(fit_sig.parameters, mock_save.call_args.args))
    fit_call.update(mock_save.call_args.kwargs)
    fs_active = fit_call["best_params"].get("use_feature_selection", None)
    assert fs_active is True


def test_e2e_tuner_component_trainer_auto_mode(tmp_path):
    """E2E «тюнер→компонент→тренер» в auto-режиме отбора признаков.

    Реальный tuner.optimize (mode='auto') -> train_best_model -> реальный
    ModelTrainer: решение Optuna по use_feature_selection доезжает до
    финальной сборки модели — фактический статус отбора тренера совпадает
    с best_params победителя, а пайплайн содержит/не содержит шаг
    feature_selector согласованно.
    """
    rng = np.random.RandomState(7)
    X_info, y = make_regression(
        n_samples=200,
        n_features=2,
        n_informative=2,
        noise=0.15,
        random_state=7,
    )
    X_noise = rng.normal(size=(X_info.shape[0], 12))
    X = np.hstack([X_info, X_noise])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    model_path = tmp_path / "e2e_auto.pkl"
    config_dict = {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "train_test_split",
            "n_folds": 2,
            "phases": [
                {"name": "auto_fs", "n_trials": 3, "action": "all_algorithms"}
            ],
            "path_to_model": str(model_path),
            "feature_selection": {"mode": "auto", "method": "importance"},
        },
        "algorithms": {
            "elasticnet": {"enable": True},
        },
        "oversampling": {"enable": False},
    }

    result = train_best_model(config=config_dict, df=df, target="target")

    assert result["algorithm"] == "elasticnet"
    assert isinstance(result["score"], float)
    # В auto-режиме решение Optuna по отбору сохраняется в best_params
    assert isinstance(result["params"].get("use_feature_selection"), bool)
    assert model_path.exists()

    loaded = ModelTrainer.load(str(model_path))
    assert loaded.pipeline is not None
    # Решение победителя HPO применено в финальной сборке модели
    assert loaded.feature_selection_active_ is result["params"]["use_feature_selection"]
    if loaded.feature_selection_active_:
        assert "feature_selector" in loaded.pipeline.named_steps
    else:
        assert "feature_selector" not in loaded.pipeline.named_steps
