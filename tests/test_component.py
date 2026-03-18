import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from configurable_automl_engine.training_engine.component import (
    _run_hpo,
    _fit_and_save,
    train_best_model
)
from configurable_automl_engine.training_engine.config_parser import (
    Config, AlgoCfg, ValidationStrategy
)
from configurable_automl_engine.tuner import InvalidAlgorithmError


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
  knn:
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
  xgboost:
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
        "knn",
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
        train_best_model(cfg_file, ["not", "a", "df"], model_path_override=tmp_path / "m.pkl")

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
  rf:
    enable: false
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(empty_cfg, "utf-8")
    
    with pytest.raises(ValueError):
        train_best_model(cfg_file, small_dataset)

# --------------------------------------------------------------------------- #
#  UNSUPPORTED ALGORITHM SHOULD RAISE
# --------------------------------------------------------------------------- #
def test_unsupported_algorithm(tmp_path: Path, small_dataset):
    """
    Проверка вызова InvalidAlgorithmError при попытке использовать 
    неподдерживаемый алгоритм (XGBoost в данном BROKEN_CFG).
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(BROKEN_CFG.format(model_path="dummy_path"), "utf-8")
    
    with pytest.raises(InvalidAlgorithmError):
        train_best_model(cfg_file, small_dataset, model_path_override=tmp_path / "m.pkl")

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
        return_value=expected_result
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
        tuner="mock_tuner",
        trainer_module="mock_trainer",
        hyperparameters=None
    )
@pytest.fixture
def base_config_dict():
    """Полный валидный словарь для Pydantic модели Config"""
    return {
        "general": {
            "comparison_metric": "accuracy",
            "validation_strategy": "k_fold",
            "n_folds": 5,
            "phases": [
                {"name": "fast", "n_trials": 2, "action": "all_algorithms"} # Исправлено 'all' -> 'all_algorithms'
            ],
            "path_to_model": "model.pkl",
            "log_to_file": None
        },
        "algorithms": {
            "rf": {
                "enable": True,
                "tuner": "mock_tuner",
                "trainer_module": "mock_trainer"
            }
        },
        "oversampling": {"data_oversampling": False}
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
                metric_name_sklearn="accuracy", 
                n_trials=1,
                validation_strategy=ValidationStrategy.k_fold
            )
        
        # Проверяем, что текст ошибки сохранился
        assert "Test Error" in str(excinfo.value)
    # Исправленный тест на отсутствие ModelTrainer
    @patch("configurable_automl_engine.training_engine.component._load_module")
    def test_fit_and_save_missing_trainer_class(self, mock_load, mock_algo_cfg, sample_df, base_config_dict):
        mock_load.return_value = MagicMock(spec=[]) 
        # Используем валидный конфиг вместо неполного словаря
        cfg = Config.model_validate(base_config_dict)
        
        with pytest.raises(AttributeError, match="lacks `ModelTrainer` class"):
            _fit_and_save("rf", mock_algo_cfg, sample_df, sample_df["target"], {}, Path("mod.pkl"), cfg)
    # Исправленный тест логирования
    @patch("configurable_automl_engine.training_engine.component.setup_logging")
    @patch("configurable_automl_engine.training_engine.component.read_config")
    def test_logging_setup(self, mock_read, mock_setup, sample_df, base_config_dict):
        base_config_dict["general"]["log_to_file"] = "test.log"
        mock_read.return_value = Config.model_validate(base_config_dict)
        
        # Мокаем HPO, чтобы не запускать реальное обучение
        with patch("configurable_automl_engine.training_engine.component._run_hpo", return_value=(0.9, {})):
            with patch("configurable_automl_engine.training_engine.component._fit_and_save"):
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
    @patch("configurable_automl_engine.training_engine.component._run_hpo", return_value=(0.9, {"p": 1}))
    @patch("configurable_automl_engine.training_engine.component._fit_and_save")
    def test_fit_and_save_failure(self, mock_fit, mock_hpo, sample_df, base_config_dict):
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
                metric_name_sklearn="accuracy",
                n_trials=1,
                validation_strategy=ValidationStrategy.k_fold
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
            metric_name_sklearn="accuracy",
            n_trials=1,
            validation_strategy=ValidationStrategy.train_test_split
        )
        assert score == 0.95
        assert params == {"param": 1}
# Ошибка, если target_col отсутствует в DataFrame
def test_train_best_model_missing_target_column():
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    config = {"dummy": "config"} # Неважно, так как упадет раньше
    with pytest.raises(ValueError, match="Target column 'missing_col' not found"):
        train_best_model(config=config, df=df, target="missing_col")

def test_train_best_model_config_from_dict_and_refine_flow():
    valid_config_dict = {
        "general": {
            "comparison_metric": "accuracy",
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "parallel_strategy": "none",
            "phases": [
                {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
                {"name": "p2", "n_trials": 1, "action": "refine_winner"}
            ],
            "path_to_model": "model.pkl"
        },
        "algorithms": {
            "logreg": {
                "enable": True, 
                "tuner": "unittest.mock", 
                "trainer_module": "unittest.mock"
            }
        },
        "oversampling": {
            "enable": False,
            "multiplier": 1.0,
            "algorithm": "random"
        }
    }
    
    df = pd.DataFrame({"f": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    # Патчим _run_hpo (вызывается внутри вложенной _execute_hpo_phase)
    # и _fit_and_save (вызывается в конце)
    with patch("configurable_automl_engine.training_engine.component._run_hpo", return_value=(0.9, {"C": 1.0})) as mock_hpo:
        with patch("configurable_automl_engine.training_engine.component._fit_and_save") as mock_save:
            result = train_best_model(config=valid_config_dict, df=df, target="target")
            
            assert result["algorithm"] == "logreg"
            # Ожидаем 2 вызова: по одному на каждую фазу
            assert mock_hpo.call_count == 2
            mock_save.assert_called_once()
def test_train_best_model_refine_winner_error_coverage():
    """ Ошибка при refine_winner в самой первой фазе"""
    invalid_dict = {
        "general": {
            "comparison_metric": "accuracy",
            "validation_strategy": "train_test_split",
            "phases": [{"name": "fail", "n_trials": 1, "action": "refine_winner"}],
            "path_to_model": "test.pkl"
        },
        "algorithms": {"a": {"enable": True, "tuner": "t", "trainer_module": "m"}},
        "oversampling": {"enable": False}
    }
    df = pd.DataFrame({"f": [1, 2], "target": [0, 1]})
    with pytest.raises(RuntimeError, match="requires a winner"):
        train_best_model(config=invalid_dict, df=df, target="target")
