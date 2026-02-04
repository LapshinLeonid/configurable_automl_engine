from pathlib import Path
import pytest
import pandas as pd

from configurable_automl_engine.training_engine.component import train_best_model
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
