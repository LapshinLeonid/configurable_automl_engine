import pytest
from .data_factory import create_mock_df

from pathlib import Path
from configurable_automl_engine.training_engine.config_parser import Config, GeneralCfg, HPOPhaseCfg
@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """
    Создает объект Config с изолированными путями во временной директории.
    """
    model_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            path_to_model=str(model_dir / "best_model.pkl"),
            log_to_file=str(log_dir / "test_run.log"),
            phases=[
                HPOPhaseCfg(name="Test Phase", n_trials=2, action="all_algorithms")
            ]
        ),
        algorithms={
            "random_forest": {"enable": True, "limit_hyperparameters": True},
            "ridge": {"enable": True, "limit_hyperparameters": True}
        }
    )

@pytest.fixture
def small_dataset():
    """
    Фикстура для быстрого тестирования базовой логики.
    Создает минимальный набор данных: 10 строк, 3 признака.
    """
    return create_mock_df(rows=10, cols=3, target="target")

@pytest.fixture
def regression_dataset():
    """
    Фикстура для тестирования алгоритмов регрессии или обучения.
    Создает расширенный набор данных: 200 строк, 10 признаков.
    """
    return create_mock_df(rows=200, cols=10, target="yield_score")