import pytest
from .data_factory import create_mock_df

from pathlib import Path
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    GeneralCfg,
    HPOPhaseCfg,
)


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """
    Creates a Config object with isolated paths in a temporary directory.
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
            ],
        ),
        algorithms={
            "random_forest": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"n_estimators": [10, 20]},
            },
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            },
        },
    )


@pytest.fixture
def small_dataset():
    """
    Fixture for quickly testing the basic logic.
    Creates a minimal dataset: 10 rows, 3 features.
    """
    return create_mock_df(rows=10, cols=3, target="target")


@pytest.fixture
def regression_dataset():
    """
    Fixture for testing regression algorithms or training.
    Creates an expanded dataset: 200 rows, 10 features.
    """
    return create_mock_df(rows=200, cols=10, target="yield_score")
