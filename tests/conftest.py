"""Shared fixtures of the engine test suite.

The contract tests are collected from the same repository and must run in an
environment that has only ``automl-contracts`` installed. Engine imports are
therefore performed inside the fixtures; an environment without the engine
simply never uses them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def test_config(tmp_path: Path) -> Any:
    """Create a Config object with isolated paths in a temporary directory."""
    from configurable_automl_engine.training_engine.config_parser import (
        Config,
        GeneralCfg,
        HPOPhaseCfg,
    )

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
def small_dataset() -> Any:
    """Create a minimal dataset: 10 rows, 3 features."""
    from .data_factory import create_mock_df

    return create_mock_df(rows=10, cols=3, target="target")


@pytest.fixture
def regression_dataset() -> Any:
    """Create an expanded synthetic dataset: 200 rows, 10 features."""
    from .data_factory import create_mock_df

    return create_mock_df(rows=200, cols=10, target="yield_score")


@pytest.fixture(autouse=True)
def _reset_oversampling_experimental_warnings() -> Any:
    """Reset the warn-once state of the oversampling warning around each test."""
    try:
        from configurable_automl_engine.oversampling import (
            reset_experimental_warnings,
        )
    except ImportError:
        yield
        return
    reset_experimental_warnings()
    yield
    reset_experimental_warnings()


@pytest.fixture(autouse=True)
def _reset_allowed_hyperparameters_cache() -> Any:
    """Clear ``get_allowed_hyperparameters`` cache around each engine test."""
    try:
        from configurable_automl_engine.models import get_allowed_hyperparameters
    except ImportError:
        yield
        return
    get_allowed_hyperparameters.cache_clear()
    yield
    get_allowed_hyperparameters.cache_clear()
