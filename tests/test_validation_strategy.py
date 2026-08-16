import pytest
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    ValidationStrategy,
)

BASE_CFG = """
{
    "general": {
        "validation_strategy": "k_fold",
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            {"name": "refine", "n_trials": 1, "action": "refine_winner"}
        ]
    },
    "algorithms": {
        "elasticnet": {"enable": true}
    }
}
"""


@pytest.mark.parametrize("v", ["train_test_split", "k_fold", "loo", "auto"])
def test_valid_values(v):
    cfg = BASE_CFG.replace('"k_fold"', f'"{v}"')
    parsed = Config.model_validate_json(cfg)
    assert parsed.general.validation_strategy.value == v


def test_auto_accepts_default_n_folds():
    """'auto' passes parsing without requiring a manual n_folds (>= 2)."""
    cfg = BASE_CFG.replace('"k_fold"', '"auto"')
    # default n_folds = 5 is used; 'auto' must not demand n_folds >= 2 strictly.
    parsed = Config.model_validate_json(cfg)
    assert parsed.general.validation_strategy is ValidationStrategy.auto
    assert parsed.general.n_folds == 5


def test_default():
    cfg = BASE_CFG.replace('"validation_strategy": "k_fold",', "")
    parsed = Config.model_validate_json(cfg)
    assert parsed.general.validation_strategy is ValidationStrategy.auto
    assert parsed.general.n_folds == 5


def test_invalid_value():
    bad = BASE_CFG.replace('"k_fold"', '"wrong"')
    with pytest.raises(ValidationError):
        Config.model_validate_json(bad)
