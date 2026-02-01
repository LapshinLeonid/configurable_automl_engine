import pytest
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import Config, ValidationStrategy

BASE_CFG = """
{
    "general": {
        "project_name": "demo",
        "validation_strategy": "k_fold",
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            {"name": "refine", "n_trials": 1, "action": "refine_winner"}
        ]
    },
    "algorithms": {
        "logreg": {"enable": true}
    }
}
"""


@pytest.mark.parametrize("v", ["train_test_split", "k_fold", "loo"])
def test_valid_values(v):
    cfg = BASE_CFG.replace('"k_fold"', f'"{v}"')
    parsed = Config.model_validate_json(cfg)
    assert parsed.general.validation_strategy.value == v

def test_default():
    cfg = BASE_CFG.replace('"validation_strategy": "k_fold",', '')
    parsed = Config.model_validate_json(cfg)
    assert parsed.general.validation_strategy is ValidationStrategy.k_fold

def test_invalid_value():
    bad = BASE_CFG.replace('"k_fold"', '"wrong"')
    with pytest.raises(ValidationError):
        Config.model_validate_json(bad)
