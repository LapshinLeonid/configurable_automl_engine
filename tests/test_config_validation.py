import pytest
from pydantic import ValidationError
from configurable_automl_engine.training_engine.config_parser import Config, ValidationStrategy

BASE = {
    "general": {
        "comparison_metric": "nrmse",
        "validation_strategy": "k_fold",
        "n_folds": 5,
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            {"name": "refine", "n_trials": 1, "action": "refine_winner"}
        ]
    },
    "oversampling": {},
    "algorithms": {"lgbm": {"enable": True}},
}

def test_n_folds_ok():
    cfg = Config.model_validate(BASE)
    assert cfg.general.n_folds == 5

def test_n_folds_bad():
    bad = BASE | {"general": {**BASE["general"], "n_folds": 1}}
    with pytest.raises(ValidationError):
        Config.model_validate(bad)

def test_n_folds_ignored_for_loo():
    loo = BASE | {"general": {**BASE["general"], "validation_strategy": "loo", "n_folds": 1}}
    cfg = Config.model_validate(loo)  # не должно падать
    assert cfg.general.validation_strategy == ValidationStrategy.loo