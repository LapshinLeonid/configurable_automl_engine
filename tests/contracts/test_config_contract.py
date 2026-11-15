"""Structural validation and runtime boundary of the configuration contract."""

from __future__ import annotations

import pytest
from automl_contracts import Config
from pydantic import ValidationError

REFERENCE_CONFIG = {
    "general": {
        "comparison_metric": "r2",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "phases": [
            {"name": "search", "n_trials": 2, "action": "all_algorithms"},
            {
                "name": "final",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
    },
    "algorithms": {"random_forest": {"enable": True}},
}


def test_reference_config_is_validated() -> None:
    cfg = Config.model_validate(REFERENCE_CONFIG)
    assert cfg.general.comparison_metric == "r2"
    assert cfg.general.validation_strategy.value == "k_fold"
    assert cfg.algorithms.random_forest.enable is True


def test_unknown_algorithm_and_field_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {**REFERENCE_CONFIG, "algorithms": {"no_such_algo": {"enable": True}}}
        )
    with pytest.raises(ValidationError):
        Config.model_validate({**REFERENCE_CONFIG, "unknown_section": {}})


def test_at_least_one_algorithm_must_be_enabled() -> None:
    with pytest.raises(ValidationError, match="At least one algorithm"):
        Config.model_validate(
            {**REFERENCE_CONFIG, "algorithms": {"ridge": {"enable": False}}}
        )


def test_runtime_checks_are_not_part_of_structural_validation() -> None:
    data = {
        **REFERENCE_CONFIG,
        "general": {
            **REFERENCE_CONFIG["general"],
            "parallel_mode": "threads",
            "phase_timeout": 30,
        },
    }
    cfg = Config.model_validate(data)
    assert cfg.general.phase_timeout == 30


def test_phase_and_metric_rules_are_structural() -> None:
    with pytest.raises(ValidationError, match="last phase"):
        Config.model_validate(
            {
                **REFERENCE_CONFIG,
                "general": {
                    **REFERENCE_CONFIG["general"],
                    "phases": [{"name": "search", "n_trials": 1}],
                },
            }
        )
    with pytest.raises(ValidationError, match="unknown metric"):
        Config.model_validate(
            {
                **REFERENCE_CONFIG,
                "general": {**REFERENCE_CONFIG["general"], "comparison_metric": "nope"},
            }
        )
