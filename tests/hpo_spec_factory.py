"""Test helper: an ``_HpoPhaseSpec`` (one HPO phase snapshot) with neutral defaults."""

from __future__ import annotations

from typing import Any

import pandas as pd

from configurable_automl_engine.training_engine.config_parser import (
    HPOPhaseCfg,
    ValidationStrategy,
)
from configurable_automl_engine.training_engine.state import _HpoPhaseSpec


def make_phase_spec(**overrides: Any) -> _HpoPhaseSpec:
    """Build an HPO phase snapshot; keyword arguments override the defaults."""
    values: dict[str, Any] = {
        "phase": HPOPhaseCfg(name="coarse", n_trials=3, action="all_algorithms"),
        "prev_results": {},
        "y": pd.Series([1.0, 2.0, 3.0]),
        "metric_sklearn": "r2",
        "resolved_validation": ValidationStrategy.k_fold,
        "resolved_n_folds": 4,
        "resolved_test_size": 0.2,
        "categorical_features": ["c"],
        "numerical_features": ["a"],
        "oversampling_enabled": False,
        "oversampling_multiplier": 1.0,
        "oversampling_algorithm": "random",
        "oversampling_target_bins": 8,
        "oversampling_add_noise": False,
        "oversampling_noise_level": 0.01,
        "pruning_cfg": None,
        "encoding": "one_hot",
        "high_cardinality_threshold": 50,
        "high_cardinality_encoding": "frequency",
        "hashing_n_components": 16,
        "target_encoding_smoothing": 20.0,
        "target_encoding_fallback": None,
        "feature_selection_cfg": None,
    }
    values.update(overrides)
    return _HpoPhaseSpec(**values)
