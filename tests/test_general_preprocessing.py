"""Tests for the run-level preprocessing default (``general.preprocessing``).

Cover the precedence resolution of
:func:`~configurable_automl_engine.preprocessing_presets.resolve_preprocessing_preset`
with ``default_override`` (class preset < general < algorithm), the public
merge helper, the config validation of the new block, the pass-through of the
default through the HPO request of the training engine, and the parity of the
preset resolved in the HPO trial with the preset of the final model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    merge_preprocessing_overrides,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.tuner import OptimizeResult
from configurable_automl_engine.training_engine.config.general import GeneralCfg

from configurable_automl_engine.training_engine import hpo_phases

_HPO_PHASES = hpo_phases


# ──────────────────────────────────────────────────────────────────────────
#  Precedence matrix: preset < general (default_override) < algorithm
# ──────────────────────────────────────────────────────────────────────────
def test_preset_only_no_overrides():
    """No overrides: the automatic class preset is used unchanged."""
    preset = resolve_preprocessing_preset("ridge")
    assert preset.imputation_strategy == "mean"
    assert preset.scaling.type == "standard"


def test_general_only_overrides_the_preset():
    """A field set only in the general block replaces the preset field."""
    preset = resolve_preprocessing_preset(
        "ridge", default_override={"imputation_strategy": "median"}
    )
    assert preset.imputation_strategy == "median"
    # The field the general block leaves unset keeps the preset value.
    assert preset.scaling.type == "standard"


def test_algorithm_only_overrides_the_preset():
    """A field set only in the algorithm block replaces the preset field."""
    preset = resolve_preprocessing_preset("ridge", {"scaling": "none"})
    assert preset.scaling.type == "none"
    assert preset.imputation_strategy == "mean"


def test_general_and_algorithm_together_algorithm_wins():
    """The algorithm level wins on the fields it sets."""
    preset = resolve_preprocessing_preset(
        "ridge",
        {"imputation_strategy": "mean", "scaling": "none"},
        default_override={"imputation_strategy": "median", "scaling": "robust"},
    )
    assert preset.imputation_strategy == "mean"
    assert preset.scaling.type == "none"


def test_unset_algorithm_field_falls_through_to_general():
    """A field the algorithm level leaves unset falls through to general."""
    preset = resolve_preprocessing_preset(
        "ridge",
        {"scaling": "none"},
        default_override={"imputation_strategy": "median"},
    )
    assert preset.imputation_strategy == "median"
    assert preset.scaling.type == "none"


def test_unset_general_field_falls_through_to_preset():
    """A field the general level leaves unset falls through to the preset."""
    preset = resolve_preprocessing_preset(
        "ridge",
        None,
        default_override={"scaling": "robust"},
    )
    assert preset.imputation_strategy == "mean"
    assert preset.scaling.type == "robust"


def test_default_override_accepts_model_instance():
    """``default_override`` accepts a ``PreprocessingOverride`` instance."""
    default = PreprocessingOverride(imputation_strategy="median")
    preset = resolve_preprocessing_preset("random_forest", default_override=default)
    assert preset.imputation_strategy == "median"
    assert preset.scaling.type == "none"


def test_default_override_invalid_value_rejected():
    """An invalid value in the general block is rejected (Pydantic)."""
    with pytest.raises(ValidationError):
        resolve_preprocessing_preset(
            "ridge", default_override={"imputation_strategy": "mode"}
        )


def test_default_override_invalid_type_rejected():
    """An unsupported default_override type is rejected with TypeError."""
    with pytest.raises(TypeError, match="preprocessing_override"):
        resolve_preprocessing_preset("ridge", default_override="median")


def test_positional_signature_unchanged():
    """The existing positional call signature keeps working."""
    assert resolve_preprocessing_preset("ridge", None) == resolve_preprocessing_preset(
        "ridge"
    )


# ──────────────────────────────────────────────────────────────────────────
#  merge_preprocessing_overrides
# ──────────────────────────────────────────────────────────────────────────
def test_merge_none_none_gives_empty_override():
    """Merging two absent blocks gives an empty override."""
    merged = merge_preprocessing_overrides(None, None)
    assert merged.imputation_strategy is None
    assert merged.scaling is None


def test_merge_specific_wins_per_field():
    """A field set at the algorithm level replaces the general field."""
    merged = merge_preprocessing_overrides(
        {"imputation_strategy": "median", "scaling": "robust"},
        {"scaling": "none"},
    )
    assert merged.imputation_strategy == "median"
    assert merged.scaling.type == "none"


def test_merge_unset_field_falls_through():
    """A field unset at the algorithm level keeps the general value."""
    merged = merge_preprocessing_overrides(
        {"imputation_strategy": "median"}, {"scaling": "none"}
    )
    assert merged.imputation_strategy == "median"
    assert merged.scaling.type == "none"


def test_merge_none_specific_gives_specific():
    """Without a general block the result is the coerced algorithm block."""
    merged = merge_preprocessing_overrides(None, {"scaling": "none"})
    assert merged.imputation_strategy is None
    assert merged.scaling.type == "none"


def test_merge_does_not_mutate_inputs():
    """The inputs are not mutated (a new model instance is returned)."""
    default = PreprocessingOverride(imputation_strategy="median")
    specific = PreprocessingOverride(scaling="none")
    merged = merge_preprocessing_overrides(default, specific)
    assert merged.imputation_strategy == "median"
    assert default.imputation_strategy == "median"
    assert specific.scaling.type == "none"


def test_merge_rejects_unknown_keys():
    """Unknown keys are rejected in both merge inputs."""
    with pytest.raises(ValidationError):
        merge_preprocessing_overrides({"unknown_field": 1}, None)
    with pytest.raises(ValidationError):
        merge_preprocessing_overrides(None, {"unknown_field": 1})


# ──────────────────────────────────────────────────────────────────────────
#  Config validation of general.preprocessing
# ──────────────────────────────────────────────────────────────────────────
def test_general_preprocessing_defaults_to_none():
    """Without the block the config field is None (no run-level default)."""
    assert GeneralCfg().preprocessing is None


def test_general_preprocessing_valid_block_accepted():
    """A valid general block is accepted and coerced to the model."""
    cfg = GeneralCfg(preprocessing={"imputation_strategy": "median"})
    assert cfg.preprocessing is not None
    assert cfg.preprocessing.imputation_strategy == "median"
    assert cfg.preprocessing.scaling is None


def test_general_preprocessing_unknown_key_rejected():
    """Unknown keys in general.preprocessing are rejected."""
    with pytest.raises(ValidationError):
        GeneralCfg(preprocessing={"unknown_field": 1})


def test_general_preprocessing_invalid_value_rejected():
    """Invalid field values in general.preprocessing are rejected."""
    with pytest.raises(ValidationError):
        GeneralCfg(preprocessing={"scaling": "quantile"})


# ──────────────────────────────────────────────────────────────────────────
#  Engine pass-through and parity (e2e)
# ──────────────────────────────────────────────────────────────────────────
def _parity_cfg_dict() -> dict[str, Any]:
    """A compact config: general default median, ridge overrides scaling."""
    return {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "train_test_split",
            "preprocessing": {"imputation_strategy": "median"},
            "phases": [
                {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            },
        },
    }


def _small_df(rows: int = 40) -> pd.DataFrame:
    """A small numeric regression dataset with a 'target' column."""
    import numpy as np

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(rows, 3), columns=["a", "b", "c"])
    X["target"] = X["a"] * 2 - X["b"] + rng.randn(rows) * 0.1
    return X


def test_engine_passes_general_default_and_parity_holds(
    tmp_path: Path, monkeypatch
):
    """For a config with general.preprocessing the preset resolved by the
    HPO trial equals ModelTrainer.preprocessing_preset after the final fit.

    The run is driven with a fake HPO (the request is captured to assert the
    pass-through of both levels); the final model is trained for real."""
    from configurable_automl_engine import train_best_model
    from configurable_automl_engine.trainer import ModelTrainer

    captured: list[Any] = []

    def fake_run(X, y, request):
        captured.append(request)
        return OptimizeResult(
            model=None, params={"alpha": 0.5}, score=0.9, trials=[]
        )

    monkeypatch.setattr(hpo_phases, "run_optimization", fake_run)

    cfg = _parity_cfg_dict()
    df = _small_df()
    res = train_best_model(
        cfg, df, target="target", model_path_override=tmp_path / "m.pkl"
    )

    assert res["algorithm"] == "ridge"
    assert len(captured) == 1
    request = captured[0]
    # The run-level default travels in the HPO request...
    assert request.preprocessing_default is not None
    assert request.preprocessing_default.imputation_strategy == "median"
    # ...and the per-algorithm block is absent from the config, so the
    # override argument of the request stays None (the default applies).
    assert request.preprocessing_override is None

    # Parity: the preset resolved by the trial pipeline equals the preset of
    # the final model.
    hpo_preset = resolve_preprocessing_preset(
        request.algo_name,
        request.preprocessing_override,
        default_override=request.preprocessing_default,
    )
    trainer = ModelTrainer.load(res["model_path"])
    assert trainer.preprocessing_preset == hpo_preset
    # The general level applied; no algorithm block was configured.
    assert trainer.preprocessing_preset.imputation_strategy == "median"
    assert trainer.preprocessing_preset.scaling.type == "standard"


def test_engine_parity_with_algorithm_override(tmp_path: Path, monkeypatch):
    """A config with general.preprocessing AND an override on one algorithm:
    the HPO trial preset equals the final model preset; the algorithm level
    wins per field (the engine merges the blocks before ModelTrainer)."""
    from configurable_automl_engine import train_best_model
    from configurable_automl_engine.trainer import ModelTrainer

    captured: list[Any] = []

    def fake_run(X, y, request):
        captured.append(request)
        return OptimizeResult(
            model=None, params={"alpha": 0.5}, score=0.9, trials=[]
        )

    monkeypatch.setattr(hpo_phases, "run_optimization", fake_run)

    cfg = _parity_cfg_dict()
    cfg["algorithms"]["ridge"]["preprocessing"] = {
        "imputation_strategy": "mean",
        "scaling": "robust",
    }
    df = _small_df()
    res = train_best_model(
        cfg, df, target="target", model_path_override=tmp_path / "m.pkl"
    )

    assert res["algorithm"] == "ridge"
    request = captured[0]
    hpo_preset = resolve_preprocessing_preset(
        request.algo_name,
        request.preprocessing_override,
        default_override=request.preprocessing_default,
    )
    trainer = ModelTrainer.load(res["model_path"])
    assert trainer.preprocessing_preset == hpo_preset
    # The algorithm level won on both fields.
    assert trainer.preprocessing_preset.imputation_strategy == "mean"
    assert trainer.preprocessing_preset.scaling.type == "robust"


def test_config_without_general_block_unchanged(tmp_path: Path, monkeypatch):
    """Regression: a config without the block behaves exactly as before —
    the HPO request carries no default and the preset is the class preset."""
    from configurable_automl_engine import train_best_model
    from configurable_automl_engine.trainer import ModelTrainer

    captured: list[Any] = []

    def fake_run(X, y, request):
        captured.append(request)
        return OptimizeResult(
            model=None, params={"alpha": 0.5}, score=0.9, trials=[]
        )

    monkeypatch.setattr(hpo_phases, "run_optimization", fake_run)

    cfg = _parity_cfg_dict()
    del cfg["general"]["preprocessing"]
    df = _small_df()
    res = train_best_model(
        cfg, df, target="target", model_path_override=tmp_path / "m.pkl"
    )

    assert res["algorithm"] == "ridge"
    assert captured[0].preprocessing_default is None
    assert captured[0].preprocessing_override is None
    trainer = ModelTrainer.load(res["model_path"])
    assert trainer.preprocessing_preset == resolve_preprocessing_preset("ridge")
    assert trainer.preprocessing_preset.imputation_strategy == "mean"
    assert trainer.preprocessing_preset.scaling.type == "standard"
