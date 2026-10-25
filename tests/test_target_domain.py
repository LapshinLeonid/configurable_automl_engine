"""Tests for the pre-HPO target-domain guard of GLM algorithms.

``gammaregressor`` (``y > 0``), ``poissonregressor`` (``y >= 0``) and
``tweedieregressor`` / ``glm`` (domain depends on ``power``) must be excluded
before HPO when the target does not fit their loss domain; a Tweedie ``power``
range that is only partly invalid is clipped to its valid part instead.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.hyperopt_defaults import SearchSpaceEntry
from configurable_automl_engine.common.target_domain import (
    check_target_domain,
    tweedie_power_is_valid,
)
from configurable_automl_engine.training_engine.component import (
    PhaseContext,
    _apply_target_domain_guard,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import AlgoCfg

Y_NEG = pd.Series([-2.0, 0.5, 1.0, 3.0])
Y_ZERO = pd.Series([0.0, 0.5, 1.0, 3.0])
Y_POS = pd.Series([0.1, 0.5, 1.0, 3.0])


def _entry(spec: list) -> SearchSpaceEntry:
    return SearchSpaceEntry.model_validate(spec)


# ──────────────────────────────────────────────────────────────────────────────
# tweedie_power_is_valid
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("power", "y_min", "expected"),
    [
        (0.0, -5.0, True),
        (-1.0, -5.0, True),
        (0.5, -0.1, False),
        (1.0, -0.1, False),
        (1.0, 0.0, True),
        (1.5, 0.0, True),
        (2.0, 0.0, False),
        (2.0, 0.1, True),
        (3.0, 0.1, True),
    ],
)
def test_tweedie_power_is_valid(power, y_min, expected):
    assert tweedie_power_is_valid(power, y_min) is expected


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-domain GLMs: gamma / poisson
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("y", [Y_NEG, Y_ZERO])
def test_gamma_disqualified_for_non_positive_target(y):
    check = check_target_domain("gammaregressor", {}, y)
    assert check.disqualify_reason is not None
    assert "y > 0" in check.disqualify_reason
    assert "before HPO" in check.disqualify_reason
    assert check.clipped_entries == {}


def test_gamma_accepted_for_positive_target():
    check = check_target_domain("gammaregressor", {}, Y_POS)
    assert check.disqualify_reason is None
    assert check.clipped_entries == {}


def test_poisson_disqualified_for_negative_target():
    check = check_target_domain("poissonregressor", {}, Y_NEG)
    assert check.disqualify_reason is not None
    assert "y >= 0" in check.disqualify_reason
    assert "-2" in check.disqualify_reason


@pytest.mark.parametrize("y", [Y_ZERO, Y_POS])
def test_poisson_accepted_for_non_negative_target(y):
    assert check_target_domain("poissonregressor", {}, y).disqualify_reason is None


def test_other_algorithms_are_not_checked():
    check = check_target_domain("ridge", {"power": _entry([1.0, 3.0])}, Y_NEG)
    assert check.disqualify_reason is None
    assert check.clipped_entries == {}


@pytest.mark.parametrize(
    "y",
    [
        pd.Series([], dtype=float),
        pd.Series([np.nan, np.nan]),
        pd.Series(["a", "b"]),
    ],
)
def test_target_without_finite_values_is_not_checked(y):
    """Nothing to check — other diagnostics own these targets."""
    assert check_target_domain("gammaregressor", {}, y).disqualify_reason is None


def test_target_nan_values_ignored():
    y = pd.Series([np.nan, 0.5, 1.0])
    assert check_target_domain("gammaregressor", {}, y).disqualify_reason is None


def test_numpy_target_supported():
    check = check_target_domain("gammaregressor", {}, np.array([[-1.0], [2.0]]))
    assert check.disqualify_reason is not None


# ──────────────────────────────────────────────────────────────────────────────
# Tweedie / glm: power handling
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("algo", ["tweedieregressor", "glm"])
def test_tweedie_without_power_accepts_negative_target(algo):
    """The default power=0 (normal) is valid for any target."""
    space = {"alpha": _entry([1e-6, 1e-1, "float_log"])}
    check = check_target_domain(algo, space, Y_NEG)
    assert check.disqualify_reason is None
    assert check.clipped_entries == {}


@pytest.mark.parametrize("algo", ["tweedieregressor", "glm"])
def test_tweedie_power_range_fully_invalid_disqualifies(algo):
    check = check_target_domain(algo, {"power": _entry([1.0, 3.0])}, Y_NEG)
    assert check.disqualify_reason is not None
    assert "power" in check.disqualify_reason
    assert "[1.0, 3.0]" in check.disqualify_reason


def test_tweedie_power_range_above_two_on_zero_target_disqualifies():
    check = check_target_domain("glm", {"power": _entry([2.0, 3.0])}, Y_ZERO)
    assert check.disqualify_reason is not None


def test_tweedie_power_range_valid_unchanged():
    entry = _entry([1.0, 3.0])
    check = check_target_domain("glm", {"power": entry}, Y_POS)
    assert check.disqualify_reason is None
    assert check.clipped_entries == {}


def test_tweedie_power_range_clipped_on_negative_target():
    check = check_target_domain("glm", {"power": _entry([-1.0, 3.0])}, Y_NEG)
    assert check.disqualify_reason is None
    clipped = check.clipped_entries["power"].config
    assert (clipped.low, clipped.high) == (-1.0, 0.0)
    assert check.messages and "[-1.0, 3.0] -> [-1.0, 0.0]" in check.messages[0]


def test_tweedie_power_range_clipped_to_single_point():
    """[0, 3] on a negative target keeps only power=0 (a degenerate range)."""
    check = check_target_domain("glm", {"power": _entry([0.0, 3.0])}, Y_NEG)
    clipped = check.clipped_entries["power"].config
    assert (clipped.low, clipped.high) == (0.0, 0.0)


def test_tweedie_power_range_clipped_below_two_on_zero_target():
    check = check_target_domain("tweedieregressor", {"power": _entry([1.0, 3.0])}, Y_ZERO)
    clipped = check.clipped_entries["power"].config
    assert clipped.low == 1.0
    assert clipped.high < 2.0
    assert math.isclose(clipped.high, 2.0)
    assert tweedie_power_is_valid(clipped.high, 0.0)


def test_tweedie_power_range_with_step_stays_on_grid():
    check = check_target_domain("glm", {"power": _entry([0.0, 3.0, "float", 0.5])}, Y_ZERO)
    clipped = check.clipped_entries["power"].config
    assert (clipped.low, clipped.high, clipped.step) == (0.0, 1.5, 0.5)


def test_tweedie_power_int_range_clipped():
    check = check_target_domain("glm", {"power": _entry([0, 3, "int"])}, Y_ZERO)
    clipped = check.clipped_entries["power"].config
    assert clipped.type == "int"
    assert (clipped.low, clipped.high) == (0, 1)
    assert isinstance(clipped.high, int)


def test_tweedie_power_int_range_with_step_clipped():
    check = check_target_domain("glm", {"power": _entry([-4, 4, "int", 3])}, Y_NEG)
    clipped = check.clipped_entries["power"].config
    assert (clipped.low, clipped.high) == (-4, -1)


def test_tweedie_power_float_log_range():
    """float_log has low > 0: clipped below 2 on a zero target, invalid on y < 0."""
    entry = _entry([0.5, 3.0, "float_log"])
    clipped = check_target_domain("glm", {"power": entry}, Y_ZERO)
    assert clipped.clipped_entries["power"].config.type == "float_log"
    assert clipped.clipped_entries["power"].config.high < 2.0
    assert check_target_domain("glm", {"power": entry}, Y_NEG).disqualify_reason


def test_tweedie_power_categorical_filtered():
    entry = _entry([[0, 1.5, 2, 3], "categorical"])
    check = check_target_domain("glm", {"power": entry}, Y_ZERO)
    assert check.clipped_entries["power"].config.options == [0, 1.5]


def test_tweedie_power_categorical_fully_invalid_disqualifies():
    entry = _entry([[1, 2], "categorical"])
    check = check_target_domain("glm", {"power": entry}, Y_NEG)
    assert check.disqualify_reason is not None


def test_tweedie_power_categorical_valid_unchanged():
    entry = _entry([[0, 1.5], "categorical"])
    check = check_target_domain("glm", {"power": entry}, Y_ZERO)
    assert check.clipped_entries == {}


def test_tweedie_original_entry_not_mutated():
    entry = _entry([0.0, 3.0])
    check_target_domain("glm", {"power": entry}, Y_NEG)
    assert entry.config.high == 3.0


@pytest.mark.parametrize(
    ("power", "y", "disqualified"),
    [(2.0, Y_ZERO, True), (0.0, Y_NEG, False), (1.5, Y_ZERO, False), ("x", Y_NEG, False)],
)
def test_tweedie_fixed_power_value(power, y, disqualified):
    check = check_target_domain("glm", {"power": power}, y)
    assert (check.disqualify_reason is not None) is disqualified


# ──────────────────────────────────────────────────────────────────────────────
# _apply_target_domain_guard (pipeline integration)
# ──────────────────────────────────────────────────────────────────────────────
def test_guard_disqualifies_and_clips(caplog):
    algos = {
        "gammaregressor": AlgoCfg(),
        "glm": AlgoCfg.model_validate({"hyperparameters": {"power": [-1.0, 3.0]}}),
        "ridge": AlgoCfg(),
        "poissonregressor": AlgoCfg(enable=False),
    }
    ctx = PhaseContext()
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        guarded = _apply_target_domain_guard(algos, Y_NEG, ctx)

    assert set(ctx.disqualified_algorithms) == {"gammaregressor"}
    assert "y > 0" in ctx.disqualified_algorithms["gammaregressor"]
    # The disabled algorithm is not checked.
    assert "poissonregressor" not in ctx.disqualified_algorithms
    assert guarded["ridge"] is algos["ridge"]
    assert guarded["poissonregressor"] is algos["poissonregressor"]
    power = guarded["glm"].hyperparameters["power"].config
    assert (power.low, power.high) == (-1.0, 0.0)
    # The input config is not mutated.
    assert algos["glm"].hyperparameters["power"].config.high == 3.0
    gamma_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "gammaregressor" in r.getMessage()
    ]
    assert len(gamma_warnings) == 1
    assert "clipped" in caplog.text


def test_guard_respects_limit_hyperparameters():
    algos = {
        "glm": AlgoCfg.model_validate(
            {"limit_hyperparameters": True, "hyperparameters": {"power": [1.0, 3.0]}}
        ),
    }
    ctx = PhaseContext()
    _apply_target_domain_guard(algos, Y_NEG, ctx)
    assert "glm" in ctx.disqualified_algorithms


def test_guard_no_op_for_valid_target():
    algos = {"gammaregressor": AlgoCfg(), "poissonregressor": AlgoCfg()}
    ctx = PhaseContext()
    guarded = _apply_target_domain_guard(algos, Y_POS, ctx)
    assert ctx.disqualified_algorithms == {}
    assert guarded == algos


# ──────────────────────────────────────────────────────────────────────────────
# E2E: target-domain guard
# ──────────────────────────────────────────────────────────────────────────────
def _negative_target_df(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    df["target"] = 2.0 * df["f0"] - df["f1"] + rng.normal(scale=0.1, size=rows)
    assert (df["target"] < 0).any()
    return df


def _cfg(model_path: Path, algorithms: dict) -> dict:
    return {
        "general": {
            "comparison_metric": "rmse",
            "path_to_model": str(model_path),
            "phases": [
                {"name": "Coarse Search", "n_trials": 3, "action": "all_algorithms"},
                {
                    "name": "Final Select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": algorithms,
    }


def test_e2e_r18_gamma_and_poisson_excluded_before_any_trial(tmp_path, caplog):
    import configurable_automl_engine.tuner as tuner_module

    real_create_model = tuner_module.create_model
    created: list[str] = []

    def capturing(algo: str, **params):
        created.append(algo)
        return real_create_model(algo, **params)

    model_path = tmp_path / "model.pkl"
    cfg = _cfg(
        model_path,
        {
            "gammaregressor": {"enable": True},
            "poissonregressor": {"enable": True},
            "ridge": {"enable": True},
        },
    )
    with (
        patch.object(tuner_module, "create_model", side_effect=capturing),
        caplog.at_level(logging.WARNING),
    ):
        res = train_best_model(
            config=cfg,
            df=_negative_target_df(),
            target="target",
            model_path_override=model_path,
        )

    assert res["algorithm"] == "ridge"
    assert set(res["disqualified_algorithms"]) == {"gammaregressor", "poissonregressor"}
    # No trial ever ran for the excluded algorithms.
    assert "gammaregressor" not in created
    assert "poissonregressor" not in created
    # Zero fold-failure warnings; exactly one WARNING per excluded algorithm.
    assert "out of the valid range" not in caplog.text
    for algo in ("gammaregressor", "poissonregressor"):
        records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and algo in r.getMessage()
        ]
        assert len(records) == 1, [r.getMessage() for r in records]


def test_e2e_tweedie_power_range_clipped(tmp_path, caplog):
    model_path = tmp_path / "model.pkl"
    cfg = _cfg(
        model_path,
        {
            "glm": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"power": [0.0, 3.0]},
            },
        },
    )
    with caplog.at_level(logging.WARNING):
        res = train_best_model(
            config=cfg,
            df=_negative_target_df(),
            target="target",
            model_path_override=model_path,
        )
    assert res["algorithm"] == "glm"
    assert "disqualified_algorithms" not in res
    assert res["params"]["power"] == 0.0
    assert "out of the valid range" not in caplog.text


def test_e2e_all_algorithms_excluded_raises(tmp_path):
    model_path = tmp_path / "model.pkl"
    cfg = _cfg(model_path, {"gammaregressor": {"enable": True}})
    with pytest.raises(RuntimeError, match="Disqualified algorithms.*gammaregressor"):
        train_best_model(
            config=cfg,
            df=_negative_target_df(),
            target="target",
            model_path_override=model_path,
        )
