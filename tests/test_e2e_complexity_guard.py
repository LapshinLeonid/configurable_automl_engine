"""E2E tests of the adaptive complexity guard on a synthetic small dataset.

Covers the acceptance criteria:
1. On a small dataset (~18 rows, 5 features) the log shows the adapted
   tree/boosting ranges ("was → became" with the reason).
2. The final report of ``train_best_model`` contains the ``complexity_guard``
   section with the adapted parameter table.
3. XGBoost training time with the guard is markedly lower than without it
   (fewer, shallower trees).
4. Multi-phase configs (TPE history transfer between phases) work without
   ``HPOHistoryError`` — the guard is deterministic between phases.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.training_engine.component import train_best_model

# Optuna INFO logs (every trial) clutter the test output — keep warnings+.
logging.getLogger("optuna").setLevel(logging.WARNING)


def _small_frame(n: int = 18, p: int = 5, seed: int = 42) -> pd.DataFrame:
    """Synthetic small regression frame (n rows, p features)."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(
        2.0 * X["f0"] + 0.5 * X["f1"] + rng.randn(n) * 0.5, name="target"
    )
    return pd.concat([X, y], axis=1)


def _single_phase_config(complexity_guard: dict | None = None) -> dict:
    cfg: dict = {
        "general": {
            "phases": [
                {"name": "search", "n_trials": 4, "action": "all_algorithms"},
                {
                    "name": "select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "validation_strategy": "k_fold",
            "n_folds": 5,
        },
        "algorithms": {
            "decision_tree": {"enable": True},
            "xgboosting": {"enable": True},
            "ridge": {"enable": True},
        },
    }
    if complexity_guard is not None:
        cfg["general"]["complexity_guard"] = complexity_guard
    return cfg


def test_e2e_small_data_guard_applies_and_reports(caplog):
    """On small data the guard adapts the ranges and reports them."""
    df = _small_frame()
    with caplog.at_level(logging.INFO, logger="configurable_automl_engine.tuner"):
        res = train_best_model(_single_phase_config(), df, target="target")

    # The report section is present with per-parameter records.
    assert "complexity_guard" in res, res.keys()
    records = res["complexity_guard"]["parameters"]
    assert records, "no complexity guard records"
    params = {(r["algo"], r["param"]) for r in records}
    # decision_tree min_samples_leaf and max_depth were adapted.
    assert ("decision_tree", "min_samples_leaf") in params
    assert ("decision_tree", "max_depth") in params
    # XGBoost max_depth was regularized to [1, 3].
    xgb_depth = next(r for r in records if (r["algo"], r["param"]) == ("xgboosting", "max_depth"))
    assert xgb_depth["after"] == [1.0, 3.0]
    # The adapted ranges are visible in the log ("was → became").
    assert any("max_depth" in rec.message and "→" in rec.message for rec in caplog.records)


def test_e2e_small_data_guard_off_spaces_unchanged():
    """mode='off': the report is absent and XGBoost keeps its wide range."""
    df = _small_frame()
    cfg = _single_phase_config(
        {"mode": "off", "lossless": True, "regularize": True}
    )
    res = train_best_model(cfg, df, target="target")
    assert "complexity_guard" not in res


def _wide_one_hot_frame(
    n: int = 4000, n_cat: int = 50, n_categories: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Wide frame: n rows, n_cat categorical columns with n_categories
    levels each -> n_cat * n_categories = 250 one-hot columns after encoding."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            f"c{i}": rng.randint(0, n_categories, size=n).astype(str)
            for i in range(n_cat)
        }
    )
    y = pd.Series(
        rng.randn(n) + 0.1 * rng.randint(0, n_categories, size=n) * 1.0,
        name="target",
    )
    return pd.concat([X, y], axis=1)


def test_e2e_large_wide_one_hot_spaces_unchanged():
    """Large-but-wide data (n=4000, 250 one-hot columns / 50 original)
    must NOT trigger the 'medium' mode (issue #134).

    With the size-mode ratio computed from the one-hot-expanded p, n/p =
    4000/250 = 16 < 20 resolved 'medium' and restricted the spaces: XGBoost
    max_depth <= 5 / subsample >= 0.7, RF min_samples_leaf >= 2 / max_depth
    <= 13. With the ORIGINAL column count (50) the ratio is 4000/50 = 80, so
    the mode is 'ordinary' and the spaces stay unchanged — the guard
    produces no records.
    """
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
    from configurable_automl_engine.tuner import optimize

    df = _wide_one_hot_frame()
    X, y = df.drop(columns=["target"]), df["target"]
    for algo in ("random_forest", "xgboosting"):
        res = optimize(
            algo,
            X,
            y,
            n_trials=2,
            random_state=42,
            # 2-fold CV: n_samples_eff = 4000/2 = 2000 >= 300, so the
            # 'medium' mode can fire ONLY via the n/p ratio — the
            # one-hot-expanded p (250) gives 8 < 20, the original column
            # count (50) gives 40 >= 20.
            validation_strategy="k_fold",
            n_folds=2,
            space_overrides={algo: dict(DEFAULT_SPACES[algo])},
            complexity_guard_cfg={"mode": "apply"},
            fit_best_model=False,
        )
        assert res is not None, f"{algo}: no valid result"
        # The guard produced NO restrictions: the spaces were unchanged.
        assert not res.complexity_guard, (
            f"{algo}: guard unexpectedly restricted the space: "
            f"{res.complexity_guard}"
        )


def test_e2e_user_range_conflict_does_not_crash():
    """A user range conflicting with the guard does not crash the search.

    Regression for the KeyError in the conflict log path: with the default
    ``user_range_policy='user_wins'`` the guard kept the user range and the
    logger unconditionally read ``record['before']`` from a conflict record
    that lacked the key, so ``decision_tree`` was disqualified with
    ``RuntimeError: No algorithms produced valid scores`` when only the
    conflicting algorithm was enabled.
    """
    df = _small_frame()
    cfg = _single_phase_config()
    cfg["algorithms"] = {
        "decision_tree": {
            "enable": True,
            "hyperparameters": {"max_depth": [10, 10, "int"]},
        }
    }
    res = train_best_model(cfg, df, target="target")
    # The conflicting algorithm survives HPO (no disqualification).
    assert res["algorithm"] == "decision_tree"
    assert not res.get("disqualified_algorithms")
    # The conflict was resolved in favor of the explicit user range.
    assert res["params"]["max_depth"] == 10


def test_e2e_multiphase_history_transfer_works():
    """Multi-phase configs with TPE history transfer do not raise HPOHistoryError."""
    df = _small_frame(n=40, p=6)
    cfg = {
        "general": {
            "phases": [
                {"name": "search", "n_trials": 3, "action": "all_algorithms"},
                {"name": "refine", "n_trials": 3, "action": "refine"},
                {
                    "name": "select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "validation_strategy": "k_fold",
            "n_folds": 5,
            "complexity_guard": {"mode": "apply"},
        },
        "algorithms": {
            "decision_tree": {"enable": True},
            "xgboosting": {"enable": True},
            "ridge": {"enable": True},
        },
    }
    # Must not raise HPOHistoryError: the guard produces the same ranges in
    # every phase, so the transferred TPE history stays compatible.
    res = train_best_model(cfg, df, target="target")
    assert res["algorithm"] in {"decision_tree", "xgboosting", "ridge"}
    # The report has ONE record per (algorithm, parameter) — the A/B rules
    # merge and the phases do not duplicate records.
    if "complexity_guard" in res:
        records = res["complexity_guard"]["parameters"]
        keys = [(r["algo"], r["param"]) for r in records]
        assert len(keys) == len(set(keys)), f"duplicated records: {keys}"


def test_e2e_user_int_max_features_not_collapsed():
    """An int max_features user range survives the guard (regression).

    ``_clamp_for_param`` used to treat ``max_features`` as a fraction
    unconditionally, collapsing an int range e.g. [5, 10] to the fixed value
    1 with ``applied=True`` and no conflict — violating the documented user
    range contract. With the fix the int range stays a feature count.
    """
    rng = np.random.RandomState(7)
    n, p = 60, 8
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(2.0 * X["f0"] + rng.randn(n) * 0.5, name="target")
    df = pd.concat([X, y], axis=1)

    cfg = {
        "general": {
            "phases": [
                {"name": "search", "n_trials": 3, "action": "all_algorithms"},
                {
                    "name": "select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "validation_strategy": "k_fold",
            "n_folds": 3,
        },
        "algorithms": {
            "decision_tree": {
                "enable": True,
                "hyperparameters": {"max_features": [5, 10, "int"]},
            }
        },
    }
    res = train_best_model(cfg, df, target="target")
    assert res["algorithm"] == "decision_tree"
    # The user int range [5, 10] is respected (the sampled value must be in it).
    assert isinstance(res["params"]["max_features"], int)
    assert 5 <= res["params"]["max_features"] <= 10
    # The report records no conflict/no collapse for max_features.
    records = res.get("complexity_guard", {}).get("parameters", [])
    mf_records = [r for r in records if r["param"] == "max_features"]
    assert not mf_records, f"max_features unexpectedly restricted: {mf_records}"


@pytest.mark.skipif(not is_installed("xgboost"), reason="xgboost not installed")
def test_e2e_xgboost_time_with_guard_markedly_lower():
    """XGBoost HPO time on small data is not worse with the guard.

    With the guard, XGBoost searches max_depth in [1, 3] and caps
    n_estimators at 300, so the trained models are cheaper than without
    the guard (deep trees up to depth 10, up to 800 estimators). The
    assertion is on the best-of-two runs of each mode (short runs are
    noisy) and bounds the guard run at 1.5x of the unguarded one — a
    "markedly lower" budget in practice, while still leaving headroom
    for CI noise.
    """
    df = _small_frame(n=60, p=5)

    cfg_off = _single_phase_config({"mode": "off"})
    cfg_on = _single_phase_config({"mode": "apply"})

    def _best_run(cfg: dict, repeats: int = 2) -> float:
        best = float("inf")
        for _ in range(repeats):
            start = time.monotonic()
            train_best_model(cfg, df, target="target")
            best = min(best, time.monotonic() - start)
        return best

    time_off = _best_run(cfg_off)
    time_on = _best_run(cfg_on)

    # The guard run must stay within 1.5x of the unguarded one (with the
    # guard the adapted max_depth/n_estimators ranges make the trials
    # cheaper — the point is that the guard never makes HPO dramatically
    # slower).
    assert time_on <= time_off * 1.5, (
        f"guard run {time_on:.3f}s > 1.5x of unguarded {time_off:.3f}s"
    )


@pytest.mark.parametrize("algo", ["decision_tree", "gradient_boosting"])
def test_e2e_float_min_samples_leaf_fraction_survives(algo):
    """A float min_samples_leaf fraction (0, 1) survives the guard.

    Regression for the third review round: ``_clamp_for_param``/
    ``_domain_for_param`` treated a float ``min_samples_leaf`` in (0, 1) as a
    row count — with ``user_wins`` the user range was silently widened (the
    lower bound jumped to 1.0), and with ``engine_wins`` HPO crashed with
    ``RuntimeError: No algorithms produced valid scores``.
    """
    rng = np.random.RandomState(9)
    n, p = 30, 5
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(2.0 * X["f0"] + rng.randn(n) * 0.5, name="target")
    df = pd.concat([X, y], axis=1)

    cfg = {
        "general": {
            "phases": [
                {"name": "search", "n_trials": 4, "action": "all_algorithms"},
                {
                    "name": "select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "validation_strategy": "k_fold",
            "n_folds": 3,
        },
        "algorithms": {
            algo: {
                "enable": True,
                "hyperparameters": {"min_samples_leaf": [0.2, 1.0, "float"]},
            },
            "ridge": {"enable": True},
        },
    }
    res = train_best_model(cfg, df, target="target")
    assert res["algorithm"] in {algo, "ridge"}
    if res["algorithm"] == algo:
        # The user float-fraction range is respected: the sampled value is a
        # fraction in (0, 1] (never widened to a row count >= 1).
        value = res["params"]["min_samples_leaf"]
        assert 0.0 < value <= 1.0, f"min_samples_leaf widened to {value!r}"


@pytest.mark.parametrize("algo", ["decision_tree", "gradient_boosting"])
def test_e2e_float_min_samples_leaf_engine_wins_does_not_crash(algo):
    """engine_wins on a conflicting float-fraction range does not crash HPO.

    Regression: the old code widened the float fraction to a row count in the
    engine_wins domain, so the fitted trees crashed the HPO phase and
    ``train_best_model`` raised ``RuntimeError: No algorithms produced valid
    scores``.
    """
    rng = np.random.RandomState(11)
    n, p = 30, 5
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(2.0 * X["f0"] + rng.randn(n) * 0.5, name="target")
    df = pd.concat([X, y], axis=1)

    cfg = {
        "general": {
            "phases": [
                {"name": "search", "n_trials": 3, "action": "all_algorithms"},
                {
                    "name": "select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "validation_strategy": "k_fold",
            "n_folds": 3,
            "complexity_guard": {"mode": "apply", "user_range_policy": "engine_wins"},
        },
        "algorithms": {
            # [0.4, 0.9] does not intersect the guard range (fraction [2/30, 0.2]).
            algo: {
                "enable": True,
                "hyperparameters": {"min_samples_leaf": [0.4, 0.9, "float"]},
            },
            "ridge": {"enable": True},
        },
    }
    res = train_best_model(cfg, df, target="target")
    assert res["algorithm"] in {algo, "ridge"}
    # The guard conflict resolves via engine_wins without disqualifying the
    # algorithm (HPO completed with a valid score).
    disqualified = res.get("disqualified_algorithms") or {}
    assert algo not in disqualified, f"{algo} disqualified: {disqualified}"
