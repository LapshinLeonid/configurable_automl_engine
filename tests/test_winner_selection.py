"""
Unit tests for the pure winner-selection logic and the winner-score invariant.

They cover ``select_winner`` and ``is_valid_winner_score`` from
``training_engine/component.py``:

• select_winner — a deterministic tie-break on equal scores
  (the first in configuration order wins);
• is_valid_winner_score — the worst-score sentinel class threshold
  (score <= float32 minimum → invalid), including the 'raw' float32
  minimum that the old filter via ``math.isclose(rel_tol=1e-9)`` missed.

Integration of the two-stage winner selection:

• the finalists pool (``select_finalists``) + ``select_robust_winner``:
  the ``off`` / ``active`` modes and the fallback when the whole pool
  fails; the ``warn_only`` mode was removed along with the report keys
  ``sanity_gate_warn_only``/``winner_if_active``;
• the specification case: ElasticNet CV=0.081, SVR=0.085, RF=0.089,
  δ=0.15 → a pool of three; the RMSE_oof winner is SVR or RF, but not
  ElasticNet;
• 'golden' scenarios on synthetic data N=10–30: the winner is an honest
  model, while the degenerate (a)–(d) are disqualified under
  ``mode=active``.

All tests use fixed seeds (the repository standard).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine.training_engine.component import (
    FinalistCandidate,
    PhaseResult,
    is_valid_winner_score,
    select_finalists,
    select_robust_winner,
    select_winner,
)
from configurable_automl_engine.training_engine.metrics import oof_rmse
from configurable_automl_engine.training_engine.sanity_gate import ModelSanityGate
from configurable_automl_engine.tuner import (
    HPO_WORST_SCORE,
    WORST_SCORE_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# select_winner: deterministic winner selection
# ──────────────────────────────────────────────────────────────────────────────
def test_select_winner_single_algorithm():
    """A single algorithm — it wins (a string name is returned)."""
    results = {"ridge": PhaseResult(score=0.7, params={"alpha": 1.0})}
    winner = select_winner(results)
    assert isinstance(winner, str)
    assert winner == "ridge"


def test_select_winner_max_score_wins():
    """The algorithm with the maximum 'raw' score wins."""
    results = {
        "ridge": PhaseResult(score=0.4, params={"alpha": 1.0}),
        "random_forest": PhaseResult(score=0.9, params={"n_estimators": 10}),
        "extra_trees": PhaseResult(score=0.6, params={"n_estimators": 20}),
    }
    assert select_winner(results) == "random_forest"


def test_select_winner_tie_breaks_by_configuration_order():
    """Equal scores → the first one in configuration order wins.

    ``max`` is stable and a dict preserves insertion order, so a tie is
    resolved by the iteration order (the configuration order for the
    sequential path).
    """
    results = {
        "random_forest": PhaseResult(score=0.5, params={"n_estimators": 10}),
        "extra_trees": PhaseResult(score=0.5, params={"n_estimators": 20}),
        "elasticnet": PhaseResult(score=0.5, params={"alpha": 0.5}),
    }
    assert select_winner(results) == "random_forest"


def test_select_winner_tie_reorders_with_negative_scores():
    """The tie-break is correct for negative (inverted) scores too."""
    results = {
        "extra_trees": PhaseResult(score=-0.5, params={}),
        "random_forest": PhaseResult(score=-0.5, params={}),
    }
    assert select_winner(results) == "extra_trees"


def test_select_winner_empty_raises_value_error():
    """An empty results dict → ValueError (there is no winner)."""
    with pytest.raises(ValueError, match="empty results"):
        select_winner({})


# ──────────────────────────────────────────────────────────────────────────────
# is_valid_winner_score: the winner-score invariant
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "score",
    [
        0.42,
        1.0,
        -0.9,
        -3.4e38,  # very small, but a valid score (above the float32 min)
        -3.39e38,
        0.0,
        5,  # an integer score
        np.float32(0.5),
        np.float64(-0.75),
    ],
)
def test_is_valid_winner_score_accepts_finite_above_threshold(score: Any):
    """Valid scores: numbers that are finite and strictly above the float32-min threshold."""
    assert is_valid_winner_score(score) is True


@pytest.mark.parametrize(
    "score",
    [
        None,
        "not-a-number",
        "0.5",  # a numeric string — still garbage
        b"0.5",
        (0.5,),
        [0.5],
        {"value": 0.5},
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32("nan"),
        np.float32("inf"),
        np.float32("-inf"),
        HPO_WORST_SCORE,
        float(np.finfo(np.float32).min),  # 'raw' float32 min
        np.float32(np.finfo(np.float32).min),
        -1e39,  # below the float32 min — the sentinel class
        float("-inf"),
    ],
)
def test_is_valid_winner_score_rejects_sentinel_and_nonfinite(score: Any):
    """Invalid scores: non-numbers, garbage, NaN/±inf, and the worst-score sentinel class."""
    assert is_valid_winner_score(score) is False


def test_threshold_covers_hpo_worst_score_and_raw_f32_min():
    """The sentinel-class threshold covers both known signatures.

    HPO_WORST_SCORE (-3.4028235e38) and the 'raw' float(np.finfo(np.float32).min)
    differ by ~9.88e-9 relative — the old isclose(rel_tol=1e-9) missed the
    second one. Both must fall into the sentinel class.
    """
    raw_f32_min = float(np.finfo(np.float32).min)
    assert HPO_WORST_SCORE != raw_f32_min  # different double values
    assert not math.isclose(raw_f32_min, HPO_WORST_SCORE, rel_tol=1e-9)
    assert raw_f32_min <= WORST_SCORE_THRESHOLD
    assert HPO_WORST_SCORE <= WORST_SCORE_THRESHOLD
    # A valid score must be strictly above the threshold
    assert -3.4e38 > WORST_SCORE_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
#  Integration of the two-stage winner selection
# ──────────────────────────────────────────────────────────────────────────────

R = dict[str, PhaseResult]


def _golden_dataset(
    n: int = 30, seed: int = 7, noise: float = 0.5, p: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset: X0/X1 are informative, the rest is noise.

    N=10–30 as in the real golden cases; fixed seed.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * noise
    return X, y


def _fit_oof_full(
    factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Honest OOF vector (the fold model has not seen the row) + full-fit.

    Returns (y_pred_oof, y_pred_full, full_model). The number of folds is 5
    for small datasets N≈30, so each row lands in the test set exactly once.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        model = factory()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    full_model = factory().fit(X, y)
    return oof, full_model.predict(X), full_model


def _candidate(
    name: str,
    cv_score: float,
    rmse_oof: float,
    y_oof: np.ndarray,
    y_full: np.ndarray,
    model: Any,
) -> FinalistCandidate:
    """Build a ``FinalistCandidate`` from a trained model and its vectors."""
    return FinalistCandidate(
        algo_name=name,
        model=model,
        params={},
        cv_score=cv_score,
        rmse_oof=rmse_oof,
        y_pred_oof=y_oof,
        y_pred_full=y_full,
    )


def _golden_candidates(
    X: np.ndarray, y: np.ndarray
) -> dict[str, FinalistCandidate]:
    """Train the 'golden' pool (a)–(e) on synthetic data N=10–30.

    (a) ElasticNet with zeroed coefficients ('shelf') → flat OOF;
    (b) SVR-sigmoid stuck on the tanh plateau → almost constant predictions;
    (c) a depth-20 tree (RMSE_train≈0) → Circuit D;
    (d) 1-NN → Circuit D;
    (e) an honest model (shallow gradient boosting) → passes the audit.

    The dict order is the deterministic pool order from the pool builder
    (by descending CV score), exactly as ``select_finalists`` produces it.
    """
    factories: dict[str, tuple[Callable[[], Any], float]] = {
        # CV score in the optimizer's 'raw' semantics (neg-RMSE): the best is
        # the closest to zero. The degenerate models 'win' by CV (per the spec).
        "elasticnet_zero": (lambda: ElasticNet(alpha=500.0, random_state=0), -0.081),
        "svr_sigmoid": (
            lambda: SVR(kernel="sigmoid", gamma=0.001, coef0=0.0),
            -0.085,
        ),
        "deep_tree": (lambda: DecisionTreeRegressor(max_depth=20, random_state=0), -0.089),
        "knn_1": (lambda: KNeighborsRegressor(n_neighbors=1), -0.093),
        "honest_gbr": (
            lambda: GradientBoostingRegressor(
                max_depth=1, n_estimators=20, learning_rate=0.05, random_state=0
            ),
            -0.11,
        ),
    }
    candidates: dict[str, FinalistCandidate] = {}
    for name, (factory, cv_score) in factories.items():
        oof, full, model = _fit_oof_full(factory, X, y)
        candidates[name] = _candidate(
            name,
            cv_score=cv_score,
            rmse_oof=float(oof_rmse(y, oof)),
            y_oof=oof,
            y_full=full,
            model=model,
        )
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
#  Modes off / active (pool + robust winner integration)
# ──────────────────────────────────────────────────────────────────────────────


def test_integration_off_and_active_pool_to_robust_winner():
    """Finalists pool → select_robust_winner: the semantics of the modes.

    The CV leader is a degenerate model (flat ElasticNet): in ``off`` it
    wins as before; in ``active`` it is disqualified and an honest model
    with the best RMSE_oof wins. The ``warn_only`` mode was removed.
    """
    X, y = _golden_dataset(n=80, seed=11)
    results: R = {
        "elasticnet_zero": PhaseResult(score=-0.081, params={"alpha": 500.0}),
        "svr": PhaseResult(score=-0.085, params={"C": 1.0}),
        "random_forest": PhaseResult(score=-0.089, params={"n_estimators": 50}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet_zero", "svr", "random_forest"]

    # Train the finalists in pool order (the deterministic pool tie-break).
    factories = {
        "elasticnet_zero": (lambda: ElasticNet(alpha=500.0, random_state=0), -0.081),
        "svr": (lambda: SVR(kernel="rbf", C=1.0), -0.085),
        "random_forest": (
            lambda: RandomForestRegressor(max_depth=3, n_estimators=50, random_state=0),
            -0.089,
        ),
    }
    cands: dict[str, FinalistCandidate] = {}
    for name in pool:
        factory, cv_score = factories[name]
        oof, full, model = _fit_oof_full(factory, X, y)
        cands[name] = _candidate(
            name, cv_score=cv_score, rmse_oof=float(oof_rmse(y, oof)),
            y_oof=oof, y_full=full, model=model,
        )

    gate = ModelSanityGate()
    X_df, y_s = pd.DataFrame(X), pd.Series(y)

    off = select_robust_winner(cands, X_df, y_s, gate, mode="off")
    assert off.winner_algo == "elasticnet_zero"  # legacy behavior: best CV
    assert off.audit == {}
    assert off.disqualified == {}

    active = select_robust_winner(cands, X_df, y_s, gate, mode="active")
    assert active.winner_algo in {"svr", "random_forest"}
    assert active.winner_algo != "elasticnet_zero"
    assert active.winner_rmse_oof == pytest.approx(cands[active.winner_algo].rmse_oof)
    assert "elasticnet_zero" in active.disqualified
    assert active.fallback_used is False


def test_warn_only_mode_removed():
    """Boundary: the ``warn_only`` mode was removed — ValueError."""
    rng = np.random.RandomState(0)
    n = 40
    y = 5.0 * np.linspace(0.0, 1.0, n) + rng.randn(n) * 0.1
    cand = FinalistCandidate(
        algo_name="a",
        model=object(),
        params={},
        cv_score=-0.08,
        rmse_oof=0.3,
        y_pred_oof=y + rng.randn(n) * 0.2,
        y_pred_full=y,
    )
    with pytest.raises(ValueError, match="mode must be 'off' or 'active'"):
        select_robust_winner(
            {"a": cand},
            pd.DataFrame(np.zeros((n, 2))),
            pd.Series(y),
            ModelSanityGate(),
            mode="warn_only",  # type: ignore[arg-type]
        )


def test_boundary_active_reports_disqualifications_without_changing_pool():
    """Boundary: ``active`` applies disqualifications but does not change the pool.

    Circuit D in the strict ``reject`` mode: degenerate deep_tree / 1-NN
    (RMSE_full≈0) are disqualified as in the old behavior (the default warn
    mode would leave them valid with soft_warnings).
    """
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    gate = ModelSanityGate(generalization_gap_mode="reject")
    X_df, y_s = pd.DataFrame(X), pd.Series(y)

    off = select_robust_winner(cands, X_df, y_s, gate, mode="off")
    active = select_robust_winner(cands, X_df, y_s, gate, mode="active")
    # The pool is the same in both modes; only the disqualifications differ.
    assert active.pool == off.pool
    assert set(active.disqualified) == {
        "elasticnet_zero",
        "svr_sigmoid",
        "deep_tree",
        "knn_1",
    }
    assert active.winner_algo == "honest_gbr"


def test_boundary_pool_exactly_top_k_candidates():
    """Boundary: a pool of exactly top_k candidates (all inside the δ corridor)."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "svr": PhaseResult(score=-0.085, params={}),
        "random_forest": PhaseResult(score=-0.089, params={}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]
    assert len(pool) == 3


# ──────────────────────────────────────────────────────────────────────────────
#  Case from the specification: ElasticNet 0.081 / SVR 0.085 / RF 0.089, δ=0.15
# ──────────────────────────────────────────────────────────────────────────────


def test_case_elasticnet_svr_rf_delta_015_winner_not_elasticnet():
    """ElasticNet CV=0.081, SVR=0.085, RF=0.089, δ=0.15 → a pool of three.

    ElasticNet with zeroed coefficients is disqualified; the RMSE_oof winner
    is SVR or RF (but not ElasticNet).
    """
    X, y = _golden_dataset(n=80, seed=11, noise=0.5, p=5)
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={"alpha": 500.0}),
        "svr": PhaseResult(score=-0.085, params={"C": 1.0}),
        "random_forest": PhaseResult(score=-0.089, params={"n_estimators": 50}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]  # a pool of three

    factories = {
        "elasticnet": (lambda: ElasticNet(alpha=500.0, random_state=0), -0.081),
        "svr": (lambda: SVR(kernel="rbf", C=1.0), -0.085),
        "random_forest": (
            lambda: RandomForestRegressor(max_depth=3, n_estimators=50, random_state=0),
            -0.089,
        ),
    }
    cands: dict[str, FinalistCandidate] = {}
    for name in pool:
        factory, cv_score = factories[name]
        oof, full, model = _fit_oof_full(factory, X, y)
        cands[name] = _candidate(
            name, cv_score=cv_score, rmse_oof=float(oof_rmse(y, oof)),
            y_oof=oof, y_full=full, model=model,
        )

    res = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="active"
    )
    assert res.winner_algo in {"svr", "random_forest"}
    assert res.winner_algo != "elasticnet"
    assert "elasticnet" in res.disqualified
    assert res.fallback_used is False


# ──────────────────────────────────────────────────────────────────────────────
#  Golden scenarios (N=10–30): (a)–(d) disqualified, winner is (e)
# ──────────────────────────────────────────────────────────────────────────────


def test_golden_small_n_winner_is_honest_model_mode_active():
    """Golden: under mode=active the winner is an honest model (e).

The degenerate (a) 'shelf' ElasticNet, (b) SVR-sigmoid on the tanh
    plateau, (c) depth-20 tree and (d) 1-NN are disqualified; the honest
    gradient boosting (e) passes the audit and wins by RMSE_oof.
    Circuit D is in the strict ``reject`` mode (the default warn mode would
    leave deep_tree / 1-NN valid with soft_warnings).
    """
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    res = select_robust_winner(
        cands,
        pd.DataFrame(X),
        pd.Series(y),
        ModelSanityGate(generalization_gap_mode="reject"),
        mode="active",
    )
    assert res.winner_algo == "honest_gbr"
    assert res.winner_rmse_oof == pytest.approx(cands["honest_gbr"].rmse_oof)
    assert set(res.disqualified) == {
        "elasticnet_zero",
        "svr_sigmoid",
        "deep_tree",
        "knn_1",
    }
    assert res.audit["honest_gbr"].is_valid is True
    assert res.fallback_used is False


def test_golden_small_n_mode_off_old_behavior():
    """Golden: mode=off — the legacy select_winner behavior.

    The winner is the best CV score (a degenerate model); no audit is run.
    """
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    res = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="off"
    )
    assert res.winner_algo == "elasticnet_zero"  # best 'raw' CV score
    assert res.audit == {}
    assert res.disqualified == {}
    assert res.fallback_used is False


def test_golden_small_n_determinism_repeated_runs():
    """Golden: reproducibility — repeated runs are identical."""
    X, y = _golden_dataset(n=30, seed=7)
    cands = _golden_candidates(X, y)
    first = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="active"
    )
    for _ in range(3):
        again = select_robust_winner(
            cands, pd.DataFrame(X), pd.Series(y), ModelSanityGate(), mode="active"
        )
        assert again.winner_algo == first.winner_algo
        assert again.disqualified == first.disqualified
        assert again.pool == first.pool
        assert again.fallback_used == first.fallback_used


# ──────────────────────────────────────────────────────────────────────────────
#  Fallback: all candidates rejected
# ──────────────────────────────────────────────────────────────────────────────


def test_boundary_all_candidates_rejected_fallback():
    """Boundary: the whole pool is rejected → the fallback criterion.

    The formal criterion: minimum number of violated circuits → minimum
    severity (D > A > B > C) → best RMSE_oof. The run does not fail.
    """
    from unittest.mock import MagicMock

    from configurable_automl_engine.training_engine.sanity_gate import SanityCheckResult

    rng = np.random.RandomState(0)
    n = 60
    y = rng.randn(n) * 5.0
    y_oof = y + rng.randn(n) * 0.5
    x_df = pd.DataFrame(np.zeros((n, 2)))

    def _cand(name: str, cv: float, rmse: float) -> FinalistCandidate:
        return FinalistCandidate(
            algo_name=name,
            model=object(),
            params={},
            cv_score=cv,
            rmse_oof=rmse,
            y_pred_oof=y_oof,
            y_pred_full=y_oof,
        )

    cands = {
        "one_violation": _cand("one_violation", -0.08, 0.3),
        "two_violations": _cand("two_violations", -0.09, 0.2),
    }
    gate = MagicMock()
    gate.check.side_effect = [
        SanityCheckResult(
            is_valid=False, reasons=["Circuit D (generalization gap): gap=2.0 > 1.5"]
        ),
        SanityCheckResult(
            is_valid=False,
            reasons=[
                "Circuit A (diversity): ratio=0.05 < 0.15",
                "Circuit B (unique): nunique=2 < 5",
            ],
        ),
    ]
    res = select_robust_winner(cands, x_df, pd.Series(y), gate, mode="active")
    assert res.fallback_used is True
    # Fewer violated circuits → the 'least problematic' model.
    assert res.winner_algo == "one_violation"
    assert res.winner_rmse_oof == pytest.approx(0.3)