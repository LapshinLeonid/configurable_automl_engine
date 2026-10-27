"""Tests for two-stage winner selection with the Sanity Gate.

Coverage:

1. ``select_robust_winner`` — the ``off`` / ``active`` modes (the
   ``warn_only`` mode was removed along with the report keys
   ``sanity_gate_warn_only``/``winner_if_active``).
2. Ranking of audit-passing candidates by ``RMSE_oof``: CV and ``RMSE_full``
   do NOT participate in the ranking (there is no composite score).
3. Negative: a 'flat' ElasticNet leader (constant OOF) is disqualified;
   a deep tree with ``RMSE_train≈0`` is cut off by Circuit D; SVR-sigmoid
   on the tanh plateau is cut off by Circuit A.
4. Boundary: ``mode=off`` (legacy behavior); all candidates rejected
   (fallback with a formal criterion); equal ``RMSE_oof`` (deterministic
   tie-break by pool order).
5. Fallback: minimum number of violated circuits → minimum severity
   (circuit significance D > A > B > C, ``_circuit_rank``) → best
   ``RMSE_oof``; a WARNING with the reasons is logged.
6. Integration with ``train_best_model`` through an explicit select phase:
   the ``sanity_gate`` section in the report is keyed by the phase name,
   gate disqualifications reach the circuit breaker, and the finalist is
   trained exactly once.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_friedman3
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine.training_engine.component import train_best_model
from configurable_automl_engine.training_engine.finalists import select_robust_winner
from configurable_automl_engine.training_engine.state import (
    FinalistCandidate,
    PhaseResult,
)
from configurable_automl_engine.training_engine.finalists import (
    _circuit_rank,
    _severity_signature,
)
from configurable_automl_engine.training_engine.config_parser import (
    ERR_REMOVED_SANITY_GATE_MODE,
    Config,
    GeneralCfg,
    SanityGateCfg,
)
from configurable_automl_engine.common.metrics import oof_rmse
from configurable_automl_engine.training_engine.sanity_gate import (
    ModelSanityGate,
    SanityCheckResult,
)


# Terminal select step: the last phase of any valid config.
FINAL_SELECT_PHASE = {
    "name": "final_select",
    "n_trials": 1,
    "action": "select",
    "strategy": "cv_best",
    "output": "winner",
}

# The orchestrator module logger name: component.py uses
# ``logging.getLogger(__name__)``, i.e. the full package path.
_COMPONENT_LOGGER = "configurable_automl_engine.training_engine.component"


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _dataset(seed: int = 7, n: int = 150, p: int = 6, noise: float = 1.0):
    """Clean dataset: X0/X1 are informative, the rest is noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * noise
    return X, y


def _honest_vectors(
    n: int = 120, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Honest vectors: y with variance, OOF/full with small noise.

    Circuit A: diversity ≈ 1 >> 0.15; Circuit B: nunique ~ N; Circuit D:
    gap = RMSE_oof/RMSE_full = 0.25/0.2 = 1.25 <= 1.5 — the model is valid.
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 1.0, n)
    y = 5.0 * x + rng.randn(n) * 0.1
    y_oof = y + rng.randn(n) * 0.25
    y_full = y + rng.randn(n) * 0.2
    return y, y_oof, y_full


def _oof_with_rmse(y: np.ndarray, target_rmse: float, seed: int) -> np.ndarray:
    """An OOF vector with the given RMSE relative to ``y`` (normalized noise)."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(y))
    noise = noise / np.sqrt(np.mean(noise**2)) * target_rmse
    return y + noise


def _flat_vectors(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat (constant) predictions — a degenerate model."""
    const = float(np.mean(y))
    return np.full(len(y), const), np.full(len(y), const)


def _candidate(
    name: str,
    cv_score: float,
    rmse_oof: float,
    y_oof: np.ndarray,
    y_full: np.ndarray | None = None,
    model: Any = None,
    rmse_full: float | None = None,
) -> FinalistCandidate:
    """Build a candidate with the given vectors and scores."""
    return FinalistCandidate(
        algo_name=name,
        model=model,
        params={},
        cv_score=cv_score,
        rmse_oof=rmse_oof,
        y_pred_oof=y_oof,
        y_pred_full=y_full if y_full is not None else y_oof,
        rmse_full=rmse_full,
    )


def _gate(**kwargs: Any) -> ModelSanityGate:
    """A gate without Circuit C (unit tests provide no models)."""
    return ModelSanityGate(check_dead_features=False, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
#  select_robust_winner: basic invariants and the off mode
# ──────────────────────────────────────────────────────────────────────────────


def test_robust_winner_empty_pool_raises():
    gate = _gate()
    with pytest.raises(RuntimeError, match="empty candidate pool"):
        select_robust_winner({}, pd.DataFrame(), pd.Series(dtype=float), gate)


def test_robust_winner_unknown_mode_raises():
    y, y_oof, y_full = _honest_vectors()
    cand = _candidate("a", cv_score=-0.1, rmse_oof=0.25, y_oof=y_oof, y_full=y_full)
    with pytest.raises(ValueError, match="mode"):
        select_robust_winner(
            {"a": cand},
            pd.DataFrame(np.zeros((len(y), 2))),
            pd.Series(y),
            _gate(),
            mode="weird",  # type: ignore[arg-type]
        )


def test_off_mode_picks_cv_leader_without_audit():
    """mode=off: the legacy select_winner behavior; no audit is performed."""
    y, y_oof, y_full = _honest_vectors()
    cands = {
        "svr": _candidate(
            "svr", cv_score=-0.081, rmse_oof=0.09, y_oof=_oof_with_rmse(y, 0.09, 1)
        ),
        "rf": _candidate(
            "rf", cv_score=-0.089, rmse_oof=0.085, y_oof=_oof_with_rmse(y, 0.085, 2)
        ),
    }
    res = select_robust_winner(
        cands, pd.DataFrame(np.zeros((len(y), 2))), pd.Series(y), _gate(), mode="off"
    )
    assert res.winner_algo == "svr"  # best 'raw' CV score
    assert res.audit == {}
    assert res.disqualified == {}
    assert res.fallback_used is False
    assert res.mode == "off"


def test_off_mode_tie_breaks_by_pool_order():
    y, y_oof, y_full = _honest_vectors()
    cands = {
        "first": _candidate("first", cv_score=-0.05, rmse_oof=0.3, y_oof=y_oof),
        "second": _candidate("second", cv_score=-0.05, rmse_oof=0.2, y_oof=y_oof),
    }
    res = select_robust_winner(
        cands, pd.DataFrame(np.zeros((len(y), 2))), pd.Series(y), _gate(), mode="off"
    )
    assert res.winner_algo == "first"  # deterministic tie-break


# ──────────────────────────────────────────────────────────────────────────────
#  Active mode: ranking by RMSE_oof among audit-passing candidates
# ──────────────────────────────────────────────────────────────────────────────


def test_active_winner_is_best_rmse_oof_among_valid():
    """Winner = best RMSE_oof among those that passed the audit (CV is not involved)."""
    y, _, _ = _honest_vectors()
    cands = {
        # CV leader, but degenerate (flat OOF) → disqualified
        "elasticnet": _candidate(
            "elasticnet",
            cv_score=-0.081,
            rmse_oof=1.0,
            y_oof=_flat_vectors(y)[0],
            y_full=_flat_vectors(y)[1],
        ),
        # Honest models: rf better by RMSE_oof than svr but worse by CV
        "svr": _candidate(
            "svr", cv_score=-0.085, rmse_oof=0.09, y_oof=_oof_with_rmse(y, 0.09, 1)
        ),
        "random_forest": _candidate(
            "random_forest",
            cv_score=-0.089,
            rmse_oof=0.085,
            y_oof=_oof_with_rmse(y, 0.085, 2),
        ),
    }
    res = select_robust_winner(
        cands, pd.DataFrame(np.zeros((len(y), 2))), pd.Series(y), _gate(), mode="active"
    )
    assert res.winner_algo == "random_forest"
    assert res.winner_rmse_oof == pytest.approx(0.085)
    assert "elasticnet" in res.disqualified
    assert res.fallback_used is False
    # Disqualification reasons are available for the report
    assert any("Circuit A" in r for r in res.disqualified["elasticnet"])
    assert any("Circuit B" in r for r in res.disqualified["elasticnet"])


def test_active_disqualified_best_rmse_cannot_win():
    """A candidate with the best RMSE_oof but degenerate predictions cannot win."""
    y, _, _ = _honest_vectors()
    cands = {
        "flat": _candidate(
            "flat",
            cv_score=-0.09,
            rmse_oof=0.01,
            y_oof=_flat_vectors(y)[0],
            y_full=_flat_vectors(y)[1],
        ),
        "honest": _candidate(
            "honest", cv_score=-0.08, rmse_oof=0.3, y_oof=_oof_with_rmse(y, 0.3, 3)
        ),
    }
    res = select_robust_winner(
        cands, pd.DataFrame(np.zeros((len(y), 2))), pd.Series(y), _gate(), mode="active"
    )
    assert res.winner_algo == "honest"
    assert "flat" in res.disqualified


def test_active_equal_rmse_oof_tie_breaks_by_pool_order():
    """Equal RMSE_oof → deterministic tie-break by pool order."""
    y, _, _ = _honest_vectors()
    cands = {
        "first": _candidate(
            "first", cv_score=-0.05, rmse_oof=0.2, y_oof=_oof_with_rmse(y, 0.2, 1)
        ),
        "second": _candidate(
            "second", cv_score=-0.04, rmse_oof=0.2, y_oof=_oof_with_rmse(y, 0.2, 2)
        ),
    }
    res = select_robust_winner(
        cands, pd.DataFrame(np.zeros((len(y), 2))), pd.Series(y), _gate(), mode="active"
    )
    assert res.winner_algo == "first"


def test_active_audit_failure_is_disqualification():
    """An audit failure itself (an exception in check) disqualifies the candidate."""
    y, y_oof, y_full = _honest_vectors()
    cand = _candidate("broken", cv_score=-0.1, rmse_oof=0.2, y_oof=y_oof, y_full=y_full)

    class _BrokenGate(ModelSanityGate):
        def check(self, **kwargs: Any) -> SanityCheckResult:
            raise RuntimeError("boom")

    res = select_robust_winner(
        {"broken": cand},
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        _BrokenGate(check_dead_features=False),
        mode="active",
    )
    assert res.disqualified["broken"] == ["Sanity audit failed: boom"]
    # A single candidate, rejected → fallback still picks it
    assert res.fallback_used is True
    assert res.winner_algo == "broken"


# ──────────────────────────────────────────────────────────────────────────────
#  Audit time budget
# ──────────────────────────────────────────────────────────────────────────────


def test_audit_time_budget_zero_audits_all_candidates():
    """Budget = 0 (default): no limit, all candidates are audited."""
    y, y_oof, y_full = _honest_vectors()
    cands = {
        "a": _candidate("a", cv_score=-0.08, rmse_oof=0.3, y_oof=y_oof, y_full=y_full),
        "b": _candidate("b", cv_score=-0.09, rmse_oof=0.2, y_oof=y_oof, y_full=y_full),
    }
    res = select_robust_winner(
        cands,
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        _gate(),
        mode="active",
        audit_time_budget_seconds=0,
    )
    assert set(res.audit) == {"a", "b"}
    assert res.disqualified == {}


def test_audit_time_budget_exhausted_skips_remaining_audits(mocker):
    """Budget exhaustion: remaining candidates are not audited (a reason in the report).

    ``time.monotonic`` is mocked: start at 0.0, the first candidate is
    checked at 1.0 (< budget 5.0), the rest at 10.0 (>= budget) → skipped.
    """
    import configurable_automl_engine.training_engine.finalists as finalists_mod

    y, y_oof, y_full = _honest_vectors()
    cands = {
        "a": _candidate("a", cv_score=-0.08, rmse_oof=0.3, y_oof=y_oof, y_full=y_full),
        "b": _candidate("b", cv_score=-0.09, rmse_oof=0.2, y_oof=y_oof, y_full=y_full),
        "c": _candidate("c", cv_score=-0.10, rmse_oof=0.25, y_oof=y_oof, y_full=y_full),
    }
    mocker.patch.object(
        finalists_mod.time,
        "monotonic",
        side_effect=[0.0, 1.0, 10.0, 10.0],
    )
    res = select_robust_winner(
        cands,
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        _gate(),
        mode="active",
        audit_time_budget_seconds=5.0,
    )
    # Only the first candidate is audited; the rest are skipped.
    assert set(res.audit) == {"a", "b", "c"}
    assert res.audit["a"].is_valid is True
    assert res.audit["b"].is_valid is False
    assert res.audit["c"].is_valid is False
    assert all(
        "audit time budget" in r for name in ("b", "c") for r in res.disqualified[name]
    )
    # A skipped candidate cannot win in active mode.
    assert res.winner_algo == "a"
    assert "a" not in res.disqualified


# ──────────────────────────────────────────────────────────────────────────────
#  Fallback: all candidates rejected
# ──────────────────────────────────────────────────────────────────────────────


def _result(reasons: list[str]) -> SanityCheckResult:
    return SanityCheckResult(is_valid=False, reasons=reasons, severity=2)


def test_fallback_min_number_of_violated_circuits():
    """Fallback: fewer violated circuits → the 'least problematic' model."""
    y, y_oof, y_full = _honest_vectors()
    gate = _gate()
    mocker_gate = MagicMock()
    mocker_gate.check.side_effect = [
        _result(["Circuit D (generalization gap): gap=2.0 > 1.5"]),
        _result(
            [
                "Circuit A (diversity): ratio=0.05 < 0.15",
                "Circuit B (unique): nunique=2 < 5",
            ]
        ),
    ]
    cands = {
        "one_violation": _candidate("one_violation", -0.08, 0.3, y_oof),
        "two_violations": _candidate("two_violations", -0.09, 0.2, y_oof),
    }
    res = select_robust_winner(
        cands,
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        mocker_gate,
        mode="active",
    )
    assert res.fallback_used is True
    assert res.winner_algo == "one_violation"


def test_fallback_tie_breaks_by_severity_circuit_order():
    """Fallback: equal violation counts → severity in the order D > A > B > C.

    A candidate violating A (rank 2) is 'lighter' than one violating D (rank 3).
    """
    y, y_oof, y_full = _honest_vectors()
    gate = _gate()
    mocker_gate = MagicMock()
    mocker_gate.check.side_effect = [
        _result(["Circuit D (generalization gap): gap=2.0 > 1.5"]),  # D — the heaviest
        _result(["Circuit A (diversity): ratio=0.05 < 0.15"]),  # A — lighter than D
    ]
    cands = {
        "violates_G": _candidate("violates_G", -0.08, 0.3, y_oof),
        "violates_A": _candidate("violates_A", -0.09, 0.3, y_oof),
    }
    res = select_robust_winner(
        cands,
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        mocker_gate,
        mode="active",
    )
    assert res.winner_algo == "violates_A"


def test_fallback_tie_breaks_by_best_rmse_oof():
    """Fallback: equal violation count and severity → the best RMSE_oof."""
    y, y_oof, y_full = _honest_vectors()
    mocker_gate = MagicMock()
    mocker_gate.check.side_effect = [
        _result(["Circuit A (diversity): ratio=0.05 < 0.15"]),
        _result(["Circuit A (diversity): ratio=0.05 < 0.15"]),
    ]
    cands = {
        "worse_oof": _candidate("worse_oof", -0.08, 0.5, y_oof),
        "better_oof": _candidate("better_oof", -0.09, 0.2, y_oof),
    }
    res = select_robust_winner(
        cands,
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        mocker_gate,
        mode="active",
    )
    assert res.winner_algo == "better_oof"


def test_fallback_logs_warning_with_reasons(caplog):
    """Fallback: a WARNING with the reasons is logged; the run does not fail."""
    y, y_oof, y_full = _honest_vectors()
    mocker_gate = MagicMock()
    mocker_gate.check.side_effect = [
        _result(["Circuit D (generalization gap): gap=2.0 > 1.5"]),
    ]
    cands = {"only": _candidate("only", -0.08, 0.3, y_oof)}
    with caplog.at_level("WARNING", logger=_COMPONENT_LOGGER):
        res = select_robust_winner(
            cands,
            pd.DataFrame(np.zeros((len(y), 2))),
            pd.Series(y),
            mocker_gate,
            mode="active",
        )
    assert res.fallback_used is True
    assert res.winner_algo == "only"
    assert any("falling back" in r.message for r in caplog.records)
    assert any("Circuit D" in r.message for r in caplog.records)


def test_circuit_rank_order_fixed():
    """Circuit significance is fixed in the code: D(3) > A(2) > B(1) > C(0)."""
    assert _circuit_rank("Circuit A (diversity): ...") == 2
    assert _circuit_rank("Circuit B (unique): ...") == 1
    assert _circuit_rank("Circuit C (dead features): ...") == 0
    assert _circuit_rank("Circuit D (generalization gap): ...") == 3
    assert _circuit_rank("custom reason without prefix") == 0


def test_severity_signature_sorted_descending():
    sig = _severity_signature(
        ["Circuit B (unique): ...", "Circuit D (generalization gap): ..."]
    )
    assert sig == (3, 1)  # D first, B second


# ──────────────────────────────────────────────────────────────────────────────
#  Negative: real degenerate models (Circuits D and A)
# ──────────────────────────────────────────────────────────────────────────────


def test_negative_deep_tree_rejected_by_circuit_g():
    """A deep tree (RMSE_train≈0) is cut off by Circuit D before ranking.

    Explicit ``generalization_gap_mode="reject"``: the default warn mode
    moves RMSE_full≈0 into soft_warnings and does not disqualify.
    """
    X, y = _dataset(n=120)
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        m = DecisionTreeRegressor(max_depth=20, random_state=0)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    tree = DecisionTreeRegressor(max_depth=20, random_state=0).fit(X, y)
    y_full = tree.predict(X)
    cand = _candidate(
        "deep_tree",
        cv_score=-0.05,
        rmse_oof=float(oof_rmse(y, oof)),
        y_oof=oof,
        y_full=y_full,
        model=tree,
    )
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, generalization_gap_mode="reject"
    )
    res = select_robust_winner(
        {"deep_tree": cand}, pd.DataFrame(X), pd.Series(y), gate, mode="active"
    )
    assert any("Circuit D" in r for r in res.disqualified["deep_tree"])
    assert res.fallback_used is True  # the only candidate is rejected


def test_warn_mode_extra_trees_with_infinite_gap_wins_by_rmse_oof():
    """Warn mode: a candidate with gap=∞ is not disqualified.

    ExtraTrees (RMSE_full≈0, gap=inf, best RMSE_oof) stays valid (Circuit D
    in soft_warnings), participates in the RMSE_oof ranking and wins —
    instead of silently replacing the winner with the 'best of those that
    passed the gate'.
    """
    X, y = make_friedman3(n_samples=200, noise=0.05, random_state=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_et = np.full(len(y), np.nan)
    oof_ridge = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        et = ExtraTreesRegressor(random_state=0).fit(X[tr], y[tr])
        rg = Ridge().fit(X[tr], y[tr])
        oof_et[te] = et.predict(X[te])
        oof_ridge[te] = rg.predict(X[te])
    et = ExtraTreesRegressor(random_state=0).fit(X, y)
    y_full = et.predict(X)
    assert np.sqrt(np.mean((y - y_full) ** 2)) < 1e-12  # RMSE_full ≈ 0
    cands = {
        "extra_trees": _candidate(
            "extra_trees",
            cv_score=-0.05,
            rmse_oof=float(oof_rmse(y, oof_et)),
            y_oof=oof_et,
            y_full=y_full,
            model=et,
        ),
        "ridge": _candidate(
            "ridge",
            cv_score=-0.08,
            rmse_oof=float(oof_rmse(y, oof_ridge)),
            y_oof=oof_ridge,
            y_full=y_full,
        ),
    }
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        generalization_gap_mode="warn",
    )
    res = select_robust_winner(
        cands, pd.DataFrame(X), pd.Series(y), gate, mode="active"
    )
    assert res.winner_algo == "extra_trees"
    assert "extra_trees" not in res.disqualified
    assert res.fallback_used is False
    assert np.isinf(res.audit["extra_trees"].generalization_gap)
    assert any(
        "Circuit D" in w and "RMSE_full" in w
        for w in res.audit["extra_trees"].soft_warnings
    )


def test_negative_svr_sigmoid_plateau_rejected_by_circuit_a():
    """SVR-sigmoid on the tanh plateau is cut off by Circuit A (diversity on OOF)."""
    X, y = _dataset(n=150, seed=11)
    sigmoid = SVR(kernel="sigmoid", gamma=0.0, coef0=0.0).fit(X, y)
    pred = sigmoid.predict(X)
    assert pred.std() < 0.1 * y.std()  # precondition: the tanh plateau
    cand = _candidate(
        "svr_sigmoid",
        cv_score=-0.05,
        rmse_oof=float(oof_rmse(y, pred)),
        y_oof=pred,
        y_full=pred,
        model=sigmoid,
    )
    gate = ModelSanityGate(check_generalization_gap=False)
    res = select_robust_winner(
        {"svr_sigmoid": cand}, pd.DataFrame(X), pd.Series(y), gate, mode="active"
    )
    assert any("Circuit A" in r for r in res.disqualified["svr_sigmoid"])


# ──────────────────────────────────────────────────────────────────────────────
#  Integration with train_best_model (explicit select phase)
# ──────────────────────────────────────────────────────────────────────────────


# A terminal select with strategy: sanity_gate replaces the implicit gate.
_SANITY_CFG = """
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  validation_strategy: train_test_split
  phases:
    - name: "Search"
      n_trials: 2
      action: "all_algorithms"
    - name: "GateSelect"
      n_trials: 1
      action: "select"
      strategy: "{strategy}"
      output: "{output}"
  sanity_gate:
    top_k_candidates: 2
algorithms:
  ridge:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
  elasticnet:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
      l1_ratio: [0.2, 0.8]
"""


def _tiny_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, 4))
    df["target"] = 2.0 * df[0] + rng.randn(n) * 0.3
    return df


def test_train_best_model_cv_best_select_has_no_sanity_key(tmp_path: Path):
    """select(cv_best, winner): artifact without the sanity_gate key (no audit)."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        _SANITY_CFG.format(model_path="m.pkl", strategy="cv_best", output="winner"),
        encoding="utf-8",
    )
    res = train_best_model(cfg_file, _tiny_df(), model_path_override=tmp_path / "m.pkl")
    assert "sanity_gate" not in res
    assert Path(res["model_path"]).exists()
    assert res["algorithm"] in {"ridge", "elasticnet"}


def test_train_best_model_sanity_gate_select_report(tmp_path: Path):
    """select(sanity_gate, winner): the report section is keyed by the phase name."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        _SANITY_CFG.format(model_path="m.pkl", strategy="sanity_gate", output="winner"),
        encoding="utf-8",
    )
    res = train_best_model(cfg_file, _tiny_df(), model_path_override=tmp_path / "m.pkl")
    assert Path(res["model_path"]).exists()
    # Report sections are keyed by the select phase name.
    assert set(res["sanity_gate"]) == {"GateSelect"}
    sg = res["sanity_gate"]["GateSelect"]
    assert sg["strategy"] == "sanity_gate"
    assert set(sg["pool"]) <= {"ridge", "elasticnet"}
    assert sg["pool"]  # non-empty
    assert "audit" in sg
    assert isinstance(sg["winner_rmse_oof"], float)
    assert sg["overhead_seconds"] >= 0
    # The winner must come from the pool
    assert res["algorithm"] in sg["pool"]
    # Audits were run for all pool finalists
    assert set(sg["audit"].keys()) == set(sg["pool"])
    # The winner is valid or the fallback fired (the whole pool was rejected).
    assert (
        sg["audit"][res["algorithm"]]["is_valid"] is True or sg["fallback_used"] is True
    )
    # The removed warn_only keys no longer appear.
    for key in ("disqualified_by_sanity_gate", "sanity_gate_warn_only"):
        assert key not in res


def test_train_best_model_sanity_winner_not_retrained_in_persist(
    tmp_path: Path, mocker
):
    """Finalists are trained once: persist_artifact does not retrain the winner.

    In the two-stage mode a finalist's training on 100% of the data has
    already been done; ``persist_artifact`` receives the ready ``ModelTrainer``
    and does not call ``_fit_and_save`` (no duplicated training).
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        _SANITY_CFG.format(model_path="m.pkl", strategy="sanity_gate", output="winner"),
        encoding="utf-8",
    )
    fit_and_save = mocker.patch(
        "configurable_automl_engine.training_engine.artifact._fit_and_save"
    )
    res = train_best_model(cfg_file, _tiny_df(), model_path_override=tmp_path / "m.pkl")
    assert Path(res["model_path"]).exists()
    fit_and_save.assert_not_called()


class _FakeTrainer:
    """A minimal fake ModelTrainer for the _train_finalists tests."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.pipeline = object()
        self.oof_score_ = 0.42
        self.oof_predictions_ = np.zeros(n)

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))


def test_train_finalists_failure_skips_candidate_and_logs_reason(
    mocker, caplog, tmp_path: Path
):
    """A finalist training failure → move on to the next one; the reason is logged."""
    import configurable_automl_engine.training_engine.component as comp
    import configurable_automl_engine.training_engine.finalists as finalists_mod

    cfg = Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            phases=[
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
        ),
        algorithms={
            "ridge": {"enable": True},
            "elasticnet": {"enable": True},
        },
    )
    df = _tiny_df()
    prepared = comp.prepare_dataset(cfg, df, "target")
    phase_results = {
        "ridge": PhaseResult(score=-0.08, params={"alpha": 1.0}),
        "elasticnet": PhaseResult(score=-0.081, params={"alpha": 0.5, "l1_ratio": 0.5}),
    }

    def _fake_fit(algo_name, *args, **kwargs):
        if algo_name == "elasticnet":
            raise RuntimeError("boom: fit failed")
        return _FakeTrainer(len(prepared.y))

    mocker.patch.object(finalists_mod, "_fit_finalist", side_effect=_fake_fit)
    with caplog.at_level("WARNING", logger=_COMPONENT_LOGGER):
        candidates = finalists_mod._train_finalists(
            ["elasticnet", "ridge"], phase_results, cfg, prepared, cfg.general.random_state
        )
    assert set(candidates) == {"ridge"}
    assert any(
        "elasticnet" in r.message and "boom" in r.message for r in caplog.records
    )


def test_train_best_model_all_finalists_fail_raises(tmp_path: Path):
    """All finalists failed to train on 100% of the data → a clear RuntimeError."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        _SANITY_CFG.format(model_path="m.pkl", strategy="sanity_gate", output="winner"),
        encoding="utf-8",
    )
    from unittest.mock import patch

    with patch(
        "configurable_automl_engine.training_engine.finalists._fit_finalist",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="No finalist could be trained"):
            train_best_model(
                cfg_file, _tiny_df(), model_path_override=tmp_path / "m.pkl"
            )


# ──────────────────────────────────────────────────────────────────────────────
#  SanityGateCfg configuration
# ──────────────────────────────────────────────────────────────────────────────


def test_sanity_gate_config_defaults():
    cfg = Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            phases=[
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
        ),
        algorithms={"ridge": {"enable": True}},
    )
    sg = cfg.general.sanity_gate
    assert "mode" not in SanityGateCfg.model_fields  # the mode field was removed
    assert sg.top_k_candidates == 3
    assert sg.corridor_delta == 0.15
    assert sg.corridor_mode == "auto"
    assert sg.enforce_family_diversity is False
    assert sg.max_generalization_gap == 1.5
    assert sg.check_permutation_sensitivity is True
    assert sg.audit_time_budget_seconds == 0


def test_sanity_gate_config_parses_mode_and_thresholds():
    data = {
        "general": {
            "comparison_metric": "rmse",
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
            "sanity_gate": {
                "top_k_candidates": 5,
                "corridor_delta": 0.2,
                "min_prediction_diversity": 0.25,
                "max_generalization_gap": 2.0,
            },
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    cfg = Config.model_validate(data)
    sg = cfg.general.sanity_gate
    assert sg.top_k_candidates == 5
    assert sg.corridor_delta == 0.2
    assert sg.min_prediction_diversity == 0.25
    assert sg.max_generalization_gap == 2.0


@pytest.mark.parametrize("legacy_mode", ["off", "warn_only", "active", "weird"])
def test_sanity_gate_config_mode_removed_with_migration_hint(legacy_mode):
    """The sanity_gate.mode field was removed: an error with a migration hint.

    The error is identified by a stable code
    (``ERR_REMOVED_SANITY_GATE_MODE`` in error['type']); the meaningful
    parts of the hint (mentioning the field and the select phase) are
    checked separately.
    """
    from pydantic import ValidationError

    data = {
        "general": {
            "comparison_metric": "rmse",
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
            "sanity_gate": {"mode": legacy_mode},
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(data)
    error_types = {e["type"] for e in exc_info.value.errors()}
    assert ERR_REMOVED_SANITY_GATE_MODE in error_types
    msg = str(exc_info.value)
    assert "sanity_gate.mode" in msg
    assert "select" in msg


@pytest.mark.parametrize(
    "bad, expected",
    [
        ({"top_k_candidates": 0}, "top_k_candidates"),
        ({"corridor_delta": -0.1}, "corridor_delta"),
        ({"corridor_delta": 0}, "corridor_delta"),
        ({"corridor_delta": 1.0}, "corridor_delta"),
        ({"min_prediction_diversity": 0}, "min_prediction_diversity"),
        ({"min_prediction_diversity": 1.5}, "min_prediction_diversity"),
        ({"min_unique_ratio": 0}, "min_unique_ratio"),
        ({"min_unique_ratio": 1.5}, "min_unique_ratio"),
        ({"max_generalization_gap": 0.5}, "max_generalization_gap"),
        ({"permutation_repeats": 0}, "permutation_repeats"),
        ({"permutation_max_rows": 0}, "permutation_max_rows"),
        ({"permutation_max_features": -1}, "permutation_max_features"),
        ({"audit_time_budget_seconds": -1}, "audit_time_budget_seconds"),
        ({"unknown_key": True}, "unknown_key"),
    ],
)
def test_sanity_gate_config_invalid_values_rejected(bad, expected):
    from pydantic import ValidationError

    data = {
        "general": {
            "comparison_metric": "rmse",
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
            "sanity_gate": bad,
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    with pytest.raises(ValidationError, match=expected):
        Config.model_validate(data)


def test_sanity_gate_config_empty_block_is_valid_thresholds_only():
    """An empty sanity_gate block (``{}``) is valid — a source of thresholds
    without a behavioral flag (the mode field was removed)."""
    data = {
        "general": {
            "comparison_metric": "rmse",
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
            "sanity_gate": {},
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    cfg = Config.model_validate(data)
    sg = cfg.general.sanity_gate
    assert "mode" not in SanityGateCfg.model_fields
    assert sg.corridor_delta == 0.15
    assert sg.min_prediction_diversity == 0.15
    assert sg.adaptive_diversity is True  # the constant-model baseline default
    assert sg.max_dead_feature_ratio == 0.4
    assert sg.max_generalization_gap == 1.5


def test_sanity_gate_config_full_v2_block_parses():
    """The full sanity_gate v2 block parses."""
    data = {
        "general": {
            "comparison_metric": "rmse",
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
            "sanity_gate": {
                "top_k_candidates": 5,
                "corridor_delta": 0.2,
                "corridor_mode": "multiplicative",
                "min_prediction_diversity": 0.25,
                "max_dead_feature_ratio": 0.5,
                "dead_features_require_low_diversity": True,
                "max_generalization_gap": 2.0,
                "generalization_gap_mode": "warn",
                "min_unique_ratio": 0.1,
                "min_unique_count": 10,
                "enforce_family_diversity": True,
                "algorithm_families": {"ridge": "linear", "svr": "kernel"},
                "allowed_families": ["linear", "kernel"],
                "family_diversity_multiplier": 1.3,
                "check_permutation_sensitivity": False,
                "permutation_max_rows": 2000,
                "permutation_max_features": 50,
                "permutation_repeats": 5,
                "permutation_seed": 7,
                "permutation_tolerance": 1e-2,
                "audit_time_budget_seconds": 120.5,
                "adaptive_diversity": True,
                "zero_coef_tolerance": 1e-10,
                "check_diversity": True,
                "check_unique": True,
                "check_dead_features": True,
                "check_generalization_gap": True,
            },
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    cfg = Config.model_validate(data)
    sg = cfg.general.sanity_gate
    assert sg.top_k_candidates == 5
    assert sg.corridor_delta == 0.2
    assert sg.corridor_mode == "multiplicative"
    assert sg.min_prediction_diversity == 0.25
    assert sg.max_dead_feature_ratio == 0.5
    assert sg.max_generalization_gap == 2.0
    assert sg.generalization_gap_mode == "warn"
    assert sg.min_unique_ratio == 0.1
    assert sg.min_unique_count == 10
    assert sg.enforce_family_diversity is True
    assert sg.check_permutation_sensitivity is False
    assert sg.permutation_max_rows == 2000
    assert sg.permutation_max_features == 50
    assert sg.permutation_repeats == 5
    assert sg.permutation_seed == 7
    assert sg.permutation_tolerance == 1e-2
    assert sg.audit_time_budget_seconds == 120.5
    assert sg.dead_features_require_low_diversity is True


def test_sanity_gate_config_cost_control_defaults():
    """Defaults of the new fields: permutations on, unlimited budget."""
    cfg = Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            phases=[
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
        ),
        algorithms={"ridge": {"enable": True}},
    )
    sg = cfg.general.sanity_gate
    assert sg.check_permutation_sensitivity is True
    assert sg.permutation_max_rows == 5000
    assert sg.permutation_max_features == 100
    assert sg.permutation_repeats == 3
    assert sg.permutation_seed == 42
    assert sg.audit_time_budget_seconds == 0


@pytest.mark.parametrize(
    "values",
    [
        # Boundary values (equality with the threshold is valid).
        {"min_prediction_diversity": 1.0},
        {"max_generalization_gap": 1.0},
        {"min_unique_ratio": 1.0},
        {"max_dead_feature_ratio": 1.0},
        {"max_dead_feature_ratio": 0.0},
        # corridor_delta → 0 (strictly positive, but arbitrarily small).
        {"corridor_delta": 1e-9},
        # top_k_candidates=1 — allowed (a pool of one leader).
        {"top_k_candidates": 1},
        # audit_time_budget_seconds=0 — no limit.
        {"audit_time_budget_seconds": 0},
    ],
)
def test_sanity_gate_config_boundary_values_valid(values):
    data = {
        "general": {
            "comparison_metric": "rmse",
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                FINAL_SELECT_PHASE,
            ],
            "sanity_gate": values,
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    cfg = Config.model_validate(data)
    for name, expected in values.items():
        assert getattr(cfg.general.sanity_gate, name) == expected


# ──────────────────────────────────────────────────────────────────────────────
#  Sanity Gate report and logs
# ──────────────────────────────────────────────────────────────────────────────


def test_active_disqualification_logs_warning_per_candidate(caplog):
    """Logging: each rejected candidate logs a WARNING with the reasons."""
    y, y_oof, y_full = _honest_vectors()
    mocker_gate = MagicMock()
    mocker_gate.check.side_effect = [
        _result(["Circuit A (diversity): ratio=0.05 < 0.15"]),
        SanityCheckResult(is_valid=True, reasons=[], severity=0),
    ]
    cands = {
        "flat": _candidate(
            "flat",
            cv_score=-0.08,
            rmse_oof=0.3,
            y_oof=_flat_vectors(y)[0],
            y_full=_flat_vectors(y)[1],
        ),
        "honest": _candidate("honest", cv_score=-0.09, rmse_oof=0.2, y_oof=y_oof),
    }
    with caplog.at_level("WARNING", logger=_COMPONENT_LOGGER):
        res = select_robust_winner(
            cands,
            pd.DataFrame(np.zeros((len(y), 2))),
            pd.Series(y),
            mocker_gate,
            mode="active",
        )
    assert res.winner_algo == "honest"
    disq_msgs = [
        r for r in caplog.records if "disqualified by the sanity gate" in r.message
    ]
    assert len(disq_msgs) == 1
    assert "flat" in disq_msgs[0].message
    assert "Circuit A" in disq_msgs[0].message
    assert disq_msgs[0].levelno == logging.WARNING


def test_fallback_logs_formal_criterion(caplog):
    """Logging: the fallback WARNING names the applied formal criterion."""
    y, y_oof, y_full = _honest_vectors()
    mocker_gate = MagicMock()
    mocker_gate.check.side_effect = [
        _result(["Circuit D (generalization gap): gap=2.0 > 1.5"]),
    ]
    cands = {"only": _candidate("only", -0.08, 0.3, y_oof)}
    with caplog.at_level("WARNING", logger=_COMPONENT_LOGGER):
        res = select_robust_winner(
            cands,
            pd.DataFrame(np.zeros((len(y), 2))),
            pd.Series(y),
            mocker_gate,
            mode="active",
        )
    assert res.fallback_used is True
    fallback_msgs = [r for r in caplog.records if "falling back" in r.message]
    assert len(fallback_msgs) == 1
    assert "Formal criterion" in fallback_msgs[0].message
    assert "min violated circuits" in fallback_msgs[0].message
    assert "D > A > B > C" in fallback_msgs[0].message


# A configuration with a strict Circuit D threshold (gap > 1.0): linear
# finalists on noisy data almost surely produce RMSE_oof > RMSE_full, so
# the pool is guaranteed to be disqualified — a deterministic scenario.
_SANITY_CFG_STRICT_GAP = """
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  validation_strategy: train_test_split
  phases:
    - name: "Search"
      n_trials: 2
      action: "all_algorithms"
    - name: "GateSelect"
      n_trials: 1
      action: "select"
      strategy: "sanity_gate"
      output: "winner"
  sanity_gate:
    top_k_candidates: 2
    max_generalization_gap: 1.0
    generalization_gap_mode: reject
algorithms:
  ridge:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
  elasticnet:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
      l1_ratio: [0.2, 0.8]
"""


def _run_sanity(tmp_path: Path) -> dict[str, Any]:
    """Run train_best_model with a strict Circuit D threshold (determinism)."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        _SANITY_CFG_STRICT_GAP.format(model_path="m.pkl"),
        encoding="utf-8",
    )
    return train_best_model(
        cfg_file, _tiny_df(), model_path_override=tmp_path / "m.pkl"
    )


def test_train_best_model_sanity_gate_disqualifications_in_circuit_breaker(
    tmp_path: Path,
):
    """Positive: gate disqualifications reach the circuit breaker.

    The reasons are available in the report's gate section; when the whole
    pool fails, the fallback fires and the run does not fail.
    """
    res = _run_sanity(tmp_path)
    sg = res["sanity_gate"]["GateSelect"]
    dq = sg["disqualified"]
    assert dq  # present and non-empty
    assert isinstance(dq, dict)
    assert set(dq) <= {"ridge", "elasticnet"}
    assert all(isinstance(reasons, list) and reasons for reasons in dq.values())
    assert all(
        any("Circuit D" in reason for reason in reasons) for reasons in dq.values()
    )
    # Gate disqualifications reach the circuit breaker
    # (disqualified_algorithms) alongside HPO failures.
    cb = res.get("disqualified_algorithms")
    assert cb
    assert set(cb) == set(dq)
    assert all("Circuit D" in reason for reason in cb.values())
    # The whole pool was rejected → fallback with the formal criterion.
    assert sg["fallback_used"] is True
    # The removed warn_only keys no longer appear.
    for key in ("disqualified_by_sanity_gate", "sanity_gate_warn_only"):
        assert key not in res


def test_train_best_model_sanity_gate_overhead_and_winner_audit_logged(
    tmp_path: Path, caplog
):
    """overhead_seconds in the gate section and an INFO log of the winner audit summary."""
    with caplog.at_level(logging.INFO, logger=_COMPONENT_LOGGER):
        res = _run_sanity(tmp_path)
    sg = res["sanity_gate"]["GateSelect"]
    assert isinstance(sg["overhead_seconds"], float)
    assert sg["overhead_seconds"] >= 0
    # INFO with the winner's audit summary: diversity/unique/dead features/gap.
    assert re.search(
        r"\[sanity_gate\] Winner \w+ audit summary: "
        r"diversity_ratio=(?:-?\d+\.\d{4}|n/a), "
        r"unique_ratio=(?:-?\d+\.\d{4}|n/a), "
        r"dead_features_count=\d+, "
        r"generalization_gap=(?:-?\d+\.\d{4}|n/a)",
        caplog.text,
    )
