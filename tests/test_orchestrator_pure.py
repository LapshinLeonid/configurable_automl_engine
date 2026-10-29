"""Direct unit tests of the pure orchestrator steps.

The select steps (``select_step_outcome``), the HPO worker result sorting
(``_consume_worker_results``) and the target-domain guard
(``_apply_target_domain_guard``) only read their inputs and return an
outcome; ``component`` applies it to the ``PhaseContext``. Every test checks
that the inputs are unchanged and that two calls give equal outcomes
(``tests.purity.assert_pure``), then checks the outcome itself.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import configurable_automl_engine.training_engine.finalists as finalists_mod
import configurable_automl_engine.training_engine.robust_winner as robust_winner_mod
import configurable_automl_engine.training_engine.select_steps as select_steps_mod
from configurable_automl_engine.hpo.types import HPOHistoryError
from configurable_automl_engine.training_engine import component
from configurable_automl_engine.training_engine.config_parser import AlgoCfg
from configurable_automl_engine.training_engine.hpo_phases import (
    _apply_target_domain_guard,
    _consume_worker_results,
)
from configurable_automl_engine.training_engine.sanity_gate import SanityCheckResult
from configurable_automl_engine.training_engine.select_steps import (
    _drop_kept_from_circuit_breaker,
    _gate_disqualifications,
    select_step_outcome,
)
from configurable_automl_engine.training_engine.state import (
    PhaseContext,
    PhaseResult,
    SelectOutcome,
    _HPOWorkerResult,
)
from tests.purity import assert_pure
from tests.test_select_phase import (
    _GATE_THRESHOLDS,
    _hpo_cfg,
    _prepared,
    _result,
    _select_phase,
)


def _ctx(**fields: Any) -> PhaseContext:
    working_set = {
        "ridge": PhaseResult(score=-0.10, params={"p": 1}),
        "extra_trees": PhaseResult(score=-0.08, params={"p": 2}),
    }
    return PhaseContext(working_set=working_set, **fields)


class _Trainer:
    """A full-fit trainer stand-in with perfect predictions."""

    def __init__(self, y: np.ndarray) -> None:
        self.oof_predictions_ = y.copy()
        self.oof_score_ = 0.0
        self.pipeline = None
        self.y = y

    def predict(self, X: Any) -> np.ndarray:
        return self.y[: len(X)]


# ── select steps ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("output", "k", "expected"),
    [("winner", None, ["extra_trees"]), ("top_k", 2, ["extra_trees", "ridge"])],
)
def test_select_cv_best_outcome_is_pure(output, k, expected):
    ctx = _ctx(
        disqualified_algorithms={"lasso": "HPO failure"},
        full_fit_trainers={"ridge": "kept"},
    )
    outcome = assert_pure(
        select_step_outcome,
        _select_phase(output=output, k=k),
        _hpo_cfg(),
        _prepared(),
        ctx,
    )
    assert isinstance(outcome, SelectOutcome)
    assert list(outcome.working_set) == expected
    assert outcome.disqualified_algorithms == {"lasso": "HPO failure"}
    assert outcome.sanity_gate_section is None
    if output == "winner":
        assert outcome.winner_algo == "extra_trees"
        assert outcome.pre_trained_trainer is None
    else:
        assert outcome.winner_algo is None
        # cv_best keeps the full-fit trainers of an earlier sanity select.
        assert outcome.full_fit_trainers == {"ridge": "kept"}


@pytest.mark.parametrize("output", ["winner", "pool"])
def test_select_sanity_gate_outcome_is_pure(output):
    prepared = _prepared()
    y = prepared.y.to_numpy()
    ctx = _ctx(disqualified_algorithms={"lasso": "HPO failure"})
    robust = _result(
        ["extra_trees", "ridge"],
        {
            "extra_trees": SanityCheckResult(is_valid=True),
            "ridge": SanityCheckResult(is_valid=False, reasons=["Circuit A"]),
        },
        winner="extra_trees",
    )
    with (
        patch.object(
            finalists_mod, "_fit_finalist", side_effect=lambda *a, **kw: _Trainer(y)
        ),
        patch.object(select_steps_mod, "select_robust_winner", return_value=robust),
        patch.object(select_steps_mod.time, "monotonic", return_value=1.0),
    ):
        outcome = assert_pure(
            select_step_outcome,
            _select_phase(name="gate", strategy="sanity_gate", output=output),
            _hpo_cfg(**_GATE_THRESHOLDS),
            prepared,
            ctx,
        )
    assert list(outcome.working_set) == ["extra_trees"]
    assert set(outcome.full_fit_trainers) == {"extra_trees"}
    assert outcome.disqualified_algorithms == {
        "lasso": "HPO failure",
        "ridge": "Circuit A",
    }
    assert outcome.sanity_gate_section["pool"] == ["extra_trees", "ridge"]
    assert outcome.sanity_gate_section["overhead_seconds"] == 0.0
    if output == "winner":
        assert outcome.winner_algo == "extra_trees"
        assert outcome.pre_trained_trainer is outcome.full_fit_trainers["extra_trees"]
    else:
        assert outcome.winner_algo is None
        assert outcome.pre_trained_trainer is None


def test_apply_select_outcome_assigns_every_field():
    ctx = _ctx()
    outcome = SelectOutcome(
        working_set={"ridge": ctx.working_set["ridge"]},
        winner_algo="ridge",
        pre_trained_trainer="trainer",
        full_fit_trainers={"ridge": "trainer"},
        disqualified_algorithms={"lasso": "x"},
        sanity_gate_section={"strategy": "sanity_gate"},
    )
    component._apply_select_outcome(ctx, outcome, "gate")
    assert list(ctx.working_set) == ["ridge"]
    assert ctx.winner_algo == "ridge"
    assert ctx.pre_trained_trainer == "trainer"
    assert ctx.full_fit_trainers == {"ridge": "trainer"}
    assert ctx.disqualified_algorithms == {"lasso": "x"}
    assert ctx.sanity_gate_sections == {"gate": {"strategy": "sanity_gate"}}


def test_drop_kept_from_circuit_breaker_is_pure():
    result = assert_pure(
        _drop_kept_from_circuit_breaker,
        {"a": "x", "b": "y", "c": "z"},
        {"b"},
    )
    assert result == {"a": "x", "c": "z"}


def test_gate_disqualifications_skip_budget_skips():
    robust = _result(
        ["a", "b"],
        {
            "a": SanityCheckResult(is_valid=False, reasons=["Circuit A"]),
            "b": SanityCheckResult(
                is_valid=False,
                reasons=[
                    f"{robust_winner_mod._AuditSkipReason.BUDGET_EXCEEDED.value} (5s) exceeded"
                ],
            ),
        },
        winner="a",
    )
    merged, this_step = assert_pure(_gate_disqualifications, {"c": "old"}, robust)
    assert merged == {"c": "old", "a": "Circuit A"}
    assert this_step == {"a"}


# ── HPO worker results ──────────────────────────────────────────────────────
def test_consume_worker_results_is_pure():
    oof = np.array([[1.0, 2.0]])
    results = [
        None,
        _HPOWorkerResult(name="ridge", score=0.5, params={"alpha": 1.0}),
        _HPOWorkerResult(name="svr", disqualify_reason="boom"),
        _HPOWorkerResult(name="lasso", score=None, params=None),
        _HPOWorkerResult(
            name="knn", score=0.4, params={"n_neighbors": 3}, oof_predictions=oof
        ),
    ]
    outcome = assert_pure(_consume_worker_results, results)
    assert list(outcome.results) == ["ridge", "knn"]
    assert outcome.results["ridge"].params == {"alpha": 1.0}
    assert outcome.disqualified == {"svr": "boom"}
    assert len(outcome.oof_pool) == 1 and outcome.oof_pool[0] is oof


def test_consume_worker_results_raises_history_error():
    with pytest.raises(HPOHistoryError, match="bad history"):
        _consume_worker_results(
            [_HPOWorkerResult(name="ridge", history_error="bad history")]
        )


def test_apply_worker_outcome_merges_in_order():
    ctx = PhaseContext(disqualified_algorithms={"a": "x"})
    phase_results: dict[str, PhaseResult] = {}
    outcome = _consume_worker_results(
        [
            _HPOWorkerResult(name="svr", disqualify_reason="boom"),
            _HPOWorkerResult(
                name="knn",
                score=0.4,
                params={},
                oof_predictions=np.zeros((1, 2)),
            ),
        ]
    )
    component._apply_worker_outcome(ctx, phase_results, outcome)
    assert ctx.disqualified_algorithms == {"a": "x", "svr": "boom"}
    assert list(phase_results) == ["knn"]
    assert len(ctx.oof_pool) == 1


# ── target-domain guard ─────────────────────────────────────────────────────
def test_target_domain_guard_is_pure():
    algos = {
        "gammaregressor": AlgoCfg(),
        "glm": AlgoCfg.model_validate({"hyperparameters": {"power": [-1.0, 3.0]}}),
        "ridge": AlgoCfg(),
    }
    y = pd.Series([-1.0, 0.0, 2.0, 3.0])
    outcome = assert_pure(_apply_target_domain_guard, algos, y, ("ridge",))
    assert set(outcome.disqualified) == {"gammaregressor"}
    assert outcome.algorithms["ridge"] is algos["ridge"]
    assert outcome.algorithms["glm"].hyperparameters["power"].config.high == 0.0
