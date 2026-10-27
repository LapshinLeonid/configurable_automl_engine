"""Tests for the explicit pipeline select phase.

Coverage:

1. Positive: select ``cv_best`` with each ``output`` (winner/top_k/pool);
   select ``sanity_gate`` with ``winner``/``top_k``/``pool``; select after
   select; persist_artifact of the terminal select without retraining (the
   finalist trainer is reused via ``pre_trained_trainer``).
2. Negative: full disqualification of a select → fallback; an empty select
   input set → RuntimeError.
3. Reporting: ``sanity_gate`` sections keyed by phase name; no gate section
   for ``cv_best``; the removed keys ``warn_only``/``winner_if_active`` no
   longer appear; gate disqualifications reach the circuit breaker.
4. Handler unit level: strategy ranking (``_rank_sanity_candidates``), audit
   serialization, gate assembly from the threshold section, and repeat
   sanity-select retrains the finalists (old full-fit trainers are not
   reused).

The config schema requires the last phase to be a terminal select
(``output: winner``), so the internals of intermediate select steps are
checked by direct handler calls on a handcrafted ``PhaseContext``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import configurable_automl_engine.training_engine.component as component
import configurable_automl_engine.training_engine.artifact as artifact_mod
import configurable_automl_engine.training_engine.finalists as finalists_mod
import configurable_automl_engine.training_engine.hpo_phases as hpo_phases_mod
import configurable_automl_engine.training_engine.select_steps as select_steps_mod
from configurable_automl_engine.training_engine.component import (
    FinalistCandidate,
    PhaseContext,
    PhaseResult,
    PreparedData,
    RobustWinnerResult,
    execute_phases,
    persist_artifact,
    prepare_dataset,
    train_best_model,
)
from configurable_automl_engine.training_engine.select_steps import (
    _build_sanity_gate,
    _rank_sanity_candidates,
    _serialize_audit_result,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    HPOPhaseCfg,
    SanityGateCfg,
    ValidationStrategy,
)
from configurable_automl_engine.training_engine.sanity_gate import SanityCheckResult

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


def _select_phase(**overrides: Any) -> HPOPhaseCfg:
    """Build an HPOPhaseCfg select phase on top of defaults."""
    base: dict[str, Any] = {
        "name": "sel",
        "n_trials": 1,
        "action": "select",
        "strategy": "cv_best",
        "output": "winner",
    }
    base.update(overrides)
    return HPOPhaseCfg(**base)


def _cfg_dict(
    phases: list[dict[str, Any]],
    *,
    extra_algorithms: dict[str, dict[str, Any]] | None = None,
    **general_extra: Any,
) -> dict[str, Any]:
    """A valid config: an HPO phase + the given phases (the terminal one at the end).

    Args:
        phases: Pipeline phases (added after the mandatory HPO phase).
        extra_algorithms: Extra algorithm sections (merged with the base
            ridge/extra_trees ones).
        **general_extra: Extra ``general`` fields (e.g. ``sanity_gate``).
    """
    algorithms: dict[str, dict[str, Any]] = {
        "ridge": {
            "enable": True,
        },
        "extra_trees": {
            "enable": True,
        },
    }
    if extra_algorithms:
        algorithms.update(extra_algorithms)
    return {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "train_test_split",
            "phases": [
                {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
                *phases,
            ],
            "path_to_model": "model.pkl",
            **general_extra,
        },
        "algorithms": algorithms,
        "oversampling": {"data_oversampling": False},
    }


def _df(n_rows: int = 20, seed: int = 0) -> pd.DataFrame:
    """A numeric DataFrame with a target column."""
    rng = np.random.RandomState(seed)
    data = {f"f{i}": rng.randn(n_rows) for i in range(3)}
    data["target"] = 2.0 * data["f0"] + rng.randn(n_rows) * 0.1
    return pd.DataFrame(data)


def _prepared(y: pd.Series | None = None) -> PreparedData:
    """A minimal PreparedData with defaults filled in."""
    if y is None:
        y = pd.Series([1.0, 2.0, 3.0])
    return PreparedData(
        X=pd.DataFrame({"a": np.arange(len(y), dtype=float)}),
        y=y,
        metric_user="mae",
        metric_sklearn="mae",
        resolved_validation=ValidationStrategy.train_test_split,
        resolved_n_folds=5,
        resolved_test_size=0.2,
        categorical_features=[],
        numerical_features=["a"],
    )


class _FakeTrainer:
    """A fake ModelTrainer with controllable OOF/full prediction quality.

    ``oof_rmse`` is the RMSE of the OOF vector relative to y; ``full_rmse``
    is the RMSE of the full-fit predictions (Circuit D: gap =
    oof_rmse / full_rmse).
    """

    def __init__(self, y: np.ndarray, oof_rmse: float, full_rmse: float) -> None:
        rng = np.random.RandomState(0)

        def _noise(scale: float) -> np.ndarray:
            noise = rng.randn(len(y))
            return noise / np.sqrt(np.mean(noise**2)) * scale

        # The pipeline has a predict method: Circuit C of the audit predicts on
        # permuted X (pipeline = self is enough).
        self.pipeline = self
        self.oof_predictions_ = y + _noise(oof_rmse)
        self.oof_score_ = oof_rmse
        self._full_pred = y + _noise(full_rmse)

    def predict(self, X: Any) -> np.ndarray:
        return self._full_pred[: len(X)]

    def save(self, path: Any) -> None:
        """The artifact is 'saved' by creating a file (persist_artifact)."""
        from pathlib import Path

        Path(path).write_bytes(b"fake-model")


def _make_ctx(scores: dict[str, PhaseResult]) -> PhaseContext:
    """A phase context with the given working set."""
    return PhaseContext(working_set=dict(scores))


def _hpo_cfg(**general_extra: Any) -> Config:
    """A config for direct handler calls (phases are not executed)."""
    return Config.model_validate(
        _cfg_dict(
            [
                {"name": "final", "n_trials": 1, "action": "all_algorithms"},
                _select_phase(),
            ],
            **general_extra,
        )
    )


_GATE_THRESHOLDS = {"sanity_gate": {"top_k_candidates": 2, "corridor_delta": 0.9}}


def _patch_hpo():
    """Mock HPO: deterministic scores — 'ridge' worse than 'extra_trees'."""
    return patch.object(
        hpo_phases_mod,
        "_run_hpo",
        side_effect=lambda **kw: (
            PhaseResult(score=-0.10, params={"p": 1}) if kw["request"].algo_name == "ridge" else PhaseResult(score=-0.08, params={"p": 1})
        ),
    )


def _patch_honest_finalists(y: pd.Series):
    """Mock finalist training with an honest trainer (gap ≈ 1.2 is valid)."""
    return patch.object(
        finalists_mod,
        "_fit_finalist",
        side_effect=lambda *a, **kw: _FakeTrainer(y.to_numpy(), 0.30, 0.25),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  select cv_best: each output (handler unit level)
# ──────────────────────────────────────────────────────────────────────────────
def test_select_cv_best_winner():
    """select(cv_best, winner): working set = CV leader, no gate section."""
    cfg = _hpo_cfg()
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    select_steps_mod._execute_select_phase(
        phase=_select_phase(), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"extra_trees"}
    assert ctx.winner_algo == "extra_trees"
    assert ctx.pre_trained_trainer is None
    assert ctx.sanity_gate_sections == {}


def test_select_cv_best_top_k():
    """select(cv_best, top_k): the k best by CV score."""
    cfg = _hpo_cfg()
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    select_steps_mod._execute_select_phase(
        phase=_select_phase(output="top_k", k=1), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"extra_trees"}
    assert ctx.winner_algo is None  # a non-terminal select does not fix the winner


def test_select_cv_best_top_k_tie_keeps_insertion_order():
    """select(cv_best, top_k): a CV-score tie keeps insertion order (tie-break)."""
    cfg = _hpo_cfg()
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.10, params={"p": 1})})
    select_steps_mod._execute_select_phase(
        phase=_select_phase(output="top_k", k=1), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"ridge"}


def test_select_cv_best_pool_corridor_and_cap():
    """select(cv_best, pool): corridor pool (δ ∩ top_k_candidates)."""
    cfg = _hpo_cfg(sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.05})
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    select_steps_mod._execute_select_phase(
        phase=_select_phase(output="pool"), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    # Narrow corridor δ=0.05: 0.10 > 0.08 * 1.05 → only the leader.
    assert set(ctx.working_set) == {"extra_trees"}

    # Wide corridor + top_k_candidates=2 cap: both candidates.
    cfg2 = _hpo_cfg(sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.9})
    ctx2 = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    select_steps_mod._execute_select_phase(
        phase=_select_phase(name="sel2", output="pool"),
        cfg=cfg2,
        prepared=_prepared(),
        ctx=ctx2,
    )
    assert list(ctx2.working_set) == ["extra_trees", "ridge"]
    assert ctx2.sanity_gate_sections == {}  # no gate section is created

    # Cap boundary: the pool is wide (both inside the δ=0.9 corridor), but
    # top_k_candidates=1 cuts it down to the leader.
    cfg3 = _hpo_cfg(sanity_gate={"top_k_candidates": 1, "corridor_delta": 0.9})
    ctx3 = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    select_steps_mod._execute_select_phase(
        phase=_select_phase(name="sel3", output="pool"),
        cfg=cfg3,
        prepared=_prepared(),
        ctx=ctx3,
    )
    assert list(ctx3.working_set) == ["extra_trees"]


def test_select_cv_best_pool_excludes_disqualified():
    """select(cv_best, pool): candidates disqualified by the circuit breaker are not in the pool."""
    cfg = _hpo_cfg(sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.9})
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    ctx.disqualified_algorithms = {"ridge": "HPO failure"}
    select_steps_mod._execute_select_phase(
        phase=_select_phase(output="pool"), cfg=cfg, prepared=_prepared(), ctx=ctx
    )
    assert set(ctx.working_set) == {"extra_trees"}


# ──────────────────────────────────────────────────────────────────────────────
#  select sanity_gate: each output (handler unit level)
# ──────────────────────────────────────────────────────────────────────────────
def _result(
    pool: list[str],
    audits: dict[str, SanityCheckResult],
    winner: str,
    fallback: bool = False,
) -> RobustWinnerResult:
    """A RobustWinnerResult for the ``select_robust_winner`` mock."""
    return RobustWinnerResult(
        winner_algo=winner,
        mode="active",
        pool=pool,
        audit=audits,
        disqualified={
            name: list(res.reasons) for name, res in audits.items() if res.reasons
        },
        fallback_used=fallback,
        winner_rmse_oof=0.2,
    )


def test_select_sanity_gate_winner_sets_pre_trained_trainer():
    """select(sanity_gate, winner): the winner's full-fit trainer goes into the context."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    trainer = object()

    def fake_fit(algo_name, *args, **kwargs):
        return _FakeTrainer(_prepared().y.to_numpy(), 0.2, 0.25)

    with (
        patch.object(finalists_mod, "_fit_finalist", side_effect=fake_fit),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ) as mock_select,
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="winner"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    mock_select.assert_called_once()
    assert mock_select.call_args.kwargs["mode"] == "active"
    assert set(ctx.working_set) == {"extra_trees"}
    assert ctx.winner_algo == "extra_trees"
    assert ctx.pre_trained_trainer is ctx.full_fit_trainers["extra_trees"]


def test_select_sanity_gate_top_k_ranks_by_rmse_oof():
    """select(sanity_gate, top_k): the k best by RMSE_oof (not by CV score)."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    # The CV leader is 'ridge', but its RMSE_oof is worse: top_k by
    # RMSE_oof picks 'extra_trees'.
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.08, params={"p": 1}), "extra_trees": PhaseResult(score=-0.09, params={"p": 1})})

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), {"ridge": 0.5, "extra_trees": 0.2}[a[0]], 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["ridge", "extra_trees"],
                {
                    "ridge": SanityCheckResult(is_valid=True),
                    "extra_trees": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=1
            ),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert set(ctx.working_set) == {"extra_trees"}
    section = ctx.sanity_gate_sections["gate"]
    assert section["pool"] == ["ridge", "extra_trees"]
    assert section["fallback_used"] is False
    assert section["overhead_seconds"] >= 0
    assert section["winner_rmse_oof"] == 0.2


def test_select_sanity_gate_pool_keeps_passed_candidates():
    """select(sanity_gate, pool): all candidates that passed the audit remain in the working set."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert list(ctx.working_set) == ["extra_trees", "ridge"]


def test_select_sanity_gate_pool_full_disqualification_falls_back_to_ranking():
    """select(sanity_gate, pool): total disqualification → fallback ranking.

    The pipeline does not end up with an empty set (documented fallback).
    Output invariant: fallback candidates that return to the working set are
    removed from the circuit breaker — otherwise the next phase would filter
    them out and fail. The audit is preserved in the gate section.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                    "ridge": SanityCheckResult(
                        is_valid=False,
                        reasons=["Circuit A", "Circuit B"],
                        severity=3,
                    ),
                },
                winner="extra_trees",
                fallback=True,
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # Fewer violations (extra_trees) — higher in the fallback ranking.
    assert list(ctx.working_set) == ["extra_trees", "ridge"]
    # Invariant: the working set does not overlap with the circuit breaker —
    # fallback candidates are not permanently disqualified, otherwise the
    # next refine/select would filter them out and crash with a RuntimeError.
    assert ctx.disqualified_algorithms == {}
    section = ctx.sanity_gate_sections["gate"]
    assert section["fallback_used"] is True
    # The detailed audit is preserved in the gate section of the report.
    assert set(section["disqualified"]) == {"extra_trees", "ridge"}


def test_select_sanity_gate_top_k_fallback_kept_candidate_not_disqualified():
    """select(sanity_gate, top_k): a fallback candidate in the output is not disqualified.

    The strategy ranking puts valid candidates first; if ``k`` also covers
    rejected ones (the fallback semantics of top_k), the candidates that
    return to the working set are removed from the circuit breaker —
    otherwise the next refine/select would filter them out and crash with a
    RuntimeError.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=2
            ),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert set(ctx.working_set) == {"extra_trees", "ridge"}
    # Invariant: ridge returned to the top_k output → not in the circuit breaker.
    assert ctx.disqualified_algorithms == {}


def test_select_sanity_gate_top_k_drops_failed_outside_k():
    """select(sanity_gate, top_k): a rejected candidate outside the slice stays disqualified."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=1
            ),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert set(ctx.working_set) == {"extra_trees"}
    # ridge outside top_k: excluded from the working set and stays in the circuit breaker.
    assert set(ctx.disqualified_algorithms) == {"ridge"}
    assert ctx.disqualified_algorithms["ridge"] == "Circuit A"


def test_select_sanity_gate_pool_partial_failure_keeps_only_passed():
    """select(sanity_gate, pool): partial audit failure.

    Passing candidates stay in the working set; truly rejected ones are out
    of the output and in the circuit breaker (they will not return in the
    following phases).
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    assert list(ctx.working_set) == ["extra_trees"]
    assert ctx.disqualified_algorithms == {"ridge": "Circuit A"}
    assert ctx.sanity_gate_sections["gate"]["fallback_used"] is False


def test_select_sanity_gate_budget_skipped_not_permanently_disqualified():
    """select(sanity_gate, pool): candidates not audited because of the
    ``audit_time_budget`` are not permanently disqualified.

    Skipping an audit is a phase resource limitation, not a model-quality
    issue: such a candidate cannot win the current select (no audit), but is
    not placed into the circuit breaker and can return in later phases.
    The reasons remain in the gate section of the report.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    # The reason format is taken from the code (single source).
    budget_reason = finalists_mod._audit_budget_skip_reason(5.0)

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=[budget_reason], severity=0
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # The passing one stays; the skipped one is not permanently disqualified.
    assert list(ctx.working_set) == ["extra_trees"]
    assert ctx.disqualified_algorithms == {}
    # The report preserves the skip reason.
    section = ctx.sanity_gate_sections["gate"]
    assert section["audit"]["ridge"]["is_valid"] is False
    assert section["disqualified"]["ridge"] == [budget_reason]


def test_select_sanity_gate_budget_skipped_outside_output_not_disqualified():
    """Budget-only reasons of a candidate OUTSIDE the working set.

    A candidate skipped by the budget did not make it into the ``pool``
    output (only valid ones did), but is NOT permanently disqualified —
    unlike a real failure, which stays in the circuit breaker. Both
    categories contrast within a single select step.
    """
    cfg = Config.model_validate(
        _cfg_dict(
            [
                _select_phase(name="gate", strategy="sanity_gate", output="pool"),
                _select_phase(name="final"),  # terminal select (schema validity)
            ],
            sanity_gate={"top_k_candidates": 3, "corridor_delta": 0.9},
            extra_algorithms={
                "svr": {
                    "enable": True,
                }
            },
        )
    )
    ctx = _make_ctx(
        {
            "extra_trees": PhaseResult(score=-0.08, params={"p": 1}),
            "ridge": PhaseResult(score=-0.10, params={"p": 1}),
            "svr": PhaseResult(score=-0.12, params={"p": 1}),
        }
    )
    budget_reason = finalists_mod._audit_budget_skip_reason(5.0)

    with (
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge", "svr"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(
                        is_valid=False, reasons=["Circuit A"], severity=2
                    ),
                    "svr": SanityCheckResult(
                        is_valid=False, reasons=[budget_reason], severity=0
                    ),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # Valid — in the output; a real failure — in the circuit breaker; the
    # budget-skipped one — neither in the output nor in the circuit breaker.
    assert list(ctx.working_set) == ["extra_trees"]
    assert ctx.disqualified_algorithms == {"ridge": "Circuit A"}
    assert "svr" not in ctx.disqualified_algorithms
    # The report preserves the reasons of both invalid categories.
    section = ctx.sanity_gate_sections["gate"]
    assert section["disqualified"]["ridge"] == ["Circuit A"]
    assert section["disqualified"]["svr"] == [budget_reason]


def test_select_sanity_gate_pool_keeps_pre_existing_disqualification():
    """select(sanity_gate, pool): a pre-existing disqualification is not lifted.

    Only the failures disqualified by this step and returned by the fallback
    into the output are lifted. A candidate disqualified earlier (for
    example, by the HPO circuit breaker) stays in the circuit breaker even
    if it ended up in the step output.
    """
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    ctx.disqualified_algorithms = {"ridge": "HPO failure"}

    with (
        patch.object(
            select_steps_mod,
            "_build_finalists_pool",
            return_value=["extra_trees", "ridge"],
        ),
        patch.object(
            finalists_mod,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                _prepared().y.to_numpy(), 0.2, 0.25
            ),
        ),
        patch.object(
            select_steps_mod,
            "select_robust_winner",
            return_value=_result(
                ["extra_trees", "ridge"],
                {
                    "extra_trees": SanityCheckResult(is_valid=True),
                    "ridge": SanityCheckResult(is_valid=True),
                },
                winner="extra_trees",
            ),
        ),
    ):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="gate", strategy="sanity_gate", output="pool"),
            cfg=cfg,
            prepared=_prepared(),
            ctx=ctx,
        )

    # Both are valid → both in the pool output; ridge did NOT fail the audit
    # in this step, so its pre-existing disqualification is kept.
    assert list(ctx.working_set) == ["extra_trees", "ridge"]
    assert ctx.disqualified_algorithms == {"ridge": "HPO failure"}


def test_is_audit_budget_skip_rejects_partial_prefix_match():
    """Full-match detection: a partial prefix match is not a budget skip.

    Previously ``startswith`` falsely triggered on any reason merely starting
    with the prefix; now the reason must fully match the
    '{prefix} (Ns) exceeded' format.
    """
    assert (
        finalists_mod._is_audit_budget_skip(
            ["Sanity audit skipped: audit time budget is not a reason"]
        )
        is False
    )
    assert finalists_mod._is_audit_budget_skip(["Sanity audit skipped by user"]) is False
    # A candidate with a real reason plus a 'similar' one is also not 'skip only'.
    assert (
        finalists_mod._is_audit_budget_skip(
            [finalists_mod._audit_budget_skip_reason(5.0), "Sanity audit skipped by user"]
        )
        is False
    )


def test_is_audit_budget_skip_detects_only_budget_reasons():
    """``_is_audit_budget_skip`` distinguishes a budget skip from a real failure."""
    reason = finalists_mod._audit_budget_skip_reason(5.0)
    assert finalists_mod._is_audit_budget_skip([reason]) is True
    # Mixed reasons (budget + circuit) — not 'skip only'.
    assert finalists_mod._is_audit_budget_skip([reason, "Circuit A"]) is False
    assert finalists_mod._is_audit_budget_skip(["Circuit A"]) is False
    assert finalists_mod._is_audit_budget_skip([]) is False


def test_select_after_select():
    """select after select: the first one's output replaces the second one's input."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1}), "extra_trees": PhaseResult(score=-0.08, params={"p": 1})})
    prepared = _prepared()

    with _patch_honest_finalists(prepared.y):
        # First select: sanity_gate top_k=1 (best by RMSE_oof — per the ranking).
        select_steps_mod._execute_select_phase(
            phase=_select_phase(
                name="gate", strategy="sanity_gate", output="top_k", k=1
            ),
            cfg=cfg,
            prepared=prepared,
            ctx=ctx,
        )
        assert len(ctx.working_set) == 1
        # Second select: cv_best winner narrows down to the terminal winner.
        select_steps_mod._execute_select_phase(
            phase=_select_phase(name="final"),
            cfg=cfg,
            prepared=prepared,
            ctx=ctx,
        )

    assert len(ctx.working_set) == 1
    assert ctx.winner_algo == list(ctx.working_set)[0]


def test_select_top_k_without_k_raises():
    """select(sanity_gate, top_k) without k → ValueError (the guard duplicates the schema)."""
    cfg = _hpo_cfg(**_GATE_THRESHOLDS)
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1})})
    phase = HPOPhaseCfg.model_construct(
        name="gate",
        n_trials=1,
        action="select",
        strategy="sanity_gate",
        output="top_k",
        k=None,
    )
    with pytest.raises(ValueError, match="requires"):
        select_steps_mod._execute_select_phase(
            phase=phase, cfg=cfg, prepared=_prepared(), ctx=ctx
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Integration through train_best_model / execute_phases
# ──────────────────────────────────────────────────────────────────────────────
def test_terminal_sanity_select_persists_without_retraining(tmp_path):
    """Terminal sanity-select: the artifact is persisted without retraining the winner."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)
    with (
        _patch_hpo(),
        _patch_honest_finalists(df["target"]),
        patch.object(artifact_mod, "_fit_and_save") as mock_fit,
    ):
        res = train_best_model(cfg, df, target="target")

    mock_fit.assert_not_called()  # the finalist trainer is reused
    import pathlib

    assert pathlib.Path(res["model_path"]).exists()
    sg = res["sanity_gate"]["gate"]
    assert res["algorithm"] in sg["pool"]
    assert sg["fallback_used"] is False
    assert sg["winner_rmse_oof"] is not None


def test_select_empty_working_set_raises():
    """An empty select input set → RuntimeError with a clear message."""
    cfg = _hpo_cfg()
    ctx = PhaseContext()
    with pytest.raises(RuntimeError, match="empty working set"):
        select_steps_mod._execute_select_phase(
            phase=_select_phase(), cfg=cfg, prepared=_prepared(), ctx=ctx
        )


def test_select_sanity_gate_full_disqualification_falls_back(tmp_path):
    """Total select disqualification → fallback; the run does not fail.

    A strict Circuit D threshold (gap > 1.0) with full-fit better than OOF
    disqualifies the whole pool: the winner is chosen by the fallback
    criterion, and disqualifications reach the circuit breaker and the
    report's gate section.
    """
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")],
        sanity_gate={
            "top_k_candidates": 2,
            "corridor_delta": 0.9,
            "max_generalization_gap": 1.0,
            "generalization_gap_mode": "reject",
        },
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)

    def _overfit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        # RMSE_oof=0.5 vs RMSE_full=0.05: gap=10 > 1.0 → Circuit D.
        return _FakeTrainer(df["target"].to_numpy(), 0.5, 0.05)

    with (
        _patch_hpo(),
        patch.object(finalists_mod, "_fit_finalist", side_effect=_overfit_finalist),
    ):
        res = train_best_model(cfg, df, target="target")

    sg = res["sanity_gate"]["gate"]
    assert sg["fallback_used"] is True
    assert sg["disqualified"]  # all finalists were rejected
    # Gate disqualifications reach the circuit breaker.
    assert set(res["disqualified_algorithms"]) == set(sg["disqualified"])
    assert all("Circuit D" in r for rs in sg["disqualified"].values() for r in rs)
    # The fallback winner comes from the finalists pool.
    assert res["algorithm"] in sg["pool"]


def test_pipeline_pool_fallback_survives_following_refine_and_select(tmp_path):
    """Regression: a fallback select(pool) does not break the following refine/select.

    The full disqualification of the first sanity-select returns the whole
    ranking into the working set. Without lifting the returned candidates
    from the circuit breaker, the next refine would filter everyone out and
    crash with a RuntimeError ('no valid results'), and select(sanity_gate)
    in ``select_finalists`` as well. With the fix, the DAG
    …select(pool) → refine → select(winner)… reaches the terminal winner.
    """
    cfg_dict = _cfg_dict(
        [
            _select_phase(name="gate_pool", strategy="sanity_gate", output="pool"),
            {"name": "refine", "n_trials": 2, "action": "refine"},
            _select_phase(name="gate_final", strategy="sanity_gate"),
        ],
        sanity_gate={
            "top_k_candidates": 2,
            "corridor_delta": 0.9,
            "max_generalization_gap": 1.0,
            "generalization_gap_mode": "reject",
        },
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)

    def _overfit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        # RMSE_oof=0.5 vs RMSE_full=0.05: gap=10 > 1.0 → Circuit D.
        return _FakeTrainer(df["target"].to_numpy(), 0.5, 0.05)

    with (
        _patch_hpo(),
        patch.object(finalists_mod, "_fit_finalist", side_effect=_overfit_finalist),
    ):
        res = train_best_model(cfg, df, target="target")

    assert res["algorithm"] in {"ridge", "extra_trees"}
    pool_section = res["sanity_gate"]["gate_pool"]
    final_section = res["sanity_gate"]["gate_final"]
    # First select: all finalists failed the audit → fallback ranking.
    assert pool_section["fallback_used"] is True
    assert set(pool_section["disqualified"]) == {"ridge", "extra_trees"}
    # The second select also falls back (the finalists are still degenerate) —
    # the final winner is chosen by the formal criterion.
    assert final_section["fallback_used"] is True
    assert res["algorithm"] in final_section["pool"]
    # The terminal select (winner) keeps the disqualifications in the circuit
    # breaker: the fallback winner is still recorded in the report.
    assert res["algorithm"] in res["disqualified_algorithms"]


def test_select_sanity_gate_warn_mode_keeps_gap_winner(tmp_path):
    """Warn mode of circuit D (default): overfitting does not disqualify.

    A candidate with a finite gap > 1.5 stays valid (a Circuit D soft_warning),
    wins by RMSE_oof, does not appear in ``disqualified`` or in the circuit
    breaker (``disqualified_algorithms``) — otherwise the next refine/select
    would lose it. The gate section shows the warning and the applied mode.
    """
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")],
        sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.9},
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)

    def _overfit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        # gap = 2.0 > 1.5 for both (a soft warning), but extra_trees has a better
        # RMSE_oof — it must win by RMSE_oof, not Ridge.
        if a[0] == "ridge":
            return _FakeTrainer(df["target"].to_numpy(), 0.5, 0.25)  # gap = 2.0
        return _FakeTrainer(df["target"].to_numpy(), 0.2, 0.1)  # gap = 2.0, better RMSE_oof

    with (
        _patch_hpo(),
        patch.object(finalists_mod, "_fit_finalist", side_effect=_overfit_finalist),
    ):
        res = train_best_model(cfg, df, target="target")

    sg = res["sanity_gate"]["gate"]
    assert sg["fallback_used"] is False
    assert sg["disqualified"] == {}
    assert "disqualified_algorithms" not in res  # circuit breaker empty
    assert sg["generalization_gap_mode"] == "warn"
    assert all(
        audit["generalization_gap_mode"] == "warn"
        for audit in sg["audit"].values()
    )
    # The best-by-RMSE_oof candidate won; both are valid with a Circuit D warning.
    assert res["algorithm"] == "extra_trees"
    assert sg["winner_rmse_oof"] == pytest.approx(0.2, rel=1e-6)
    for audit in sg["audit"].values():
        assert audit["is_valid"] is True
        assert audit["generalization_gap"] == pytest.approx(2.0, rel=1e-4)
        assert any(
            "Circuit D" in w and "max_generalization_gap=1.500" in w
            for w in audit["soft_warnings"]
        )
        assert audit["severity"] == 1


def test_select_sanity_gate_warn_mode_infinite_gap_winner(tmp_path):
    """Warn mode: a candidate with gap=∞ (RMSE_full≈0) wins.

    A candidate with zero training error is not disqualified: it wins by
    RMSE_oof, is absent from ``disqualified`` and the circuit breaker
    (``disqualified_algorithms``); the gate section shows a Circuit D
    warning about RMSE_full≈0, the warn mode and an explicit marker of the
    infinite gap (``generalization_gap_infinite: true``).
    """
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")],
        sanity_gate={"top_k_candidates": 2, "corridor_delta": 0.9},
    )
    cfg = Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)

    def _overfit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        # extra_trees: RMSE_full = 0 (gap = ∞), RMSE_oof better than ridge.
        if a[0] == "ridge":
            return _FakeTrainer(df["target"].to_numpy(), 0.5, 0.25)  # gap = 2.0
        return _FakeTrainer(df["target"].to_numpy(), 0.2, 0.0)  # gap = ∞

    with (
        _patch_hpo(),
        patch.object(finalists_mod, "_fit_finalist", side_effect=_overfit_finalist),
    ):
        res = train_best_model(cfg, df, target="target")

    sg = res["sanity_gate"]["gate"]
    assert sg["fallback_used"] is False
    assert sg["disqualified"] == {}
    assert "disqualified_algorithms" not in res  # circuit breaker empty
    assert sg["generalization_gap_mode"] == "warn"
    assert res["algorithm"] == "extra_trees"
    assert sg["winner_rmse_oof"] == pytest.approx(0.2, rel=1e-6)
    winner_audit = sg["audit"]["extra_trees"]
    assert winner_audit["is_valid"] is True
    # The inf value is serialized with an explicit flag, not lost in None.
    assert winner_audit["generalization_gap"] is None
    assert winner_audit["generalization_gap_infinite"] is True
    assert any(
        "Circuit D" in w and "RMSE_full" in w for w in winner_audit["soft_warnings"]
    )
    assert winner_audit["severity"] == 1


def test_select_sanity_gate_all_finalists_fail_to_train():
    """No finalist could be trained on 100% of the data → RuntimeError."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    prepared = prepare_dataset(cfg, _df(), "target")

    with (
        _patch_hpo(),
        patch.object(finalists_mod, "_fit_finalist", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="No finalist could be trained"):
            execute_phases(cfg, prepared)


def test_repeat_sanity_select_retrains_finalists():
    """A repeat sanity-select after an HPO phase: the finalists are retrained.

    The first sanity phase's full-fit trainers are not reused after the
    HPO phase.
    """
    cfg_dict = _cfg_dict(
        [
            _select_phase(name="gate", strategy="sanity_gate", output="top_k", k=2),
            {"name": "second_hpo", "n_trials": 2, "action": "all_algorithms"},
            _select_phase(name="gate_final", strategy="sanity_gate"),
        ],
        **_GATE_THRESHOLDS,
    )
    cfg = Config.model_validate(cfg_dict)
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    # Trainers accumulate per sanity-phase name. The phase name is determined
    # dynamically: the finalists pool is built at the START of a sanity-select,
    # so previous sanity phases are already in the report while the current
    # one is not yet.
    trainers_by_phase: dict[str, list[_FakeTrainer]] = {}
    current_phase: list[str] = []

    real_build_pool = select_steps_mod._build_finalists_pool

    def _build_pool(cfg_arg: Any, prepared_arg: Any, ctx_arg: Any) -> list[str]:
        pending = [
            p.name
            for p in cfg_arg.general.phases
            if p.action == "select"
            and p.strategy == "sanity_gate"
            and p.name not in ctx_arg.sanity_gate_sections
        ]
        current_phase.append(pending[0])
        return real_build_pool(cfg_arg, prepared_arg, ctx_arg)

    def _fit(*a: Any, **kw: Any) -> _FakeTrainer:
        trainer = _FakeTrainer(prepared.y.to_numpy(), 0.30, 0.25)
        trainers_by_phase.setdefault(current_phase[-1], []).append(trainer)
        return trainer

    with (
        _patch_hpo(),
        patch.object(select_steps_mod, "_build_finalists_pool", side_effect=_build_pool),
        patch.object(finalists_mod, "_fit_finalist", side_effect=_fit),
    ):
        ctx = execute_phases(cfg, prepared)

    # Finalists were trained in both sanity phases with fresh trainer instances;
    # the phase split does not depend on pool sizes or the set of algorithms.
    first_phase = trainers_by_phase["gate"]
    final_phase = trainers_by_phase["gate_final"]
    assert first_phase and final_phase
    # The terminal sanity-select returns the winner's trainer trained in ITS
    # own phase (without reusing the first sanity phase's trainers).
    assert ctx.pre_trained_trainer in final_phase
    assert ctx.pre_trained_trainer not in first_phase
    # Full disjointness: the intersection of the trainer sets is empty.
    assert not set(map(id, first_phase)) & set(map(id, final_phase))


# ──────────────────────────────────────────────────────────────────────────────
#  Reporting
# ──────────────────────────────────────────────────────────────────────────────
def test_report_warn_only_keys_removed():
    """The removed warn_only keys no longer appear in the report."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    prepared = _prepared()
    ctx = _make_ctx({"ridge": PhaseResult(score=-0.10, params={"p": 1})})
    ctx.winner_algo = "ridge"
    ctx.sanity_gate_sections = {
        "gate": {
            "strategy": "sanity_gate",
            "pool": ["ridge"],
            "audit": {},
            "disqualified": {},
            "fallback_used": False,
            "winner_rmse_oof": 0.2,
            "overhead_seconds": 0.1,
        }
    }
    with patch.object(artifact_mod, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=-0.10,
            final_params={"p": 1},
            model_path_override=None,
            disqualified_algorithms=ctx.disqualified_algorithms,
            sanity_gate_sections=ctx.sanity_gate_sections,
        )
    for key in (
        "disqualified_by_sanity_gate",
        "sanity_gate_warn_only",
        "sanity_gate_overhead_seconds",
    ):
        assert key not in result
    assert "winner_if_active" not in result["sanity_gate"]["gate"]


def test_report_audit_section_serialized():
    """The audit in the gate section is serialized into JSON-compatible dicts."""
    cfg_dict = _cfg_dict(
        [_select_phase(name="gate", strategy="sanity_gate")], **_GATE_THRESHOLDS
    )
    cfg = Config.model_validate(cfg_dict)
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")
    with _patch_hpo(), _patch_honest_finalists(df["target"]):
        ctx = execute_phases(cfg, prepared)
    section = ctx.sanity_gate_sections["gate"]
    for audit in section["audit"].values():
        assert isinstance(audit, dict)
        assert set(audit) == {
            "is_valid",
            "reasons",
            "soft_warnings",
            "diversity_ratio",
            "unique_ratio",
            "dead_features_count",
            "generalization_gap",
            "generalization_gap_infinite",
            "generalization_gap_mode",
            "severity",
        }
# The applied circuit D mode is present in the report.
    assert section["generalization_gap_mode"] == "warn"
    assert all(
        audit["generalization_gap_mode"] == "warn" for audit in section["audit"].values()
    )
    # Gate disqualifications are synchronized with the circuit breaker.
    assert set(section["disqualified"]) <= set(ctx.disqualified_algorithms)


# ──────────────────────────────────────────────────────────────────────────────
#  Unit-level helpers
# ──────────────────────────────────────────────────────────────────────────────
def test_rank_sanity_candidates_valid_first_by_rmse_oof():
    """sanity_gate ranking: valid candidates by RMSE_oof, then fallback order."""
    result = MagicMock()
    result.pool = ["flat", "honest", "second"]
    result.audit = {
        "flat": SanityCheckResult(is_valid=False, reasons=["Circuit A"], severity=2),
        "honest": SanityCheckResult(is_valid=True, reasons=[], severity=0),
        "second": SanityCheckResult(is_valid=True, reasons=[], severity=0),
    }
    cands = {
        "flat": FinalistCandidate(
            algo_name="flat",
            model=object(),
            params={},
            cv_score=-0.08,
            rmse_oof=0.01,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
        "honest": FinalistCandidate(
            algo_name="honest",
            model=object(),
            params={},
            cv_score=-0.09,
            rmse_oof=0.3,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
        "second": FinalistCandidate(
            algo_name="second",
            model=object(),
            params={},
            cv_score=-0.10,
            rmse_oof=0.2,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
    }
    ranked = _rank_sanity_candidates(result, cands)
    assert ranked == ["second", "honest", "flat"]  # RMSE_oof: 0.2 < 0.3, flat last


def test_rank_sanity_candidates_none_rmse_last():
    """A candidate with an unavailable RMSE_oof (None) is ranked last."""
    result = MagicMock()
    result.pool = ["a", "b"]
    result.audit = {
        "a": SanityCheckResult(is_valid=True, reasons=[], severity=0),
        "b": SanityCheckResult(is_valid=True, reasons=[], severity=0),
    }
    cands = {
        "a": FinalistCandidate(
            algo_name="a",
            model=object(),
            params={},
            cv_score=-0.08,
            rmse_oof=None,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
        "b": FinalistCandidate(
            algo_name="b",
            model=object(),
            params={},
            cv_score=-0.09,
            rmse_oof=0.5,
            y_pred_oof=np.zeros(3),
            y_pred_full=np.zeros(3),
        ),
    }
    assert _rank_sanity_candidates(result, cands) == ["b", "a"]


def test_serialize_audit_result_converts_nan_to_none():
    """NaN statistics of disabled circuits are replaced with None (JSON safety)."""
    audit = SanityCheckResult(
        is_valid=False,
        reasons=["Circuit A"],
        soft_warnings=["w"],
        diversity_ratio=float("nan"),
        unique_ratio=0.5,
        dead_features_count=2,
        generalization_gap=float("nan"),
        severity=2,
    )
    serialized = _serialize_audit_result(audit, "reject")
    assert serialized["diversity_ratio"] is None
    assert serialized["generalization_gap"] is None
    assert serialized["generalization_gap_infinite"] is False
    assert serialized["generalization_gap_mode"] == "reject"
    assert serialized["unique_ratio"] == 0.5
    assert serialized["dead_features_count"] == 2


def test_serialize_audit_result_infinite_gap_flagged_explicitly():
    """An infinite gap (division by zero) — an explicit flag in the report.

    Without the flag, inf would turn into None and become indistinguishable
    from 'circuit disabled or uncomputable'.
    """
    audit = SanityCheckResult(
        is_valid=True,
        soft_warnings=["Circuit D (generalization gap): RMSE_full≈0"],
        generalization_gap=float("inf"),
        severity=1,
    )
    serialized = _serialize_audit_result(audit, "warn")
    assert serialized["generalization_gap"] is None
    assert serialized["generalization_gap_infinite"] is True


def test_build_sanity_gate_passes_thresholds():
    """The ``general.sanity_gate`` section is the source of thresholds for ModelSanityGate."""
    sg_cfg = SanityGateCfg(
        min_prediction_diversity=0.25,
        max_generalization_gap=2.0,
        generalization_gap_mode="reject",
        check_dead_features=False,
        audit_time_budget_seconds=12.5,
    )
    gate = _build_sanity_gate(sg_cfg)
    assert gate.min_prediction_diversity == 0.25
    assert gate.max_generalization_gap == 2.0
    assert gate.generalization_gap_mode == "reject"
    assert gate.check_dead_features is False
    # The gate's warn_only is not used: sanity-select disqualifies.
    assert gate.warn_only is False
