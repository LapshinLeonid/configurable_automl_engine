"""E2E tests of the target pipeline.

Target DAG: ``all_algorithms → select(sanity_gate, top_k) → refine →
select(sanity_gate, winner)``.

Coverage:

1. E2E scenario: full DAG run on a test dataset — the gate winner is refined
   (refine receives its parameters as ``initial_params`` from the gate
   context), the repeated select re-audits the updated models (finalists are
   retrained — the honest price), the final artifact is saved via
   ``pre_trained_trainer`` without retraining the winner, and the report
   contains ``sanity_gate`` sections for both select steps with their phase
   names.
2. Negative end-to-end scenarios: a config with legacy entities
   (``refine_winner``, ``sanity_gate.mode``) reaches a friendly validation
   error at the ``train_best_model`` entry point.
3. Edge cases: refine after select(top_k) keeps all k candidates; when all
   models are disqualified again after refine, the terminal select falls back
   to the fallback criterion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import configurable_automl_engine.training_engine.component as component
from configurable_automl_engine.training_engine.component import (
    PhaseResult,
    execute_phases,
    prepare_dataset,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    ERR_REMOVED_REFINE_WINNER,
    ERR_REMOVED_SANITY_GATE_MODE,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Common helpers
# ──────────────────────────────────────────────────────────────────────────────


def _target_dag_phases(
    top_k: int = 2,
    *,
    first_select_name: str = "gate_select",
    refine_name: str = "refine",
    terminal_select_name: str = "gate_final",
) -> list[dict[str, Any]]:
    """Phases of the target DAG."""
    return [
        {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
        {
            "name": first_select_name,
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "top_k",
            "k": top_k,
        },
        {"name": refine_name, "n_trials": 2, "action": "refine"},
        {
            "name": terminal_select_name,
            "n_trials": 1,
            "action": "select",
            "strategy": "sanity_gate",
            "output": "winner",
        },
    ]


def _cfg_dict(
    phases: list[dict[str, Any]],
    *,
    gate_thresholds: dict[str, Any] | None = None,
    **general_extra: Any,
) -> dict[str, Any]:
    """Valid config with the given phases (the last one is the terminal select)."""
    general: dict[str, Any] = {
        "comparison_metric": "mae",
        "validation_strategy": "train_test_split",
        "phases": phases,
        "path_to_model": "model.pkl",
        **general_extra,
    }
    if gate_thresholds is not None:
        general["sanity_gate"] = gate_thresholds
    return {
        "general": general,
        "algorithms": {
            "ridge": {
                "enable": True,
                "tuner": "mock.mock_tuner",
                "trainer_module": "mock.mock_trainer",
            },
            "extra_trees": {
                "enable": True,
                "tuner": "mock.mock_tuner",
                "trainer_module": "mock.mock_trainer",
            },
        },
        "oversampling": {"data_oversampling": False},
    }


def _df(n_rows: int = 40, seed: int = 0) -> pd.DataFrame:
    """Numeric DataFrame with a target column."""
    rng = np.random.RandomState(seed)
    data = {f"f{i}": rng.randn(n_rows) for i in range(3)}
    data["target"] = 2.0 * data["f0"] + rng.randn(n_rows) * 0.1
    return pd.DataFrame(data)


_GATE_THRESHOLDS = {"top_k_candidates": 2, "corridor_delta": 0.9}


class _FakeTrainer:
    """Fake ModelTrainer with controlled quality of OOF/full predictions.

    ``oof_rmse`` — RMSE of the OOF vector with respect to y; ``full_rmse`` —
    RMSE of the full-fit predictions (Circuit D: gap = oof_rmse / full_rmse).
    """

    def __init__(self, y: np.ndarray, oof_rmse: float, full_rmse: float) -> None:
        rng = np.random.RandomState(0)

        def _noise(scale: float) -> np.ndarray:
            noise = rng.randn(len(y))
            return noise / np.sqrt(np.mean(noise**2)) * scale

        self.pipeline = self
        self.oof_predictions_ = y + _noise(oof_rmse)
        self.oof_score_ = oof_rmse
        self._full_pred = y + _noise(full_rmse)

    def predict(self, X: Any) -> np.ndarray:
        return self._full_pred[: len(X)]

    def save(self, path: Any) -> None:
        Path(path).write_bytes(b"fake-model")


def _houndstooth_hpo(**kwargs: Any) -> PhaseResult:
    """Deterministic HPO: extra_trees beats ridge; refine boosts.

    The coarse phase returns fixed ``PhaseResult`` records. The refine phase
    is called with ``initial_params`` from the gate context (verified below)
    and returns improved (score', params').
    """
    algo = kwargs["algo_name"]
    if kwargs.get("initial_params") is None:
        # coarse (all_algorithms): gate-context parameters of phase 1.
        return {
            "ridge": PhaseResult(score=-0.10, params={"alpha": 1.0}),
            "extra_trees": PhaseResult(score=-0.08, params={"n_estimators": 10}),
        }[algo]
    # refine: warm start from the output parameters of the first select (gate context).
    if algo == "ridge":
        assert kwargs["initial_params"] == {"alpha": 1.0}, kwargs["initial_params"]
        return PhaseResult(score=-0.09, params={"alpha": 1.5})
    assert kwargs["initial_params"] == {"n_estimators": 10}, kwargs["initial_params"]
    return PhaseResult(score=-0.07, params={"n_estimators": 20})


# ──────────────────────────────────────────────────────────────────────────────
#  1. E2E of the target DAG
# ──────────────────────────────────────────────────────────────────────────────


def test_e2e_target_dag_refine_select_persist_without_retraining(tmp_path: Path):
    """End-to-end run of the target DAG: refine, re-audit, artifact.

    Checks:
    - refine receives the gate winner's parameters as ``initial_params``
      (the winner's tuner starts from the gate-context parameters);
    - the terminal select re-audits the updated models — finalists are
      retrained (the honest price, old full-fit trainers are not reused);
    - the artifact is saved via ``pre_trained_trainer`` — ``_fit_and_save``
      is not called;
    - the report contains ``sanity_gate`` sections with the names of both
      select phases.
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=2),
        gate_thresholds=_GATE_THRESHOLDS,
    )
    cfg = component.Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    # Finalist training: phase 1 (gate_select) and phase 2 (gate_final)
    # train finalists with new instances; refine does not train finalists.
    trainers_by_phase: dict[str, list[_FakeTrainer]] = {}
    current_phase: list[str] = []
    real_build_pool = component._build_finalists_pool

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

    def _fit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        trainer = _FakeTrainer(prepared.y.to_numpy(), 0.30, 0.25)
        trainers_by_phase.setdefault(current_phase[-1], []).append(trainer)
        return trainer

    with (
        patch.object(
            component, "_run_hpo", side_effect=_houndstooth_hpo
        ) as mock_hpo,
        patch.object(component, "_build_finalists_pool", side_effect=_build_pool),
        patch.object(component, "_fit_finalist", side_effect=_fit_finalist),
        patch.object(component, "_fit_and_save") as mock_fit,
    ):
        result = train_best_model(cfg, df, target="target", model_path_override=tmp_path / "m.pkl")

    # 1a. The artifact is saved without retraining the winner (pre_trained_trainer).
    mock_fit.assert_not_called()
    assert Path(result["model_path"]).exists()

    # 1b. Refine started with a warm start from the gate-context parameters.
    # _run_hpo calls: 2 (coarse) + 2 (refine). For each candidate the refine
    # initial_params equal the parameters from the first select output.
    refine_calls = [
        call
        for call in mock_hpo.call_args_list
        if call.kwargs.get("initial_params") is not None
    ]
    assert len(refine_calls) == 2
    refine_params = {
        call.kwargs["algo_name"]: call.kwargs["initial_params"]
        for call in refine_calls
    }
    assert refine_params == {
        "ridge": {"alpha": 1.0},
        "extra_trees": {"n_estimators": 10},
    }

    # 1c. The repeated select re-audits the updated models: the terminal
    # select finalists are retrained (the first select's trainers are not
    # reused).
    assert set(trainers_by_phase) == {"gate_select", "gate_final"}
    assert all(trainers_by_phase.values())
    first = trainers_by_phase["gate_select"]
    final = trainers_by_phase["gate_final"]
    assert not set(map(id, first)) & set(map(id, final))
    # The terminal select winner is from its phase's finalist pool.
    assert result["algorithm"] in result["sanity_gate"]["gate_final"]["pool"]

    # 1d. Report: sanity_gate sections for both select steps with phase names.
    assert set(result["sanity_gate"]) == {"gate_select", "gate_final"}
    for section in result["sanity_gate"].values():
        assert section["strategy"] == "sanity_gate"
        assert section["fallback_used"] is False
        assert section["winner_rmse_oof"] is not None
        assert section["overhead_seconds"] >= 0


def test_e2e_target_dag_winner_refined_params_in_gate_context():
    """The terminal select winner is indeed refined.

    refine replaces the winner's parameters with improved ones (warm start):
    the terminal select audits the updated model and puts its parameters in
    result.
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=1),
        gate_thresholds={"top_k_candidates": 2, "corridor_delta": 0.9},
    )
    cfg = component.Config.model_validate(cfg_dict)
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    with (
        patch.object(component, "_run_hpo", side_effect=_houndstooth_hpo),
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                prepared.y.to_numpy(), 0.30, 0.25
            ),
        ),
    ):
        ctx = execute_phases(cfg, prepared)

    # top_k=1: after the first select the working set holds a single candidate —
    # the best one by RMSE_oof. Its refine parameters are an improved version
    # of the coarse parameters (the warm start came from the gate context).
    assert len(ctx.working_set) == 1
    winner = ctx.winner_algo
    assert winner is not None
    winner_record = ctx.working_set[winner]
    # The winner was updated by the refine phase: parameters from the refine stub.
    assert winner_record.params in ({"alpha": 1.5}, {"n_estimators": 20})
    assert winner_record.score in (-0.09, -0.07)
    assert ctx.pre_trained_trainer is not None
    assert set(ctx.sanity_gate_sections) == {"gate_select", "gate_final"}


def test_e2e_target_dag_infinite_gap_candidate_survives_to_refine(tmp_path: Path):
    """Regression: a candidate with gap=∞ survives from select to refine.

    The first sanity-select (top_k) in the warn mode does not lose a candidate
    with RMSE_full≈0 (gap=∞): refine processes both candidates, and the
    terminal select picks the winner by RMSE_oof (not "the best of those who
    passed the gate").
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=2),
        gate_thresholds=_GATE_THRESHOLDS,
    )
    cfg = component.Config.model_validate(cfg_dict)
    cfg.general.path_to_model = tmp_path / "m.pkl"
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    def _fit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        if a[0] == "ridge":
            return _FakeTrainer(prepared.y.to_numpy(), 0.5, 0.25)  # gap = 2.0
        return _FakeTrainer(prepared.y.to_numpy(), 0.2, 0.0)  # gap = ∞, better RMSE_oof

    with (
        patch.object(component, "_run_hpo", side_effect=_houndstooth_hpo),
        patch.object(component, "_fit_finalist", side_effect=_fit_finalist),
        patch.object(component, "_fit_and_save"),
    ):
        result = train_best_model(cfg, df, target="target")

    first = result["sanity_gate"]["gate_select"]
    terminal = result["sanity_gate"]["gate_final"]
    # The first select: the candidate with gap=∞ is valid, not in disqualified
    # and not in the circuit breaker (otherwise the next refine would lose it).
    assert first["disqualified"] == {}
    assert first["fallback_used"] is False
    et_audit = first["audit"]["extra_trees"]
    assert et_audit["is_valid"] is True
    assert et_audit["generalization_gap_infinite"] is True
    assert any(
        "Circuit D" in w and "RMSE_full" in w for w in et_audit["soft_warnings"]
    )
    # Both candidates survived to the terminal select (full pool).
    assert set(terminal["pool"]) == {"ridge", "extra_trees"}
    # The winner — the minimum RMSE_oof among the finalists (extra_trees).
    assert result["algorithm"] == "extra_trees"
    assert terminal["winner_rmse_oof"] == pytest.approx(0.2, rel=1e-6)
    assert terminal["disqualified"] == {}


# ──────────────────────────────────────────────────────────────────────────────
#  2. Negative end-to-end scenarios: legacy entities -> friendly validation
# ──────────────────────────────────────────────────────────────────────────────

def test_train_best_model_rejects_legacy_refine_winner_action():
    """A config with ``refine_winner`` fails at the ``train_best_model`` entry point.

    The error is a friendly validation with the stable code
    ``removed_refine_winner_action`` and a migration hint.
    """
    cfg_dict = {
        "general": {
            "comparison_metric": "r2",
            "phases": [
                {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
                {"name": "refine", "n_trials": 1, "action": "refine_winner"},
            ],
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    with pytest.raises(ValidationError) as excinfo:
        train_best_model(config=cfg_dict, df=_df(), target="target")

    errors = excinfo.value.errors()
    assert any(
        error["type"] == ERR_REMOVED_REFINE_WINNER for error in errors
    ), errors
    assert any("refine_winner" in error.get("msg", "") for error in errors)


def test_train_best_model_rejects_legacy_sanity_gate_mode():
    """A config with ``sanity_gate.mode`` fails at the ``train_best_model`` entry point.

    The error is a friendly validation with the stable code
    ``removed_sanity_gate_mode`` and a migration hint.
    """
    cfg_dict = {
        "general": {
            "comparison_metric": "r2",
            "phases": [
                {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "sanity_gate",
                    "output": "winner",
                },
            ],
            "sanity_gate": {"mode": "active"},
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    with pytest.raises(ValidationError) as excinfo:
        train_best_model(config=cfg_dict, df=_df(), target="target")

    errors = excinfo.value.errors()
    assert any(
        error["type"] == ERR_REMOVED_SANITY_GATE_MODE for error in errors
    ), errors
    assert any("sanity_gate.mode" in error.get("msg", "") for error in errors)


# ──────────────────────────────────────────────────────────────────────────────
#  3. Edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_refine_after_select_top_k_keeps_all_k_candidates():
    """refine after select(sanity_gate, top_k) keeps all k candidates.

    The schema requires the terminal select as the last phase, so the config
    is validated with it, and for the run the phases are truncated to
    [all_algorithms, select(top_k), refine].
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=2),
        gate_thresholds=_GATE_THRESHOLDS,
    )
    cfg = component.Config.model_validate(cfg_dict)
    cfg.general.phases = cfg.general.phases[:-1]  # drop the terminal select
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    with (
        patch.object(component, "_run_hpo", side_effect=_houndstooth_hpo),
        patch.object(
            component,
            "_fit_finalist",
            side_effect=lambda *a, **kw: _FakeTrainer(
                prepared.y.to_numpy(), 0.30, 0.25
            ),
        ),
    ):
        ctx = execute_phases(cfg, prepared)

    # The first select (top_k=2) did not narrow the set: both candidates remain,
    # and refine keeps all of them (all passed refine-HPO).
    assert set(ctx.working_set) == {"ridge", "extra_trees"}
    assert len(ctx.working_set) == 2


def test_all_disqualified_after_refine_terminal_select_falls_back_t4():
    """All candidates disqualified again after refine → fallback in the terminal select.

    The first sanity select passes honest finalists; after refine the terminal
    select audits the updated (degenerate) models — the whole pool is
    disqualified, the winner is chosen by the fallback criterion, and the run
    does not fail.
    """
    cfg_dict = _cfg_dict(
        _target_dag_phases(top_k=2),
        gate_thresholds={
            "top_k_candidates": 2,
            "corridor_delta": 0.9,
            "max_generalization_gap": 1.0,
            "generalization_gap_mode": "reject",
        },
    )
    cfg = component.Config.model_validate(cfg_dict)
    cfg.general.path_to_model = "m.pkl"
    df = _df(n_rows=40)
    prepared = prepare_dataset(cfg, df, "target")

    current_phase: list[str] = []
    real_build_pool = component._build_finalists_pool

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

    def _fit_finalist(*a: Any, **kw: Any) -> _FakeTrainer:
        if current_phase[-1] == "gate_select":
            # First select: honest finalists (gap=0.83 < 1.0 — strictly below the
            # Circuit D threshold; floating-point equality does not trigger).
            return _FakeTrainer(prepared.y.to_numpy(), 0.25, 0.30)
        # Terminal select: degenerate models (gap=10 > 1.0 → Circuit D).
        return _FakeTrainer(prepared.y.to_numpy(), 0.5, 0.05)

    with (
        patch.object(component, "_run_hpo", side_effect=_houndstooth_hpo),
        patch.object(component, "_build_finalists_pool", side_effect=_build_pool),
        patch.object(component, "_fit_finalist", side_effect=_fit_finalist),
        patch.object(component, "_fit_and_save"),
    ):
        result = train_best_model(cfg, df, target="target")

    terminal = result["sanity_gate"]["gate_final"]
    assert terminal["fallback_used"] is True
    assert terminal["disqualified"]  # all finalists rejected after refine
    assert all("Circuit D" in r for rs in terminal["disqualified"].values() for r in rs)
    # The winner (fallback) is from the terminal select's finalist pool.
    assert result["algorithm"] in terminal["pool"]
    assert result["algorithm"] in result["sanity_gate"]["gate_select"]["pool"]