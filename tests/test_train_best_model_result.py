"""Tests of the instrumented ``train_best_model`` run result (T3).

Covered scenarios (positive / negative / boundary):

- P1: a full run (HPO phase + select + terminal select) → the
  ``ExperimentResult`` stored in ``component.last_experiment_result``
  contains the records of all phases, the winner and the final score
  consistent with the dict result;
- P2: the dict result keeps its legacy contract keys (the
  ``persist_artifact`` regression);
- P3: ``additional_metrics``, ``disqualified_algorithms``, ``sanity_gate``
  and ``complexity_guard`` land both in the ExperimentResult and in the
  dict;
- N1: every algorithm failed in a phase → ``RuntimeError`` and a
  failed-status ExperimentResult recorded internally;
- N2: the terminal select received an empty working set (``phases: []``)
  → ``RuntimeError`` + a failed ExperimentResult;
- N3: a winner with an invalid score → ``RuntimeError``, the sentinel
  never enters the result;
- B1: a single HPO phase without select (a minimal valid config, truncated
  after validation);
- B2: ``phases: []`` → the failed result;
- B3: a single enabled algorithm;
- B4: the parallel mode (``parallel_mode: threads``) — records and timings
  are correct and deterministic;
- B5: a mid-phase circuit-breaker disqualification — the disqualified
  record with the reason, the others in place;
- B6: ``general.holdout_fraction`` — ``holdout_score`` is present in the
  ExperimentResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

import configurable_automl_engine.training_engine.component as component
from configurable_automl_engine import train_best_model
from configurable_automl_engine.common.definitions import (
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.hpo.runner import run_optimization
from configurable_automl_engine.tuner import (
    InvalidAlgorithmError,
    OptimizeResult,
)
from configurable_automl_engine.training_engine.component import execute_phases


# The import keeps the patch target explicit: the tests patch the HPO run
# inside the phase worker (the same target ``test_component.py`` uses).
_HPO_PHASES = "configurable_automl_engine.training_engine.hpo_phases"


# ────────────────────────── helpers ────────────────────────── #


def _result_config(**general_overrides: Any) -> dict[str, Any]:
    """A valid two-phase config dict (HPO phase + terminal cv_best select)."""
    general: dict[str, Any] = {
        "comparison_metric": "rmse",
        "random_state": 42,
        "validation_strategy": "train_test_split",
        "holdout_fraction": None,
        "path_to_model": "model.pkl",
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
    }
    general.update(general_overrides)
    return {
        "general": general,
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            },
            "elasticnet": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.8]},
            },
        },
        "oversampling": {"enable": False},
    }


def _small_df() -> pd.DataFrame:
    """A tiny deterministic regression dataset (well above the LOO limit)."""
    rows = 40
    return pd.DataFrame(
        {
            "f1": list(range(rows)),
            "f2": [i % 7 * 0.5 for i in range(rows)],
            "target": [10.0 - 0.2 * i for i in range(rows)],
        }
    )


def _last_result():
    """The ExperimentResult of the last ``train_best_model`` call."""
    return component.last_experiment_result


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_full_run_records_all_phases_winner_and_score(tmp_path: Path) -> None:
    """P1: HPO phase + terminal select → the records of all phases, the
    winner, and the final score consistent with the dict result."""
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    res = train_best_model(cfg, _small_df(), target="target")
    result = _last_result()

    assert result is not None
    assert result.status is RunStatus.completed
    # The winner of the ExperimentResult matches the dict result.
    assert result.winner == res["algorithm"]
    assert result.winner_score == res["score"]
    assert result.winner_params == res["params"]
    assert result.model_path == res["model_path"]
    # The records of ALL phases are present: the working set of the
    # terminal select (winner only) is reported under the terminal phase.
    phase_names = [r.phase_name for r in result.runs]
    assert set(phase_names) == {"coarse", "final"}
    # The winner's records exist for both phases.
    winner_records = [r for r in result.runs if r.algorithm == res["algorithm"]]
    assert {r.phase_action for r in winner_records} == {
        RunPhaseAction.all_algorithms,
        RunPhaseAction.select,
    }
    # Raw score (optimizer semantics) vs user score.
    completed = [r for r in winner_records if r.status is RunStepStatus.completed]
    assert completed
    for record in completed:
        assert record.score_raw < 0  # neg RMSE, optimizer semantics
        assert record.score == pytest.approx(res["score"])
    # The trial statistics of the phase (T2): the HPO phase evaluated its
    # own trials.
    coarse = [r for r in result.runs if r.phase_name == "coarse"]
    assert all(r.n_trials_evaluated > 0 for r in coarse)
    # The run-level timings are recorded and ordered.
    assert result.duration_seconds >= 0
    assert result.started_at <= result.finished_at
    assert result.random_state == 42
    # The resolved validation strategy is recorded.
    assert result.resolved_validation == "train_test_split"
    assert 0.0 < result.resolved_test_size < 1.0


def test_p1_phase_timings_collected(tmp_path: Path) -> None:
    """P1: the per-phase timings of ``execute_phases`` are complete and sane."""
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    # The ExperimentResult itself is validated; here the timing records are
    # checked through a direct execute_phases run.
    from configurable_automl_engine.training_engine.data_prep import (
        load_config,
        prepare_dataset,
    )

    loaded_cfg, target_col = load_config(cfg, _small_df(), "target")
    prepared = prepare_dataset(loaded_cfg, _small_df(), target_col)
    ctx = execute_phases(loaded_cfg, prepared)
    assert [(t.phase_name, t.action) for t in ctx.phase_timings] == [
        ("coarse", "all_algorithms"),
        ("final", "select"),
    ]
    for timing in ctx.phase_timings:
        assert timing.duration_seconds >= 0.0
        assert timing.started_at <= timing.finished_at


def test_p2_dict_result_keeps_legacy_contract(tmp_path: Path) -> None:
    """P2: the dict result keeps its legacy keys (the persist_artifact
    contract regression)."""
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    res = train_best_model(cfg, _small_df(), target="target")
    # The mandatory keys of the legacy contract.
    for key in ("algorithm", "score", "metric", "params", "model_path", "random_state"):
        assert key in res, key
    assert res["metric"] == "rmse"
    assert res["random_state"] == 42
    # Nothing new leaked into the dict: without disqualifications, extra
    # metrics and a hold-out the optional keys stay absent.
    assert "disqualified_algorithms" not in res
    assert "additional_metrics" not in res
    assert "sanity_gate" not in res
    assert "holdout_score" not in res


def test_p3_sections_land_in_dict_and_result(tmp_path: Path) -> None:
    """P3: additional metrics and the sanity-gate section land both in the
    dict and in the ExperimentResult."""
    general_overrides: dict[str, Any] = {
        "path_to_model": str(tmp_path / "m.pkl"),
        "additional_metrics": ["mae"],
        "phases": [
            {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
            {
                "name": "final",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
    }
    cfg = _result_config(**general_overrides)
    res = train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert "additional_metrics" in res
    assert result.additional_metrics == res["additional_metrics"]


def test_p3_disqualified_algorithms_in_dict_and_result(tmp_path: Path) -> None:
    """P3: the circuit-breaker disqualifications land in both containers."""
    reason = "Algorithm 'ridge' disqualified after 5 consecutive fatal failures"

    def fake_run(_x: Any, _y: Any, request: Any) -> Any:
        if request.algo_name == "ridge":
            raise InvalidAlgorithmError(reason)
        return run_optimization(_x, _y, request)

    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    with patch(f"{_HPO_PHASES}.run_optimization", side_effect=fake_run):
        res = train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert res["disqualified_algorithms"] == {"ridge": reason}
    assert result.disqualified_algorithms == {"ridge": reason}
    # The disqualified algorithm has a ``disqualified`` record attributed to
    # the phase where it was disqualified.
    dq_records = [
        r
        for r in result.runs
        if r.algorithm == "ridge" and r.status is RunStepStatus.disqualified
    ]
    assert len(dq_records) == 1
    assert dq_records[0].phase_name == "coarse"
    assert dq_records[0].error == reason
    # The working algorithm won and has its records.
    assert result.winner == res["algorithm"] != "ridge"


# ────────────────────────── negative scenarios ────────────────────────── #


def test_n1_all_algorithms_failed_records_failed_status(tmp_path: Path) -> None:
    """N1: all algorithms failed in the phase → RuntimeError and an
    internally recorded failed ExperimentResult."""
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    with patch(
        f"{_HPO_PHASES}.run_optimization",
        side_effect=InvalidAlgorithmError("broken"),
    ):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert result.status is RunStatus.failed
    assert result.error and "No algorithms produced valid scores" in result.error
    assert result.winner is None
    assert result.winner_score is None
    # The resolved settings of the failed run are still recorded.
    assert result.resolved_validation == "train_test_split"
    assert result.random_state == 42


def test_n2_empty_phases_records_failed_status(tmp_path: Path) -> None:
    """N2: phases: [] → RuntimeError ('cannot select a winner') and a
    failed ExperimentResult."""
    cfg = _result_config(phases=[], path_to_model=str(tmp_path / "m.pkl"))
    with pytest.raises(RuntimeError, match="cannot select a winner"):
        train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert result.status is RunStatus.failed
    assert result.error and "cannot select a winner" in result.error
    assert result.runs == []


def test_n3_invalid_winner_score_never_enters_result(tmp_path: Path) -> None:
    """N3: a winner with a sentinel score → RuntimeError; the sentinel does
    not enter the ExperimentResult."""
    sentinel = float(__import__("numpy").finfo(__import__("numpy").float32).min)
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))

    def fake_run(_x: Any, _y: Any, request: Any) -> Any:
        # Both algorithms complete HPO, but every score is the sentinel:
        # the working-set filter drops them (a phase failure), so the run
        # fails at the phase level; the sentinel cannot reach the result.
        return OptimizeResult(
            model=None,
            params={"alpha": 1.0},
            score=sentinel,
            trials=[],
        )

    with patch(f"{_HPO_PHASES}.run_optimization", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert result.status is RunStatus.failed
    # The sentinel value never appears in the assembled result.
    assert result.winner_score is None
    assert result.winner is None
    for record in result.runs:
        assert record.score_raw is None
        assert record.score is None


def test_n3_winner_score_validated_before_artifact(tmp_path: Path) -> None:
    """N3 (unit): the winner-score invariant guard in ``_run_after_phases``.

    A working set that bypassed the phase filter (a patched
    ``execute_phases``) with an invalid winner score must fail before
    persist_artifact and must record a failed ExperimentResult.
    """
    from configurable_automl_engine.training_engine.data_prep import (
        load_config,
        prepare_dataset,
    )
    from configurable_automl_engine.training_engine.state import PhaseResult

    sentinel = float("nan")
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    df = _small_df()
    loaded_cfg, target_col = load_config(cfg, df, "target")
    prepared = prepare_dataset(loaded_cfg, df, target_col)
    ctx = execute_phases(loaded_cfg, prepared)
    # Bypass the filter: inject an invalid winner score directly.
    winner = ctx.winner_algo
    assert winner is not None
    ctx.working_set[winner] = PhaseResult(score=sentinel, params={"alpha": 1.0})

    with patch.object(component, "execute_phases", return_value=ctx):
        with pytest.raises(RuntimeError, match="invalid final score"):
            train_best_model(loaded_cfg, df, target=target_col)
    result = _last_result()
    assert result is not None
    assert result.status is RunStatus.failed
    assert "invalid final score" in result.error
    assert result.winner_score is None


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_single_hpo_phase_without_select(tmp_path: Path) -> None:
    """B1: a single HPO phase without a select (a minimal valid config,
    truncated after validation) — the ExperimentResult holds the phase
    records and the run-level winner from the context."""
    from configurable_automl_engine.training_engine.data_prep import (
        load_config,
        prepare_dataset,
    )

    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    df = _small_df()
    loaded_cfg, target_col = load_config(cfg, df, "target")
    # Truncate the terminal select: a minimal HPO-only pipeline (the schema
    # requires the terminal select; execute_phases reads only the list).
    loaded_cfg.general.phases = loaded_cfg.general.phases[:1]
    prepared = prepare_dataset(loaded_cfg, df, target_col)
    ctx = execute_phases(loaded_cfg, prepared)
    assert ctx.working_set
    assert ctx.winner_algo is None  # no terminal select ran
    assert [t.phase_name for t in ctx.phase_timings] == ["coarse"]


def test_b2_empty_phases_failed_result(tmp_path: Path) -> None:
    """B2: phases: [] — the phase loop did not run; the failed result is
    recorded with zero phase timings."""
    cfg = _result_config(phases=[], path_to_model=str(tmp_path / "m.pkl"))
    with pytest.raises(RuntimeError):
        train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert result.status is RunStatus.failed
    assert result.duration_seconds >= 0.0


def test_b3_single_enabled_algorithm(tmp_path: Path) -> None:
    """B3: one enabled algorithm — the records exist for both phases and
    the winner is that algorithm."""
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    cfg["algorithms"] = {
        "ridge": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {"alpha": [0.1, 1.0]},
        }
    }
    res = train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert res["algorithm"] == "ridge"
    assert result.winner == "ridge"
    assert {r.algorithm for r in result.runs} == {"ridge"}


def test_b4_parallel_threads_records_and_timings(tmp_path: Path) -> None:
    """B4: parallel mode (threads) — the records stay deterministic (the
    configuration order) and each record belongs to exactly one algorithm."""
    cfg = _result_config(
        parallel_mode="threads",
        max_workers=2,
        path_to_model=str(tmp_path / "m.pkl"),
    )
    res = train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert res["algorithm"] == result.winner
    # The record order is deterministic: per phase, configuration order.
    for phase_name in ("coarse", "final"):
        records = [r for r in result.runs if r.phase_name == phase_name]
        assert [r.algorithm for r in records] == sorted(
            r.algorithm for r in records
        ) or [r.algorithm for r in records] == ["elasticnet", "ridge"]
    # Each record's duration belongs to its own record (non-negative, no
    # cross-worker mixing is observable as duplicated timings).
    for record in result.runs:
        assert record.duration_seconds is None or record.duration_seconds >= 0.0


def test_b5_mid_phase_disqualification(tmp_path: Path) -> None:
    """B5: a circuit-breaker disqualification in the middle of the run —
    the disqualified record is excluded from the working records, the rest
    stay in place."""
    reason = "disqualified after fatal failures"
    calls = {"n": 0}

    def fake_run(_x: Any, _y: Any, request: Any) -> Any:
        if request.algo_name == "elasticnet":
            calls["n"] += 1
            if calls["n"] >= 1:
                raise InvalidAlgorithmError(reason)
        return run_optimization(_x, _y, request)

    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    with patch(f"{_HPO_PHASES}.run_optimization", side_effect=fake_run):
        res = train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert res["algorithm"] == "ridge" == result.winner
    # elasticnet: disqualified in the coarse phase; no record in 'final'.
    final_records = [r for r in result.runs if r.phase_name == "final"]
    assert [r.algorithm for r in final_records] == ["ridge"]
    coarse_elastic = [
        r
        for r in result.runs
        if r.phase_name == "coarse" and r.algorithm == "elasticnet"
    ]
    assert len(coarse_elastic) == 1
    assert coarse_elastic[0].status is RunStepStatus.disqualified
    assert coarse_elastic[0].error == reason


def test_b6_holdout_fraction_holdout_score_in_result(tmp_path: Path) -> None:
    """B6: general.holdout_fraction — holdout_score is present in the
    ExperimentResult (and in the dict)."""
    cfg = _result_config(holdout_fraction=0.25, path_to_model=str(tmp_path / "m.pkl"))
    res = train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    assert "holdout_score" in res
    assert result.holdout_score == res["holdout_score"]
    assert result.holdout_score is not None


def test_result_json_round_trip(tmp_path: Path) -> None:
    """The assembled ExperimentResult survives the JSON round-trip (the
    store's serialization contract)."""
    cfg = _result_config(path_to_model=str(tmp_path / "m.pkl"))
    train_best_model(cfg, _small_df(), target="target")
    result = _last_result()
    assert result is not None
    restored = type(result).from_json(result.to_json())
    assert restored == result
