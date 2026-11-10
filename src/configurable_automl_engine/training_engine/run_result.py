"""Assembly of the ``ExperimentResult`` from the pipeline sources.

``build_experiment_result`` is the single assembly point of the
instrumented run result: it converts the pipeline state
(``PhaseContext`` — the working set, the circuit breaker, the winner,
the phase timings) plus the prepared data and the resolved settings into
the pure ``ExperimentResult`` model. The assembly uses
the SAME sources as the legacy dict result (``persist_artifact``), not the
dict itself, so the optional dict keys stay backward-compatible.

Key semantics:

* per-algorithm / per-phase records (``AlgorithmRun``): the working-set
  record of each phase, ordered deterministically (phase name, then
  configuration order — independent of the parallel-worker completion);
  raw scores keep the optimizer semantics, the user scores are derived by
  ``to_user_value``; trial statistics come from ``TrialStats``;
* the winner of the terminal select step (``output: winner``) is taken
  from ``ctx.winner_algo``; ``winner_score`` is the user-semantic score of
  the same record the legacy ``result["score"]`` comes from, so the two
  cannot diverge;
* scores that fail the winner-score invariant (``None``, non-finite,
  the worst-score sentinel class) never enter the model: the record is
  marked ``failed`` with the exclusion reason, the model keeps only the
  valid values (the same filtering ``execute_phases`` applies to the
  working set);
* a failed run (``status='failed'``) is assembled by
  ``build_failed_experiment_result`` when the orchestration raises
  (a phase produced no valid scores, a select received an empty working
  set, an invalid winner score): the already-known run metadata and the
  records of the completed phases are preserved so the run history
  keeps the partial evidence of the failure.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.common.metrics import (
    to_user_value,
    user_direction,
)
from configurable_automl_engine.experiment.result import (
    AlgorithmRun,
    ExperimentResult,
)
from configurable_automl_engine.training_engine.config_parser import Config

from .finalists import is_valid_winner_score
from .state import PhaseContext, PhaseResult, PreparedData

_LOG = logging.getLogger(__name__)

__all__ = [
    "build_experiment_result",
    "build_failed_experiment_result",
    "mark_result_owner",
    "result_owned_by",
]


def mark_result_owner(result: ExperimentResult, token: uuid.UUID) -> None:
    """Mark ``result`` as produced for the caller identified by ``token``.

    The ``ExperimentResult`` model has a closed field set (extra =
    ``"forbid"``), so the ownership token is attached as an attribute —
    it never reaches ``model_dump`` / ``to_json`` / the JSONL history and
    needs no model change.

    The token lets an orchestrator (the ``AutoMLExperiment`` facade)
    recognize *its own* run result when reading the module-level
    ``component.last_experiment_result`` global: the global is the
    result of the last ``train_best_model`` call in the process and is
    not reset between calls, so a consumer that cannot snapshot it
    before the run (or runs concurrently in another thread) would
    otherwise pick up a stranger's result. The engine stamps the token
    right after assembling the result; the facade passes its private
    token into ``train_best_model`` and matches it afterwards.

    Args:
        result: The assembled result model to stamp (mutated in place).
        token: The caller's private run token (an opaque UUID).
    """
    object.__setattr__(result, "_owner_token", token)


def result_owned_by(result: ExperimentResult, token: uuid.UUID) -> bool:
    """Whether ``result`` was stamped with ``token`` (see
    :func:`mark_result_owner`). A result without a token (built by an
    older engine version or by hand) belongs to nobody.
    """
    return getattr(result, "_owner_token", None) == token


def build_experiment_result(
    *,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
    result: dict[str, Any],
    run_started_at: dt.datetime,
    run_finished_at: dt.datetime,
    run_duration_seconds: float,
) -> ExperimentResult:
    """Assemble the ``ExperimentResult`` of a completed run.

    The model is built from the pipeline sources (``cfg`` / ``prepared`` /
    ``ctx``) — the same sources the legacy dict result is assembled from;
    the values that only exist after the final fit/save (``model_path``,
    ``additional_metrics``, ``oof_score`` / ``holdout_score`` /
    ``bbc_score``) are read from ``result`` — the dict
    ``persist_artifact`` returned.

    Args:
        cfg: The configuration object.
        prepared: The prepared data and resolved settings.
        ctx: The final phase context of ``execute_phases``.
        result: The legacy dict result of ``persist_artifact``.
        run_started_at: The run start time (UTC).
        run_finished_at: The run finish time (UTC).
        run_duration_seconds: The total wall-clock duration (monotonic
            clock).

    Returns:
        ExperimentResult: The validated run-result model.

    Raises:
        pydantic.ValidationError: If the assembled data violates the model
            invariants (a programming error: the engine must never produce
            an invalid result).
    """
    metric_user = prepared.metric_user
    runs = _build_algorithm_runs(cfg=cfg, prepared=prepared, ctx=ctx)
    winner = ctx.winner_algo
    winner_score_raw = ctx.working_set[winner].score if winner else None
    return ExperimentResult(
        run_id=uuid.uuid4(),
        engine_version=_engine_version(),
        started_at=run_started_at,
        finished_at=run_finished_at,
        duration_seconds=max(run_duration_seconds, 0.0),
        status=RunStatus.completed,
        config=cfg.model_dump(mode="json"),
        resolved_validation=prepared.resolved_validation.value,
        resolved_n_folds=prepared.resolved_n_folds,
        resolved_test_size=prepared.resolved_test_size,
        random_state=cfg.general.random_state,
        metric_user=metric_user,
        metric_sklearn=prepared.metric_sklearn,
        metric_direction=MetricDirection(user_direction(metric_user)),
        runs=runs,
        winner=winner,
        winner_score=(
            to_user_value(metric_user, winner_score_raw)
            if winner_score_raw is not None
            else None
        ),
        winner_params=result.get("params", {}),
        model_path=result.get("model_path"),
        additional_metrics=dict(result.get("additional_metrics", {})),
        disqualified_algorithms=dict(ctx.disqualified_algorithms),
        sanity_gate=(
            {name: dict(section) for name, section in ctx.sanity_gate_sections.items()}
            if ctx.sanity_gate_sections
            else None
        ),
        complexity_guard=(
            {"parameters": list(ctx.complexity_guard_records)}
            if ctx.complexity_guard_records
            else None
        ),
        ensemble=(
            dict(result["ensemble"])
            if isinstance(result.get("ensemble"), dict)
            else None
        ),
        oof_score=_opt_float(result.get("oof_score")),
        holdout_score=_opt_float(result.get("holdout_score")),
        bbc_score=_opt_float(result.get("bbc_score")),
        n_trials_evaluated=(
            int(result["n_trials_evaluated"])
            if result.get("n_trials_evaluated")
            else None
        ),
    )


def build_failed_experiment_result(
    *,
    cfg: Config,
    prepared: PreparedData | None,
    ctx: PhaseContext | None,
    error_message: str,
    run_started_at: dt.datetime,
    run_finished_at: dt.datetime,
    run_duration_seconds: float,
) -> ExperimentResult:
    """Assemble the ``ExperimentResult`` of a failed orchestration run.

    Called when the orchestration raises after ``prepare_dataset`` (a phase
    produced no valid scores, a select step received an empty working set,
    the winner's score failed the invariant). Everything already known —
    the resolved settings and the records of the completed phases — is
    preserved so the run history store keeps the partial evidence
    of the failure.

    Args:
        cfg: The configuration object.
        prepared: The prepared data, or ``None`` when the failure happened
            before the dataset was prepared.
        ctx: The final phase context, or ``None`` when the failure happened
            before ``execute_phases`` produced a context.
        error_message: The orchestration error message.
        run_started_at: The run start time (UTC).
        run_finished_at: The run finish time (UTC).
        run_duration_seconds: The total wall-clock duration (monotonic
            clock).

    Returns:
        ExperimentResult: The validated failed-run model.
    """
    metric_user = prepared.metric_user if prepared is not None else "rmse"
    metric_sklearn = (
        prepared.metric_sklearn
        if prepared is not None
        else "neg_root_mean_squared_error"
    )
    resolved_validation = (
        prepared.resolved_validation.value
        if prepared is not None
        else "train_test_split"
    )
    return ExperimentResult(
        run_id=uuid.uuid4(),
        engine_version=_engine_version(),
        started_at=run_started_at,
        finished_at=run_finished_at,
        duration_seconds=max(run_duration_seconds, 0.0),
        status=RunStatus.failed,
        config=cfg.model_dump(mode="json"),
        resolved_validation=resolved_validation,
        resolved_n_folds=prepared.resolved_n_folds if prepared is not None else 5,
        resolved_test_size=(
            prepared.resolved_test_size if prepared is not None else 0.2
        ),
        random_state=cfg.general.random_state,
        metric_user=metric_user,
        metric_sklearn=metric_sklearn,
        metric_direction=MetricDirection(user_direction(metric_user)),
        runs=(
            _build_algorithm_runs(cfg=cfg, prepared=prepared, ctx=ctx)  # type: ignore[arg-type]
            if ctx is not None and prepared is not None
            else []
        ),
        disqualified_algorithms=(
            dict(ctx.disqualified_algorithms) if ctx is not None else {}
        ),
        sanity_gate=(
            {name: dict(section) for name, section in ctx.sanity_gate_sections.items()}
            if ctx is not None and ctx.sanity_gate_sections
            else None
        ),
        complexity_guard=(
            {"parameters": list(ctx.complexity_guard_records)}
            if ctx is not None and ctx.complexity_guard_records
            else None
        ),
        error=error_message,
    )


def _build_algorithm_runs(
    *,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> list[AlgorithmRun]:
    """Build the per-algorithm / per-phase records of the run.

    Sources and order:

    * the working-set records of every HPO phase, plus the terminal select
      record; the phases contribute only what their final working set
      holds (the working set is not accumulated — each HPO phase rebuilds
      it, and the select output replaces it);
    * the order is deterministic: by phase name, then by the configuration
      order of the algorithms (independent of the parallel-worker
      completion order);
    * the algorithms disqualified by the circuit breaker
      (``ctx.disqualified_algorithms``) that have no valid record become
      ``disqualified`` records with the circuit-breaker reason.
    """
    records: list[AlgorithmRun] = []
    metric_user = prepared.metric_user
    config_order = {
        name: idx for idx, name in enumerate(type(cfg.algorithms).model_fields)
    }
    # The circuit-breaker disqualifications are attributed to the phase
    # where they happened (``ctx.disqualification_phases``); a select step
    # un-disqualifying a fallback candidate (``_drop_kept_from_circuit_
    # breaker``) removes it from the map — the record disappears too.
    disqualification_phase = getattr(ctx, "disqualification_phases", {})
    for phase in cfg.general.phases:
        phase_records: list[AlgorithmRun] = []
        covered: set[str] = set()
        # The working set of the phase: rebuilt by every HPO phase and
        # replaced by every select step — records of the previous phases
        # do not "resurrect" in the current one.
        for name, record in ctx.working_set.items():
            algo_run = _algorithm_run_from_phase_result(
                name=name,
                record=record,
                phase_name=phase.name,
                phase_action=phase.action,
                metric_user=metric_user,
                phase_n_trials=phase.n_trials,
            )
            phase_records.append(algo_run)
            covered.add(name)
        # Disqualified algorithms without a valid record in the phase: a
        # ``disqualified`` record with the circuit-breaker reason (the
        # leaderboard excludes them, the run history keeps the reason).
        for name, reason in sorted(ctx.disqualified_algorithms.items()):
            if name in covered:
                continue
            if disqualification_phase.get(name, "") != phase.name:
                continue
            phase_records.append(
                AlgorithmRun(
                    algorithm=name,
                    phase_name=phase.name,
                    phase_action=RunPhaseAction(phase.action),
                    status=RunStepStatus.disqualified,
                    error=reason,
                    n_trials_requested=(
                        phase.n_trials if phase.action != "select" else None
                    ),
                )
            )
        phase_records.sort(
            key=lambda r: config_order.get(r.algorithm, len(config_order))
        )
        records.extend(phase_records)
    return records


def _algorithm_run_from_phase_result(
    *,
    name: str,
    record: PhaseResult,
    phase_name: str,
    phase_action: str,
    metric_user: str,
    phase_n_trials: int,
) -> AlgorithmRun:
    """Convert one working-set record into an ``AlgorithmRun``.

    The raw score keeps the optimizer semantics ("higher is better"); the
    user score is derived by ``to_user_value``. A record whose score fails
    the winner-score invariant must not exist in the working set
    (``execute_phases`` filters it), so it is treated defensively: the
    record is dropped from the result instead of being validated into the
    model.
    """
    score_raw = record.score
    if not is_valid_winner_score(score_raw):
        # Defensive: the working set holds only valid scores. A record that
        # violates the invariant cannot be represented in the model (the
        # score fields reject non-finite/sentinel values), so the record is
        # reported as failed without scores.
        _LOG.warning(
            "Algorithm %s: phase record score %r fails the winner-score "
            "invariant; reported without scores",
            name,
            score_raw,
        )
        return AlgorithmRun(
            algorithm=name,
            phase_name=phase_name,
            phase_action=RunPhaseAction(phase_action),
            status=RunStepStatus.failed,
            error=(
                f"Invalid phase score {score_raw!r} "
                "(non-finite or worst-score sentinel)"
            ),
            params=dict(record.params),
            n_trials_requested=phase_n_trials,
            n_trials_evaluated=int(record.n_trials_evaluated or 0),
            duration_seconds=record.trial_stats.duration_seconds,
        )
    stats = record.trial_stats
    duration = stats.duration_seconds
    return AlgorithmRun(
        algorithm=name,
        phase_name=phase_name,
        phase_action=RunPhaseAction(phase_action),
        status=RunStepStatus.completed,
        score_raw=float(score_raw),
        score=to_user_value(metric_user, float(score_raw)),
        params=dict(record.params),
        n_trials_requested=phase_n_trials,
        n_trials_evaluated=int(record.n_trials_evaluated or 0),
        n_trials_completed=stats.n_completed,
        n_trials_pruned=stats.n_pruned,
        n_trials_failed=stats.n_failed,
        duration_seconds=duration,
    )


def _opt_float(value: Any) -> float | None:
    """Normalize an optional dict-report float to a plain ``float``."""
    if value is None:
        return None
    return float(value)


def _engine_version() -> str:
    """The engine version recorded in the result (setuptools-scm)."""
    # The generated ``_version.py`` is absent in a fresh checkout; the
    # module-level import mirrors ``trainer_artifact`` / the package ``__init__``.
    return str(_engine_version_value)


try:
    from configurable_automl_engine._version import (
        __version__ as _engine_version_value,
    )
except ImportError:  # pragma: no cover - absent in a fresh checkout
    _engine_version_value = "0.0.0"
