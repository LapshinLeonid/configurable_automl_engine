"""Explicit select steps of the training pipeline (Training Engine).

Implements the ``select`` phases: the ``cv_best`` strategy (ranking by the CV
score without an audit) and the ``sanity_gate`` strategy (finalists pool,
training on 100% of the data, Sanity Gate audit, ranking by ``RMSE_oof``);
the step output replaces the pipeline working set. The steps only read the
``PhaseContext`` and return a ``SelectOutcome`` (the new state) that
``component`` applies. It may import ``state`` and ``finalists`` and must not
import ``component``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from configurable_automl_engine.training_engine.config_parser import (
    Config,
    HPOPhaseCfg,
    SanityGateCfg,
)

from .finalists import (
    _fallback_sort_key,
    _fmt_audit_value,
    _is_audit_budget_skip,
    _train_finalists,
    select_finalists,
    select_robust_winner,
    select_winner,
)
from .metrics import to_user_value
from .sanity_gate import ModelSanityGate, SanityCheckResult
from .state import (
    FinalistCandidate,
    PhaseContext,
    PhaseResult,
    PreparedData,
    RobustWinnerResult,
    SelectOutcome,
)

# The records keep the logger name of the pre-split ``component`` module: the
# logger name is part of the observable behavior (handlers, filters and
# ``caplog`` assertions select records by it).
_LOG = logging.getLogger("configurable_automl_engine.training_engine.component")


# --------------------------------------------------------------------------- #
#  Select steps                                                                #
#  Explicit selection steps: strategy cv_best (ranking by CV score without    #
#  audit) and sanity_gate (finalists pool → training on 100% of the data →   #
#  audit → ranking by RMSE_oof). The select step output replaces the          #
#  pipeline working set.                                                       #
# --------------------------------------------------------------------------- #
def _rank_cv_best(working_set: dict[str, PhaseResult]) -> list[str]:
    """Rank the working set by the raw CV score in descending order.

    The optimizer semantics are the same as in ``select_winner``/
    ``select_finalists``: ties are resolved by insertion order (the
    configuration order, a documented tie-break).

    Args:
        working_set (dict[str, PhaseResult]): The working set
            "algorithm → PhaseResult".

    Returns:
        list[str]: Algorithm names, best CV score first.
    """
    return sorted(working_set, key=lambda n: working_set[n].score, reverse=True)


def _rank_sanity_candidates(
    result: RobustWinnerResult,
    candidate_models: Mapping[str, FinalistCandidate],
) -> list[str]:
    """Deterministic ranking of the ``sanity_gate`` select step.

    The strategy's ranking metric is ``RMSE_oof``: candidates that passed the
    audit are sorted by ascending ``RMSE_oof`` (None — last, ties — pool
    order), followed by the rejected ones per the fallback criterion. The
    first element of the ranking always matches ``result.winner_algo``.

    Args:
        result (RobustWinnerResult): The result of ``select_robust_winner``.
        candidate_models (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        list[str]: Candidate names in the strategy ranking order.
    """

    def _rmse_key(name: str) -> tuple[bool, float]:
        rmse = candidate_models[name].rmse_oof
        return (rmse is None, rmse if rmse is not None else float("inf"))

    valid = [n for n in result.pool if result.audit[n].is_valid]
    invalid = [n for n in result.pool if not result.audit[n].is_valid]
    return sorted(valid, key=_rmse_key) + sorted(
        invalid, key=lambda n: _fallback_sort_key(n, result.audit, candidate_models)
    )


def _serialize_audit_result(
    audit: SanityCheckResult, generalization_gap_mode: str
) -> dict[str, Any]:
    """Serialize a ``SanityCheckResult`` into a JSON-compatible dict.

    NaN statistics of disabled/non-computable circuits are replaced with
        ``None``. An infinite ``generalization_gap`` (division by zero, circuit D)
        is serialized with an explicit ``generalization_gap_infinite`` flag: in
        the ``warn`` mode an infinite gap is a soft warning, and without the flag
        the reader could not tell it apart from "circuit disabled or not
        computable". The applied circuit D mode (``generalization_gap_mode``) is
        included in the serialization of each candidate: in warn mode the reader
        sees next to a Circuit D warning that the violation did NOT disqualify
        the model, and in reject mode — that it became a reason.

        Args:
            audit (SanityCheckResult): The audit result of a single finalist.
            generalization_gap_mode (str): The applied circuit D mode:
                'warn' | 'reject'.

        Returns:
            dict[str, Any]: A dict for the select step's report section.
    """

    def _num(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "is_valid": bool(audit.is_valid),
        "reasons": list(audit.reasons),
        "soft_warnings": list(audit.soft_warnings),
        "diversity_ratio": _num(audit.diversity_ratio),
        "unique_ratio": _num(audit.unique_ratio),
        "dead_features_count": int(audit.dead_features_count),
        "generalization_gap": _num(audit.generalization_gap),
        "generalization_gap_infinite": bool(audit.generalization_gap == float("inf")),
        "generalization_gap_mode": generalization_gap_mode,
        "severity": int(audit.severity),
    }


def _build_sanity_gate(sg_cfg: SanityGateCfg) -> ModelSanityGate:
    """Build a ``ModelSanityGate`` from the ``general.sanity_gate`` section.

    The config section is the source of thresholds: all circuit thresholds and
    enable flags are passed to the gate constructor unchanged, including the
    circuit D mode ``generalization_gap_mode``: ``"warn"`` (default) —
    a gap violation (finite or infinite) goes to ``soft_warnings``;
    ``"reject"`` — strict disqualification (the old behavior). The gate's
    ``warn_only`` mode is not used: the ``sanity_gate`` select step applies
    disqualifications.

    Args:
        sg_cfg (SanityGateCfg): The ``general.sanity_gate`` section.

    Returns:
        ModelSanityGate: The configured gate instance.
    """
    return ModelSanityGate(
        min_prediction_diversity=sg_cfg.min_prediction_diversity,
        adaptive_diversity=sg_cfg.adaptive_diversity,
        min_unique_count=sg_cfg.min_unique_count,
        min_unique_ratio=sg_cfg.min_unique_ratio,
        max_dead_feature_ratio=sg_cfg.max_dead_feature_ratio,
        dead_features_require_low_diversity=sg_cfg.dead_features_require_low_diversity,
        zero_coef_tolerance=sg_cfg.zero_coef_tolerance,
        max_generalization_gap=sg_cfg.max_generalization_gap,
        generalization_gap_mode=sg_cfg.generalization_gap_mode,
        check_permutation_sensitivity=sg_cfg.check_permutation_sensitivity,
        permutation_max_rows=sg_cfg.permutation_max_rows,
        permutation_max_features=sg_cfg.permutation_max_features,
        permutation_repeats=sg_cfg.permutation_repeats,
        permutation_seed=sg_cfg.permutation_seed,
        permutation_tolerance=sg_cfg.permutation_tolerance,
        check_diversity=sg_cfg.check_diversity,
        check_unique=sg_cfg.check_unique,
        check_dead_features=sg_cfg.check_dead_features,
        check_generalization_gap=sg_cfg.check_generalization_gap,
    )


def _build_finalists_pool(
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> list[str]:
    """Build the finalists pool from the current working set.

    The thresholds (corridor δ, ``top_k_candidates``, family top-up) come from
    ``general.sanity_gate``; algorithms disqualified by the circuit breaker
    (HPO failures and previous select steps) do not enter the pool.

    Args:
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        ctx (PhaseContext): The pipeline phase context.

    Returns:
        list[str]: The ordered finalists pool (best first).
    """
    sg_cfg = cfg.general.sanity_gate
    return select_finalists(
        ctx.working_set,
        metric_user=prepared.metric_user,
        top_k_candidates=sg_cfg.top_k_candidates,
        corridor_delta=sg_cfg.corridor_delta,
        corridor_mode=sg_cfg.corridor_mode,
        enforce_family_diversity=sg_cfg.enforce_family_diversity,
        algorithm_families=sg_cfg.algorithm_families,
        allowed_families=(
            set(sg_cfg.allowed_families)
            if sg_cfg.allowed_families is not None
            else None
        ),
        family_diversity_multiplier=sg_cfg.family_diversity_multiplier,
        disqualified=tuple(ctx.disqualified_algorithms),
    )


def _select_cv_best(
    phase: HPOPhaseCfg,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> SelectOutcome:
    """Handle a select step with the ``cv_best`` strategy.

    Ranks the incoming set by the raw CV score (the equivalent of the legacy
    path with ``sanity_gate.mode: off``): no audit, no gate report section is
    created. Output forms:

    - ``winner`` — the single winner (terminal step);
    - ``top_k`` — the first ``k`` candidates of the ranking;
    - ``pool`` — the corridor-based finalists pool (corridor δ ∩
      ``top_k_candidates``, thresholds from ``general.sanity_gate``), ranked by
      CV score, without a ``ModelSanityGate`` audit.

    The output replaces the pipeline working set; the other state is kept.

    Args:
        phase (HPOPhaseCfg): The select phase configuration.
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        ctx (PhaseContext): The pipeline phase context (read only).

    Returns:
        SelectOutcome: The new pipeline state.
    """
    if phase.output == "winner":
        winner_algo = select_winner(ctx.working_set)
        # Consistent user semantics: like the HPO phase log, the select step
        # prints the natural metric value (for errors — positive RMSE/MAE),
        # not the raw inverted optimizer scorer value.
        _LOG.info(
            "Select phase '%s' (strategy: cv_best): winner %s (score %.5f)",
            phase.name,
            winner_algo,
            to_user_value(prepared.metric_user, ctx.working_set[winner_algo].score),
        )
        return SelectOutcome(
            working_set={winner_algo: ctx.working_set[winner_algo]},
            winner_algo=winner_algo,
            pre_trained_trainer=None,
            full_fit_trainers=dict(ctx.full_fit_trainers),
            disqualified_algorithms=dict(ctx.disqualified_algorithms),
            sanity_gate_section=None,
        )

    if phase.output == "top_k":
        # The ranking is built lazily: not needed for winner (the source of
        # truth is select_winner), and the pool is built separately.
        kept = _rank_cv_best(ctx.working_set)[: phase.k]
        _LOG.info(
            "Select phase '%s' (strategy: cv_best): top_k=%d kept %s",
            phase.name,
            phase.k,
            kept,
        )
    else:
        # output == "pool": corridor-based finalists pool by CV score (no
        # audit).
        kept = _build_finalists_pool(cfg, prepared, ctx)
        _LOG.info(
            "Select phase '%s' (strategy: cv_best): corridor pool %s",
            phase.name,
            kept,
        )
    return SelectOutcome(
        working_set={name: ctx.working_set[name] for name in kept},
        winner_algo=ctx.winner_algo,
        pre_trained_trainer=ctx.pre_trained_trainer,
        full_fit_trainers=dict(ctx.full_fit_trainers),
        disqualified_algorithms=dict(ctx.disqualified_algorithms),
        sanity_gate_section=None,
    )


def _drop_kept_from_circuit_breaker(
    disqualified: Mapping[str, str], returned_by_fallback: Iterable[str]
) -> dict[str, str]:
    """Un-disqualify candidates returned by the select step output.

    A select step with ``strategy: sanity_gate`` (``output: top_k``/``pool``)
    deliberately returns candidates that did not pass the audit (fallback) to
    the working set — otherwise the pipeline would end up with an empty set.
    Keeping them in ``disqualified_algorithms`` is not allowed: the next
    phase would filter them out and fail (refine — RuntimeError "no valid
    results", the next sanity-select — ``select_finalists``).

    Only the names passed via ``returned_by_fallback`` are removed: failures
    disqualified by this step and returned into its output.
    Pre-existing disqualifications (the HPO circuit breaker and others) are
    not touched — other phases are responsible for them.

    Args:
        disqualified (Mapping[str, str]): The circuit breaker
            "algorithm → reason".
        returned_by_fallback (Iterable[str]): Names returned into the step
            output and disqualified by the same step.

    Returns:
        dict[str, str]: The circuit breaker without the returned names (the
            order of the remaining entries is kept).
    """
    kept = set(returned_by_fallback)
    return {name: reason for name, reason in disqualified.items() if name not in kept}


def _gate_disqualifications(
    disqualified: Mapping[str, str], result: RobustWinnerResult
) -> tuple[dict[str, str], set[str]]:
    """Merge the gate disqualifications of a select step into the circuit breaker.

    Gate disqualifications enter the circuit breaker on par with HPO failures:
    the algorithm will not return to the pools of subsequent select steps.
    The exception — candidates skipped because of an exhausted
    ``audit_time_budget_seconds``: they are NOT audited rather than rejected,
    so they must not be disqualified forever from all subsequent phases.

    Args:
        disqualified (Mapping[str, str]): The incoming circuit breaker.
        result (RobustWinnerResult): The result of ``select_robust_winner``.

    Returns:
        tuple[dict[str, str], set[str]]: The new circuit breaker and the
            names disqualified by this step (without budget skips).
    """
    merged = dict(disqualified)
    disqualified_this_step: set[str] = set()
    for name, reasons in result.disqualified.items():
        if _is_audit_budget_skip(reasons):
            _LOG.info(
                "[sanity_gate] Candidate %s was not audited (audit time "
                "budget exceeded); not permanently disqualified",
                name,
            )
            continue
        merged[name] = "; ".join(reasons)
        disqualified_this_step.add(name)
    return merged, disqualified_this_step


def _sanity_gate_section(
    result: RobustWinnerResult,
    generalization_gap_mode: str,
    overhead_seconds: float,
) -> dict[str, Any]:
    """Build the report section of a ``sanity_gate`` select step.

    The applied circuit D mode is included in the report together with the
    audit so that the reader sees that a Circuit D warning did not disqualify
    the model.

    Args:
        result (RobustWinnerResult): The result of ``select_robust_winner``.
        generalization_gap_mode (str): The applied circuit D mode.
        overhead_seconds (float): The wall time of the training and audit.

    Returns:
        dict[str, Any]: The section (stored under the select phase name).
    """
    return {
        "strategy": "sanity_gate",
        "generalization_gap_mode": generalization_gap_mode,
        "pool": list(result.pool),
        "audit": {
            name: _serialize_audit_result(res, generalization_gap_mode)
            for name, res in result.audit.items()
        },
        "disqualified": {
            name: list(reasons) for name, reasons in result.disqualified.items()
        },
        "fallback_used": result.fallback_used,
        "winner_rmse_oof": result.winner_rmse_oof,
        "overhead_seconds": overhead_seconds,
    }


def _select_sanity_gate(
    phase: HPOPhaseCfg,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> SelectOutcome:
    """Handle a select step with the ``sanity_gate`` strategy.

    Reuses the two-stage selection:

    1. Finalists pool (corridor δ ∩ ``top_k_candidates`` from
       ``general.sanity_gate``);
    2. ``_train_finalists`` — train each finalist on 100% of the data exactly
       once (a repeated select after HPO phases trains the finalists anew —
       old full-fit trainers are not reused);
    3. ``ModelSanityGate`` audit;
    4. ``select_robust_winner`` (``active`` mode): rank the candidates that
       passed the audit by ``RMSE_oof``, with a fallback criterion on total
       disqualification.

    Gate disqualifications enter the circuit breaker
    (``disqualified_algorithms``) on par with HPO failures; the exception is
    candidates skipped because of an exhausted ``audit_time_budget_seconds``:
    they were not audited rather than rejected, so they are not disqualified
    forever. The outcome carries the report section of the step. Output
    forms:

    - ``winner`` — the single winner (terminal step); its full-fit trainer is
      passed to ``persist_artifact`` via ``pre_trained_trainer``;
    - ``top_k`` — the first ``k`` candidates of the ranking (by ``RMSE_oof``);
    - ``pool`` — all candidates that passed the selection (audit); on total
      disqualification — the full ranking per the fallback criterion (the
      pipeline does not end up with an empty set).

    Output invariant: the working set does not intersect
    ``disqualified_algorithms`` — failures disqualified by this step and
    returned by the fallback into the ``top_k``/``pool`` output are removed
    from the circuit breaker (otherwise the next phase would filter them out
    and fail). Pre-existing disqualifications (the HPO circuit breaker and
    others) are not removed.

    Args:
        phase (HPOPhaseCfg): The select phase configuration.
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        ctx (PhaseContext): The pipeline phase context (read only).

    Returns:
        SelectOutcome: The new pipeline state.

    Raises:
        RuntimeError: If no finalist of the pool was trained on 100% of the
            data.
    """
    # 1. Finalists pool.
    pool = _build_finalists_pool(cfg, prepared, ctx)
    _LOG.info(
        "Select phase '%s' (strategy: sanity_gate): finalists pool %s",
        phase.name,
        pool,
    )

    # 2. Train the finalists on 100% of the data (exactly once; a repeated
    # select after HPO phases trains them anew).
    gate_started = time.monotonic()
    candidates = _train_finalists(
        pool, ctx.working_set, cfg, prepared, cfg.general.random_state
    )
    if not candidates:
        raise RuntimeError(
            f"No finalist could be trained on the full dataset in select "
            f"phase '{phase.name}' (pool: {pool})."
        )
    full_fit_trainers = {
        name: cand.trainer
        for name, cand in candidates.items()
        if cand.trainer is not None
    }

    # 3-4. Audit and ranking by RMSE_oof (active mode).
    result = select_robust_winner(
        candidates,
        prepared.X,
        prepared.y,
        _build_sanity_gate(cfg.general.sanity_gate),
        mode="active",
        audit_time_budget_seconds=cfg.general.sanity_gate.audit_time_budget_seconds,
    )
    overhead_seconds = time.monotonic() - gate_started

    disqualified, disqualified_this_step = _gate_disqualifications(
        ctx.disqualified_algorithms, result
    )

    ranked = _rank_sanity_candidates(result, candidates)
    winner_audit = result.audit[result.winner_algo]
    _LOG.info(
        "[sanity_gate] Winner %s audit summary: diversity_ratio=%s, "
        "unique_ratio=%s, dead_features_count=%s, generalization_gap=%s",
        result.winner_algo,
        _fmt_audit_value(winner_audit.diversity_ratio),
        _fmt_audit_value(winner_audit.unique_ratio),
        winner_audit.dead_features_count,
        _fmt_audit_value(winner_audit.generalization_gap),
    )
    section = _sanity_gate_section(
        result, cfg.general.sanity_gate.generalization_gap_mode, overhead_seconds
    )

    # Output form: replace the working set.
    if phase.output == "winner":
        winner_algo = result.winner_algo
        # Only the winner's trainer is needed for the artifact: the other
        # finalists' trainers are not kept in memory until the end of
        # train_best_model.
        return SelectOutcome(
            working_set={winner_algo: ctx.working_set[winner_algo]},
            winner_algo=winner_algo,
            pre_trained_trainer=candidates[winner_algo].trainer,
            full_fit_trainers={winner_algo: candidates[winner_algo].trainer},
            disqualified_algorithms=disqualified,
            sanity_gate_section=section,
        )

    if phase.output == "top_k":
        kept = ranked[: phase.k]
        _LOG.info(
            "Select phase '%s' (strategy: sanity_gate): top_k=%d kept %s",
            phase.name,
            phase.k,
            kept,
        )
    else:
        # output == "pool": all candidates that passed the selection; on
        # total disqualification — the fallback ranking (otherwise the
        # pipeline would end up with an empty set).
        kept = [n for n in result.pool if result.audit[n].is_valid]
        if not kept:
            kept = ranked
            _LOG.warning(
                "Select phase '%s' (strategy: sanity_gate): all %d finalist(s) "
                "failed the audit; output 'pool' falls back to the full ranked "
                "pool %s (fallback criterion)",
                phase.name,
                len(result.pool),
                kept,
            )
        _LOG.info(
            "Select phase '%s' (strategy: sanity_gate): pool of passed candidates %s",
            phase.name,
            kept,
        )
    # Output invariant: the returned fallback candidates are removed from the
    # circuit breaker, otherwise the next phase would filter them out and fail
    # with RuntimeError. Only this step's failures are removed.
    return SelectOutcome(
        working_set={name: ctx.working_set[name] for name in kept},
        winner_algo=ctx.winner_algo,
        pre_trained_trainer=ctx.pre_trained_trainer,
        full_fit_trainers={
            name: tr for name, tr in full_fit_trainers.items() if name in kept
        },
        disqualified_algorithms=_drop_kept_from_circuit_breaker(
            disqualified, set(kept) & disqualified_this_step
        ),
        sanity_gate_section=section,
    )


def select_step_outcome(
    phase: HPOPhaseCfg,
    cfg: Config,
    prepared: PreparedData,
    ctx: PhaseContext,
) -> SelectOutcome:
    """Run a select step and return the new pipeline state (strategy dispatcher).

    Select runs synchronously (outside ``run_parallel``). An empty incoming
    set is an error: there is nothing to select from. The context is only
    read; ``component._apply_select_outcome`` applies the outcome.

    Args:
        phase (HPOPhaseCfg): The select phase configuration.
        cfg (Config): The configuration object.
        prepared (PreparedData): The prepared data and settings.
        ctx (PhaseContext): The pipeline phase context (read only).

    Returns:
        SelectOutcome: The new pipeline state.

    Raises:
        RuntimeError: If the incoming working set is empty.
        ValueError: If the phase strategy is unknown (protection; the config
            schema allows only 'cv_best'/'sanity_gate').
    """
    if not ctx.working_set:
        raise RuntimeError(
            f"Select phase '{phase.name}' received an empty working set: "
            "nothing to select from. Check that a preceding HPO phase "
            "produced valid results."
        )
    # Common protection for all strategies: top_k without k makes no sense
    # (the config schema already requires k >= 1; the check is duplicated
    # explicitly).
    if phase.output == "top_k" and phase.k is None:
        raise ValueError(
            f"Select phase '{phase.name}': output 'top_k' requires "
            "the integer parameter k (>= 1)"
        )
    _LOG.info(
        f"--- Select Phase: {phase.name} (strategy: {phase.strategy}, "
        f"output: {phase.output}) on {len(ctx.working_set)} candidate(s) ---"
    )
    if phase.strategy == "cv_best":
        return _select_cv_best(phase, cfg, prepared, ctx)
    if phase.strategy == "sanity_gate":
        return _select_sanity_gate(phase, cfg, prepared, ctx)
    raise ValueError(  # pragma: no cover — protected by the config schema
        f"Select phase '{phase.name}': unknown strategy {phase.strategy!r}"
    )
