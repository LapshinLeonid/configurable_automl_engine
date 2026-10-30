"""Robust winner selection over the trained finalists (Training Engine).

Holds ``select_robust_winner`` — the Sanity Gate audit of the finalists in
the ``off`` / ``active`` modes, the fallback ranking by circuit severity when
every candidate is rejected, and the audit-budget handling — with its ranking
and audit helpers. It may import ``state`` and ``sanity_gate`` and must not
import ``finalists``, ``select_steps`` or ``component``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from enum import Enum

import numpy as np
import pandas as pd

from configurable_automl_engine.training_engine.config_parser import (
    SanityGateMode,
)

from .sanity_gate import ModelSanityGate, SanityCheckResult
from .state import FinalistCandidate, RobustWinnerResult

_LOG = logging.getLogger(__name__)


# Circuit significance for the fallback criterion: D > A > B > C.
# The higher the rank, the "heavier" the violation. Ranks are fixed in code.
# Note: in the warn mode of circuit D no violation (neither a finite gap
# above the threshold nor an infinite gap from division by zero) produces a
# reason — everything goes to soft_warnings and does not take part in the
# fallback ranking; the D rank (3) applies only to the reject mode.
_CIRCUIT_RANK_BY_LETTER = {
    "A": 2,  # A — diversity
    "B": 1,  # B — unique
    "C": 0,  # C — dead features
    "D": 3,  # D — generalization gap (the most severe)
}


def _circuit_rank(reason: str) -> int:
    """Significance rank of a circuit from the reason string (D > A > B > C).

    Parses the ``Circuit <letter>`` prefix in reason (``ModelSanityGate``
    format). An unknown format is treated as the least significant circuit
    (rank 0) — a defensive fallback so the fallback ranking never fails on
    custom reasons.

    Args:
        reason (str): The disqualification reason text.

    Returns:
        int: The significance rank (0..3); 3 — circuit D, 2 — A, 1 — B,
            0 — C/unknown.
    """
    match = re.search(r"Circuit\s+([A-D])", reason)
    if match is None:
        return 0
    return _CIRCUIT_RANK_BY_LETTER.get(match.group(1), 0)


def _severity_signature(reasons: list[str]) -> tuple[int, ...]:
    """Violation severity signature for the fallback criterion.

    A tuple of circuit significance ranks sorted in descending order (the most
    severe violation first). A lexicographically smaller signature means a less
    problematic model given the same number of violated circuits.

    Args:
        reasons (list[str]): The candidate's disqualification reasons.

    Returns:
        tuple[int, ...]: The severity signature sorted in descending order.
    """
    return tuple(sorted((_circuit_rank(r) for r in reasons), reverse=True))


def _best_by_rmse_oof(
    names: list[str], candidates: Mapping[str, FinalistCandidate]
) -> str:
    """Select the candidate with the best (minimal) ``RMSE_oof``.

    Tie-break: on equal ``RMSE_oof`` the first in list order wins (the pool
    order is deterministic). A candidate with unavailable ``RMSE_oof`` (None)
    is ranked last.

    Args:
        names (list[str]): Candidates in deterministic order.
        candidates (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        str: The winner's name.
    """

    def _key(name: str) -> tuple[int, float]:
        rmse = candidates[name].rmse_oof
        return (rmse is None, rmse if rmse is not None else float("inf"))

    return min(names, key=_key)


def _fallback_sort_key(
    name: str,
    audits: Mapping[str, SanityCheckResult],
    candidate_models: Mapping[str, FinalistCandidate],
) -> tuple[int, tuple[int, ...], bool, float]:
    """Fallback ranking key (see ``_select_active_winner``).

    Minimum number of violated circuits → minimum severity (D > A > B > C) →
    best ``RMSE_oof``. Hoisted to module level so the select step ranking
    (``output: top_k``/``pool``) matches the winner selection.

    Args:
        name (str): The candidate name.
        audits (Mapping[str, SanityCheckResult]): The audit of each finalist.
        candidate_models (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        tuple[int, tuple[int, ...], bool, float]: The sort key.
    """
    reasons = audits[name].reasons
    rmse = candidate_models[name].rmse_oof
    return (
        len(reasons),
        _severity_signature(reasons),
        rmse is None,
        rmse if rmse is not None else float("inf"),
    )


def _select_active_winner(
    pool_order: list[str],
    audits: Mapping[str, SanityCheckResult],
    candidate_models: Mapping[str, FinalistCandidate],
) -> str:
    """Select the winner in ``active`` mode.

    First — the best ``RMSE_oof`` among the candidates that passed the audit
    (CV and ``RMSE_full`` do not participate in the ranking). If there are no
    valid candidates (total pool failure) — the formal fallback criterion of
    the "least problematic" model: minimum number of violated circuits →
    minimum severity (circuit significance is fixed in code: D > A > B > C,
    see ``_circuit_rank``) → best ``RMSE_oof``. The run never fails.

    Args:
        pool_order (list[str]): The ordered finalists pool.
        audits (Mapping[str, SanityCheckResult]): The audit of each finalist.
        candidate_models (Mapping[str, FinalistCandidate]): Candidates.

    Returns:
        str: The winner's name.
    """
    valid = [n for n in pool_order if audits[n].is_valid]
    if valid:
        return _best_by_rmse_oof(valid, candidate_models)
    return min(
        pool_order,
        key=lambda n: _fallback_sort_key(n, audits, candidate_models),
    )


def _fmt_audit_value(value: float) -> str:
    """Format a numeric audit metric for logs.

    NaN/±inf (circuit disabled or not computable) are rendered as ``"n/a"`` so
    that the winner's audit summary log stays compact and readable.

    Note: the formatter is used in the INFO log of the audit summary of the
    ``sanity_gate`` select step; it is also used by the report mechanisms (see
    also ``select_finalists`` / ``_train_finalists`` / ``select_robust_winner``).

    Args:
        value (float): The audit metric value (diversity/unique/gap).

    Returns:
        str: ``"%.4f"`` for finite values, ``"n/a"`` otherwise.
    """
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


class _AuditSkipReason(str, Enum):
    """Reasons for skipping the Sanity Gate audit.

    An audit skip differs from a failure: the check did not run because of a
    resource limit of the phase (``audit_time_budget_seconds``), not because
    of the model quality. A candidate with such a reason cannot win the
    current select step (there is no audit), but must NOT be disqualified
    forever: the limit of one phase does not throw the algorithm out of all
    subsequent ones.
    """

    BUDGET_EXCEEDED = "Sanity audit skipped: audit time budget"


def _audit_budget_skip_reason(budget_seconds: float) -> str:
    """Build the reason text for a time-budget audit skip.

    The single source of the format: used by ``select_robust_winner`` when
    building the ``SanityCheckResult`` and by tests — the string is not
    duplicated anywhere.

    Args:
        budget_seconds (float): The phase's ``audit_time_budget_seconds`` value.

    Returns:
        str: A reason like "Sanity audit skipped: audit time budget (Ns) exceeded".
    """
    return f"{_AuditSkipReason.BUDGET_EXCEEDED.value} ({budget_seconds:g}s) exceeded"


# Full match of the audit-skip reason to the "{prefix} (Ns) exceeded" format,
# where N is a number in the %g format (integer/decimal/exponent). The
# fullmatch check rules out false positives on reasons that only start with
# the prefix.
_AUDIT_BUDGET_SKIP_RE = re.compile(
    rf"{re.escape(_AuditSkipReason.BUDGET_EXCEEDED.value)} "
    r"\([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?s\) exceeded"
)


def _is_audit_budget_skip(reasons: list[str]) -> bool:
    """True if ALL candidate reasons are a time-budget audit skip.

    Used in ``select_steps._gate_disqualifications`` so that such candidates do not
    enter the circuit breaker (``disqualified_algorithms``) — the phase
    budget must not throw the algorithm out of all subsequent phases.

    Args:
        reasons (list[str]): The candidate's audit invalidity reasons.

    Returns:
        bool: True if the reasons reduce to a skip caused by an exhausted
            ``audit_time_budget_seconds`` (at least one reason exists).
    """
    return bool(reasons) and all(
        _AUDIT_BUDGET_SKIP_RE.fullmatch(reason) is not None for reason in reasons
    )


def select_robust_winner(
    candidate_models: Mapping[str, FinalistCandidate],
    X: pd.DataFrame,
    y: pd.Series,
    sanity_gate: ModelSanityGate,
    *,
    mode: SanityGateMode = "off",
    audit_time_budget_seconds: float = 0,
) -> RobustWinnerResult:
    """Two-stage winner selection with a Sanity Gate audit.

    Iterates over the candidate pool (already trained on 100% of the data),
    runs each candidate through ``ModelSanityGate.check`` (diversity/unique are
    computed on the OOF vector; circuit D uses full-fit predictions) and ranks
    the candidates that passed the audit by ``RMSE_oof`` (CV and ``RMSE_full``
    do NOT participate in the ranking — the composite score was removed after
    the v2 review).

    Modes (``mode``):
    - ``off``: legacy ``select_winner`` behavior — winner = best ``cv_score``
      (no audit; ``audit`` stays empty);
    - ``active``: disqualifications are applied; winner = best ``RMSE_oof``
      among those that passed all checks. Pipeline select steps with
      ``strategy: sanity_gate`` run exactly this mode. The ``warn_only`` mode
      was removed along with the ``sanity_gate_warn_only``/``winner_if_active``
      report keys.

    Audit time budget (``audit_time_budget_seconds``): the total time of the
    audit phase is limited; when the budget is exhausted the remaining
    candidates are not audited and receive the explicit reason "audit skipped:
    budget exceeded" (in ``active`` mode they cannot win). ``0`` — no limit.

    Fallback (formalized): if all candidates are rejected, the "least
    problematic" model is chosen — minimum number of violated circuits, on ties
    — minimum violation severity (circuit significance is fixed in code:
    D > A > B > C, see ``_circuit_rank``), on further ties — best ``RMSE_oof``.
    The run never fails; a WARNING with the reasons is logged.

    Args:
        candidate_models (Mapping[str, FinalistCandidate]): The candidate pool
            (trained on 100% of the data, OOF vectors available). Insertion
            order is the deterministic pool order.
        X (pd.DataFrame): The full feature matrix.
        y (pd.Series): The full target vector.
        sanity_gate (ModelSanityGate): The configured gate instance.
        mode (SanityGateMode): The operation mode ('off'/'active').
        audit_time_budget_seconds (float): The total audit phase time budget in
            seconds; 0 — no limit.

    Returns:
        RobustWinnerResult: The winner, the audit of each candidate, the
            applied disqualifications, and the fallback flag.

    Raises:
        RuntimeError: If the candidate pool is empty — no winner exists.
        ValueError: If ``mode`` is unknown.
    """
    if not candidate_models:
        raise RuntimeError("Cannot select a robust winner from an empty candidate pool")

    pool_order = list(candidate_models)

    def _cv_winner() -> str:
        """Legacy select_winner behavior: best raw score, ties — order."""
        return max(pool_order, key=lambda n: candidate_models[n].cv_score)

    if mode == "off":
        winner = _cv_winner()
        return RobustWinnerResult(
            winner_algo=winner,
            mode=mode,
            pool=pool_order,
            winner_rmse_oof=candidate_models[winner].rmse_oof,
        )

    if mode not in ("off", "active"):
        raise ValueError(f"mode must be 'off' or 'active'; got {mode!r}")

    # Audit each finalist: diversity/unique on the OOF vector, circuit D on the
    # full-fit predictions. The total audit phase time is limited by the
    # audit_time_budget_seconds budget (0 — no limit).
    audits: dict[str, SanityCheckResult] = {}
    audit_started = time.monotonic()

    def _audit_budget_exhausted() -> bool:
        return (
            audit_time_budget_seconds > 0
            and time.monotonic() - audit_started >= audit_time_budget_seconds
        )

    for name in pool_order:
        if _audit_budget_exhausted():
            _LOG.warning(
                "[sanity_gate] audit time budget (%.3fs) exceeded; "
                "skipping audit of candidate %s",
                audit_time_budget_seconds,
                name,
            )
            audits[name] = SanityCheckResult(
                is_valid=False,
                reasons=[_audit_budget_skip_reason(audit_time_budget_seconds)],
                severity=0,
            )
            continue
        cand = candidate_models[name]
        try:
            audits[name] = sanity_gate.check(
                y=y,
                y_pred_oof=cand.y_pred_oof,
                y_pred_full=cand.y_pred_full,
                X=X,
                model=cand.model,
                candidate_name=name,
            )
        except Exception as err:  # noqa: BLE001 — audit failure = disqualification
            _LOG.warning("Sanity audit failed for candidate %s: %s", name, err)
            audits[name] = SanityCheckResult(
                is_valid=False,
                reasons=[f"Sanity audit failed: {err}"],
                severity=4,
            )

    disqualified = {
        name: list(audits[name].reasons) for name in pool_order if audits[name].reasons
    }

    # For each rejected candidate — a WARNING listing the disqualification
    # reasons (active mode applies to winner selection).
    for name, reasons in disqualified.items():
        _LOG.warning(
            "[sanity_gate active] Candidate %s disqualified by the sanity gate: %s",
            name,
            reasons,
        )

    # mode == "active": apply disqualifications, rank by RMSE_oof.
    winner = _select_active_winner(pool_order, audits, candidate_models)
    all_failed = not any(audits[n].is_valid for n in pool_order)
    if all_failed:
        # Fallback: all candidates were rejected — pick the "least
        # problematic" model. The formal criterion is stated explicitly in the
        # log.
        _LOG.warning(
            "[sanity_gate active] All %d candidate(s) failed the sanity audit; "
            "falling back to the least problematic model %s. Formal criterion: "
            "min violated circuits -> min severity (D > A > B > C) -> "
            "best RMSE_oof. Reasons: %s",
            len(pool_order),
            winner,
            {n: audits[n].reasons for n in pool_order},
        )
    return RobustWinnerResult(
        winner_algo=winner,
        mode=mode,
        pool=pool_order,
        audit=audits,
        disqualified=disqualified,
        fallback_used=all_failed,
        winner_rmse_oof=candidate_models[winner].rmse_oof,
    )
