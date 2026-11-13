"""Aggregation of the Optuna trial states of one HPO phase."""

from __future__ import annotations

import optuna
from optuna.trial import FrozenTrial

from .types import TrialStats


def count_trial_states(
    trials: list[FrozenTrial], duration_seconds: float | None = None
) -> TrialStats:
    """Aggregate the state counters of the CURRENT phase's own trials.

    Args:
        trials: The trials of the current phase ONLY (transferred
            ``previous_trials`` must not be passed — they would inflate the
            counters).
        duration_seconds: The wall-clock duration of the optimization loop,
            in seconds (``None`` — not measured).

    Returns:
        TrialStats: The counters normalized to plain Python ``int`` /
        ``float`` (Optuna stores trial numbers and values as numpy scalars;
        the aggregate must survive serialization boundaries without
        changing its type).
    """
    states = [t.state for t in trials]
    n_completed = sum(1 for s in states if s == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for s in states if s == optuna.trial.TrialState.PRUNED)
    n_failed = sum(1 for s in states if s == optuna.trial.TrialState.FAIL)
    # RUNNING/WAITING trials (a crashed worker left one behind) are counted in
    # n_trials but in none of the state counters — the counters never overlap.
    return TrialStats(
        n_trials=len(trials),
        n_completed=n_completed,
        n_pruned=n_pruned,
        n_failed=n_failed,
        duration_seconds=duration_seconds,
    )
