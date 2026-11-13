"""Public types of the hyperparameter optimisation (HPO) package.

The exceptions (``HyperoptError`` and its subclasses), the self-documenting
phase result ``OptimizeResult``, its aggregated trial statistics
``TrialStats`` and the worst-score sentinel constants
(``HPO_WORST_SCORE`` / ``WORST_SCORE_THRESHOLD``). They are re-exported by
``configurable_automl_engine.tuner``, their documented import path (the
classes are defined here, so ``__module__`` names this module). The
module holds definitions only and must not import any other module of the
``hpo`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from optuna.trial import FrozenTrial

from configurable_automl_engine.common.definitions import (
    WORST_SCORE_THRESHOLD as _WORST_SCORE_THRESHOLD,
)


# ═════════════════════════════════════ exceptions ════════════════════════════
class HyperoptError(Exception):
    """Base error of the hyperparameter optimisation module."""


class InvalidAlgorithmError(HyperoptError):
    """Error raised when an algorithm is not found or is marked as unused."""


class InvalidDataError(HyperoptError):
    """Error raised when invalid X or y data structures are passed."""


class HPOHistoryError(HyperoptError):
    """Error raised when TPE history cannot be transferred between HPO phases.

    Raised in two situations:
    - the transferred trials have distributions incompatible with the current
      search space (``study.add_trials`` desync) — fail fast instead of
      silently skipping the history;
    - the current search space cannot be reproduced by the distribution
      pre-check (an exotic custom ``space_fn``) — the transfer cannot be
      validated, and silently dropping the history would degrade refine to a
      cold run (the exact bug this feature fixes).
    """


# ═════════════════════════════ trial statistics ══════════════════════════════
@dataclass
class TrialStats:
    """Aggregated statistics of the current phase's own trials.

    The ready-made counters of one HPO phase so the callers (the training
    engine's leaderboard columns, logs, reports) do not have to recount the
    trial states by iterating over ``FrozenTrial.state``. The counters are
    computed over the CURRENT phase's own trials only — transferred
    ``previous_trials`` never count (symmetric with ``OptimizeResult.
    n_trials_evaluated``).

    All counters are plain Python ``int``: Optuna/numpy scalar types
    (``np.int64`` and friends) are normalized at construction so the values
    survive serialization boundaries (parallel workers, reports) without
    changing their type.

    Attributes:
        n_trials: The number of the phase's own trials (requested and
            actually run by Optuna; an enqueued ``initial_params`` trial
            counts too).
        n_completed: The trials that finished with a valid score
            (``TrialState.COMPLETE``). A trial whose metric was non-finite
            still completes (with the worst-score sentinel), so it is
            counted here.
        n_pruned: The trials pruned by the early stopping pruner or by a
            non-fatal evaluation error (``optuna.TrialPruned``).
        n_failed: The trials that failed with an uncaught exception
            (``TrialState.FAIL``).
        duration_seconds: The wall-clock duration of the phase's
            optimization loop, in seconds. ``None`` when the duration was
            not measured (e.g. a custom runner or ``n_trials=0``).
    """

    n_trials: int = 0
    n_completed: int = 0
    n_pruned: int = 0
    n_failed: int = 0
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        """Normalize the counters to plain Python scalars."""
        self.n_trials = int(self.n_trials)
        self.n_completed = int(self.n_completed)
        self.n_pruned = int(self.n_pruned)
        self.n_failed = int(self.n_failed)
        self.duration_seconds = (
            None if self.duration_seconds is None else float(self.duration_seconds)
        )


@dataclass
class OptimizeResult:
    """Result of a single HPO phase run (self-documenting return type).

    Replaces the positional ``(model, params, score)`` tuple of ``optimize``:
    the fields have explicit names, and the current phase's trials travel with
    the result so the caller can feed them into the next phase
    (``previous_trials``).

    Attributes:
        model: The final model trained on all data with the best parameters.
            ``None`` when ``fit_best_model=False`` (the full-data fit is
            skipped for HPO-only callers); otherwise never ``None`` on a
            non-``None`` ``OptimizeResult``.
        params: The best found hyperparameters (optimizer semantics, as for
            the score).
        score: The best raw "optimizer" score of the phase (higher is
            better).
        trials: The frozen trials of the current phase ONLY — transferred
            ``previous_trials`` are excluded. Always present on a non-``None``
            result; the list may contain COMPLETED/PRUNED/FAILED trials of
            the phase.
        complexity_guard: Log records of the adaptive complexity guard for
            the phase (one per restricted parameter, "before → after" with
            the reason). ``None`` when the guard is disabled or produced no
            records.
        n_trials_evaluated: The number of trials actually evaluated by the
            phase — the count of the phase's own COMPLETED trials
            (transferred ``previous_trials`` never count). Reported as the
            honest denominator next to the best-trial ``score``: on a fixed
            number of folds the score is the maximum over the evaluated
            trials, so the more trials were evaluated, the more optimistic the
            best score is expected to be. Defaults to 0 when the field is
            not computed.
        oof_predictions: The out-of-fold predictions of the phase's own
            COMPLETED trials with a valid score, one row per trial and repeat
            (shape ``(n_trials × n_repeats, n_rows)``; with one repeat —
            ``(n_trials, n_rows)``). Rows are aligned to ``X`` by position;
            rows a trial did not predict — e.g. the train part of
            ``train_test_split`` — are NaN. Collected only when
            ``collect_oof=True`` (the input of the BBC-CV estimate);
            ``None`` otherwise, when no trial qualifies, or when the total
            size exceeds ``BBC_MAX_OOF_CELLS``.
        trial_stats: The aggregated statistics of the current phase's own
            trials (state counters and the phase duration); see
            ``TrialStats``. Always present on a non-``None`` result.
    """

    model: Any | None
    params: dict[str, Any] | None
    score: float | None
    trials: list[FrozenTrial]
    complexity_guard: list[dict[str, Any]] | None = None
    n_trials_evaluated: int = 0
    oof_predictions: np.ndarray | None = None
    trial_stats: TrialStats = field(default_factory=TrialStats)


# ═══════════════════════════════════ constants ═══════════════════════════════
# The worst possible trial score — an analog of the float32 minimum or
# float('-inf'). Returned as the objective value when the real metric is not
# finite (NaN/+inf, e.g. an nrmse error). The value is practically unreachable
# for real metrics, hence used as a sentinel.
#
# Used in two places:
# 1. Inside `_objective`: a non-finite trial avg_score is replaced with this
#    constant (the trial completes, but with a deliberately worst score).
# 2. On the orchestrator side: the `valid_results` filter in
#    `training_engine/component.py` (``_valid_phase_results``) drops results whose score belongs to the
#    worst-score sentinel class — any score at or below WORST_SCORE_THRESHOLD —
#    treating the algorithm as failed.
HPO_WORST_SCORE = -3.4028235e38


# The lower bound of the "worst-score sentinel" class (see the docstring of
# ``common.definitions.WORST_SCORE_THRESHOLD`` — the canonical definition);
# re-exported here so the HPO-side checks and the documented import path
# (``configurable_automl_engine.tuner.WORST_SCORE_THRESHOLD``) keep working.
WORST_SCORE_THRESHOLD = _WORST_SCORE_THRESHOLD
