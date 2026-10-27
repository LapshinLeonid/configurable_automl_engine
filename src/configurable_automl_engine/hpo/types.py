"""Public types of the hyperparameter optimisation (HPO) package.

The exceptions (``HyperoptError`` and its subclasses), the self-documenting
phase result ``OptimizeResult`` and the worst-score sentinel constants
(``HPO_WORST_SCORE`` / ``WORST_SCORE_THRESHOLD``). They are re-exported by
``configurable_automl_engine.tuner``, their documented import path. The
module holds definitions only and must not import any other module of the
``hpo`` package.
"""

from __future__ import annotations

# ─────────────────────────────── stdlib
from dataclasses import dataclass
from typing import Any

# ──────────────────────────── third-party
import numpy as np
from optuna.trial import FrozenTrial


# ═════════════════════════════════════ exceptions ════════════════════════════
class HyperoptError(Exception):
    """Base error of the hyperparameter optimisation module."""

    # The documented identity: the class is imported as
    # ``configurable_automl_engine.tuner.HyperoptError`` (re-exported there), and
    # tracebacks and pickles keep that path.
    __module__ = "configurable_automl_engine.tuner"


class InvalidAlgorithmError(HyperoptError):
    """Error raised when an algorithm is not found or is marked as unused."""

    # The documented identity: the class is imported as
    # ``configurable_automl_engine.tuner.InvalidAlgorithmError`` (re-exported there), and
    # tracebacks and pickles keep that path.
    __module__ = "configurable_automl_engine.tuner"


class InvalidDataError(HyperoptError):
    """Error raised when invalid X or y data structures are passed."""

    # The documented identity: the class is imported as
    # ``configurable_automl_engine.tuner.InvalidDataError`` (re-exported there), and
    # tracebacks and pickles keep that path.
    __module__ = "configurable_automl_engine.tuner"


class HPOHistoryError(HyperoptError):
    """Error raised when TPE history cannot be transferred between HPO phases.

    Raised in three situations:
    - the tuner does not accept the ``previous_trials`` argument, but the
      pipeline needs to transfer history (e.g. a custom tuner in a refine
      phase) — fail fast instead of silently degrading refine to a cold run;
    - the transferred trials have distributions incompatible with the current
      search space (``study.add_trials`` desync) — fail fast instead of
      silently skipping the history;
    - the current search space cannot be reproduced by the distribution
      pre-check (an exotic custom ``space_fn``) — the transfer cannot be
      validated, and silently dropping the history would degrade refine to a
      cold run (the exact bug this feature fixes).
    """

    # The documented identity: the class is imported as
    # ``configurable_automl_engine.tuner.HPOHistoryError`` (re-exported there), and
    # tracebacks and pickles keep that path.
    __module__ = "configurable_automl_engine.tuner"


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
            best score is expected to be. Defaults to 0 for legacy code paths
            that construct ``OptimizeResult`` without the field.
        oof_predictions: The out-of-fold predictions of the phase's own
            COMPLETED trials with a valid score, one row per trial (shape
            ``(n_trials, n_rows)``, rows aligned to ``X`` by position; rows a
            trial did not predict — e.g. the train part of
            ``train_test_split`` — are NaN). Collected only when
            ``collect_oof=True`` (the input of the BBC-CV estimate);
            ``None`` otherwise or when no trial qualifies.
    """

    model: Any | None
    params: dict[str, Any] | None
    score: float | None
    trials: list[FrozenTrial]
    complexity_guard: list[dict[str, Any]] | None = None
    n_trials_evaluated: int = 0
    oof_predictions: np.ndarray | None = None


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
#    `training_engine/component.py` drops results whose score belongs to the
#    worst-score sentinel class — any score at or below WORST_SCORE_THRESHOLD —
#    treating the algorithm as failed.
HPO_WORST_SCORE = -3.4028235e38


# The lower bound of the "worst-score sentinel" class: the exact float32
# minimum. Any score <= this bound is treated as a sentinel — real metrics of
# such magnitude are practically impossible (documented threshold decision).
# Comparing against this bound (instead of the exact
# HPO_WORST_SCORE via math.isclose) also catches the raw
# float(np.finfo(np.float32).min) value that numpy-cast metrics or custom
# tuners may return: its relative difference from HPO_WORST_SCORE is
# ≈ 9.88e-9, which the old isclose(rel_tol=1e-9) check silently missed.
WORST_SCORE_THRESHOLD = float(np.finfo(np.float32).min)
