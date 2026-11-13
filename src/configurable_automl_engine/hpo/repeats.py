"""Repeat handling of the multi-split HPO scoring.

Helpers shared by the trial evaluators (``hpo.evaluation``): the deterministic
repeat seed, the per-repeat OOF buffer, the per-repeat failure policy
(a non-finite repeat becomes ``HPO_WORST_SCORE`` with a WARNING; all failed
repeats fail the trial), the per-repeat aggregation (median/mean) and the
stacking of the trial OOF vectors along the repeat axis. The functions are
pure (no runner state) and may import ``hpo.plans`` / ``hpo.types`` and lower
layers only.
"""

from __future__ import annotations

import logging as _logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from optuna.trial import FrozenTrial
from sklearn.model_selection import KFold

from configurable_automl_engine.validation import (
    SUPPORTED_SPLIT_AGGREGATIONS,
    aggregate_split_scores,
    repeated_seed,
    resolved_n_repeats,
)

from .plans import ValidationPlan
from .types import HPO_WORST_SCORE

log = _logging.getLogger(__name__)

#: A per-repeat evaluator: (repeat index, repeat seed, repeat OOF buffer) -> score.
RepeatScorer = Callable[[int, "int | None", "np.ndarray | None"], float]


def plan_repeats(
    *,
    n_repeats: int,
    split_aggregation: str,
    random_state: int | None,
    method: str,
) -> tuple[int, tuple[int | None, ...], str]:
    """Resolve the repeat plan of a validation method.

    The effective number of repeats is resolved against the RESOLVED method
    (LOO has deterministic folds — its repeats are ignored with a WARNING by
    ``resolved_n_repeats``), and every repeat gets the deterministic seed
    ``random_state + r``. The aggregation policy is validated here, so an
    invalid value is rejected once per run instead of once per repeat.

    Args:
        n_repeats: The requested number of repeats.
        split_aggregation: The per-repeat aggregation policy.
        random_state: The base split seed.
        method: The RESOLVED validation method.

    Returns:
        tuple: ``(effective n_repeats, repeat seeds, aggregation policy)``.

    Raises:
        ValueError: If ``n_repeats`` or ``split_aggregation`` is invalid.
    """
    if split_aggregation not in SUPPORTED_SPLIT_AGGREGATIONS:
        raise ValueError(
            "Unknown split_aggregation "
            f"{split_aggregation!r}; expected one of "
            f"{SUPPORTED_SPLIT_AGGREGATIONS}"
        )
    count = resolved_n_repeats(n_repeats, method)
    seeds = tuple(repeated_seed(random_state, r) for r in range(count))
    return count, seeds, split_aggregation


def repeat_seed(
    validation: ValidationPlan, random_state: int | None, repeat_index: int
) -> int | None:
    """Return the seed of a repeat.

    The plan's ``repeat_seeds`` are the single source of the repeat seeds when
    they were built for the plan's ``n_repeats``; a manually constructed plan
    without them falls back to ``repeated_seed(random_state, repeat_index)``
    (``random_state + r``, wrapped into the sklearn seed range).

    Args:
        validation: The resolved validation plan.
        random_state: The base split seed of the run.
        repeat_index: The zero-based repeat index.

    Returns:
        The repeat seed (``None`` — a non-deterministic run).
    """
    seeds = validation.repeat_seeds
    if len(seeds) == validation.n_repeats and repeat_index < len(seeds):
        return seeds[repeat_index]
    return repeated_seed(random_state, repeat_index)


def repeat_cv(validation: ValidationPlan, seed: int | None) -> Any:
    """Return the CV splitter of one repeat (same folds as ``iter_splits``).

    The repeated k-fold splits are re-drawn with the deterministic per-repeat
    seed; LOO has deterministic folds (its repeats are resolved away before
    this helper is reached), so the plan's own splitter is reused.

    Args:
        validation: The resolved validation plan.
        seed: The seed of the repeat.

    Returns:
        The sklearn CV splitter of the repeat.
    """
    if validation.method == "loo":
        return validation.cv
    return KFold(n_splits=validation.n_folds, shuffle=True, random_state=seed)


def repeat_oof_buffer(
    oof_out: np.ndarray | None, repeat_index: int
) -> np.ndarray | None:
    """Return the OOF row-buffer of one repeat.

    With several repeats the collector allocates a ``(n_repeats, n_rows)``
    matrix; with one repeat the classic flat ``(n_rows,)`` buffer is kept.

    Args:
        oof_out: The OOF buffer of the trial (``None`` — not collecting).
        repeat_index: The zero-based repeat index.

    Returns:
        The buffer of the repeat, or ``None``.
    """
    if oof_out is None or oof_out.ndim == 1:
        return oof_out
    return oof_out[repeat_index]


def penalize_non_finite_repeat(value: float, repeat_index: int) -> float:
    """Replace a non-finite repeat score with ``HPO_WORST_SCORE`` (with a WARNING).

    A degenerate metric value of one repeat (NaN/±inf) must not fail the
    whole trial: the repeat is penalized, the remaining repeats still carry
    the evidence. Only when EVERY repeat is non-finite the aggregate becomes
    the worst-score sentinel as well (the classic score semantics).

    Args:
        value: The raw score of the repeat.
        repeat_index: The zero-based repeat index (for the WARNING).

    Returns:
        The finite score, or ``HPO_WORST_SCORE``.
    """
    score = float(value)
    if np.isfinite(score):
        return score
    log.warning(
        "Repeat %d returned a non-finite score (%r) — the repeat is "
        "penalized with HPO_WORST_SCORE; the trial score is aggregated over "
        "the remaining repeats.",
        repeat_index,
        score,
    )
    return float(HPO_WORST_SCORE)


def run_repeat_scores(
    score_repeat: RepeatScorer,
    *,
    validation: ValidationPlan | None,
    n_repeats: int | None,
    random_state: int | None,
    oof_out: np.ndarray | None,
    split_aggregation: str,
    all_failed_message: str,
    scores_out: list[float] | None = None,
    mask_out: list[bool] | None = None,
) -> float:
    """Evaluate every repeat and aggregate the per-repeat scores.

    A repeat whose evaluation raises a non-fatal ``ValueError`` (all folds of
    the repeat failed, or non-finite hold-out predictions) is penalized with
    ``HPO_WORST_SCORE`` and a WARNING; the other repeats continue. When ALL
    repeats fail, the trial fails as before (the ``ValueError`` is re-raised).

    Args:
        score_repeat: The per-repeat evaluator.
        validation: The resolved validation plan (``None`` for a direct
            evaluator call — the seeds fall back to ``random_state + r``).
        n_repeats: The number of repeats (``None`` — the plan's count).
        random_state: The base split seed of the run.
        oof_out: The OOF buffer of the trial.
        split_aggregation: ``'median'`` or ``'mean'``.
        all_failed_message: The error message of the all-repeats-failed path.
        scores_out: An optional collector receiving the per-repeat scores
            (the trial report / pruning diagnostics).
        mask_out: An optional collector receiving one flag per repeat —
            ``True`` only for a repeat that produced a finite score. The OOF
            row of a failed/non-finite repeat must not enter BBC-CV.

    Returns:
        The aggregated score (``HPO_WORST_SCORE`` when every repeat was
        penalized for a non-finite score, a ``ValueError`` when every repeat
        failed to evaluate).
    """
    scores: list[float] = []
    success: list[bool] = []
    n_failed = 0
    count = (
        n_repeats
        if n_repeats is not None
        else (validation.n_repeats if validation is not None else 1)
    )
    for repeat_index in range(count):
        seed = (
            repeat_seed(validation, random_state, repeat_index)
            if validation is not None
            else repeated_seed(random_state, repeat_index)
        )
        sub_oof = repeat_oof_buffer(oof_out, repeat_index)
        try:
            value = score_repeat(repeat_index, seed, sub_oof)
        except ValueError as err:
            n_failed += 1
            log.warning(
                "Repeat %d failed to evaluate: %s: %s — the repeat is "
                "penalized with HPO_WORST_SCORE.",
                repeat_index,
                type(err).__name__,
                err,
            )
            scores.append(float(HPO_WORST_SCORE))
            success.append(False)
            continue
        finite = bool(np.isfinite(float(value)))
        scores.append(penalize_non_finite_repeat(value, repeat_index))
        success.append(finite)
    if n_failed >= count:
        raise ValueError(all_failed_message)
    if scores_out is not None:
        scores_out.extend(scores)
    if mask_out is not None:
        mask_out.extend(success)
    if len(scores) > 1:
        log.info(
            "Multi-split trial score: repeats=%d, aggregation=%s, mean=%.6f, std=%.6f",
            len(scores),
            split_aggregation,
            float(np.mean(scores)),
            float(np.std(scores)),
        )
    return aggregate_split_scores(scores, split_aggregation)


def phase_oof_matrix(
    phase_trials: list[FrozenTrial],
    trial_oof: dict[int, np.ndarray],
    *,
    max_cells: int,
) -> np.ndarray | None:
    """Stack the kept OOF rows of the phase's completed trials.

    With ``n_repeats > 1`` every trial stores one OOF vector per repeat
    (``(n_repeats, n_rows)``): the repeats are stacked along the
    configuration axis, so every repeat of every trial becomes a separate
    BBC-CV configuration. A single-repeat trial keeps the classic
    ``(n_rows,)`` vector and contributes one row. Above ``max_cells`` the
    matrix is not built (the BBC-CV budget) and ``None`` is returned with a
    WARNING.

    Args:
        phase_trials: The phase's own trials.
        trial_oof: The kept OOF vectors by trial number.
        max_cells: The maximum number of OOF values to keep.

    Returns:
        np.ndarray | None: A ``(n_configurations, n_rows)`` matrix, or
            ``None`` when no completed trial kept its predictions or the
            budget is exceeded.
    """
    blocks: list[np.ndarray] = []
    for trial in phase_trials:
        if (
            trial.state != optuna.trial.TrialState.COMPLETE
            or trial.number not in trial_oof
        ):
            continue
        oof = np.asarray(trial_oof[trial.number], dtype=float)
        if oof.ndim == 1:
            blocks.append(oof.reshape(1, -1))
        elif oof.ndim == 2 and oof.shape[1] > 0:
            blocks.append(oof)
        else:
            log.warning(
                "Trial %d kept OOF predictions of an unsupported shape %s — "
                "the vector is ignored by BBC-CV.",
                trial.number,
                oof.shape,
            )
    if not blocks:
        return None
    total = sum(block.size for block in blocks)
    if total > max_cells:
        log.warning(
            "BBC-CV estimate skipped: keeping %d OOF values (trials x "
            "repeats x rows) exceeds the budget of %d values.",
            total,
            max_cells,
        )
        return None
    return np.vstack(blocks)


def successful_repeat_oof(
    oof: np.ndarray | None, mask: list[bool], n_repeats: int
) -> np.ndarray | None:
    """Keep only the OOF rows of the successful repeats.

    A repeat penalized with ``HPO_WORST_SCORE`` (a failed evaluation or a
    non-finite score) may have a partially or fully NaN OOF row. Such a row
    must not enter BBC-CV: the common-row filter would otherwise reject every
    row of the estimate. With one repeat (or an incomplete mask) the buffer is
    returned unchanged.

    Args:
        oof: The trial OOF buffer (``(n_repeats, n_rows)`` for a multi-split).
        mask: The per-repeat success flags.
        n_repeats: The number of repeats of the trial.

    Returns:
        The OOF buffer without the rows of the failed repeats.
    """
    if oof is None or n_repeats <= 1 or len(mask) != n_repeats:
        return oof
    return oof[np.asarray(mask, dtype=bool)]
