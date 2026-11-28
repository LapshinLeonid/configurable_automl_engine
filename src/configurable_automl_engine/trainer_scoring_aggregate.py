"""Aggregation of validation scoring outcomes.

The module turns the per-repeat fold outcomes into the user-facing
``ValidationScores``: it computes the per-repeat values, combines them with the
configured aggregation policy, finalizes the OOF channel and emits the OOF
warnings. It is a pure module: neither the trainer nor the HPO modules are
imported here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from configurable_automl_engine import validation as _val
from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.common.metrics import (
    direction_label,
    oof_rmse,
    pooled_scorer_value,
    to_user_value,
    warn_pooled_fallback_once,
)
from configurable_automl_engine.common.types import (
    FloatArray,
    Frame,
    Scorer,
    Target,
)

from .trainer_data import _align_oof_vector, _count_valid_pairs
from .trainer_state import ValidationScores

_LOG = logging.getLogger(__name__)

#: The worst possible ``RMSE_oof`` of a repeat. A repeat whose OOF channel
#: contains non-finite predictions is excluded from the aggregate by the
#: finite-value filter of ``validation.aggregate_split_scores`` (an RMSE is a
#: "lower is better" error, so +inf is its worst outcome).
_WORST_OOF_SCORE = float("inf")


@dataclass(frozen=True)
class _RepeatScoring:
    """The fold metrics and OOF buffers of one repeated validation split."""

    fold_raw: list[float]
    add_raw: dict[str, list[float]]
    oof_pred: FloatArray
    oof_y: FloatArray
    fold_errors: list[str]
    oof_failed: bool
    repeat_index: int


@dataclass(frozen=True)
class ScoringPlan:
    """Inputs that define how repeated validation outcomes are aggregated."""

    outcomes: list[_RepeatScoring]
    X: Frame
    n_samples: int
    method: str
    metric: str
    scorer: Scorer
    add_scorers: dict[str, Scorer]
    pool_primary: bool
    pool_additive: dict[str, bool]
    algorithm: str
    split_aggregation: str


@dataclass(frozen=True)
class _RepeatAggregate:
    """The contribution of one repeat to the multi-repeat aggregates."""

    repeat_score: float
    add_scores: dict[str, float]
    oof_score: float
    oof_vector: Target
    n_valid_pairs: int


@dataclass(frozen=True)
class _RepeatAggregation:
    """The per-repeat values collected from all repeat outcomes."""

    repeat_scores: list[float]
    add_repeat: dict[str, list[float]]
    oof_repeat_scores: list[float]
    oof_vectors: list[Target]
    n_valid_pairs: int


@dataclass(frozen=True)
class _ScoreSummary:
    """Mean, median and standard deviation of the finite values."""

    mean: float | None
    median: float | None
    std: float | None


def _single_repeat_scores(
    outcome: _RepeatScoring,
    X: Frame,
    n_samples: int,
    *,
    method: str,
    metric: str,
    scorer: Scorer,
    add_scorers: dict[str, Scorer],
    pool_primary: bool,
    pool_additive: dict[str, bool],
    algorithm: str,
) -> ValidationScores:
    """Build the single-split result."""
    if not outcome.fold_raw:
        _raise_all_folds_failed(outcome)

    # Non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D²,
    # explained_variance, max error, median absolute error) are computed
    # over the pooled OOF vector at once, not averaged by folds: each row
    # enters the calculation exactly once. The pooled value is obtained
    # from the already-collected OOF buffer when the vector covers the
    # whole data (k_fold/loo); for train_test_split the buffer covers only
    # the hold-out part, so the metric is estimated on that subset — the
    # SAME subset as ``RMSE_oof`` (``_finalize_oof`` uses the identical
    # buffers), so val_score and oof_score_ stay mutually consistent by
    # construction. This is a documented limitation (identical to the OOF
    # channel: rows from the train part have no honest out-of-fold
    # prediction). The raw per-fold values still drive the fail-fast
    # check above (no fold produced a valid score).
    raw_mean = _aggregate_raw_values(
        scorer,
        outcome.fold_raw,
        pooled=pool_primary,
        oof_y=outcome.oof_y,
        oof_pred=outcome.oof_pred,
        algorithm=algorithm,
        phase_name=f"final_fit:{metric}",
        failure_reason="the pooled value over the OOF buffer cannot be "
        "computed (no raw scorer metric function or no valid "
        "pairs)",
        empty_reason="the OOF buffer contains no finite (y_true, y_pred) pairs",
    )
    val_score = to_user_value(metric, raw_mean)
    # User-facing semantics: val_score is a value in user representation,
    # so the direction in the log is user_direction ("min better" for
    # errors), not the optimizer greater_is_better flag (which is true for
    # neg_ metrics even though a smaller value is better for the user).
    _LOG.debug(
        f"Metric calculation: raw={raw_mean:.4f},"
        f" final val_score={val_score:.4f} "
        f"({direction_label(metric)})"
    )

    additional_scores: dict[str, float] = {}
    for mname, raw_values in outcome.add_raw.items():
        if not raw_values:
            # All folds failed for this metric — it was already skipped with a WARNING.
            continue
        raw_values_agg = _aggregate_raw_values(
            add_scorers[mname],
            raw_values,
            pooled=pool_additive[mname],
            oof_y=outcome.oof_y,
            oof_pred=outcome.oof_pred,
            algorithm=algorithm,
            phase_name=f"final_fit:{mname}",
            failure_reason=f"the pooled value for additional metric "
            f"'{mname}' cannot be computed (no raw scorer metric "
            f"function or no valid pairs)",
            empty_reason=f"the OOF buffer contains no finite "
            f"(y_true, y_pred) pairs for additional metric "
            f"'{mname}'",
        )
        additional_scores[mname] = to_user_value(mname, raw_values_agg)

    # OOF channel: RMSE_oof on the whole vector, warnings about
    # coverage/small N, and alignment of y_pred_oof to the original row order.
    oof_predictions, oof_score, oof_coverage = _finalize_oof(
        outcome.oof_pred,
        outcome.oof_y,
        X,
        n_samples,
        method=method,
    )
    return ValidationScores(
        val_score=val_score,
        additional_scores=additional_scores,
        oof_predictions=oof_predictions,
        oof_score=oof_score,
        oof_coverage=oof_coverage,
    )


def _raise_all_folds_failed(outcome: _RepeatScoring) -> None:
    """Raise the primary-metric failure with the unique fold errors."""
    detail = "; ".join(dict.fromkeys(outcome.fold_errors)) or "unknown failure"
    raise TrainingError(f"Validation scoring failed for all folds: {detail}")


def _multi_repeat_scores(plan: ScoringPlan) -> ValidationScores:
    """Aggregate the per-repeat estimates and OOF vectors.

    Args:
        plan: The scoring inputs and the per-repeat outcomes.

    Returns:
        ValidationScores: The aggregated validation scores.

    Raises:
        TrainingError: If no repeat produced a valid primary-metric score.
    """
    _ensure_some_fold_scored(plan.outcomes)
    collected = _collect_repeat_aggregates(plan)
    summary = _score_summary(collected.repeat_scores)
    oof_summary = _score_summary(collected.oof_repeat_scores)
    oof_score = _aggregate_finite_scores(
        collected.oof_repeat_scores, plan.split_aggregation
    )
    oof_coverage = _oof_coverage(
        collected.n_valid_pairs, plan.n_samples, len(plan.outcomes)
    )
    _log_multi_repeat_oof(
        collected.n_valid_pairs,
        plan.n_samples,
        len(plan.outcomes),
        method=plan.method,
        coverage=oof_coverage,
        oof_score=oof_score,
        split_aggregation=plan.split_aggregation,
    )
    return ValidationScores(
        val_score=float(
            _val.aggregate_split_scores(collected.repeat_scores, plan.split_aggregation)
        ),
        additional_scores=_aggregate_additional_scores(
            collected.add_repeat, plan.split_aggregation
        ),
        oof_predictions=collected.oof_vectors[
            _primary_vector_index(collected.oof_repeat_scores, oof_score)
        ],
        oof_score=oof_score,
        oof_coverage=oof_coverage,
        repeat_scores=collected.repeat_scores,
        score_mean=summary.mean,
        score_median=summary.median,
        score_std=summary.std,
        oof_repeat_scores=collected.oof_repeat_scores,
        oof_score_std=oof_summary.std,
        oof_predictions_by_repeat=collected.oof_vectors,
    )


def _ensure_some_fold_scored(outcomes: list[_RepeatScoring]) -> None:
    """Raise when no repeat produced a valid primary-metric fold score."""
    if any(outcome.fold_raw for outcome in outcomes):
        return
    detail = "; ".join(_unique_fold_errors(outcomes)) or "unknown failure"
    raise TrainingError(f"Validation scoring failed for all folds: {detail}")


def _unique_fold_errors(outcomes: list[_RepeatScoring]) -> list[str]:
    """Return the fold errors of all repeats without duplicates."""
    return list(
        dict.fromkeys(error for outcome in outcomes for error in outcome.fold_errors)
    )


def _collect_repeat_aggregates(plan: ScoringPlan) -> _RepeatAggregation:
    """Collect the per-repeat values and OOF vectors of every outcome."""
    repeat_scores: list[float] = []
    add_repeat: dict[str, list[float]] = {m: [] for m in plan.add_scorers}
    oof_repeat_scores: list[float] = []
    oof_vectors: list[Target] = []
    n_valid_pairs = 0
    for outcome in plan.outcomes:
        aggregate = _repeat_aggregate(outcome, plan)
        repeat_scores.append(aggregate.repeat_score)
        for mname in plan.add_scorers:
            add_repeat[mname].append(aggregate.add_scores[mname])
        oof_repeat_scores.append(aggregate.oof_score)
        oof_vectors.append(aggregate.oof_vector)
        n_valid_pairs += aggregate.n_valid_pairs
    return _RepeatAggregation(
        repeat_scores=repeat_scores,
        add_repeat=add_repeat,
        oof_repeat_scores=oof_repeat_scores,
        oof_vectors=oof_vectors,
        n_valid_pairs=n_valid_pairs,
    )


def _repeat_aggregate(outcome: _RepeatScoring, plan: ScoringPlan) -> _RepeatAggregate:
    """Build the aggregate of a failed or a successful repeat."""
    if not outcome.fold_raw or outcome.oof_failed:
        return _failed_repeat_aggregate(outcome, plan)
    return _successful_repeat_aggregate(outcome, plan)


def _finite_values(values: list[float]) -> list[float]:
    """Return the finite values of the list."""
    return [value for value in values if np.isfinite(value)]


def _score_summary(values: list[float]) -> _ScoreSummary:
    """Summarize the finite values with their mean, median and std."""
    finite = _finite_values(values)
    if not finite:
        return _ScoreSummary(mean=None, median=None, std=None)
    array = np.asarray(finite, dtype=float)
    return _ScoreSummary(
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=float(np.std(array)),
    )


def _aggregate_finite_scores(
    values: list[float], split_aggregation: str
) -> float | None:
    """Combine the finite values or return ``None`` when there are none."""
    finite = _finite_values(values)
    if not finite:
        return None
    return float(_val.aggregate_split_scores(finite, split_aggregation))


def _primary_vector_index(
    oof_repeat_scores: list[float], oof_score: float | None
) -> int:
    """Pick the repeat whose OOF score is closest to the aggregate."""
    if oof_score is None:
        return 0
    finite_indices = [
        index for index, value in enumerate(oof_repeat_scores) if np.isfinite(value)
    ]
    return min(finite_indices, key=lambda i: abs(oof_repeat_scores[i] - oof_score))


def _aggregate_additional_scores(
    add_repeat: dict[str, list[float]], split_aggregation: str
) -> dict[str, float]:
    """Combine the per-repeat additional-metric values that have a finite value."""
    additional_scores: dict[str, float] = {}
    for mname, values in add_repeat.items():
        if not _finite_values(values):
            continue
        additional_scores[mname] = float(
            _val.aggregate_split_scores(values, split_aggregation)
        )
    return additional_scores


def _oof_coverage(n_valid_pairs: int, n_samples: int, n_repeats: int) -> float:
    """Return the fraction of OOF cells that hold a valid prediction/target pair."""
    total_cells = n_samples * n_repeats
    return float(n_valid_pairs / total_cells) if total_cells > 0 else 0.0


def _failed_repeat_aggregate(
    outcome: _RepeatScoring, plan: ScoringPlan
) -> _RepeatAggregate:
    """Build the aggregate of a repeat excluded from every aggregate.

    Args:
        outcome: The outcome of the failed repeat.
        plan: The scoring inputs.

    Returns:
        _RepeatAggregate: NaN repeat/additional scores, the worst OOF score
            and the aligned OOF vector.
    """
    if not outcome.fold_raw:
        _LOG.warning(
            "Repeat %d failed: no fold produced a valid score for the "
            "primary metric; the repeat is excluded from the aggregate.",
            outcome.repeat_index,
        )
    elif outcome.oof_failed:
        _LOG.warning(
            "Repeat %d has non-finite OOF predictions on a fold: the "
            "whole repeat gets the worst score and is excluded from the "
            "primary, additional and OOF aggregates.",
            outcome.repeat_index,
        )
    return _RepeatAggregate(
        repeat_score=float("nan"),
        add_scores={mname: float("nan") for mname in plan.add_scorers},
        oof_score=_WORST_OOF_SCORE,
        oof_vector=_align_oof_vector(outcome.oof_pred, plan.X),
        n_valid_pairs=_count_valid_pairs(outcome.oof_pred, outcome.oof_y),
    )


def _successful_repeat_aggregate(
    outcome: _RepeatScoring, plan: ScoringPlan
) -> _RepeatAggregate:
    """Build the aggregate of a repeat whose folds were all scored.

    Args:
        outcome: The outcome of the successful repeat.
        plan: The scoring inputs.

    Returns:
        _RepeatAggregate: The user-facing repeat score, the additional
            scores, the OOF score and the aligned OOF vector.
    """
    raw_mean = _aggregate_raw_values(
        plan.scorer,
        outcome.fold_raw,
        pooled=plan.pool_primary,
        oof_y=outcome.oof_y,
        oof_pred=outcome.oof_pred,
        algorithm=plan.algorithm,
        phase_name=f"final_fit:{plan.metric}",
        failure_reason="the pooled value over the OOF buffer cannot be "
        "computed (no raw scorer metric function or no valid "
        "pairs)",
        empty_reason="the OOF buffer contains no finite (y_true, y_pred) pairs",
    )
    repeat_score = to_user_value(plan.metric, raw_mean)

    add_scores: dict[str, float] = {mname: float("nan") for mname in plan.add_scorers}
    for mname, raw_values in outcome.add_raw.items():
        if not raw_values:
            continue
        raw_values_agg = _aggregate_raw_values(
            plan.add_scorers[mname],
            raw_values,
            pooled=plan.pool_additive[mname],
            oof_y=outcome.oof_y,
            oof_pred=outcome.oof_pred,
            algorithm=plan.algorithm,
            phase_name=f"final_fit:{mname}",
            failure_reason=f"the pooled value for additional metric "
            f"'{mname}' cannot be computed (no raw scorer metric "
            f"function or no valid pairs)",
            empty_reason=f"the OOF buffer contains no finite "
            f"(y_true, y_pred) pairs for additional metric "
            f"'{mname}'",
        )
        add_scores[mname] = to_user_value(mname, raw_values_agg)

    vector, repeat_oof_score, _coverage = _finalize_oof(
        outcome.oof_pred,
        outcome.oof_y,
        plan.X,
        plan.n_samples,
        method=plan.method,
        emit_warnings=False,
    )
    if repeat_oof_score is None:
        _LOG.warning(
            "Repeat %d produced no computable RMSE_oof; it is excluded "
            "from the aggregate.",
            outcome.repeat_index,
        )
        oof_score = _WORST_OOF_SCORE
    else:
        oof_score = float(repeat_oof_score)
    return _RepeatAggregate(
        repeat_score=repeat_score,
        add_scores=add_scores,
        oof_score=oof_score,
        oof_vector=vector,
        n_valid_pairs=_count_valid_pairs(outcome.oof_pred, outcome.oof_y),
    )


def _log_multi_repeat_oof(
    n_valid_pairs: int,
    n_samples: int,
    n_repeats: int,
    *,
    method: str,
    coverage: float,
    oof_score: float | None,
    split_aggregation: str,
) -> None:
    """Emit the aggregated OOF warnings and the summary log line."""
    total_cells = n_samples * n_repeats
    if method == "train_test_split":
        _LOG.warning(
            "OOF evaluation with 'train_test_split' across %d repeats covers "
            "only the validation part (%d/%d cells, %.0f%%): rows from the "
            "training part have no honest out-of-fold prediction, so "
            "RMSE_oof is estimated on the covered subset only.",
            n_repeats,
            n_valid_pairs,
            total_cells,
            coverage * 100.0,
        )
    if n_samples <= 5:
        _LOG.warning(
            "OOF evaluation on very small data (N=%d, repeats=%d): the OOF "
            "vectors are short and RMSE_oof has high variance.",
            n_samples,
            n_repeats,
        )
    if n_valid_pairs == 0:
        _LOG.warning(
            "RMSE_oof could not be computed: no valid out-of-fold "
            "prediction/target pairs in any repeat."
        )
        return
    dropped = total_cells - n_valid_pairs
    if dropped and method != "train_test_split":
        _LOG.warning(
            "%d OOF cell(s) excluded from RMSE_oof (non-finite prediction "
            "or target, or a failed repeat); coverage=%.3f.",
            dropped,
            coverage,
        )
    _LOG.info(
        "OOF evaluation across %d repeats: RMSE_oof(%s)=%.4f (min better) "
        "on %d/%d cells (coverage=%.3f).",
        n_repeats,
        split_aggregation,
        oof_score if oof_score is not None else float("nan"),
        n_valid_pairs,
        total_cells,
        coverage,
    )


def _aggregate_raw_values(
    scorer: Scorer,
    raw_values: list[float],
    *,
    pooled: bool,
    oof_y: FloatArray,
    oof_pred: FloatArray,
    algorithm: str,
    phase_name: str,
    failure_reason: str,
    empty_reason: str,
) -> float:
    """Aggregate the raw values of a metric: pooled OOF value or fold mean.

    A non-additive metric is computed over the pooled OOF buffer when it
    holds a finite pair; when the pooled value is impossible, the fold mean
    is used and the fallback WARNING is emitted once per (algorithm, phase)."""
    if pooled and np.any(np.isfinite(oof_pred) & np.isfinite(oof_y)):
        try:
            return pooled_scorer_value(scorer, oof_y, oof_pred)
        except ValueError:
            # The pooled value is impossible (no valid pairs); fall back to
            # the fold mean (RMSE_oof handles its own None semantics). The
            # WARNING is emitted once per (algorithm, metric, 'final_fit').
            warn_pooled_fallback_once(
                algo_name=algorithm, phase_name=phase_name, reason=failure_reason
            )
            return float(np.mean(raw_values))
    if pooled:
        warn_pooled_fallback_once(
            algo_name=algorithm, phase_name=phase_name, reason=empty_reason
        )
    return float(np.mean(raw_values))


def _finalize_oof(
    oof_pred: FloatArray,
    oof_y: FloatArray,
    X: Frame,
    n_samples: int,
    *,
    method: str,
    emit_warnings: bool = True,
) -> tuple[Target, float | None, float]:
    """Finalize the OOF evaluation channel.

    Based on the prediction/target buffers collected on validation folds
    (``score_on_validation_splits``), computes ``RMSE_oof`` over the whole
    vector at once via :func:`metrics.oof_rmse`, returns the OOF vector,
    ``RMSE_oof`` and the coverage, and logs warnings:
    - ``train_test_split`` covers only the validation part with OOF
      (a documented limitation);
    - very small N (2-5): a short OOF vector with high variance;
    - non-finite predictions/targets are excluded from RMSE_oof.

    Args:
        oof_pred: OOF prediction buffer (NaN for uncovered rows).
        oof_y: True value buffer (NaN for uncovered rows).
        X: Original feature matrix (to restore the pandas index).
        n_samples: Total number of rows in the original data.
        method: The validation method actually applied (after a possible
            fallback ``k_fold`` → ``train_test_split``), which drives the
            coverage warnings.
        emit_warnings: Whether to log the coverage/small-N warnings (the
            multi-repeat path emits them once for the aggregated coverage).
    """
    oof_vector = _align_oof_vector(oof_pred, X)
    valid = np.isfinite(oof_pred) & np.isfinite(oof_y)
    n_valid = int(valid.sum())
    coverage = float(n_valid / n_samples) if n_samples > 0 else 0.0

    # train_test_split limitation: OOF covers only the validation part.
    # Explicitly documented in the log.
    if emit_warnings and method == "train_test_split":
        _LOG.warning(
            "OOF evaluation with 'train_test_split' covers only the "
            "validation part (%d/%d rows, %.0f%%): rows from the training "
            "part have no honest out-of-fold prediction, so RMSE_oof is "
            "estimated on the covered subset only.",
            n_valid,
            n_samples,
            coverage * 100.0,
        )

    # Very small N: the OOF vector is short and the estimate has high variance.
    if emit_warnings and n_samples <= 5:
        _LOG.warning(
            "OOF evaluation on very small data (N=%d): the OOF vector is "
            "short and RMSE_oof has high variance.",
            n_samples,
        )

    if n_valid == 0:
        if emit_warnings:
            _LOG.warning(
                "RMSE_oof could not be computed: no valid out-of-fold "
                "prediction/target pairs."
            )
        return oof_vector, None, coverage

    dropped = n_samples - n_valid
    # For train_test_split the uncovered rows are the train part; the
    # limitation was already documented by a separate WARNING above, so the
    # generic row-exclusion warning is logged only for non-finite
    # predictions/targets. With the non-finite-prediction policy (folds with
    # NaN/Inf predictions are not collected at all, see
    # ``_collect_fold_oof``) the remaining exclusions come from non-finite
    # targets only.
    if emit_warnings and dropped and method != "train_test_split":
        _LOG.warning(
            "%d row(s) excluded from RMSE_oof (non-finite prediction or "
            "target); coverage=%.3f.",
            dropped,
            coverage,
        )

    oof_score: float | None
    try:
        oof_score = oof_rmse(oof_y, oof_pred)
    except ValueError as err:
        oof_score = None
        if emit_warnings:
            _LOG.warning("RMSE_oof could not be computed: %s", err)
    # RMSE_oof is a user-facing error value ("min better"): the direction
    # is stated explicitly, and the optimizer flag is not applied.
    if emit_warnings:
        _LOG.info(
            "OOF evaluation: RMSE_oof=%.4f (min better) on %d/%d rows (coverage=%.3f).",
            oof_score if oof_score is not None else float("nan"),
            n_valid,
            n_samples,
            coverage,
        )
    return oof_vector, oof_score, coverage
