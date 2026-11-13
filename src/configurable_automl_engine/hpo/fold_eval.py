"""Per-fold helpers of the trial evaluation.

The fold fit/predict step and its WARNINGs, shared by the additive
(``_evaluate_with_intermediate_reports``) and pooled (``_evaluate_pooled_cv``)
evaluation paths of ``hpo.evaluation``. The functions are pure helpers; the
callers keep the failure policy (a non-fatal fold failure penalizes the fold).
"""

from __future__ import annotations

import logging as _logging
from typing import Any

import numpy as np
from sklearn.base import clone

from configurable_automl_engine.common.metrics import (
    pooled_scorer_value,
    warn_pooled_fallback_once,
)

from .types import InvalidDataError

log = _logging.getLogger(__name__)


def _warn_failed_fold(fold_idx: int, err: Exception, stage: str) -> None:
    """Log the penalty of a failed fold at the given evaluation stage.

    Shared by the per-fold evaluation loops so the warning text stays uniform:
    a fold whose fit/predict/scoring raised a non-fatal exception makes the
    trial's estimate NaN — the ``cross_val_score(error_score=np.nan)``
    contract (a failed fold became NaN, i.e. the worst possible outcome), so
    a model that crashes on most folds cannot win over an honest one."""
    log.warning(
        "Fold %d failed to %s: %s: %s — the fold is penalized: the trial "
        "estimate becomes NaN (the worst outcome)",
        fold_idx,
        stage,
        type(err).__name__,
        err,
    )


def _warn_invalid_fold_predictions(fold_idx: int, n_finite: int, n_rows: int) -> None:
    """Log the penalty of a fold with non-finite predictions.

    A fold that completed but returned NaN/Inf predictions is penalized like
    a fold that raised: its rows cannot enter the pooled estimate, so the
    trial's estimate becomes NaN — the ``cross_val_score(error_score=np.nan)``
    contract (the worst possible outcome). The values are never replaced
    with zeros: that would also distort the metrics."""
    log.warning(
        "Fold %d returned non-finite predictions (%d of %d rows are "
        "NaN/Inf) — the fold is penalized: the trial estimate becomes NaN "
        "(the worst outcome)",
        fold_idx,
        n_rows - n_finite,
        n_rows,
    )


def _fit_predict_fold(
    estimator: Any, X_tr: Any, y_tr: Any, X_te: Any, fold_idx: int
) -> tuple[Any, np.ndarray] | None:
    """Fit a clone of the estimator on a fold and predict its test rows.

    Clone the estimator for every fold (as cross_val_score does) to avoid
    state accumulating between folds. A fold whose fit/predict raises a
    NON-FATAL exception is PENALIZED: ``None`` is returned (its rows stay NaN
    in the pooled buffers, they cannot enter the estimate), the intermediate
    reports of the surviving folds are still published, and a warning is
    logged (the ``cross_val_score(error_score=np.nan)`` contract: a failed
    fold became NaN — the worst possible outcome — and must not abort the
    trial, but a model that crashes on most folds must not win over an
    honest one). A fold that COMPLETES but returns non-finite (NaN/Inf)
    predictions is penalized the same way: the model was required to predict
    these rows, and masking them out of the pooled estimate would let a
    model that fails on most of the validation rows win with a good estimate
    computed on the few rows it happened to predict (the values are never
    replaced with zeros — that would also distort the metrics). Fatal
    exception types (MemoryError / RuntimeError / InvalidDataError) are NOT
    swallowed here — they propagate to ``handle_failure`` and keep the
    circuit breaker active.

    Args:
        estimator: The estimator (possibly a pipeline) of the fold.
        X_tr: The train features of the fold.
        y_tr: The train target of the fold.
        X_te: The test features of the fold.
        fold_idx: The 1-based fold index (for the warning).

    Returns:
        tuple[Any, np.ndarray] | None: The fitted fold estimator and its
            (finite) predictions, or ``None`` — the fold failed and is
            penalized.
    """
    try:
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_tr, y_tr)
    except (MemoryError, RuntimeError, InvalidDataError):
        raise
    except Exception as err:  # noqa: BLE001 — per-fold failure, cross_val_score(error_score=nan) semantics
        _warn_failed_fold(fold_idx, err, "fit or predict")
        return None
    try:
        fold_pred = np.asarray(fold_estimator.predict(X_te)).reshape(-1)
    except (MemoryError, RuntimeError, InvalidDataError):
        raise
    except Exception as err:  # noqa: BLE001 — per-fold failure, cross_val_score(error_score=nan) semantics
        _warn_failed_fold(fold_idx, err, "fit or predict")
        return None
    # The fold's test rows are rows the model was REQUIRED to predict: a
    # non-finite prediction is a quality defect of the model (not a coverage
    # gap), so the fold is penalized exactly like a fold that raised.
    if not np.isfinite(fold_pred.astype(float, copy=False)).all():
        _warn_invalid_fold_predictions(
            fold_idx,
            int(np.isfinite(fold_pred.astype(float, copy=False)).sum()),
            len(fold_pred),
        )
        return None
    return fold_estimator, fold_pred


def _final_pooled_estimate(
    scorer: Any,
    fold_scores: list[float],
    fold_pairs: list[tuple[np.ndarray, np.ndarray]],
    pooled_y: np.ndarray,
    pooled_pred: np.ndarray,
    *,
    poolable: bool,
    n_folds_total: int,
    n_failed_folds: int,
    algo_name: str,
    phase_name: str,
) -> float:
    """The final estimate of ``_evaluate_pooled_cv``.

    Returns:
        The pooled value, the per-fold fallback mean, or NaN when a fold
        failed (the caller turns it into ``HPO_WORST_SCORE``).
    """
    if n_folds_total > 0 and n_failed_folds == n_folds_total:
        raise ValueError(
            f"All {n_folds_total} fold(s) failed during evaluation — the "
            "pooled estimate is impossible"
        )
    # A failed fold is a guaranteed worst outcome — the semantics of
    # ``cross_val_score(error_score=np.nan)``, whose NaN fold poisoned
    # the estimate. A model that crashes on part of the folds must not win
    # over an honest one.
    if n_failed_folds > 0:
        return float("nan")
    if poolable:
        try:
            return pooled_scorer_value(scorer, pooled_y, pooled_pred)
        except ValueError:
            # The pooled value is impossible (all rows masked out, or the raw
            # metric function raised on the pooled vector): fall back to the
            # mean of the per-fold values, scored lazily from the RETAINED
            # per-fold pairs. The scorer is invoked only now — not on the
            # common path — so per-fold scoring noise (e.g. LOO+R², where
            # every fold R² is undefined) does not pollute the logs, and NO
            # split is iterated or fold re-trained a second time.
            warn_pooled_fallback_once(
                algo_name=algo_name or "unknown",
                phase_name=phase_name or "untitled_phase",
                reason="the pooled value over the collected OOF rows cannot "
                "be computed (no valid pairs, or the raw metric function "
                "raised on the pooled vector)",
            )
            for yt, yp in fold_pairs:
                try:
                    fold_scores.append(pooled_scorer_value(scorer, yt, yp))
                except ValueError:
                    pass
            return float(np.mean(fold_scores)) if fold_scores else float("nan")
    warn_pooled_fallback_once(
        algo_name=algo_name or "unknown",
        phase_name=phase_name or "untitled_phase",
        reason="the scorer does not expose a raw metric function "
        "(the pooled-evaluation contract of common.sklearn_compat)",
    )
    return float(np.mean(fold_scores)) if fold_scores else float("nan")


def _running_pooled_value(
    scorer: Any,
    fold_estimator: Any,
    X_te: Any,
    y_te: Any,
    pooled_y_buf: np.ndarray,
    pooled_pred_buf: np.ndarray,
    fold_scores: list[float],
) -> float:
    """The running pooled metric value over the rows covered so far.

    The per-fold scorer value is used only as the fallback when the pooled
    value cannot be computed; for non-additive metrics a fold with non-finite
    predictions must not abort the trial (those rows are masked out of the
    pooled value). The scorer is called lazily — only when the pooled value
    turns out to be impossible — so that per-fold scoring noise (e.g. LOO+R²,
    where every fold R² is undefined) does not pollute the logs on the common
    path.

    Returns:
        The running value (NaN when no estimate is possible yet).
    """
    try:
        return pooled_scorer_value(scorer, pooled_y_buf, pooled_pred_buf)
    except ValueError:
        # A transient pooled failure (e.g. the first fold predicted
        # all-NaN, so the accumulated buffer has no valid pair yet)
        # is NOT a metric degradation: later folds may still produce a
        # valid pooled value. Only the per-fold fallback value is used
        # to publish the intermediate step; no WARNING is emitted here
        # (the warning is reserved for the final return point, where a
        # real fallback decision is made).
        try:
            fold_scores.append(float(scorer(fold_estimator, X_te, y_te)))
        except Exception:  # noqa: BLE001, S110 — only a fallback for the pooled value
            pass
        return float(np.mean(fold_scores)) if fold_scores else float("nan")


def _final_intermediate_estimate(
    scorer: Any,
    fold_scores: list[float],
    pooled_y_buf: np.ndarray,
    pooled_pred_buf: np.ndarray,
    *,
    pool_metric: bool,
    n_folds_total: int,
    n_failed_folds: int,
    algo_name: str,
    phase_name: str,
) -> float:
    """The final estimate of ``_evaluate_with_intermediate_reports``.

    Returns:
        The fold mean (additive metrics) or the pooled value (non-additive
        metrics); NaN when a fold failed.
    """
    if n_folds_total > 0 and n_failed_folds == n_folds_total:
        raise ValueError(
            f"All {n_folds_total} fold(s) failed during evaluation — the "
            "estimate is impossible"
        )
    # A failed fold is a guaranteed worst outcome: its rows could not enter
    # the pooled estimate (pooled path) or its score is NaN in the additive
    # fold mean — the trial estimate becomes NaN, mirroring
    # ``cross_val_score(error_score=np.nan)``, whose NaN fold poisoned the
    # mean. A model that crashes on part of the folds must not win over an
    # honest one.
    if n_failed_folds > 0:
        return float("nan")
    if pool_metric:
        try:
            return pooled_scorer_value(scorer, pooled_y_buf, pooled_pred_buf)
        except ValueError:
            warn_pooled_fallback_once(
                algo_name=algo_name or "unknown",
                phase_name=phase_name or "untitled_phase",
                reason="the pooled value over the accumulated OOF rows "
                "cannot be computed (no raw scorer metric function or no "
                "valid pairs)",
            )
            return float(np.mean(fold_scores)) if fold_scores else float("nan")
    return float(np.mean(fold_scores)) if fold_scores else float("nan")
