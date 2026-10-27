"""Trial evaluation of the HPO runner.

Scores a trial pipeline with the resolved validation strategy
(``score_trial``: hold-out, fold-wise with intermediate reports for the
Optuna pruner, the pooled out-of-fold vector for non-additive metrics, or
the cross-validated fold mean) and returns the out-of-fold predictions for
BBC-CV. It also holds the scorer factory and the stratified hold-out split
helpers. It may import ``hpo.types`` and ``hpo.plans`` only; it must not
import ``hpo.runner`` or ``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

import logging as _logging
from typing import Any, cast

import numpy as np
import optuna
from optuna.trial import FrozenTrial, Trial
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.utils import _safe_indexing

from configurable_automl_engine.training_engine.metrics import (
    get_scorer_object,
    pooled_scorer_value,
    warn_pooled_fallback_once,
)
from configurable_automl_engine.validation import iter_splits

from .plans import TrialScore, ValidationPlan
from .types import (
    HPO_WORST_SCORE,
    HyperoptError,
    InvalidDataError,
)

log = _logging.getLogger(__name__)


def _build_scorer(name: str) -> Any:
    """Create a scorer object from a metric name.

    Args:
        name (str): String metric name
            (e.g. 'r2' or 'neg_mean_squared_error').
    Returns:
        Any: A metric object compatible with the sklearn API.
    Raises:
        HyperoptError: If an unknown metric name is given.
    """
    try:
        # Use the new API, which returns either a make_scorer object or a string
        return get_scorer_object(name)
    except Exception as err:
        raise HyperoptError(f"Unknown metric name: '{name}'") from err


def _can_stratify(y: Any) -> bool:
    """Determine whether stratification is possible for the y vector.

    Args:
        y (Any): Target variable vector.
    Returns:
        bool: True if the data is discrete (integers/bool) and the number of
            unique classes does not exceed 15. Otherwise — False.
    """
    # np.asarray guarantees a plain np.ndarray and np.dtype for mypy (no ExtensionArray/Dtype)
    arr = np.asarray(y)

    if arr.ndim != 1:
        return False

    uniq = np.unique(arr)
    return uniq.size <= 15 and (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)
    )


def _split_train_test(
    X: Any, y: Any, *, test_size: float = 0.2, random_state: int | None = 42
) -> tuple[Any, Any, Any, Any]:
    """Split data into train and test sets with automatic stratification.

    Args:
        X (Any): Feature matrix.
        y (Any): Target vector.
        test_size (float): Test set fraction. Defaults to 0.2.
        random_state (int | None): Random number generator seed. Defaults to 42.
    Returns:
        tuple[Any, Any, Any, Any]: Tuple of (X_train, X_test, y_train, y_test).
    """
    strat = y if _can_stratify(y) else None
    try:
        return cast(
            tuple[Any, Any, Any, Any],
            train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=True,
                random_state=random_state,
                stratify=strat,
            ),
        )
    except ValueError:
        return cast(
            tuple[Any, Any, Any, Any],
            train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_state
            ),
        )


def _warn_failed_fold(fold_idx: int, err: Exception, stage: str) -> None:
    """Log the penalty of a failed fold at the given evaluation stage.

    Shared by the per-fold evaluation loops (``_evaluate_with_intermediate_
    reports`` and ``_evaluate_pooled_cv``) so the warning text stays uniform:
    a fold whose fit/predict/scoring raised a non-fatal exception makes the
    trial's estimate NaN — the ``cross_val_score(error_score=np.nan)``
    contract (a failed fold became NaN, i.e. the worst possible outcome), so
    a model that crashes on most folds cannot win over an honest one.

    Args:
        fold_idx (int): 1-based fold number.
        err (Exception): The raised exception (the message is logged with its
            type name).
        stage (str): One of 'fit or predict' / 'score' — the operation that
            failed for this fold.
    """
    log.warning(
        "Fold %d failed to %s: %s: %s — the fold is penalized: the trial "
        "estimate becomes NaN (the worst outcome)",
        fold_idx,
        stage,
        type(err).__name__,
        err,
    )


def _evaluate_with_intermediate_reports(
    trial: Trial,
    estimator: Any,
    X: Any,
    y: Any,
    *,
    method: str,
    n_folds: int,
    test_size: float,
    random_state: int | None,
    scorer: Any,
    pool_metric: bool = False,
    algo_name: str = "",
    phase_name: str = "",
    oof_out: np.ndarray | None = None,
) -> float:
    """Evaluate the estimator fold by fold, publishing intermediate results.

    Serves as the early stopping enabler: after each fold the running estimate
    of the metric is published via ``trial.report``, followed by a call to
    ``trial.should_prune()``. A hopeless trial is interrupted by an
    ``optuna.TrialPruned`` exception before the full evaluation completes.

    For non-additive metrics (``pool_metric=True``: RMSE, MSE, R², nRMSE,
    RMSLE, D², explained_variance, max error, median absolute error) the
    published intermediate value is the metric
    computed over the predictions accumulated up to the current step (the
    pooled value over the covered rows), not the cumulative mean of per-fold
    values — each row enters the calculation exactly once, in the same way as
    the final pooled estimate. Non-finite predictions of individual folds are
    masked out of the pooled value (the same contract as
    ``metrics.oof_rmse`` and the final ``ModelTrainer`` OOF channel): a model
    that merely predicts NaN on some rows is NOT penalized — only its
    non-finite rows are excluded from the estimate. A fold whose fit/predict
    raises a NON-FATAL exception (e.g. ``TypeError``, ``KeyError``,
    ``IndexError``, ``ValueError``/``numpy.linalg.LinAlgError``) is PENALIZED:
    its rows are
    marked with NaN (they cannot enter the pooled estimate) and a warning is
    logged — the semantics of the legacy ``cross_val_score(error_score=np.nan)``
    (a failed fold became NaN, i.e. the worst possible outcome; it must not
    abort the trial, and a model that crashes on most folds must not win over
    an honest one). Fatal exceptions (``MemoryError``,
    ``RuntimeError``, ``InvalidDataError``) are NOT swallowed — they propagate
    to ``handle_failure`` so the circuit breaker stays active. If EVERY fold
    fails, a ``ValueError`` is raised — the same
    "no valid result" contract as before. A running pooled value that is
    itself non-finite
    (e.g. R² on a single LOO pair, which is undefined) is NOT published to
    ``trial.report``: Optuna pruners cannot handle NaN and would emit
    RuntimeWarnings and compare incorrectly. The step is skipped until the
    accumulated rows admit a finite pooled estimate. Additive metrics (MAE and
    similar per-sample means, ``pool_metric=False``) keep publishing the
    cumulative fold mean (which equals the pooled value); a failed fold is
    penalized with NaN in the cumulative mean, so the trial's FINAL estimate
    is NaN — the worst outcome.

    When the pooled value is impossible FINALLY (the scorer exposes no raw
    metric function, or no finite pair remains by the end of the evaluation),
    a WARNING about the fallback to the per-fold mean is logged once per
    (``algo_name``, ``phase_name``) pair. A TRANSIENT pooled failure inside
    the fold loop (e.g. the first fold predicted all-NaN) is not a metric
    degradation — the final pooled value may still be valid — so no WARNING
    is emitted there.

    Args:
        trial (Trial): The current Optuna trial.
        estimator (Any): The estimator (possibly a pipeline) to evaluate.
        X (Any): Feature matrix.
        y (Any): Target variable vector.
        method (str): Split strategy ('k_fold' or 'loo').
        n_folds (int): Number of folds for k_fold.
        test_size (float): Test fraction (unused for k_fold/loo; passed to
            keep a uniform iter_splits call).
        random_state (int | None): Randomness seed.
        scorer (Any): sklearn metric object.
        pool_metric (bool): Whether the metric is non-additive and must be
            pooled over the accumulated OOF predictions. Defaults to False —
            the legacy cumulative-fold-mean behavior.
        algo_name (str): The algorithm being evaluated, for the fallback
            WARNING deduplication.
        phase_name (str): The pipeline phase, for the fallback WARNING
            deduplication.
        oof_out (np.ndarray | None): An optional buffer of length
            ``len(X)``; when given, the out-of-fold predictions of the
            successful folds are written into it by position (the rows of
            failed folds stay untouched).

    Returns:
        float: The mean metric value across all folds (for additive metrics)
            or the pooled value over all rows (for non-additive metrics).
            NaN whenever at least one fold failed during evaluation — the
            caller turns it into ``HPO_WORST_SCORE`` (the worst outcome), so
            a failing model cannot win over an honest one.
    Raises:
        optuna.TrialPruned: If the pruner decides to prune the trial.
        ValueError: If every fold fails during evaluation (fit/predict) — the
            estimate is impossible.
        MemoryError, RuntimeError, InvalidDataError: A fatal per-fold failure —
            propagated so the circuit breaker stays active.
        Exception: A genuine failure outside a single fold (e.g. split
            iteration) propagates (fatal or non-fatal depending on the
            exception type, as before).
    """
    fold_scores: list[float] = []
    # Non-additive metrics (RMSE, MSE, R², nRMSE, ...) are pooled over the
    # rows covered by the folds processed so far; the running pooled value is
    # published as the intermediate result. The OOF rows are accumulated in
    # pre-allocated full-size buffers (filled by positional indices) and
    # pooled via ``pooled_scorer_value``, whose non-finite masking drops the
    # not-yet-covered rows — no repeated array concatenation (O(k·n) copies
    # on the old code, O(n²) for LOO).
    n_rows = len(X)
    pooled_y_buf = np.full(n_rows, np.nan, dtype=float)
    pooled_pred_buf = np.full(n_rows, np.nan, dtype=float)
    n_folds_total = 0
    n_failed_folds = 0
    for fold_idx, (X_tr, X_te, y_tr, y_te, test_idx) in enumerate(
        iter_splits(
            X,
            y,
            method=method,
            n_folds=n_folds,
            test_size=test_size,
            random_state=random_state,
            include_indices=True,
        ),
        start=1,
    ):
        n_folds_total += 1
        fitted = _fit_predict_fold(estimator, X_tr, y_tr, X_te, fold_idx)
        if fitted is None:
            n_failed_folds += 1
            continue
        fold_estimator, fold_pred = fitted
        if oof_out is not None:
            oof_out[np.asarray(test_idx)] = fold_pred
        fold_true = np.asarray(y_te, dtype=float).reshape(-1)
        if pool_metric:
            idx = np.asarray(test_idx)
            pooled_y_buf[idx] = fold_true
            pooled_pred_buf[idx] = fold_pred
            running_value = _running_pooled_value(
                scorer,
                fold_estimator,
                X_te,
                y_te,
                pooled_y_buf,
                pooled_pred_buf,
                fold_scores,
            )
            # Optuna pruners cannot compare NaN intermediate values (they
            # produce RuntimeWarnings and incorrect decisions), so a
            # non-finite pooled value (e.g. LOO+R² on the first single-row
            # step) is not published — the step is skipped until the
            # accumulated rows admit a finite estimate.
            if np.isfinite(running_value):
                trial.report(running_value, step=fold_idx)
        else:
            try:
                fold_scores.append(float(scorer(fold_estimator, X_te, y_te)))
            except (MemoryError, RuntimeError, InvalidDataError):
                raise
            except Exception as err:  # noqa: BLE001 — per-fold scoring, same cross_val_score(error_score=nan) semantics
                n_failed_folds += 1
                _warn_failed_fold(fold_idx, err, "score")
                continue
            running_value = float(np.mean(fold_scores))
            trial.report(running_value, step=fold_idx)
        if trial.should_prune():
            log.debug(
                "Trial %d pruned by early stopping at step %d",
                trial.number,
                fold_idx,
            )
            raise optuna.TrialPruned()
    return _final_intermediate_estimate(
        scorer,
        fold_scores,
        pooled_y_buf,
        pooled_pred_buf,
        pool_metric=pool_metric,
        n_folds_total=n_folds_total,
        n_failed_folds=n_failed_folds,
        algo_name=algo_name,
        phase_name=phase_name,
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
    trial, but a model that crashes on most folds must not win over an honest
    one). Fatal exception types (MemoryError / RuntimeError /
    InvalidDataError) are NOT swallowed here — they propagate to
    ``handle_failure`` and keep the circuit breaker active.

    Returns:
        The fitted fold estimator and its predictions, or ``None`` when the
        fold failed.
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
    return fold_estimator, fold_pred


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

    Raises:
        ValueError: If every fold failed.
    """
    if n_folds_total > 0 and n_failed_folds == n_folds_total:
        raise ValueError(
            f"All {n_folds_total} fold(s) failed during evaluation — the "
            "estimate is impossible"
        )
    # A failed fold is a guaranteed worst outcome: its rows could not enter
    # the pooled estimate (pooled path) or its score is NaN in the additive
    # fold mean — the trial estimate becomes NaN, exactly like the legacy
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


def _evaluate_pooled_cv(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    method: str,
    n_folds: int,
    test_size: float,
    random_state: int | None,
    scorer: Any,
    algo_name: str = "",
    phase_name: str = "",
    oof_out: np.ndarray | None = None,
) -> float:
    """Evaluate a trial pipeline over pooled OOF predictions (non-additive).

    Iterates the folds (the same splits as ``cross_val_score(cv=cv_obj)`` via
    ``iter_splits``), collects the out-of-fold prediction/target vectors, and
    computes the metric over the whole vector at once — the same estimate the
    final ``ModelTrainer`` and the select phase use. This is the required form
    for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D²,
    explained_variance, max error, median absolute error): the pooled value
    differs from the average of per-fold values.

    Non-finite predictions of individual folds are masked out of the pooled
    value (the same contract as ``metrics.oof_rmse`` and the final
    ``ModelTrainer`` OOF channel): a model that merely predicts NaN on some
    rows is NOT penalized — only its non-finite rows are excluded from the
    estimate (e.g. isotonic regression on a degenerate split does not abort
    the trial). A fold whose fit/predict (or per-fold scoring, on the
    non-poolable path)
    raises a NON-FATAL exception (e.g. ``TypeError``, ``KeyError``,
    ``IndexError``, ``ValueError``/``numpy.linalg.LinAlgError``) is PENALIZED:
    the trial's estimate becomes NaN — the semantics of the legacy
    ``cross_val_score(error_score=np.nan)`` (a failed fold became NaN, i.e.
    the worst possible outcome), so a model that crashes on most folds cannot
    win over an honest one. Fatal exceptions (``MemoryError``,
    ``RuntimeError``, ``InvalidDataError``) are NOT swallowed — they propagate
    to ``handle_failure`` so the circuit breaker stays active. If EVERY fold
    fails at the fit/predict stage, the evaluation is impossible and a
    ``ValueError`` is raised — the same "no valid result" contract as before
    (mirroring ``cross_val_score``, which raises ``ValueError`` when every
    fit fails, and returns an all-NaN array when the fits succeed but every
    score is impossible).

    The per-fold scorer value is computed lazily — only when the pooled value
    cannot be computed — so per-fold scoring noise (e.g. LOO+R², where every
    fold R² is undefined on a single row) does not pollute the logs on the
    common path.

    If the pooled value cannot be computed (no valid pairs, or the scorer does
    not expose a raw metric function), the mean of the available per-fold
    values is returned as a defensive fallback, and a WARNING about the
    fallback is logged once per (``algo_name``, ``phase_name``) pair.

    Args:
        estimator (Any): The estimator (possibly a pipeline) to evaluate.
        X (Any): Feature matrix.
        y (Any): Target variable vector.
        method (str): Split strategy ('k_fold' or 'loo').
        n_folds (int): Number of folds for k_fold.
        test_size (float): Test fraction (unused for k_fold/loo; passed to
            keep a uniform iter_splits call).
        random_state (int | None): Randomness seed.
        scorer (Any): sklearn metric object.
        algo_name (str): The algorithm being evaluated, for the fallback
            WARNING deduplication.
        phase_name (str): The pipeline phase, for the fallback WARNING
            deduplication.
        oof_out (np.ndarray | None): An optional buffer of length
            ``len(X)``; when given, the out-of-fold predictions of the
            successful folds are written into it by position.

    Returns:
        float: The pooled metric value over all out-of-fold rows. NaN
            whenever at least one fold failed during evaluation — the caller
            turns it into ``HPO_WORST_SCORE`` (the worst outcome), so a
            failing model cannot win over an honest one.

    Raises:
        ValueError: If every fold fails during evaluation (fit/predict or the
            per-fold scoring fallback) — the pooled estimate is impossible.
        MemoryError, RuntimeError, InvalidDataError: A fatal per-fold failure —
            propagated so the circuit breaker stays active.
    """
    fold_scores: list[float] = []
    n_rows = len(X)
    # Whether the scorer exposes a raw metric function: if it does not, the
    # pooled value is impossible and the per-fold path is taken inside the
    # SAME loop (no double iteration of the splits).
    poolable = getattr(scorer, "_score_func", None) is not None
    # OOF rows are accumulated in pre-allocated full-size buffers (filled by
    # positional indices); the pooled value is computed over the buffers with
    # non-finite masking — no repeated array concatenation.
    pooled_y = np.full(n_rows, np.nan, dtype=float)
    pooled_pred = np.full(n_rows, np.nan, dtype=float)
    # Per-fold (true, pred) pairs are retained so that the per-fold fallback
    # (used when the pooled value is impossible) can be scored WITHOUT
    # re-iterating the splits or re-training the folds.
    fold_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    n_folds_total = 0
    n_failed_folds = 0
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X,
        y,
        method=method,
        n_folds=n_folds,
        test_size=test_size,
        random_state=random_state,
        include_indices=True,
    ):
        n_folds_total += 1
        # Clone the estimator for every fold (as cross_val_score does) to
        # avoid state accumulating between folds. A fold whose fit/predict
        # raises a NON-FATAL exception is PENALIZED: the trial's estimate
        # becomes NaN, and a warning is logged (the
        # ``cross_val_score(error_score=np.nan)`` contract: a failed fold
        # became NaN — the worst possible outcome — and must not abort the
        # trial, but a model that crashes on part of the folds must not win
        # over an honest one). Fatal exception types (MemoryError
        # / RuntimeError / InvalidDataError) are NOT swallowed here — they
        # propagate to ``handle_failure`` and keep the circuit breaker active.
        try:
            fold_estimator = clone(estimator)
            fold_estimator.fit(X_tr, y_tr)
            fold_pred = np.asarray(fold_estimator.predict(X_te)).reshape(-1)
        except (MemoryError, RuntimeError, InvalidDataError):
            raise
        except Exception as err:  # noqa: BLE001 — per-fold failure, cross_val_score(error_score=nan) semantics
            n_failed_folds += 1
            _warn_failed_fold(n_folds_total, err, "fit or predict")
            continue
        if oof_out is not None:
            oof_out[np.asarray(test_idx)] = fold_pred
        fold_true = np.asarray(y_te, dtype=float).reshape(-1)
        if not poolable:
            # The scorer does not expose a raw metric function: pooling is
            # impossible, collect the per-fold values as the fallback. A
            # non-fatal scoring failure is penalized the same way as a fit
            # failure.
            try:
                fold_scores.append(float(scorer(fold_estimator, X_te, y_te)))
            except (MemoryError, RuntimeError, InvalidDataError):
                raise
            except Exception as err:  # noqa: BLE001 — per-fold scoring, same cross_val_score(error_score=nan) semantics
                n_failed_folds += 1
                _warn_failed_fold(n_folds_total, err, "score")
            continue
        idx = np.asarray(test_idx)
        pooled_y[idx] = fold_true
        pooled_pred[idx] = fold_pred
        fold_pairs.append((fold_true, fold_pred))
    if n_folds_total > 0 and n_failed_folds == n_folds_total:
        raise ValueError(
            f"All {n_folds_total} fold(s) failed during evaluation — the "
            "pooled estimate is impossible"
        )
    # A failed fold is a guaranteed worst outcome — the semantics of the
    # legacy ``cross_val_score(error_score=np.nan)``, whose NaN fold poisoned
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
        "(_score_func/_sign/_kwargs)",
    )
    return float(np.mean(fold_scores)) if fold_scores else float("nan")


def _phase_oof_matrix(
    phase_trials: list[FrozenTrial], trial_oof: dict[int, np.ndarray]
) -> np.ndarray | None:
    """Stack the kept OOF vectors of the phase's own COMPLETED trials.

    Args:
        phase_trials: The trials of the current phase.
        trial_oof: The kept OOF vectors by trial number.

    Returns:
        np.ndarray | None: A ``(n_trials, n_rows)`` matrix in trial
            order, or ``None`` when no completed trial kept its
            predictions (OOF collection off included).
    """
    rows = [
        trial_oof[t.number]
        for t in phase_trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.number in trial_oof
    ]
    if not rows:
        return None
    return np.vstack(rows)


def score_trial(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    trial: Trial,
    validation: ValidationPlan,
    scorer: Any,
    pruning_active: bool,
    pool_metric: bool,
    random_state: int | None,
    algo_name: str = "",
    phase_name: str = "",
    collect_oof: bool = False,
) -> TrialScore:
    """Score the trial pipeline with the selected validation strategy.

    Evaluates ``estimator`` with the validation strategy of the plan; the
    estimator itself is not fitted (every split fits a clone). A non-finite
    result (+inf on an nrmse error or NaN) is replaced with
    ``HPO_WORST_SCORE`` — the "worst outcome" sentinel (the
    ``train_test_split`` hold-out value is returned as is). Hyperparameter
    generation and pipeline assembly happen in ``_objective`` BEFORE this
    function is invoked and outside try/except: errors from a user space
    function or the model constructor must propagate (trial FAIL) rather
    than turn into pruning/disqualification. With ``pruning_active`` the
    intermediate values are reported to ``trial`` for the Optuna pruner.

    Args:
        estimator (Any): The pipeline/model of the current trial.
        X (Any): The feature matrix.
        y (Any): The target vector.
        trial (Trial): The current Optuna trial object.
        validation (ValidationPlan): The resolved validation strategy.
        scorer (Any): The sklearn scorer of the target metric.
        pruning_active (bool): Whether the pruner is applied (fold-wise
            evaluation with intermediate reports).
        pool_metric (bool): Whether the metric is non-additive (pooled over
            the out-of-fold vector).
        random_state (int | None): The seed of the data splits.
        algo_name (str): The algorithm name (fallback WARNING deduplication).
        phase_name (str): The phase name (fallback WARNING deduplication).
        collect_oof (bool): Whether the out-of-fold predictions are
            returned.
    Returns:
        TrialScore: The target metric value and, with ``collect_oof``, the
            out-of-fold predictions by row position.
    Raises:
        ValueError: Non-fatal evaluation error (the trial is pruned).
        MemoryError, RuntimeError, InvalidDataError: Fatal failures —
            counted in the ``handle_failure`` counter.
        optuna.TrialPruned: Pruning by the early stopping pruner.
    """
    oof = np.full(len(X), np.nan, dtype=float) if collect_oof else None
    if validation.method == "train_test_split":
        # The hold-out value is returned as is (a non-finite one fails the
        # trial in Optuna).
        return TrialScore(
            value=_score_holdout(
                estimator, X, y, validation, scorer, random_state, oof
            ),
            oof=oof,
        )
    if pruning_active:
        # Early stopping: evaluate fold by fold and publish intermediate
        # results for the Optuna pruner. For non-additive metrics the
        # published value is the pooled metric over the predictions
        # accumulated up to the current step.
        value = _evaluate_with_intermediate_reports(
            trial,
            estimator,
            X,
            y,
            method=validation.method,
            n_folds=validation.n_folds,
            test_size=validation.test_size,
            random_state=random_state,
            scorer=scorer,
            pool_metric=pool_metric,
            algo_name=algo_name,
            phase_name=phase_name,
            oof_out=oof,
        )
    elif pool_metric:
        # Non-additive metrics (RMSE, MSE, R², nRMSE): evaluate over the
        # pooled OOF vector at once (the same estimate the final
        # ModelTrainer and the select phase use) instead of averaging
        # per-fold values. The splits are the same as
        # ``cross_val_score(cv=cv_obj)``.
        value = _evaluate_pooled_cv(
            estimator,
            X,
            y,
            method=validation.method,
            n_folds=validation.n_folds,
            test_size=validation.test_size,
            random_state=random_state,
            scorer=scorer,
            algo_name=algo_name,
            phase_name=phase_name,
            oof_out=oof,
        )
    else:
        value = _score_fold_mean(estimator, X, y, validation, scorer, oof)
    # If we got +inf (an nrmse error) or NaN, return the worst float
    # (analog of the float32 minimum or float('-inf')).
    if not np.isfinite(value):
        return TrialScore(value=HPO_WORST_SCORE, oof=oof)
    return TrialScore(value=value, oof=oof)


def _score_holdout(
    estimator: Any,
    X: Any,
    y: Any,
    validation: ValidationPlan,
    scorer: Any,
    random_state: int | None,
    oof_out: np.ndarray | None,
) -> float:
    """Score a clone of the estimator on the ``train_test_split`` hold-out.

    If 'auto' was resolved in make_cv, the already computed
    (dataset-dependent) integer test size is used so that ``iter_splits``
    does not recompute the decision; otherwise — the fixed fraction.

    Returns:
        float: The raw metric value on the hold-out rows.
    """
    model = clone(estimator)
    if oof_out is None:
        X_tr, X_te, y_tr, y_te = next(
            iter_splits(
                X,
                y,
                method="train_test_split",
                test_size=validation.split_test_size,
                random_state=random_state,
            )
        )
        model.fit(X_tr, y_tr)
        return float(scorer(model, X_te, y_te))
    # OOF collection: the same split, with the test row positions.
    X_tr, X_te, y_tr, y_te, test_idx = next(
        iter_splits(
            X,
            y,
            method="train_test_split",
            test_size=validation.split_test_size,
            random_state=random_state,
            include_indices=True,
        )
    )
    model.fit(X_tr, y_tr)
    oof_out[np.asarray(test_idx)] = np.asarray(
        model.predict(X_te), dtype=float
    ).reshape(-1)
    return float(scorer(model, X_te, y_te))


def _score_fold_mean(
    estimator: Any,
    X: Any,
    y: Any,
    validation: ValidationPlan,
    scorer: Any,
    oof_out: np.ndarray | None,
) -> float:
    """The fold-average score of additive metrics (``cross_val_score``).

    ``validation.cv`` is guaranteed non-None here: a non-train_test_split and
    non-pruning branch means k_fold/loo, where make_cv always creates a cv
    object.

    Returns:
        float: The mean of the per-fold scores.
    """
    if oof_out is None:
        scores = model_selection.cross_val_score(
            estimator,
            X,
            y,
            cv=validation.cv,
            scoring=scorer,
            n_jobs=1,
        )
        return float(np.mean(scores))
    # Additive metrics with OOF collection: the same fold scores as
    # ``cross_val_score`` (same cv object and scorer), with the fold
    # estimators kept to predict their test rows.
    cv_res = model_selection.cross_validate(
        estimator,
        X,
        y,
        cv=validation.cv,
        scoring=scorer,
        n_jobs=1,
        return_estimator=True,
        return_indices=True,
    )
    avg_score = float(np.mean(cv_res["test_score"]))
    if not np.isfinite(avg_score):
        return avg_score
    for fold_est, test_idx in zip(
        cv_res["estimator"], cv_res["indices"]["test"], strict=True
    ):
        oof_out[test_idx] = np.asarray(
            fold_est.predict(_safe_indexing(X, test_idx)), dtype=float
        ).reshape(-1)
    return avg_score
