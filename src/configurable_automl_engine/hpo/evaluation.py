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

from configurable_automl_engine.common.bbc_cv import BBC_MAX_OOF_CELLS
from configurable_automl_engine.common.errors import InvalidPredictionError
from configurable_automl_engine.common.metrics import get_scorer_object
from configurable_automl_engine.common.sklearn_compat import scorer_raw_parts
from configurable_automl_engine.validation import iter_splits

from .fold_eval import (
    _final_intermediate_estimate,
    _final_pooled_estimate,
    _fit_predict_fold,
    _running_pooled_value,
    _warn_failed_fold,
    _warn_invalid_fold_predictions,
)
from .plans import TrialScore, ValidationPlan
from .repeats import (
    phase_oof_matrix,
    repeat_cv,
    run_repeat_scores,
    successful_repeat_oof,
)
from .types import (
    HPO_WORST_SCORE,
    HyperoptError,
    InvalidDataError,
)

log = _logging.getLogger(__name__)


def _build_scorer(name: str) -> Any:
    """Create a scorer object from a metric name."""
    try:
        # Use the new API, which returns either a make_scorer object or a string
        return get_scorer_object(name)
    except Exception as err:
        raise HyperoptError(f"Unknown metric name: '{name}'") from err


def _can_stratify(y: Any) -> bool:
    """Determine whether stratification is possible for the y vector."""
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
        X: Feature matrix.
        y: Target vector.
        test_size: Test set fraction. Defaults to 0.2.
        random_state: Random number generator seed. Defaults to 42."""
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
    n_repeats: int = 1,
    split_aggregation: str = "median",
    step_offset: int = 0,
    validation: ValidationPlan | None = None,
    scores_out: list[float] | None = None,
    mask_out: list[bool] | None = None,
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
    the final pooled estimate. A fold whose fit/predict
    raises a NON-FATAL exception (e.g. ``TypeError``, ``KeyError``,
    ``IndexError``, ``ValueError``/``numpy.linalg.LinAlgError``) is PENALIZED:
    its rows are
    marked with NaN (they cannot enter the pooled estimate) and a warning is
    logged — a failed fold becomes NaN, i.e. the worst possible outcome; it
    must not abort the trial, and a model that crashes on most folds must not
    win over an honest one. A fold that COMPLETES but returns non-finite
    (NaN/Inf) predictions is penalized the same way (see
    ``_fit_predict_fold``): the model was required to predict these rows, and
    masking them out of the pooled value would reward a model that fails on
    most of the validation rows. Fatal exceptions (``MemoryError``,
    ``RuntimeError``, ``InvalidDataError``) are NOT swallowed — they propagate
    to ``handle_failure`` so the circuit breaker stays active. If EVERY fold
    fails, a ``ValueError`` is raised — the "no valid result" contract. A
    running pooled value that is
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
        trial: The current Optuna trial.
        estimator: The estimator (possibly a pipeline) to evaluate.
        X: Feature matrix.
        y: Target variable vector.
        method: Split strategy ('k_fold' or 'loo').
        n_folds: Number of folds for k_fold.
        test_size: Test fraction (unused for k_fold/loo; passed to
            keep a uniform iter_splits call).
        random_state: Randomness seed.
        scorer: sklearn metric object.
        pool_metric: Whether the metric is non-additive and must be
            pooled over the accumulated OOF predictions. Defaults to False —
            the cumulative-fold-mean behavior.
        algo_name: The algorithm being evaluated, for the fallback
            WARNING deduplication.
        phase_name: The pipeline phase, for the fallback WARNING
            deduplication.
        oof_out: An optional buffer of length
            ``len(X)``; when given, the out-of-fold predictions of the
            successful folds are written into it by position (the rows of
            failed folds stay untouched).
        n_repeats: The number of repeats (``1`` — the classic single split).
        split_aggregation: How the per-repeat scores are combined.
        step_offset: The global Optuna step offset of the repeat (the
            intermediate reports of every repeat keep distinct steps).
        validation: The resolved validation plan (the repeat seeds).
        scores_out: An optional collector of the per-repeat scores.
        mask_out: An optional collector of the per-repeat success flags.


    Returns:
        float: The mean metric value across all folds (for additive metrics)
            or the pooled value over all rows (for non-additive metrics).
            NaN whenever at least one fold failed during evaluation — the
            caller turns it into ``HPO_WORST_SCORE`` (the worst outcome), so
            a failing model cannot win over an honest one."""
    if n_repeats > 1:
        # Every repeat is evaluated fold by fold with its own deterministic
        # split seed; its intermediate reports are shifted by the global step
        # offset, so the pruner keeps one comparable intermediate step per
        # (repeat, fold) while the final trial value is the aggregate over the
        # repeats (median by default).
        return run_repeat_scores(
            lambda repeat_index, seed, sub_oof: _evaluate_with_intermediate_reports(
                trial,
                estimator,
                X,
                y,
                method=method,
                n_folds=n_folds,
                test_size=test_size,
                random_state=seed,
                scorer=scorer,
                pool_metric=pool_metric,
                algo_name=algo_name,
                phase_name=phase_name,
                oof_out=sub_oof,
                step_offset=step_offset + repeat_index * n_folds,
            ),
            validation=validation,
            n_repeats=n_repeats,
            random_state=random_state,
            oof_out=oof_out,
            split_aggregation=split_aggregation,
            all_failed_message=(
                f"All {n_repeats} repeat(s) failed during evaluation — the "
                "estimate is impossible"
            ),
            scores_out=scores_out,
            mask_out=mask_out,
        )
    fold_scores: list[float] = []
    # Non-additive metrics (RMSE, MSE, R², nRMSE, ...) are pooled over the
    # rows covered by the folds processed so far; the running pooled value is
    # published as the intermediate result. The OOF rows are accumulated in
    # pre-allocated full-size buffers (filled by positional indices) and
    # pooled via ``pooled_scorer_value``, whose non-finite masking drops the
    # not-yet-covered rows — no repeated array concatenation (O(n²) copies
    # with the naive approach, O(k·n) here).
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
                trial.report(running_value, step=step_offset + fold_idx)
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
            trial.report(running_value, step=step_offset + fold_idx)
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
    n_repeats: int = 1,
    split_aggregation: str = "median",
    validation: ValidationPlan | None = None,
    scores_out: list[float] | None = None,
    mask_out: list[bool] | None = None,
) -> float:
    """Evaluate a trial pipeline over pooled OOF predictions (non-additive).

    Iterates the folds (the same splits as ``cross_val_score(cv=cv_obj)`` via
    ``iter_splits``), collects the out-of-fold prediction/target vectors, and
    computes the metric over the whole vector at once — the same estimate the
    final ``ModelTrainer`` and the select phase use. This is the required form
    for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D²,
    explained_variance, max error, median absolute error): the pooled value
    differs from the average of per-fold values.

    The per-fold predictions must be finite: the fold's test rows are rows
    the model was REQUIRED to predict, so a completed fold with non-finite
    (NaN/Inf) predictions is a quality defect of the model, not a coverage
    gap — it is PENALIZED exactly like a fold that raised (its rows stay NaN
    in the pooled buffers, the trial's estimate becomes NaN; the values are
    never replaced with zeros). A fold whose fit/predict (or per-fold scoring,
    on the
    non-poolable path)
    raises a NON-FATAL exception (e.g. ``TypeError``, ``KeyError``,
    ``IndexError``, ``ValueError``/``numpy.linalg.LinAlgError``) is PENALIZED:
    the trial's estimate becomes NaN — mirroring
    ``cross_val_score(error_score=np.nan)`` (a failed fold became NaN, i.e.
    the worst possible outcome), so a model that crashes on most folds cannot
    win over an honest one. Fatal exceptions (``MemoryError``,
    ``RuntimeError``, ``InvalidDataError``) are NOT swallowed — they propagate
    to ``handle_failure`` so the circuit breaker stays active. If EVERY fold
    fails at the fit/predict stage, the evaluation is impossible and a
    ``ValueError`` is raised — the "no valid result" contract
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
        estimator: The estimator (possibly a pipeline) to evaluate.
        X: Feature matrix.
        y: Target variable vector.
        method: Split strategy ('k_fold' or 'loo').
        n_folds: Number of folds for k_fold.
        test_size: Test fraction (unused for k_fold/loo; passed to
            keep a uniform iter_splits call).
        random_state: Randomness seed.
        scorer: sklearn metric object.
        algo_name: The algorithm being evaluated, for the fallback
            WARNING deduplication.
        phase_name: The pipeline phase, for the fallback WARNING
            deduplication.
        oof_out: An optional buffer of length
            ``len(X)``; when given, the out-of-fold predictions of the
            successful folds are written into it by position.
        n_repeats: The number of repeats (``1`` — the classic single split).
        split_aggregation: How the per-repeat scores are combined.
        validation: The resolved validation plan (the repeat seeds).
        scores_out: An optional collector of the per-repeat scores.
        mask_out: An optional collector of the per-repeat success flags.


    Returns:
        float: The pooled metric value over all out-of-fold rows. NaN
            whenever at least one fold failed during evaluation — the caller
            turns it into ``HPO_WORST_SCORE`` (the worst outcome), so a
            failing model cannot win over an honest one.
    """
    if n_repeats > 1:
        return run_repeat_scores(
            lambda _r, seed, sub_oof: _evaluate_pooled_cv(
                estimator,
                X,
                y,
                method=method,
                n_folds=n_folds,
                test_size=test_size,
                random_state=seed,
                scorer=scorer,
                algo_name=algo_name,
                phase_name=phase_name,
                oof_out=sub_oof,
            ),
            validation=validation,
            n_repeats=n_repeats,
            random_state=random_state,
            oof_out=oof_out,
            split_aggregation=split_aggregation,
            all_failed_message=(
                f"All {n_repeats} repeat(s) failed during evaluation — the "
                "pooled estimate is impossible"
            ),
            scores_out=scores_out,
            mask_out=mask_out,
        )
    fold_scores: list[float] = []
    n_rows = len(X)
    # Whether the scorer exposes a raw metric function: if it does not, the
    # pooled value is impossible and the per-fold path is taken inside the
    # SAME loop (no double iteration of the splits).
    poolable = scorer_raw_parts(scorer)[0] is not None
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
        # The fold's test rows are rows the model was REQUIRED to predict:
        # a completed fold with non-finite (NaN/Inf) predictions is a quality
        # defect of the model (not a coverage gap), so it is penalized exactly
        # like a fold that raised — its rows stay NaN in the pooled buffers
        # and the trial estimate becomes NaN (never replaced with zeros).
        if not np.isfinite(fold_pred.astype(float, copy=False)).all():
            n_failed_folds += 1
            _warn_invalid_fold_predictions(
                n_folds_total,
                int(np.isfinite(fold_pred.astype(float, copy=False)).sum()),
                len(fold_pred),
            )
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
    return _final_pooled_estimate(
        scorer,
        fold_scores,
        fold_pairs,
        pooled_y,
        pooled_pred,
        poolable=poolable,
        n_folds_total=n_folds_total,
        n_failed_folds=n_failed_folds,
        algo_name=algo_name,
        phase_name=phase_name,
    )


def _phase_oof_matrix(
    phase_trials: list[FrozenTrial], trial_oof: dict[int, np.ndarray]
) -> np.ndarray | None:
    """Stack the kept OOF vectors of the phase's own COMPLETED trials.

    A multi-split trial stores one OOF vector per repeat — the repeats are
    stacked along the configuration axis (every repeat is a separate BBC-CV
    configuration). The total size is bounded by ``BBC_MAX_OOF_CELLS``: over
    the budget the matrix is not built and ``None`` is returned.

    Returns:
        np.ndarray | None: A ``(n_configurations, n_rows)`` matrix in trial
            (then repeat) order, or ``None`` when no completed trial kept its
            predictions (OOF collection off included) or the budget is
            exceeded.
    """
    return phase_oof_matrix(phase_trials, trial_oof, max_cells=BBC_MAX_OOF_CELLS)


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
    n_repeats: int | None = None,
    split_aggregation: str | None = None,
) -> TrialScore:
    """Score the trial pipeline with the selected validation strategy.

    Evaluates ``estimator`` with the validation strategy of the plan; the
    estimator itself is not fitted (every split fits a clone). A non-finite
    result (+inf on an nrmse error or NaN) is replaced with
    ``HPO_WORST_SCORE`` — the "worst outcome" sentinel (the
    ``train_test_split`` hold-out value is returned as is). A model that
    returns non-finite (NaN/Inf) predictions on rows it was REQUIRED to
    predict (the hold-out rows, or a CV fold's test rows) does not get a
    masked estimate: a fold with such predictions is penalized to NaN (the
    worst outcome) and the hold-out path raises ``InvalidPredictionError``
    (a non-fatal ``ValueError`` — the trial is pruned). Hyperparameter
    generation and pipeline assembly happen in ``_objective`` BEFORE this
    function is invoked and outside try/except: errors from a user space
    function or the model constructor must propagate (trial FAIL) rather
    than turn into pruning/disqualification. With ``pruning_active`` the
    intermediate values are reported to ``trial`` for the Optuna pruner.

    Args:
        estimator: The pipeline/model of the current trial.
        X: The feature matrix.
        y: The target vector.
        trial: The current Optuna trial object.
        validation: The resolved validation strategy.
        scorer: The sklearn scorer of the target metric.
        pruning_active: Whether the pruner is applied (fold-wise
            evaluation with intermediate reports).
        pool_metric: Whether the metric is non-additive (pooled over
            the out-of-fold vector).
        random_state: The seed of the data splits.
        algo_name: The algorithm name (fallback WARNING deduplication).
        phase_name: The phase name (fallback WARNING deduplication).
        collect_oof: Whether the out-of-fold predictions are
            returned.
        n_repeats: The number of repeats (``None`` — the plan's
            ``n_repeats``; the plan is the canonical source).
        split_aggregation: The per-repeat aggregation policy (``None`` — the
            plan's ``split_aggregation``).
    Returns:
        TrialScore: The target metric value and, with ``collect_oof``, the
            out-of-fold predictions by row position.
    Raises:
        ValueError: Non-fatal evaluation error (the trial is pruned).
        MemoryError, RuntimeError, InvalidDataError: Fatal failures —
            counted in the ``handle_failure`` counter.
        optuna.TrialPruned: Pruning by the early stopping pruner.
    """
    n_repeats = validation.n_repeats if n_repeats is None else n_repeats
    aggregation = (
        validation.split_aggregation if split_aggregation is None else split_aggregation
    )
    repeat_scores: list[float] = []
    repeat_ok: list[bool] = []
    if collect_oof:
        oof = (
            np.full((n_repeats, len(X)), np.nan, dtype=float)
            if n_repeats > 1
            else np.full(len(X), np.nan, dtype=float)
        )
    else:
        oof = None
    # The repeat-specific keyword arguments are passed only for a real
    # multi-split run: with one repeat the classic evaluator call (and its
    # monkeypatchable signature) stays exactly as before.
    repeat_options: dict[str, Any] = {}
    fold_options: dict[str, Any] = {}
    if n_repeats > 1:
        repeat_options = {
            "n_repeats": n_repeats,
            "split_aggregation": aggregation,
            "scores_out": repeat_scores,
            "mask_out": repeat_ok,
        }
        fold_options = {**repeat_options, "validation": validation}
    if validation.method == "train_test_split":
        # The hold-out value is returned as is (a non-finite one fails the
        # trial in Optuna).
        value = _score_holdout(
            estimator, X, y, validation, scorer, random_state, oof, **repeat_options
        )
        return TrialScore(
            value=value,
            oof=successful_repeat_oof(oof, repeat_ok, n_repeats),
            repeat_scores=tuple(repeat_scores),
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
            **fold_options,
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
            **fold_options,
        )
    else:
        if n_repeats > 1:
            repeat_options["random_state"] = random_state
        value = _score_fold_mean(
            estimator, X, y, validation, scorer, oof, **repeat_options
        )
    # If we got +inf (an nrmse error) or NaN, return the worst float
    # (analog of the float32 minimum or float('-inf')).
    selected_oof = successful_repeat_oof(oof, repeat_ok, n_repeats)
    if not np.isfinite(value):
        return TrialScore(
            value=HPO_WORST_SCORE,
            oof=selected_oof,
            repeat_scores=tuple(repeat_scores),
        )
    return TrialScore(value=value, oof=selected_oof, repeat_scores=tuple(repeat_scores))


def _score_holdout(
    estimator: Any,
    X: Any,
    y: Any,
    validation: ValidationPlan,
    scorer: Any,
    random_state: int | None,
    oof_out: np.ndarray | None,
    *,
    n_repeats: int = 1,
    split_aggregation: str = "median",
    scores_out: list[float] | None = None,
    mask_out: list[bool] | None = None,
) -> float:
    """Score a clone of the estimator on the ``train_test_split`` hold-out.

    If 'auto' was resolved in make_cv, the already computed
    (dataset-dependent) integer test size is used so that ``iter_splits``
    does not recompute the decision; otherwise — the fixed fraction.

    The hold-out rows are rows the model was REQUIRED to predict: a
    non-finite (NaN/Inf) prediction on any of them raises
    ``InvalidPredictionError`` — a quality defect of the model that must not
    be masked out of the estimate (a model that fails on most of the hold-out
    rows must not get a good score from the few rows it predicted; the
    values are never replaced with zeros). The error is a non-fatal
    ``ValueError``: the trial is pruned by the caller's failure contract.

    With several repeats every repeat draws its own hold-out split (the
    deterministic seed of the repeat); a failed repeat is penalized, only all
    failed repeats fail the trial.

    Returns:
        float: The raw metric value on the hold-out rows (the aggregate over
            the repeats when ``n_repeats > 1``).

    Raises:
        InvalidPredictionError: If the model returned a non-finite
            prediction for at least one hold-out row (the single-repeat
            path).
    """
    if n_repeats > 1:
        return run_repeat_scores(
            lambda _r, seed, sub_oof: _score_holdout(
                estimator, X, y, validation, scorer, seed, sub_oof
            ),
            validation=validation,
            n_repeats=n_repeats,
            random_state=random_state,
            oof_out=oof_out,
            split_aggregation=split_aggregation,
            all_failed_message=(
                f"All {n_repeats} hold-out repeat(s) failed during "
                "evaluation — the estimate is impossible"
            ),
            scores_out=scores_out,
            mask_out=mask_out,
        )
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
        _check_holdout_predictions_finite(model, X_te)
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
    holdout_pred = np.asarray(model.predict(X_te), dtype=float).reshape(-1)
    # The hold-out rows are rows the model was REQUIRED to predict: a
    # non-finite prediction is a quality defect (see the docstring).
    if not np.isfinite(holdout_pred).all():
        raise InvalidPredictionError(
            "Model returned non-finite predictions for validation samples "
            f"({int((~np.isfinite(holdout_pred)).sum())} of "
            f"{len(holdout_pred)} hold-out rows)"
        )
    oof_out[np.asarray(test_idx)] = holdout_pred
    return float(scorer(model, X_te, y_te))


def _check_holdout_predictions_finite(model: Any, X_te: Any) -> None:
    """Raise ``InvalidPredictionError`` on non-finite hold-out predictions."""
    pred = np.asarray(model.predict(X_te), dtype=float).reshape(-1)
    if not np.isfinite(pred).all():
        raise InvalidPredictionError(
            "Model returned non-finite predictions for validation samples "
            f"({int((~np.isfinite(pred)).sum())} of {len(pred)} hold-out rows)"
        )


def _score_fold_mean(
    estimator: Any,
    X: Any,
    y: Any,
    validation: ValidationPlan,
    scorer: Any,
    oof_out: np.ndarray | None,
    *,
    n_repeats: int = 1,
    split_aggregation: str = "median",
    random_state: int | None = None,
    scores_out: list[float] | None = None,
    mask_out: list[bool] | None = None,
) -> float:
    """The fold-average score of additive metrics (``cross_val_score``).

    With several repeats every repeat is scored on its own deterministic
    k-fold split (the same splits as ``iter_splits``), the per-repeat scores
    are aggregated (median by default). ``validation.cv`` is guaranteed
    non-None here: a non-train_test_split and non-pruning branch means
    k_fold/loo, where make_cv always creates a cv object.
    """
    if n_repeats > 1:
        return run_repeat_scores(
            lambda _r, seed, sub_oof: _score_fold_mean_once(
                estimator,
                X,
                y,
                repeat_cv(validation, seed),
                scorer,
                sub_oof,
            ),
            validation=validation,
            n_repeats=n_repeats,
            random_state=random_state,
            oof_out=oof_out,
            split_aggregation=split_aggregation,
            all_failed_message=(
                f"All {n_repeats} repeat(s) failed during evaluation — the "
                "fold-mean estimate is impossible"
            ),
            scores_out=scores_out,
            mask_out=mask_out,
        )
    return _score_fold_mean_once(estimator, X, y, validation.cv, scorer, oof_out)


def _score_fold_mean_once(
    estimator: Any,
    X: Any,
    y: Any,
    cv: Any,
    scorer: Any,
    oof_out: np.ndarray | None,
) -> float:
    """The fold-average score of one repeat (the classic path)."""
    if oof_out is None:
        scores = model_selection.cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scorer,
            n_jobs=1,
        )
        return float(np.mean(scores))
    # Additive metrics with OOF collection: the same fold scores as
    # ``cross_val_score`` (same cv object and scorer), with the fold
    # estimators kept to predict their test rows. The fold-average
    # ``cross_validate`` scoring mirrors the ``cross_val_score`` contract: a
    # fold whose predictions break the metric is set to NaN by sklearn, so a
    # non-finite average already means the per-fold non-finite check below
    # raised (a fold with NaN/Inf predictions cannot coexist with a finite
    # average).
    cv_res = model_selection.cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scorer,
        n_jobs=1,
        return_estimator=True,
        return_indices=True,
    )
    avg_score = float(np.mean(cv_res["test_score"]))
    for fold_est, test_idx in zip(
        cv_res["estimator"], cv_res["indices"]["test"], strict=True
    ):
        fold_pred = np.asarray(
            fold_est.predict(_safe_indexing(X, test_idx)), dtype=float
        ).reshape(-1)
        # The fold's test rows are rows the model was REQUIRED to predict: a
        # non-finite prediction is a quality defect (not a coverage gap) —
        # the OOF vector must not carry it as a masked-out row (the value is
        # never replaced with zeros).
        if not np.isfinite(fold_pred).all():
            raise InvalidPredictionError(
                "Model returned non-finite predictions for validation samples "
                f"({int((~np.isfinite(fold_pred)).sum())} of {len(fold_pred)} "
                "rows in one fold)"
            )
        oof_out[test_idx] = fold_pred
    return avg_score
