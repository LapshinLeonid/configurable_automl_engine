"""Validation scoring of the single-model trainer.

``score_on_validation_splits`` scores the trainer pipeline on the same
splits HPO uses (``validation.iter_splits``), aggregates the primary and the
additional metrics (fold mean or pooled OOF value), and builds the
out-of-fold prediction vector with ``RMSE_oof`` (``_finalize_oof`` /
``_align_oof_vector``). The functions are pure: the fold pipelines come from
a ``make_steps`` callable passed in, and the result is returned as
``ValidationScores``. The module must not import
``configurable_automl_engine.trainer``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from configurable_automl_engine.training_engine.metrics import (
    direction_label,
    get_scorer_object,
    is_non_additive_metric,
    oof_rmse,
    pooled_scorer_value,
    to_user_value,
    warn_pooled_fallback_once,
)
from configurable_automl_engine.validation import iter_splits, make_cv

from .trainer_errors import TrainingError
from .trainer_state import ValidationScores

#: Builds fresh pipeline steps (clones of the preprocessor and the model):
#: ``make_steps(feature_selection_active, X_for_estimate)``.
StepsFactory = Callable[[bool, Any], list[tuple[str, Any]]]

# The records keep the logger name of ``ModelTrainer`` (its ``logger``).
_LOG = logging.getLogger("configurable_automl_engine.trainer")


def score_on_validation_splits(
    X: Any,
    y: Any,
    make_steps: StepsFactory,
    *,
    feature_selection_active: bool,
    metric: str,
    additional_metrics: list[str],
    validation_strategy: str,
    n_folds: int,
    test_size: float,
    random_state: int | None,
    algorithm: str,
) -> ValidationScores:
    """Evaluate model quality on held-out data.

    A single mechanism shared with HPO: splits are built via
    ``validation.iter_splits`` (the same factory used by ``tuner.optimize``),
    so ``val_score``/``additional_scores`` match the estimate used to
    compare models, up to sign correction.

    How it works:
    1. On each fold a fresh pipeline is assembled by ``make_steps`` (clones
       of the preprocessor and the model plus selector/oversampler per
       config) and trained on the fold's train part.
    2. Raw values of the primary metric and all additional metrics are
       computed on the fold's validation part.
    3. ``val_score = to_user_value(metric, mean(raw))`` —
       ``additional_scores[m] = to_user_value(m, mean(raw_m))`` — for
       additive metrics; for non-additive metrics (RMSE, MSE, R², nRMSE,
       RMSLE, D², explained_variance, max error, median absolute error
       and their aliases) the value is the pooled estimate over the OOF
       vector (each row enters the calculation exactly once).
    4. In parallel, an OOF channel is collected: predictions from the
       validation parts of the folds are accumulated into a single
       ``y_pred_oof`` vector aligned to the original row order, and
       ``RMSE_oof`` is computed on the whole vector at once (see
       ``_finalize_oof``).

    A failure of a single additional metric (scorer error/None on a fold) →
    WARNING + the metric is skipped; failure of all folds of the primary
    metric → ``TrainingError`` (fail-fast, before final training).
    Non-finite fold scores (e.g., inf for NRMSE on a constant target) are
    returned as is. An OOF collection failure on a single fold does not
    crash the validation scoring: it only reduces the coverage of the OOF
    vector (WARNING).

    Note (strategy interface): building the OOF channel does not depend on
    a particular strategy — only the positional ``test_idx`` provided by
    ``iter_splits`` for any strategy is used. Order-aware strategies
    (``TimeSeriesSplit`` and similar) registered through
    ``validation.SUPPORTED_VALIDATION_METHODS``/``make_cv`` will work
    without rewriting this method.

    Args:
        X: Feature matrix (full data).
        y: Target variable (full data).
        make_steps: Builds fresh pipeline steps for a fold
            (``make_steps(feature_selection_active, X_for_estimate)``).
        feature_selection_active: Flag enabling the feature selection step.
        metric: The primary metric.
        additional_metrics: The additional (informational) metrics.
        validation_strategy: The validation method.
        n_folds: The number of folds of ``k_fold``.
        test_size: The hold-out size of ``train_test_split``.
        random_state: The seed of the splits (``None`` — the fixed fallback
            seed 42).
        algorithm: The algorithm name (fallback WARNING deduplication).

    Returns:
        ValidationScores: ``val_score``, the additional scores and the OOF
            channel.

    Raises:
        TrainingError: If no fold produced a valid score for the primary
            metric, or on a critical split-initialization error.
    """
    # Deterministic splits: with random_state=None (non-fixed main
    # training) the validation splits must still be reproducible (a fixed
    # fallback seed, as in the feature selection auto-check).
    splits_seed = random_state if random_state is not None else 42
    method = validation_strategy

    scorer, add_scorers = _build_validation_scorers(metric, additional_metrics, y)

    # Materialize the splits before the fold loop: validation initialization
    # errors (insufficient data, unknown method, etc.) must abort training
    # immediately rather than being masked as a "failed fold".
    # include_indices=True: the splits additionally provide positional
    # indices of the validation rows for building an OOF vector aligned to
    # the original order.
    try:
        splits = list(
            iter_splits(
                X,
                y,
                method=method,
                n_folds=n_folds,
                test_size=test_size,
                random_state=splits_seed,
                include_indices=True,
            )
        )
    except Exception as err:  # noqa: BLE001
        raise TrainingError(f"Error calculating metrics on validation: {err}")

    n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
    effective_method = _effective_validation_method(
        X,
        n_samples,
        method=method,
        n_folds=n_folds,
        test_size=test_size,
        random_state=splits_seed,
    )

    fold_raw: list[float] = []
    add_raw: dict[str, list[float]] = {m: [] for m in add_scorers}
    fold_errors: list[str] = []
    # OOF channel buffers: filled by the positions of validation rows in
    # the original data; uncovered rows remain NaN and are discarded when
    # computing the pooled metrics (RMSE_oof and, for non-additive
    # metrics, val_score/additional_scores).
    oof_pred_buffer = np.full(n_samples, np.nan, dtype=float)
    oof_y_buffer = np.full(n_samples, np.nan, dtype=float)
    # Whether the primary/additional metric is non-additive and therefore
    # must be computed over the pooled OOF vector rather than averaged by
    # folds (RMSE, MSE, R², nRMSE). MAE and similar per-sample means stay
    # on the fold-average path (their fold mean equals the pooled value).
    pool_primary = is_non_additive_metric(metric)
    pool_additive = {mname: is_non_additive_metric(mname) for mname in add_scorers}
    for X_tr, X_te, y_tr, y_te, test_idx in splits:
        try:
            # A fresh pipeline per fold: clones of preprocessor/base_model
            # (sklearn Pipeline does not clone the passed steps, so shared
            # instances would mutate the state of a neighboring fold).
            # The estimate of the isotonic guard is based on the fold's
            # training part so it matches the fold-fitted preprocessor output
            # (a category absent from the train part must not trigger the
            # guard).
            fold_pipe = Pipeline(make_steps(feature_selection_active, X_tr))
            fold_pipe.fit(X_tr, y_tr)

            # For non-additive metrics the per-fold scorer is invoked only
            # until the first successful value (a fail-fast check that the
            # scorer works and does not return None): the final
            # val_score/additional_scores come from the pooled OOF vector,
            # so per-fold scoring noise (e.g. LOO+R², where every fold R²
            # is undefined) must not pollute the logs on the common path.
            # Additive metrics (MAE and similar per-sample means) keep
            # per-fold scoring on every fold — their fold mean IS the
            # estimate.
            if not pool_primary or not fold_raw:
                raw = scorer(fold_pipe, X_te, y_te)
                if raw is None:
                    raise TrainingError("Scorer returned None")
                fold_raw.append(float(raw))

            _score_additional_metrics_on_fold(
                fold_pipe, X_te, y_te, add_scorers, pool_additive, add_raw
            )
            _collect_fold_oof(
                fold_pipe, X_te, y_te, test_idx, oof_pred_buffer, oof_y_buffer
            )
        except Exception as err:  # noqa: BLE001
            fold_errors.append(str(err))

    if not fold_raw:
        detail = "; ".join(dict.fromkeys(fold_errors)) or "unknown failure"
        raise TrainingError(f"Validation scoring failed for all folds: {detail}")

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
        fold_raw,
        pooled=pool_primary,
        oof_y=oof_y_buffer,
        oof_pred=oof_pred_buffer,
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
    for mname, raw_values in add_raw.items():
        if not raw_values:
            # All folds failed for this metric — it was already skipped with a WARNING.
            continue
        raw_values_agg = _aggregate_raw_values(
            add_scorers[mname],
            raw_values,
            pooled=pool_additive[mname],
            oof_y=oof_y_buffer,
            oof_pred=oof_pred_buffer,
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
        oof_pred_buffer,
        oof_y_buffer,
        X,
        n_samples,
        method=effective_method,
    )
    return ValidationScores(
        val_score=val_score,
        additional_scores=additional_scores,
        oof_predictions=oof_predictions,
        oof_score=oof_score,
        oof_coverage=oof_coverage,
    )


def _build_validation_scorers(
    metric: str, additional_metrics: list[str], y: Any
) -> tuple[Callable[..., Any], dict[str, Callable[..., Any]]]:
    """Build the primary scorer and the additional-metric scorers.

    global_y=y: the global NRMSE is normalized by the full target (the range
    does not depend on a given fold). An additional metric whose scorer
    cannot be built is skipped with a WARNING.

    Args:
        metric: The primary metric.
        additional_metrics: The additional metrics.
        y: Target variable (full data).

    Returns:
        The primary scorer and the scorers of the additional metrics.

    Raises:
        TrainingError: If the primary scorer cannot be built.
    """
    try:
        scorer = cast(Callable[..., Any], get_scorer_object(metric, global_y=y))
    except Exception as err:  # noqa: BLE001
        raise TrainingError(f"Error calculating metrics on validation: {err}")
    add_scorers: dict[str, Callable[..., Any]] = {}
    for mname in additional_metrics:
        try:
            add_scorers[mname] = cast(
                Callable[..., Any], get_scorer_object(mname, global_y=y)
            )
        except Exception as err:  # noqa: BLE001
            _LOG.warning(
                "Additional metric '%s' could not be computed: %s. "
                "Training result and artifact are unaffected.",
                mname,
                err,
            )
    return scorer, add_scorers


def _effective_validation_method(
    X: Any,
    n_samples: int,
    *,
    method: str,
    n_folds: int,
    test_size: float,
    random_state: int,
) -> str:
    """Resolve the validation method actually applied by ``iter_splits``.

    The method actually applied after a possible fallback (k_fold ->
    train_test_split for small N): OOF channel warnings must rely on the
    splits actually used rather than the requested strategy. Resolved via the
    same make_cv that iter_splits calls internally — the values are
    guaranteed to match.

    Returns:
        The effective validation method name.
    """
    n_features = X.shape[1] if method == "auto" else None
    try:
        effective_method, _, _ = make_cv(
            n_samples,
            val_method=method,
            n_folds=n_folds,
            random_state=random_state,
            test_size=test_size,
            n_features=n_features,
        )
    except Exception:  # noqa: BLE001
        # Splits were already materialized above — we get here only in an
        # impossible/degenerate case; fall back to the requested method.
        effective_method = method
    return cast(str, effective_method)


def _score_additional_metrics_on_fold(
    fold_pipe: Any,
    X_te: Any,
    y_te: Any,
    add_scorers: dict[str, Callable[..., Any]],
    pool_additive: dict[str, bool],
    add_raw: dict[str, list[float]],
) -> None:
    """Score the additional metrics on a validation fold (failures warn)."""
    for mname, m_scorer in add_scorers.items():
        if pool_additive[mname] and add_raw[mname]:
            continue
        try:
            m_raw = m_scorer(fold_pipe, X_te, y_te)
            if m_raw is None:
                raise TrainingError(
                    f"Scorer returned None for additional metric '{mname}'"
                )
            add_raw[mname].append(float(m_raw))
        except Exception as err:  # noqa: BLE001
            _LOG.warning(
                "Additional metric '%s' could not be computed on a "
                "validation fold: %s. Training result and artifact "
                "are unaffected.",
                mname,
                err,
            )


def _collect_fold_oof(
    fold_pipe: Any,
    X_te: Any,
    y_te: Any,
    test_idx: Any,
    oof_pred_buffer: np.ndarray,
    oof_y_buffer: np.ndarray,
) -> None:
    """Collect the out-of-fold predictions of a validation fold.

    OOF channel: a prediction on the validation part of the fold — the model
    was trained only on the train part, so the row is guaranteed not to have
    been used in training (no leakage). A separate try: an OOF collection
    failure does not crash the fold scoring, it only reduces the OOF vector
    coverage (WARNING).
    """
    try:
        preds = np.asarray(fold_pipe.predict(X_te)).reshape(-1)
        idx = np.asarray(test_idx)
        if preds.shape[0] != idx.shape[0]:
            raise TrainingError(
                "OOF prediction length mismatch: got "
                f"{preds.shape[0]} predictions for {idx.shape[0]} rows"
            )
        if y_te is not None:
            oof_pred_buffer[idx] = preds
            oof_y_buffer[idx] = np.asarray(y_te, dtype=float).reshape(-1)
    except Exception as err:  # noqa: BLE001
        _LOG.warning(
            "OOF predictions could not be collected on a "
            "validation fold: %s. RMSE_oof coverage reduced.",
            err,
        )


def _aggregate_raw_values(
    scorer: Callable[..., Any],
    raw_values: list[float],
    *,
    pooled: bool,
    oof_y: np.ndarray,
    oof_pred: np.ndarray,
    algorithm: str,
    phase_name: str,
    failure_reason: str,
    empty_reason: str,
) -> float:
    """Aggregate the raw values of a metric: pooled OOF value or fold mean.

    A non-additive metric is computed over the pooled OOF buffer when it
    holds a finite pair; when the pooled value is impossible, the fold mean
    is used and the fallback WARNING is emitted once per (algorithm, phase).

    Returns:
        The raw (optimizer-semantics) metric value.
    """
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
    oof_pred: np.ndarray,
    oof_y: np.ndarray,
    X: Any,
    n_samples: int,
    *,
    method: str,
) -> tuple[pd.Series | np.ndarray, float | None, float]:
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

    Returns:
        The OOF vector in the original row order, ``RMSE_oof`` (``None``
        when impossible) and the coverage.
    """
    oof_vector = _align_oof_vector(oof_pred, X)
    valid = np.isfinite(oof_pred) & np.isfinite(oof_y)
    n_valid = int(valid.sum())
    coverage = float(n_valid / n_samples) if n_samples > 0 else 0.0

    # train_test_split limitation: OOF covers only the validation part.
    # Explicitly documented in the log.
    if method == "train_test_split":
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
    if n_samples <= 5:
        _LOG.warning(
            "OOF evaluation on very small data (N=%d): the OOF vector is "
            "short and RMSE_oof has high variance.",
            n_samples,
        )

    if n_valid == 0:
        _LOG.warning(
            "RMSE_oof could not be computed: no valid out-of-fold "
            "prediction/target pairs."
        )
        return oof_vector, None, coverage

    dropped = n_samples - n_valid
    # For train_test_split the uncovered rows are the train part; the
    # limitation was already documented by a separate WARNING above, so the
    # generic row-exclusion warning is logged only for non-finite
    # predictions/targets.
    if dropped and method != "train_test_split":
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
        _LOG.warning("RMSE_oof could not be computed: %s", err)
    # RMSE_oof is a user-facing error value ("min better"): the direction
    # is stated explicitly, and the optimizer flag is not applied.
    _LOG.info(
        "OOF evaluation: RMSE_oof=%.4f (min better) on %d/%d rows (coverage=%.3f).",
        oof_score if oof_score is not None else float("nan"),
        n_valid,
        n_samples,
        coverage,
    )
    return oof_vector, oof_score, coverage


def _align_oof_vector(
    oof_pred: np.ndarray,
    X: Any,
) -> pd.Series | np.ndarray:
    """Align the OOF vector ``y_pred_oof`` to the original row order.

    For pandas inputs (``DataFrame``/``Series``) with a unique index, a
    ``pd.Series`` with preserved row labels is returned, including
    MultiIndex. For numpy inputs and data with an unrecoverable index
    (duplicate labels), an ``np.ndarray`` aligned by position is returned:
    no sorting by index value is performed.

    Note: the ``Index.hasnans`` property is not defined for ``MultiIndex``
    (``NotImplementedError``), so the check is wrapped in try/except —
    otherwise ``fit()`` would fail on data with a MultiIndex even though
    such input worked on main.

    Args:
        oof_pred: OOF prediction buffer (NaN for uncovered rows), aligned
            by position to the original data.
        X: Original feature matrix.

    Returns:
        The ``y_pred_oof`` vector in the original row order.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)) and X.index.is_unique:
        try:
            has_nan_labels = bool(X.index.hasnans)
        except (NotImplementedError, TypeError):
            # MultiIndex and other indexes without hasnans support: NaN
            # labels cannot be checked, but uniqueness guarantees unambiguous
            # positional alignment — keep the row labels.
            has_nan_labels = False
        if not has_nan_labels:
            return pd.Series(oof_pred, index=X.index, name="y_pred_oof")
    return oof_pred
