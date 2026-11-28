"""Validation scoring of the single-model trainer.

``score_on_validation_splits`` uses the same splits as HPO, aggregates the
metrics, builds the OOF vector and combines repeated splits by the configured
policy. Pure functions take pipelines from ``make_steps`` and return
``ValidationScores``; the module must not import the trainer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import chain, groupby
from typing import cast

import numpy as np
from imblearn.pipeline import Pipeline

from configurable_automl_engine import validation as _val
from configurable_automl_engine.common.casts import as_scorer
from configurable_automl_engine.common.errors import (
    InvalidPredictionError,
    TrainingError,
)
from configurable_automl_engine.common.metrics import (
    get_scorer_object,
    is_non_additive_metric,
)
from configurable_automl_engine.common.types import (
    Array,
    Estimator,
    FloatArray,
    Frame,
    Scorer,
    Target,
)
from configurable_automl_engine.validation import iter_splits

from .trainer_data import _align_oof_vector as _align_oof_vector
from .trainer_scoring_aggregate import (
    ScoringPlan,
    _finalize_oof as _finalize_oof,
    _multi_repeat_scores,
    _RepeatScoring,
    _single_repeat_scores,
)
from .trainer_state import ValidationScores

#: Builds fresh pipeline steps (clones of the preprocessor and the model):
#: ``make_steps(feature_selection_active, X_for_estimate)``.
StepsFactory = Callable[[bool, Frame | None], list[tuple[str, object]]]

_LOG = logging.getLogger(__name__)


def score_on_validation_splits(
    X: Frame,
    y: Target,
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
    n_repeats: int | None = None,
    split_aggregation: str = "median",
) -> ValidationScores:
    """Evaluate model quality on held-out data.

    A single mechanism shared with HPO: splits are built via
    ``validation.iter_splits`` (the same factory used by ``tuner.optimize``),
    so ``val_score``/``additional_scores`` match the estimate used to
    compare models, up to sign correction. With ``n_repeats = R > 1`` the
    base strategy is re-drawn R times with the deterministic repeat seeds;
    the per-repeat estimates are combined by ``split_aggregation`` (median
    by default, mean on request).

    How it works:
    1. On each fold a fresh pipeline is assembled by ``make_steps`` and
       trained on the fold's train part.
    2. Raw values of the primary metric and all additional metrics are
       computed on the fold's validation part.
    3. ``val_score = to_user_value(metric, aggregate(raw))`` —
       ``additional_scores[m]`` is aggregated the same way. Additive metrics
       use the fold mean; non-additive metrics use the pooled estimate over
       the repeat's OOF vector (each row enters the calculation exactly
       once). The per-repeat values are then combined by
       ``split_aggregation``.
    4. In parallel, an OOF channel is collected: predictions from the
       validation parts of the folds are accumulated into a single
       ``y_pred_oof`` vector aligned to the original row order (see
       ``_finalize_oof``). ``oof_score`` is the median (or mean) of the
       per-repeat ``RMSE_oof`` values.

    A failure of a single additional metric (scorer error/None on a fold) →
    WARNING + the metric is skipped; failure of all folds of the primary
    metric in every repeat → ``TrainingError`` (fail-fast, before final
    training). Non-finite fold scores (e.g., inf for NRMSE on a constant
    target) are returned as is for a single repeat. A fold whose predictions
    contain a non-finite value (NaN/Inf) is not collected into the OOF
    channel — the validation rows are rows the model was REQUIRED to
    predict, so such a prediction is a quality defect of the model, not a
    coverage gap: the fold's rows cannot enter the pooled estimate and the
    coverage is reduced (WARNING), the values are never replaced with zeros.
    With ``n_repeats > 1`` such a repeat is excluded from all aggregates
    (primary NaN, ``RMSE_oof`` inf); the remaining repeats still contribute.
    An OOF collection failure on a single fold does not crash the validation
    scoring: it invalidates the whole repeat and reduces the concatenated
    coverage (WARNING).

    Note (strategy interface): building the OOF channel does not depend on
    a particular strategy — only the positional ``test_idx`` provided by
    ``validation.iter_splits`` for any strategy is used. Order-aware strategies
    (``TimeSeriesSplit`` and similar) registered through
    ``validation.SUPPORTED_VALIDATION_METHODS``/``validation.make_cv`` will work
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
            seed 42). Repeat r uses ``random_state + r``.
        algorithm: The algorithm name (fallback WARNING deduplication).
        n_repeats: The number of repeated validation splits (``None`` → 1).
            The effective count is resolved by ``validation.iter_splits`` (LOO repeats
            are ignored with a WARNING).
        split_aggregation: How the per-repeat values are combined: 'median'
            (default) or 'mean'. Used only when more than one repeat is
            produced.

    Returns:
        ValidationScores: ``val_score``, the additional scores and the OOF
            channel. With more than one repeat, also the per-repeat vectors
            and the mean/median/std summary.

    Raises:
        TrainingError: If no fold produced a valid score for the primary
            metric in any repeat, or on a critical split-initialization
            error.
    """
    # Deterministic splits: with random_state=None (non-fixed main
    # training) the validation splits must still be reproducible (a fixed
    # fallback seed, as in the feature selection auto-check).
    splits_seed = random_state if random_state is not None else 42
    method = validation_strategy

    scorer, add_scorers = _build_validation_scorers(metric, additional_metrics, y)

    # Pull the first split eagerly so validation initialization errors
    # (insufficient data, unknown method, etc.) abort training immediately
    # instead of being masked as a "failed fold". The remaining groups are
    # streamed one repeat at a time. include_indices=True provides the
    # positional validation-row indices for the aligned OOF vector.
    try:
        split_iterator = iter_splits(
            X,
            y,
            method=method,
            n_folds=n_folds,
            test_size=test_size,
            random_state=splits_seed,
            include_indices=True,
            n_repeats=n_repeats,
        )
        first_split = next(split_iterator, None)
    except Exception as err:
        raise TrainingError(f"Error calculating metrics on validation: {err}") from err

    n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
    effective_method = _effective_validation_method(
        X,
        n_samples,
        method=method,
        n_folds=n_folds,
        test_size=test_size,
        random_state=splits_seed,
    )

    pool_primary = is_non_additive_metric(metric)
    pool_additive = {mname: is_non_additive_metric(mname) for mname in add_scorers}

    outcomes: list[_RepeatScoring] = []
    if first_split is not None:
        try:
            repeat_groups = groupby(
                chain((first_split,), split_iterator), key=_val.repeat_index_of
            )
            for repeat_index, repeat_splits in repeat_groups:
                outcomes.append(
                    _score_repeat(
                        list(repeat_splits),
                        repeat_index,
                        n_samples,
                        make_steps,
                        feature_selection_active=feature_selection_active,
                        scorer=scorer,
                        add_scorers=add_scorers,
                        pool_primary=pool_primary,
                        pool_additive=pool_additive,
                    )
                )
        except TrainingError:
            raise
        except Exception as err:
            raise TrainingError(f"Error calculating metrics on validation: {err}") from err  # fmt: skip

    if len(outcomes) <= 1:
        outcome = outcomes[0] if outcomes else _empty_repeat_outcome(n_samples)
        return _single_repeat_scores(
            outcome,
            X,
            n_samples,
            method=effective_method,
            metric=metric,
            scorer=scorer,
            add_scorers=add_scorers,
            pool_primary=pool_primary,
            pool_additive=pool_additive,
            algorithm=algorithm,
        )
    return _multi_repeat_scores(
        ScoringPlan(
            outcomes=outcomes,
            X=X,
            n_samples=n_samples,
            method=effective_method,
            metric=metric,
            scorer=scorer,
            add_scorers=add_scorers,
            pool_primary=pool_primary,
            pool_additive=pool_additive,
            algorithm=algorithm,
            split_aggregation=split_aggregation,
        )
    )


def _empty_repeat_outcome(n_samples: int) -> _RepeatScoring:
    """Build an empty outcome for a split generator that produced nothing."""
    return _RepeatScoring(
        fold_raw=[],
        add_raw={},
        oof_pred=np.full(n_samples, np.nan, dtype=float),
        oof_y=np.full(n_samples, np.nan, dtype=float),
        fold_errors=[],
        oof_failed=False,
        repeat_index=0,
    )


def _score_repeat(
    splits: list[_val.SplitWithIndices],
    repeat_index: int,
    n_samples: int,
    make_steps: StepsFactory,
    *,
    feature_selection_active: bool,
    scorer: Scorer,
    add_scorers: dict[str, Scorer],
    pool_primary: bool,
    pool_additive: dict[str, bool],
) -> _RepeatScoring:
    """Score all folds of one repeated split (a pure collection step)."""
    fold_raw: list[float] = []
    add_raw: dict[str, list[float]] = {m: [] for m in add_scorers}
    fold_errors: list[str] = []
    # OOF channel buffers: filled by the positions of validation rows in
    # the original data; uncovered rows remain NaN and are discarded when
    # computing the pooled metrics (RMSE_oof and, for non-additive
    # metrics, val_score/additional_scores).
    oof_pred_buffer = np.full(n_samples, np.nan, dtype=float)
    oof_y_buffer = np.full(n_samples, np.nan, dtype=float)
    oof_failed = False
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
                fold_pipe,
                X_te,
                cast(Target, y_te),
                add_scorers,
                pool_additive,
                add_raw,
            )
            if not _collect_fold_oof(
                fold_pipe,
                X_te,
                cast(Target, y_te),
                test_idx,
                oof_pred_buffer,
                oof_y_buffer,
            ):
                oof_failed = True
        except Exception as err:  # noqa: BLE001
            fold_errors.append(str(err))
    return _RepeatScoring(
        fold_raw=fold_raw,
        add_raw=add_raw,
        oof_pred=oof_pred_buffer,
        oof_y=oof_y_buffer,
        fold_errors=fold_errors,
        oof_failed=oof_failed,
        repeat_index=repeat_index,
    )


def _build_validation_scorers(
    metric: str, additional_metrics: list[str], y: Target
) -> tuple[Scorer, dict[str, Scorer]]:
    """Build the primary scorer and the additional-metric scorers.

    global_y=y: the global NRMSE is normalized by the full target (the range
    does not depend on a given fold). An additional metric whose scorer
    cannot be built is skipped with a WARNING."""
    try:
        scorer = as_scorer(get_scorer_object(metric, global_y=np.asarray(y)))
    except Exception as err:
        raise TrainingError(f"Error calculating metrics on validation: {err}") from err
    add_scorers: dict[str, Scorer] = {}
    for mname in additional_metrics:
        try:
            add_scorers[mname] = as_scorer(
                get_scorer_object(mname, global_y=np.asarray(y))
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
    X: Frame,
    n_samples: int,
    *,
    method: str,
    n_folds: int,
    test_size: float,
    random_state: int,
) -> str:
    """Resolve the validation method actually applied by ``validation.iter_splits``.

    The method actually applied after a possible fallback (k_fold ->
    train_test_split for small N): OOF channel warnings must rely on the
    splits actually used rather than the requested strategy. Resolved via the
    same ``validation.make_cv`` that ``validation.iter_splits`` calls internally — the values are
    guaranteed to match."""
    n_features = X.shape[1] if method == "auto" else None
    try:
        effective_method, _, _ = _val.make_cv(
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
    return effective_method


def _score_additional_metrics_on_fold(
    fold_pipe: Estimator,
    X_te: Frame,
    y_te: Target,
    add_scorers: dict[str, Scorer],
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
    fold_pipe: Estimator,
    X_te: Frame,
    y_te: Target,
    test_idx: Array,
    oof_pred_buffer: FloatArray,
    oof_y_buffer: FloatArray,
) -> bool:
    """Collect the out-of-fold predictions of a validation fold.

    OOF channel: a prediction on the validation part of the fold — the model
    was trained only on the train part, so the row is guaranteed not to have
    been used in training (no leakage). A separate try: an OOF collection
    failure does not crash the fold scoring, it only reduces the OOF vector
    coverage (WARNING).

    Non-finite (NaN/Inf) predictions of a fold are treated as a collection
    failure: the fold's validation rows are rows the model was REQUIRED to
    predict, and writing NaN into the buffer would mask them out of the
    pooled estimate (the values are never replaced with zeros) — the buffer
    keeps the previous values (NaN for uncovered rows) and the coverage is
    reduced, so the pooled metrics stay consistent with the reported
    coverage.

    Returns:
        bool: True when the fold was collected (or has no target to collect),
            False when the OOF channel could not be extended (the caller
            invalidates the repeat's ``RMSE_oof``).

    Raises:
        InvalidPredictionError: When the fold returned at least one
            non-finite prediction (raised inside the try and turned into a
            ``False`` result by this helper).
    """
    try:
        preds = np.asarray(fold_pipe.predict(X_te)).reshape(-1)
        idx = np.asarray(test_idx)
        if preds.shape[0] != idx.shape[0]:
            raise TrainingError(
                "OOF prediction length mismatch: got "
                f"{preds.shape[0]} predictions for {idx.shape[0]} rows"
            )
        if not np.isfinite(preds.astype(float, copy=False)).all():
            raise InvalidPredictionError(
                "Model returned non-finite predictions for validation samples "
                f"({int((~np.isfinite(preds.astype(float, copy=False))).sum())} "
                f"of {len(preds)} rows in the fold)"
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
        return False
    return True
