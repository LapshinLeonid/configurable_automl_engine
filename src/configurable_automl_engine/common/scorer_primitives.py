"""Low-level scoring primitives shared by the metric and BBC-CV modules.

The raw RMSE/NRMSE implementations, the pooled scorer evaluation and the
preconfigured scorer registry are consumed by both ``common.metrics`` (the
metric registry and direction helpers) and ``common.bbc_cv`` (the
bootstrap bias-corrected CV estimate). Keeping them in a module of their own
makes the dependency one-directional: both consumers import the primitives,
and ``common.bbc_cv`` no longer imports the private names of
``common.metrics``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from configurable_automl_engine.common.errors import InvalidPredictionError
from configurable_automl_engine.common.sklearn_compat import scorer_raw_parts

logger = logging.getLogger(__name__)

__all__ = [
    "SCORER_OBJECTS",
    "nrmse",
    "pooled_scorer_value",
    "resolve_expected_valid_mask",
    "rmse",
]


# --------------------------------------------------------------------------- #
#  The metrics themselves
# --------------------------------------------------------------------------- #
def rmse(y_true: Any, y_pred: Any) -> float:
    """Compute the root mean squared error (RMSE).

    The MSE is computed with scikit-learn's ``mean_squared_error`` and the
    square root is taken from the result; the value is cast to ``float`` for
    consistency.

    Args:
        y_true: The true values.
        y_pred: The predictions aligned with ``y_true``.

    Returns:
        float: The root mean squared error (lower is better).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def resolve_expected_valid_mask(
    yt: np.ndarray, yp: np.ndarray, expected: np.ndarray | None
) -> np.ndarray:
    """Resolve the shared masking contract of ``oof_rmse``/``pooled_scorer_value``.

    The two cases of the non-finite prediction policy:

    - a row the split strategy did not require to predict (outside the
      ``expected`` mask, e.g. the train part of ``train_test_split``) stays
      masked out of the estimate even when its values are finite — a normal
      coverage gap by design;
    - a row the model WAS required to predict (inside ``expected``) must
      carry a finite prediction: a model-returned NaN/Inf there is a quality
      defect, not a coverage gap, and raises ``InvalidPredictionError``
      instead of being silently masked (a model that fails on most
      validation rows must not get a good estimate from the few rows it
      predicted).

    Args:
        yt: The true values as a float array.
        yp: The predictions as a float array.
        expected: An optional boolean mask aligned with ``yp``: True where
            the model was required to produce a finite prediction.
            ``None`` — every row is expected: any non-finite pair is dropped.

    Returns:
        np.ndarray: The boolean mask of the rows entering the estimate.

    Raises:
        InvalidPredictionError: If any expected row has a non-finite
            prediction (with ``expected`` given).
    """
    if expected is None:
        return np.isfinite(yt) & np.isfinite(yp)
    exp = np.asarray(expected, dtype=bool)
    if not np.isfinite(yp[exp]).all():
        raise InvalidPredictionError(
            "Model returned non-finite predictions for validation samples "
            f"({int((~np.isfinite(yp[exp])).sum())} of {int(exp.sum())} "
            "expected rows)"
        )
    # Rows outside the expected set (e.g. the train part of a hold-out)
    # are excluded from the estimate even when their values are finite.
    return exp & np.isfinite(yt) & np.isfinite(yp)


def pooled_scorer_value(
    scorer: Any, y_true: Any, y_pred: Any, *, expected: Any | None = None
) -> float:
    """Compute a scorer value over the whole aligned vector at once.

    Applies the raw metric function (with the scorer's sign and kwargs) over
    the concatenated vector, dropping non-finite pairs — the same masking
    contract as ``oof_rmse``. Used to estimate non-additive metrics (RMSE,
    MSE, R², nRMSE) over the pooled OOF vector instead of averaging per-fold
    values.

    With ``expected`` the masking splits into the two fundamentally different
    cases (the same contract as ``oof_rmse``): rows outside the expected set
    (e.g. the train part of ``train_test_split``) stay masked out — a normal
    coverage gap by design — while a non-finite prediction on an EXPECTED row
    raises ``InvalidPredictionError``: the model was required to predict it,
    and silently scoring only the rows it happened to predict would reward a
    model that fails on most of the validation rows. Without ``expected``
    any non-finite pair is dropped.

    Args:
        scorer: A scikit-learn ``make_scorer`` object (its raw metric parts
            are read through the stable adapter
            ``common.sklearn_compat.scorer_raw_parts``).
        y_true: True values aligned with ``y_pred`` by position.
        y_pred: Predictions aligned with ``y_true`` by position.
        expected: An optional boolean mask aligned with ``y_pred``: True
            where the model was required to produce a finite prediction
            (validation rows). ``None`` — every row is expected.

    Returns:
        float: The raw (sign-applied) metric value over all valid pairs.

    Raises:
        InvalidPredictionError: If any expected row has a non-finite
            prediction (with ``expected`` given).
        ValueError: If no valid pair remains after dropping non-finite pairs,
            or if the scorer does not expose a raw ``_score_func`` (it cannot
            be pooled).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    valid = resolve_expected_valid_mask(yt, yp, expected)
    if not np.any(valid):
        raise ValueError("No valid (y_true, y_pred) pairs for the pooled metric.")
    score_func, sign, kwargs = scorer_raw_parts(scorer)
    if score_func is None:
        raise ValueError(
            "The scorer cannot be pooled: no raw metric function is exposed."
        )
    return float(sign * score_func(yt[valid], yp[valid], **kwargs))


# Per-fold NRMSE
def nrmse(y_true: Any, y_pred: Any) -> float:
    """Compute the local normalized RMSE (NRMSE).

    The standard RMSE for the current sample is divided by the range
    (max - min) of the passed true values. If the range is below 1e-6, the
    normalization is undefined: a perfect prediction (RMSE = 0) is scored as
    0.0 — the error-metric counterpart of scikit-learn's R² force_finite
    rule, which reports 1.0 for a perfect constant-target prediction —
    otherwise ``inf`` is returned with a warning to avoid division by zero.

    Args:
        y_true: The true values of the current split.
        y_pred: The predictions aligned with ``y_true``.

    Returns:
        float: The locally normalized RMSE (lower is better).
    """
    value = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.max(y_true) - np.min(y_true)

    if denom < 1e-6:
        if value == 0.0:
            return 0.0
        logger.warning(
            f"NRMSE: target is constant (range < 1e-6) for split of size "
            f"{len(y_true)}. Returning +inf (will be inverted to -inf by scorer)."
        )
        return float("inf")
    return float(value / denom)


# --------------------------------------------------------------------------- #
#  Registry of ready-made scorer objects for use in the sklearn API
# --------------------------------------------------------------------------- #
# For each metric the optimization direction (is_greater_better) is stored
# explicitly next to the scorer: True — "higher is better" (r2, as well as
# inverted neg_ metrics), False — "lower is better" (rmse, mae, etc.).
# The direction is never inferred heuristically from name substrings — only
# from this registry or from the sklearn scorer object itself.
SCORER_OBJECTS: dict[str, tuple[Callable[..., Any], bool]] = {
    # For errors set greater_is_better=False;
    # sklearn itself will return negative values for maximization
    "nrmse": (make_scorer(nrmse, greater_is_better=False), False),
    "rmse": (make_scorer(rmse, greater_is_better=False), False),
    "neg_root_mean_squared_error": (make_scorer(rmse, greater_is_better=False), True),
    "mae": (make_scorer(mean_absolute_error, greater_is_better=False), False),
    "mse": (make_scorer(mean_squared_error, greater_is_better=False), False),
    "r2": (make_scorer(r2_score, greater_is_better=True), True),
}
