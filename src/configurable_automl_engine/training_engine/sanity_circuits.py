"""The four audit circuits of the Sanity Gate (A–D) and their helpers.

Holds the threshold comparisons, the numerical tolerances and the helpers of
the circuits (diversity, unique ratio, dead features through coefficients or
permutation, generalization gap). It depends only on numpy, pandas,
scikit-learn and :mod:`.sanity_result`; it must not import the gate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Threshold comparisons: THE single place that fixes the comparison side (< or ≤)
# --------------------------------------------------------------------------- #
# Circuits A/B fail on a STRICT shortfall below the threshold; circuits C/D —
# on a STRICT excess. Equality with the threshold is never a failure.
# Tuning other behavior (≤ instead of <) is allowed only here.
def _fails_below(value: float, threshold: float) -> bool:
    """Fail on a strict shortfall: ``value < threshold`` (equality passes)."""
    return value < threshold


def _fails_above(value: float, threshold: float) -> bool:
    """Fail on a strict excess: ``value > threshold`` (equality passes)."""
    return value > threshold


def _diversity_fails(
    diversity: float, threshold: float, adaptive: bool, r2_oof: float
) -> bool:
    """Fail side of circuit A, fixed in one place (module level).

    Non-adaptive mode: strict shortfall — ``diversity < threshold``, equality
    with the threshold passes. Adaptive mode is the only documented exception
    to the strict-``<`` convention: the reference is the constant model whose
    prediction spread is exactly zero, so a tie with it (``diversity`` at the
    numerical zero tolerance ``_ZERO_VAR_EPS``) fails — the model is not
    strictly better than the constant baseline. In
    the adaptive mode a model with a negative ``R²_oof`` (strictly worse than
    the constant prediction) fails even when numerical round-off makes
    ``diversity`` a tiny positive number.

    Args:
        diversity: ``Var(y_pred_oof) / Var(y)``.
        threshold: effective threshold (``min_prediction_diversity``).
        adaptive: whether the adaptive constant-model baseline is used.
        r2_oof: out-of-fold R² of the predictions (used in the adaptive
            mode only).

    Returns:
        bool: True when circuit A must fail for the given values.
    """
    if adaptive:
        return diversity <= _ZERO_VAR_EPS or (np.isfinite(r2_oof) and r2_oof < 0.0)
    return diversity < threshold


def _r2_oof(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-fold R² of the predictions (constant-model baseline).

    ``1 - Var(y - pred) / Var(y)`` — the fraction of variance explained.
    Used by circuit A in the adaptive mode to reject models that are worse
    than a constant prediction: a negative R² means the residual variance
    exceeds the target variance, i.e. the model is strictly worse than
    predicting the mean. The diversity ratio (``Var(pred) / Var(y)``) is
    subject to round-off and may come out as a tiny positive number when the
    predictions are numerically constant while R² is clearly negative, so
    the explicit R² test is the authoritative negative-signal detector.

    Args:
        y: the target (on the valid pairs).
        y_pred: the OOF predictions (on the same valid pairs).

    Returns:
        float: R²_oof, or NaN when ``Var(y) == 0`` (undefined).
    """
    var_y = float(np.var(y))
    if var_y <= _ZERO_VAR_EPS:
        return float("nan")
    residual = y - y_pred
    return float(1.0 - np.var(residual) / var_y)


# Numerical tolerances and rounding constants.
_ZERO_VAR_EPS = 1e-12  # target variance below this is treated as "constant y"
_ZERO_RMSE_EPS = 1e-12  # RMSE below this is considered zero (division by zero)
_MSE_EPS = 1e-12  # denominator stabilizer in the relative ΔMSE
_FALLBACK_DECIMALS = 6  # rounding precision when std(y)==0
_MIN_DECIMALS = -12
_MAX_DECIMALS = 10


def _rounding_decimals(std_y: float) -> int:
    """Rounding precision for predictions derived from the target scale (B).

    The rounding granularity equals 1% of the order of magnitude of ``std(y)``:
    ``10 ** (floor(log10(std_y)) - 2)``. Relative to ``std(y)`` itself this is
    0.1–1% — predictions that differ by less than roughly one hundredth of the
    noise scale of the target are treated as equal. For a constant target
    (``std_y == 0``) a fixed fallback of ``_FALLBACK_DECIMALS`` decimals is
    used; note that for large-scale targets this may collapse predictions
    differing by less than the fallback granularity — pre-scale ``y`` if finer
    granularity is needed.

    Args:
        std_y: standard deviation of the target.

    Returns:
        int: number of decimals for ``np.round`` (may be negative for
            large-scale y).
    """
    if not np.isfinite(std_y) or std_y <= 0.0:
        return _FALLBACK_DECIMALS
    order = math.floor(math.log10(std_y))
    return int(min(max(2 - order, _MIN_DECIMALS), _MAX_DECIMALS))


def _core_estimator(model: Any) -> Any:
    """Unwrap sklearn/imblearn pipelines down to the final estimator.

    Args:
        model: a model object (possibly a Pipeline).

    Returns:
        Any: the final pipeline step (or ``model`` itself when not a pipeline).
    """
    seen: set[int] = set()
    while hasattr(model, "named_steps"):
        if id(model) in seen:
            break
        seen.add(id(model))
        steps = model.named_steps
        try:
            values = list(steps.values())
        except AttributeError:
            break
        if not values:
            break
        model = values[-1]
    return model


def _copy_with_permuted_column(X: Any, col: int, rng: np.random.Generator) -> Any:
    """Return a copy of ``X`` with the values of column ``col`` shuffled.

    Preserves the input type (a DataFrame stays a DataFrame with the same
    columns) so that ``model.predict`` receives the expected shape. Scipy
    sparse matrices (detected by duck typing via ``toarray``) are densified
    for the subsample — ``np.array`` on a sparse matrix yields a 0-D object
    array, which would crash column permutation.

    Args:
        X: the feature matrix (ndarray, DataFrame or scipy sparse).
        col: index of the column to shuffle.
        rng: randomness generator (deterministic for a fixed seed).

    Returns:
        Any: a copy of ``X`` with column ``col`` permuted.
    """
    perm = rng.permutation(X.shape[0])
    if isinstance(X, pd.DataFrame):
        X2: Any = X.copy()
        col_name = X2.columns[col]
        X2[col_name] = X2[col_name].to_numpy()[perm]
        return X2
    if hasattr(X, "toarray"):  # scipy sparse → dense copy (subsample only)
        X = X.toarray()
    X2 = np.array(X, copy=True)
    X2[:, col] = X2[perm, col]
    return X2


class _PermutationAbort(Exception):
    """Ends the permutation check early with a circuit-C reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
