"""The four audit circuits of the Sanity Gate (A-D) and their helpers.

Each circuit is a function with explicit inputs (arrays, thresholds, the
seed) that returns ``(metric, reasons, warnings)``; ``ModelSanityGate``
calls the enabled ones and routes the results. The module depends only on
numpy, pandas, scikit-learn and :mod:`.sanity_result`; it must not import
the gate. The user-facing description of the circuits lives in
API_REFERENCE.md ("How the Sanity Gate works"); below is only what the code
relies on.

Circuit summary (defaults and the adaptive/threshold logic are in the
check functions):

A. Diversity Ratio (``check_diversity``) — ``Var(y_pred_oof) / Var(y)`` on
   out-of-fold predictions. Adaptive mode (default): the reference is the
   constant model — a numerically zero prediction spread or ``R2_oof < 0``
   fails, any visibly positive variance passes; the fixed threshold
   ``min_prediction_diversity`` applies only with
   ``adaptive_diversity=False``. A constant target skips the check with a
   soft warning.
B. Unique Ratio (``check_unique``) — predictions are rounded to the scale of
   the target (decimals derived from ``std(y)``); fails when
   ``nunique < max(min_unique_count, min_unique_ratio * N)`` with the ratio
   term capped by ``nunique(y)`` — a model predicting ``y`` exactly always
   reaches the threshold.
C. Dead Feature Check (``check_dead_features``) — the share of zero
   coefficients (linear ``coef_`` path) is a soft signal; a standalone
   disqualification requires the combination with low diversity
   (``dead_features_require_low_diversity``). Nonlinear models: permutation
   sensitivity with cost controls (``permutation_max_rows``,
   ``permutation_max_features``, ``permutation_repeats``, fixed
   ``permutation_seed``); a feature is "dead" when the relative MSE change
   under permutation stays below ``permutation_tolerance``. Correlated
   neighbours produce false-dead reports, so a high dead share is a soft
   warning, not a disqualification.
D. Generalization Gap (``check_generalization_gap``) — ``RMSE_oof /
   RMSE_full``; a finite gap above ``max_generalization_gap`` is routed by
   ``generalization_gap_mode``: ``"warn"`` (default) — ``soft_warnings``
   (including an infinite gap: a perfect in-sample fit says nothing about
   quality on new data), ``"reject"`` — a reason and disqualification.
   ``RMSE_oof ~= 0`` additionally raises a WARNING — a deliberately visible
   signal for possible target leakage.

Threshold comparison sides are fixed in one place (module-level helpers
``_fails_below`` / ``_fails_above`` / ``_diversity_fails``):

* circuits A and B fail on a strict shortfall (``value < threshold``);
* circuits C and D fail on a strict excess (``value > threshold``);
* circuit A in the adaptive mode is the only documented exception: the
  reference is the constant model whose prediction spread is exactly zero, so
  a tie with it (``diversity`` at the numerical zero tolerance
  ``_ZERO_VAR_EPS``) fails — the model is not strictly better
  than the constant baseline — and a model strictly worse than it
  (``R2_oof < 0``) fails as well (see ``_diversity_fails``).

Equality with a threshold never fails (boundary tests rely on this).

NaN handling: pairs with a non-finite true value or prediction are masked
before computing any statistic (consistent with ``metrics.oof_rmse``). If no
valid pair remains, the circuit fails with an explicit reason. Any internal
error — e.g. a model that cannot predict on a modified ``X`` — is converted
into a failure reason instead of raising.

Determinism: the permutation test uses ``np.random.default_rng`` seeded with
``permutation_seed``; row/column subsampling and column shuffles are all drawn
from this generator. Threshold-side semantics are fixed in the helpers below,
so results are reproducible across runs for the same inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from configurable_automl_engine.training_engine.sanity_result import _to_float_vector


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

    Objects exposing a ``regressor_`` attribute (fitted
    ``TransformedTargetRegressor`` wrappers — e.g. the trainer pipeline's
    ``model`` step with ``target_transform``) are unwrapped too: the linear
    path of circuit C (``coef_``) must see the inner regressor's
    coefficients, not stop silently on the wrapper. Unfitted wrappers
    (without ``regressor_``) are returned as is. Loop protection: every
    object is visited at most once (the ``seen`` id set), so a ``regressor_``
    pointing back cannot loop forever."""
    seen: set[int] = set()
    while hasattr(model, "named_steps") or hasattr(model, "regressor_"):
        if id(model) in seen:
            break
        seen.add(id(model))
        if hasattr(model, "named_steps"):
            steps = model.named_steps
            try:
                values = list(steps.values())
            except AttributeError:
                break
            if not values:
                break
            model = values[-1]
        else:
            model = model.regressor_
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


@dataclass(frozen=True)
class DeadFeatureSettings:
    """The settings of circuit C (dead features), as passed by the gate.

    Attributes:
        check_diversity: Whether circuit A runs (the combination rule needs it).
        check_permutation_sensitivity: Whether the permutation path runs for
            nonlinear models.
        zero_coef_tolerance: Coefficients with ``abs(coef) <=`` this value
            count as zero (linear path).
        max_dead_feature_ratio: The dead-feature share threshold.
        dead_features_require_low_diversity: Whether the combined
            disqualification "many dead features AND low diversity" applies.
        permutation_max_rows: The row subsampling cap of the permutation path.
        permutation_max_features: The cap on checked columns.
        permutation_repeats: The shuffle repeats per column.
        permutation_seed: The seed of the permutation generator.
        permutation_tolerance: The relative "deadness" threshold of a column.
    """

    check_diversity: bool
    check_permutation_sensitivity: bool
    zero_coef_tolerance: float
    max_dead_feature_ratio: float
    dead_features_require_low_diversity: bool
    permutation_max_rows: int
    permutation_max_features: int
    permutation_repeats: int
    permutation_seed: int
    permutation_tolerance: float


# ──────────────────────────────────────────────────────────────────────── #
#  Circuit A — Diversity Ratio (on OOF predictions)
# ──────────────────────────────────────────────────────────────────────── #
def check_diversity(
    y: np.ndarray,
    y_pred_oof: np.ndarray,
    valid: np.ndarray,
    *,
    adaptive: bool,
    min_prediction_diversity: float,
) -> tuple[float, list[str], list[str]]:
    """Compute ``Var(y_pred_oof) / Var(y)`` and check the threshold.

    Args:
        y: the target.
        y_pred_oof: OOF predictions.
        valid: mask of finite pairs.
        adaptive: whether the adaptive constant-model baseline is used
            (``adaptive_diversity``).
        min_prediction_diversity: the threshold of the non-adaptive mode.

    Returns:
        tuple[float, list[str], list[str]]: (diversity_ratio, reasons,
            soft_warnings).
    """
    if not valid.any():
        return (
            float("nan"),
            ["Circuit A (diversity): no finite (y, y_pred_oof) pairs"],
            [],
        )
    var_y = float(np.var(y[valid]))
    if var_y <= _ZERO_VAR_EPS:
        # Constant target: the ratio is undefined (division by zero) —
        # NaN is reported and circuit B takes over.
        return (
            float("nan"),
            [],
            [
                (
                    "Circuit A (diversity): target variance is zero — check "
                    "skipped (unique-ratio circuit applies)"
                )
            ],
        )
    diversity = float(np.var(y_pred_oof[valid]) / var_y)
    # In the adaptive mode the fraction of explained variance
    # (R²_oof ≈ diversity for an honest model) is the decisive signal:
    # a model worse than the constant prediction is rejected explicitly.
    r2_oof = _r2_oof(y[valid], y_pred_oof[valid]) if adaptive else 0.0
    threshold = 0.0 if adaptive else min_prediction_diversity
    if not _diversity_fails(diversity, threshold, adaptive, r2_oof):
        return diversity, [], []
    if adaptive:
        if np.isfinite(r2_oof) and r2_oof < 0.0:
            reason = (
                f"Circuit A (diversity): R²_oof={r2_oof:.4f} < 0 — "
                "model is worse than the constant prediction"
            )
        else:
            reason = (
                "Circuit A (diversity): ratio=0.0000 equals the "
                "constant-model baseline — no expressiveness"
            )
    else:
        reason = (
            f"Circuit A (diversity): ratio={diversity:.4f} < "
            f"min_prediction_diversity={threshold:.4f}"
        )
    return diversity, [reason], []


# ──────────────────────────────────────────────────────────────────────── #
#  Circuit B — Unique Ratio (rounding from std(y))
# ──────────────────────────────────────────────────────────────────────── #
def check_unique(
    y: np.ndarray,
    y_pred_oof: np.ndarray,
    valid: np.ndarray,
    *,
    min_unique_count: int,
    min_unique_ratio: float,
) -> tuple[float, list[str], list[str]]:
    """Compute the share of unique predictions and check the lower bound.

    Predictions are rounded relative to the scale of the **target**
    (``std(y)``), not of the predictions — fixed digits are inapplicable to
    differently scaled and discrete targets.

    The required number of distinct predictions is
    ``max(min_unique_count, min_unique_ratio * N)`` with the ratio term
    capped by ``nunique(y)`` — the number of distinct target values on the
    valid rows at the same rounding granularity that is applied to the
    predictions (so a model predicting y exactly always reaches the cap).
    Without the cap a high ``min_unique_ratio`` on a low-cardinality
    target demands more distinct predictions than the most expressive
    model (one predicting y exactly) can produce: e.g. 18 rows with 13
    distinct y values and ``min_unique_ratio=0.9`` require 16.2 ≥ 17
    distinct predictions — even a perfect model would be disqualified.
    The cap makes the ceiling equal to the expressing power
    of a perfect model; the final threshold is additionally never allowed
    to exceed the cap even when ``min_unique_count`` is set above it.

    Args:
        y: the target (source of the rounding scale).
        y_pred_oof: OOF predictions.
        valid: mask of finite pairs.
        min_unique_count: the absolute lower bound for nunique.
        min_unique_ratio: the lower bound for the share of unique predictions.

    Returns:
        tuple[float, list[str], list[str]]: (unique_ratio, reasons,
            soft_warnings).
    """
    n_valid = int(valid.sum())
    if n_valid == 0:
        return (
            float("nan"),
            ["Circuit B (unique): no finite (y, y_pred_oof) pairs"],
            [],
        )
    std_y = float(np.std(y[valid]))
    decimals = _rounding_decimals(std_y)
    rounded = np.round(y_pred_oof[valid], decimals=decimals)
    nunique = len(np.unique(rounded))
    unique_ratio = nunique / n_valid
    # Distinct target values on the valid rows at the same rounding
    # granularity that is applied to the predictions — the natural
    # ceiling for how many distinct predictions the most expressive
    # model (one predicting y exactly) can produce.
    n_unique_y = len(np.unique(np.round(y[valid], decimals=decimals)))
    ratio_bound = min(min_unique_ratio * n_valid, n_unique_y)
    required = min(max(min_unique_count, ratio_bound), n_unique_y)
    if _fails_below(nunique, required):
        return (
            unique_ratio,
            [
                (
                    f"Circuit B (unique): nunique={nunique} < "
                    f"max(min_unique_count={min_unique_count}, "
                    f"min_unique_ratio*N={min_unique_ratio * n_valid:.3f} "
                    f"capped by nunique(y)={n_unique_y} -> "
                    f"{ratio_bound:.3f}) = {required:.3f}"
                )
            ],
            [],
        )
    return unique_ratio, [], []


# ──────────────────────────────────────────────────────────────────────── #
#  Circuit C — Dead Feature Check
# ──────────────────────────────────────────────────────────────────────── #
def check_dead_features(
    *,
    model: Any,
    X: Any | None,
    y: np.ndarray,
    diversity_failed: bool,
    settings: DeadFeatureSettings,
) -> tuple[int, list[str], list[str]]:
    """Check dead features: the linear or the permutation path.

    Args:
        model: the fitted model (may be a pipeline).
        X: the feature matrix (needed by the permutation path).
        y: the target.
        diversity_failed: whether circuit A failed (for the combined
            disqualification).
        settings: the circuit-C settings of the gate.

    Returns:
        tuple[int, list[str], list[str]]: (dead_features_count, reasons,
            soft_warnings).
    """
    if model is None:
        return (
            0,
            [
                (
                    "Circuit C (dead features): model is not provided "
                    "(check_dead_features=True)"
                )
            ],
            [],
        )
    estimator = _core_estimator(model)
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        return _check_dead_features_linear(coef, diversity_failed, settings)
    if not settings.check_permutation_sensitivity:
        # The permutation path is disabled by the user: for nonlinear
        # models circuit C is idle — an intentional cost decision.
        return 0, [], []
    if X is None:
        return (
            0,
            [
                (
                    "Circuit C (dead features): X is not provided — permutation "
                    "check impossible for a nonlinear model"
                )
            ],
            [],
        )
    return _check_dead_features_permutation(
        model=model, X=X, y=y, diversity_failed=diversity_failed, settings=settings
    )


def _check_dead_features_linear(
    coef: Any, diversity_failed: bool, settings: DeadFeatureSettings
) -> tuple[int, list[str], list[str]]:
    """Linear path: zero-coefficient share is a signal + a combination."""
    coef_arr = np.asarray(coef, dtype=float).ravel()
    if coef_arr.size == 0:
        return 0, ["Circuit C (dead features): model exposes an empty coef_"], []
    zero_count = int(np.sum(np.abs(coef_arr) <= settings.zero_coef_tolerance))
    zero_ratio = zero_count / coef_arr.size
    reasons: list[str] = []
    warnings: list[str] = []
    if _fails_above(zero_ratio, settings.max_dead_feature_ratio):
        warnings.append(
            f"Circuit C (dead features): zero-coefficient ratio="
            f"{zero_ratio:.3f} > max_dead_feature_ratio="
            f"{settings.max_dead_feature_ratio:.3f} (signal — on wide/noisy "
            f"data Lasso-family models legitimately zero out features)"
        )
        if settings.dead_features_require_low_diversity:
            if diversity_failed:
                reasons.append(
                    f"Circuit C (dead features): combination — "
                    f"zero-coefficient ratio={zero_ratio:.3f} above "
                    f"max_dead_feature_ratio AND prediction diversity "
                    f"below its threshold"
                )
            elif not settings.check_diversity:
                # The combination cannot be computed: circuit A is
                # disabled — explicitly report that the rule was not
                # applied (not silently).
                warnings.append(
                    "Circuit C (dead features): combination rule inactive — "
                    "check_diversity=False (dead-feature share is a signal "
                    "only)"
                )
    return zero_count, reasons, warnings


def _check_dead_features_permutation(
    *,
    model: Any,
    X: Any,
    y: np.ndarray,
    diversity_failed: bool,
    settings: DeadFeatureSettings,
) -> tuple[int, list[str], list[str]]:
    """Permutation path for nonlinear models (with cost controls).

    A feature is "dead" when the relative change of MSE under permutation
    of its column stays below ``permutation_tolerance``. Row subsampling
    and the column cap bound the cost; base predictions are cached; the
    seed is fixed (determinism). All ΔMSE repeats are computed on the same
    fixed set of valid rows (derived from the base predictions), so their
    average is well-defined even when a repeat returns non-finite values
    for some rows (such a repeat is skipped instead of changing the mask).

    Any unexpected error inside the permutation machinery (sparse or exotic
    ``X``, failing row subsampling, etc.) is converted into a circuit-C
    reason instead of escaping from ``check()``."""
    try:
        return _permutation_impl(
            model=model, X=X, y=y, diversity_failed=diversity_failed, settings=settings
        )
    except Exception as err:  # noqa: BLE001 — convert into a reason
        return (
            0,
            [f"Circuit C (dead features): permutation check failed: {err}"],
            [],
        )


def _permutation_impl(
    *,
    model: Any,
    X: Any,
    y: np.ndarray,
    diversity_failed: bool,
    settings: DeadFeatureSettings,
) -> tuple[int, list[str], list[str]]:
    """Core permutation check; errors are caught by the caller wrapper.

    All random draws come from one generator seeded with
    ``permutation_seed``, in this order: the row subsample, the column
    subsample, then the permutations column by column and repeat by
    repeat. A step that cannot continue raises :class:`_PermutationAbort`
    with its circuit-C reason.
    """
    rng = np.random.default_rng(settings.permutation_seed)
    try:
        X_sub, y_sub, col_idx = _permutation_subsample(X, y, rng, settings)
        base_valid, base_mse = _base_prediction(model, X_sub, y_sub)
        dead_count = 0
        for col in col_idx:
            if _is_dead_column(
                model, X_sub, y_sub, int(col), rng, base_valid, base_mse, settings
            ):
                dead_count += 1
    except _PermutationAbort as abort:
        return 0, [abort.reason], []
    return _dead_features_verdict(
        dead_count, int(col_idx.size), diversity_failed, settings
    )


def _permutation_subsample(
    X: Any, y: np.ndarray, rng: np.random.Generator, settings: DeadFeatureSettings
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Draw the row subsample, then the (sorted) column subsample.

    Raises:
        _PermutationAbort: when the rows cannot be selected or ``X`` is
            not a 2-D matrix with at least one column.
    """
    n_rows = y.shape[0]
    row_idx = np.arange(n_rows)
    if n_rows > settings.permutation_max_rows:
        row_idx = rng.choice(n_rows, size=settings.permutation_max_rows, replace=False)
    try:
        X_sub = X.iloc[row_idx] if isinstance(X, pd.DataFrame) else X[row_idx]
    except Exception as err:  # converted into a reason by the caller
        raise _PermutationAbort(
            f"Circuit C (dead features): row subsampling of X failed: {err}"
        ) from err
    y_sub = y[row_idx]

    if X_sub.ndim != 2 or X_sub.shape[1] == 0:
        raise _PermutationAbort(
            "Circuit C (dead features): X must be a 2-D matrix with at least one column"
        )
    n_cols = X_sub.shape[1]
    col_idx = np.arange(n_cols)
    if n_cols > settings.permutation_max_features:
        col_idx = rng.choice(
            n_cols, size=settings.permutation_max_features, replace=False
        )
    return X_sub, y_sub, np.sort(col_idx)


def _base_prediction(
    model: Any, X_sub: Any, y_sub: np.ndarray
) -> tuple[np.ndarray, float]:
    """Predict on the subsample once (cached base for every column)."""
    try:
        base_pred = np.asarray(model.predict(X_sub), dtype=float).reshape(-1)
    except Exception as err:  # converted into a reason by the caller
        raise _PermutationAbort(
            f"Circuit C (dead features): prediction failed on subsampled X: {err}"
        ) from err

    if base_pred.shape != y_sub.shape:
        raise _PermutationAbort(
            f"Circuit C (dead features): prediction returned "
            f"{base_pred.shape[0]} values for {y_sub.shape[0]} rows"
        )
    base_valid = np.isfinite(y_sub) & np.isfinite(base_pred)
    if not base_valid.any():
        raise _PermutationAbort(
            "Circuit C (dead features): no finite (y, prediction) pairs "
            "on the permutation subsample"
        )
    base_mse = float(mean_squared_error(y_sub[base_valid], base_pred[base_valid]))
    return base_valid, base_mse


def _is_dead_column(
    model: Any,
    X_sub: Any,
    y_sub: np.ndarray,
    col: int,
    rng: np.random.Generator,
    base_valid: np.ndarray,
    base_mse: float,
    settings: DeadFeatureSettings,
) -> bool:
    """Whether permuting column ``col`` leaves the MSE (almost) unchanged.

    Args:
        model, X_sub, y_sub: the fitted model and the subsample.
        col: the column index.
        rng: the permutation generator.
        base_valid, base_mse: the fixed mask of valid pairs and the base
            MSE on it.
        settings: the circuit-C settings (repeats and tolerance).


    Returns:
        bool: True when the relative ΔMSE (mean over the usable repeats)
            is below ``permutation_tolerance``; False when no repeat is
            usable.
    """
    deltas: list[float] = []
    for _ in range(settings.permutation_repeats):
        delta = _column_delta(model, X_sub, y_sub, col, rng, base_valid, base_mse)
        if delta is not None:
            deltas.append(delta)
    if not deltas:
        return False
    mean_delta = float(np.mean(deltas))
    # Relative change of MSE; the _MSE_EPS stabilizer keeps the ratio
    # well-defined for a perfect in-sample fit (base_mse == 0): in that
    # case only an exactly unchanged prediction counts as "dead".
    rel_change = abs(mean_delta) / (abs(base_mse) + _MSE_EPS)
    return bool(rel_change < settings.permutation_tolerance)


def _column_delta(
    model: Any,
    X_sub: Any,
    y_sub: np.ndarray,
    col: int,
    rng: np.random.Generator,
    base_valid: np.ndarray,
    base_mse: float,
) -> float | None:
    """ΔMSE of one permutation repeat of column ``col``."""
    try:
        X_perm = _copy_with_permuted_column(X_sub, col, rng)
    except Exception as err:  # converted into a reason by the caller
        raise _PermutationAbort(
            f"Circuit C (dead features): column permutation "
            f"failed (column {int(col)}): {err}"
        ) from err
    try:
        perm_pred = np.asarray(model.predict(X_perm), dtype=float).reshape(-1)
    except Exception as err:  # converted into a reason by the caller
        raise _PermutationAbort(
            f"Circuit C (dead features): prediction failed on "
            f"permuted X (column {int(col)}): {err}"
        ) from err
    # Fixed valid-pair mask (from the base predictions): every
    # repeat of the same column is evaluated on the same rows, so
    # the mean ΔMSE is not biased by varying masks. A repeat that
    # returns non-finite values for a base-valid row is skipped
    # rather than silently switching to a different row set.
    if perm_pred.shape != y_sub.shape or not np.all(np.isfinite(perm_pred[base_valid])):
        return None
    return (
        float(mean_squared_error(y_sub[base_valid], perm_pred[base_valid])) - base_mse
    )


def _dead_features_verdict(
    dead_count: int,
    n_checked: int,
    diversity_failed: bool,
    settings: DeadFeatureSettings,
) -> tuple[int, list[str], list[str]]:
    """Turn the dead-column count into the circuit-C result."""
    reasons: list[str] = []
    warnings: list[str] = []
    dead_ratio = dead_count / n_checked if n_checked else 0.0
    if not _fails_above(dead_ratio, settings.max_dead_feature_ratio):
        return dead_count, reasons, warnings
    warnings.append(
        f"Circuit C (dead features): dead-feature ratio="
        f"{dead_ratio:.3f} > max_dead_feature_ratio="
        f"{settings.max_dead_feature_ratio:.3f} on permutation check "
        f"({dead_count}/{n_checked} checked columns; note: correlated "
        f"features may produce 'false-dead' results)"
    )
    if settings.dead_features_require_low_diversity:
        if diversity_failed:
            reasons.append(
                "Circuit C (dead features): combination — high "
                "dead-feature ratio AND prediction diversity below its "
                "threshold"
            )
        elif not settings.check_diversity:
            # The combination cannot be computed: circuit A is
            # disabled — explicitly report that the rule was not
            # applied (not silently).
            warnings.append(
                "Circuit C (dead features): combination rule inactive — "
                "check_diversity=False (dead-feature share is a signal "
                "only)"
            )
    return dead_count, reasons, warnings


# ──────────────────────────────────────────────────────────────────────── #
#  Circuit D — Generalization Gap
# ──────────────────────────────────────────────────────────────────────── #
def check_generalization_gap(
    y: np.ndarray,
    y_pred_oof: np.ndarray,
    y_pred_full: Any | None,
    *,
    max_generalization_gap: float,
) -> tuple[float, list[str], list[str]]:
    """Check ``RMSE_oof / RMSE_full`` against ``max_generalization_gap``.

            Both RMSEs are computed on the **same** joint mask of finite pairs
            (y, y_pred_oof, y_pred_full), so the ratio compares apples to apples.

            Returns three lists separately so that ``check()`` can route them by
    Returns three lists separately so that ``check()`` can route them by
            ``generalization_gap_mode``: a gap violation — finite
            (strictly above the threshold) or infinite (division by zero:
            ``RMSE_full ≈ 0`` perfect in-sample fit, ``RMSE_oof ≈ 0``) — is
            returned as a ``warnings`` entry, which ``check()`` turns into a
            reason in ``"reject"`` mode or into ``soft_warnings`` in ``"warn"``
            mode (in the warn mode the circuit never disqualifies because
            of the gap magnitude). Contract violations (``y_pred_full = None``)
            and non-computability (no finite pairs) are returned as ``reasons``
            in **both** modes. ``RMSE_oof ≈ 0`` is the only case additionally
            logged at ``WARNING`` level in warn mode (possible target leakage).

            Args:
                y: the target.
                y_pred_oof: OOF predictions.
                y_pred_full: full-fit predictions (None → fail with a reason).
                max_generalization_gap: the circuit-D threshold.

            Returns:
                tuple[float, list[str], list[str]]: (generalization_gap, reasons,
                    warnings).
    """
    if y_pred_full is None:
        return (
            float("nan"),
            ["Circuit D (generalization gap): y_pred_full is not provided"],
            [],
        )
    full_arr = _to_float_vector(y_pred_full, "y_pred_full")
    if full_arr.shape != y.shape:
        raise ValueError(
            f"y and y_pred_full must have the same length "
            f"(got {y.shape} and {full_arr.shape})"
        )
    mask = np.isfinite(y) & np.isfinite(y_pred_oof) & np.isfinite(full_arr)
    if not mask.any():
        return (
            float("nan"),
            ["Circuit D (generalization gap): no finite pairs to compute RMSE"],
            [],
        )
    rmse_full = float(np.sqrt(mean_squared_error(y[mask], full_arr[mask])))
    rmse_oof = float(np.sqrt(mean_squared_error(y[mask], y_pred_oof[mask])))
    if rmse_oof <= _ZERO_RMSE_EPS:
        # RMSE_oof ≈ 0: either a perfect model or a target leakage.
        # In warn mode — a soft warning with an explicit leakage
        # suspicion (additionally a WARNING in the log, see check());
        # in reject mode — a reason.
        return (
            float("inf"),
            [],
            [
                (
                    "Circuit D (generalization gap): RMSE_oof≈0 — "
                    "suspiciously perfect out-of-fold fit, possible "
                    "target leakage (division by zero)"
                )
            ],
        )
    if rmse_full <= _ZERO_RMSE_EPS:
        # A perfect train fit (a depth-20 tree, 1-NN,
        # ExtraTrees with min_samples_leaf=1, ...) — a normal property of
        # a whole class of models; the gap goes to infinity, but this
        # says nothing about the quality on new data.
        return (
            float("inf"),
            [],
            [
                (
                    "Circuit D (generalization gap): RMSE_full≈0 (perfect "
                    "in-sample fit) — division by zero"
                )
            ],
        )
    gap = rmse_oof / rmse_full
    if _fails_above(gap, max_generalization_gap):
        return (
            gap,
            [],
            [
                (
                    f"Circuit D (generalization gap): gap={gap:.3f} > "
                    f"max_generalization_gap={max_generalization_gap:.3f}"
                )
            ],
        )
    return gap, [], []
