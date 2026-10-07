"""ModelSanityGate: module for auditing model degeneracy (epic #61, task T2).

Protects the final winner selection of the AutoML engine from "degenerate"
models — constant/flat predictions, step functions, unused features and
overfit learners. The module is standalone: it depends only on numpy, pandas
and scikit-learn and knows nothing about the rest of the engine.

Four independent check circuits are implemented; each can be disabled through
a constructor parameter:

A. Diversity Ratio (``check_diversity``)
   ``Var(y_pred_oof) / Var(y)`` computed on **out-of-fold** predictions, not
   on a full fit. Fails when the ratio is strictly below
   ``min_prediction_diversity`` (default 0.15 — deliberately soft: on noisy
   data an honest model with R2 ≈ 0.3 yields a ratio of ≈ 0.3 by definition
   and must not be disqualified). A constant target (``Var(y) == 0``) skips
   the check with a soft warning — the unique-ratio circuit takes over.
   Optional adaptive mode (``adaptive_diversity``) compares against the
   constant-model baseline: any model with zero prediction variance ties the
   constant model and fails, any strictly positive variance passes.

B. Unique Ratio (``check_unique``)
   Predictions are rounded to the scale of the target: the number of decimals
   is derived from ``std(y)`` (not a fixed 3 digits, which depend on the
   scale of y and break for discrete targets). Fails when
   ``nunique < max(min_unique_count, min_unique_ratio * N)``.

C. Dead Feature Check (``check_dead_features``)
   * Linear models (Lasso, ElasticNet, SGDRegressor, ... — anything exposing
     ``coef_``): the share of zero coefficients is a **signal**
     (``soft_warnings``), never a standalone disqualification — on wide noisy
     data Lasso is *supposed* to zero out most features. Standalone
     disqualification happens only in combination: share of zeros above
     ``max_dead_feature_ratio`` **AND** prediction diversity below the
     diversity threshold (rule controlled by
     ``dead_features_require_low_diversity``, default True).
   * Nonlinear models (SVR, trees, ...): permutation sensitivity with cost
     controls — ``permutation_max_rows`` (row subsampling cap),
     ``permutation_max_features`` (cap on checked columns),
     ``permutation_repeats`` (repeat count) and a fixed ``permutation_seed``
     for full determinism. Base predictions are cached; a feature is "dead"
     when the relative change of MSE under permutation stays below
     ``permutation_tolerance``. A high share of dead features produces a soft
     warning, not a disqualification: on correlated features permutation
     reports "false-dead" features because the model borrows the information
     from a neighbour column. The whole permutation path can be turned off
     with ``check_permutation_sensitivity=False`` (cost control, task T5);
     the linear ``coef_`` path is unaffected.

D. Generalization Gap (``check_generalization_gap``)
   ``RMSE_oof / RMSE_full`` (the formula written as ``RMSE_full / RMSE_oof``
   in the issue is equivalent to requiring this ratio below
   ``1 / max_generalization_gap``; the stored value grows with overfitting).
   Fails when the gap strictly exceeds ``max_generalization_gap``
   (default 1.5). A near-zero denominator — ``RMSE_full ≈ 0`` (perfect
   in-sample fit such as a depth-20 tree or 1-NN) or ``RMSE_oof ≈ 0`` — is
   treated as a failure with an explicit reason (division by zero).

Threshold comparison sides are fixed in one place (module-level helpers
``_fails_below`` / ``_fails_above`` / ``_diversity_fails``):

* circuits A and B fail on a strict shortfall (``value < threshold``);
* circuits C and D fail on a strict excess (``value > threshold``);
* circuit A in the adaptive mode is the only documented exception: the
  reference is the constant model whose prediction spread is exactly zero, so
  a tie with it (``diversity <= 0``) fails — the model is not strictly better
  than the constant baseline (see ``_diversity_fails``).

Equality with a threshold never fails (boundary tests rely on this).

NaN handling: pairs with a non-finite true value or prediction are masked
before computing any statistic (consistent with ``metrics.oof_rmse``). If no
valid pair remains, the circuit fails with an explicit reason. Any internal
error — e.g. a model that cannot predict on a modified ``X`` — is converted
into a failure reason instead of raising.

``warn_only`` mode: the gate still computes every statistic and fills
``reasons`` with the would-be disqualifications, but ``is_valid`` is forced to
True and ``severity`` is capped at the "warning" level — the upper layer
(task T4) decides whether to apply the verdict to winner selection.

Determinism: the permutation test uses ``np.random.default_rng`` seeded with
``permutation_seed``; row/column subsampling and column shuffles are all drawn
from this generator. Threshold-side semantics are fixed in the helpers below,
so results are reproducible across runs for the same inputs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

__all__ = ["ModelSanityGate", "SanityCheckResult"]


# --------------------------------------------------------------------------- #
#  Пороговые сравнения: ЕДИНСТВЕННАЯ точка фиксации стороны (< или ≤)
# --------------------------------------------------------------------------- #
# Контуры А/Б проваливаются при СТРОГОМ недоборе до порога; контуры В/Г —
# при СТРОГОМ превышении. Равенство порогу никогда не является провалом.
# Настройка другого поведения (≤ вместо <) разрешена только здесь.
def _fails_below(value: float, threshold: float) -> bool:
    """Fail on a strict shortfall: ``value < threshold`` (equality passes)."""
    return value < threshold


def _fails_above(value: float, threshold: float) -> bool:
    """Fail on a strict excess: ``value > threshold`` (equality passes)."""
    return value > threshold


def _diversity_fails(diversity: float, threshold: float, adaptive: bool) -> bool:
    """Fail side of circuit A, fixed in one place (module level).

    Non-adaptive mode: strict shortfall — ``diversity < threshold``, equality
    with the threshold passes. Adaptive mode is the only documented exception
    to the strict-``<`` convention: the reference is the constant model whose
    prediction spread is exactly zero, so a tie with it (``diversity <= 0``)
    fails — the model is not strictly better than the constant baseline.

    Args:
        diversity: ``Var(y_pred_oof) / Var(y)``.
        threshold: effective threshold (``min_prediction_diversity``).
        adaptive: whether the adaptive constant-model baseline is used.

    Returns:
        bool: True when circuit A must fail for the given values.
    """
    if adaptive:
        return diversity <= 0.0
    return diversity < threshold


# Численные допуски и константы округления.
_ZERO_VAR_EPS = 1e-12  # дисперсия таргета ниже этого — «константный y»
_ZERO_RMSE_EPS = 1e-12  # RMSE ниже этого считается нулевым (деление на ноль)
_MSE_EPS = 1e-12  # стабилизатор знаменателя в относительном ΔMSE
_FALLBACK_DECIMALS = 6  # precision округления при std(y)==0
_MIN_DECIMALS = -12
_MAX_DECIMALS = 10

_SEVERITY_CLEAN = 0
_SEVERITY_WARNING = 1


@dataclass(frozen=True)
class SanityCheckResult:
    """Result of a model degeneracy audit.

    Attributes:
        is_valid: True when no circuit disqualified the model. In ``warn_only``
            mode it is always True (the gate does not disqualify).
        reasons: disqualification reasons (each a string naming the circuit).
        soft_warnings: signals for the report — they never disqualify alone.
        diversity_ratio: ``Var(y_pred_oof) / Var(y)`` (circuit A); NaN when the
            circuit is disabled or not computable.
        unique_ratio: share of distinct (post-rounding) predictions
            ``nunique / N`` (circuit B); NaN when the circuit is disabled.
        dead_features_count: number of "dead" features (circuit C): zero
            coefficients for linear models or permutation-insensitive columns
            for nonlinear ones; 0 when the circuit is disabled.
        generalization_gap: ``RMSE_oof / RMSE_full`` (circuit D); NaN when the
            circuit is disabled or not computable.
        severity: formalized severity for the fallback (T4):
            0 — clean; 1 — soft warnings only;
            2/3/4 — one/two/three or more disqualification reasons.
            In ``warn_only`` mode it never exceeds 1.
    """

    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    diversity_ratio: float = float("nan")
    unique_ratio: float = float("nan")
    dead_features_count: int = 0
    generalization_gap: float = float("nan")
    severity: int = _SEVERITY_CLEAN


def _compute_severity(
    reasons: list[str], soft_warnings: list[str], warn_only: bool
) -> int:
    """Map reasons/warnings to a formalized severity level (T4 fallback).

    Args:
        reasons: disqualification reasons.
        soft_warnings: soft signals.
        warn_only: True — the gate does not disqualify (severity ≤ 1).

    Returns:
        int: 0 — clean; 1 — warnings only; 2/3/4 — one/two/three+ reasons.
            Not higher than 1 in ``warn_only`` mode.
    """
    if reasons:
        severity = min(2 + len(reasons) - 1, 4)
    elif soft_warnings:
        severity = _SEVERITY_WARNING
    else:
        severity = _SEVERITY_CLEAN
    return min(severity, _SEVERITY_WARNING) if warn_only else severity


def _to_float_vector(values: Any, name: str) -> np.ndarray:
    """Convert an input to a 1-D float array (with shape validation).

    Args:
        values: any array-like (numpy, pandas, list).
        name: argument name used in error messages.

    Returns:
        np.ndarray: a 1-D float array.

    Raises:
        ValueError: if ``values`` is None, is not 1-D after collapsing
            (n, 1)/(1, n), or cannot be converted to float.
    """
    if values is None:
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D vector (got shape {arr.shape}); "
            "pass y / y_pred_oof / y_pred_full as flat arrays or Series."
        )
    return arr


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


class ModelSanityGate:
    """Audit a model for degeneracy through four optional circuits (epic #61).

    Each circuit is optional and disabled with a ``check_*`` flag. Errors and
    NaN are treated as a failure of the corresponding circuit with a clear
    reason (except partial NaN in predictions — such pairs are masked, as in
    ``metrics.oof_rmse``). Default thresholds are deliberately soft (v2
    statement), so honest models on noisy/wide data are not disqualified.
    """

    def __init__(
        self,
        *,
        min_prediction_diversity: float = 0.15,
        adaptive_diversity: bool = False,
        min_unique_count: int = 5,
        min_unique_ratio: float = 0.05,
        max_dead_feature_ratio: float = 0.4,
        dead_features_require_low_diversity: bool = True,
        zero_coef_tolerance: float = 1e-12,
        max_generalization_gap: float = 1.5,
        check_permutation_sensitivity: bool = True,
        permutation_max_rows: int = 5000,
        permutation_max_features: int = 100,
        permutation_repeats: int = 3,
        permutation_seed: int = 42,
        permutation_tolerance: float = 1e-3,
        check_diversity: bool = True,
        check_unique: bool = True,
        check_dead_features: bool = True,
        check_generalization_gap: bool = True,
        warn_only: bool = False,
    ) -> None:
        """Initialize the gate with thresholds and circuit flags.

        Args:
            min_prediction_diversity: circuit A threshold, in (0, 1]
                (default 0.15 — soft).
            adaptive_diversity: adaptive mode A relative to the constant model:
                fails only for a zero prediction spread (the absolute threshold
                is not applied).
            min_unique_count: absolute lower bound for nunique (circuit B).
            min_unique_ratio: lower bound for the share of unique predictions,
                in (0, 1].
            max_dead_feature_ratio: dead-feature share threshold (circuit C):
                above it — a soft signal, and combined with a circuit A failure
                (when ``dead_features_require_low_diversity``) — disqualification.
            dead_features_require_low_diversity: allow the combined
                disqualification "many dead features AND low diversity".
            zero_coef_tolerance: coefficients with ``abs(coef) <=`` this value
                count as zero (linear path of circuit C).
            max_generalization_gap: circuit D threshold (≥ 1, default 1.5).
            check_permutation_sensitivity: enable the permutation sensitivity
                audit of circuit C for nonlinear models (default True — the
                cost-controlled permutation path runs; False — the path is
                skipped and only the linear ``coef_`` path of circuit C works).
            permutation_max_rows: row subsampling for permutations (>= 1,
                default 5000).
            permutation_max_features: cap on checked columns (>= 1,
                default 100).
            permutation_repeats: shuffle repeats per column.
            permutation_seed: fixed seed for permutation determinism.
            permutation_tolerance: relative "deadness" threshold of a feature:
                ``|ΔMSE| / (|mse_base| + eps) < tolerance``.
            check_diversity: enable circuit A.
            check_unique: enable circuit B.
            check_dead_features: enable circuit C.
            check_generalization_gap: enable circuit D.
            warn_only: mode without the right to disqualify (statistics are
                computed, ``is_valid`` is always True).

        Raises:
            ValueError: for invalid thresholds or parameter combinations.
        """
        if not 0 < min_prediction_diversity <= 1:
            raise ValueError("min_prediction_diversity must be in (0, 1]")
        if min_unique_count < 1:
            raise ValueError("min_unique_count must be >= 1")
        if not 0 < min_unique_ratio <= 1:
            raise ValueError("min_unique_ratio must be in (0, 1]")
        if not 0 <= max_dead_feature_ratio <= 1:
            raise ValueError("max_dead_feature_ratio must be in [0, 1]")
        if max_generalization_gap < 1:
            raise ValueError("max_generalization_gap must be >= 1")
        if zero_coef_tolerance < 0:
            raise ValueError("zero_coef_tolerance must be >= 0")
        if permutation_repeats < 1:
            raise ValueError("permutation_repeats must be >= 1")
        if permutation_max_rows < 1:
            raise ValueError("permutation_max_rows must be >= 1")
        if permutation_max_features < 1:
            raise ValueError("permutation_max_features must be >= 1")
        if permutation_tolerance < 0:
            raise ValueError("permutation_tolerance must be >= 0")

        self.min_prediction_diversity = min_prediction_diversity
        self.adaptive_diversity = adaptive_diversity
        self.min_unique_count = min_unique_count
        self.min_unique_ratio = min_unique_ratio
        self.max_dead_feature_ratio = max_dead_feature_ratio
        self.dead_features_require_low_diversity = dead_features_require_low_diversity
        self.zero_coef_tolerance = zero_coef_tolerance
        self.max_generalization_gap = max_generalization_gap
        self.check_permutation_sensitivity = check_permutation_sensitivity
        self.permutation_max_rows = permutation_max_rows
        self.permutation_max_features = permutation_max_features
        self.permutation_repeats = permutation_repeats
        self.permutation_seed = permutation_seed
        self.permutation_tolerance = permutation_tolerance
        self.check_diversity = check_diversity
        self.check_unique = check_unique
        self.check_dead_features = check_dead_features
        self.check_generalization_gap = check_generalization_gap
        self.warn_only = warn_only

    # ──────────────────────────────────────────────────────────────────────── #
    #  Public API
    # ──────────────────────────────────────────────────────────────────────── #
    def check(
        self,
        *,
        y: Any,
        y_pred_oof: Any,
        y_pred_full: Any | None = None,
        X: Any | None = None,
        model: Any = None,
    ) -> SanityCheckResult:
        """Run the model through every enabled audit circuit.

        Args:
            y: the true target (1-D; may contain NaN — pairs are masked).
            y_pred_oof: OOF predictions aligned with ``y`` by rows (NaN for
                uncovered rows is masked).
            y_pred_full: full-fit predictions on the same rows (needed by
                circuit D); None with circuit D enabled → fail with a reason.
            X: feature matrix in the form accepted by ``model.predict`` (needed
                by the permutation path of circuit C for nonlinear models).
            model: the fitted model (needed by circuit C); pipelines are
                unwrapped to the final estimator.

        Returns:
            SanityCheckResult: the full audit result.

        Raises:
            ValueError: when the input contract is violated (None arguments,
                length mismatch between y/y_pred_oof/X).
        """
        y_arr = _to_float_vector(y, "y")
        oof_arr = _to_float_vector(y_pred_oof, "y_pred_oof")
        if y_arr.shape != oof_arr.shape:
            raise ValueError(
                f"y and y_pred_oof must have the same length "
                f"(got {y_arr.shape} and {oof_arr.shape})"
            )
        if X is not None:
            if not hasattr(X, "shape"):
                raise ValueError(
                    f"X must be array-like with a .shape attribute "
                    f"(ndarray, pandas.DataFrame or scipy sparse), got "
                    f"{type(X).__name__}"
                )
            x_len = X.shape[0]
            if x_len != y_arr.shape[0]:
                raise ValueError(
                    f"X and y must have the same number of rows "
                    f"(got {x_len} and {y_arr.shape[0]})"
                )

        reasons: list[str] = []
        soft_warnings: list[str] = []
        diversity_ratio = float("nan")
        unique_ratio = float("nan")
        dead_features_count = 0
        generalization_gap = float("nan")
        diversity_failed = False

        # Общая маска конечных пар для контуров, работающих с предсказаниями.
        valid_pairs = np.isfinite(y_arr) & np.isfinite(oof_arr)

        if self.check_diversity:
            diversity_ratio, d_reasons, d_warnings = self._check_diversity(
                y_arr, oof_arr, valid_pairs
            )
            reasons.extend(d_reasons)
            soft_warnings.extend(d_warnings)
            diversity_failed = bool(d_reasons)

        if self.check_unique:
            unique_ratio, u_reasons, u_warnings = self._check_unique(
                y_arr, oof_arr, valid_pairs
            )
            reasons.extend(u_reasons)
            soft_warnings.extend(u_warnings)

        if self.check_dead_features:
            dead_features_count, c_reasons, c_warnings = self._check_dead_features(
                model=model,
                X=X,
                y=y_arr,
                diversity_failed=diversity_failed,
            )
            reasons.extend(c_reasons)
            soft_warnings.extend(c_warnings)

        if self.check_generalization_gap:
            generalization_gap, g_reasons = self._check_generalization_gap(
                y_arr, oof_arr, y_pred_full
            )
            reasons.extend(g_reasons)

        is_valid = not reasons
        if self.warn_only:
            is_valid = True
        severity = _compute_severity(reasons, soft_warnings, self.warn_only)

        if reasons:
            logger.info(
                "Sanity gate%s: %d reason(s): %s",
                " (warn_only)" if self.warn_only else "",
                len(reasons),
                reasons,
            )
        elif soft_warnings:
            logger.info(
                "Sanity gate: valid with %d soft warning(s)", len(soft_warnings)
            )

        return SanityCheckResult(
            is_valid=is_valid,
            reasons=reasons,
            soft_warnings=soft_warnings,
            diversity_ratio=diversity_ratio,
            unique_ratio=unique_ratio,
            dead_features_count=dead_features_count,
            generalization_gap=generalization_gap,
            severity=severity,
        )

    # ──────────────────────────────────────────────────────────────────────── #
    #  Circuit A — Diversity Ratio (on OOF predictions)
    # ──────────────────────────────────────────────────────────────────────── #
    def _check_diversity(
        self, y: np.ndarray, y_pred_oof: np.ndarray, valid: np.ndarray
    ) -> tuple[float, list[str], list[str]]:
        """Compute ``Var(y_pred_oof) / Var(y)`` and check the threshold.

        Args:
            y: the target.
            y_pred_oof: OOF predictions.
            valid: mask of finite pairs.

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
        threshold = 0.0 if self.adaptive_diversity else self.min_prediction_diversity
        if not _diversity_fails(diversity, threshold, self.adaptive_diversity):
            return diversity, [], []
        if self.adaptive_diversity:
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
    def _check_unique(
        self, y: np.ndarray, y_pred_oof: np.ndarray, valid: np.ndarray
    ) -> tuple[float, list[str], list[str]]:
        """Compute the share of unique predictions and check the lower bound.

        Predictions are rounded relative to the scale of the **target**
        (``std(y)``), not of the predictions — fixed digits are inapplicable to
        differently scaled and discrete targets.

        Args:
            y: the target (source of the rounding scale).
            y_pred_oof: OOF predictions.
            valid: mask of finite pairs.

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
        required = max(self.min_unique_count, self.min_unique_ratio * n_valid)
        if _fails_below(nunique, required):
            return (
                unique_ratio,
                [
                    (
                        f"Circuit B (unique): nunique={nunique} < "
                        f"max(min_unique_count={self.min_unique_count}, "
                        f"min_unique_ratio*N={self.min_unique_ratio * n_valid:.3f}) "
                        f"= {required:.3f}"
                    )
                ],
                [],
            )
        return unique_ratio, [], []

    # ──────────────────────────────────────────────────────────────────────── #
    #  Circuit C — Dead Feature Check
    # ──────────────────────────────────────────────────────────────────────── #
    def _check_dead_features(
        self,
        *,
        model: Any,
        X: Any | None,
        y: np.ndarray,
        diversity_failed: bool,
    ) -> tuple[int, list[str], list[str]]:
        """Check dead features: the linear or the permutation path.

        Args:
            model: the fitted model (may be a pipeline).
            X: the feature matrix (needed by the permutation path).
            y: the target.
            diversity_failed: whether circuit A failed (for the combined
                disqualification).

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
            return self._check_dead_features_linear(coef, diversity_failed)
        if not self.check_permutation_sensitivity:
            # Пермутационный путь отключён пользователем (T5): для нелинейных
            # моделей контур В бездействует — это осознанный выбор стоимости.
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
        return self._check_dead_features_permutation(
            model=model, X=X, y=y, diversity_failed=diversity_failed
        )

    def _check_dead_features_linear(
        self, coef: Any, diversity_failed: bool
    ) -> tuple[int, list[str], list[str]]:
        """Linear path: zero-coefficient share is a signal + a combination.

        Args:
            coef: the model's coefficient vector (``coef_``).
            diversity_failed: whether circuit A failed.

        Returns:
            tuple[int, list[str], list[str]]: (dead_features_count, reasons,
                soft_warnings).
        """
        coef_arr = np.asarray(coef, dtype=float).ravel()
        if coef_arr.size == 0:
            return 0, ["Circuit C (dead features): model exposes an empty coef_"], []
        zero_count = int(np.sum(np.abs(coef_arr) <= self.zero_coef_tolerance))
        zero_ratio = zero_count / coef_arr.size
        reasons: list[str] = []
        warnings: list[str] = []
        if _fails_above(zero_ratio, self.max_dead_feature_ratio):
            warnings.append(
                f"Circuit C (dead features): zero-coefficient ratio="
                f"{zero_ratio:.3f} > max_dead_feature_ratio="
                f"{self.max_dead_feature_ratio:.3f} (signal — on wide/noisy "
                f"data Lasso-family models legitimately zero out features)"
            )
            if self.dead_features_require_low_diversity:
                if diversity_failed:
                    reasons.append(
                        f"Circuit C (dead features): combination — "
                        f"zero-coefficient ratio={zero_ratio:.3f} above "
                        f"max_dead_feature_ratio AND prediction diversity "
                        f"below its threshold"
                    )
                elif not self.check_diversity:
                    # Комбинация не вычислима: контур А отключён — явно
                    # сообщаем, что правило не применялось (не молчаливо).
                    warnings.append(
                        "Circuit C (dead features): combination rule inactive — "
                        "check_diversity=False (dead-feature share is a signal "
                        "only)"
                    )
        return zero_count, reasons, warnings

    def _check_dead_features_permutation(
        self,
        *,
        model: Any,
        X: Any,
        y: np.ndarray,
        diversity_failed: bool,
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
        reason instead of escaping from ``check()``.

        Args:
            model: the fitted model able to ``predict`` on a modified X.
            X: the feature matrix (ndarray, DataFrame or scipy sparse).
            y: the target.
            diversity_failed: whether circuit A failed.

        Returns:
            tuple[int, list[str], list[str]]: (dead_features_count, reasons,
                soft_warnings).
        """
        try:
            return self._permutation_impl(
                model=model, X=X, y=y, diversity_failed=diversity_failed
            )
        except Exception as err:  # noqa: BLE001 — конвертируем в reason
            return (
                0,
                [f"Circuit C (dead features): permutation check failed: {err}"],
                [],
            )

    def _permutation_impl(
        self,
        *,
        model: Any,
        X: Any,
        y: np.ndarray,
        diversity_failed: bool,
    ) -> tuple[int, list[str], list[str]]:
        """Core permutation check; errors are caught by the caller wrapper."""
        rng = np.random.default_rng(self.permutation_seed)
        n_rows = y.shape[0]

        row_idx = np.arange(n_rows)
        if n_rows > self.permutation_max_rows:
            row_idx = rng.choice(n_rows, size=self.permutation_max_rows, replace=False)
        try:
            X_sub = X.iloc[row_idx] if isinstance(X, pd.DataFrame) else X[row_idx]
        except Exception as err:  # noqa: BLE001 — конвертируем в reason
            return (
                0,
                [(f"Circuit C (dead features): row subsampling of X failed: {err}")],
                [],
            )
        y_sub = y[row_idx]

        if X_sub.ndim != 2 or X_sub.shape[1] == 0:
            return (
                0,
                [
                    (
                        "Circuit C (dead features): X must be a 2-D matrix with at "
                        "least one column"
                    )
                ],
                [],
            )
        n_cols = X_sub.shape[1]
        col_idx = np.arange(n_cols)
        if n_cols > self.permutation_max_features:
            col_idx = rng.choice(
                n_cols, size=self.permutation_max_features, replace=False
            )
        col_idx = np.sort(col_idx)

        try:
            base_pred = np.asarray(model.predict(X_sub), dtype=float).reshape(-1)
        except Exception as err:  # noqa: BLE001 — конвертируем в reason
            return (
                0,
                [
                    (
                        f"Circuit C (dead features): prediction failed on subsampled "
                        f"X: {err}"
                    )
                ],
                [],
            )

        if base_pred.shape != y_sub.shape:
            return (
                0,
                [
                    (
                        f"Circuit C (dead features): prediction returned "
                        f"{base_pred.shape[0]} values for {y_sub.shape[0]} rows"
                    )
                ],
                [],
            )
        base_valid = np.isfinite(y_sub) & np.isfinite(base_pred)
        if not base_valid.any():
            return (
                0,
                [
                    (
                        "Circuit C (dead features): no finite (y, prediction) pairs "
                        "on the permutation subsample"
                    )
                ],
                [],
            )
        base_mse = float(mean_squared_error(y_sub[base_valid], base_pred[base_valid]))

        dead_count = 0
        for col in col_idx:
            deltas: list[float] = []
            for _ in range(self.permutation_repeats):
                try:
                    X_perm = _copy_with_permuted_column(X_sub, col, rng)
                except Exception as err:  # noqa: BLE001 — конвертируем в reason
                    return (
                        0,
                        [
                            (
                                f"Circuit C (dead features): column permutation "
                                f"failed (column {int(col)}): {err}"
                            )
                        ],
                        [],
                    )
                try:
                    perm_pred = np.asarray(model.predict(X_perm), dtype=float).reshape(
                        -1
                    )
                except Exception as err:  # noqa: BLE001 — конвертируем в reason
                    return (
                        0,
                        [
                            (
                                f"Circuit C (dead features): prediction failed on "
                                f"permuted X (column {int(col)}): {err}"
                            )
                        ],
                        [],
                    )
                # Fixed valid-pair mask (from the base predictions): every
                # repeat of the same column is evaluated on the same rows, so
                # the mean ΔMSE is not biased by varying masks. A repeat that
                # returns non-finite values for a base-valid row is skipped
                # rather than silently switching to a different row set.
                if perm_pred.shape != y_sub.shape or not np.all(
                    np.isfinite(perm_pred[base_valid])
                ):
                    continue
                deltas.append(
                    float(mean_squared_error(y_sub[base_valid], perm_pred[base_valid]))
                    - base_mse
                )
            if not deltas:
                continue
            mean_delta = float(np.mean(deltas))
            # Relative change of MSE; the _MSE_EPS stabilizer keeps the ratio
            # well-defined for a perfect in-sample fit (base_mse == 0): in that
            # case only an exactly unchanged prediction counts as "dead".
            rel_change = abs(mean_delta) / (abs(base_mse) + _MSE_EPS)
            if rel_change < self.permutation_tolerance:
                dead_count += 1

        n_checked = int(col_idx.size)
        reasons: list[str] = []
        warnings: list[str] = []
        dead_ratio = dead_count / n_checked if n_checked else 0.0
        if _fails_above(dead_ratio, self.max_dead_feature_ratio):
            warnings.append(
                f"Circuit C (dead features): dead-feature ratio="
                f"{dead_ratio:.3f} > max_dead_feature_ratio="
                f"{self.max_dead_feature_ratio:.3f} on permutation check "
                f"({dead_count}/{n_checked} checked columns; note: correlated "
                f"features may produce 'false-dead' results)"
            )
            if self.dead_features_require_low_diversity:
                if diversity_failed:
                    reasons.append(
                        "Circuit C (dead features): combination — high "
                        "dead-feature ratio AND prediction diversity below its "
                        "threshold"
                    )
                elif not self.check_diversity:
                    # Комбинация не вычислима: контур А отключён — явно
                    # сообщаем, что правило не применялось (не молчаливо).
                    warnings.append(
                        "Circuit C (dead features): combination rule inactive — "
                        "check_diversity=False (dead-feature share is a signal "
                        "only)"
                    )
        return dead_count, reasons, warnings

    # ──────────────────────────────────────────────────────────────────────── #
    #  Circuit D — Generalization Gap
    # ──────────────────────────────────────────────────────────────────────── #
    def _check_generalization_gap(
        self, y: np.ndarray, y_pred_oof: np.ndarray, y_pred_full: Any | None
    ) -> tuple[float, list[str]]:
        """Check ``RMSE_oof / RMSE_full`` against ``max_generalization_gap``.

        Both RMSEs are computed on the **same** joint mask of finite pairs
        (y, y_pred_oof, y_pred_full), so the ratio compares apples to apples.

        Args:
            y: the target.
            y_pred_oof: OOF predictions.
            y_pred_full: full-fit predictions (None → fail with a reason).

        Returns:
            tuple[float, list[str]]: (generalization_gap, reasons).
        """
        if y_pred_full is None:
            return float("nan"), [
                "Circuit D (generalization gap): y_pred_full is not provided"
            ]
        full_arr = _to_float_vector(y_pred_full, "y_pred_full")
        if full_arr.shape != y.shape:
            raise ValueError(
                f"y and y_pred_full must have the same length "
                f"(got {y.shape} and {full_arr.shape})"
            )
        mask = np.isfinite(y) & np.isfinite(y_pred_oof) & np.isfinite(full_arr)
        if not mask.any():
            return float("nan"), [
                "Circuit D (generalization gap): no finite pairs to compute RMSE"
            ]
        rmse_full = float(np.sqrt(mean_squared_error(y[mask], full_arr[mask])))
        rmse_oof = float(np.sqrt(mean_squared_error(y[mask], y_pred_oof[mask])))
        if rmse_oof <= _ZERO_RMSE_EPS:
            # Деление на ноль по знаменателю OOF — провал с явным reason.
            return float("inf"), [
                "Circuit D (generalization gap): RMSE_oof≈0 — division by zero"
            ]
        if rmse_full <= _ZERO_RMSE_EPS:
            # Идеальное обучение на train (дерево глубины 20, 1-NN) —
            # классическое переобучение, gap уходит в бесконечность.
            return float("inf"), [
                (
                    "Circuit D (generalization gap): RMSE_full≈0 (perfect "
                    "in-sample fit) — division by zero"
                )
            ]
        gap = rmse_oof / rmse_full
        if _fails_above(gap, self.max_generalization_gap):
            return gap, [
                (
                    f"Circuit D (generalization gap): gap={gap:.3f} > "
                    f"max_generalization_gap={self.max_generalization_gap:.3f}"
                )
            ]
        return gap, []
