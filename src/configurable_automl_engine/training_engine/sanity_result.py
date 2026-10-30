"""The result record of a Sanity Gate audit.

Holds :class:`SanityCheckResult` (the verdict, the reasons, the soft warnings
and the metrics of the four circuits), the severity mapping
(``_compute_severity``) and the input checks shared by the gate and the
circuits (``_to_float_vector``, ``_validate_check_inputs``). Like the gate, it depends only on numpy,
pandas and scikit-learn; it must not import the gate or the circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
            circuit is disabled or not computable, ``inf`` for the
            division-by-zero cases (``RMSE_full ≈ 0`` / ``RMSE_oof ≈ 0``).
            In ``generalization_gap_mode = "warn"`` every gap violation —
            finite or infinite — is reported in ``soft_warnings`` and never
            disqualifies; in ``"reject"`` mode it becomes a
            disqualification reason.
        severity: formalized severity for the fallback:
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
    """Map reasons/warnings to a formalized severity level.

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


def _validate_check_inputs(
    y: Any, y_pred_oof: Any, X: Any | None
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the ``check()`` inputs and convert them to float vectors.

    Args:
        y: the true target.
        y_pred_oof: OOF predictions aligned with ``y``.
        X: the feature matrix (or None).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``y`` and ``y_pred_oof`` as 1-D
            float vectors.

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
    return y_arr, oof_arr
