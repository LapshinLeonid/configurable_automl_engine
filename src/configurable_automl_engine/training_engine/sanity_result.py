"""The result record of a Sanity Gate audit.

Holds :class:`SanityCheckResult` (the verdict, the reasons, the soft warnings
and the metrics of the four circuits), the severity mapping
(``_compute_severity``) and the input checks shared by the gate and the
circuits (``_to_float_vector``, ``_validate_check_inputs``). Like the gate, it depends only on numpy,
pandas and scikit-learn; it must not import the gate or the circuits.
"""

from __future__ import annotations

from collections.abc import Sequence
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
        n_repeats: The total number of repeats recorded by the trainer,
            including repeats invalidated by the M3 scoring layer. The select
            step fills this value; a direct ``ModelSanityGate.check`` caller
            sees the number of valid repeat vectors it passed.
        n_valid_repeats: The number of valid repeat vectors used by this
            audit (0 when the audit was skipped or failed). More than one
            valid vector is aggregated by the median; exactly one valid
            vector uses the single-split path even when ``n_repeats > 1``.
            ``1`` also covers a classic single OOF vector.
    """

    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    diversity_ratio: float = float("nan")
    unique_ratio: float = float("nan")
    dead_features_count: int = 0
    generalization_gap: float = float("nan")
    severity: int = _SEVERITY_CLEAN
    n_repeats: int = 1
    n_valid_repeats: int = 1


def _compute_severity(
    reasons: list[str], soft_warnings: list[str], warn_only: bool
) -> int:
    """Map reasons/warnings to a formalized severity level.

    Args:
        reasons: disqualification reasons.
        soft_warnings: soft signals.
        warn_only: True — the gate does not disqualify (severity ≤ 1).
    """
    if reasons:
        severity = min(2 + len(reasons) - 1, 4)
    elif soft_warnings:
        severity = _SEVERITY_WARNING
    else:
        severity = _SEVERITY_CLEAN
    return min(severity, _SEVERITY_WARNING) if warn_only else severity


def _to_float_vector(values: Any, name: str) -> np.ndarray:
    """Convert an input to a 1-D float array (with shape validation)."""
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
    """Validate the ``check()`` inputs and convert them to float vectors."""
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


def _validate_repeat_vectors(
    y: np.ndarray, repeats: Sequence[Any] | None
) -> list[np.ndarray] | None:
    """Validate the per-repeat OOF vectors of a multi-split audit.

    ``None`` and an empty sequence mean "no repeat information" — the caller
    falls back to single-vector input. Every provided vector
    must be one-dimensional and aligned to ``y`` by position (the same
    contract as ``y_pred_oof``).

    Args:
        y: The validated target vector.
        repeats: The per-repeat OOF vectors (already aligned to ``y``), or
            ``None``.

    Returns:
        list[np.ndarray] | None: The vectors as float arrays, or ``None``
            when no repeat information was passed.

    Raises:
        ValueError: When a vector is not one-dimensional or its length does
            not match ``y``.
    """
    if repeats is None:
        return None
    vectors = list(repeats)
    if not vectors:
        return None
    validated: list[np.ndarray] = []
    for index, vector in enumerate(vectors):
        arr = _to_float_vector(vector, f"y_pred_oof_by_repeat[{index}]")
        if arr.shape != y.shape:
            raise ValueError(
                "y and every y_pred_oof_by_repeat vector must have the same "
                f"length (got {y.shape} and {arr.shape} for repeat {index})"
            )
        validated.append(arr)
    return validated
