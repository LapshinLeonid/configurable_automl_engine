"""Median aggregation of Sanity Gate circuits A/B over repeated splits.

Repeated validation produces one OOF vector per split. Circuits A and B are
evaluated independently on every vector and their metrics are combined by the
median, exactly like circuit D's gap ratio. The reported value is the median
of the per-repeat values: identical repeat vectors keep the same ratio and
verdict at any ``R``, while adding informative or degenerate repeats changes
the median (a majority of degenerate repeats still fails through it). A
single repeat follows the single-split circuit functions.

The module depends only on ``sanity_circuits``/``sanity_result``; it must not
import the gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from configurable_automl_engine.training_engine.sanity_circuits import (
    _ZERO_VAR_EPS,
    _diversity_fails,
    _fails_below,
    _r2_oof,
    _rounding_decimals,
    check_diversity,
    check_unique,
)
from configurable_automl_engine.training_engine.sanity_result import (
    _to_float_vector,
)


def _repeat_vectors(y: np.ndarray, vectors: Sequence[Any]) -> list[np.ndarray]:
    """Convert the per-repeat vectors to validated 1-D float arrays."""
    prepared = [
        _to_float_vector(vector, f"y_pred_oof_by_repeat[{index}]")
        for index, vector in enumerate(vectors)
    ]
    for index, vector in enumerate(prepared):
        if vector.shape != y.shape:
            raise ValueError(
                "y and every y_pred_oof_by_repeat vector must have the same "
                f"length (got {y.shape} and {vector.shape} for repeat {index})"
            )
    return prepared


def check_diversity_repeats(
    y: np.ndarray,
    y_pred_oof_by_repeat: Sequence[Any],
    *,
    adaptive: bool,
    min_prediction_diversity: float,
) -> tuple[float, list[str], list[str]]:
    """Circuit A on every repeat, aggregated by the median ratio.

    A repeat with no finite pairs or a constant target is excluded from the
    aggregate. When no repeat yields a ratio, every-repeat missing pairs is a
    reason while a constant target (with finite pairs) is a soft warning.
    One repeat follows :func:`check_diversity`.

    Args:
        y: The target vector, one-dimensional. Non-finite entries are masked
            together with non-finite predictions.
        y_pred_oof_by_repeat: The OOF vector of every repeat, each aligned to
            ``y`` by position and one-dimensional (NaN marks an uncovered
            row). An empty sequence follows the single-split path with all
            predictions missing.
        adaptive: Whether circuit A uses the constant-model reference instead
            of the fixed diversity threshold.
        min_prediction_diversity: The fixed threshold used when ``adaptive``
            is false.

    Returns:
        tuple[float, list[str], list[str]]: ``(median_diversity_ratio,
            reasons, warnings)``. ``NaN`` is returned when no repeat yields a
            ratio.

    Raises:
        ValueError: If a repeat has a different length from ``y`` or cannot be
            converted to a one-dimensional float vector.
    """
    vectors = _repeat_vectors(y, y_pred_oof_by_repeat)
    if len(vectors) <= 1:
        vector = vectors[0] if vectors else np.full(y.shape, np.nan, dtype=float)
        valid = np.isfinite(y) & np.isfinite(vector)
        return check_diversity(
            y,
            vector,
            valid,
            adaptive=adaptive,
            min_prediction_diversity=min_prediction_diversity,
        )

    ratios: list[float] = []
    r2_values: list[float] = []
    no_pairs = 0
    for vector in vectors:
        valid = np.isfinite(y) & np.isfinite(vector)
        if not valid.any():
            no_pairs += 1
            continue
        var_y = float(np.var(y[valid]))
        if var_y <= _ZERO_VAR_EPS:
            continue
        ratios.append(float(np.var(vector[valid]) / var_y))
        if adaptive:
            r2_values.append(_r2_oof(y[valid], vector[valid]))
    if not ratios:
        if no_pairs == len(vectors):
            return (
                float("nan"),
                [
                    (
                        "Circuit A (diversity): no finite (y, y_pred_oof) "
                        "pairs in any repeat"
                    )
                ],
                [],
            )
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

    diversity = float(np.median(ratios))
    r2_oof = float(np.median(r2_values)) if r2_values else 0.0
    threshold = 0.0 if adaptive else min_prediction_diversity
    if not _diversity_fails(diversity, threshold, adaptive, r2_oof):
        return diversity, [], []
    if adaptive:
        if np.isfinite(r2_oof) and r2_oof < 0.0:
            reason = (
                f"Circuit A (diversity): median R²_oof={r2_oof:.4f} < 0 — "
                "model is worse than the constant prediction"
            )
        else:
            reason = (
                "Circuit A (diversity): median ratio=0.0000 equals the "
                "constant-model baseline — no expressiveness"
            )
    else:
        reason = (
            f"Circuit A (diversity): median ratio={diversity:.4f} < "
            f"min_prediction_diversity={threshold:.4f}"
        )
    return diversity, [reason], []


def check_unique_repeats(
    y: np.ndarray,
    y_pred_oof_by_repeat: Sequence[Any],
    *,
    min_unique_count: int,
    min_unique_ratio: float,
) -> tuple[float, list[str], list[str]]:
    """Circuit B on every repeat, aggregated by the median ratio.

    The required ratio is derived per repeat with the same rounding scale,
    ``min_unique_count`` floor and ``nunique(y)`` cap as the single-split
    circuit; the median of the observed ratios is compared with the median of
    the required ratios. Equality with the threshold passes. One repeat
    follows :func:`check_unique`.

    Args:
        y: The target vector, one-dimensional. Non-finite entries are masked
            together with non-finite predictions.
        y_pred_oof_by_repeat: The OOF vector of every repeat, each aligned to
            ``y`` by position and one-dimensional (NaN marks an uncovered
            row). An empty sequence follows the single-split path with all
            predictions missing.
        min_unique_count: The absolute floor on the number of distinct
            predictions.
        min_unique_ratio: The required share of distinct predictions before
            the ``nunique(y)`` cap is applied.

    Returns:
        tuple[float, list[str], list[str]]: ``(median_unique_ratio, reasons,
            warnings)``. ``NaN`` is returned when no repeat has finite pairs.

    Raises:
        ValueError: If a repeat has a different length from ``y`` or cannot be
            converted to a one-dimensional float vector.
    """
    vectors = _repeat_vectors(y, y_pred_oof_by_repeat)
    if len(vectors) <= 1:
        vector = vectors[0] if vectors else np.full(y.shape, np.nan, dtype=float)
        valid = np.isfinite(y) & np.isfinite(vector)
        return check_unique(
            y,
            vector,
            valid,
            min_unique_count=min_unique_count,
            min_unique_ratio=min_unique_ratio,
        )

    ratios: list[float] = []
    required_ratios: list[float] = []
    for vector in vectors:
        valid = np.isfinite(y) & np.isfinite(vector)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        std_y = float(np.std(y[valid]))
        decimals = _rounding_decimals(std_y)
        nunique = len(np.unique(np.round(vector[valid], decimals=decimals)))
        n_unique_y = len(np.unique(np.round(y[valid], decimals=decimals)))
        ratio_bound = min(min_unique_ratio * n_valid, n_unique_y)
        required = min(max(min_unique_count, ratio_bound), n_unique_y)
        ratios.append(nunique / n_valid)
        required_ratios.append(required / n_valid)
    if not ratios:
        return (
            float("nan"),
            ["Circuit B (unique): no finite (y, y_pred_oof) pairs in any repeat"],
            [],
        )

    unique_ratio = float(np.median(ratios))
    required_ratio = float(np.median(required_ratios))
    if _fails_below(unique_ratio, required_ratio):
        return (
            unique_ratio,
            [
                (
                    f"Circuit B (unique): median unique ratio="
                    f"{unique_ratio:.4f} < required median ratio="
                    f"{required_ratio:.4f}"
                )
            ],
            [],
        )
    return unique_ratio, [], []
