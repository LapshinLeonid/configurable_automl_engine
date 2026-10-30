"""Binning of a regression target and decoding of the resampled target.

Holds the quantile binning that turns a numeric target into pseudo-classes
for imbalanced-learn (``_bin_regression_target`` and its helpers), the
per-class row counts of the sampling strategy (``_sampling_strategy``), the
decoding of the resampled target (``_decode_regression_y`` and the legacy
quantile-bin decode ``_decode_continuous_y``) and the ``_BinnedTarget``
record. The module must not import the other oversampling modules.
"""

from collections import Counter
from math import ceil
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

#: Default number of quantile bins used to discretize a regression target
#: before the sampling strategy is computed. The binned groups imitate (and
#: balance) "classes" of a regression target.
DEFAULT_TARGET_BINS = 8

#: Minimum number of quantile bins a discretized regression target must have:
#: imbalanced-learn needs at least two groups to resample.
MIN_TARGET_BINS = 2

#: Backward-compatible aliases of :data:`DEFAULT_TARGET_BINS` and
#: :data:`MIN_TARGET_BINS` kept for earlier callers of the module.
CONTINUOUS_TARGET_QUANTILES = DEFAULT_TARGET_BINS
CONTINUOUS_TARGET_MIN_BINS = MIN_TARGET_BINS


#: Backward-compatible alias of :func:`_bin_regression_target` kept for
#: earlier callers of the module.
def _bin_continuous_target(
    y: pd.Series,
) -> tuple[bool, pd.Series, dict[int, float]]:
    """Discretize a continuous regression target into quantile bin codes.

    Backward-compatible alias of :func:`_bin_regression_target` with the
    default bin count.

    Args:
        y (pd.Series): The target variable vector.

    Returns:
        tuple[bool, pd.Series, dict[int, float]]: A triple of
            (is_regression, codes, bin_mids).
    """
    return _bin_regression_target(y, DEFAULT_TARGET_BINS)


def _bin_regression_target(
    y: pd.Series, target_bins: int
) -> tuple[bool, pd.Series, dict[int, float]]:
    """Discretize a numeric regression target into quantile bin codes.

    Runs a single quantile discretization and reuses it both for the
    regression-mode detection and for the backward decoding, so the hot path
    (:meth:`DataOversampler._fit_resample`) bins the target only once.

    A target is eligible for the regression path when it is numeric (``NaN``
    allowed — the natural regression case) and can be discretized into at
    least :data:`MIN_TARGET_BINS` groups. Non-numeric, bool and NaN-only
    targets stay on the classification path (the caller keeps the bin codes
    only when the tuple's first element is ``True``).

    Args:
        y (pd.Series): The target variable vector.
        target_bins (int): Number of quantile bins.

    Returns:
        tuple[bool, pd.Series, dict[int, float]]: A triple of
            (is_regression, codes, bin_mids). When the target is not numeric
            (or cannot be discretized into at least :data:`MIN_TARGET_BINS`
            groups), returns ``(False, y, {})`` so the caller keeps the
            classification path.
    """
    if y.empty or not is_numeric_dtype(y.dtype):
        return False, y, {}
    if is_bool_dtype(y.dtype):
        return False, y, {}
    finite = y.dropna()
    if finite.empty:
        return False, y, {}
    values = np.asarray(finite, dtype=float)
    if not np.all(np.isfinite(values)):
        return False, y, {}
    try:
        finite_bins = pd.qcut(finite, q=target_bins, duplicates="drop")
    except (TypeError, ValueError):
        return False, y, {}
    if finite_bins.cat.categories.size < MIN_TARGET_BINS:
        return False, y, {}

    finite_mask = y.notna()
    codes = np.asarray(finite_bins.cat.codes, dtype=np.int32)
    binned: pd.Series = pd.Series(
        np.full(len(y), -1, dtype=np.int32), index=y.index, name=y.name
    )
    binned.iloc[np.asarray(finite_mask)] = codes
    bin_mids: dict[int, float] = {
        int(code): float(interval.mid)
        for code, interval in enumerate(finite_bins.cat.categories)
    }
    return True, binned, bin_mids


def _is_continuous_target(y: pd.Series) -> bool:
    """Determine whether a target vector is continuous regression data.

    Kept for backward compatibility with earlier callers of the module: a
    target is considered continuous exactly when it is numeric and can be
    discretized into at least :data:`MIN_TARGET_BINS` quantile bins. The
    engine itself no longer distinguishes continuous from integer targets —
    the regression path (see :func:`_bin_regression_target`) is used for any
    numeric target.

    Args:
        y (pd.Series): The target variable vector.

    Returns:
        bool: ``True`` if the target should be treated as regression data.
    """
    return _bin_regression_target(y, DEFAULT_TARGET_BINS)[0]


def _prepare_continuous_bins(y: pd.Series) -> tuple[pd.Series, dict[int, float]]:
    """Discretize a continuous target into quantile bins (class codes).

    Kept for backward compatibility: equivalent to
    :func:`_bin_regression_target` with the default bin count. The bin codes
    are computed on the finite target values; rows with a missing target keep
    code ``-1`` and are passed through the samplers unchanged (like an extra
    class that is not resampled).

    Args:
        y (pd.Series): The continuous target vector.

    Returns:
        tuple[pd.Series, dict[int, float]]: A pair of the integer (``int32``)
            code series (the quantile bin number per row, ``-1`` for a
            missing target) and the per-bin representative values (interval
            midpoints) used to restore continuous targets after resampling.

    Raises:
        ValueError: If ``y`` cannot be discretized into at least
            :data:`MIN_TARGET_BINS` quantile bins.
    """
    is_continuous, binned, bin_mids = _bin_regression_target(y, DEFAULT_TARGET_BINS)
    if not is_continuous:
        raise ValueError(
            "a continuous target with at least "
            f"{MIN_TARGET_BINS} quantile bins is required"
        )
    return binned, bin_mids


def _decode_continuous_y(
    y_binned: Any,
    y_original: pd.Series,
    y_orig_codes: pd.Series,
    bin_mids: dict[int, float],
) -> np.ndarray:
    """Decode quantile-bin codes back into continuous target values.

    Original rows keep their real target values; each synthetic row receives
    the midpoint of the quantile bin it was generated for (identified by its
    own bin code, so the per-row association is preserved).

    The decoding relies on the samplers prepending the original rows
    unchanged (the current behavior of RandomOverSampler/SMOTE/ADASYN in
    imbalanced-learn, where ``X_resampled``/``y_resampled`` start with a copy
    of the input). The prefix is verified explicitly so that a future change
    in the row ordering of imbalanced-learn cannot silently corrupt the
    original target values — it fails loudly instead.

    Args:
        y_binned: The resampled bin-code vector (first ``len(y_original)``
            entries are the original codes, the tail is synthetic).
        y_original (pd.Series): The original (continuous) target vector.
        y_orig_codes (pd.Series): The bin codes of the original target
            vector (the first element of the tuple returned by
            :func:`_prepare_continuous_bins`).
        bin_mids (dict[int, float]): Per-bin representative values produced
            by :func:`_prepare_continuous_bins`.

    Returns:
        np.ndarray: The continuous target vector for the resampled rows
            (original values followed by the synthetic ones).

    Raises:
        ValueError: If the resampled vector does not start with the original
            rows (unexpected row ordering from the sampler).
        KeyError: If a synthetic row carries a bin code absent from
            ``bin_mids`` (i.e. a code that was never generated by
            :func:`_prepare_continuous_bins`).
    """
    y_binned_arr = np.asarray(y_binned)
    n_original = len(y_original)
    y_res = np.asarray(y_original, dtype=float).copy()
    if len(y_binned_arr) < n_original:
        raise ValueError(
            "The resampled bin-code vector is shorter than the original "
            "target; cannot decode the continuous target."
        )
    if not np.array_equal(y_binned_arr[:n_original], np.asarray(y_orig_codes)):
        raise ValueError(
            "The resampled bin-code vector does not start with the original "
            "rows; cannot safely decode the continuous target. This may "
            "indicate a change in the row ordering of imbalanced-learn."
        )
    if len(y_binned_arr) == n_original:
        return y_res
    synth_values = np.array(
        [float(bin_mids[int(c)]) for c in y_binned_arr[n_original:]],
        dtype=float,
    )
    return np.concatenate([y_res, synth_values])


class _BinnedTarget(NamedTuple):
    """The target of one resampling call, binned for the regression path.

    Attributes:
        y_s (pd.Series): The original target values.
        is_regression (bool): Whether the regression path is used.
        y_fit (pd.Series): The values resampled as classes (bin codes for
            the regression path, ``y_s`` itself for classification).
        bin_mids (dict[int, float]): Per-bin representative values.
    """

    y_s: pd.Series
    is_regression: bool
    y_fit: pd.Series
    bin_mids: dict[int, float]


def _decode_regression_y(
    y_res: Any,
    y_original: pd.Series,
    y_fit: pd.Series,
    bin_mids: dict[int, float],
    sampler: Any,
) -> np.ndarray:
    """Restore real regression targets after resampling.

    * ``random``: every row (original or duplicated) gets the real
      target of its source row, looked up through
      ``RandomOverSampler.sample_indices_``.
    * ``smote``/``adasyn`` (regression path): the synthetic rows already
      carry interpolated real values; original rows keep theirs.

    Original rows are verified to stay in front (the current behavior of
    the imbalanced-learn samplers); any deviation fails loudly.

    Args:
        y_res: The resampled target vector (real values for the engine
            generators, bin codes for ``random``).
        y_original (pd.Series): The original target vector.
        y_fit (pd.Series): The bin codes of the original target.
        bin_mids (dict[int, float]): Per-bin representative values (only
            used by the legacy decode path).
        sampler: The fitted ``RandomOverSampler`` (``None`` for the
            engine generators).

    Returns:
        np.ndarray: The regression target vector for the resampled rows.
    """
    if sampler is not None:
        indices = getattr(sampler, "sample_indices_", None)
        if indices is None or len(np.asarray(indices)) != len(y_res):
            # Fall back to the quantile-bin decode when the sampler does
            # not expose sample_indices_ (defensive; the current
            # imbalanced-learn RandomOverSampler always sets it).
            return _decode_continuous_y(y_res, y_original, y_fit, bin_mids)
        return np.asarray(y_original, dtype=float)[np.asarray(indices)]
    y_arr = np.asarray(y_res)
    if len(y_arr) < len(y_original):
        raise ValueError(
            "The resampled target vector is shorter than the original "
            "target; cannot decode the regression target."
        )
    if not np.allclose(
        y_arr[: len(y_original)],
        np.asarray(y_original, dtype=float),
        equal_nan=True,
    ):
        raise ValueError(
            "The resampled target vector does not start with the original "
            "rows; cannot safely decode the regression target."
        )
    return np.asarray(y_arr, dtype=float)


def _sampling_strategy(
    y: pd.Series, multiplier: float, *, balance: bool, skip_missing_code: bool = False
) -> dict[Any, int]:
    """Compute the target number of instances for each class.

    For a classification target (or the pseudo-classes of a binned
    regression target) the values are the *final* per-class row counts;
    the overall result size is therefore approximately
    ``N * multiplier`` (exact for ``balance=False``, where each class is
    scaled independently). For a binned regression target, classes that
    end up smaller than ``ceil(count * multiplier)`` (e.g. singleton
    bins that cannot be synthesized) are dropped from the strategy by
    the caller and pass through unchanged, so the final size may be
    smaller than ``N * multiplier`` — this is an intentional
    "balance-like" behavior, not a bug.

    Args:
        y (pd.Series): The target variable vector.
        multiplier (float): The scaling factor.
        balance (bool): Whether to bring all classes up to the majority class
            size (``DataOversampler.balance``).
        skip_missing_code (bool): Whether to exclude the pseudo-class
            ``-1`` (rows with a missing target in the regression mode)
            from the strategy — it must never be rebalanced or scaled
            (pass-through only).
    Returns:
        Dict[Any, int]: A dictionary where keys are class labels
            and values are the final number of rows.
    """
    counts = Counter(y)
    # The pseudo-class "-1" is reserved for rows with a missing target in
    # the regression mode: imbalanced-learn treats it as a real class, so
    # it must never be rebalanced or scaled (pass-through only).
    if skip_missing_code and -1 in counts:
        del counts[-1]
    # If balancing is enabled, bring all classes up to the majority class,
    # multiplied by the factor. Otherwise scale each class proportionally.
    if balance:
        # Base is the largest class
        base_size = max(counts.values())
        return {cls: ceil(base_size * multiplier) for cls in counts}

    # Important: imbalanced-learn expects the final number of instances
    # (Total), not the delta
    return {cls: ceil(cnt * multiplier) for cls, cnt in counts.items()}
