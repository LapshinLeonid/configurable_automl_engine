"""Binning of a regression target and decoding of the resampled target.

Holds the quantile binning that turns a numeric target into pseudo-classes
for imbalanced-learn (``_bin_regression_target`` and its helpers), the
decoding of the binned target after resampling (``_decode_continuous_y``)
and the ``_BinnedTarget`` record. The module must not import the other
oversampling modules.
"""

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
