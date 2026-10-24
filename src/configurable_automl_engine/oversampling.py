"""
Data Oversampling Engine: An intelligent sample expander.
The class provides class balancing and synthetic data expansion
while preserving data types and supporting multithreading.

Key features:
    1. Multi-Algorithm: Supports Random, SMOTE and ADASYN.
    2. Noise Injection: Optional Gaussian noise for regularization.
    3. Type Integrity: The `_restore_dtypes` mechanism prevents type
       degradation (e.g. Categorical -> Object cast) after Numpy transforms.
    4. Thread-Safety: Uses an RLock for
    safe access to parameters inside a Pipeline.

The engine is regression-only, so by default any numeric target (float or
integer) is treated as a regression target and oversampled through quantile
bins: the sampling strategy is computed on binned ``y`` regardless of
integrality. The classification path is kept only for standalone
``DataOversampler`` calls with an explicit ``task="classification"``
argument.
"""

import logging
import threading
from collections import Counter
from math import ceil
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import RandomOverSampler
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

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

#: Experimental status of the oversampling feature, driven by the nightly
#: evidence benchmark (``benchmarks/run_oversampling_benchmark.py``, gate
#: T13). When any oversampling algorithm lowers the median held-out R² by
#: more than 0.01 relative to the no-oversampling baseline, the benchmark
#: reports the violation and this flag is set to ``True``: README and
#: API_REFERENCE then mark the feature experimental, and a WARNING is logged
#: every time the oversampler actually runs.
#
#: Current status: the nightly benchmark observed a median held-out R²
#: degradation of ADASYN on the count dataset (Δ ≈ +0.019 > 0.01), so the
#: feature is experimental until the evidence gate passes.
OVERSAMPLING_EXPERIMENTAL: bool = True


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


logger = logging.getLogger(__name__)


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


def _is_categorical_col(s: pd.Series) -> bool:
    """Determine whether a column should be considered categorical.

    Detection rule:
    * Numeric dtype (``is_numeric_dtype``) -> the column is numeric.
    * Non-numeric dtype (``object``/``category``/``string``):
        - if all values are 'purely numeric' (``pd.to_numeric`` without NaN)
          -> the column is treated as numeric (the 'numeric strings' edge
          case, e.g. IDs ``'10'``, ``'20'``);
        - otherwise (at least one non-numeric value) -> categorical.

    Returns:
        bool: ``True`` if the column is categorical.
    """
    if is_numeric_dtype(s):
        return False
    try:
        converted = pd.to_numeric(s, errors="coerce")
    except (TypeError, ValueError):
        return True
    return bool(converted.isna().any())


class _SMOTERGenerator:
    """SMOTER-style synthetic sample generator for regression targets.

    imbalanced-learn does not expose the (row, neighbor, gap) triplets it
    uses internally, so the engine implements its own generator following the
    SMOTER scheme (Torgo et al.): for a synthetic sample between row ``i``
    and its neighbor ``n`` (from the same bin) with a random gap ``u`` in
    ``(0, 1)``, both the features and the target are interpolated:

    ``x = x_i + u * (x_n - x_i)`` and ``y = y_i + u * (y_n - y_i)``.

    Neighbors are found with ``NearestNeighbors`` within each bin, which
    keeps the synthetic targets honest: the interpolated ``y`` always lies
    strictly between the two parents' values instead of collapsing every
    synthetic row onto a bin midpoint.

    Categorical columns follow the SMOTENC rules: a category that differs
    between the two parents keeps the first parent's (row ``i``) value;
    a shared category is kept as is.

    ``density_weighted=True`` switches the parent selection to the ADASYN
    scheme: the per-parent number of generated rows is proportional to its
    "density" — the share of k-NN neighbors (over the whole dataset) that
    belong to OTHER bins. Rows in sparse regions of the feature space are
    therefore copied more often than rows in dense regions, which visibly
    differs from the uniform random parent selection of SMOTE.
    """

    def __init__(
        self,
        *,
        random_state: int | None = None,
        categorical_features: list[int] | None = None,
        density_weighted: bool = False,
    ):
        """Initialize the generator.

        Args:
            random_state (int | None): Random seed for reproducibility.
            categorical_features (list[int] | None): Indices of categorical
                columns; they are never interpolated (SMOTENC
                copy-from-first-parent rules apply instead).
            density_weighted (bool): Use ADASYN-style density weighting of
                the parent selection instead of uniform random choice.
        """
        self.random_state = random_state
        self.categorical_features = set(categorical_features or [])
        self.density_weighted = density_weighted

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_binned: np.ndarray,
        n_to_generate: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate ``n_to_generate[bin]`` synthetic rows per quantile bin.

        Args:
            X (np.ndarray): Numeric (already encoded) feature matrix.
            y (np.ndarray): The real regression target values per row.
            y_binned (np.ndarray): Quantile-bin code per row (``-1`` —
                missing target, never used as a synthetic parent).
            n_to_generate (dict[int, int]): Number of synthetic rows to
                create per bin code.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of the stacked synthetic
                feature rows and the interpolated synthetic target values.
        """
        rng = np.random.RandomState(self.random_state)
        X_synth: list[np.ndarray] = []
        y_synth: list[np.ndarray] = []

        # ADASYN-style density: for every row, the share of its k-NN
        # neighbors (over the whole dataset) that belong to OTHER bins.
        # Rows whose neighborhood is dominated by other bins are "harder"
        # and receive proportionally more synthetic clones.
        density: np.ndarray | None = None
        if self.density_weighted and len(X) >= 2:
            k_global = min(5, len(X) - 1)
            nn_global = NearestNeighbors(n_neighbors=k_global + 1).fit(X)
            neigh_global = nn_global.kneighbors(X, return_distance=False)[:, 1:]
            # Neighbor "belongs to another bin" when its bin differs from
            # the row's; missing-target rows (code -1) count as different.
            other = y_binned[neigh_global] != y_binned[:, None]
            density = other.sum(axis=1) / k_global

        for _bin, n in n_to_generate.items():
            if n <= 0:
                continue
            mask = y_binned == _bin
            if mask.sum() < 2:
                continue
            bin_X = X[mask]
            bin_y = y[mask]
            n_bin = len(bin_X)
            k = min(5, n_bin - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(bin_X)
            neigh = nn.kneighbors(bin_X, return_distance=False)[:, 1:]
            # Rows with a k-NN query that returns no neighbor cannot be
            # synthesized (SMOTE contract: n_samples >= k_neighbors + 1).
            queryable = neigh.sum(axis=1) >= 1
            if not queryable.any():
                continue
            pool_idx = np.flatnonzero(queryable)
            n_gen = int(n)
            if self.density_weighted and density is not None:
                # ADASYN: distribute the bin's synthetic budget among the
                # queryable parents proportionally to their density.
                parents_mask = np.zeros(len(bin_X), dtype=bool)
                parents_mask[pool_idx] = True
                w = density[mask][parents_mask]
                w_sum = w.sum()
                if w_sum <= 0:
                    # No queryable parent has a cross-bin neighbor (constant
                    # or well-separated features): fall back to the uniform
                    # parent selection of SMOTE.
                    chosen = rng.randint(0, len(pool_idx), size=n_gen)
                    rows = pool_idx[chosen]
                else:
                    w = w / w_sum
                    counts = rng.multinomial(n_gen, w)
                    rows = np.repeat(pool_idx, counts)
            else:
                chosen = rng.randint(0, len(pool_idx), size=n_gen)
                rows = pool_idx[chosen]
            neigh_chosen = neigh[rows, 0]
            gaps = rng.uniform(0.0, 1.0, size=len(rows)).reshape(-1, 1)
            feat_diff = bin_X[neigh_chosen] - bin_X[rows]
            new_X = bin_X[rows] + gaps * feat_diff
            # Categorical columns: SMOTENC rule — a category differing
            # between the parents keeps the first parent's value.
            if self.categorical_features:
                new_X = new_X.copy()
                for col in self.categorical_features:
                    new_X[:, col] = bin_X[rows, col]
            new_y = bin_y[rows] + gaps.ravel() * (bin_y[neigh_chosen] - bin_y[rows])
            X_synth.append(new_X)
            y_synth.append(new_y)
        if not X_synth:
            return np.empty((0, X.shape[1])), np.empty(0)
        return np.vstack(X_synth), np.concatenate(y_synth)


class DataOversampler(BaseSampler):  # type: ignore[misc]
    """Increase the data volume and balance the classes.

    An intelligent wrapper class around imbalanced-learn adapted for Pandas.
    It supports automatic handling of categorical features (SMOTENC) and
    type integrity preservation after synthetic generation.

    Attributes:
        multiplier (float | int): The sample multiplier.
        algorithm (str): The algorithm name ('random', 'smote', 'adasyn').
        add_noise (bool): Whether to add Gaussian noise to numeric features.
        balance (bool): Whether to bring all classes up to the majority class size.
        random_state (int | None): Random seed.
        noise_level (float): Noise intensity relative to the standard deviation.
        task (str): 'regression' (default) — bin numeric targets into
            quantiles and use the regression path for any numeric ``y``;
            'classification' — treat ``y`` as class labels.
        target_bins (int): Number of quantile bins for the regression path.
    """

    _sampling_type = "over-sampling"

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "multiplier": [float, int],
        "algorithm": [str],
        "add_noise": ["boolean"],
        "balance": ["boolean"],
        "random_state": [int, np.random.RandomState, None],
        "noise_level": [float],
        "task": [str],
        "target_bins": [int],
    }

    def __init__(
        self,
        *,
        multiplier: float = 1.0,
        algorithm: str = "random",
        add_noise: bool = False,
        balance: bool = False,
        random_state: int | None = 42,
        noise_level: float = 0.01,
        task: str = "regression",
        target_bins: int = DEFAULT_TARGET_BINS,
    ):
        # Store exactly what was passed so that clone works later
        self.multiplier = multiplier
        self.algorithm = algorithm
        self.add_noise = add_noise
        self.balance = balance
        self.random_state = random_state
        self.noise_level = noise_level
        self.task = task
        self.target_bins = target_bins

        super().__init__()
        self._lock = threading.RLock()

    def _strategy(
        self, y: pd.Series, multiplier: float, skip_missing_code: bool = False
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
        if self.balance:
            # Base is the largest class
            base_size = max(counts.values())
            return {cls: ceil(base_size * multiplier) for cls in counts}

        # Important: imbalanced-learn expects the final number of instances
        # (Total), not the delta
        return {cls: ceil(cnt * multiplier) for cls, cnt in counts.items()}

    def _add_gaussian_noise(
        self, df: pd.DataFrame, exclude_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Add Gaussian noise to numeric columns and numeric elements of object columns.
        Applies vectorized noise computation for performance.
        For non-numeric columns it attempts a safe type conversion.
        Args:
            df (pd.DataFrame): The dataset to modify.
            exclude_cols (list[str] | None): Columns to which noise should NOT be applied
                (e.g. ordinal-encoded categorical ones).
        Returns:
            pd.DataFrame: The dataset with added noise.
        """
        rng = np.random.default_rng(self.random_state)

        # 1. First force-convert "numeric strings" to numbers
        # This is critical for pandas 3.0 and StringDtype
        potential_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in potential_cols:
            # If the column contains '1.0', it becomes float.
            # If it contains 'abc', it stays object/string.
            converted = pd.to_numeric(df[col], errors="coerce")
            if not converted.isna().any():
                df[col] = converted
        # 2. Now select ALL numeric columns
        # (including the ones just converted)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Exclude columns that must not be noised (e.g. ordinal-encoded)
        if exclude_cols:
            numeric_cols = numeric_cols.difference(exclude_cols)

        if not numeric_cols.empty:
            # Vectorized noise addition for all numeric columns at once
            stds = df[numeric_cols].std().fillna(1.0).replace(0, 1.0).to_numpy()
            scales = stds * self.noise_level
            noise = rng.normal(0, 1.0, size=(len(df), len(numeric_cols))) * scales

            # Use .to_numpy() for speed
            df[numeric_cols] = df[numeric_cols].to_numpy() + noise
        return df

    def __getstate__(self) -> dict[str, Any]:
        # Copy the object state
        state = self.__dict__.copy()
        # Remove the non-serializable lock object under its correct name
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore the state
        self.__dict__.update(state)
        # Re-initialize the lock after deserialization
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Section 1: Overriding fit_resample to bypass _check_X_y            #
    # ------------------------------------------------------------------ #

    def fit_resample(self, X: Any, y: Any, **params: Any) -> tuple[Any, Any]:
        """Perform resampling while bypassing _check_X_y (validate_data).

        The parent ``BaseSampler.fit_resample`` -> ``SamplerMixin.fit_resample``
        calls ``_check_X_y`` -> ``validate_data`` -> ``check_array`` with
        ``dtype=np.float64``, which raises a ``ValueError`` on non-numeric data
        BEFORE the custom validation in ``_fit_resample`` runs.

        This override safely converts X/y and delegates directly to
        ``_fit_resample``, preserving all existing validation logic
        (algorithm support, multiplier validity, data types, etc.).

        Args:
            X: Features (DataFrame, ndarray or array-like).
            y: The target variable.
            **params: Additional parameters (accepted for compatibility with the
                sklearn interface, but unused; they are passed directly to
                _fit_resample, which does not support them).

        Returns:
            tuple: (X_resampled, y_resampled) — the resampling results.
        """
        # Basic conversion for compatibility with _fit_resample
        # Sparse matrices are not converted — this would break the format
        is_sparse = hasattr(X, "tocsr")
        if not is_sparse and not isinstance(X, (pd.DataFrame, np.ndarray)):
            X = np.asarray(X)
        if not isinstance(y, (pd.Series, np.ndarray)):
            y = np.asarray(y)
        return self._fit_resample(X, y)

    # ------------------------------------------------------------------ #
    #  Section 2: Implementation of the protected _fit_resample method    #
    # ------------------------------------------------------------------ #

    def _restore_dtypes(
        self, df: pd.DataFrame, original_dtypes: pd.Series
    ) -> pd.DataFrame:
        """Restore data types with adaptation to the changes made.

        Restoration logic:
        1. Categorical: Restored strictly (including categories and order).
        2. Integer: If add_noise=True, cast to Float (32/64) to preserve the
           fractional part of the noise. Without noise — the original Integer is returned.
        3. Object: Forced back to object to prevent a cast to float/int
           by the generation libraries.
        4. Floating: Original precision is preserved (32/64 bits).
        Args:
            df (pd.DataFrame): The transformed dataset.
            original_dtypes (pd.Series): A series with the original types (`X.dtypes`).
        Returns:
            pd.DataFrame: The dataset with restored typing.
        """
        if original_dtypes is None:
            return df

        for col, dtype in original_dtypes.items():
            if col not in df.columns:
                continue
            try:
                # 1. If it is a category — restore it strictly
                #  (important for memory and logic)
                if isinstance(dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype(dtype)

                # 2. If it is an object (strings) — return the object type
                elif (
                    dtype is object or isinstance(dtype, np.dtype) and dtype.kind == "O"
                ):
                    # If noise was added and the column became numeric,
                    # then do NOT roll back to object
                    if self.add_noise and is_numeric_dtype(df[col]):
                        pass
                    else:
                        df[col] = df[col].astype(object)

                # 3. If it is an integer:
                elif np.issubdtype(dtype, np.integer):
                    # If noise was added, we MUST NOT return int, or precision is lost
                    if not self.add_noise:
                        # Use pd.to_numeric for safety
                        # and cast to the original int type
                        converted = pd.to_numeric(df[col], errors="coerce")
                        df[col] = (
                            converted.astype(dtype)
                            if not converted.isna().any()
                            else converted
                        )
                    else:
                        # PRECISION CHOICE: int64 -> float64, otherwise float32
                        if dtype == np.int64:
                            df[col] = df[col].astype(np.float64)
                        else:
                            df[col] = df[col].astype(np.float32)
                # 4. If the original type is a floating point (float32 or float64):
                elif np.issubdtype(dtype, np.floating):
                    # Preserve the original type
                    # (if it was float64 -> float64, if float32 -> float32)
                    df[col] = df[col].astype(dtype)

                # 5. Otherwise (float, etc.) try to return the original type
                else:
                    if not pd.Series(df[col]).isna().any():
                        df[col] = df[col].astype(dtype)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to restore type for column {col}: {e}")
        return df

    def _fit_resample(self, X: Any, y: Any) -> tuple[Any, np.ndarray]:
        """Resample the data according to the selected algorithm.

        The engine is regression-only: by default (``task="regression"``)
        ANY numeric target (float or integer — counts, whole-unit prices,
        etc.) is discretized into ``target_bins`` quantile bins, the bins
        are resampled as pseudo-classes, and the real target values are
        restored afterwards. The classification path is used only for
        explicit ``task="classification"`` calls or for non-numeric targets.

        For ``random`` each duplicated row keeps its source row's real
        target (``RandomOverSampler.sample_indices_``). For ``smote`` /
        ``adasyn`` synthetic rows are generated SMOTER-style
        (:class:`_SMOTERGenerator`): the features AND the target are
        interpolated between a row and a same-bin neighbor, so the synthetic
        ``y`` values are honest (they lie between the two parents) instead of
        collapsing onto bin midpoints.

        Automatically switches to SMOTENC if non-numeric columns are detected.
        For ADASYN a transparent fallback to SMOTENC is implemented when categories are present,
        because the original ADASYN only works with numeric data.

        Args:
            X (Any): Features (array, DataFrame or sparse matrix).
            y (Any): The target variable.
        Returns:
            tuple: A tuple (X_resampled, y_resampled). For a sparse input
            (output of hashing encoding) X_resampled stays a
            ``scipy.sparse.csr_matrix`` for the 'random' algorithm and becomes
            dense for SMOTE/ADASYN (they require dense data).
        Raises:

            ValueError: If `multiplier` < 1, the algorithm is not supported,
                or ``target_bins`` < 2.
            TypeError: If the data has no numeric features (needed for
                       distance computation in SMOTE/ADASYN) or when
                       attempting to add noise to a sparse matrix.
        """
        with self._lock:
            # Store the result in an attribute with a leading underscore
            # (sklearn standard)
            self.multiplier_ = float(self.multiplier)
            # Use a local variable instead of mutating self.algorithm
            algo_local = self.algorithm.lower().replace(" ", "_")

        if self.multiplier_ < 1:
            raise ValueError("multiplier must be >= 1")
        if self.target_bins < MIN_TARGET_BINS:
            raise ValueError(
                f"target_bins must be >= {MIN_TARGET_BINS}, got {self.target_bins}"
            )
        if algo_local not in ("random", "smote", "adasyn"):
            raise ValueError(f"Unsupported algorithm: {algo_local}")

        # Evidence gate (T13): when the nightly benchmark has shown that
        # oversampling degrades the held-out R² beyond the threshold, the
        # feature is experimental — warn whenever the oversampler actually
        # generates rows (multiplier > 1, or balancing that brings minority
        # groups up to the majority size even at multiplier == 1).
        if (self.multiplier_ > 1.0 or self.balance) and OVERSAMPLING_EXPERIMENTAL:
            logger.warning(
                "Oversampling is EXPERIMENTAL: the nightly evidence benchmark "
                "observed a median held-out R² degradation exceeding 0.01. "
                "Disable oversampling unless you have verified the effect on "
                "your own data."
            )

        # Check for sparse matrices (scipy.sparse)
        is_sparse = hasattr(X, "tocsr")

        # Sparse matrices are incompatible with noise addition.
        # Adding Gaussian noise turns every zero element into a non-zero one,
        # which causes an explosive growth of memory usage (densification).
        if self.add_noise and is_sparse:
            raise TypeError(
                f"The '{self.algorithm}' algorithm with "
                "noise injection does not support "
                "sparse matrices. Please convert your data to a dense format "
                "using X.toarray() or set add_noise=False."
            )

        try:
            # Standard input data validation
            y_validated = np.asarray(y)
            y_s = pd.Series(y_validated)

            # Regression mode (default): the engine never treats the target
            # as a class label. ANY numeric target (float or integer) is
            # discretized into quantile bins; the bins are resampled as
            # pseudo-classes and the real values are restored afterwards.
            # The classification path is used only for explicit
            # task="classification" calls or for non-numeric targets.
            is_regression = self.task.lower() != "classification"
            if is_regression:
                is_regression, y_fit, bin_mids = _bin_regression_target(
                    y_s, self.target_bins
                )
            else:
                y_fit = y_s
                bin_mids = {}

            # Sparse branch (output of hashing encoding, csr_matrix):
            # the data is already numeric, there are no categorical columns
            # (OrdinalEncoder is not needed) and pd.DataFrame(csr) would create a
            # silently garbage dataframe object. RandomOverSampler works with sparse
            # directly (memory-efficient); SMOTE/ADASYN mathematically require
            # dense distance matrices, so for them a sparse input
            # is converted to dense (as before the hashing switch to sparse).
            if is_sparse:
                strategy_sparse = self._strategy(
                    y_fit, self.multiplier_, skip_missing_code=is_regression
                )
                if algo_local == "random":
                    sampler = RandomOverSampler(
                        sampling_strategy=strategy_sparse,
                        random_state=self.random_state,
                    )
                    X_res_raw, y_res = sampler.fit_resample(X, y_fit)
                    logger.info(
                        "%s resample: %d -> %d",
                        algo_local.upper(),
                        X.shape[0],
                        X_res_raw.shape[0],
                    )
                    if is_regression:
                        return X_res_raw, self._decode_regression_y(
                            y_res, y_s, y_fit, bin_mids, sampler
                        )
                    return X_res_raw, y_res.to_numpy()
                if algo_local in ("smote", "adasyn"):
                    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                else:
                    raise ValueError(f"Unsupported algorithm: {algo_local}")

            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

            # Determine the indices of non-numeric columns for SMOTENC.
            # Columns with numeric strings ("1.0", "2.0") are not
            # categorical — they will be converted to numbers.
            cat_features_idx = [
                i
                for i, col in enumerate(X_df.columns)
                if _is_categorical_col(X_df[col])
            ]
            cat_cols = [X_df.columns[i] for i in cat_features_idx]
            # Check for the absence of numeric columns for SMOTE/ADASYN
            if algo_local in ["smote", "adasyn"] and len(cat_features_idx) == len(
                X_df.columns
            ):
                raise TypeError(
                    f"{self.algorithm.upper()} requires at least one numeric feature."
                )

            # For all algorithms the full dataset is passed in the current logic
            X_to_resample = X_df

            # Determine the algorithm and perform resampling
            strategy = self._strategy(
                y_fit, self.multiplier_, skip_missing_code=is_regression
            )

            if algo_local == "random":
                sampler = RandomOverSampler(
                    sampling_strategy=strategy, random_state=self.random_state
                )
                # Encode categorical features to pass
                # _check_X_y inside RandomOverSampler, which forcibly
                # converts the DataFrame to float.
                if cat_features_idx:
                    ordinal_encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    X_to_resample = X_to_resample.copy()
                    X_to_resample[cat_cols] = ordinal_encoder.fit_transform(
                        X_to_resample[cat_cols]
                    )
                else:
                    ordinal_encoder = None
                X_res_raw, y_res = sampler.fit_resample(X_to_resample, y_fit)
                random_sampler: RandomOverSampler | None = sampler
            else:
                ordinal_encoder = None
                random_sampler = None
                counts = Counter(y_fit)
                # "-1" (missing target in the regression mode) is a
                # pass-through pseudo-class: it is excluded from the strategy
                # AND from the k_neighbors estimation.
                if counts and is_regression:
                    skip_missing = True
                else:
                    skip_missing = False
                active_counts = Counter(
                    {
                        cls: cnt
                        for cls, cnt in counts.items()
                        if not (skip_missing and cls == -1)
                    }
                )
                if not active_counts:
                    raise ValueError(
                        "SMOTE/ADASYN require at least one non-missing target value."
                    )

                if is_regression:
                    # Regression mode: only bins with at least two rows can
                    # produce a same-bin neighbor pair; a singleton bin (or a
                    # bin that is entirely too small for the k-NN) passes
                    # through unchanged.
                    eligible = dict(active_counts)
                    growth_eligible = {
                        cls: cnt for cls, cnt in eligible.items() if cnt >= 2
                    }
                    if self.balance:
                        strategy_final = {
                            cls: tgt
                            for cls, tgt in strategy.items()
                            if cls in growth_eligible and cls != -1
                        }
                    else:
                        # Proportional mode: keep only the bins that can
                        # actually be synthesized.
                        strategy_final = {
                            cls: tgt
                            for cls, tgt in strategy.items()
                            if cls in growth_eligible
                        }
                    # Number of synthetic rows per bin = final - original count.
                    counts_binned = Counter(y_fit)
                    n_to_generate: dict[int, int] = {}
                    for cls, tgt in strategy_final.items():
                        orig = counts_binned.get(cls, 0)
                        n_to_generate[int(cls)] = max(0, int(tgt) - orig)
                    # Encode categorical columns numerically before the
                    # distance computation (SMOTENC-style); the categories are
                    # decoded back by the shared code below through
                    # ``ordinal_encoder``.
                    X_syn_source = X_df
                    if cat_features_idx:
                        ordinal_encoder = OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        )
                        X_syn_source = X_syn_source.copy()
                        X_syn_source[cat_cols] = ordinal_encoder.fit_transform(
                            X_syn_source[cat_cols]
                        )
                    X_np = X_syn_source.to_numpy(dtype=float)
                    y_np = np.asarray(y_s, dtype=float)
                    y_fit_np = np.asarray(y_fit)
                    generator = _SMOTERGenerator(
                        random_state=self.random_state,
                        categorical_features=cat_features_idx,
                        density_weighted=(algo_local == "adasyn"),
                    )
                    X_synth, y_synth = generator.fit_resample(
                        X_np, y_np, y_fit_np, n_to_generate
                    )
                    X_res_raw = np.vstack([X_np, X_synth]) if len(X_synth) else X_np
                    y_res = np.concatenate([y_np, y_synth]) if len(y_synth) else y_np
                else:
                    # Classification mode: keep all classes in the strategy and
                    # derive k from the smallest class count. A singleton
                    # minority class is left to imbalanced-learn, which raises
                    # an informative ValueError (the documented imblearn
                    # contract).
                    min_samples = min(active_counts.values())
                    k_neighbors = max(1, min(5, min_samples - 1))

                    if algo_local == "smote":
                        if cat_features_idx:
                            from imblearn.over_sampling import SMOTENC

                            sampler = SMOTENC(
                                categorical_features=cat_features_idx,
                                sampling_strategy=strategy,
                                k_neighbors=k_neighbors,
                                random_state=self.random_state,
                            )
                        else:
                            from imblearn.over_sampling import SMOTE

                            sampler = SMOTE(
                                sampling_strategy=strategy,
                                k_neighbors=k_neighbors,
                                random_state=self.random_state,
                            )
                    elif algo_local == "adasyn":
                        if cat_features_idx:
                            # IMPORTANT: Log the substitution to avoid misleading
                            logger.warning(
                                "ADASYN does not support categorical columns. "
                                "Automatically falling back to SMOTENC for data integrity."
                            )
                            from imblearn.over_sampling import ADASYN, SMOTENC

                            sampler = SMOTENC(
                                categorical_features=cat_features_idx,
                                sampling_strategy=strategy,
                                k_neighbors=k_neighbors,
                                random_state=self.random_state,
                            )
                        else:
                            from imblearn.over_sampling import ADASYN

                            sampler = ADASYN(
                                sampling_strategy=strategy,
                                n_neighbors=k_neighbors,
                                random_state=self.random_state,
                            )
                    else:
                        raise ValueError(f"Unsupported algorithm: {algo_local}")
                    X_res_raw, y_res = sampler.fit_resample(X_to_resample, y_fit)

            # Cast to DF since SMOTE may return numpy
            X_res_df = pd.DataFrame(X_res_raw, columns=X_df.columns)[X_df.columns]

            # A single point of noise application for all algorithms
            # Categorical columns are excluded from the noise so as not to break
            # the subsequent decoding via OrdinalEncoder.inverse_transform
            if self.add_noise:
                noise_exclude = cat_cols if ordinal_encoder is not None else None
                X_res_df = self._add_gaussian_noise(
                    X_res_df, exclude_cols=noise_exclude
                )

            # Decode categorical columns back (only for random)
            if ordinal_encoder is not None:
                X_res_df[cat_cols] = ordinal_encoder.inverse_transform(
                    X_res_df[cat_cols]
                )

            logger.info(
                "%s resample: %d -> %d", algo_local.upper(), len(X_df), len(X_res_df)
            )

            if is_regression:
                return X_res_df.to_numpy(), self._decode_regression_y(
                    y_res, y_s, y_fit, bin_mids, random_sampler
                )
            return X_res_df.to_numpy(), np.asarray(y_res)
        except (ValueError, TypeError, ImportError, RuntimeError):
            # Re-raise known data-related or configuration errors directly
            raise
        except Exception as e:
            logger.debug("Unexpected error trace in _fit_resample:", exc_info=True)
            logger.error(f"Critical error during data oversampling: {e}")
            raise

    def _decode_regression_y(
        self,
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

    def oversample(self, data: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """Increase the sample as a DataFrame while preserving metadata.
        Args:
            data (pd.DataFrame): The source DataFrame.
            target (str, optional): The name of the target column. If None, only the
                'random' algorithm is possible (duplicating the whole sample). For SMOTE/ADASYN
                and balancing the parameter is required.
        Returns:
            pd.DataFrame: The resulting DataFrame with new rows.
        Raises:
            ValueError: If `target` is not specified with balancing enabled
                or when using SMOTE/ADASYN algorithms.
        """

        try:
            algo_local = self.algorithm.lower().replace(" ", "_")

            valid_algorithms = ["smote", "adasyn", "random"]
            if algo_local not in valid_algorithms:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

            # CHECK: If the target is not specified, balancing
            # or synthetic algorithms cannot be used
            if target is None and (self.balance or algo_local in ("smote", "adasyn")):
                raise ValueError(
                    f"The 'target' parameter must be specified for "
                    f"the '{self.algorithm}' algorithm "
                    f"or when balance=True. Without a target, only 'random' "
                    f"oversampling of the entire dataset is possible."
                )

            # EARLY CHECK: For SMOTE/ADASYN all features must be
            # numeric (at least one). This check raises a TypeError
            # BEFORE the call to self.fit_resample(), preventing imblearn from
            # crashing with an uninformative ValueError at the _check_X_y stage.
            # Use the same category detection as in _fit_resample,
            # so that 'numeric strings' are not treated as categorical.
            feature_cols = [col for col in data.columns if col != target]
            if algo_local in ("smote", "adasyn") and all(
                _is_categorical_col(data[col]) for col in feature_cols
            ):
                raise TypeError(
                    f"{self.algorithm.upper()} requires at least one numeric feature."
                )

            # If the target is not specified, create a dummy vector for
            # compatibility with the imblearn API,
            # which allows 'random' oversampling
            # of the whole table without labels.
            if target is None and algo_local == "random":
                X = data
                # Create a dummy y
                y_array = np.zeros(len(data), dtype=int)
                if len(y_array) > 1:
                    y_array[0] = 1
                y = pd.Series(y_array, name="temp_target")
            else:
                y = data[target]
                X = data.drop(target, axis=1)

            original_dtypes = X.dtypes
            target_dtype = y.dtype
            # 3. Call the public resampling method
            X_res_np, y_res_np = self.fit_resample(X, y)
            # 4. Assemble the final DataFrame
            res_df = pd.DataFrame(X_res_np, columns=X.columns)

            # 5. Restore data types (fix for lost types)
            res_df = self._restore_dtypes(res_df, original_dtypes)

            if target is not None:
                y_res_s = pd.Series(y_res_np)
                if self.task.lower() == "classification":
                    # The classification path restores the exact class labels.
                    res_df[target] = y_res_s.astype(target_dtype)
                elif is_numeric_dtype(target_dtype):
                    # The regression path returns the honest targets: original
                    # rows keep their real values, synthetic rows carry the
                    # SMOTER-interpolated (float) values. Casting the whole
                    # vector back to an integer dtype would silently round the
                    # interpolation away (the main point of the feature) and
                    # crash on nullable integer dtypes with NaN — so the
                    # result is float64 (NaN preserved).
                    res_df[target] = y_res_s.astype(np.float64)
                else:
                    # Non-numeric targets (strings, etc.) are handled on the
                    # classification fallback path inside _fit_resample: the
                    # original dtype is restored as-is.
                    res_df[target] = y_res_s.astype(target_dtype)

            return res_df.reset_index(drop=True)

        except Exception:
            logger.exception("Oversampling error")
            raise


# ------------------------------------------------------------------ #
#  Functional interfaces                                             #
# ------------------------------------------------------------------ #


def oversample(
    data: pd.DataFrame,
    multiplier: float = 1.0,
    algorithm: str = "random",
    add_noise: bool = False,
    balance: bool | None = False,
    target: str | None = None,
    random_state: int | None = 42,
    noise_level: float = 0.01,
    task: str = "regression",
    target_bins: int = DEFAULT_TARGET_BINS,
) -> pd.DataFrame:
    """Increase the DataFrame volume through a functional interface.
    Creates a temporary `DataOversampler` instance and applies it to the data.
    Args:
        data (pd.DataFrame): The input dataset.
        multiplier (float): The row multiplier. Defaults to 1.0.
        algorithm (str): The algorithm ('random', 'smote', 'adasyn'). Defaults to 'random'.
        add_noise (bool): Whether to add noise. Defaults to False.
        balance (bool, optional): Whether to balance classes. Defaults to False.
        target (str, optional): The name of the target column. Defaults to None.
        random_state (int, optional): The randomness seed. Defaults to 42.
        noise_level (float): The noise level. Defaults to 0.01.
        task (str): 'regression' (default) or 'classification'.
        target_bins (int): Number of quantile bins for the regression path.
    Returns:
        pd.DataFrame: The expanded dataset.
    """
    sampler = DataOversampler(
        multiplier=multiplier,
        algorithm=algorithm,
        add_noise=add_noise,
        balance=balance if balance is not None else False,
        random_state=random_state,
        noise_level=noise_level,
        task=task,
        target_bins=target_bins,
    )
    return sampler.oversample(data, target=target)
