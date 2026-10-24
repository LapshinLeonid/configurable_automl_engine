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
"""

import logging
import threading
from collections import Counter
from math import ceil
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder

#: Number of quantile bins used to discretize a continuous regression target
#: before class-based oversampling. The binned groups imitate (and balance)
#: "classes" of a regression target; after resampling the bins are decoded
#: back into the original continuous values.
CONTINUOUS_TARGET_QUANTILES = 8

#: Minimum number of "classes" a target must have to be considered
#: continuous: pd.qcut of a target with fewer effective quantiles produces
#: fewer bins (Peter-Hall estimator), and imbalanced-learn needs at least two
#: groups to resample.
CONTINUOUS_TARGET_MIN_BINS = 2

logger = logging.getLogger(__name__)


def _bin_continuous_target(
    y: pd.Series,
) -> tuple[bool, pd.Series, dict[int, float]]:
    """Detect a continuous regression target and discretize it into bins.

    Runs a single quantile discretization and reuses it for both the
    continuous detection and the backward decoding, so the hot path
    (:meth:`DataOversampler._fit_resample`) bins the target only once.

    Args:
        y (pd.Series): The target variable vector.

    Returns:
        tuple[bool, pd.Series, dict[int, float]]: A triple of
            (is_continuous, codes, bin_mids). When the target is not
            continuous — or cannot be discretized into at least
            :data:`CONTINUOUS_TARGET_MIN_BINS` groups — returns
            ``(False, y, {})`` so the caller keeps the classification path.
    """
    if y.empty or not is_numeric_dtype(y.dtype):
        return False, y, {}
    if is_bool_dtype(y.dtype):
        return False, y, {}
    finite = y.dropna()
    if finite.empty:
        return False, y, {}
    if not np.issubdtype(np.asarray(finite).dtype, np.floating):
        return False, y, {}
    # A float target that contains any non-integer-representable value is
    # "continuous" for sklearn/imblearn and must be binned. Integer-valued
    # floats (``0.0``/``1.0``, ``0.0``/``1.0``/``2.0``, ...) are valid
    # multiclass targets and stay on the classification path.
    values = np.asarray(finite, dtype=float)
    if not np.all(np.isfinite(values)):
        return False, y, {}
    if np.all(values == np.floor(values)):
        return False, y, {}
    try:
        finite_bins = pd.qcut(finite, q=CONTINUOUS_TARGET_QUANTILES, duplicates="drop")
    except (TypeError, ValueError):
        return False, y, {}
    if finite_bins.cat.categories.size < CONTINUOUS_TARGET_MIN_BINS:
        return False, y, {}

    finite_mask = y.notna()
    codes = np.asarray(finite_bins.cat.codes, dtype=np.int8)
    binned: pd.Series = pd.Series(
        np.full(len(y), -1, dtype=np.int8), index=y.index, name=y.name
    )
    binned.iloc[np.asarray(finite_mask)] = codes
    bin_mids: dict[int, float] = {
        int(code): float(interval.mid)
        for code, interval in enumerate(finite_bins.cat.categories)
    }
    return True, binned, bin_mids


def _is_continuous_target(y: pd.Series) -> bool:
    """Determine whether a target vector is continuous regression data.

    The classification-style samplers of imbalanced-learn reject float
    targets that contain at least one non-integer value
    (``ValueError: Unknown label type: continuous`` — sklearn's
    ``type_of_target`` rule). A target is considered continuous exactly when
    imbalanced-learn would reject it: a float dtype (``NaN`` allowed — the
    natural regression case) with a finite value that is not integer-
    representable (e.g. ``2.5``). Integer dtypes, bools and integer-valued
    floats (``0.0``/``1.0`` binary targets, ``0.0``/``1.0``/``2.0``
    multiclass targets, ...) are valid classification targets and stay on
    the classification path. The binned path additionally requires a
    quantile discretization into at least
    :data:`CONTINUOUS_TARGET_MIN_BINS` groups.

    Args:
        y (pd.Series): The target variable vector.

    Returns:
        bool: ``True`` if the target should be treated as continuous.
    """
    return _bin_continuous_target(y)[0]


def _prepare_continuous_bins(y: pd.Series) -> tuple[pd.Series, dict[int, float]]:
    """Discretize a continuous target into quantile bins (class codes).

    The bin codes are computed on the finite target values; rows with a
    missing target keep code ``-1`` and are passed through the samplers
    unchanged (like an extra class that is not resampled). The selected
    number of bins is derived from the quantile discretization itself
    (``duplicates="drop"``), so the same bin structure is used both for the
    resampling strategy and for the backward decoding.

    Args:
        y (pd.Series): The continuous target vector.

    Returns:
        tuple[pd.Series, dict[int, float]]: A pair of the integer (``int8``)
            code series (the quantile bin number per row, ``-1`` for a
            missing target) and the per-bin representative values (interval
            midpoints) used to restore continuous targets after resampling.

    Raises:
        ValueError: If ``y`` cannot be discretized into at least
            :data:`CONTINUOUS_TARGET_MIN_BINS` quantile bins.
    """
    is_continuous, binned, bin_mids = _bin_continuous_target(y)
    if not is_continuous:
        raise ValueError(
            "a continuous target with at least "
            f"{CONTINUOUS_TARGET_MIN_BINS} quantile bins is required"
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
    """

    _sampling_type = "over-sampling"

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "multiplier": [float, int],
        "algorithm": [str],
        "add_noise": ["boolean"],
        "balance": ["boolean"],
        "random_state": [int, np.random.RandomState, None],
        "noise_level": [float],
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
    ):
        # Store exactly what was passed so that clone works later
        self.multiplier = multiplier
        self.algorithm = algorithm
        self.add_noise = add_noise
        self.balance = balance
        self.random_state = random_state
        self.noise_level = noise_level

        super().__init__()
        self._lock = threading.RLock()

    def _strategy(
        self, y: pd.Series, multiplier: float, skip_missing_code: bool = False
    ) -> dict[Any, int]:
        """Compute the target number of instances for each class.

        For a classification target (or the pseudo-classes of a binned
        continuous target) the values are the *final* per-class row counts;
        the overall result size is therefore approximately
        ``N * multiplier`` (exact for ``balance=False``, where each class is
        scaled independently). For a binned continuous target, classes that
        end up smaller than ``ceil(count * multiplier)`` (e.g. singleton
        bins that cannot be synthesized) are dropped from the strategy by
        the caller and pass through unchanged, so the final size may be
        smaller than ``N * multiplier`` — this is an intentional
        "balance-like" behavior, not a bug.

        Args:
            y (pd.Series): The target variable vector.
            multiplier (float): The scaling factor.
            skip_missing_code (bool): Whether to exclude the pseudo-class
                ``-1`` (rows with a missing target in the continuous mode)
                from the strategy — it must never be rebalanced or scaled
                (pass-through only).
        Returns:
            Dict[Any, int]: A dictionary where keys are class labels
                and values are the final number of rows.
        """
        counts = Counter(y)
        # The pseudo-class "-1" is reserved for rows with a missing target in
        # the continuous mode: imbalanced-learn treats it as a real class, so
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
        Automatically switches to SMOTENC if non-numeric columns are detected.
        For ADASYN a transparent fallback to SMOTENC is implemented when categories are present,
        because the original ADASYN only works with numeric data.
        For SMOTE/ADASYN: Dynamic calculation of k_neighbors.
        Prevents ValueError if the number of samples in the minority class
        is smaller than the standard value (5). It is set as min(5, n_samples-1).
        A continuous (regression) target is detected automatically and
        oversampled through quantile bins: the target is discretized into
        equal-frequency bins (see :data:`CONTINUOUS_TARGET_QUANTILES`), the
        bins are resampled as pseudo-classes, and the real continuous values
        are restored afterwards (original rows keep their values, synthetic
        rows receive the midpoint of their bin). This replaces the previous
        failure when every unique continuous ``y`` value was treated as a
        separate class, which imbalanced-learn rejects with
        ``ValueError: Unknown label type: continuous``.
        Args:
            X (Any): Features (array, DataFrame or sparse matrix).
            y (Any): The target variable.
        Returns:
            tuple: A tuple (X_resampled, y_resampled). For a sparse input
            (output of hashing encoding) X_resampled stays a
            ``scipy.sparse.csr_matrix`` for the 'random' algorithm and becomes
            dense for SMOTE/ADASYN (they require dense data).
        Raises:

            ValueError: If `multiplier` < 1 or the algorithm is not supported.
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

            # Continuous-target mode: the imblearn samplers reject continuous
            # labels ("Unknown label type: continuous"). Discretize the target
            # into quantile bins, resample the bins as pseudo-classes and
            # restore the real continuous values afterwards. A single
            # quantile discretization is reused for the detection and the
            # binning.
            is_continuous, y_fit, bin_mids = _bin_continuous_target(y_s)

            # Sparse branch (output of hashing encoding, csr_matrix):
            # the data is already numeric, there are no categorical columns
            # (OrdinalEncoder is not needed) and pd.DataFrame(csr) would create a
            # silently garbage dataframe object. RandomOverSampler works with sparse
            # directly (memory-efficient); SMOTE/ADASYN mathematically require
            # dense distance matrices, so for them a sparse input
            # is converted to dense (as before the hashing switch to sparse).
            if is_sparse:
                strategy_sparse = self._strategy(
                    y_fit, self.multiplier_, skip_missing_code=is_continuous
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
                    if is_continuous:
                        return X_res_raw, _decode_continuous_y(
                            y_res, y_s, y_fit, bin_mids
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
                y_fit, self.multiplier_, skip_missing_code=is_continuous
            )

            if algo_local == "random":
                sampler = RandomOverSampler(
                    sampling_strategy=strategy, random_state=self.random_state
                )
                # Encode categorical features to pass
                # _check_X_y inside RandomOverSampler, which forcibly
                # converts the DataFrame to float.
                ordinal_encoder: OrdinalEncoder | None = None
                if cat_features_idx:
                    ordinal_encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    X_to_resample = X_to_resample.copy()
                    X_to_resample[cat_cols] = ordinal_encoder.fit_transform(
                        X_to_resample[cat_cols]
                    )
                y_to_fit: Any = y_fit
            else:
                ordinal_encoder = None
                # SMOTE / ADASYN case
                counts = Counter(y_fit)
                # "-1" (missing target in the continuous mode) is a
                # pass-through pseudo-class: it is excluded from the strategy
                # AND from the k_neighbors estimation.
                skip_missing = is_continuous
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

                if is_continuous:
                    # Continuous mode: quantile-bin pseudo-classes that are
                    # too small for the current k cannot be synthesized.
                    # imblearn fits a k-NN with (k_neighbors + 1) points per
                    # resampled class and drops the point itself, so a class
                    # with n samples can only be synthesized when
                    # n >= k_neighbors + 1. Such classes are excluded from
                    # the strategy and pass through unchanged; k is recomputed
                    # from the minimum of the remaining classes until the set
                    # stabilizes. When every class is too small (e.g. all
                    # quantile bins are singletons) the strategy becomes empty
                    # and the data is kept unchanged.
                    strategy = {
                        cls: tgt
                        for cls, tgt in strategy.items()
                        if not (skip_missing and cls == -1)
                    }
                    eligible = dict(active_counts)
                    k_neighbors = 1
                    while eligible:
                        k_candidate = min(5, min(eligible.values()) - 1)
                        k_candidate = max(1, k_candidate)
                        too_small = {
                            cls: cnt
                            for cls, cnt in eligible.items()
                            if cnt <= k_candidate
                        }
                        if not too_small:
                            k_neighbors = k_candidate
                            break
                        eligible = {
                            cls: cnt
                            for cls, cnt in eligible.items()
                            if cnt > k_candidate
                        }
                    strategy = {
                        cls: tgt for cls, tgt in strategy.items() if cls in eligible
                    }
                else:
                    # Classification mode: keep all classes in the strategy and
                    # derive k from the smallest class count. A singleton
                    # minority class is left to imbalanced-learn, which raises
                    # a ValueError (the documented imblearn contract).
                    min_samples = min(active_counts.values())
                    k_neighbors = max(1, min(5, min_samples - 1))

                if algo_local == "smote":
                    if cat_features_idx:
                        sampler = SMOTENC(
                            categorical_features=cat_features_idx,
                            sampling_strategy=strategy,
                            k_neighbors=k_neighbors,
                            random_state=self.random_state,
                        )
                    else:
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
                        sampler = SMOTENC(
                            categorical_features=cat_features_idx,
                            sampling_strategy=strategy,
                            k_neighbors=k_neighbors,
                            random_state=self.random_state,
                        )
                    else:
                        sampler = ADASYN(
                            sampling_strategy=strategy,
                            n_neighbors=k_neighbors,
                            random_state=self.random_state,
                        )
                else:
                    raise ValueError(f"Unsupported algorithm: {algo_local}")
                y_to_fit = y_fit

            X_res_raw, y_res = sampler.fit_resample(X_to_resample, y_to_fit)
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

            if is_continuous:
                return X_res_df.to_numpy(), _decode_continuous_y(
                    y_res, y_s, y_fit, bin_mids
                )
            return X_res_df.to_numpy(), y_res.to_numpy()
        except (ValueError, TypeError, ImportError, RuntimeError):
            # Re-raise known data-related or configuration errors directly
            raise
        except Exception as e:
            logger.debug("Unexpected error trace in _fit_resample:", exc_info=True)
            logger.error(f"Critical error during data oversampling: {e}")
            raise

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
                res_df[target] = pd.Series(y_res_np).astype(target_dtype)

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
    )
    return sampler.oversample(data, target=target)
