"""The oversampler: ``DataOversampler`` and the functional ``oversample``.

Holds :class:`DataOversampler` — the imbalanced-learn sampler that expands
and balances the data (random, SMOTE/SMOTENC, ADASYN, the SMOTER path for a
regression target), adds optional Gaussian noise and restores the dtypes —
and the functional wrapper :func:`oversample`. ``DataOversampler`` is
pickled inside saved models, so its module path is part of the artifact
format (``trainer_state.ARTIFACT_FORMAT_VERSION``). It builds on the other
oversampling modules; none of them may import it.
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
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder

from configurable_automl_engine.oversampling import experimental
from configurable_automl_engine.oversampling.columns import (
    _binary_indicator_columns,
    _is_categorical_col,
)
from configurable_automl_engine.oversampling.experimental import (
    warn_experimental_once,
)
from configurable_automl_engine.oversampling.smoter import _SMOTERGenerator
from configurable_automl_engine.oversampling.targets import (
    DEFAULT_TARGET_BINS,
    MIN_TARGET_BINS,
    _bin_regression_target,
    _BinnedTarget,
    _decode_continuous_y,
)

logger = logging.getLogger(__name__)


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
        self,
        df: pd.DataFrame,
        exclude_cols: list[str] | None = None,
        *,
        nonzero_only: bool = False,
    ) -> pd.DataFrame:
        """Add Gaussian noise to numeric columns and numeric elements of object columns.
        Applies vectorized noise computation for performance.
        For non-numeric columns it attempts a safe type conversion.
        Args:
            df (pd.DataFrame): The dataset to modify.
            exclude_cols (list[str] | None): Columns to which noise should NOT be applied
                (e.g. ordinal-encoded categorical ones).
            nonzero_only (bool): Noise only the non-zero cells (zeros stay
                exact zeros). Used for data that arrived as a sparse matrix
                and was densified for SMOTE/ADASYN, so the result matches
                :meth:`_add_sparse_gaussian_noise`.
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
            values = df[numeric_cols].to_numpy()
            if nonzero_only:
                noise = noise * (values != 0)
            df[numeric_cols] = values + noise
        return df

    def _add_sparse_gaussian_noise(self, X: Any) -> Any:
        """Add Gaussian noise to the stored non-zero elements of a sparse matrix.

        Noising every cell would turn each zero into a non-zero value and
        densify the matrix (memory explosion on hashing / one-hot output);
        besides, a zero there means "feature absent" and should stay exact.
        So only the explicitly stored non-zero values are perturbed: the
        sparsity pattern is preserved and the cost is O(nnz). Binary indicator
        columns (at most two distinct values, implicit zeros included — one-hot
        / hashing output, and also scaled 0/1 flags and missing-value
        indicators, see :func:`_binary_indicator_columns`) are left untouched
        as well: a noised "1.003" is not a category. The decision is made on
        ``X`` itself, which is valid because random oversampling only
        duplicates rows. The noise scale
        of a column is ``noise_level`` times its standard deviation computed
        over ALL rows (zeros included, ddof=1), the same as the dense path.

        Args:
            X: A scipy.sparse matrix.

        Returns:
            Any: A new CSR matrix (float64) with noised non-zero elements.
        """
        rng = np.random.default_rng(self.random_state)
        X_csr = X.tocsr().astype(np.float64, copy=True)
        # Explicitly stored zeros must stay zeros
        X_csr.eliminate_zeros()
        n_rows = X_csr.shape[0]
        if X_csr.nnz == 0 or n_rows == 0:
            return X_csr

        mean = np.asarray(X_csr.mean(axis=0)).ravel()
        mean_sq = np.asarray(X_csr.multiply(X_csr).mean(axis=0)).ravel()
        var = np.maximum(mean_sq - mean**2, 0.0)
        if n_rows > 1:
            var = var * n_rows / (n_rows - 1)
        stds = np.sqrt(var)
        stds[~np.isfinite(stds) | (stds == 0)] = 1.0
        scales = stds * self.noise_level

        # Columns with more than two distinct values (an implicit zero counts
        # as one) carry real magnitudes and are noised; two-valued columns are
        # binary indicators, possibly shifted/scaled by the preprocessor
        n_cols = X_csr.shape[1]
        order = np.lexsort((X_csr.data, X_csr.indices))
        cols_sorted = X_csr.indices[order]
        data_sorted = X_csr.data[order]
        is_new = np.ones(cols_sorted.size, dtype=bool)
        is_new[1:] = (cols_sorted[1:] != cols_sorted[:-1]) | (
            data_sorted[1:] != data_sorted[:-1]
        )
        n_distinct = np.bincount(cols_sorted[is_new], minlength=n_cols)
        n_distinct += np.bincount(X_csr.indices, minlength=n_cols) < n_rows
        not_binary = n_distinct > 2
        noise = rng.normal(0.0, 1.0, size=X_csr.nnz) * scales[X_csr.indices]
        X_csr.data += noise * not_binary[X_csr.indices]
        return X_csr

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
                       distance computation in SMOTE/ADASYN).
        """
        multiplier, target_bins, algo_local = self._validated_params()

        # Evidence gate: when the nightly benchmark has shown that
        # oversampling degrades the held-out R² beyond the threshold, the
        # feature is experimental — warn (once per configuration) when the
        # oversampler actually generates rows (multiplier > 1, or balancing
        # that brings minority groups up to the majority size even at
        # multiplier == 1).
        if (
            multiplier > 1.0 or self.balance
        ) and experimental.OVERSAMPLING_EXPERIMENTAL:
            warn_experimental_once(algorithm=algo_local, multiplier=multiplier)

        try:
            return self._resample(X, y, algo_local, multiplier, target_bins)
        except (ValueError, TypeError, ImportError, RuntimeError):
            # Re-raise known data-related or configuration errors directly
            raise
        except Exception as e:
            logger.debug("Unexpected error trace in _fit_resample:", exc_info=True)
            logger.error(f"Critical error during data oversampling: {e}")
            raise

    def _validated_params(self) -> tuple[float, int, str]:
        """Validate the sampler parameters.

        Stores ``multiplier_`` (sklearn standard: a fitted attribute with a
        trailing underscore) and normalizes the algorithm name into a local
        value instead of mutating ``self.algorithm``.

        Returns:
            tuple[float, int, str]: (multiplier, target_bins, algorithm).

        Raises:
            ValueError: If ``multiplier`` < 1, ``target_bins`` < 2 or the
                algorithm is not supported.
        """
        with self._lock:
            self.multiplier_ = float(self.multiplier)
            algo_local = self.algorithm.lower().replace(" ", "_")

        if self.multiplier_ < 1:
            raise ValueError("multiplier must be >= 1")
        if self.target_bins < MIN_TARGET_BINS:
            raise ValueError(
                f"target_bins must be >= {MIN_TARGET_BINS}, got {self.target_bins}"
            )
        # The only algorithm check: the resampling steps below rely on it.
        if algo_local not in ("random", "smote", "adasyn"):
            raise ValueError(f"Unsupported algorithm: {algo_local}")
        return self.multiplier_, self.target_bins, algo_local

    def _resample(
        self, X: Any, y: Any, algo: str, multiplier: float, target_bins: int
    ) -> tuple[Any, np.ndarray]:
        """Bin the target and dispatch to the resampling path of ``algo``.

        Returns:
            tuple: (X_resampled, y_resampled), see :meth:`_fit_resample`.
        """
        # Regression mode (default): the engine never treats the target
        # as a class label. ANY numeric target (float or integer) is
        # discretized into quantile bins; the bins are resampled as
        # pseudo-classes and the real values are restored afterwards.
        # The classification path is used only for explicit
        # task="classification" calls or for non-numeric targets.
        y_s = pd.Series(np.asarray(y))
        if self.task.lower() != "classification":
            target = _BinnedTarget(y_s, *_bin_regression_target(y_s, target_bins))
        else:
            target = _BinnedTarget(y_s, False, y_s, {})

        # Sparse input with add_noise: only the stored non-zero elements are
        # noised (see _add_sparse_gaussian_noise) — noising zeros would
        # densify the matrix and corrupt the "feature absent" encoding.
        noise_nonzero_only = False
        if hasattr(X, "tocsr"):
            if algo == "random":
                return self._resample_sparse(
                    X, target, self._target_strategy(target, multiplier)
                )
            # SMOTE/ADASYN mathematically require dense distance matrices,
            # so a sparse input is converted to dense (as before the hashing
            # switch to sparse), keeping the sparse noise semantics.
            X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            noise_nonzero_only = True

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Binary indicator columns (one-hot / missing indicators) are
        # detected BEFORE resampling: SMOTE/ADASYN interpolate them into
        # fractions, after which they would no longer look binary.
        binary_cols = _binary_indicator_columns(X_df) if self.add_noise else []
        cat_idx = self._categorical_indices(X_df, algo)
        cat_cols = [X_df.columns[i] for i in cat_idx]
        strategy = self._target_strategy(target, multiplier)

        sampler: RandomOverSampler | None = None
        encoder: OrdinalEncoder | None = None
        if algo == "random":
            X_res_raw, y_res, encoder, sampler = self._resample_random(
                X_df, cat_cols, target, strategy
            )
        elif target.is_regression:
            X_res_raw, y_res, encoder = self._resample_smoter(
                X_df, cat_idx, cat_cols, target, strategy, algo
            )
        else:
            X_res_raw, y_res = self._resample_classification(
                X_df, cat_idx, target, strategy, algo
            )

        X_res_df = self._postprocess(
            X_res_raw,
            X_df,
            cat_cols=cat_cols,
            encoder=encoder,
            binary_cols=binary_cols,
            noise_nonzero_only=noise_nonzero_only,
            algo=algo,
        )
        if target.is_regression:
            return X_res_df.to_numpy(), self._decode_regression_y(
                y_res, target.y_s, target.y_fit, target.bin_mids, sampler
            )
        return X_res_df.to_numpy(), np.asarray(y_res)

    def _target_strategy(
        self, target: _BinnedTarget, multiplier: float
    ) -> dict[Any, int]:
        """The sampling strategy of ``target`` (``-1`` skipped in regression)."""
        return self._strategy(
            target.y_fit, multiplier, skip_missing_code=target.is_regression
        )

    def _categorical_indices(self, X_df: pd.DataFrame, algo: str) -> list[int]:
        """Return the positions of the categorical columns (for SMOTENC).

        Columns with numeric strings ("1.0", "2.0") are not categorical —
        they will be converted to numbers.

        Args:
            X_df (pd.DataFrame): The features.
            algo (str): The normalized algorithm name.

        Returns:
            list[int]: The positions of the non-numeric columns.

        Raises:
            TypeError: If SMOTE/ADASYN get no numeric feature (needed for
                the distance computation).
        """
        cat_idx = [
            i for i, col in enumerate(X_df.columns) if _is_categorical_col(X_df[col])
        ]
        if algo in ("smote", "adasyn") and len(cat_idx) == len(X_df.columns):
            raise TypeError(
                f"{self.algorithm.upper()} requires at least one numeric feature."
            )
        return cat_idx

    def _resample_sparse(
        self, X: Any, target: _BinnedTarget, strategy: dict[Any, int]
    ) -> tuple[Any, np.ndarray]:
        """Random oversampling of a sparse matrix (output of hashing encoding).

        The data is already numeric, there are no categorical columns
        (OrdinalEncoder is not needed) and ``pd.DataFrame(csr)`` would create
        a silently garbage dataframe object. RandomOverSampler works with
        sparse directly (memory-efficient); the result stays a CSR matrix.

        Returns:
            tuple: (X_resampled, y_resampled).
        """
        sampler = RandomOverSampler(
            sampling_strategy=strategy, random_state=self.random_state
        )
        X_res_raw, y_res = sampler.fit_resample(X, target.y_fit)
        logger.info("%s resample: %d -> %d", "RANDOM", X.shape[0], X_res_raw.shape[0])
        if self.add_noise:
            X_res_raw = self._add_sparse_gaussian_noise(X_res_raw)
        if target.is_regression:
            return X_res_raw, self._decode_regression_y(
                y_res, target.y_s, target.y_fit, target.bin_mids, sampler
            )
        return X_res_raw, y_res.to_numpy()

    @staticmethod
    def _encode_categoricals(
        X_df: pd.DataFrame, cat_cols: list[Any]
    ) -> tuple[pd.DataFrame, OrdinalEncoder | None]:
        """Ordinal-encode the categorical columns (a copy; no-op without them).

        Args:
            X_df (pd.DataFrame): The features.
            cat_cols (list[Any]): The categorical columns.

        Returns:
            tuple: (X_encoded, encoder), the encoder being ``None`` when
                there is no categorical column.
        """
        if not cat_cols:
            return X_df, None
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_encoded = X_df.copy()
        X_encoded[cat_cols] = encoder.fit_transform(X_encoded[cat_cols])
        return X_encoded, encoder

    def _resample_random(
        self,
        X_df: pd.DataFrame,
        cat_cols: list[Any],
        target: _BinnedTarget,
        strategy: dict[Any, int],
    ) -> tuple[Any, Any, OrdinalEncoder | None, RandomOverSampler]:
        """Random oversampling of a dense frame.

        Categorical features are encoded to pass ``_check_X_y`` inside
        RandomOverSampler, which forcibly converts the DataFrame to float.

        Returns:
            tuple: (X_resampled, y_resampled, encoder, sampler).
        """
        sampler = RandomOverSampler(
            sampling_strategy=strategy, random_state=self.random_state
        )
        X_encoded, encoder = self._encode_categoricals(X_df, cat_cols)
        X_res_raw, y_res = sampler.fit_resample(X_encoded, target.y_fit)
        return X_res_raw, y_res, encoder, sampler

    @staticmethod
    def _active_counts(target: _BinnedTarget) -> Counter[Any]:
        """Per-class counts used by SMOTE/ADASYN.

        "-1" (missing target in the regression mode) is a pass-through
        pseudo-class: it is excluded from the strategy AND from the
        k_neighbors estimation.

        Raises:
            ValueError: If no non-missing target value is left.
        """
        counts = Counter(target.y_fit)
        active_counts = Counter(
            {
                cls: cnt
                for cls, cnt in counts.items()
                if not (target.is_regression and cls == -1)
            }
        )
        if not active_counts:
            raise ValueError(
                "SMOTE/ADASYN require at least one non-missing target value."
            )
        return active_counts

    def _resample_smoter(
        self,
        X_df: pd.DataFrame,
        cat_idx: list[int],
        cat_cols: list[Any],
        target: _BinnedTarget,
        strategy: dict[Any, int],
        algo: str,
    ) -> tuple[np.ndarray, np.ndarray, OrdinalEncoder | None]:
        """SMOTER-style generation for a binned regression target.

        Only bins with at least two rows can produce a same-bin neighbor
        pair; a singleton bin (or a bin that is entirely too small for the
        k-NN) passes through unchanged. The pseudo-class ``-1`` is already
        excluded from ``strategy`` and the active counts in the regression
        mode, so the balanced and the proportional mode share this filter.

        Returns:
            tuple: (X_resampled, y_resampled, encoder).
        """
        active_counts = self._active_counts(target)
        growth_eligible = {cls for cls, cnt in active_counts.items() if cnt >= 2}
        # Number of synthetic rows per bin = final - original count.
        counts_binned = Counter(target.y_fit)
        n_to_generate: dict[int, int] = {
            int(cls): max(0, int(tgt) - counts_binned.get(cls, 0))
            for cls, tgt in strategy.items()
            if cls in growth_eligible
        }
        # Encode categorical columns numerically before the distance
        # computation (SMOTENC-style); the categories are decoded back in
        # _postprocess through the encoder.
        X_syn_source, encoder = self._encode_categoricals(X_df, cat_cols)
        X_np = X_syn_source.to_numpy(dtype=float)
        y_np = np.asarray(target.y_s, dtype=float)
        generator = _SMOTERGenerator(
            random_state=self.random_state,
            categorical_features=cat_idx,
            density_weighted=(algo == "adasyn"),
        )
        X_synth, y_synth = generator.fit_resample(
            X_np, y_np, np.asarray(target.y_fit), n_to_generate
        )
        X_res_raw = np.vstack([X_np, X_synth]) if len(X_synth) else X_np
        y_res = np.concatenate([y_np, y_synth]) if len(y_synth) else y_np
        return X_res_raw, y_res, encoder

    def _resample_classification(
        self,
        X_df: pd.DataFrame,
        cat_idx: list[int],
        target: _BinnedTarget,
        strategy: dict[Any, int],
        algo: str,
    ) -> tuple[Any, Any]:
        """SMOTE / SMOTENC / ADASYN for a classification target.

        All classes stay in the strategy and k is derived from the smallest
        class count. A singleton minority class is left to imbalanced-learn,
        which raises an informative ValueError (the documented imblearn
        contract). ADASYN falls back to SMOTENC when categories are present,
        because the original ADASYN only works with numeric data.

        Returns:
            tuple: (X_resampled, y_resampled).
        """
        min_samples = min(self._active_counts(target).values())
        k_neighbors = max(1, min(5, min_samples - 1))
        sampler: Any
        if cat_idx:
            if algo == "adasyn":
                # IMPORTANT: Log the substitution to avoid misleading
                logger.warning(
                    "ADASYN does not support categorical columns. "
                    "Automatically falling back to SMOTENC for data integrity."
                )
            sampler = SMOTENC(
                categorical_features=cat_idx,
                sampling_strategy=strategy,
                k_neighbors=k_neighbors,
                random_state=self.random_state,
            )
        elif algo == "smote":
            sampler = SMOTE(
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
        return sampler.fit_resample(X_df, target.y_fit)

    def _postprocess(
        self,
        X_res_raw: Any,
        X_df: pd.DataFrame,
        *,
        cat_cols: list[Any],
        encoder: OrdinalEncoder | None,
        binary_cols: list[Any],
        noise_nonzero_only: bool,
        algo: str,
    ) -> pd.DataFrame:
        """Add the noise, decode the categories back and log the resample.

        ``encoder`` is the categorical encoder of the random and SMOTER paths
        (``None`` otherwise); ``noise_nonzero_only`` noises only the non-zero
        cells of dense data that arrived as a sparse matrix.

        Returns:
            pd.DataFrame: The resampled features.
        """
        # Cast to DF since SMOTE may return numpy
        X_res_df = pd.DataFrame(X_res_raw, columns=X_df.columns)[X_df.columns]

        # A single point of noise application for all algorithms
        # Categorical columns are excluded from the noise so as not to break
        # the subsequent decoding via OrdinalEncoder.inverse_transform
        if self.add_noise:
            noise_exclude = list(cat_cols) if encoder is not None else []
            noise_exclude.extend(binary_cols)
            X_res_df = self._add_gaussian_noise(
                X_res_df,
                exclude_cols=noise_exclude,
                nonzero_only=noise_nonzero_only,
            )

        # Decode the ordinal-encoded categorical columns back (random and
        # SMOTER paths; SMOTENC keeps the original categories itself)
        if encoder is not None:
            X_res_df[cat_cols] = encoder.inverse_transform(X_res_df[cat_cols])

        logger.info("%s resample: %d -> %d", algo.upper(), len(X_df), len(X_res_df))
        return X_res_df

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
