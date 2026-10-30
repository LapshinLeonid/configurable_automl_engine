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
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder

from configurable_automl_engine.oversampling import experimental
from configurable_automl_engine.oversampling.columns import (
    _add_gaussian_noise,
    _add_sparse_gaussian_noise,
    _binary_indicator_columns,
    _is_categorical_col,
    _restore_dtypes,
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
    _decode_regression_y,
    _sampling_strategy,
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
        """Compute the target number of rows per class (see ``targets``)."""
        return _sampling_strategy(
            y,
            multiplier,
            balance=self.balance,
            skip_missing_code=skip_missing_code,
        )

    def _add_gaussian_noise(
        self,
        df: pd.DataFrame,
        exclude_cols: list[str] | None = None,
        *,
        nonzero_only: bool = False,
    ) -> pd.DataFrame:
        """Add Gaussian noise to the numeric columns (see ``columns``)."""
        return _add_gaussian_noise(
            df,
            exclude_cols,
            noise_level=self.noise_level,
            random_state=self.random_state,
            nonzero_only=nonzero_only,
        )

    def _add_sparse_gaussian_noise(self, X: Any) -> Any:
        """Add Gaussian noise to the non-zero elements (see ``columns``)."""
        return _add_sparse_gaussian_noise(
            X, noise_level=self.noise_level, random_state=self.random_state
        )

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
        """Restore the original data types (see ``columns``)."""
        return _restore_dtypes(df, original_dtypes, add_noise=self.add_noise)

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
        """Restore real regression targets after resampling (see ``targets``)."""
        return _decode_regression_y(y_res, y_original, y_fit, bin_mids, sampler)

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
