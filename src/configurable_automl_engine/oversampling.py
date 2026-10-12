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
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


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

    def _strategy(self, y: pd.Series, multiplier: float) -> dict[Any, int]:
        """Compute the target number of instances for each class.
        Args:
            y (pd.Series): The target variable vector.
            multiplier (float): The scaling factor.
        Returns:
            Dict[Any, int]: A dictionary where keys are class labels
                and values are the final number of rows.
        """
        counts = Counter(y)
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

            # Sparse branch (output of hashing encoding, csr_matrix):
            # the data is already numeric, there are no categorical columns
            # (OrdinalEncoder is not needed) and pd.DataFrame(csr) would create a
            # silently garbage dataframe object. RandomOverSampler works with sparse
            # directly (memory-efficient); SMOTE/ADASYN mathematically require
            # dense distance matrices, so for them a sparse input
            # is converted to dense (as before the hashing switch to sparse).
            if is_sparse:
                strategy_sparse = self._strategy(y_s, self.multiplier_)
                if algo_local == "random":
                    sampler = RandomOverSampler(
                        sampling_strategy=strategy_sparse,
                        random_state=self.random_state,
                    )
                    X_res_raw, y_res = sampler.fit_resample(X, y_s)
                    logger.info(
                        "%s resample: %d -> %d",
                        algo_local.upper(),
                        X.shape[0],
                        X_res_raw.shape[0],
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
            strategy = self._strategy(y_s, self.multiplier_)

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
            else:
                ordinal_encoder = None
                # SMOTE / ADASYN case
                counts = Counter(y_s)

                # For SMOTE/ADASYN: Dynamic calculation of k_neighbors.
                # Prevents ValueError if a class has fewer
                # than the default 5 neighbors.
                min_samples = min(counts.values())
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

            X_res_raw, y_res = sampler.fit_resample(X_to_resample, y_s)
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
