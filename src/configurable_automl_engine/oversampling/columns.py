"""Column-level helpers of the oversampler.

Holds the detection of categorical columns (``_is_categorical_col``) and of
binary indicator columns that the Gaussian noise must leave untouched
(``_binary_indicator_columns``), the Gaussian noise itself (dense and sparse
input) and the restoration of the original dtypes after resampling — as
functions with explicit inputs (``DataOversampler`` keeps thin methods that
pass its settings). The module must not import the other oversampling
modules.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

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


def _binary_indicator_columns(df: pd.DataFrame) -> list[Any]:
    """Find the binary indicator columns of a DataFrame.

    A column is an indicator when its non-missing values take at most two
    distinct values (one-hot / hashing output, missing-value indicators, raw
    0/1 flags; numeric strings such as ``'0'``/``'1'`` count too). The rule
    deliberately does not require the values to be exactly ``0``/``1``: the
    oversampler runs after the preprocessor, and for scale-sensitive
    algorithms ``StandardScaler`` / ``RobustScaler`` map a 0/1 flag or a
    missing-value indicator to two other values (e.g. ``-1.02``/``0.98``).
    Both scalers are per-column affine maps, so the number of distinct values
    is preserved and "binary before scaling" is recognized after it. Gaussian
    noise on such a column produces meaningless values between its two
    levels, so the oversampler excludes them from the noise. Boolean columns
    are not returned: the noise step never touches them anyway.

    Args:
        df (pd.DataFrame): The feature matrix (before resampling).

    Returns:
        list[Any]: Labels of the binary indicator columns.
    """
    numeric = df.select_dtypes(include=[np.number])
    binary: list[Any] = []
    if not numeric.empty:
        n_distinct = numeric.nunique(dropna=True).to_numpy()
        mask = (n_distinct >= 1) & (n_distinct <= 2)
        binary.extend(numeric.columns[mask])
    for col in df.select_dtypes(include=["object", "string"]).columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().sum() != df[col].isna().sum():
            continue  # truly categorical — handled by the encoder logic
        if 1 <= converted.nunique(dropna=True) <= 2:
            binary.append(col)
    return binary


def _add_gaussian_noise(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    *,
    noise_level: float,
    random_state: int | None,
    nonzero_only: bool = False,
) -> pd.DataFrame:
    """Add Gaussian noise to numeric columns and numeric elements of object columns.
    Applies vectorized noise computation for performance.
    For non-numeric columns it attempts a safe type conversion.
    Args:
        df (pd.DataFrame): The dataset to modify.
        exclude_cols (list[str] | None): Columns to which noise should NOT be applied
            (e.g. ordinal-encoded categorical ones).
        noise_level (float): Noise intensity relative to the column's standard
            deviation.
        random_state (int | None): Seed of the noise generator.
        nonzero_only (bool): Noise only the non-zero cells (zeros stay
            exact zeros). Used for data that arrived as a sparse matrix
            and was densified for SMOTE/ADASYN, so the result matches
            :func:`_add_sparse_gaussian_noise`.
    Returns:
        pd.DataFrame: The dataset with added noise.
    """
    rng = np.random.default_rng(random_state)

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
        scales = stds * noise_level
        noise = rng.normal(0, 1.0, size=(len(df), len(numeric_cols))) * scales

        # Use .to_numpy() for speed
        values = df[numeric_cols].to_numpy()
        if nonzero_only:
            noise = noise * (values != 0)
        df[numeric_cols] = values + noise
    return df


def _add_sparse_gaussian_noise(
    X: Any, *, noise_level: float, random_state: int | None
) -> Any:
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
        noise_level (float): Noise intensity relative to the column's standard
            deviation.
        random_state (int | None): Seed of the noise generator.

    Returns:
        Any: A new CSR matrix (float64) with noised non-zero elements.
    """
    rng = np.random.default_rng(random_state)
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
    scales = stds * noise_level

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


def _restore_dtypes(
    df: pd.DataFrame, original_dtypes: pd.Series, *, add_noise: bool
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
        add_noise (bool): Whether Gaussian noise was added to the numeric
            columns (they then keep the fractional part).
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
            elif dtype is object or isinstance(dtype, np.dtype) and dtype.kind == "O":
                # If noise was added and the column became numeric,
                # then do NOT roll back to object
                if add_noise and is_numeric_dtype(df[col]):
                    pass
                else:
                    df[col] = df[col].astype(object)

            # 3. If it is an integer:
            elif np.issubdtype(dtype, np.integer):
                # If noise was added, we MUST NOT return int, or precision is lost
                if not add_noise:
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
