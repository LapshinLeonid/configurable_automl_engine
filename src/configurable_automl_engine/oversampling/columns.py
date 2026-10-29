"""Column-level helpers of the oversampler.

Holds the detection of categorical columns (``_is_categorical_col``) and of
binary indicator columns that the Gaussian noise must leave untouched
(``_binary_indicator_columns``). The module must not import the other
oversampling modules.
"""

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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
