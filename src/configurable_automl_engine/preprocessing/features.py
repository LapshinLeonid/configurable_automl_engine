"""Feature inspection before the preprocessor is built.

Holds :func:`detect_feature_types` (categorical vs numerical columns),
:func:`_clean_uninformative_features` (removal of empty and constant
columns, applied identically in HPO and in final training) and
:func:`estimate_n_features_after_preprocessing`. The module must not import
the preprocessor builder.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from configurable_automl_engine.common.definitions import EncodingStrategy

logger = logging.getLogger(__name__)


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Automatically classify columns into numerical and categorical.

    Columns with dtype ``object``, ``category`` and ``bool`` are considered
    categorical; numerical columns have a dtype from the ``number`` family.

    Args:
        X: Feature DataFrame (without the target variable).

    Returns:
        A tuple ``(categorical_features, numerical_features)`` — lists of
        column names of each type.

    Note:
        The function expects exactly ``pd.DataFrame``; for ``np.ndarray``
        without column names auto-detection is impossible (documented limitation).

    Note:
        Classification is based on the column ``dtype``. This intentionally
        differs from the value-based detection in ``oversampling._is_categorical_col``:
        here the preprocessor handles data before oversampling, so ``bool`` and
        'numeric strings' (``'10','20'``) are treated as categories (one-hot).
        The oversampler receives already encoded numerical features and applies
        its own value-based logic for standalone calls.
    """
    categorical = X.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    return categorical, numerical


def _clean_uninformative_features(
    X: pd.DataFrame,
    max_nan_ratio: float = 0.95,
    min_unique: int = 2,
) -> pd.DataFrame:
    """Drop uninformative columns: empty and strictly constant ones.

    A column is considered uninformative if at least one of the following
    conditions holds:

    1. Missing ratio ``>= max_nan_ratio`` (0.95 by default): imputation with
       mean/median replaces almost all values with a single constant; after
       scaling the column either carries no information (zero contribution to
       distances) or biases the distance matrix of metric models
       (SVR/KNN/Ridge).
    2. Number of unique values excluding NaN ``< min_unique`` (2 by default):
       a strictly constant column has zero variance after imputation and
       scaling; for ``RobustScaler`` division by a zero IQR (``q75 == q25``)
       yields NaN/±inf, which breaks training. A constant categorical column
       is useless even after one-hot encoding (a constant binary column), so
       it is also dropped.

    The decision is made **only from the training data** (no leakage) and is
    fixed at the fitting stage: a column that is empty on train but
    informative at inference will be dropped in production too (documented
    behavior). This is consistent with ``detect_feature_types``, which is also
    computed on the full data. The function is idempotent: a repeated call on
    already cleaned data does not change the result.

    Guard rules:

    * A DataFrame without columns (``X.shape[1] == 0``) or without rows
      (``X.shape[0] == 0``) is returned unchanged — cleaning does not fail.
    * The single column is never dropped, even if it is constant or empty:
      otherwise an empty feature matrix would remain.
    * If **all** columns fall under removal (with ``P >= 2``), a
      :class:`ValueError` is raised: an empty feature set is not allowed. This
      is analogous to ``FeatureSelector.min_features_guard``
      (``feature_selection.py``), but at the data preparation stage the error
      is explicit rather than a silent passthrough.

    Args:
        X: Feature DataFrame (without the target variable).
        max_nan_ratio: Missing ratio threshold in ``(0, 1]``: columns with a
            NaN ratio ``>=`` the threshold are dropped. ``1.0`` — only fully
            empty columns are dropped.
        min_unique: Minimum number of unique values excluding NaN (``>= 1``)
            at which a column is considered informative. The value ``2``
            (default) drops strictly constant columns; the value ``1`` keeps
            them (useful only together with the missing-ratio rule).

    Returns:
        A copy of ``X`` without the uninformative columns (the original object
        is not mutated). If there is nothing to drop, the original ``X`` is
        returned.

    Raises:
        TypeError: If ``X`` is not a ``pd.DataFrame``. For ``np.ndarray``
            without column names, cleaning is impossible (documented
            limitation, as with :func:`detect_feature_types`).
        ValueError: If the thresholds are invalid or all columns fall under
            removal (an empty feature set).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "_clean_uninformative_features expects a pandas.DataFrame, got "
            f"{type(X).__name__}. Column names are required to identify "
            "uninformative features (same limitation as detect_feature_types)."
        )
    if not 0.0 < max_nan_ratio <= 1.0:
        raise ValueError(f"max_nan_ratio must be in (0, 1], got {max_nan_ratio!r}.")
    if min_unique < 1:
        raise ValueError(f"min_unique must be >= 1, got {min_unique!r}.")

    n_cols = X.shape[1]
    # An empty DataFrame (0 rows / 0 columns) is handled gracefully without
    # failures: there is nothing to clean.
    if n_cols == 0 or X.shape[0] == 0:
        return X
    # The single column is always kept, even if it is constant or empty:
    # otherwise an empty feature matrix would remain.
    if n_cols == 1:
        return X

    # 1. Columns with a missing ratio >= max_nan_ratio (including fully empty ones).
    nan_ratios = X.isna().mean()
    drop_cols = list(nan_ratios[nan_ratios >= max_nan_ratio].index)

    # 2. Strictly constant columns: exactly one unique value excluding NaN (for a
    # fully empty column nunique() == 0, but such columns are already covered
    # by the missing-ratio rule at max_nan_ratio <= 1.0).
    for col in X.columns:
        if col not in drop_cols and X[col].dropna().nunique() < min_unique:
            drop_cols.append(col)

    if not drop_cols:
        logger.debug(
            "No uninformative columns to drop (max_nan_ratio=%s, min_unique=%d, "
            "n_columns=%d)",
            max_nan_ratio,
            min_unique,
            n_cols,
        )
        return X
    if len(drop_cols) >= n_cols:
        raise ValueError(
            "All columns are uninformative and would be dropped: "
            f"{drop_cols}. Refusing to produce an empty feature matrix."
        )

    logger.warning("Dropping uninformative/empty columns: %s", drop_cols)
    logger.debug(
        "Cleanup thresholds: max_nan_ratio=%s, min_unique=%d; dropped %d of %d columns",
        max_nan_ratio,
        min_unique,
        len(drop_cols),
        n_cols,
    )
    return X.drop(columns=drop_cols)


def estimate_n_features_after_preprocessing(
    X: Any,
    *,
    categorical_features: list[str],
    numerical_features: list[str],
    encoding: EncodingStrategy = "one_hot",
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: EncodingStrategy | None = None,
    hashing_n_components: int = 16,
    add_indicator: bool = True,
) -> int:
    """Estimate the number of columns the preprocessor will produce.

    A deterministic estimate from the data and encoding settings (no fitting
    stage), shared between the tuner (data-dependent search spaces, e.g. the
    isotonic ``feature_index`` range) and the final training
    (``ModelTrainer``, which passes the count to the isotonic adapter for the
    estimate-vs-actual guard). Numerical features contribute one column each,
    plus one missing-value indicator per column with missing values when
    ``add_indicator=True``; categorical columns contribute their number of
    unique values for ``one_hot``, ``n_components`` for ``hashing``, and one
    column for the remaining strategies (ordinal/target/frequency). When the
    automatic high-cardinality mode is enabled, each column is counted with
    the strategy it actually receives at fit time (columns whose cardinality
    strictly exceeds the threshold use ``high_cardinality_encoding`` — the
    same rule as :class:`SplitCategoricalEncoder`).

    Args:
        X: Feature matrix. A ``pd.DataFrame`` is required for the categorical
            expansion; for other inputs only the numerical block is counted.
        categorical_features: Names of the categorical columns.
        numerical_features: Names of the numerical columns.
        encoding: The categorical encoding strategy.
        high_cardinality_threshold: High-cardinality threshold (``None`` — the
            automatic mode is disabled).
        high_cardinality_encoding: Strategy for high-cardinality columns.
        hashing_n_components: Number of hashing output columns per
            categorical column.
        add_indicator: Whether missing-value indicators are added to the
            numerical features.

    Returns:
        The estimated post-preprocessing column count (>= 1).
    """
    cats = list(categorical_features)
    nums = list(numerical_features)
    n = 0
    if nums:
        n += len(nums)
        if add_indicator and isinstance(X, pd.DataFrame):
            try:
                n += int(X[nums].isna().any().sum())
            except Exception:  # noqa: BLE001 — estimate only
                n += 0
    if cats and isinstance(X, pd.DataFrame):
        for col in cats:
            if col not in X.columns:
                continue
            # The automatic high-cardinality split: the column is encoded
            # with the HC strategy when its cardinality strictly exceeds the
            # threshold (the same rule as SplitCategoricalEncoder).
            effective = encoding
            if (
                high_cardinality_threshold is not None
                and high_cardinality_encoding is not None
            ):
                try:
                    if int(X[col].nunique(dropna=True)) > high_cardinality_threshold:
                        effective = high_cardinality_encoding
                except Exception:  # noqa: BLE001, S110 — estimate only
                    # Cardinality is unknown: keep the main strategy.
                    pass
            if effective == "hashing":
                n += hashing_n_components
            elif effective == "one_hot":
                try:
                    n += int(X[col].nunique(dropna=True))
                except Exception:  # noqa: BLE001 — estimate only
                    n += 1
            else:
                n += 1
    return max(1, n)
