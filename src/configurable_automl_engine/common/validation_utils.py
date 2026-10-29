from __future__ import annotations

import logging
import math
from collections import Counter
from math import sqrt
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd

from configurable_automl_engine.common.definitions import ValidationStrategy

_InfInput = TypeVar("_InfInput", bound=pd.DataFrame | np.ndarray)

#: Fraction of unique values (nunique / n) above which an object/string column
#: is considered ID-like and flagged with a WARNING.
ID_LIKE_UNIQUE_RATIO = 0.95


def ceil_div(a: int, b: int) -> int:
    """Ceiling integer division: ceil(a / b)."""
    return (a + b - 1) // b


def resolve_auto_no_features_fallback(n_samples: int, n_folds: int) -> tuple[str, int]:
    """Resolve the 'auto' strategy when P (feature count) is unavailable.

    Falls back to k-fold with ``n_folds`` (clamped to a minimum of 2). If the
    sample size is too small for a meaningful k-fold
    (``n_samples < max(4, 2 * k)``), it falls back to a train/test split.

    This is the single source of truth shared by ``make_cv`` and
    ``get_effective_train_size`` so the resolved method and effective train
    size never drift apart.

    Args:
        n_samples: Total number of observations.
        n_folds: User-requested fold count (may be < 2; clamped internally).

    Returns:
        A ``(method, k)`` tuple where ``method`` is ``"k_fold"`` or
        ``"train_test_split"`` and ``k`` is the resolved fold count.
    """
    k = max(2, n_folds)
    min_required = max(4, 2 * k)
    if n_samples < min_required:
        return "train_test_split", k
    return "k_fold", k


def _effective_size_train_test(n_total: int, test_size: float) -> int:
    """Compute the effective train size for a train/test split.

    ``test_size`` follows sklearn semantics: a fraction in [0.01, 0.99] is
    clamped to that range (to avoid producing 0 or ``n_total``, which would
    crash most training algorithms); an integer value is interpreted as the
    absolute number of test rows (the 'auto' resolution via
    ``choose_validation_method`` returns an integer test size).

    Args:
        n_total: Total number of rows.
        test_size: Desired test fraction (or absolute row count).

    Returns:
        The number of rows the model effectively trains on.
    """
    if isinstance(test_size, int) and not isinstance(test_size, bool):
        return max(1 if n_total >= 2 else 0, n_total - test_size)
    safe_test_size = max(0.01, min(0.99, test_size))
    effective_size = math.floor(n_total * (1 - safe_test_size))
    # Ensure that if there are at least 2 rows, Neff is at least 1
    return max(1 if n_total >= 2 else 0, effective_size)


def choose_validation_method(n_samples: int, n_features: int) -> dict[str, Any]:
    """Deterministically select a validation method for regression.

    The decision is based on ``n_samples`` (N, number of observations) and
    ``n_features`` (P, number of features). The returned dict describes the
    selected method and, when needed, the derived regime, test split size and
    number of folds. The heuristic is fully deterministic given ``(N, P)`` so
    every caller that resolves 'auto' arrives at the same decision.

    Decision branches (applied in order):

    1. **LOO** for critically small samples (N <= 15) with a good N/P ratio
       (N >= 2*P): leave-one-out is the only affordable way to get per-point
       estimates when there are too few rows for k-fold.
    2. **Fixed k-fold** for small samples (16 <= N <= 50): each fold tests the
       model on a slice of several distinct points (6-8 rows), which is more
       reliable than LOO — LOO would measure the ability to repeat the global
       trend instead of generalizing to new point slices. Uses k=4 for N >= 20
       and k=3 otherwise. A low-confidence warning is attached when N < 2*P.
    3. **Fixed 5-fold CV** for mid-size samples (200 <= N < 50000) with a good
       N/P ratio (N >= 10*P): a single hold-out of max(30, 2*P) rows compares
       models on too few points — on 30 rows the RMSE spread is
       about ±13% while the confidence corridor is δ=0.15, so the winner is
       selected almost at random and the Sanity Gate OOF vector covers only
       ~3% of the data. 5-fold CV scores every row exactly once (100% OOF
       coverage) and stays affordable up to a few tens of thousands of rows.
    4. **Standard train-test split** for large samples (N >= 50000) with a
       good N/P ratio: at this scale the smallest hold-out (max(30, 2*P)
       rows, ~1-6% of the data) is already large enough for a reliable
       comparison, while 5-fold CV would cost 5x compute for no benefit.
       Note the deliberate discrete jump at N=50000: below the threshold 5-fold
       CV scores 100% of rows; from the threshold on a single hold-out scores
       ~1-6% — the hold-out floor scales with N, so the comparison stays
       meaningful while the compute cost drops 5x.
    5. **High-dimensional train-test split** for N >= 50000 when P is very
       large relative to N (5P <= N < 10P).
    6. **Continuous k-fold heuristic** for everything else; k == N is relabeled
       to LOO. Micro-samples (N <= 5) are protected from k > N.

    Args:
        n_samples: Number of observations (rows) in the dataset.
        n_features: Number of features (columns) in the dataset.

    Returns:
        A dict describing the chosen method. Possible shapes:
            ``{"method": "LOO"}``
            ``{"method": "kfold", "k", "average_test_size", "warning"}``
            (fixed k-fold for 16 <= N <= 50; ``average_test_size`` is an
            integer floor of N/k)
            ``{"method": "train_test_split", "regime", "test_percent",
            "test_size", "train_size", ["warning"]}``
            ``{"method": "kfold", "k", "average_test_size", ["warning"]}``
            ``{"method": "LOO", "note", ["warning"]}`` (relabeled k-fold)
            ``{"method": "invalid", "reason"}`` when N < 2.

    Raises:
        ValueError: If n_features (P) is not positive.
    """
    if n_features <= 0:
        raise ValueError(f"n_features (P) must be > 0, got {n_features}")
    if n_samples < 2:
        return {"method": "invalid", "reason": "N < 2: insufficient data"}

    # 1. LOO for critically small samples (up to 15 rows) with a good N/P ratio.
    if n_samples <= 15 and n_samples >= 2 * n_features:
        return {"method": "LOO"}

    # 2. Small samples 16 <= N <= 50: a fixed k-fold (3-4 folds) is more
    # reliable than LOO. Each fold tests the model on a slice of several
    # distinct points instead of a single one.
    if 16 <= n_samples <= 50:
        k = 4 if n_samples >= 20 else 3
        # Defensive guard: keep k within [1, N]. k > N is impossible for
        # 16..50 today (k <= 4 < 16 <= N), but the clamp protects against
        # future range edits, invalid callers and division by zero.
        k = max(1, min(k, n_samples))
        fixed_kfold_warning = (
            "Small dataset: fixed k-fold CV (3-4 folds) to prevent LOO overfitting."
        )
        # Uniform low-confidence flag with the other branches: N barely
        # sufficient for P (N < 2*P) takes precedence over the generic note.
        if n_samples < 2 * n_features:
            fixed_kfold_warning = (
                "Low confidence: High variance expected (N is barely sufficient for P)."
            )
        return {
            "method": "kfold",
            "k": k,
            "average_test_size": n_samples // k,
            "warning": fixed_kfold_warning,
        }

    # 3. Mid-size samples with a good N/P ratio (200 <= N < 50000, N >= 10*P):
    # fixed 5-fold CV. A single hold-out of max(30, 2*P) rows would compare
    # models on far too few points: on 30 rows the RMSE spread is about ±13%
    # against a confidence corridor of δ=0.15, so the winner would be chosen
    # almost at random, and the Sanity Gate OOF vector would cover only ~3%
    # of the data. 5-fold CV scores every row exactly once (100% OOF coverage)
    # and is affordable up to a few tens of thousands of rows.
    if 200 <= n_samples < 50000 and n_samples >= 10 * n_features:
        return {
            "method": "kfold",
            "k": 5,
            "average_test_size": round(n_samples / 5, 1),
        }

    # 4. Standard train-test split for large samples (N >= 50000).
    # At this scale the minimum test set — the larger of 30 rows or 2*P
    # features (enough to estimate the target distribution) — is already a
    # statistically reliable slice (~1-6% of the data), and 5-fold CV would
    # cost 5x compute for no benefit. The test fraction is capped at 30%.
    min_test_standard = max(30, 2 * n_features)
    if (
        n_samples >= 50000
        and n_samples >= 10 * n_features
        and n_samples - min_test_standard >= 50
    ):
        test_percent = min(30, max(1, ceil_div(100 * min_test_standard, n_samples)))
        n_test = ceil_div(test_percent * n_samples, 100)
        return {
            "method": "train_test_split",
            "regime": "standard",
            "test_percent": test_percent,
            "test_size": n_test,
            "train_size": n_samples - n_test,
        }

    # 5. High-dimensional train-test split: when P is very large relative to N
    # (5P <= N < 10P), k-fold becomes too costly, so a single split is preferred.
    min_test_high_dimensional = max(1000, 2 * n_features)
    if (
        n_samples >= 50000
        and n_samples >= 5 * n_features
        and n_samples < 10 * n_features
        and 100 * min_test_high_dimensional <= 40 * n_samples
    ):
        test_percent = min(
            40, max(1, ceil_div(100 * min_test_high_dimensional, n_samples))
        )
        n_test = ceil_div(test_percent * n_samples, 100)
        return {
            "method": "train_test_split",
            "regime": "high_dimensional",
            "test_percent": test_percent,
            "test_size": n_test,
            "train_size": n_samples - n_test,
            "warning": (
                "Low confidence: this regime trades train-set reliability "
                "(N/P below the standard r_min=10) for compute savings vs k-fold."
            ),
        }

    # 6. K-Fold (continuous heuristic).
    # Pre-check for micro-samples (protection against k > N).
    if n_samples <= 5:
        k = n_samples
    else:
        # Compute budget constraint (cap folds to keep each fold large enough).
        budget_k = max(5, min(10, round(sqrt(n_samples))))
        # Statistical reliability constraint (min rows per class ~ max(20, 2*P)).
        reliability_k = max(2, n_samples // max(20, 2 * n_features))
        # Take the minimum to satisfy both constraints.
        k = max(2, min(budget_k, reliability_k))

    # Unified confidence flag (Low confidence): N barely sufficient for P.
    low_confidence_ratio = n_samples < 2 * n_features
    warning = None
    if k == 2 or low_confidence_ratio:
        warning = (
            "Low confidence: High variance expected (N is barely sufficient for P)."
        )

    # Relabel to LOO if k == N (preserving the warning).
    if k == n_samples:
        result: dict[str, Any] = {
            "method": "LOO",
            "note": "k-fold with k=N is equivalent to LOO",
        }
        if warning:
            result["warning"] = warning
        return result

    result = {
        "method": "kfold",
        "k": k,
        "average_test_size": round(n_samples / k, 1),
    }
    if warning:
        result["warning"] = warning

    return result


def effective_train_size_from_decision(n_total: int, decision: dict[str, Any]) -> int:
    """Return the effective train size implied by a resolved 'auto' decision.

    Maps a decision dict produced by :func:`choose_validation_method` to the
    number of rows the model trains on per iteration. Kept in one place so
    ``get_effective_train_size`` and the tuner share the same mapping and the
    value used to clip the search space can never drift from the actual split.

    Args:
        n_total: Total number of observations.
        decision: The decision dict from :func:`choose_validation_method`.

    Returns:
        The effective train size for the selected method.
    """
    method = decision["method"]

    if method == "LOO":
        return max(0, n_total - 1)

    if method == "kfold":
        k = max(2, int(decision["k"]))
        return math.floor(n_total * (1 - 1 / k))

    if method == "train_test_split":
        train_size = int(decision["train_size"])
        return max(1 if n_total >= 2 else 0, train_size)

    # method == "invalid" (N < 2): nothing reliable to train on.
    return max(0, n_total - 1)


def validate_df_not_empty(df: pd.DataFrame) -> None:
    """
    Checks if the input is a non-empty pandas DataFrame.
    Raises TypeError if not a DataFrame, and ValueError if empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data must be a pandas.DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError("Input dataframe is empty")


def check_target_exists(df: pd.DataFrame, target_col: str) -> None:
    """
    Verifies that the target column exists in the DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found"
            f" in dataframe columns: {list(df.columns)}"
        )


def prepare_X_y(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features (X) and target (y).

    Assumptions:
    - df is a valid non-empty DataFrame.
    - target_col exists in df.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    check_target_numeric(y)
    return X, y


def check_target_numeric(y: Any) -> None:
    """
    Verify that the target vector is numeric and free of NaN/±inf.

    A non-finite target makes per-fold evaluations depend on which rows
    happen to land in which fold (the folds containing the bad values
    crash/fail, the rest look "clean"), so the HPO budget is spent on
    randomly clean folds and the final full-data fit fails with an unclear
    error. The check fails fast on the pipeline entry point.

    Args:
        y (Any): The target vector (pd.Series or any array-like of numbers).

    Raises:
        ValueError: If the target is non-numeric or contains NaN/±inf.
    """
    try:
        y_float = np.asarray(y, dtype=float).reshape(-1)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Target column '{getattr(y, 'name', 'y')}' must be numeric"
        ) from err
    if y_float.size != len(y):
        raise ValueError(
            f"Target column '{getattr(y, 'name', 'y')}' must be a single "
            f"column of numbers; got {y_float.size} values for {len(y)} rows"
        )
    if not np.all(np.isfinite(y_float)):
        n_non_finite = int(np.sum(~np.isfinite(y_float)))
        raise ValueError(
            f"Target column '{getattr(y, 'name', 'y')}' must not contain NaN "
            f"or infinite values; found {n_non_finite} non-finite value(s)"
        )


def check_target_constant(y: Any) -> None:
    """Verify that the target vector is not constant.

    A constant target (a single unique value) makes training meaningless:
    every model trivially predicts the constant and reports a perfect score
    (e.g. R² = 1.0), which is silently misleading. The check fails fast at
    the pipeline entry point.

    The "constant" test uses the count of unique values (not ``np.std``,
    which is numerically fragile for tiny values such as ~1e-200): a vector
    with more than one distinct finite value passes even when its standard
    deviation rounds to zero.

    Args:
        y (Any): The target vector (pd.Series or any array-like of numbers).

    Raises:
        ValueError: If the target contains a single unique value.
    """
    try:
        y_float = np.asarray(y, dtype=float).reshape(-1)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Target column '{getattr(y, 'name', 'y')}' must be numeric"
        ) from err
    if y_float.size == 0:
        return
    if not np.all(np.isfinite(y_float)):
        return
    # Uniqueness is checked in the original dtype first: a float64 conversion
    # loses precision for large integers (> 2**53), so distinct int64 values
    # could collapse into one float. The float path is only a fallback for
    # inputs without a usable native dtype.
    y_native = np.asarray(y).reshape(-1)
    y_unique = y_native if y_native.dtype.kind in "iub" else y_float
    if np.unique(y_unique).size <= 1:
        raise ValueError(
            f"Target column '{getattr(y, 'name', 'y')}' is constant "
            f"(single unique value {y_unique[0].item()!r}); a regression "
            "model cannot be trained on a constant target — every model would "
            "trivially report a perfect score. Remove the target and re-check "
            "the data, or switch to a different prediction problem."
        )


def _format_column_list(columns: list[str]) -> str:
    """Format a list of column names for a diagnostic message.

    Args:
        columns: Column names.

    Returns:
        A human-readable listing (``"a, b, c"``), with quotes around names
        that are not plain identifiers.
    """
    if not columns:
        return ""
    quoted = [repr(c) if not str(c).isidentifier() else str(c) for c in columns]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


#: Datetime components generated by the ``"expand"`` policy: component name
#: (the suffix of the generated column) → ``Series.dt`` attribute. ``hour``
#: is only generated when it is not all-zero on the training data.
_DATETIME_COMPONENTS: dict[str, str] = {
    "year": "year",
    "month": "month",
    "day": "day",
    "dayofweek": "day_of_week",
    "hour": "hour",
}

#: Timedelta components generated by the ``"expand"`` policy.
_TIMEDELTA_COMPONENTS: dict[str, str] = {
    "days": "days",
    "seconds": "seconds",
    "microseconds": "microseconds",
}


def _datetime_parts(series: pd.Series) -> dict[str, pd.Series]:
    """Derive numeric components from a datetime/timedelta column.

    For datetime columns the components are ``year``, ``month``, ``day``,
    ``dayofweek`` and — when the hour is not all-zero — ``hour``. For
    timedelta columns the components are ``days``, ``seconds`` and
    ``microseconds``.

    Args:
        series: A datetime64 or timedelta64 column.

    Returns:
        An ordered dict ``component name → numeric component series``.
    """
    if pd.api.types.is_timedelta64_dtype(series.dtype):
        return {
            part: getattr(series.dt, attr)
            for part, attr in _TIMEDELTA_COMPONENTS.items()
        }
    parts: dict[str, pd.Series] = {
        part: getattr(series.dt, attr)
        for part, attr in _DATETIME_COMPONENTS.items()
        if part != "hour"
    }
    hour = series.dt.hour
    if bool((hour != 0).any()):
        parts["hour"] = hour
    return parts


def _is_datetime_like(series: pd.Series) -> bool:
    """Whether a column has a datetime64 or timedelta64 dtype.

    Args:
        series: A DataFrame column.

    Returns:
        ``True`` for datetime64 (tz-naive or tz-aware) and timedelta64 columns.
    """
    return bool(
        pd.api.types.is_datetime64_any_dtype(series.dtype)
        or pd.api.types.is_timedelta64_dtype(series.dtype)
    )


def datetime_expansion_map(
    X: pd.DataFrame, generated: list[str]
) -> dict[str, list[str]]:
    """Map each expanded datetime/timedelta column to its generated components.

    The mapping is recorded at training time so that inference can apply the
    same expansion to raw input data (see :func:`expand_datetime_columns`).

    Args:
        X: The feature DataFrame BEFORE :func:`validate_features` (it still
            contains the original datetime/timedelta columns).
        generated: The ``generated_columns`` returned by
            :func:`validate_features` for ``X``.

    Returns:
        A dict ``source column → generated component columns`` (in the
        generation order). Empty when nothing was expanded.
    """
    if not generated:
        return {}
    generated_set = set(generated)
    expansion: dict[str, list[str]] = {}
    for col in X.columns:
        series = X[col]
        if not isinstance(series, pd.Series) or not _is_datetime_like(series):
            continue
        components = (
            _TIMEDELTA_COMPONENTS
            if pd.api.types.is_timedelta64_dtype(series.dtype)
            else _DATETIME_COMPONENTS
        )
        names = [
            f"{col}_{part}" for part in components if f"{col}_{part}" in generated_set
        ]
        if names:
            expansion[str(col)] = names
    return expansion


def expand_datetime_columns(
    X: pd.DataFrame, expansion: dict[str, list[str]]
) -> pd.DataFrame:
    """Apply a recorded datetime expansion to new (raw) data.

    Inference counterpart of the ``datetime_policy="expand"`` step of
    :func:`validate_features`: each source column listed in ``expansion`` is
    replaced by exactly the components generated at training time (the
    component set is taken from the mapping, not re-derived from the new
    data, so e.g. ``hour`` is produced even when it is all-zero in the new
    rows). A source column with an ``object``/string dtype is converted with
    ``pd.to_datetime`` / ``pd.to_timedelta`` first.

    Columns are left untouched when the source column is absent, or when all
    of its components are already present (the data was expanded upstream).

    Args:
        X: The input DataFrame.
        expansion: The mapping from :func:`datetime_expansion_map`.

    Returns:
        The DataFrame with the expansion applied (``X`` itself when nothing
        had to be expanded; otherwise a new DataFrame).

    Raises:
        ValueError: If a source column cannot be converted to datetime /
            timedelta values.
    """
    out = X
    for col, names in expansion.items():
        if col not in X.columns or all(name in X.columns for name in names):
            continue
        part_names = [name[len(col) + 1 :] for name in names]
        is_timedelta = any(part in _TIMEDELTA_COMPONENTS for part in part_names)
        series = X[col]
        if not _is_datetime_like(series):
            try:
                series = (
                    pd.to_timedelta(series) if is_timedelta else pd.to_datetime(series)
                )
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Column {col!r} was expanded into datetime components "
                    f"{_format_column_list(names)} at training time, but its "
                    "values in the prediction data cannot be parsed as "
                    f"{'timedelta' if is_timedelta else 'datetime'} values."
                ) from err
        components = _TIMEDELTA_COMPONENTS if is_timedelta else _DATETIME_COMPONENTS
        if out is X:
            out = X.copy()
        for name, part in zip(names, part_names, strict=True):
            out[name] = np.asarray(getattr(series.dt, components[part]), dtype=float)
        out = out.drop(columns=[col])
    return out


def apply_inf_policy(X: _InfInput, inf_policy: str = "as_missing") -> _InfInput:
    """Apply the ±inf policy to the numeric features of ``X``.

    This is the single implementation of the ``general.inf_policy`` rule,
    shared by the training path (:func:`validate_features`) and the
    inference path (``ModelTrainer.predict``) so that both treat infinite
    values identically.

    For a ``pd.DataFrame`` only the numeric columns are inspected. For an
    ``np.ndarray`` the check runs when the array has a numeric dtype;
    columns are reported by their positional index. Arrays of other dtypes
    (e.g. ``object``) are returned unchanged. The input is never modified in
    place: a copy is returned when values are replaced.

    Args:
        X: The feature matrix (``pd.DataFrame`` or ``np.ndarray``).
        inf_policy: ``"as_missing"`` (default) — replace ±inf with NaN and
            emit a WARNING listing the columns and counts; ``"error"`` —
            raise ``ValueError`` naming the columns.

    Returns:
        ``X`` with ±inf replaced by NaN (or ``X`` itself when it contains no
        infinite values).

    Raises:
        ValueError: If ``inf_policy`` is ``"error"`` and ``X`` contains ±inf,
            or if ``inf_policy`` is unknown.
    """
    if inf_policy not in ("as_missing", "error"):
        raise ValueError(
            f"Unknown inf_policy: {inf_policy!r}. Expected 'as_missing' or 'error'."
        )

    numeric_cols: list[Any] = []
    labels: list[str]
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return X
        numeric_data = X[numeric_cols].to_numpy(dtype=float)
        labels = [str(col) for col in numeric_cols]
    else:
        if X.size == 0 or not np.issubdtype(X.dtype, np.number):
            return X
        numeric_data = X.reshape(X.shape[0], -1) if X.ndim != 2 else X
        labels = [f"column {i}" for i in range(numeric_data.shape[1])]

    inf_mask = np.isinf(numeric_data)
    if not inf_mask.any():
        return X
    inf_counts_per_col = inf_mask.sum(axis=0)
    offenders = [
        f"{col} ({int(count)} value(s))"
        for col, count in zip(labels, inf_counts_per_col, strict=True)
        if count > 0
    ]
    detail = _format_column_list(offenders)
    if inf_policy == "error":
        raise ValueError(
            "Numeric feature(s) contain infinite values (±inf): "
            f"{detail}. Infinite values break "
            "distance-based and linear models. Replace them with NaN "
            "(set general.inf_policy: 'as_missing') or remove the "
            "offending rows."
        )
    logging.getLogger(__name__).warning(
        "Infinite values (±inf) found in numeric feature(s) "
        "%s; replacing them with NaN (general.inf_policy: "
        "'as_missing'). The missing values will be imputed by the "
        "preprocessor.",
        detail,
    )
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return cast(_InfInput, out)
    out_arr = X.astype(np.result_type(X.dtype, np.float64), copy=True)
    out_arr[np.isinf(out_arr)] = np.nan
    return cast(_InfInput, out_arr)


def validate_features(
    X: pd.DataFrame,
    *,
    inf_policy: str = "as_missing",
    datetime_policy: str = "expand",
    drop_id_like_columns: bool = False,
    warn_id_like: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate the feature matrix and optionally repair it.

    Runs all input-data diagnostics required for a meaningful regression
    run. Messages always name the offending column(s) and suggest the fix.

    Checks (applied in order):

    1. **Duplicate column names** → ``ValueError`` listing the duplicates.
       Duplicated names make ``X[col]`` return a DataFrame instead of a
       Series, which surfaces later as an unclear pandas error
       ("The truth value of a Series is ambiguous").

    2. **±inf in numeric features** → governed by ``inf_policy``:
       ``"as_missing"`` (default) replaces them with NaN and emits a WARNING
       listing columns and counts; ``"error"`` raises ``ValueError``.

    3. **Datetime/timedelta columns** → governed by ``datetime_policy``:
       ``"expand"`` (default) replaces each column by the components
       ``year, month, day, dayofweek`` (and ``hour`` if it is not all-zero)
       and emits a WARNING listing the generated columns; ``"drop"`` removes
       the column and emits a WARNING; ``"error"`` raises ``ValueError``.

    4. **ID-like columns** → a WARNING naming the column. A column is
       ID-like if it is an object/string column with ``nunique/n >= 0.95``,
       or an integer column that is unique and monotonic. With
       ``drop_id_like_columns=True`` such columns are dropped (independently
       of ``warn_id_like``).

    Limitation: only columns with a ``datetime64``/``timedelta64`` dtype are
    detected as datetime columns. A column holding datetime values with an
    ``object`` dtype (e.g. after ``pd.read_csv`` without ``parse_dates``, or
    a load from a database driver returning strings) is not expanded; it may
    be reported as ID-like or be treated as a categorical feature. Convert
    such columns with ``pd.to_datetime`` before training.

    The performance budget of the checks is kept small (a single pass over
    the object columns for the ID heuristic, vectorized finiteness checks
    for the numeric columns) so that on a 50k x 50 frame the added overhead
    stays below 5% of the training runtime.

    Args:
        X: The feature DataFrame (without the target column).
        inf_policy: ``"as_missing"`` (default) — replace ±inf with NaN and
            warn; ``"error"`` — raise ``ValueError``.
        datetime_policy: ``"expand"`` (default) — replace a datetime/timedelta
            column with its components and warn; ``"drop"`` — drop the column
            and warn; ``"error"`` — raise ``ValueError``.
        drop_id_like_columns: Whether ID-like columns are dropped (default
            ``False`` — they are only reported with a WARNING).
        warn_id_like: Whether the ID-like WARNING is emitted (default
            ``True``). Set to ``False`` when the ID-like diagnostics have
            already been reported by an outer call (e.g. by
            ``prepare_dataset``) to avoid duplicate warnings per algorithm.

    Returns:
        A tuple ``(X, generated_columns)`` where ``X`` is the (possibly
        repaired) feature DataFrame and ``generated_columns`` lists the names
        of the columns created by the datetime expansion.

    Raises:
        ValueError: If duplicate column names are found, or if ``inf_policy``
            / ``datetime_policy`` is ``"error"`` and the corresponding issue
            is present. An unknown policy value also raises ``ValueError``.
    """
    if inf_policy not in ("as_missing", "error"):
        raise ValueError(
            f"Unknown inf_policy: {inf_policy!r}. Expected 'as_missing' or 'error'."
        )
    if datetime_policy not in ("expand", "drop", "error"):
        raise ValueError(
            f"Unknown datetime_policy: {datetime_policy!r}. "
            "Expected 'expand', 'drop' or 'error'."
        )

    # 1. Duplicate column names → ValueError listing the duplicates.
    _reject_duplicate_columns(X)
    # 2. ±inf in numeric features.
    out = apply_inf_policy(X.copy(), inf_policy)
    # 3. Datetime/timedelta columns.
    out, generated = _apply_datetime_policy(out, datetime_policy)
    # 4. ID-like columns.
    out = _handle_id_like_columns(
        out, _find_id_like_columns(out), drop=drop_id_like_columns, warn=warn_id_like
    )
    return out, generated


def _reject_duplicate_columns(X: pd.DataFrame) -> None:
    """Raise ``ValueError`` listing the duplicated column names of ``X``.

    Args:
        X: The feature DataFrame.

    Raises:
        ValueError: If ``X`` has duplicate column names.
    """
    duplicate_cols = sorted(
        {col for col, count in Counter(X.columns).items() if count > 1}
    )
    if duplicate_cols:
        raise ValueError(
            "Input data contains duplicate column names: "
            f"{_format_column_list(duplicate_cols)}. Duplicated names make "
            "column selection by name ambiguous (X[col] returns a DataFrame "
            "instead of a Series). Rename the duplicated columns before "
            "training."
        )


def _reject_datetime_name_collisions(X: pd.DataFrame, datetime_cols: list[str]) -> None:
    """Reject a datetime expansion that would overwrite existing columns.

    Derived component names (``<col>_year`` etc.) must not shadow existing
    columns — silently overwriting user data would be wrong, and a colliding
    datetime column would crash the ``.dt`` accessor.

    Args:
        X: The feature DataFrame.
        datetime_cols: The datetime/timedelta columns to expand.

    Raises:
        ValueError: Naming the collisions, if there are any.
    """
    existing = set(X.columns)
    collisions: list[str] = []
    for col in datetime_cols:
        collisions.extend(f"{col}_{part_name}" for part_name in _datetime_parts(X[col]))
    collisions = sorted(c for c in collisions if c in existing)
    if collisions:
        raise ValueError(
            "Datetime expansion would overwrite existing "
            "column(s): "
            f"{_format_column_list(collisions)}. Rename the "
            "conflicting columns or set general.datetime_policy: "
            "'drop' to remove the datetime column instead of "
            "expanding it."
        )


def _apply_datetime_policy(
    X: pd.DataFrame, datetime_policy: str
) -> tuple[pd.DataFrame, list[str]]:
    """Expand, drop or reject the datetime/timedelta columns of ``X``.

    Args:
        X: The feature DataFrame.
        datetime_policy: ``"expand"``, ``"drop"`` or ``"error"``.

    Returns:
        tuple[pd.DataFrame, list[str]]: The frame and the names of the
            columns generated by the expansion (in order).

    Raises:
        ValueError: Under ``"error"`` when a datetime/timedelta column is
            present, or under ``"expand"`` on a name collision.
    """
    generated: list[str] = []
    datetime_cols = [
        col
        for col in X.columns
        if pd.api.types.is_datetime64_any_dtype(X[col].dtype)
        or pd.api.types.is_timedelta64_dtype(X[col].dtype)
    ]
    if not datetime_cols:
        return X, generated
    if datetime_policy == "error":
        raise ValueError(
            "Input data contains datetime/timedelta column(s): "
            f"{_format_column_list(datetime_cols)}. Such columns are not "
            "handled by any transformer and would be silently dropped. "
            "Either set general.datetime_policy: 'expand' to derive "
            "features from them, 'drop' to remove them, or convert the "
            "columns to numeric features before training."
        )
    expand = datetime_policy == "expand"
    if expand:
        _reject_datetime_name_collisions(X, datetime_cols)
    out = X
    for col in datetime_cols:
        parts = _datetime_parts(out[col]) if expand else {}
        out = out.drop(columns=[col])
        for part_name, part_values in parts.items():
            new_col = f"{col}_{part_name}"
            # The component arrays are already numeric; a float conversion
            # handles NaT correctly (→ NaN).
            out[new_col] = np.asarray(part_values, dtype=float)
            generated.append(new_col)
    logging.getLogger(__name__).warning(
        "Datetime/timedelta column(s) %s: %s. Such columns are not "
        "handled by any transformer (they would be silently dropped); "
        "under general.datetime_policy '%s' they %s. Configure "
        "general.datetime_policy ('expand'/'drop'/'error') to control "
        "this behavior.",
        "were expanded into components" if expand else "have been dropped",
        _format_column_list(datetime_cols),
        datetime_policy,
        "were replaced by the components " + _format_column_list(generated)
        if expand
        else "were removed from the input",
    )
    return out, generated


def _is_id_like(series: pd.Series, n_rows: int) -> bool:
    """Whether a column matches the row-identifier heuristic.

    Args:
        series: The column.
        n_rows: The number of rows (> 0).

    Returns:
        bool: True for an object/string column with
            ``nunique/n >= ID_LIKE_UNIQUE_RATIO`` or a unique monotonic
            integer column.
    """
    dtype = series.dtype
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return bool(series.nunique(dropna=False) / n_rows >= ID_LIKE_UNIQUE_RATIO)
    if pd.api.types.is_integer_dtype(dtype) and series.is_unique:
        return bool(series.is_monotonic_increasing or series.is_monotonic_decreasing)
    return False


def _find_id_like_columns(X: pd.DataFrame) -> list[str]:
    """Return the ID-like columns of ``X`` (see :func:`_is_id_like`).

    Args:
        X: The feature DataFrame.

    Returns:
        list[str]: The ID-like columns, in column order (empty for 0 rows).
    """
    n_rows = len(X)
    if n_rows == 0:
        return []
    return [col for col in X.columns if _is_id_like(X[col], n_rows)]


def _handle_id_like_columns(
    X: pd.DataFrame, id_like: list[str], *, drop: bool, warn: bool
) -> pd.DataFrame:
    """Report and optionally drop the ID-like columns.

    Args:
        X: The feature DataFrame.
        id_like: The ID-like columns of ``X``.
        drop: Whether the columns are dropped.
        warn: Whether the WARNING is emitted.

    Returns:
        pd.DataFrame: ``X`` without the ID-like columns when ``drop``,
            otherwise ``X`` itself.
    """
    if not id_like:
        return X
    if warn:
        logging.getLogger(__name__).warning(
            "Column(s) %s match the heuristic for row identifiers "
            "(ID-like): an object/string column with nearly all values "
            "unique (nunique/n >= %.2f) or a unique monotonic integer "
            "column. Such columns are often uninformative for regression and "
            "can slow training down dramatically; however, the rule is a "
            "heuristic — time-dependent monotonic features (e.g. a timestamp "
            "counter) may be meaningful and should be reviewed before removal. "
            "Set general.drop_id_like_columns: true to drop them, or remove "
            "the column(s) from the input data. %s",
            _format_column_list(id_like),
            ID_LIKE_UNIQUE_RATIO,
            "They were dropped as configured (general.drop_id_like_columns: true)."
            if drop
            else "They are kept in the data by default (general.drop_id_like_columns: false).",
        )
    # The drop does not depend on whether the WARNING is emitted: a repeated
    # validation with ``warn_id_like=False`` (ModelTrainer.fit, the tuner)
    # must apply the same drop_id_like_columns policy as the first call.
    if drop:
        return X.drop(columns=id_like)
    return X


def get_effective_train_size(
    n_total: int,
    strategy: ValidationStrategy | str,
    n_folds: int = 5,
    test_size: float = 0.2,
    n_features: int | None = None,
) -> int:
    """
    Calculate the number of rows the model actually "sees" during fit()
    within a single HPO iteration or cross-validation.
    Rationale: This value is critical for dynamic search space clipping
    (e.g. n_neighbors in KNN cannot exceed the number of training examples).

    Args:
        n_total: Total number of rows in the dataset.
        strategy: Validation strategy (k_fold, loo, train_test_split, auto).
        n_folds: Number of folds (used for k_fold).
        test_size: Test size fraction (used for train_test_split).
        n_features: Number of features P. Required for the 'auto' strategy
            (used in choose_validation_method).

    Returns:
        int: Effective training sample size.
    """
    if n_total <= 0:
        return 0

    # Convert the string to an Enum if necessary
    if isinstance(strategy, str):
        try:
            strategy = ValidationStrategy(strategy)
        except ValueError:
            # If an unknown strategy is passed, return n_total as fallback
            raise ValueError(f"Unknown validation strategy string: '{strategy}'")

    if strategy == ValidationStrategy.auto:
        return _effective_size_for_auto(n_total, n_features, n_folds, test_size)

    if strategy == ValidationStrategy.k_fold:
        # Neff = floor(N_total * (1 - 1/k))
        # k cannot be less than 2 for k_fold
        k = max(2, n_folds)
        return math.floor(n_total * (1 - 1 / k))

    if strategy == ValidationStrategy.loo:
        # Neff = N_total - 1
        return max(0, n_total - 1)

    if strategy == ValidationStrategy.train_test_split:
        return _effective_size_train_test(n_total, test_size)

    raise ValueError(f"Unsupported validation strategy type or value: {strategy}")


def _effective_size_for_auto(
    n_total: int,
    n_features: int | None,
    n_folds: int,
    test_size: float,
) -> int:
    """Compute the effective train size for the 'auto' validation strategy.

    Resolves 'auto' through choose_validation_method and returns Neff for the
    actually selected method (train_size for train_test_split, N - N/k for
    k-fold, N - 1 for LOO). When P is unavailable/invalid, falls back to the
    default k-fold behaviour.
    """
    if n_features is None or n_features <= 0:
        # Safe fallback: mirror make_cv's behaviour when P is unavailable so
        # the resolved method and the effective train size stay consistent.
        method, k = resolve_auto_no_features_fallback(n_total, n_folds)
        if method == "train_test_split":
            return _effective_size_train_test(n_total, test_size)
        return math.floor(n_total * (1 - 1 / k))

    decision = choose_validation_method(n_total, n_features)
    return effective_train_size_from_decision(n_total, decision)
