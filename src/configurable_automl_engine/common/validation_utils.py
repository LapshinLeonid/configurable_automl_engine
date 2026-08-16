from __future__ import annotations

import math
from math import sqrt
from typing import Any

import pandas as pd

from configurable_automl_engine.common.definitions import ValidationStrategy

pd.options.mode.copy_on_write = True


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

    The ratio ``test_size`` is clamped to the range [0.01, 0.99] to avoid
    producing 0 or ``n_total`` (which would crash most training algorithms).

    Args:
        n_total: Total number of rows.
        test_size: Desired test fraction (clamped internally).

    Returns:
        The number of rows the model effectively trains on.
    """
    safe_test_size = max(0.01, min(0.99, test_size))
    effective_size = math.floor(n_total * (1 - safe_test_size))
    # Гарантируем, что если есть хотя бы 2 строки, то Neff будет минимум 1
    return max(1 if n_total >= 2 else 0, effective_size)


def choose_validation_method(n_samples: int, n_features: int) -> dict[str, Any]:
    """Deterministically select a validation method for regression (v6).

    The decision is based on ``n_samples`` (N, number of observations) and
    ``n_features`` (P, number of features). The returned dict describes the
    selected method and, when needed, the derived regime, test split size and
    number of folds. The heuristic is fully deterministic given ``(N, P)`` so
    every caller that resolves 'auto' arrives at the same decision.

    Args:
        n_samples: Number of observations (rows) in the dataset.
        n_features: Number of features (columns) in the dataset.

    Returns:
        A dict describing the chosen method. Possible shapes:
            ``{"method": "LOO"}``
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

    # 1. LOO for very small datasets
    if n_samples <= 30 and n_samples >= 2 * n_features:
        return {"method": "LOO"}

    # 2. LOO for small datasets with a good N/P ratio
    if 31 <= n_samples <= 50 and n_samples >= 10 * n_features:
        return {"method": "LOO"}

    # 3. Standard train-test split.
    # The minimum test set size is the larger of 30 rows or 2*P features (enough
    # to estimate the target distribution), capped at a 30% test fraction.
    min_test_standard = max(30, 2 * n_features)
    if (
        n_samples >= 200
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

    # 4. High-dimensional train-test split: when P is very large relative to N
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

    # 5. K-Fold (continuous heuristic).
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
    return X, y


def get_effective_train_size(
    n_total: int,
    strategy: ValidationStrategy | str,
    n_folds: int = 5,
    test_size: float = 0.2,
    n_features: int | None = None,
) -> int:
    """
    Рассчитывает количество строк, которые модель фактически "видит" во время fit()
    в рамках одной итерации HPO или кросс-валидации.
    Rationale: Это значение критично для динамического клиппинга пространства поиска
    (например, n_neighbors в KNN не может быть больше количества обучающих примеров).

    Args:
        n_total: Общее количество строк в датасете.
        strategy: Стратегия валидации (k_fold, loo, train_test_split, auto).
        n_folds: Количество фолдов (используется для k_fold).
        test_size: Доля теста (используется для train_test_split).
        n_features: Количество признаков P. Требуется для стратегии 'auto'
            (используется в choose_validation_method).

    Returns:
        int: Эффективный размер обучающей выборки.
    """
    if n_total <= 0:
        return 0

    # Приведение строки к Enum, если необходимо
    if isinstance(strategy, str):
        try:
            strategy = ValidationStrategy(strategy)
        except ValueError:
            # Если передана неизвестная стратегия, возвращаем n_total как fallback
            raise ValueError(f"Unknown validation strategy string: '{strategy}'")

    if strategy == ValidationStrategy.auto:
        return _effective_size_for_auto(n_total, n_features, n_folds, test_size)

    if strategy == ValidationStrategy.k_fold:
        # Neff = floor(N_total * (1 - 1/k))
        # k не может быть меньше 2 для k_fold
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
