from __future__ import annotations
import pandas as pd
from typing import Tuple
from configurable_automl_engine.common.definitions import ValidationStrategy

pd.options.mode.copy_on_write = True


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

def prepare_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
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
test_size: float = 0.2
) -> int:
    """
    Рассчитывает количество строк, которые модель фактически "видит" во время fit()
    в рамках одной итерации HPO или кросс-валидации.
    Rationale: Это значение критично для динамического клиппинга пространства поиска 
    (например, n_neighbors в KNN не может быть больше количества обучающих примеров).

    Args:
        n_total: Общее количество строк в датасете.
        strategy: Стратегия валидации (k_fold, loo, train_test_split).
        n_folds: Количество фолдов (используется для k_fold).
        test_size: Доля теста (используется для train_test_split).

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
            return n_total

    if strategy == ValidationStrategy.k_fold:
        # Neff = floor(N_total * (1 - 1/k))
        # k не может быть меньше 2 для k_fold
        k = max(2, n_folds)
        return math.floor(n_total * (1 - 1 / k))

    if strategy == ValidationStrategy.loo:
        # Neff = N_total - 1
        return max(0, n_total - 1)

    if strategy == ValidationStrategy.train_test_split:
        # Neff = floor(N_total * (1 - test_size))
        return math.floor(n_total * (1 - test_size))

    return n_total