from __future__ import annotations
import pandas as pd
from typing import Tuple

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
