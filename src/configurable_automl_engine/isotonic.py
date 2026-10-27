"""The isotonic regression data adapter.

``IsotonicDataTransformer`` feeds one numerical column of the preprocessed
feature matrix to the strictly univariate ``IsotonicRegression``: it picks
the column (by its index relative to the numerical block), imputes missing
values with the training median and returns a 2D ``(n_samples, 1)`` array.
It is a pipeline step of both the single-model trainer (``trainer``) and the
HPO trial pipelines (``hpo.trial_pipeline``), and it is part of saved models.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from configurable_automl_engine.common.errors import TrainingError

__all__ = ["IsotonicDataTransformer"]


class IsotonicDataTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Transformer for preparing data for IsotonicRegression.
    Selects the specified numerical feature and ensures correct handling of
    missing values to meet the requirements of the isotonic regression
    algorithm.

    The preprocessor (``build_preprocessor``) always emits the categorical
    block (one-hot dummies etc.) first and the numerical block
    (imputation/scaling) last. Only the numerical features are meaningful
    inputs for the strictly univariate isotonic model, so the adapter treats
    ``feature_index`` as an index RELATIVE to the numerical block of the
    input matrix and maps it to the absolute column via
    ``(n_columns - n_numeric) + feature_index``. The number of numerical
    features (``n_numeric``) is passed by the caller; the estimated total
    feature count (``n_total``) guards against a divergence between the
    estimate and the fitted preprocessor output.

    Attributes:
        feature_index (int): Ordinal index of the numerical feature to
            extract (relative to the numerical block).
        n_numeric (int | None): Number of numerical features in the input
            matrix (used to find the numerical block). ``None`` — the index
            is treated as positional into the whole matrix (standalone use).
        n_total (int | None): Estimated total feature count after
            preprocessing, used for the guard. ``None`` — no guard.
        median_ (float): Computed median used to fill missing values.
    """

    def __init__(
        self,
        feature_index: int = 0,
        n_numeric: int | None = None,
        n_total: int | None = None,
    ):
        """Initialize the transformer with the index of the target feature.

        Args:
            feature_index: Ordinal index of the numerical feature to extract
                (relative to the numerical block of the input matrix; with
                ``n_numeric=None`` — positional into the whole matrix).
            n_numeric: Expected number of numerical features in the input
                matrix (the numerical block width). ``None`` — standalone
                positional mode.
            n_total: Estimated total feature count after preprocessing
                (guard against an estimate/preprocessor divergence).
                ``None`` — no guard.
        """
        self.feature_index = feature_index
        self.n_numeric = n_numeric
        self.n_total = n_total

    def _get_dimensions(self, X: Any) -> tuple[int, int]:
        """Return the number of rows and columns of the input data
        in a universal format."""
        if hasattr(X, "shape"):
            return X.shape[0], (X.shape[1] if len(X.shape) > 1 else 1)
        n_rows = len(X)
        n_cols = len(X[0]) if n_rows > 0 and isinstance(X[0], (list, np.ndarray)) else 1
        return n_rows, n_cols

    def _resolve_absolute_index(self, n_cols: int) -> int:
        """Map the (possibly numerical-block-relative) index to an absolute column.

        Args:
            n_cols: Number of columns of the actual input matrix.

        Returns:
            The absolute column index to extract.

        Raises:
            ValueError: If the estimated total feature count disagrees with
                the actual column count, ``n_numeric`` is invalid, or
                ``feature_index`` lies outside the numerical block.
                ValueError is non-fatal for the HPO circuit breaker (unlike
                ``TrainingError`` — a ``RuntimeError`` — which is fatal).
        """
        if self.n_numeric is None:
            # Standalone mode: positional index into the whole matrix.
            if self.feature_index >= n_cols:
                raise ValueError(
                    f"IsotonicDataTransformer: feature_index {self.feature_index} "
                    f"out of bounds. Dataset has only {n_cols} columns."
                )
            return self.feature_index

        # Engine mode: feature_index is relative to the numerical block.
        if self.n_total is not None and int(self.n_total) != int(n_cols):
            raise ValueError(
                f"IsotonicDataTransformer: the estimated feature count "
                f"{self.n_total} disagrees with the fitted preprocessor "
                f"output ({n_cols} columns)."
            )
        if int(self.n_numeric) <= 0:
            raise ValueError(
                "IsotonicDataTransformer: no numerical features to select "
                "from (n_numeric=0). Isotonic regression requires at least one "
                "numerical feature."
            )
        if self.feature_index >= int(self.n_numeric):
            raise ValueError(
                f"IsotonicDataTransformer: feature_index {self.feature_index} "
                f"out of bounds: only {self.n_numeric} numerical features "
                "after preprocessing."
            )
        offset = int(n_cols) - int(self.n_numeric)
        if offset < 0:
            raise ValueError(
                f"IsotonicDataTransformer: the preprocessed matrix has fewer "
                f"columns ({n_cols}) than the estimated numerical features "
                f"({self.n_numeric})."
            )
        return offset + self.feature_index

    def _extract_column(self, X: Any, idx: int) -> pd.Series | np.ndarray:
        """Extract the ``idx``-th column of ``X`` without copying."""
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, idx]
        if isinstance(X, np.ndarray):
            return X[:, idx] if len(X.shape) > 1 else X
        return np.asarray(X)[:, idx]

    def fit(self, X: Any, y: Any = None) -> IsotonicDataTransformer:
        """Compute the median of the selected feature for subsequent missing-value imputation."""
        _, n_cols = self._get_dimensions(X)
        abs_idx = self._resolve_absolute_index(n_cols)

        X_col = self._extract_column(X, abs_idx)
        s_col = X_col if isinstance(X_col, pd.Series) else pd.Series(X_col)
        self.median_ = float(s_col.median()) if not s_col.isna().all() else 0.0
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Extract the feature, fill missing values,
        and convert to a 2D array (n_samples, 1)."""
        try:
            # 1. Get metadata without forcing a copy into a DataFrame
            _n_rows, n_cols = self._get_dimensions(X)

            # 2. Validate the estimated feature count / resolve the absolute
            #    column (numeric-block-relative feature_index).
            abs_idx = self._resolve_absolute_index(n_cols)

            # 3. Extract the column (zero-copy for numpy and pandas)
            X_col = self._extract_column(X, abs_idx)
            # 4. NaN handling logic
            # Check whether all values in the column are NaN
            # Use pd.Series for convenient median computation,
            # if it is not already a Series
            s_col = X_col if isinstance(X_col, pd.Series) else pd.Series(X_col)

            if s_col.isna().all():
                raise ValueError(
                    f"Column with index {abs_idx} contains only NaN values."
                    " Training IsotonicRegression is impossible."
                )
            # Compute the median and fill missing values
            fill_value = getattr(self, "median_", 0.0)
            X_imputed = s_col.fillna(fill_value)
            # Return the result as a 2D array, as scikit-learn expects
            result = X_imputed.to_numpy().reshape(-1, 1)
            return cast(np.ndarray, result)
        except ValueError:
            # ValueError is non-fatal for the circuit breaker in HPO (see the
            # tuner), while TrainingError (a RuntimeError) is fatal — so
            # validation/selection errors must not be wrapped.
            raise
        except Exception as e:  # noqa: BLE001
            # Unify the error message according to the tests
            raise TrainingError(f"Data transformation error: {e}")
