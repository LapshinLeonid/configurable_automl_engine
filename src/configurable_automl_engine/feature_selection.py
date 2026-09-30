"""Feature selector for reducing the feature space (issue #29).

This module implements :class:`FeatureSelector` — a standalone sklearn
transformer (``BaseEstimator`` + ``TransformerMixin``) that reduces the
feature space using one of four algorithms:

* **importance** — ``SelectFromModel`` based on the feature importances of a
  tree ensemble (``ExtraTreesRegressor``); the threshold is the mean
  importance (the default ``SelectFromModel`` behavior);
* **percentile** — ``SelectPercentile`` scored with ``f_regression``
  (linear dependencies);
* **mutual_info** — ``SelectPercentile`` scored with
  ``mutual_info_regression`` (captures non-linear dependencies);
* **variance** — ``VarianceThreshold`` (drops constant and low-variance
  columns).

The transformer guarantees:

1. Matrix type preservation on transform: ``np.ndarray`` stays a dense
   array, ``scipy.sparse.csr_matrix`` stays a sparse matrix (slicing is done
   without ``.toarray()``, which prevents OOM on large data), and
   ``pd.DataFrame`` stays a DataFrame.
2. Protection against an empty feature space (``min_features_guard``): if
   the base selector picks fewer than ``min_features`` features (down to
   zero — all features deemed irrelevant), the support mask forcibly
   activates the ``top-min(min_features, P)`` features with the highest
   scores.
3. Passthrough mode when ``P <= min_features``: selection is skipped and
   ``support_`` keeps every input feature.

The transformer is consumed by ``ModelTrainer`` (see
``configurable_automl_engine.trainer``), where it is inserted between the
preprocessor and the oversampler of the training pipeline.
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["FeatureSelector"]

_VALID_METHODS: tuple[str, ...] = (
    "importance",
    "percentile",
    "mutual_info",
    "variance",
)

# Methods that require the target y during fit().
_TARGET_METHODS: tuple[str, ...] = ("importance", "percentile", "mutual_info")


class FeatureSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Reduce the feature space with one of several backends.

    Selects features according to the chosen algorithm (``method``),
    preserves the matrix type on transform (dense / sparse / DataFrame) and
    protects the pipeline from an empty feature space:

    * if the number of input features ``P <= min_features``, the transformer
      works in passthrough mode (no selection is performed);
    * if the base selector picks ``K < min_features`` features (including
      ``K = 0``), the ``support_`` mask forcibly activates the
      ``top-min(min_features, P)`` features with the highest scores
      (importances / scores / variances).

    Args:
        method: Selection algorithm: ``'importance'`` (tree importances),
            ``'percentile'`` (regression f-statistics), ``'mutual_info'``
            (mutual information, non-linear dependencies) or ``'variance'``
            (variance threshold).
        percentile: Percentage of features kept by the ``'percentile'`` and
            ``'mutual_info'`` methods (``0 < percentile <= 100``).
        min_features: Absolute lower bound on the number of kept features
            (``>= 1``). When ``P <= min_features`` no selection is performed.
        variance_threshold: Variance threshold for the ``'variance'``
            method: columns with variance ``<= threshold`` are dropped.
        n_estimators: Number of trees in ``ExtraTreesRegressor`` for the
            ``'importance'`` method.
        random_state: Random seed for reproducibility (``'importance'`` and
            ``'mutual_info'``).

    Attributes:
        support_: Boolean mask of selected features of length
            ``n_features`` (available after ``fit``).
        n_selected_: Number of selected features.
        base_selector_: Fitted sklearn base selector or ``None`` in
            passthrough mode.
        n_features_in_: Number of input features.
        feature_names_in_: Column names (for ``pd.DataFrame`` input).
    """

    def __init__(
        self,
        method: str = "importance",
        percentile: float = 50.0,
        min_features: int = 2,
        variance_threshold: float = 0.0,
        n_estimators: int = 50,
        random_state: int | None = 42,
    ) -> None:
        self.method = method
        self.percentile = percentile
        self.min_features = min_features
        self.variance_threshold = variance_threshold
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> FeatureSelector:
        """Fit the selector on the training data.

        Args:
            X: Feature matrix ``(n_samples, n_features)``. Supports
                ``np.ndarray``, ``pd.DataFrame`` and sparse
                ``scipy.sparse`` matrices (``csr_matrix`` and others).
            y: Target variable. Required for the ``'importance'``,
                ``'percentile'`` and ``'mutual_info'`` methods; ignored for
                ``'variance'``.

        Returns:
            The fitted transformer (``self``). Selected features are
            available through :meth:`get_support` / the ``support_``
            attribute.

        Raises:
            ValueError: If the ``method`` is unknown, ``y`` is missing for
                methods that require the target, or
                ``percentile``/``min_features``/``variance_threshold``/
                ``n_estimators`` are invalid.
        """
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method: {self.method!r}. Expected one of {_VALID_METHODS}."
            )
        if self.method in _TARGET_METHODS and y is None:
            raise ValueError(f"Method {self.method!r} requires y during fit().")
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}.")
        if not 0.0 < self.percentile <= 100.0:
            raise ValueError(f"percentile must be in (0, 100], got {self.percentile}.")
        if self.variance_threshold < 0.0:
            raise ValueError(
                f"variance_threshold must be >= 0.0, got {self.variance_threshold}."
            )
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")

        if y is None:
            X_arr = check_array(X, accept_sparse=True, dtype="numeric")
            y_arr = None
        else:
            X_arr, y_arr = check_X_y(X, y, accept_sparse=True, dtype="numeric")

        self.n_features_in_ = X_arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        elif hasattr(self, "feature_names_in_"):
            # Drop stale names when re-fitting on column-less input.
            del self.feature_names_in_

        n_features = X_arr.shape[1]

        # Edge case: P <= min_features -> passthrough without selection.
        if n_features <= self.min_features:
            self.support_ = np.ones(n_features, dtype=bool)
            self.n_selected_ = n_features
            self.base_selector_ = None
            return self

        # sklearn treats the features of sparse matrices as discrete, and
        # mutual_info_regression fails with a cryptic error on continuous
        # values — raise a clear message beforehand.
        if self.method == "mutual_info" and sp.issparse(X_arr):
            _validate_sparse_mutual_info(X_arr)

        if self.method == "variance":
            support, k_selected = self._fit_variance(X_arr, n_features)
        else:
            assert y_arr is not None
            support, k_selected = self._fit_target_based(X_arr, y_arr, n_features)

        self.support_ = support
        self.n_selected_ = k_selected
        return self

    def _fit_variance(self, X_arr: Any, n_features: int) -> tuple[np.ndarray, int]:
        """Fit the ``'variance'`` branch, handling the ``K = 0`` edge case.

        ``VarianceThreshold.fit`` raises ``ValueError`` when no feature
        exceeds the threshold (all columns are dropped). Such a situation is
        handled routinely by ``min_features_guard``: variances are computed
        beforehand **with the same algorithm sklearn uses** (see
        :func:`_compute_variances`), so the decision about calling
        ``VarianceThreshold`` always matches its own, and the crash is
        impossible. On an empty selection the guard immediately activates
        the ``top-min(min_features, P)`` features with the largest variance.
        """
        variances = _compute_variances(X_arr)
        if np.any(variances > self.variance_threshold):
            selector = self._build_selector()
            selector.fit(X_arr)
            self.base_selector_ = selector
            support = np.asarray(selector.get_support(), dtype=bool)
            k_selected = int(np.count_nonzero(support))
            if k_selected < self.min_features:
                support, k_selected = self._apply_guard(
                    self._extract_scores(selector), n_features
                )
        else:
            self.base_selector_ = None
            support, k_selected = self._apply_guard(variances, n_features)
        return support, k_selected

    def _fit_target_based(
        self, X_arr: Any, y_arr: Any, n_features: int
    ) -> tuple[np.ndarray, int]:
        """Fit the target-based branches (importance/percentile/mutual_info)."""
        selector = self._build_selector()
        selector.fit(X_arr, y_arr)
        self.base_selector_ = selector
        support = np.asarray(selector.get_support(), dtype=bool)
        k_selected = int(np.count_nonzero(support))
        if k_selected < self.min_features:
            scores = self._extract_scores(selector)
            support, k_selected = self._apply_guard(scores, n_features)
        return support, k_selected

    def _build_selector(self) -> Any:
        """Create the sklearn base selector for the chosen method.

        The method is already validated in :meth:`fit`, so this branch is
        unreachable for invalid values.
        """
        if self.method == "importance":
            return SelectFromModel(
                ExtraTreesRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                ),
                threshold="mean",
            )
        if self.method == "percentile":
            return SelectPercentile(score_func=f_regression, percentile=self.percentile)
        if self.method == "mutual_info":
            # The seed fixes the stochasticity of the kNN mutual-information
            # estimate.
            score_func = partial(mutual_info_regression, random_state=self.random_state)
            return SelectPercentile(score_func=score_func, percentile=self.percentile)
        if self.method == "variance":
            return VarianceThreshold(threshold=self.variance_threshold)
        raise AssertionError(
            f"Unreachable: method {self.method!r} must be validated in fit()."
        )

    def _extract_scores(self, selector: Any) -> np.ndarray:
        """Extract feature scores from the fitted sklearn base selector."""
        if self.method == "importance":
            return np.asarray(selector.estimator_.feature_importances_, dtype=float)
        if self.method in ("percentile", "mutual_info"):
            return np.asarray(selector.scores_, dtype=float)
        if self.method == "variance":
            return np.asarray(selector.variances_, dtype=float)
        raise AssertionError(
            f"Unreachable: method {self.method!r} must be validated in fit()."
        )

    def _apply_guard(
        self, scores: np.ndarray, n_features: int
    ) -> tuple[np.ndarray, int]:
        """Forcibly activate the ``top-min(min_features, P)`` features.

        Args:
            scores: Feature scores (importances, scores or variances);
                larger is "better".
            n_features: Total number of input features ``P``.

        Returns:
            Tuple ``(support_, n_selected)``: a boolean mask with exactly
            ``min(min_features, P)`` active positions.
        """
        top_k = min(self.min_features, n_features)
        top_indices = _top_indices(scores, top_k)
        support = np.zeros(n_features, dtype=bool)
        support[top_indices] = True
        return support, int(np.count_nonzero(support))

    def transform(self, X: Any) -> np.ndarray | sp.spmatrix | pd.DataFrame:
        """Reduce the feature space according to the fitted ``support_`` mask.

        The matrix type is preserved: ``np.ndarray`` -> ``np.ndarray``,
        ``scipy.sparse`` -> ``scipy.sparse`` (slice ``X[:, support_]``
        without ``.toarray()``), ``pd.DataFrame`` -> ``pd.DataFrame``.

        Args:
            X: Feature matrix ``(n_samples, n_features)`` of the same type
                as used during ``fit``.

        Returns:
            A matrix of the same type with the columns selected in
            :meth:`fit`.

        Raises:
            sklearn.exceptions.NotFittedError: If the transformer is not
                fitted.
            ValueError: If the number of columns of ``X`` does not match the
                training one, or the column names of a ``pd.DataFrame`` do
                not match ``feature_names_in_``.
        """
        check_is_fitted(self, attributes=["support_"])
        n_input = X.shape[1]
        if n_input != len(self.support_):
            raise ValueError(
                f"X has {n_input} features, but FeatureSelector was fitted "
                f"with {len(self.support_)} features."
            )
        if isinstance(X, pd.DataFrame):
            # Name verification, like in sklearn: with the same number of
            # columns but a different order, a positional slice would
            # silently take the wrong features.
            if hasattr(self, "feature_names_in_") and list(X.columns) != list(
                self.feature_names_in_
            ):
                raise ValueError(
                    "The feature names of the input DataFrame do not match "
                    "those seen during fit. Expected "
                    f"{list(self.feature_names_in_)}, got {list(X.columns)}."
                )
            return X.iloc[:, self.support_]
        if sp.issparse(X):
            # Slicing is done in CSR (COO does not support indexing), then
            # the result is converted back to the original format:
            # CSR -> CSR, CSC -> CSC, COO -> COO etc. Without .toarray():
            # no dense copy is created (prevents OOM). scipy-stubs do not
            # type indexing and dynamic to<format> for SparseABC — Any.
            sparse_X = cast(Any, X)
            sliced = sparse_X.tocsr()[:, self.support_]
            return cast(sp.spmatrix, getattr(sliced, "to" + sparse_X.format)())
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return the mask (or the indices) of the selected features.

        Args:
            indices: If ``True``, return the indices of the selected
                features instead of the boolean mask.

        Returns:
            A boolean array of length ``n_features`` or an integer array of
            indices of the selected features.
        """
        check_is_fitted(self, attributes=["support_"])
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the names of the output (selected) features.

        Args:
            input_features: Names of the input features. When ``None``, the
                names saved in :meth:`fit` for ``pd.DataFrame``
                (``feature_names_in_``) are used, or positional names
                ``x0...xN`` otherwise.

        Returns:
            An array of names of the selected features only.

        Raises:
            ValueError: If the length of ``input_features`` does not match
                the number of input features.
        """
        check_is_fitted(self, attributes=["support_"])
        n_features = len(self.support_)
        if input_features is None and hasattr(self, "feature_names_in_"):
            input_features = self.feature_names_in_
        if input_features is None:
            names = np.asarray([f"x{i}" for i in range(n_features)], dtype=object)
        else:
            features = np.asarray(input_features, dtype=object)
            if len(features) != n_features:
                raise ValueError(
                    f"input_features has {len(features)} elements, "
                    f"expected {n_features}."
                )
            names = features
        return names[self.support_]


def _compute_variances(X_arr: Any) -> np.ndarray:
    """Compute the variance of each column the same way ``VarianceThreshold`` does.

    For sparse matrices, ``sklearn.utils.sparsefuncs.mean_variance_axis``
    is used after converting to ``csc float64`` — exactly what
    ``VarianceThreshold.fit`` does internally
    (``accept_sparse='csc', dtype=np.float64``). For dense input, ``np.var``
    on a ``float64`` copy.

    Why the ``E[x^2] - E[x]^2`` formula cannot be used: on sparse input it
    produces noisy non-zero "variances" of the order ``1e-14`` for constant
    columns (precision loss when summing squares in ``scipy.sparse.mean``),
    so the precomputation diverges from sklearn's own computation:
    ``VarianceThreshold.fit`` raises ``ValueError`` ("No feature in X meets
    the variance threshold") instead of triggering the
    ``min_features_guard``.
    """
    if sp.issparse(X_arr):
        # First tocsc(), then float64 — as in _validate_data of
        # VarianceThreshold. scipy-stubs do not type tocsc for SparseABC —
        # cast to Any.
        X_csc = cast(Any, X_arr).tocsc().astype(np.float64)
        _, variances = mean_variance_axis(X_csc, axis=0)
        return np.asarray(variances, dtype=float).ravel()
    X_dense: np.ndarray = np.asarray(X_arr, dtype=np.float64)
    return np.asarray(np.var(X_dense, axis=0), dtype=float)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` features with the highest scores.

    Non-numeric scores (NaN) are treated as ``-inf`` (they cannot enter the
    top). Ties are resolved stably by index (``argsort(kind='stable')`` in
    descending order): with equal scores the lowest indices survive, so the
    result is deterministic even for fully equal scores (for example, zero
    variances of all-constant columns).
    """
    safe = np.where(np.isfinite(scores), scores, -np.inf)
    if k >= safe.shape[0]:
        return np.arange(safe.shape[0], dtype=int)
    # Descending argsort with stable sorting: on equal scores the lowest
    # indices come first (deterministic tie-break).
    return np.argsort(-safe, kind="stable")[:k]


def _validate_sparse_mutual_info(X_arr: Any) -> None:
    """Check that the sparse input for ``mutual_info`` is discrete.

    sklearn treats all features of sparse matrices as discrete
    (``discrete_features='auto'`` -> ``True`` for sparse), and
    ``mutual_info_regression`` fails on continuous values with a cryptic
    error (an empty sample in the kNN estimate). Instead, a clear
    ``ValueError`` is raised demanding integer-valued features.

    Args:
        X_arr: Sparse matrix (already converted to a numeric dtype).

    Raises:
        ValueError: If any value is non-integer (continuous).
    """
    data = cast(Any, X_arr).data
    if np.issubdtype(X_arr.dtype, np.integer):
        return
    if np.all(np.isfinite(data)) and np.all(data == np.trunc(data)):
        return
    raise ValueError(
        "Method 'mutual_info' requires discrete (integer-valued) features "
        "for sparse input: sklearn treats all features of sparse matrices "
        "as discrete, and continuous values break the kNN-based mutual "
        "information estimation."
    )
