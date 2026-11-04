"""Standalone winsorization transformer.

This module must not import the other preprocessing modules: it is a pure,
self-contained building block with no coupling to the encoders, the column
selection or the preprocessor builder. Its classes are pickled inside saved
models, so the module path is part of the artifact format
(``trainer_state.ARTIFACT_FORMAT_VERSION``) — renaming or moving the module
would break the deserialization of previously saved models.
"""

from __future__ import annotations

import logging
import math
import warnings
from numbers import Real
from typing import Any, Literal

import numpy as np
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    _check_feature_names,
    check_is_fitted,
)

logger = logging.getLogger(__name__)

__all__ = ["Winsorizer"]


class Winsorizer(TransformerMixin, BaseEstimator):
    """Cap extreme values to the per-column bounds learned in ``fit``.

    The bounds are computed per column with ``np.nanquantile``, so missing
    values do not shift them; ``transform`` applies ``np.clip`` with the
    stored bounds and leaves ``NaN`` values unchanged.

    Two degenerate cases are handled in ``fit``:

    * an all-NaN column yields the bounds ``(-inf, +inf)`` — nothing is
      ever clipped;
    * a collapsed column (``lower == upper``, e.g. a constant column) is
      widened to ``(-inf, +inf)`` as well, so its constant value is not
      rewritten by the clip, and one WARNING per ``fit`` names all such
      columns.

    Other invariants:

    * the fitted transformer is pickled inside saved models, so this class's
      module path is part of the artifact format;
    * ``transform`` must not mutate the estimator state (sklearn
      ``check_dict_unchanged``), so the per-column clip counts of the last
      ``transform`` call are exposed through :meth:`get_last_clip_counts`
      instead of an attribute.

    Attributes:
        method: Winsorization method: ``"quantile"`` clips to
            ``[lower_quantile, upper_quantile]``, ``"iqr"`` clips to
            ``[Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR]``.
        lower_quantile: Lower quantile of the ``"quantile"`` method.
        upper_quantile: Upper quantile of the ``"quantile"`` method.
        iqr_multiplier: Multiplier of the IQR for the ``"iqr"`` method.
        lower_bounds_: Per-column lower bounds learned in ``fit``.
        upper_bounds_: Per-column upper bounds learned in ``fit``.
        n_features_in_: Number of features seen in ``fit``.
        feature_names_in_: Input feature names (DataFrame input only).
    """

    def __init__(
        self,
        method: Literal["quantile", "iqr"] = "quantile",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        iqr_multiplier: float = 1.5,
    ) -> None:
        """Initialize the transformer; no validation happens here (sklearn convention).

        Args:
            method: Winsorization method: ``"quantile"`` or ``"iqr"``.
            lower_quantile: Lower quantile of the ``"quantile"`` method.
            upper_quantile: Upper quantile of the ``"quantile"`` method.
            iqr_multiplier: Multiplier of the interquartile range for the
                ``"iqr"`` method.
        """
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.iqr_multiplier = iqr_multiplier

    def __sklearn_tags__(self) -> Any:
        """Declare estimator tags; NaN input is supported by design."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _validate_params(self) -> None:
        """Validate the constructor parameters.

        Non-numeric values (including booleans) and ``NaN`` values are
        rejected with ``ValueError`` as well, so any invalid parameter gives
        the documented exception type rather than an incidental ``TypeError``
        from a comparison.

        Raises:
            ValueError: If ``method`` is unknown, a quantile or the IQR
                multiplier is not a real number or is ``NaN``, a quantile is
                outside ``[0, 1]``, ``lower_quantile >= upper_quantile``, or
                ``iqr_multiplier <= 0``.
        """
        if self.method not in ("quantile", "iqr"):
            raise ValueError(
                f"Winsorizer: unknown method {self.method!r}; "
                "expected 'quantile' or 'iqr'."
            )
        self._validate_real_param("lower_quantile", self.lower_quantile)
        self._validate_real_param("upper_quantile", self.upper_quantile)
        self._validate_real_param("iqr_multiplier", self.iqr_multiplier)
        if not (0.0 <= self.lower_quantile <= 1.0):
            raise ValueError(
                f"Winsorizer: lower_quantile must be in [0, 1], "
                f"got {self.lower_quantile}."
            )
        if not (0.0 <= self.upper_quantile <= 1.0):
            raise ValueError(
                f"Winsorizer: upper_quantile must be in [0, 1], "
                f"got {self.upper_quantile}."
            )
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                f"Winsorizer: lower_quantile ({self.lower_quantile}) must be "
                f"strictly less than upper_quantile ({self.upper_quantile})."
            )
        if self.iqr_multiplier <= 0:
            raise ValueError(
                f"Winsorizer: iqr_multiplier must be positive, "
                f"got {self.iqr_multiplier}."
            )

    @staticmethod
    def _validate_real_param(name: str, value: Any) -> None:
        """Reject non-numeric and NaN values for a numeric parameter.

        Args:
            name: Parameter name for the error message.
            value: The parameter value to check.

        Raises:
            ValueError: If ``value`` is not a real number (booleans do not
                count) or is ``NaN``.
        """
        # bool is a subclass of int, so it must be excluded explicitly.
        # ValueError is deliberate: the module contract promises ValueError
        # for every invalid parameter, including non-numeric ones.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(  # noqa: TRY004
                f"Winsorizer: {name} must be a real number, "
                f"got {value!r} of type {type(value).__name__}."
            )
        if math.isnan(float(value)):
            raise ValueError(f"Winsorizer: {name} must not be NaN, got {value}.")

    def _to_2d_float(self, X: Any) -> np.ndarray:
        """Reduce the input to a 2D float ndarray (n_samples, n_features).

        Raises:
            TypeError: If a sparse matrix is passed or the input data is not
                convertible to float (e.g. string columns).
            ValueError: If the input has more than two dimensions, contains
                complex values, or has zero samples or features.
        """
        if sp.issparse(X):
            raise TypeError(
                "A sparse matrix was passed, but dense data is required. "
                "Winsorizer does not support sparse input."
            )
        arr = np.asarray(X)
        if np.iscomplexobj(arr):
            # The exact legacy wording is asserted by sklearn's
            # check_estimators_overwrite_params / complex-data checks.
            raise ValueError("Complex data not supported")
        try:
            # Conversion before the shape checks mirrors the original
            # behaviour; dtype checks happen via the conversion failure path
            # below so that numeric object arrays keep working (sklearn's
            # check_dtype_object relies on that).
            arr = arr.astype(float)
        except (TypeError, ValueError) as exc:
            # A string (object) input fails inside astype with a bare
            # "could not convert string to float"; re-raise it with the
            # Winsorizer context, keeping the sklearn-expected
            # "must be a string or a real number" wording for non-convertible
            # elements.
            raise TypeError(
                "Winsorizer expects numeric data; every element must be a "
                "string or a real number. "
                f"Underlying error: {exc}. "
                "Encode categorical/string columns before winsorization."
            ) from exc
        if arr.ndim == 1:
            raise ValueError(
                f"Expected 2D array, got 1D array instead:\narray={arr}.\n"
                "Reshape your data either using array.reshape(-1, 1) if your "
                "data has a single feature or array.reshape(1, -1) if it "
                "contains a single sample."
            )
        if arr.ndim != 2:
            raise ValueError(
                f"Winsorizer: expected a 2D input, got {arr.ndim} dimensions."
            )
        if arr.size == 0:
            raise ValueError(
                f"Found array with {arr.shape[0]} sample(s) and "
                f"{arr.shape[1]} feature(s) (shape={arr.shape}) while a "
                "minimum of 1 is required."
            )
        return arr

    def _validate_X_for_transform(self, X: Any) -> np.ndarray:
        """Validate a fitted-state input and reduce it to a 2D float ndarray.

        Shared by :meth:`transform` and :meth:`get_last_clip_counts` so the
        fitted-state checks and the error messages cannot drift apart.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            The input as a 2D float ndarray.

        Raises:
            TypeError: If the input is sparse or non-numeric.
            ValueError: If the input has an invalid shape or the column count
                differs from ``fit``.
            NotFittedError: If the transformer has not been fitted yet.
        """
        check_is_fitted(self, ("lower_bounds_", "upper_bounds_", "n_features_in_"))
        arr = self._to_2d_float(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, but {type(self).__name__} is "
                f"expecting {self.n_features_in_} features as input."
            )
        return arr

    def _input_names(self) -> np.ndarray | None:
        """Return the input feature names recorded by ``fit``, or ``None``."""
        return getattr(self, "feature_names_in_", None)

    def fit(self, X: Any, y: Any = None) -> Winsorizer:
        """Learn the per-column clipping bounds.

        Args:
            X: Input data of shape ``(n_samples, n_features)``; a DataFrame
                or anything convertible to a 2D float ndarray.
            y: Ignored, present for API consistency.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the parameters are invalid.
        """
        self._validate_params()
        _check_feature_names(self, X, reset=True)
        arr = self._to_2d_float(X)
        self.n_features_in_ = arr.shape[1]

        lower = np.empty(self.n_features_in_, dtype=float)
        upper = np.empty(self.n_features_in_, dtype=float)
        for col in range(self.n_features_in_):
            with warnings.catch_warnings():
                # An all-NaN column makes nanquantile return NaN; the
                # degenerate bounds below handle it without the noise.
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                if self.method == "quantile":
                    q_lo = np.nanquantile(arr[:, col], self.lower_quantile)
                    q_hi = np.nanquantile(arr[:, col], self.upper_quantile)
                else:
                    q1 = np.nanquantile(arr[:, col], 0.25)
                    q3 = np.nanquantile(arr[:, col], 0.75)
                    iqr = q3 - q1
                    q_lo = q1 - self.iqr_multiplier * iqr
                    q_hi = q3 + self.iqr_multiplier * iqr
            if np.isnan(q_lo) or np.isnan(q_hi):
                # All-NaN column: clipping to (-inf, +inf) changes nothing.
                q_lo, q_hi = -np.inf, np.inf
            elif q_lo == q_hi:
                # Collapsed column: the degenerate interval would rewrite its
                # constant value (and everything else) on clip; widen it
                # instead so the column passes through unchanged.
                q_lo, q_hi = -np.inf, np.inf
            lower[col] = q_lo
            upper[col] = q_hi

        self.lower_bounds_ = lower
        self.upper_bounds_ = upper

        collapsed_names = self._collapsed_column_names()
        if collapsed_names:
            logger.warning(
                "Winsorizer: %d column(s) have a degenerate clipping interval "
                "(constant or all-NaN in the training data) and are not "
                "clipped: %s",
                len(collapsed_names),
                ", ".join(collapsed_names),
            )
        return self

    def _collapsed_column_names(self) -> list[str]:
        """Return the names (or positional labels) of the degenerate columns."""
        names = self._input_names()
        collapsed = np.flatnonzero(
            np.isinf(self.lower_bounds_) & np.isinf(self.upper_bounds_)
        )
        return [
            str(names[col]) if names is not None else f"column_{col}"
            for col in collapsed
        ]

    def transform(self, X: Any) -> np.ndarray:
        """Clip the values to the bounds learned in ``fit``; NaN passes through.

        Args:
            X: Input data of shape ``(n_samples, n_features)`` with exactly
                ``n_features_in_`` columns.

        Returns:
            A float ndarray of the same shape with every column clipped to
            its learned ``[lower_bounds_[j], upper_bounds_[j]]`` interval.

        Raises:
            ValueError: If the column count differs from ``fit``.
            NotFittedError: If the transformer has not been fitted yet.
        """
        arr = self._validate_X_for_transform(X)
        clipped = np.clip(arr, self.lower_bounds_, self.upper_bounds_)
        # np.clip already keeps NaN, but do not rely on that: restore the
        # missing mask explicitly so NaN passes through for any dtype path.
        clipped[np.isnan(arr)] = np.nan

        n_clipped = int(np.sum((arr != clipped) & ~np.isnan(arr)))
        logger.debug(
            "Winsorizer: clipped %d value(s) across %d column(s)",
            n_clipped,
            self.n_features_in_,
        )
        return clipped

    def _clip_counts(self, X: np.ndarray, clipped: np.ndarray) -> np.ndarray:
        """Count the values that ``clip`` changed, per column.

        Args:
            X: The raw input, already validated and converted to a float
                ndarray by :meth:`_validate_X_for_transform`.
            clipped: The clipped output for the same input.

        Returns:
            An integer ndarray of length ``n_features_in_``.
        """
        # A value is counted when it differs from its clipped version and is
        # not NaN (NaN passes through unchanged).
        return np.sum((X != clipped) & ~np.isnan(X), axis=0).astype(int)

    def get_last_clip_counts(self, X: Any) -> dict[str, int]:
        """Per-column number of values that ``transform`` clips for ``X``.

        Kept out of the estimator state on purpose: ``transform`` must not
        mutate the fitted estimator (the sklearn ``check_dict_unchanged``
        contract), so the counts are computed on demand instead of being
        stored as an attribute during ``transform``.

        Args:
            X: The input to count the clipped values for; validated the same
                way as in :meth:`transform`.

        Returns:
            A mapping of column name (or positional label ``column_<j>`` for
            the nameless ndarray input) to the number of values outside the
            learned bounds.

        Raises:
            ValueError: If the column count differs from ``fit``.
            NotFittedError: If the transformer has not been fitted yet.
        """
        arr = self._validate_X_for_transform(X)
        clipped = np.clip(arr, self.lower_bounds_, self.upper_bounds_)
        counts = self._clip_counts(arr, clipped)
        names = self._input_names()
        return {
            str(names[col]) if names is not None else f"column_{col}": int(counts[col])
            for col in range(len(counts))
        }

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Map one input feature to one output feature, in order.

        Args:
            input_features: Input feature names; an array-like of length
                ``n_features_in_``. ``None`` falls back to the names seen in
                ``fit`` (DataFrame input) or to positional names.

        Returns:
            An ndarray of output feature names of length ``n_features_in_``.

        Raises:
            ValueError: If ``input_features`` has the wrong length or does
                not match the names seen in ``fit``.
            NotFittedError: If the transformer has not been fitted yet.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            names = self._input_names()
            if names is not None:
                return np.asarray(names, dtype=object)
            return np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        names = np.asarray(input_features, dtype=object)
        if names.ndim != 1 or len(names) != self.n_features_in_:
            raise ValueError(
                f"Winsorizer: input_features has {len(names)} name(s), but the "
                f"transformer was fitted with {self.n_features_in_}."
            )
        fitted_names = self._input_names()
        if fitted_names is not None and not np.array_equal(names, fitted_names):
            raise ValueError(
                "Winsorizer: input_features do not match the feature names seen in fit."
            )
        return names
