"""Fold-safe quantile scaling transformer.

Holds :class:`_FoldSafeQuantileTransformer`: a thin picklable wrapper over
:class:`sklearn.preprocessing.QuantileTransformer` that clips
``n_quantiles`` to the number of rows of the fitted fold before delegating
to sklearn. On cross-validation folds (and in leave-one-out validation in
particular) a fold can contain fewer rows than the configured
``n_quantiles``; without the clipping sklearn would emit its
``UserWarning`` ("n_quantiles is greater than the total number of samples")
on every fold fit.

The fitted transformers are pickled inside saved models, so the module path
is part of the artifact format (``trainer_state.ARTIFACT_FORMAT_VERSION``):
renaming or moving the module would break the deserialization of previously
saved models. The module must not import the other preprocessing modules.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer

# Private sklearn API, but stable across the supported range
# (scikit-learn>=1.8): it is also used by the Winsorizer
# (preprocessing/outliers.py). Revisit the import on a major sklearn
# upgrade; a fallback shim would silently change the validation semantics,
# so a hard import error is preferable.
from sklearn.utils.validation import _check_feature_names, check_is_fitted

logger = logging.getLogger(__name__)

__all__ = ["_FoldSafeQuantileTransformer"]


class _FoldSafeQuantileTransformer(TransformerMixin, BaseEstimator):
    """QuantileTransformer that clips ``n_quantiles`` to the fold size.

    ``fit`` sets ``n_quantiles = min(n_quantiles, n_rows)`` before the
    sklearn transformer is created, so sklearn never emits its
    ``UserWarning`` about ``n_quantiles`` exceeding the number of samples;
    the effective value is exposed as ``n_quantiles_``. Every clipping is
    logged once at DEBUG level.

    ``output_distribution='normal'`` is computed from the uniform output as
    ``norm.ppf(0.5/n + u * (1 - 1/n))`` (``n = n_quantiles_``) instead of
    sklearn's own normal output, which maps the training extremes to its
    ±5.2 clipping bound and thus creates outliers on small folds; the
    transformed values are bounded by ``±ppf(1 - 0.5/n)``.

    The transformer follows the standard sklearn API: ``get_params``/
    ``set_params`` (via :class:`~sklearn.base.BaseEstimator`), ``fit``/
    ``transform``/``fit_transform`` (via
    :class:`~sklearn.base.TransformerMixin`), ``get_feature_names_out``
    (one-to-one: one output column per input column — the transform never
    changes the number of columns).

    Attributes:
        n_quantiles_: The effective number of quantiles after clipping
            (``min(n_quantiles, n_rows)``), available after ``fit``.
    """

    def __init__(
        self,
        *,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        n_quantiles: int = 1000,
        subsample: int = 100_000,
        random_state: int | None = None,
    ) -> None:
        """Store the transformer parameters (no logic in ``__init__``).

        Args:
            output_distribution: Marginal distribution of the transformed
                data: ``'uniform'`` or ``'normal'``.
            n_quantiles: Requested number of quantiles (``>= 10``); clipped
                to the number of rows in ``fit``.
            subsample: Maximum number of samples used to estimate the
                quantiles (``>= n_quantiles``). Always forwarded to sklearn,
                so the behavior does not depend on the sklearn default
                (10_000 as of scikit-learn 1.8).
            random_state: Seed for the subsampling of large datasets; makes
                the transform deterministic for a fixed seed.
        """
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> _FoldSafeQuantileTransformer:
        """Clip ``n_quantiles`` to the number of rows and fit sklearn's transformer.

        Args:
            X: The input data of shape ``(n_samples, n_features)``.
            y: Ignored (present for API compatibility).

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If ``X`` is empty (no rows), or sklearn rejects the
                parameters (e.g. ``subsample < n_quantiles``).
        """
        _check_feature_names(self, X, reset=True)
        # X.shape[0] instead of np.asarray(X).shape[0]: both ndarray and
        # DataFrame expose .shape, and asarray could materialize a copy of a
        # mixed-dtype input just to read the row count.
        n_rows = int(X.shape[0])
        effective_n_quantiles = min(self.n_quantiles, n_rows)
        if effective_n_quantiles < self.n_quantiles:
            logger.debug(
                "_FoldSafeQuantileTransformer: n_quantiles clipped from %d "
                "to the fold size %d.",
                self.n_quantiles,
                effective_n_quantiles,
            )
        self.n_quantiles_ = effective_n_quantiles
        # sklearn's 'normal' output maps the training minimum/maximum to its
        # clipping bound (about ±5.2) while the neighbouring points get
        # ordinary values: on a 14-row fold the extreme row sits 3.8 away
        # from its neighbour, i.e. the scaler itself manufactures two
        # high-leverage outliers per column. The 'normal' output is built on
        # top of the 'uniform' one instead (see ``transform``).
        self._transformer_ = QuantileTransformer(
            n_quantiles=effective_n_quantiles,
            output_distribution="uniform",
            subsample=self.subsample,
            random_state=self.random_state,
        ).fit(X)
        return self

    def transform(self, X: Any) -> Any:
        """Transform the data with the fitted quantiles.

        Args:
            X: The input data of shape ``(n_samples, n_features)``.

        Returns:
            The transformed data with the same number of columns.

        Raises:
            NotFittedError: If called before ``fit``.
        """
        check_is_fitted(self, "_transformer_")
        uniform = self._transformer_.transform(X)
        # A model pickled before the 'normal' remap holds an inner
        # transformer that already outputs the normal distribution: it is
        # returned as is, so old artifacts keep predicting identically.
        if (
            self.output_distribution != "normal"
            or self._transformer_.output_distribution == "normal"
        ):
            return uniform
        return self._uniform_to_normal(uniform, self.n_quantiles_)

    @staticmethod
    def _uniform_to_normal(uniform: Any, n_quantiles: int) -> Any:
        """Map the uniform output to a normal one without extreme tails.

        ``u`` is squeezed into ``[0.5/n, 1 - 0.5/n]`` (the mid-points of
        the outermost quantile bins, ``n`` — the effective number of
        quantiles) before ``norm.ppf``, so the training extremes land at
        ``±ppf(1 - 0.5/n)`` (±1.80 on 14 rows) instead of sklearn's ±5.2
        clipping bound. NaN passes through unchanged.

        Args:
            uniform: The uniform-distributed output (dense ndarray).
            n_quantiles: The effective number of quantiles.

        Returns:
            The normal-distributed output of the same shape.
        """
        half = 0.5 / n_quantiles

        def _map(values: np.ndarray) -> np.ndarray:
            return np.asarray(
                norm.ppf(half + np.clip(values, 0.0, 1.0) * (1.0 - 2.0 * half)),
                dtype=values.dtype if values.dtype.kind == "f" else float,
            )

        return _map(np.asarray(uniform))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """One-to-one mapping: one output column per input column.

        Args:
            input_features: Input feature names (validated against the
                names seen in ``fit`` when provided).

        Returns:
            The output feature names, identical to the input ones.
        """
        check_is_fitted(self, "_transformer_")
        return self._transformer_.get_feature_names_out(input_features)
