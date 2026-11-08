"""Stable adapter over the private scikit-learn details the engine relies on.

The engine uses two families of scikit-learn internals that have no public
equivalent:

* the raw metric of a ``make_scorer`` object (``_score_func``/``_sign``/
  ``_kwargs``) — required to compute pooled out-of-fold and BBC-CV metrics
  over concatenated prediction vectors (a per-fold average is meaningless
  for non-additive metrics such as RMSE or R²);
* the feature-name recording of ``BaseEstimator`` (``_check_feature_names``)
  — required so custom transformers store ``feature_names_in_`` in ``fit``
  the same way built-in scikit-learn estimators do.

This module is the ONLY place in the package that names the private
attributes. Every consumer works with the stable contract below; when a
future scikit-learn release removes the private attributes, only this
adapter has to be revisited (and its tests) — the callers keep their
fallback semantics.

Contract:

- ``scorer_raw_parts(scorer)`` — ``(score_func, sign, kwargs)`` of a scorer;
  ``score_func`` is ``None`` when the scorer does not expose a raw metric
  (a plain callable scorer adapter or any other duck-typed scorer).
- ``scorer_sign(scorer)`` — the direction flag (+1 / -1) or ``None``.
- ``check_feature_names(estimator, X)`` — the ``reset=True`` half of
  scikit-learn's ``_check_feature_names``: record ``feature_names_in_``
  from a DataFrame-like input, drop the stale attribute on nameless input.

The private functions themselves are imported defensively: the import is
guarded so a hypothetical import failure degrades to the attribute-based
implementation instead of breaking the package import.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "check_feature_names",
    "scorer_raw_parts",
    "scorer_sign",
]

try:  # private sklearn API — may change between releases (see module docstring)
    from sklearn.utils.validation import _check_feature_names as _sk_check_feature_names
except ImportError:  # pragma: no cover - exercised only on future sklearn releases
    _sk_check_feature_names = None


def scorer_raw_parts(scorer: Any) -> tuple[Any, int, dict[str, Any]]:
    """Extract the raw metric parts of a scikit-learn ``make_scorer`` object.

    The pooled out-of-fold evaluation (``metrics.pooled_scorer_value``,
    ``metrics._columnwise_raw_values``, ``metrics.bbc_cv_score``) needs the
    raw ``(y_true, y_pred) -> float`` function with its sign and static
    keyword arguments: it re-applies the metric over the concatenated
    prediction vector instead of averaging per-fold values. These parts are
    private scikit-learn attributes (``_score_func``/``_sign``/``_kwargs``)
    with no public equivalent; this function localizes the access.

    Args:
        scorer: A scorer object (a ``make_scorer`` result, a
            ``_CallableScorer`` adapter or any duck-typed scorer).

    Returns:
        tuple[Any, int, dict[str, Any]]: ``(score_func, sign, kwargs)``;
            ``score_func`` is ``None`` when the scorer does not expose a raw
            metric function — such a scorer cannot be pooled and the caller
            falls back to per-fold averaging. ``sign`` defaults to ``+1`` and
            ``kwargs`` to ``{}`` when the respective attributes are missing.
    """
    score_func = getattr(scorer, "_score_func", None)
    sign = getattr(scorer, "_sign", 1)
    kwargs = getattr(scorer, "_kwargs", None)
    return score_func, int(sign), dict(kwargs) if kwargs else {}


def scorer_sign(scorer: Any) -> int | None:
    """Return the direction flag of a scorer (``None`` when not exposed).

    ``+1`` — "higher is better" (score metrics), ``-1`` — an inverted error
    metric (``neg_`` scorers and the ``make_scorer`` wrappers built with
    ``greater_is_better=False``). ``None`` — the scorer does not expose the
    flag (e.g. a plain callable scorer); the callers apply their documented
    conservative fallback.

    Args:
        scorer: A scorer object.

    Returns:
        int | None: The direction flag, or ``None``.
    """
    sign = getattr(scorer, "_sign", None)
    return None if sign is None else int(sign)


def _record_feature_names(estimator: Any, X: Any) -> None:
    """Record ``feature_names_in_`` on ``estimator`` from ``X`` (public fallback).

    The own implementation of the ``reset=True`` branch of scikit-learn's
    ``_check_feature_names``: DataFrame-like inputs (anything with a
    ``columns`` attribute) record their column names, nameless inputs (for
    example a plain ndarray) drop a stale ``feature_names_in_`` left by a
    previous ``fit`` on named data.

    Args:
        estimator: The estimator being fitted.
        X: The input data (a DataFrame-like object or an array).
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        # Nameless input: delete the attribute recorded by an earlier fit on
        # named data (the sklearn convention).
        if hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")
        return
    estimator.feature_names_in_ = np.asarray(list(columns), dtype=object)


def check_feature_names(estimator: Any, X: Any) -> None:
    """Record the input feature names on ``estimator`` (the ``fit`` convention).

    The ``reset=True`` half of scikit-learn's private
    ``_check_feature_names``: called from ``fit``, it stores the column
    names of a DataFrame-like ``X`` as ``feature_names_in_`` (an object
    array of strings) and removes the stale attribute when ``X`` has no
    names. The private scikit-learn function is used when available; on a
    future scikit-learn release that drops it, the own recording above
    keeps the engine behavior (recording the names and letting the
    downstream ``get_feature_names_out`` checks validate them).

    Only the recording is replicated here: the fit/reset path never
    *validates* names (the engine transformers do not check names on
    transform — the DataFrame column count mismatch is caught by the
    fitted ``n_features_in_``), so the warning/validation branches of the
    private function are not needed.

    Args:
        estimator: The estimator being fitted (``self`` of the caller).
        X: The input data of the ``fit`` call.
    """
    if _sk_check_feature_names is not None:
        _sk_check_feature_names(estimator, X, reset=True)
        return
    logger.debug(
        "sklearn.utils.validation._check_feature_names is unavailable; "
        "using the own feature-name recording of %s.",
        type(estimator).__name__,
    )
    _record_feature_names(estimator, X)
