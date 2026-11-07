"""Target transformation for the model step of the trainer pipeline.

Provides :class:`TargetTransformer` — a picklable scikit-learn-style
transformer applied to the target ``y`` — and two helpers used by the
trainer:

* :func:`wrap_with_target_transform` — wraps a regressor into a
  ``TransformedTargetRegressor`` (identity for ``TargetTransform.none``);
* :func:`validate_target_for_transform` — domain validation of ``y``
  (``log1p`` requires ``min(y) > -1``, ``boxcox`` requires ``min(y) > 0``).

The transformer supports three methods:

* ``log1p`` — stateless ``log(1 + y)`` / ``expm1``;
* ``boxcox`` — the Box-Cox power transformation via an internal
  ``PowerTransformer(method="box-cox", standardize=True)``;
* ``yeo-johnson`` — the Yeo-Johnson power transformation via
  ``PowerTransformer(method="yeo-johnson", standardize=True)``.

    After ``fit`` the transformer remembers the transformed train range
    (``t_min_`` / ``t_max_``): ``inverse_transform`` clips its input into
    ``[t_min_ - margin, t_max_ + margin]`` where
    ``margin = clip_margin * (t_max_ - t_min_)`` (``clip_margin`` itself when
    the range is zero) before inverting. The clipping bounds linear-model
    extrapolation in log space: without it ``expm1`` returns ``inf`` for a
    prediction far outside the train range. Additionally the inverse output
    is clipped into the original train range expanded by the same relative
    margin (``y_min_`` / ``y_max_``): the power transformations (a negative
    Yeo-Johnson lambda, in particular) can return NaN beyond the domain of
    their inverse formula even for a clipped input, and a prediction must
    always be finite. The number of clipped values is logged at DEBUG level.
    Because of the clipping the inverse transformation is intentionally not
    exact outside the train range — the wrapper
    (``TransformedTargetRegressor``) is therefore created with
    ``check_inverse=False``.

``transform``/``inverse_transform`` expect and return ``(n, 1)`` arrays —
the shape ``TransformedTargetRegressor`` passes them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

from configurable_automl_engine.common.definitions import TargetTransform

_LOG = logging.getLogger(__name__)

__all__ = [
    "TargetTransformer",
    "fit_transform_target",
    "normalize_target_transform",
    "validate_target_for_transform",
    "wrap_with_target_transform",
]


def _normalize_method(method: TargetTransform | str) -> str:
    """Normalize a transformation method to the enum value string.

    Args:
        method: A ``TargetTransform`` member or one of its value strings
            (``"none"``, ``"log1p"``, ``"boxcox"``, ``"yeo-johnson"``).

    Returns:
        The value string of the method.

    Raises:
        ValueError: If the method is not a valid transformation.
    """
    try:
        return TargetTransform(method).value
    except ValueError as err:
        raise ValueError(
            f"Unknown target transform method: {method!r}. Expected one of "
            f"{[m.value for m in TargetTransform]}."
        ) from err


def normalize_target_transform(value: TargetTransform | str) -> TargetTransform:
    """Normalize a ``target_transform`` argument into the enum member.

    The single normalization point shared by the public APIs
    (``tuner.optimize``, the config schema) and the internal helpers: both
    ``TargetTransform`` members and their value strings (any case) are
    accepted.

    Args:
        value: A ``TargetTransform`` member or one of its value strings
            (case-insensitive).

    Returns:
        TargetTransform: The normalized enum member.

    Raises:
        ValueError: If the value is not a valid transformation.
    """
    if isinstance(value, TargetTransform):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in TargetTransform:
            if normalized == member.value:
                return member
    raise ValueError(
        f"Unknown target_transform: {value!r}. Expected one of "
        f"{[m.value for m in TargetTransform]}."
    )


def validate_target_for_transform(y: Any, method: TargetTransform | str) -> None:
    """Validate that ``y`` lies in the domain of the transformation.

    Args:
        y: The target vector (1-D, numeric).
        method: The transformation method (``log1p`` requires
            ``min(y) > -1``, ``boxcox`` requires ``min(y) > 0``;
            ``none`` and ``yeo-johnson`` accept any real ``y``).

    Raises:
        ValueError: If ``min(y)`` violates the domain of the method; the
            message names the method, the found minimum and suggests
            ``yeo-johnson`` (no domain restrictions).
    """
    method_str = _normalize_method(method)
    if method_str not in ("log1p", "boxcox"):
        return
    arr = np.asarray(y, dtype=float).reshape(-1)
    minimum = float(np.min(arr))
    lower_bound = -1.0 if method_str == "log1p" else 0.0
    if minimum <= lower_bound:
        raise ValueError(
            f"target_transform='{method_str}' requires min(y) > "
            f"{lower_bound}, but the minimum found in y is {minimum:.6g}. "
            "Use target_transform='yeo-johnson' (any real y) or transform "
            "the target before training."
        )


class TargetTransformer(TransformerMixin, BaseEstimator):
    """Transform the target ``y`` and invert with clipping to the train range.

    Module-level class (picklable). Parameters:

    * ``method`` — ``"log1p"``, ``"boxcox"`` or ``"yeo-johnson"``;
    * ``clip_margin`` — relative margin (a fraction of the transformed
      train range) added around ``[t_min_, t_max_]`` before the inverse
      transformation.

    Attributes (after ``fit``):

    * ``t_min_`` / ``t_max_`` — the transformed train range;
    * ``power_`` — the internal ``PowerTransformer`` for ``boxcox`` /
      ``yeo-johnson`` (``None`` for the stateless ``log1p``).

    ``transform``/``inverse_transform`` expect ``(n, 1)`` arrays (the shape
    ``TransformedTargetRegressor`` uses).
    """

    def __init__(self, method: str = "log1p", clip_margin: float = 0.5):
        self.method = method
        self.clip_margin = clip_margin

    def fit(self, y: Any, X: Any = None) -> TargetTransformer:
        """Fit the transformer and record the transformed train range.

        Args:
            y: The target values, an array of shape ``(n, 1)`` or ``(n,)``.
            X: Ignored (scikit-learn API compatibility).

        Returns:
            ``self``.

        Raises:
            ValueError: For an unknown method or a degenerate fit (see
                ``_fit_power`` for the constant-``y`` handling).
        """
        method = _normalize_method(self.method)
        self.power_: PowerTransformer | None
        if method == "log1p":
            self.power_ = None
        else:
            self.power_ = _fit_power(np.asarray(y, dtype=float), method)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        t = self._forward(y_arr.reshape(-1, 1))
        self.t_min_ = float(np.min(t))
        self.t_max_ = float(np.max(t))
        # The original-scale train range (the inverse output is clipped into
        # it, expanded by the same relative margin): guarantees finite
        # predictions even when the power-transform inverse formula returns
        # NaN/inf outside its domain.
        self.y_min_ = float(np.min(y_arr))
        self.y_max_ = float(np.max(y_arr))
        return self

    def transform(self, y: Any) -> np.ndarray:
        """Transform ``y`` into the model scale (shape ``(n, 1)``)."""
        return self._forward(np.asarray(y, dtype=float))

    def inverse_transform(self, t: Any) -> np.ndarray:
        """Clip ``t`` to the transformed train range and invert.

        The clip window is ``[t_min_ - m, t_max_ + m]`` with
        ``m = clip_margin * (t_max_ - t_min_)`` (``clip_margin`` when the
        range is zero). The inverse output is additionally clipped into
        ``[y_min_ - m * (y_max_ - y_min_), y_max_ + m * (y_max_ - y_min_)]``
        (the same relative margin in the original scale): the power
        transforms may return NaN beyond the domain of their inverse formula
        even for a clipped input. The number of clipped values (the union of
        the two clip masks, a value clipped at both levels counts once) is
        reported at DEBUG level.

        Args:
            t: Values in the transformed scale, shape ``(n, 1)`` or ``(n,)``.

        Returns:
            The values in the original scale, shape ``(n, 1)``.
        """
        t_arr = np.asarray(t, dtype=float).reshape(-1, 1)
        t_min = float(self.t_min_)
        t_max = float(self.t_max_)
        spread = t_max - t_min
        margin = self.clip_margin * spread if spread > 0.0 else self.clip_margin
        low = t_min - margin
        high = t_max + margin
        input_clipped = (t_arr < low) | (t_arr > high)
        t_arr = np.clip(t_arr, low, high)
        out = self._inverse(t_arr)
        # Second, output-side clip in the original scale: guards against
        # NaN/inf from the power-transform inverse outside its domain.
        y_spread = float(self.y_max_) - float(self.y_min_)
        y_margin = self.clip_margin * y_spread if y_spread > 0.0 else self.clip_margin
        y_low = float(self.y_min_) - y_margin
        y_high = float(self.y_max_) + y_margin
        out = np.asarray(out, dtype=float)
        output_clipped = ~np.isfinite(out) | (out < y_low) | (out > y_high)
        if np.any(output_clipped):
            out = np.where(
                output_clipped,
                np.clip(np.nan_to_num(out, nan=y_low), y_low, y_high),
                out,
            )
        # Union of the two clip masks (a value clipped at both levels is
        # counted once) reported at DEBUG level.
        n_clipped = int(
            np.count_nonzero(input_clipped.ravel() | output_clipped.ravel())
        )
        if n_clipped:
            _LOG.debug(
                "TargetTransformer(%s): %d value(s) of %d clipped to the "
                "train range [%.6g, %.6g] (margin %.6g) before/after the "
                "inverse transform.",
                self.method,
                n_clipped,
                t_arr.shape[0],
                y_low,
                y_high,
                y_margin,
            )
        return out

    # ── internals ──────────────────────────────────────────────────────────
    def _forward(self, y: np.ndarray) -> np.ndarray:
        """Apply the forward transformation to ``(n,)`` or ``(n, 1)`` input."""
        y_arr = y.reshape(-1, 1)
        if self.power_ is not None:
            return np.asarray(self.power_.transform(y_arr))
        # log1p: no state (the forward direction has no domain checks here —
        # validate_target_for_transform guards it before training).
        return np.log1p(y_arr)

    def _inverse(self, t: np.ndarray) -> np.ndarray:
        """Apply the inverse transformation to a ``(n, 1)`` array."""
        if self.power_ is not None:
            return np.asarray(self.power_.inverse_transform(t))
        return np.expm1(t)


def _fit_power(y: np.ndarray, method: str) -> PowerTransformer:
    """Fit the internal ``PowerTransformer`` for a power method.

    For a constant ``y`` (zero variance) the standardized power transform
    would divide by zero. The degenerate case falls back to
    ``standardize=False`` for ``yeo-johnson`` (which fits cleanly — sklearn
    keeps the lambda at 1.0); ``boxcox`` cannot represent a constant target
    at all (sklearn rejects it), so a clear ``ValueError`` is raised. The
    constant target is also rejected by the trainer's target validation
    before training, so these paths are defensive ones.

    Args:
        y: The target values, shape ``(n, 1)`` or ``(n,)``.
        method: ``"boxcox"`` or ``"yeo-johnson"``.

    Returns:
        The fitted ``PowerTransformer``.

    Raises:
        ValueError: If the power transformation cannot fit the target
            (a constant ``y`` for ``boxcox``; a non-positive ``y`` when not
            validated upstream).
    """
    flat = y.reshape(-1)
    constant = bool(np.all(flat == flat[0]))
    if constant and method == "boxcox":
        raise ValueError(
            "Cannot fit the boxcox target transformation to a constant "
            "target: the Box-Cox power transform requires a non-constant, "
            "strictly positive y."
        )
    power = PowerTransformer(
        method="box-cox" if method == "boxcox" else "yeo-johnson",
        standardize=not constant,
    )
    power.fit(y.reshape(-1, 1))
    return power


def wrap_with_target_transform(model: Any, method: TargetTransform | str) -> Any:
    """Wrap a regressor into a target-transformed regressor.

    Args:
        model: The base regressor.
        method: ``TargetTransform.none`` → the same object is returned
            (the pipeline keeps the original model type); any other method
            → ``TransformedTargetRegressor(regressor=model,
            transformer=TargetTransformer(method), check_inverse=False)``.
            ``check_inverse`` is disabled because the clipping makes the
            inverse transformation inexact outside the train range.

    Returns:
        The wrapped regressor (or the input object for ``none``).
    """
    method_str = _normalize_method(method)
    if method_str == TargetTransform.none.value:
        return model
    return TransformedTargetRegressor(
        regressor=model,
        transformer=TargetTransformer(method=method_str),
        check_inverse=False,
    )


def fit_transform_target(y: Any, method: TargetTransform | str) -> np.ndarray:
    """Fit the transformation on ``y`` and return it in the model scale.

    The single entry point for "train on the transformed ``y``" consumers
    that need the transformed values directly (the pre-HPO GLM target-domain
    guard): for ``none`` the input is returned unchanged (a 1-D float array),
    otherwise the target is validated against the method's domain (the same
    ``validate_target_for_transform`` contract as the runner and the trainer)
    and transformed by ``TargetTransformer``.

    Args:
        y: The target vector (1-D, numeric).
        method: The transformation method.

    Returns:
        The transformed target (1-D float array; the input object itself
        for ``none`` when it already is one).
    """
    method_str = _normalize_method(method)
    if method_str == TargetTransform.none.value:
        return np.asarray(y, dtype=float).reshape(-1)
    validate_target_for_transform(y, method_str)
    transformer = TargetTransformer(method=method_str)
    return transformer.fit_transform(np.asarray(y, dtype=float).reshape(-1)).reshape(-1)
