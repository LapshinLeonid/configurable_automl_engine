"""Target transformation for the model step of the trainer pipeline.

Provides :class:`TargetTransformer` — a picklable scikit-learn-style
transformer applied to the target ``y`` — a fold-local ``auto`` wrapper
(:class:`AutoTargetTransformer`) and three helpers used by the trainer:

* :func:`wrap_with_target_transform` — wraps a regressor into a
  ``TransformedTargetRegressor`` (identity for ``TargetTransform.none``);
* :func:`validate_target_for_transform` — domain validation of ``y``
  (``log1p`` requires ``min(y) > -1``, ``boxcox`` requires ``min(y) > 0``);
* :func:`resolve_target_transform` — the single decision point of the
  ``auto`` mode (the target-skewness rule).

The transformer supports these methods:

* ``log1p`` — stateless ``log(1 + y)`` / ``expm1``;
* ``boxcox`` — the Box-Cox power transformation via an internal
  ``PowerTransformer(method="box-cox", standardize=True)``;
* ``yeo-johnson`` — the Yeo-Johnson power transformation via
  ``PowerTransformer(method="yeo-johnson", standardize=True)``.

The ``auto`` mode of ``general.target_transform`` is resolved by
:func:`resolve_target_transform` from the target skewness. The decision
point depends on the estimation design (see the engine's ``data_prep``):
with a reserved outer hold-out the concrete method is decided once in
``prepare_dataset`` on the TRAIN part of the split, and every pipeline
stage sees the resolved method — ``auto`` never reaches the trainers.
Without an outer hold-out (the honest estimate is the BBC-CV score) the
decision would otherwise be made on the full target BEFORE the internal
cross-validation: the validation-fold target values would then influence
the choice of the transformation, and the same rows would later be used to
score the pipeline that choice built (a leak). In that mode ``auto`` stays
in the configuration and reaches the trainers, where every fold resolves
its own method from its ``y_train`` via :class:`AutoTargetTransformer` —
the values a fold is scored on never influence the method that fold trains
with.

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
import warnings
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

from configurable_automl_engine.common.definitions import TargetTransform

_LOG = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TARGET_TRANSFORM_SKEW_THRESHOLD",
    "AutoTargetTransformer",
    "TargetTransformer",
    "fit_transform_target",
    "normalize_target_transform",
    "resolve_target_transform",
    "validate_target_for_transform",
    "wrap_with_target_transform",
]

#: The default |skew(y)| threshold of the ``auto`` target-transform mode
#: (``general.target_transform_skew_threshold``). A target whose absolute
#: Fisher skewness exceeds the threshold is considered skewed and receives
#: a transformation; a (more symmetric) target keeps ``none``.
DEFAULT_TARGET_TRANSFORM_SKEW_THRESHOLD = 1.0


def _normalize_method(
    method: TargetTransform | str, *, allow_auto: bool = False
) -> str:
    """Normalize a transformation method to the enum value string.

    Args:
        method: A ``TargetTransform`` member or one of its value strings
            (``"none"``, ``"log1p"``, ``"boxcox"``, ``"yeo-johnson"``).
        allow_auto: Whether the ``auto`` placeholder is accepted (only the
            consumers implementing the fold-local resolution contract pass
            ``True``).

    Returns:
        The value string of the method.

    Raises:
        ValueError: If the method is not a valid transformation. The
            ``auto`` member is rejected too (unless ``allow_auto``): it is a
            configuration-level placeholder, resolved by
            :func:`resolve_target_transform` before the pipeline runs — it
            must never reach the plain transformer.
    """
    try:
        value = TargetTransform(method).value
    except ValueError as err:
        raise ValueError(
            f"Unknown target transform method: {method!r}. Expected one of "
            f"{[m.value for m in TargetTransform]}."
        ) from err
    if value == TargetTransform.auto.value and not allow_auto:
        raise ValueError(
            "target_transform='auto' must be resolved into a concrete "
            "method (resolve_target_transform) before training; the "
            "transformer does not accept the 'auto' placeholder."
        )
    return value


def normalize_target_transform(
    value: TargetTransform | str, *, allow_auto: bool = False
) -> TargetTransform:
    """Normalize a ``target_transform`` argument into the enum member.

    The single normalization point shared by the public APIs
    (``tuner.optimize``, the config schema) and the internal helpers: both
    ``TargetTransform`` members and their value strings (any case) are
    accepted.

    The ``auto`` member is accepted ONLY through :func:`allow_auto`: it is a
    configuration-level placeholder, and a consumer that cannot honor the
    fold-local resolution contract must reject it.

    Args:
        value: A ``TargetTransform`` member or one of its value strings
            (case-insensitive).
        allow_auto: Whether the ``auto`` placeholder is accepted (the
            fold-local resolution path of the engine — the mode without an
            outer hold-out — passes ``True``; every other consumer keeps the
            strict default).

    Returns:
        TargetTransform: The normalized enum member.

    Raises:
        ValueError: If the value is not a valid transformation, or ``auto``
            is passed with ``allow_auto=False``: it must be resolved by
            :func:`resolve_target_transform` before training (with a
            reserved outer hold-out the engine resolves it in
            ``prepare_dataset``).
    """
    if isinstance(value, TargetTransform):
        if value is TargetTransform.auto and not allow_auto:
            # 'auto' is a configuration-level placeholder: it must be
            # resolved by resolve_target_transform before the pipeline
            # runs and never reaches the consumers of a normalized method
            # (the fold-local resolution path is the only exception).
            raise ValueError(
                "target_transform='auto' must be resolved into a concrete "
                "method (resolve_target_transform) before training."
            )
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == TargetTransform.auto.value and not allow_auto:
            raise ValueError(
                "target_transform='auto' must be resolved into a concrete "
                "method (resolve_target_transform) before training."
            )
        for member in TargetTransform:
            if normalized == member.value:
                return member
    raise ValueError(
        f"Unknown target_transform: {value!r}. Expected one of "
        f"{[m.value for m in TargetTransform]}."
    )


def validate_target_for_transform(
    y: Any, method: TargetTransform | str, *, allow_auto: bool = False
) -> None:
    """Validate that ``y`` lies in the domain of the transformation.

    Args:
        y: The target vector (1-D, numeric).
        method: The transformation method (``log1p`` requires
            ``min(y) > -1``, ``boxcox`` requires ``min(y) > 0``;
            ``none`` and ``yeo-johnson`` accept any real ``y``).
        allow_auto: Whether the ``auto`` placeholder is a no-op here
            (``True`` in the trainer's fit: the concrete method is not
            decided yet in the fold-local resolution mode — the domain of
            the resolved method is validated by the resolution itself).

    Raises:
        ValueError: If ``min(y)`` violates the domain of the method; the
            message names the method, the found minimum and suggests
            ``yeo-johnson`` (no domain restrictions).
    """
    method_str = _normalize_method(method, allow_auto=allow_auto)
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


class AutoTargetTransformer(TransformerMixin, BaseEstimator):
    """Fold-local resolution of ``target_transform: 'auto'``.

    A scikit-learn-style wrapper that resolves the ``auto`` placeholder at
    ``fit`` time from the ``y`` it is fitted on (the FOLD's train target,
    when used inside a cross-validated pipeline) and delegates everything
    else to the resolved :class:`TargetTransformer`.

    Honesty contract (why this class exists): in the engine's mode without
    an outer hold-out the ``auto`` decision must not be made on the full
    target before the internal cross-validation — the validation-fold
    values would influence the choice of the transformation while the same
    rows are later used to score the pipeline that choice built. This
    wrapper guarantees the decision sample of every fitted instance is
    exactly the ``y`` passed to its ``fit`` (a fold's ``y_train`` — never
    the fold's validation rows).

    Semantics:

    * ``fit(y)`` runs :func:`resolve_target_transform` on ``y`` with the
      configured ``skew_threshold`` and stores the resolved concrete
      method in ``resolved_method_``; the resolution is announced at INFO
      level (the same log line as the global ``auto`` resolution).
    * ``transform`` / ``inverse_transform`` behave exactly like the
      resolved ``TargetTransformer`` (including the inverse clipping to
      the fitted train range). ``transform`` before ``fit`` raises.
    * The resolved method is part of the fitted state: a pickled artifact
      keeps the concrete method its training folds chose (per fold), and
      ``transform`` always uses the method resolved at fit time — a later
      refit re-resolves from its own data.

    Attributes (after ``fit``):
        resolved_method_ (TargetTransform): The concrete method the fit
            resolved from its own ``y``.

    Parameters:
        skew_threshold: The ``|skew(y)|`` threshold of the ``auto``
            decision (``general.target_transform_skew_threshold``); must
            be strictly positive.

    ``transform``/``inverse_transform`` expect and return ``(n, 1)`` arrays —
    the shape ``TransformedTargetRegressor`` passes them.
    """

    def __init__(self, skew_threshold: float = DEFAULT_TARGET_TRANSFORM_SKEW_THRESHOLD):
        self.skew_threshold = skew_threshold

    def fit(self, y: Any, X: Any = None) -> AutoTargetTransformer:
        """Resolve the method from ``y`` and fit the inner transformer.

        Args:
            y: The target values the decision is made on (an array of shape
                ``(n, 1)`` or ``(n,)``; the fold's train target inside a
                cross-validated pipeline).
            X: Ignored (scikit-learn API compatibility).

        Returns:
            ``self``.

        Raises:
            ValueError: If the resolution fails (an invalid threshold or an
                empty target) or the inner ``TargetTransformer.fit`` fails
                (a degenerate target, see ``_fit_power``).
        """
        resolved = resolve_target_transform(
            y, TargetTransform.auto, skew_threshold=self.skew_threshold
        )
        self.resolved_method_: TargetTransform = resolved
        # 'none' is the identity: no inner transformer (the plain
        # TargetTransformer implements only the transforming methods).
        self.transformer_: TargetTransformer | None = (
            None
            if resolved is TargetTransform.none
            else TargetTransformer(method=resolved.value).fit(y)
        )
        return self

    def transform(self, y: Any) -> np.ndarray:
        """Transform ``y`` with the method resolved at fit time."""
        inner = self._inner()
        if inner is None:
            return np.asarray(y, dtype=float).reshape(-1, 1)
        return inner.transform(y)

    def inverse_transform(self, t: Any) -> np.ndarray:
        """Invert ``y`` with the method resolved at fit time (with clipping)."""
        inner = self._inner()
        if inner is None:
            return np.asarray(t, dtype=float).reshape(-1, 1)
        return inner.inverse_transform(t)

    def _inner(self) -> TargetTransformer | None:
        """The fitted inner transformer (``None`` — the identity, 'none')."""
        if not hasattr(self, "transformer_"):
            raise RuntimeError(
                "This AutoTargetTransformer instance is not fitted yet: "
                "call fit before transform/inverse_transform."
            )
        return self.transformer_


def wrap_with_target_transform(
    model: Any,
    method: TargetTransform | str,
    *,
    skew_threshold: float | None = None,
) -> Any:
    """Wrap a regressor into a target-transformed regressor.

    Args:
        model: The base regressor.
        method: ``TargetTransform.none`` → the same object is returned
            (the pipeline keeps the original model type); ``auto`` → the
            regressor is wrapped into a
            ``TransformedTargetRegressor`` around an
            :class:`AutoTargetTransformer`: every ``fit`` resolves the
            concrete method from ITS OWN ``y`` (the fold-local resolution
            of the engine's no-hold-out mode — see the module docstring);
            any other method → ``TransformedTargetRegressor(regressor=model,
            transformer=TargetTransformer(method), check_inverse=False)``.
            ``check_inverse`` is disabled because the clipping makes the
            inverse transformation inexact outside the train range.
        skew_threshold: The ``|skew(y)|`` threshold of the ``auto``
            decision (``general.target_transform_skew_threshold``);
            ``None`` — the :class:`AutoTargetTransformer` module default.
            Ignored for the other methods.

    Returns:
        The wrapped regressor (or the input object for ``none``).
    """
    method_str = _normalize_method(method, allow_auto=True)
    if method_str == TargetTransform.none.value:
        return model
    if method_str == TargetTransform.auto.value:
        transformer: TargetTransformer | AutoTargetTransformer = (
            AutoTargetTransformer(skew_threshold=skew_threshold)
            if skew_threshold is not None
            else AutoTargetTransformer()
        )
    else:
        transformer = TargetTransformer(method=method_str)
    return TransformedTargetRegressor(
        regressor=model,
        transformer=transformer,
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


def resolve_target_transform(
    y: Any,
    method: TargetTransform | str,
    skew_threshold: float = DEFAULT_TARGET_TRANSFORM_SKEW_THRESHOLD,
) -> TargetTransform:
    """Resolve the ``auto`` target-transform mode from the target skewness.

    The single decision point of ``general.target_transform: auto``: the
    engine computes the Fisher skewness of ``y`` (``scipy.stats.skew``) and

    * ``|skew(y)| > skew_threshold`` → ``log1p`` when ``min(y) > -1``
      (the domain of ``log1p``), otherwise ``yeo-johnson`` (any real ``y``);
    * ``|skew(y)| <= skew_threshold`` → ``none``.

    The rule keys on the skewness MAGNITUDE, not its sign — a deliberate
    simplification: ``log1p`` is optimal for the common right-skewed
    target, and for a strongly LEFT-skewed target with ``min(y) > -1``
    ``yeo-johnson`` would typically symmetrize better (a sign-aware
    choice ``skew > +thr → log1p``, ``skew < -thr → yeo-johnson`` is a
    possible future refinement). Users who need the sign-aware behavior
    set an explicit method.

    The decision is computed at the decision point of the estimation design:
    with a reserved outer hold-out the engine calls it once in
    ``prepare_dataset`` on the train part of the split, and every downstream
    consumer (the HPO trials, the final trainer, the hold-out trainer, the
    artifact metadata, the ``ExperimentResult``) receives the resolved
    concrete method — ``auto`` itself never reaches them. Without an outer
    hold-out the decision moves INSIDE every cross-validation fold (via
    :class:`AutoTargetTransformer`): the validation-fold values must not
    influence the choice of the transformation they are later scored with.
    ``auto`` is the only member that resolves: any explicit method is
    returned unchanged. A non-finite skewness (a constant or otherwise
    degenerate ``y``, where the sample skewness is undefined) resolves to
    ``none``: no skewness evidence, no transformation (the engine rejects a
    constant target before this point anyway).

    Args:
        y: The target vector (1-D, numeric). Ignored for a non-``auto``
            method.
        method: The configured method — a ``TargetTransform`` member or one
            of its value strings (case-insensitive). ``auto`` triggers the
            skewness decision; anything else is normalized and returned.
        skew_threshold: The ``|skew(y)|`` threshold of the ``auto``
            decision (``general.target_transform_skew_threshold``); must
            be strictly positive.

    Returns:
        TargetTransform: The resolved concrete method (never ``auto``).

    Raises:
        ValueError: If ``skew_threshold`` is not strictly positive, the
            method is not a valid transformation, or ``y`` is empty.
    """
    if skew_threshold <= 0:
        raise ValueError(
            f"target_transform_skew_threshold must be > 0, got {skew_threshold!r}."
        )
    if isinstance(method, str):
        normalized = method.strip().lower()
        try:
            method = TargetTransform(normalized)
        except ValueError as err:
            raise ValueError(
                f"Unknown target_transform: {method!r}. Expected one of "
                f"{[m.value for m in TargetTransform]}."
            ) from err
    if method is not TargetTransform.auto:
        return method
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot resolve target_transform='auto': y is empty.")
    # A constant (or otherwise degenerate) y makes the sample skewness
    # undefined: scipy returns NaN with RuntimeWarnings about precision
    # loss. There is no skewness evidence to act on — resolve to 'none'
    # deterministically (the engine's constant-target check rejects such
    # data before this point anyway; this guard makes the PUBLIC function
    # predictable on any input).
    with warnings.catch_warnings():
        # The precision-loss RuntimeWarnings of scipy are noise here: the
        # NaN outcome is handled explicitly below.
        warnings.simplefilter("ignore", RuntimeWarning)
        skew = float(stats.skew(arr, bias=False))
    minimum = float(np.min(arr))
    if np.isfinite(skew) and abs(skew) > skew_threshold:
        resolved = (
            TargetTransform.log1p if minimum > -1.0 else TargetTransform.yeo_johnson
        )
    else:
        resolved = TargetTransform.none
    _LOG.info(
        "target_transform='auto': skew(y) = %.4f (threshold %.4f), "
        "min(y) = %.6g → resolved target_transform='%s'.",
        skew,
        skew_threshold,
        minimum,
        resolved.value,
    )
    return resolved
