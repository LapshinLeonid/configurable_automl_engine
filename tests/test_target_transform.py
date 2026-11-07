"""Tests for the target transformation building blocks.

Coverage (``preprocessing.target_transform``):
    1. ``TargetTransformer``: the round trip (transform → inverse_transform)
       of each method on in-range data; the inverse-clip of out-of-range
       values (finite results, DEBUG log of the clipped count, zero-range
       margin); pickle/clone compatibility; empty and constant ``y``.
    2. ``validate_target_for_transform``: the domain boundary values of
       each method (-1, -0.999, 0, 1e-9, ...); the error message content
       (method, minimum, yeo-johnson hint).
    3. ``wrap_with_target_transform``: ``none`` → the same object
       (identity); other methods → a ``TransformedTargetRegressor`` with
       ``check_inverse=False`` that returns predictions in the original
       scale.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from configurable_automl_engine.common.definitions import TargetTransform
from configurable_automl_engine.preprocessing.target_transform import (
    TargetTransformer,
    validate_target_for_transform,
    wrap_with_target_transform,
)

METHODS = ["log1p", "boxcox", "yeo-johnson"]


def _positive_target(n: int = 60, seed: int = 7) -> np.ndarray:
    """A positive skewed target in the domain of every method."""
    rng = np.random.RandomState(seed)
    return np.exp(rng.randn(n) * 0.5 + 1.0).reshape(-1, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  Round trip per method
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_round_trip_in_range(method: str) -> None:
    y = _positive_target()
    t = TargetTransformer(method=method).fit(y)
    recovered = t.inverse_transform(t.transform(y))
    assert np.allclose(recovered, y, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_transform_output_shape(method: str) -> None:
    y = _positive_target()
    t = TargetTransformer(method=method).fit(y)
    assert t.transform(y).shape == (y.shape[0], 1)
    assert t.inverse_transform(t.transform(y)).shape == (y.shape[0], 1)


def test_log1p_matches_numpy() -> None:
    y = _positive_target()
    t = TargetTransformer(method="log1p").fit(y)
    assert np.allclose(t.transform(y), np.log1p(y))
    assert np.allclose(t.inverse_transform(np.log1p(y)), y)


def test_power_methods_use_power_transformer() -> None:
    y = _positive_target()
    for method in ("boxcox", "yeo-johnson"):
        t = TargetTransformer(method=method).fit(y)
        assert t.power_ is not None
        # Standardized power-transformed output has (approximately) unit
        # variance by construction.
        assert float(np.std(t.transform(y))) == pytest.approx(1.0, abs=1e-6)
    t_log = TargetTransformer(method="log1p").fit(y)
    assert t_log.power_ is None


# ──────────────────────────────────────────────────────────────────────────────
#  Inverse clipping
# ──────────────────────────────────────────────────────────────────────────────


def test_inverse_clips_out_of_range_values() -> None:
    y = _positive_target()
    t = TargetTransformer(method="log1p", clip_margin=0.5).fit(y)
    transformed = t.transform(y)
    t_min, t_max = float(transformed.min()), float(transformed.max())
    spread = t_max - t_min
    # Values far outside the train range (linear-model extrapolation):
    # without the clip expm1 would return huge/inf values.
    far = np.array([[-1e6], [1e6]])
    out = t.inverse_transform(far)
    high_bound = t_max + 0.5 * spread
    expected = np.expm1(np.clip(far, t_min - 0.5 * spread, high_bound))
    # The output-side clip into the original train range (expanded by the
    # same relative margin) binds before expm1 of the t-window here.
    y_spread = float(y.max() - y.min())
    y_high = float(y.max()) + 0.5 * y_spread
    y_low = float(y.min()) - 0.5 * y_spread
    expected = np.clip(expected, y_low, y_high)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, expected)


def test_inverse_clip_counts_logged_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    y = _positive_target()
    t = TargetTransformer(method="log1p", clip_margin=0.5).fit(y)
    with caplog.at_level(logging.DEBUG, logger="configurable_automl_engine"):
        t.inverse_transform(np.array([[1e6]]))
    assert any("clipped" in r.message for r in caplog.records)


def test_inverse_clip_zero_range_uses_margin_directly() -> None:
    # A constant y gives a zero transformed range: the margin equals
    # clip_margin itself on both clip levels (zero spread → the same
    # absolute margin in the original scale). The result is exactly
    # log1p(3) + 0.25 clipped into [2.75, 3.25] → 3.25.
    y = np.full((20, 1), 3.0)
    t = TargetTransformer(method="log1p", clip_margin=0.25).fit(y)
    assert t.t_max_ - t.t_min_ == pytest.approx(0.0)
    out = t.inverse_transform(np.array([[100.0]]))
    assert np.allclose(out, 3.25)
    out_low = t.inverse_transform(np.array([[-100.0]]))
    assert np.allclose(out_low, 2.75)


# ──────────────────────────────────────────────────────────────────────────────
#  Edge cases: constant / empty y
# ──────────────────────────────────────────────────────────────────────────────


def test_constant_y_log1p_round_trip() -> None:
    y = np.full((15, 1), 2.0)
    t = TargetTransformer(method="log1p").fit(y)
    assert np.allclose(t.inverse_transform(t.transform(y)), y)


@pytest.mark.parametrize("method", ["yeo-johnson", "boxcox"])
def test_constant_y_power_methods(method: str) -> None:
    # A constant y: the standardized PowerTransformer would divide by zero —
    # the transformer must either fit cleanly (yeo-johnson, which falls back
    # to standardize=False) or raise a clear domain error (boxcox, which
    # cannot represent a constant target). The trainer rejects a constant
    # target before training, so both are acceptable defensive behaviors.
    y = np.full((15, 1), 4.0)
    t = TargetTransformer(method=method)
    if method == "boxcox":
        with pytest.raises(ValueError, match="constant target"):
            t.fit(y)
        return
    t.fit(y)
    recovered = t.inverse_transform(t.transform(y))
    assert np.allclose(recovered, y)


def test_empty_y_fit_raises() -> None:
    t = TargetTransformer(method="log1p")
    with pytest.raises(ValueError):
        t.fit(np.empty((0, 1)))


# ──────────────────────────────────────────────────────────────────────────────
#  Pickle / clone
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_pickle_round_trip(method: str) -> None:
    y = _positive_target()
    t = TargetTransformer(method=method).fit(y)
    restored = pickle.loads(pickle.dumps(t))
    assert restored.method == t.method
    assert restored.clip_margin == t.clip_margin
    assert np.allclose(
        restored.inverse_transform(t.transform(y)),
        t.inverse_transform(t.transform(y)),
    )


def test_clone_keeps_params() -> None:
    t = TargetTransformer(method="yeo-johnson", clip_margin=0.3)
    cloned = clone(t)
    assert cloned.method == "yeo-johnson"
    assert cloned.clip_margin == 0.3
    assert not hasattr(cloned, "t_min_")


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown target transform method"):
        TargetTransformer(method="sqrt").fit(_positive_target())
    with pytest.raises(ValueError, match="Unknown target transform"):
        validate_target_for_transform(_positive_target(), "sqrt")


# ──────────────────────────────────────────────────────────────────────────────
#  Domain validation: boundary values
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("y_min", [-5.0, -1.0, -1.001])
def test_log1p_rejects_min_le_minus_one(y_min: float) -> None:
    # min(y) == -1 is INCLUSIVE violation (log1p(-1) = -inf).
    with pytest.raises(ValueError, match="log1p"):
        validate_target_for_transform(np.array([1.0, y_min]), "log1p")


def test_log1p_accepts_just_above_minus_one() -> None:
    validate_target_for_transform(np.array([-0.999, 1.0]), "log1p")


@pytest.mark.parametrize("y_min", [0.0, -0.001, -10.0])
def test_boxcox_rejects_min_le_zero(y_min: float) -> None:
    # min(y) == 0 is INCLUSIVE violation (Box-Cox requires strictly positive).
    with pytest.raises(ValueError, match="boxcox"):
        validate_target_for_transform(np.array([1.0, y_min]), "boxcox")


def test_boxcox_accepts_tiny_positive() -> None:
    validate_target_for_transform(np.array([1e-9, 1.0]), "boxcox")


def test_domain_error_message_content() -> None:
    y = np.array([2.0, -0.5])
    with pytest.raises(ValueError) as exc:
        validate_target_for_transform(y, "boxcox")
    message = str(exc.value)
    assert "boxcox" in message
    assert "-0.5" in message
    assert "yeo-johnson" in message


def test_no_domain_restriction_for_none_and_yeo_johnson() -> None:
    negative = np.array([-100.0, -1.0, 0.0, 50.0])
    validate_target_for_transform(negative, "none")
    validate_target_for_transform(negative, "yeo-johnson")
    validate_target_for_transform(negative, TargetTransform.yeo_johnson)


# ──────────────────────────────────────────────────────────────────────────────
#  wrap_with_target_transform
# ──────────────────────────────────────────────────────────────────────────────


def test_wrap_none_returns_same_object() -> None:
    model = LinearRegression()
    assert wrap_with_target_transform(model, "none") is model
    assert wrap_with_target_transform(model, TargetTransform.none) is model


def test_wrap_creates_ttr_with_check_inverse_false() -> None:
    model = LinearRegression()
    wrapped = wrap_with_target_transform(model, "log1p")
    assert isinstance(wrapped, TransformedTargetRegressor)
    assert wrapped.check_inverse is False
    assert isinstance(wrapped.transformer, TargetTransformer)
    assert wrapped.transformer.method == "log1p"


def test_wrapped_model_predicts_in_original_scale() -> None:
    rng = np.random.RandomState(3)
    X = rng.randn(80, 3)
    y = np.exp(0.5 + 0.8 * X[:, 0] + 0.05 * rng.randn(80))
    wrapped = wrap_with_target_transform(LinearRegression(), "log1p")
    wrapped.fit(X, y)
    preds = wrapped.predict(X)
    # Predictions in the original scale (same magnitude as y), not log
    # scale: with a good linear fit the mean tracks the true mean of y.
    assert np.all(np.isfinite(preds))
    assert preds.mean() == pytest.approx(y.mean(), rel=0.5)
