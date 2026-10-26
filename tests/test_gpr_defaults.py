"""Regression tests for the GPR default parameters.

Covers:
- the default kernel (amplitude × RBF + white noise) set by
  ``models.create_model``;
- target normalization (``normalize_y=True``) and several optimizer
  restarts with a fixed seed;
- the non-empty default search space (kernel family: RBF / Matern /
  RationalQuadratic) and its ``kernel_type`` translation in
  ``models.create_model``;
- a direct ``tuner.optimize("gaussian_process_regression", ...)`` call without an explicit
  ``space_overrides`` (was failing with "No search space" — the space is
  now taken from ``DEFAULT_SPACES``);
- the GPR model actually beating the constant baseline on a small sample
  (the motivating problem: bare RBF ≈ std(y)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
from configurable_automl_engine.models import create_model
from configurable_automl_engine.tuner import (
    HyperoptError,
    optimize,
)


def _small_frame() -> tuple[pd.DataFrame, pd.Series]:
    """A small noisy regression sample (the motivating use case)."""
    X, y = make_regression(
        n_samples=25,
        n_features=5,
        noise=0.2,
        random_state=7,
    )
    return pd.DataFrame(X), pd.Series(y)


# ──────────────────────────────────────────────────────────────────────────
# Default search space
# ──────────────────────────────────────────────────────────────────────────
def test_gpr_default_search_space_not_empty():
    """DEFAULT_SPACES['gaussian_process_regression'] is no longer empty."""
    space = DEFAULT_SPACES["gaussian_process_regression"]
    assert "kernel_type" in space
    options = space["kernel_type"].config.options
    assert set(options) == {"rbf", "matern", "rational_quadratic"}


def test_gpr_direct_optimize_uses_default_space():
    """Direct optimize('gaussian_process_regression', X, y) no longer raises "No search space".

    The default space (kernel family) is applied even without an explicit
    ``space_overrides``.
    """
    X, y = _small_frame()
    result = optimize(
        "gaussian_process_regression",
        X,
        y,
        n_trials=2,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
    )
    assert result is not None
    assert result.params is not None
    assert "kernel_type" in result.params
    assert result.params["kernel_type"] in {
        "rbf",
        "matern",
        "rational_quadratic",
    }


def test_gpr_optimize_beats_constant_baseline():
    """On the small sample GPR with the default space beats the constant.

    The bare-RBF default produced a predictor ≈ std(y), i.e. worse than a
    constant. The improved defaults (amplitude × kernel + noise, normalized
    target, restarts) must reach R² > 0.
    """
    X, y = _small_frame()
    baseline = float(np.var(y))
    result = optimize(
        "gaussian_process_regression",
        X,
        y,
        n_trials=3,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
    )
    assert result is not None
    # R² below/at 0 would be the "mean predictor" regression.
    assert result.score > 0.0


def test_gpr_final_params_reconstruct_model():
    """best_params (kernel_type) reconstructs a model via create_model.

    The final training path passes best_params to ModelTrainer, which
    forwards them to create_model — the kernel_type hint must translate
    to a valid kernel there.
    """
    X, y = _small_frame()
    result = optimize(
        "gaussian_process_regression",
        X,
        y,
        n_trials=2,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
    )
    assert result is not None
    model = create_model("gaussian_process_regression", **result.params)
    assert hasattr(model, "kernel")
    assert model.normalize_y is True
    assert model.n_restarts_optimizer >= 1
    assert model.random_state is not None


def test_gpr_unknown_algo_still_rejected():
    """A wrong algorithm name still raises HyperoptError."""
    X, y = _small_frame()
    try:
        optimize("no_such_gpr", X, y, n_trials=1)
    except HyperoptError:
        pass
    else:
        raise AssertionError("Expected HyperoptError for unknown algorithm")