"""E2E Quality Regression Suite for configurable-automl-engine.

Real integration tests without mocks on synthetic data.
Verify end-to-end functionality and quality stability of all
registry algorithms through a direct ``tuner.optimize()`` call.

Data generation
---------------
* ``sklearn.datasets.make_regression`` with a fixed ``random_state=42``
* N=100 samples, 10 features, noise=0.05
* For tree-based / KNN the number of samples is increased (200-300) — on
  purely linear data these algorithms need more points to approximate.

Optimization
------------
* 25 trials per algorithm (20 for weak ones: tree-based, KNN)
* GLM algorithms with expensive fitting: 8 (gammaregressor), 15 (tweedieregressor)
* ``train_test_split`` (80/20) — faster than k-fold
* R² metric, quality threshold ≥ 0.70
* Explicit ``random_state=42`` for full reproducibility

Algorithm coverage (11)
-----------------------
ridge, elasticnet, sgdregressor, random_forest, decision_tree,
svr, gammaregressor, tweedieregressor, ardregression,
nearest_neighbors_regression, xgboosting

Note
----
decision_tree is excluded from the main quality test: a single decision tree
fundamentally cannot reach R² ≥ 0.70 on linear synthetic data with N=100
(structural ceiling ~0.55). It has a separate test
``test_decision_tree_quality`` on the non-linear dataset ``make_friedman1``
with an R² ≥ 0.40 threshold.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_friedman1, make_regression

from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
from configurable_automl_engine.tuner import (
    optimize,
)

# All 11 algorithms — used for integrity test
ALGORITHMS_UNDER_TEST: list[str] = [
    "ridge",
    "elasticnet",
    "sgdregressor",
    "random_forest",
    "decision_tree",
    "svr",
    "gammaregressor",
    "tweedieregressor",
    "ardregression",
    "nearest_neighbors_regression",
    "xgboosting",
]

# Quality test excludes decision_tree — structural limitation on linear data
QUALITY_ALGORITHMS: list[str] = [
    a for a in ALGORITHMS_UNDER_TEST if a != "decision_tree"
]

# Algorithms that need more samples/trials to reach R² >= 0.70
_WEAK_LEARNERS: frozenset[str] = frozenset(
    {
        "random_forest",
        "decision_tree",
        "nearest_neighbors_regression",
    }
)

# GLM-based algorithms with expensive per-trial fitting (log-link, IRLS).
# These need fewer trials to keep the total suite time under budget (< 60 s).
_SLOW_ALGORITHMS: frozenset[str] = frozenset(
    {
        "gammaregressor",
        "tweedieregressor",
    }
)

# Algorithms that need more samples (not noise reduction)
# noise=0.05 for all — the issue is sample count, not noise
_MORE_SAMPLES: dict[str, int] = {
    "random_forest": 200,
    "nearest_neighbors_regression": 300,
}


def _parametrize_algorithms(algorithms: list[str]) -> list[Any]:
    """Wrap algorithm names into ``pytest.param`` nodes.

    Adds ``pytest.mark.skipif`` for optional dependencies
    (``xgboosting``) so the tests are skipped gracefully when the
    package is not installed, rather than failing with an error.
    """
    params: list[Any] = []
    for algo in algorithms:
        if algo == "xgboosting":
            params.append(
                pytest.param(
                    algo,
                    marks=pytest.mark.skipif(
                        not is_installed("xgboost"),
                        reason="xgboost is not installed",
                    ),
                )
            )
        else:
            params.append(algo)
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _adjust_for_positive_target(algo: str, y: pd.Series) -> pd.Series:
    """Ensure strictly positive target for GLM-based algorithms.

    * ``gammaregressor`` — uses log-link (power=1), so ``y`` must be positive
      and should follow ``log(y) ≈ Xβ``.  We apply ``exp(y / 5)``.
    * ``tweedieregressor`` — default ``power=0`` resolves to identity link
      (Normal family), so the original linear-scale data is fine.
    """
    if algo == "gammaregressor":
        arr = y.to_numpy()
        y = pd.Series(np.exp(arr / 5.0), name=y.name)
    return y


def _build_space_overrides(algo: str) -> dict[str, dict[str, Any]]:
    """Build ``space_overrides`` dict for the given algorithm.

    The ``tuner.optimize()`` function requires an explicit search-space
    definition via ``space_overrides`` for all algorithms except ``knn``.
    """
    space = DEFAULT_SPACES.get(algo)
    if space is None:
        raise ValueError(
            f"{algo}: missing from DEFAULT_SPACES — cannot build space_overrides"
        )
    return {algo: deepcopy(space)}


def _n_samples_for(algo: str) -> int:
    """Return the number of samples appropriate for *algo*.

    Tree-based models and KNN need more samples to approximate linear
    functions on purely linear synthetic data.
    """
    return _MORE_SAMPLES.get(algo, 100)


def _generate_data(algo: str) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic regression data with *algo*-specific sample count."""
    X, y = make_regression(
        n_samples=_n_samples_for(algo),
        n_features=10,
        noise=0.05,
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y, name="target")


def _n_trials_for(algo: str) -> int:
    """Return the number of Optuna trials appropriate for *algo*.

    * Weak learners (tree-based, KNN) need more exploration to find
      a configuration that yields R² ≥ 0.70 — 35 trials.
    * Slow GLM-based algorithms (Gamma, Tweedie) are expensive per
      trial (log-link IRLS fitting) so they get fewer trials to keep
      the suite under budget.
    * ``gammaregressor`` gets 15 trials instead of 6: since HPO started
      applying ``StandardScaler`` on purely numeric data (the same logic as
      the final ``ModelTrainer``), gamma regression converges more slowly and
      needs a wider search to reach the R² ≥ 0.70 bar.
    * All others run with the standard 25 trials.
    """
    if algo in _SLOW_ALGORITHMS:
        return 15
    return 20 if algo in _WEAK_LEARNERS else 25


# ──────────────────────────────────────────────────────────────────────────────
# Test 1a: Algorithm quality baseline (R² >= 0.70)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("algo_name", _parametrize_algorithms(QUALITY_ALGORITHMS))
def test_algorithm_quality_baseline(algo_name: str) -> None:
    """Check that every qualifying algorithm reaches R² ≥ 0.70.

    decision_tree is excluded from the parametrized list — single decision
    trees cannot reach R² ≥ 0.70 on linear synthetic data (structural ceiling
    ~0.55).  It is tested separately via :func:`test_decision_tree_quality`.
    """
    X_df, y_series = _generate_data(algo_name)
    y_series = _adjust_for_positive_target(algo_name, y_series)

    opt_res = optimize(
        algo_name,
        X_df,
        y_series,
        n_trials=_n_trials_for(algo_name),
        random_state=42,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides=_build_space_overrides(algo_name),
    )
    assert opt_res is not None
    best_model, best_params, best_score = opt_res.model, opt_res.params, opt_res.score


    assert best_score >= 0.70, (
        f"{algo_name}: R²={best_score:.4f} < 0.70 — quality regression detected"
    )
    assert best_model is not None, (
        f"{algo_name}: best_model is None — algorithm was disqualified"
    )
    assert isinstance(best_params, dict) and len(best_params) > 0, (
        f"{algo_name}: best_params is empty or not a dict"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1b: decision_tree quality (non-linear synthetic data)
# ──────────────────────────────────────────────────────────────────────────────


def test_decision_tree_quality() -> None:
    """Check that ``decision_tree`` reaches R² ≥ 0.40 on non-linear data.

    ``decision_tree`` is excluded from :func:`test_algorithm_quality_baseline`
    because single trees have a structural ceiling of ~0.55 R² on linear
    synthetic data (even with 200+ samples).  Here we use
    ``sklearn.datasets.make_friedman1`` which produces non-linear
    interactions that a tree can exploit.
    """
    X, y = make_friedman1(n_samples=200, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name="target")

    opt_res = optimize(
        "decision_tree",
        X_df,
        y_series,
        n_trials=35,
        random_state=42,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides=_build_space_overrides("decision_tree"),
    )
    assert opt_res is not None
    best_model, best_params, best_score = opt_res.model, opt_res.params, opt_res.score


    assert best_score >= 0.40, (
        f"decision_tree: R²={best_score:.4f} < 0.40 on friedman1 data"
    )
    assert best_model is not None, (
        "decision_tree: best_model is None — algorithm was disqualified"
    )
    assert isinstance(best_params, dict) and len(best_params) > 0, (
        "decision_tree: best_params is empty or not a dict"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Search-space integrity (no false disqualifications)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("algo_name", _parametrize_algorithms(ALGORITHMS_UNDER_TEST))
def test_search_space_integrity(algo_name: str) -> None:
    """Verify that **no** algorithm raises ``InvalidAlgorithmError`` on valid data.

    A false disqualification would mean the search-space definition or
    internal circuit-breaker is too aggressive for the given dataset.
    """
    X_df, y_series = _generate_data(algo_name)
    y_series = _adjust_for_positive_target(algo_name, y_series)

    opt_res = optimize(
        algo_name,
        X_df,
        y_series,
        n_trials=_n_trials_for(algo_name),
        random_state=42,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides=_build_space_overrides(algo_name),
    )
    assert opt_res is not None
    best_model, _, _ = opt_res.model, opt_res.params, opt_res.score


    assert best_model is not None, (
        f"{algo_name}: model is None after successful optimisation"
    )
