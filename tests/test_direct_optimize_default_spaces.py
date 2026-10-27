"""A direct ``optimize()`` call searches the same space as a config run.

Covers:
- every algorithm of ``AVAILABLE_ALGORITHMS`` is tunable by a direct
  ``optimize(algo, X, y)`` call without ``space_overrides`` (the defaults
  come from ``DEFAULT_SPACES``);
- parity of the searched Optuna distributions (after clipping and the
  complexity guard) between a direct call and the engine path
  (``hpo_phases._run_hpo`` with the request built by
  ``_build_hpo_request``): no user ranges, a partial range merged over the
  defaults, and the same range with ``limit_hyperparameters=True``;
- parity of the complexity guard records on small data;
- idempotence of ``build_search_space``;
- the engine always forwards ``user_hyperparameter_keys`` (an empty list
  included);
- the error of an algorithm without a default space and without overrides.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.hyperopt_defaults import (
    DEFAULT_SPACES,
    CategoricalSpace,
    SearchSpaceEntry,
)
from configurable_automl_engine.hpo import runner as hpo_runner
from configurable_automl_engine.hpo.search_space import (
    _canonical_distributions,
    build_search_space,
)
from configurable_automl_engine.models import AVAILABLE_ALGORITHMS
from configurable_automl_engine.training_engine import hpo_phases
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    HPOPhaseCfg,
)
from configurable_automl_engine.tuner import HyperoptError, optimize
from tests.hpo_spec_factory import make_phase_spec


def _algo_params() -> list[Any]:
    """All algorithms; ``xgboosting`` is skipped without the package."""
    return [
        pytest.param(
            algo,
            marks=pytest.mark.skipif(
                not is_installed("xgboost"), reason="xgboost not installed"
            ),
        )
        if algo == "xgboosting"
        else algo
        for algo in AVAILABLE_ALGORITHMS
    ]


def _frame(n_rows: int = 40, n_features: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    """A small numeric sample with a strictly positive target (GLMs)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        rng.rand(n_rows, n_features), columns=[f"x{i}" for i in range(n_features)]
    )
    y = pd.Series(1.0 + 2.0 * X["x0"] + X["x1"] + 0.1 * rng.rand(n_rows))
    return X, y


def _partial_range(algo: str) -> dict[str, list[Any]]:
    """A narrowed range of one default parameter (explicitly typed)."""
    space = DEFAULT_SPACES[algo]
    for name, entry in space.items():
        if isinstance(entry.config, CategoricalSpace):
            continue
        low, high = entry.low, entry.high
        if entry.dist_type == "int":
            return {name: [int(low), max(int(low), (int(low) + int(high)) // 2), "int"]}
        if entry.dist_type == "float_log":
            return {name: [float(low), math.sqrt(low * high), "float_log"]}
        return {name: [float(low), (float(low) + float(high)) / 2, "float"]}
    # Only categorical parameters (isotonic, Gaussian process): one option.
    name, entry = next(iter(space.items()))
    return {name: [[entry.config.options[0]], "categorical"]}


class _SpaceSpy:
    """Record the resolved distributions and guard records of every run."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.records: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        original = hpo_runner.resolve_search_space

        def _spy(runner: Any) -> None:
            original(runner)
            self.records.append(
                (
                    _canonical_distributions(runner) or {},
                    list(getattr(runner, "complexity_guard_logs", []) or []),
                )
            )

        monkeypatch.setattr(hpo_runner, "resolve_search_space", _spy)


def _direct(
    algo: str,
    X: pd.DataFrame,
    y: pd.Series,
    overrides: dict[str, Any] | None,
    limit: bool,
) -> None:
    optimize(
        algo,
        X,
        y,
        n_trials=1,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=7,
        sampler_seed=7,
        space_overrides={algo: overrides} if overrides is not None else None,
        limit_hyperparameters=limit,
        numerical_features=list(X.columns),
        categorical_features=[],
        fit_best_model=False,
    )


def _engine(
    algo: str,
    X: pd.DataFrame,
    y: pd.Series,
    overrides: dict[str, Any] | None,
    limit: bool,
) -> None:
    """Mirror ``hpo_phases._hpo_phase_worker`` for one algorithm."""
    hyperparameters = (
        {k: SearchSpaceEntry.model_validate(v) for k, v in overrides.items()}
        if overrides is not None
        else None
    )
    a_cfg = AlgoCfg(hyperparameters=hyperparameters, limit_hyperparameters=limit)
    spec = make_phase_spec(
        phase=HPOPhaseCfg(name="coarse", n_trials=1, action="all_algorithms"),
        y=y,
        resolved_n_folds=3,
        categorical_features=[],
        numerical_features=list(X.columns),
        random_state=7,
        sampler_seed=7,
    )
    request = hpo_phases._build_hpo_request(algo, a_cfg, spec)
    hpo_phases._run_hpo(X=X, y=y, request=request)


# ──────────────────────────────────────────────────────────────────────────
# 1. Defaults for every algorithm
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("algo", _algo_params())
def test_direct_optimize_default_space_every_algorithm(algo: str) -> None:
    """``optimize(algo, X, y)`` without overrides runs for every algorithm."""
    X, y = _frame()
    result = optimize(
        algo,
        X,
        y,
        n_trials=2,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=0,
        fit_best_model=False,
    )
    assert result is not None
    assert result.score is not None and np.isfinite(result.score)
    completed = [t.value for t in result.trials if t.value is not None]
    assert completed and all(np.isfinite(v) for v in completed)


# ──────────────────────────────────────────────────────────────────────────
# 2. Parity of the searched distributions
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["defaults", "merged", "limited"])
@pytest.mark.parametrize("algo", _algo_params())
def test_direct_and_engine_search_the_same_space(
    algo: str, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct call vs engine: equal distributions and guard records."""
    X, y = _frame()
    overrides = None if case == "defaults" else _partial_range(algo)
    limit = case == "limited"
    spy = _SpaceSpy(monkeypatch)

    _direct(algo, X, y, overrides, limit)
    _engine(algo, X, y, overrides, limit)

    assert len(spy.records) == 2
    (direct_dists, direct_guard), (engine_dists, engine_guard) = spy.records
    assert direct_dists, "the direct call must search a non-empty space"
    assert direct_dists == engine_dists
    assert direct_guard == engine_guard

    if case == "merged":
        # The partial range is merged over the defaults.
        defaults = set(build_search_space(algo, None))
        assert defaults <= set(direct_dists)
    if case == "limited":
        # Only the listed parameter is searched (isotonic additionally gets
        # its data-driven feature index).
        expected = set(overrides or {})
        if algo == "isotonic_regression":
            expected.add("feature_index")
        assert set(direct_dists) == expected


def test_knn_default_n_neighbors_matches_config_and_is_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KNN without overrides: 1–50 (DEFAULT_SPACES), clipped to n_eff - 1."""
    spy = _SpaceSpy(monkeypatch)
    X, y = _frame(n_rows=120)
    _direct("nearest_neighbors_regression", X, y, None, False)
    dist = spy.records[-1][0]["n_neighbors"]
    assert (dist.low, dist.high) == (1, 50)

    X, y = _frame(n_rows=12)  # k_fold n=3: n_eff = 8 → high = 7
    _direct("nearest_neighbors_regression", X, y, None, False)
    dist = spy.records[-1][0]["n_neighbors"]
    assert (dist.low, dist.high) == (1, 7)


def test_svr_default_epsilon_is_adaptive(monkeypatch: pytest.MonkeyPatch) -> None:
    """SVR without overrides keeps the adaptive epsilon (not DEFAULT_SPACES)."""
    from configurable_automl_engine.common.hyperopt_defaults import (
        compute_svr_epsilon_bounds,
    )

    spy = _SpaceSpy(monkeypatch)
    X, y = _frame()
    _direct("svr", X, y, None, False)
    dist = spy.records[-1][0]["epsilon"]
    low, high = compute_svr_epsilon_bounds(y)
    assert dist.log
    assert dist.low == pytest.approx(low)
    assert dist.high == pytest.approx(high)


# ──────────────────────────────────────────────────────────────────────────
# 3. Complexity guard parity on small data
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "overrides", [None, {"max_depth": [2, 20, "int"]}], ids=["defaults", "user"]
)
def test_complexity_guard_parity_small_data(
    overrides: dict[str, Any] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard tightens the same ranges on both paths."""
    spy = _SpaceSpy(monkeypatch)
    X, y = _frame(n_rows=18)
    _direct("random_forest", X, y, overrides, False)
    _engine("random_forest", X, y, overrides, False)
    (direct_dists, direct_guard), (engine_dists, engine_guard) = spy.records
    assert direct_guard, "the guard must tighten ranges on small data"
    assert any(r.get("applied") for r in direct_guard)
    assert direct_guard == engine_guard
    assert direct_dists == engine_dists


def test_direct_call_user_keys_default_to_override_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``user_hyperparameter_keys`` the override keys are the user ranges."""
    seen: list[Any] = []
    original = hpo_runner.resolve_search_space

    def _spy(runner: Any) -> None:
        original(runner)
        seen.append(runner.user_hyperparameter_keys)

    monkeypatch.setattr(hpo_runner, "resolve_search_space", _spy)
    X, y = _frame()
    _direct("random_forest", X, y, {"max_depth": [2, 5, "int"]}, False)
    _direct("random_forest", X, y, None, False)
    assert seen == [["max_depth"], []]


# ──────────────────────────────────────────────────────────────────────────
# 4. The engine always forwards user_hyperparameter_keys
# ──────────────────────────────────────────────────────────────────────────
def test_engine_always_passes_user_hyperparameter_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty list is forwarded too (not dropped as "no keys")."""
    captured: dict[str, Any] = {}

    def fake_run(X: Any, y: Any, request: Any) -> None:
        captured["user_hyperparameter_keys"] = request.user_hyperparameter_keys
        return None

    monkeypatch.setattr(hpo_phases, "run_optimization", fake_run)
    X, y = _frame()
    spec = make_phase_spec(y=y, numerical_features=list(X.columns))
    hpo_phases._run_hpo(
        X=X, y=y, request=hpo_phases._build_hpo_request("ridge", AlgoCfg(), spec)
    )
    assert captured["user_hyperparameter_keys"] == []


# ──────────────────────────────────────────────────────────────────────────
# 5. Idempotence of the merge
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("algo", sorted(DEFAULT_SPACES))
@pytest.mark.parametrize("with_user", [False, True])
def test_build_search_space_is_idempotent(algo: str, with_user: bool) -> None:
    """Merging an already-merged space again changes nothing (keys and order)."""
    user = (
        {
            k: SearchSpaceEntry.model_validate(v)
            for k, v in _partial_range(algo).items()
        }
        if with_user
        else None
    )
    once = build_search_space(algo, user)
    twice = build_search_space(algo, once)
    assert list(twice) == list(once)
    assert twice == once


def test_build_search_space_svr_epsilon() -> None:
    """SVR: a missing epsilon stays missing (adaptive); a user one is kept."""
    once = build_search_space("svr", None)
    assert "epsilon" not in once
    assert "epsilon" not in build_search_space("svr", once)

    user = {"epsilon": SearchSpaceEntry.model_validate([0.01, 0.1, "float_log"])}
    once = build_search_space("svr", user)
    assert once["epsilon"] == user["epsilon"]
    assert build_search_space("svr", once)["epsilon"] == user["epsilon"]


def test_build_search_space_limit_and_unknown() -> None:
    """Limited mode keeps only the listed keys; no defaults → empty."""
    user = {"alpha": [0.1, 1.0, "float_log"]}
    assert build_search_space("ridge", user, limit=True) == user
    assert build_search_space("ridge", None, limit=True) == {}
    assert build_search_space("fake_algo", None) == {}


# ──────────────────────────────────────────────────────────────────────────
# 6. No default space and no override
# ──────────────────────────────────────────────────────────────────────────
def test_unknown_algorithm_without_space_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error starts as before and tells how to fix it."""
    monkeypatch.setattr(hpo_runner, "_get_estimator", lambda algo: True)
    X, y = _frame()
    with pytest.raises(HyperoptError) as err:
        optimize("fake_algo", X, y, n_trials=1)
    message = str(err.value)
    assert message.startswith("No search space for algorithm 'fake_algo'")
    assert "space_overrides" in message
