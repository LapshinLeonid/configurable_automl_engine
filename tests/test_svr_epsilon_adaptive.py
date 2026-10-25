"""Tests of the adaptive ``epsilon`` range for SVR.

Coverage:
- Unit tests of the ``compute_svr_epsilon_bounds`` helper (bounds from σ(y),
  fallback for a constant/NaN target, clamping for float_log, IQR).
- Tuner integration tests: epsilon is proposed by Optuna strictly within the
  adaptive bounds; a user-specified ``epsilon`` takes priority.
- e2e through ``train_best_model``: no trial leaves the adaptive range, and
  ``best_params["epsilon"]`` stays inside it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import Ridge

import configurable_automl_engine.tuner as tuner_module
from configurable_automl_engine.common.hyperopt_defaults import (
    SVR_EPSILON_FALLBACK,
    SearchSpaceEntry,
    compute_svr_epsilon_bounds,
)
from configurable_automl_engine.training_engine.component import (
    PhaseResult,
    PreparedData,
    _prepare_search_space,
    execute_phases,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    GeneralCfg,
    HPOPhaseCfg,
    ValidationStrategy,
)
from configurable_automl_engine.tuner import (
    InvalidDataError,
    _apply_dynamic_space,
    _make_svr_space_dict,
    optimize,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Helper: bounds from the spread of y
# ──────────────────────────────────────────────────────────────────────────────
def test_helper_known_sigma_returns_relative_bounds():
    """For y with a known σ returns (0.005·σ, 0.10·σ)."""
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sigma = float(np.std(y))  # σ ≈ 1.41421356 (population std, ddof=0)
    low, high = compute_svr_epsilon_bounds(y)

    assert low == pytest.approx(0.005 * sigma)
    assert high == pytest.approx(0.10 * sigma)
    assert 0.0 < low < high


def test_helper_accepts_series_and_dataframe():
    """The helper accepts pd.Series and pd.DataFrame (single column)."""
    y_series = pd.Series(np.arange(20, dtype=float))
    y_frame = pd.DataFrame({"y": np.arange(20, dtype=float)})
    y_list = list(range(20))

    expected = compute_svr_epsilon_bounds(y_series)
    assert compute_svr_epsilon_bounds(y_frame) == expected
    assert compute_svr_epsilon_bounds(y_list) == expected


def test_helper_custom_rel_factors():
    """The rel_min/rel_max parameters are overridden."""
    y = np.array([0.0, 2.0, 4.0])
    low, high = compute_svr_epsilon_bounds(y, rel_min=0.01, rel_max=0.2)
    sigma = float(np.std(y))
    assert low == pytest.approx(0.01 * sigma)
    assert high == pytest.approx(0.2 * sigma)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([1.0, 1.0, 1.0, 1.0]),  # constant y → σ = 0
        np.array([1.0, np.nan, 3.0]),  # NaN
        np.array([np.nan] * 4),  # only NaN
        np.array([1.0, np.inf]),  # +inf
        np.array([-np.inf, 2.0]),  # -inf
        np.array([]),  # empty vector
        np.array([0.0, 0.0]),  # zero spread
    ],
)
def test_helper_fallback_for_constant_or_non_finite_y(bad_y):
    """Constant/NaN/±inf/empty y → fallback (1e-4, 1e-2) without exceptions."""
    assert compute_svr_epsilon_bounds(bad_y) == SVR_EPSILON_FALLBACK


def test_helper_very_small_sigma_clamps_low_positive():
    """Very small σ → low is clamped > 0 (float_log invariant), low < high."""
    y = np.array([1e-12, 1.1e-12, 1.2e-12, 1.3e-12])
    low, high = compute_svr_epsilon_bounds(y)

    # Without clamping, low would be ~5.6e-16; we guarantee a strictly positive minimum.
    assert low > 0.0
    assert low == pytest.approx(1e-8)  # clamped to SVR_EPSILON_MIN_LOW
    assert low < high


def test_helper_iqr_strategy_robust_to_outliers():
    """The IQR variant is robust to outliers: the spread is smaller than with std."""
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000.0], dtype=float)

    std_low, std_high = compute_svr_epsilon_bounds(y, strategy="std")
    iqr_low, iqr_high = compute_svr_epsilon_bounds(y, strategy="iqr")

    expected_spread = (np.percentile(y, 75) - np.percentile(y, 25)) / 1.349
    assert iqr_low == pytest.approx(0.005 * expected_spread)
    assert iqr_high == pytest.approx(0.10 * expected_spread)
    # An outlier inflates σ but not IQR → the robust range is narrower.
    assert iqr_high < std_high


def test_helper_invalid_strategy_raises():
    """Unknown spread strategy → ValueError."""
    with pytest.raises(ValueError, match="Unknown epsilon spread strategy"):
        compute_svr_epsilon_bounds([1, 2, 3], strategy="mad")


def test_helper_non_positive_min_low_raises():
    """min_low <= 0 → ValueError (the log scale requires low > 0)."""
    with pytest.raises(ValueError, match="min_low must be > 0"):
        compute_svr_epsilon_bounds([1, 2, 3], min_low=0.0)
    with pytest.raises(ValueError, match="min_low must be > 0"):
        compute_svr_epsilon_bounds([1, 2, 3], min_low=-1e-3)


def test_helper_target_scenario_upper_bound_near_025():
    """Regression of the target scenario: with σ ≈ 0.244 the upper bound ≈ 0.025.

    The 0.06–0.08 zone (degeneration into a flat line) is physically unreachable.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.244, size=500)
    low, high = compute_svr_epsilon_bounds(y)

    assert high == pytest.approx(0.10 * float(np.std(y)))
    assert high < 0.05
    assert 0.02 <= high <= 0.03


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tuner: SVR space factory and integration
# ──────────────────────────────────────────────────────────────────────────────
def test_make_svr_space_uses_adaptive_epsilon_bounds():
    """The factory proposes epsilon strictly within the bounds derived from y."""
    y = np.arange(50, dtype=float)
    low, high = compute_svr_epsilon_bounds(y)

    space_dict = _make_svr_space_dict(y)
    trial = MagicMock()
    _apply_dynamic_space(trial, space_dict)

    trial.suggest_float.assert_any_call("epsilon", low, high, log=True)


def _mock_optimize_infra(
    monkeypatch,
    collect_epsilons: list[float],
    model_factory=None,
):
    """Mock the optimization infrastructure and collect epsilon from trials."""

    def default_factory(algo: str, **params):
        return Ridge()

    factory = model_factory or default_factory

    def mock_create_model(algo: str, **params):
        if "epsilon" in params:
            collect_epsilons.append(float(params["epsilon"]))
        return factory(algo, **params)

    monkeypatch.setattr(
        "configurable_automl_engine.tuner.create_model", mock_create_model
    )
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.get_scorer_object",
        lambda name, global_y=None: lambda model, X, y: 0.5,
    )
    monkeypatch.setattr(
        "configurable_automl_engine.tuner._evaluate_pooled_cv",
        lambda *a, **kw: 0.5,
    )
    # The additive-metric path (MAE and similar) still uses cross_val_score;
    # it is patched too so a metric change cannot break the helpers.
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.model_selection.cross_val_score",
        lambda *a, **kw: [0.5, 0.5],
    )


def test_tuner_svr_default_space_all_trials_within_adaptive_bounds(monkeypatch):
    """Tuner (svr, default space): all trial epsilon values lie in [eps_min, eps_max]."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    low, high = compute_svr_epsilon_bounds(y)

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    opt_res = optimize("svr", X, y, n_trials=6, random_state=42)
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert len(seen) >= 6
    for eps in seen:
        assert low <= eps <= high
    assert low <= params["epsilon"] <= high


def test_tuner_svr_dict_override_without_epsilon_gets_adaptive(monkeypatch):
    """space_overrides[svr] — a dict without epsilon → adaptive bounds are added."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    low, high = compute_svr_epsilon_bounds(y)

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    space = {"C": SearchSpaceEntry.model_validate([1e-2, 100.0, "float_log"])}
    opt_res = optimize(
        "svr",
        X,
        y,
        n_trials=5,
        random_state=42,
        space_overrides={"svr": space},
    )
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert seen
    for eps in seen:
        assert low <= eps <= high
    assert low <= params["epsilon"] <= high


def test_tuner_svr_user_epsilon_override_has_priority(monkeypatch):
    """Explicit user epsilon (dict override) → adaptive logic is not applied."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    # The adaptive range for such y is guaranteed to be below 0.1 (rel_max=0.10).
    _, adaptive_high = compute_svr_epsilon_bounds(y)
    assert adaptive_high < 0.5

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    user_eps = SearchSpaceEntry.model_validate([0.5, 1.0, "float_log"])
    opt_res = optimize(
        "svr",
        X,
        y,
        n_trials=5,
        random_state=42,
        space_overrides={"svr": {"epsilon": user_eps}},
    )
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert seen
    for eps in seen:
        assert 0.5 <= eps <= 1.0
        assert eps > adaptive_high  # outside the adaptive range — override applied
    assert 0.5 <= params["epsilon"] <= 1.0


def test_tuner_svr_callable_override_unchanged(monkeypatch):
    """space_overrides[svr] as a callable (old mechanism) → behavior is unchanged."""
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    def custom_space(trial):
        return {"epsilon": trial.suggest_float("epsilon", 0.2, 0.3)}

    opt_res = optimize(
        "svr",
        X,
        y,
        n_trials=4,
        random_state=42,
        space_overrides={"svr": custom_space},
    )
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert seen
    for eps in seen:
        assert 0.2 <= eps <= 0.3
    assert 0.2 <= params["epsilon"] <= 0.3


def test_tuner_svr_initial_params_epsilon_valid_in_adaptive_range(monkeypatch):
    """enqueue_trial: epsilon from the previous phase (refine) is valid in the adaptive range.

    The bounds are computed from the same y_train in both phases, so the winner
    value of the first phase is guaranteed to lie inside the range of the second.
    """
    X = pd.DataFrame(np.random.randn(80, 5))
    y = pd.Series(np.random.randn(80))
    low, high = compute_svr_epsilon_bounds(y)
    prev_epsilon = (low + high) / 2.0  # "winner" of the previous phase

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    opt_res = optimize(
        "svr",
        X,
        y,
        n_trials=3,
        random_state=42,
        initial_params={
            "C": 1.0,
            "epsilon": prev_epsilon,
            "kernel": "rbf",
            "gamma": "scale",
        },
    )
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert low <= prev_epsilon <= high
    assert seen[0] == prev_epsilon  # the first trial is the enqueued value
    assert low <= params["epsilon"] <= high
    # The legacy categorical gamma="scale" does not fit the numeric gamma
    # range: it is dropped from the enqueued trial instead of failing it.
    assert isinstance(params["gamma"], float)


def test_tuner_svr_constant_y_with_user_epsilon_works(monkeypatch):
    """σ = 0 + user epsilon → the user value is used, not the fallback."""
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series([1.0] * 60)  # constant y → σ = 0

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    user_eps = SearchSpaceEntry.model_validate([0.5, 1.0, "float_log"])
    opt_res = optimize(
        "svr",
        X,
        y,
        n_trials=4,
        random_state=42,
        space_overrides={"svr": {"epsilon": user_eps}},
    )
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert seen
    for eps in seen:
        assert 0.5 <= eps <= 1.0  # user range, not the fallback (1e-4, 1e-2)
    assert 0.5 <= params["epsilon"] <= 1.0


def test_tuner_svr_constant_y_falls_back(monkeypatch):
    """Constant y → fallback (1e-4, 1e-2), runs without exceptions."""
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series([2.5] * 60)

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    opt_res = optimize("svr", X, y, n_trials=4, random_state=42)
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert seen
    for eps in seen:
        assert SVR_EPSILON_FALLBACK[0] <= eps <= SVR_EPSILON_FALLBACK[1]
    assert SVR_EPSILON_FALLBACK[0] <= params["epsilon"] <= SVR_EPSILON_FALLBACK[1]


def test_tuner_svr_nan_y_falls_back(monkeypatch):
    """y with NaN is rejected by the tuner before the search starts.

    A non-finite target must fail fast (``InvalidDataError``) instead of
    spending the HPO budget on randomly "clean" folds. The fallback behavior
    of ``compute_svr_epsilon_bounds`` itself is covered separately by
    ``test_helper_fallback_for_constant_or_non_finite_y``.
    """
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series([1.0, np.nan] + list(np.random.randn(58)))

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    with pytest.raises(InvalidDataError, match="non-finite"):
        optimize("svr", X, y, n_trials=4, random_state=42)
    # No trial ran: the validation happens before the search.
    assert not seen


def test_tuner_other_algorithms_unaffected(monkeypatch):
    """Behavior of the other algorithms is unchanged (e.g., ridge)."""
    X = pd.DataFrame(np.random.randn(60, 5))
    y = pd.Series(np.random.randn(60))

    seen: list[float] = []
    _mock_optimize_infra(monkeypatch, seen)

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=3,
        random_state=42,
        space_overrides={"ridge": {"alpha": SearchSpaceEntry.model_validate([0.1, 1.0, "float_log"])}},
    )
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score


    assert 0.1 <= params["alpha"] <= 1.0
    assert "epsilon" not in params


# ──────────────────────────────────────────────────────────────────────────────
# 3. e2e through train_best_model
# ──────────────────────────────────────────────────────────────────────────────
def _svr_only_cfg(model_path: str, n_trials: int = 3) -> str:
    return f"""
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  phases:
    - name: "Coarse Search"
      n_trials: {n_trials}
      action: "all_algorithms"
    - name: "Final Select"
      n_trials: 1
      action: "select"
      strategy: "cv_best"
      output: "winner"
algorithms:
  svr:
    enable: true
"""


def test_e2e_svr_adaptive_epsilon_all_trials_and_best_within_bounds(
    tmp_path: Path, monkeypatch
):
    """e2e: no trial leaves the adaptive range; best_params stays inside."""
    rows = 60
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.standard_normal((rows, 4)), columns=[f"f{i}" for i in range(4)])
    # Target variable with a deterministic spread σ(y).
    df["target"] = np.arange(rows, dtype=float)
    sigma = float(np.std(df["target"].to_numpy()))
    low, high = 0.005 * sigma, 0.10 * sigma

    # Intercept every create_model call in the tuner (all trials + the final fit).
    real_create_model = tuner_module.create_model
    seen: list[float] = []

    def capturing_create_model(algo: str, **params):
        if algo == "svr" and "epsilon" in params:
            seen.append(float(params["epsilon"]))
        return real_create_model(algo, **params)

    monkeypatch.setattr(tuner_module, "create_model", capturing_create_model)

    model_path = tmp_path / "model.pkl"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(_svr_only_cfg(str(model_path)), encoding="utf-8")

    res = train_best_model(cfg_file, df, target="target", model_path_override=model_path)

    assert res["algorithm"] == "svr"
    assert seen, "no epsilon was proposed — the test is useless"
    for eps in seen:
        assert low <= eps <= high
    assert low <= res["params"]["epsilon"] <= high


def test_e2e_svr_user_epsilon_override_respected(tmp_path: Path, monkeypatch):
    """e2e: user epsilon in YAML → the adaptive range is not applied."""
    rows = 60
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.standard_normal((rows, 4)), columns=[f"f{i}" for i in range(4)])
    # Scale the target so that the adaptive range is guaranteed to be below [0.5, 1.0].
    df["target"] = np.arange(rows, dtype=float) * 0.02
    sigma = float(np.std(df["target"].to_numpy()))
    adaptive_high = 0.10 * sigma
    assert adaptive_high < 0.5  # the user range [0.5, 1.0] is outside the adaptive one

    model_path = tmp_path / "model.pkl"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_text = f"""
general:
  comparison_metric: rmse
  path_to_model: '{model_path}'
  phases:
    - name: "Coarse Search"
      n_trials: 3
      action: "all_algorithms"
    - name: "Final Select"
      n_trials: 1
      action: "select"
      strategy: "cv_best"
      output: "winner"
algorithms:
  svr:
    enable: true
    hyperparameters:
      epsilon: [0.5, 1.0]
"""
    cfg_file.write_text(cfg_text, encoding="utf-8")

    real_create_model = tuner_module.create_model
    seen: list[float] = []

    def capturing_create_model(algo: str, **params):
        if algo == "svr" and "epsilon" in params:
            seen.append(float(params["epsilon"]))
        return real_create_model(algo, **params)

    monkeypatch.setattr(tuner_module, "create_model", capturing_create_model)

    res = train_best_model(cfg_file, df, target="target", model_path_override=model_path)

    assert res["algorithm"] == "svr"
    assert seen
    for eps in seen:
        assert 0.5 <= eps <= 1.0
        assert eps > adaptive_high  # outside the adaptive range — override applied
    assert 0.5 <= res["params"]["epsilon"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Component: prepare_search_space (user-priority contract)
# ──────────────────────────────────────────────────────────────────────────────
def _svr_config(tmp_path: Path, hyperparameters: dict | None = None) -> Config:
    """Config with a single enabled svr algorithm."""
    algos: dict[str, dict] = {"svr": {"enable": True}}
    if hyperparameters:
        algos["svr"]["hyperparameters"] = hyperparameters
    return Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            path_to_model=str(tmp_path / "m.pkl"),
            phases=[
                HPOPhaseCfg(name="Coarse Search", n_trials=1, action="all_algorithms"),
                HPOPhaseCfg(
                    name="Final Coarse Select",
                    n_trials=1,
                    action="select",
                    strategy="cv_best",
                    output="winner",
                ),
            ],
        ),
        algorithms=algos,
    )


def _prepared_data() -> PreparedData:
    return PreparedData(
        X=pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]}),
        y=pd.Series([1.0, 2.0, 3.0, 4.0]),
        metric_user="rmse",
        metric_sklearn="neg_root_mean_squared_error",
        resolved_validation=ValidationStrategy.k_fold,
        resolved_n_folds=2,
        resolved_test_size=0.2,
        categorical_features=[],
        numerical_features=["f1", "f2"],
    )


def test_prepare_search_space_svr_default_omits_epsilon(
    tmp_path: Path, monkeypatch
):
    """prepare_search_space: svr without a user epsilon → epsilon is omitted (adaptation in the tuner).

    The default blind range DEFAULT_SPACES["svr"]["epsilon"]=[1e-3, 1.0] must
    not end up in the final space: otherwise the tuner cannot tell it apart
    from an explicit user value.
    """
    cfg = _svr_config(tmp_path)
    prepared = _prepared_data()

    captured: dict[str, object] = {}

    def fake_run_hpo(**kwargs):
        captured["search_space"] = kwargs.get("search_space_override")
        return PhaseResult(score=0.9, params={"C": 1.0})

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.component._run_hpo",
        fake_run_hpo,
    )

    ctx = execute_phases(cfg, prepared)
    results = ctx.working_set

    assert "svr" in results
    space = captured["search_space"]
    assert "epsilon" not in space, (
        "default epsilon must not end up in the space — "
        "its place is taken by adaptive bounds in the tuner"
    )
    # The remaining default SVR parameters are kept.
    assert "C" in space
    assert "kernel" in space
    assert "gamma" in space


def test_prepare_search_space_svr_user_epsilon_kept(tmp_path: Path, monkeypatch):
    """prepare_search_space: an explicit user epsilon is kept (priority)."""
    cfg = _svr_config(tmp_path, hyperparameters={"epsilon": [0.5, 1.0]})
    prepared = _prepared_data()

    captured: dict[str, object] = {}

    def fake_run_hpo(**kwargs):
        captured["search_space"] = kwargs.get("search_space_override")
        return PhaseResult(score=0.9, params={"C": 1.0})

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.component._run_hpo",
        fake_run_hpo,
    )

    ctx = execute_phases(cfg, prepared)
    results = ctx.working_set

    assert "svr" in results
    space = captured["search_space"]
    assert "epsilon" in space
    assert space["epsilon"].low == 0.5
    assert space["epsilon"].high == 1.0


def test_prepare_search_space_other_algorithms_unaffected(tmp_path: Path, monkeypatch):
    """prepare_search_space: behavior for the other algorithms is unchanged."""
    cfg_ridge = Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            path_to_model=str(tmp_path / "m.pkl"),
            phases=[
                HPOPhaseCfg(name="Coarse Search", n_trials=1, action="all_algorithms"),
                HPOPhaseCfg(
                    name="Final Coarse Select",
                    n_trials=1,
                    action="select",
                    strategy="cv_best",
                    output="winner",
                ),
            ],
        ),
        algorithms={"ridge": {"enable": True}},
    )

    captured: dict[str, object] = {}

    def fake_run_hpo(**kwargs):
        captured["search_space"] = kwargs.get("search_space_override")
        return PhaseResult(score=0.9, params={"alpha": 0.1})

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.component._run_hpo",
        fake_run_hpo,
    )

    ctx = execute_phases(cfg_ridge, _prepared_data())
    results = ctx.working_set

    assert "ridge" in results
    space = captured["search_space"]
    # The default ridge alpha is kept as before (nothing is omitted).
    assert "alpha" in space
    assert space["alpha"].dist_type == "float_log"


# ──────────────────────────────────────────────────────────────────────────────
# Legacy-name remap in _prepare_search_space (n_iter → max_iter for ARD)
# ──────────────────────────────────────────────────────────────────────────────


def test_prepare_search_space_remaps_ard_default_n_iter():
    """The default ARDRegression space exposes max_iter (not the legacy n_iter).

    The default space itself uses the modern name, so a per-trial remap ---
    and its repeated "Remapped ..." DEBUG log line --- is not needed.
    """
    space = _prepare_search_space("ardregression", None)
    assert "max_iter" in space
    assert "n_iter" not in space
    assert space["max_iter"].low == 100
    assert space["max_iter"].high == 1000


def test_prepare_search_space_remaps_ard_user_legacy_n_iter():
    """A user override with the legacy n_iter key is remapped once."""
    space = _prepare_search_space(
        "ardregression",
        {"n_iter": SearchSpaceEntry.model_validate([50, 500, "int", 50])},
    )
    assert "max_iter" in space
    assert "n_iter" not in space
    assert space["max_iter"].low == 50
    assert space["max_iter"].high == 500


def test_prepare_search_space_other_algos_not_remapped():
    """Algorithms without legacy mappings keep their keys untouched."""
    space = _prepare_search_space("elasticnet", {"alpha": [1e-4, 1.0]})
    assert "alpha" in space
    assert "max_iter" not in space
