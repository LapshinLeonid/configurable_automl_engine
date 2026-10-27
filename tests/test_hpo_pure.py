"""Direct unit tests of the pure HPO functions (plans, trial scoring, helpers).

Every function is called with plain values — no ``_OptimizeRunner`` is
constructed or mocked. ``assert_pure`` checks that the inputs are unchanged
after the call (data frames are compared with a copy taken before the call)
and that two calls give equal results.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import FixedTrial, FrozenTrial, TrialState
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from configurable_automl_engine.common.complexity_guard import ComplexityGuardConfig
from configurable_automl_engine.common.hyperopt_defaults import SearchSpaceEntry
from configurable_automl_engine.hpo.evaluation import (
    _build_scorer,
    _phase_oof_matrix,
    score_trial,
)
from configurable_automl_engine.hpo.plans import (
    PreprocessingPlan,
    PruningPlan,
    SearchSpacePlan,
    ValidationPlan,
)
from configurable_automl_engine.hpo.pruning import plan_pruning
from configurable_automl_engine.common.settings import (
    EncodingSettings,
    InputPolicies,
    OversamplingSettings,
)
from configurable_automl_engine.hpo.request import (
    HpoRequest,
    as_complexity_guard_config,
)
from configurable_automl_engine.hpo.runner import (
    _normalize_feature_selection_cfg,
    plan_validation,
)
from configurable_automl_engine.hpo.search_space import (
    _canonical_distributions,
    _enqueueable_initial_params,
    _is_guard_boundary,
    _natural_domain_bounds,
    _normalize_space_dict,
    _warn_if_optimum_at_boundary,
    build_search_space,
    log_complexity_guard,
    plan_search_space,
)
from configurable_automl_engine.hpo.trial_pipeline import (
    _estimate_n_features_after_preprocessing,
    _fs_transformer_factory,
    _isotonic_max_feature_index,
    _missing_indicator_added,
    _resolve_apply_fs,
    _resolve_isotonic_feature_name,
    assemble_estimator,
    plan_preprocessing,
)
from configurable_automl_engine.hpo.types import HPO_WORST_SCORE, HyperoptError
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from tests.purity import assert_pure


# ──────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _frame(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric + categorical columns, an uninformative constant column."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "city": rng.choice(["a", "b", "c"], size=n),
            "const": np.ones(n),
        }
    )
    y = pd.Series(2.0 * X["x1"] - X["x2"] + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def _numeric_frame(n: int = 30, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(3.0 * X["a"] + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def _validation(X: pd.DataFrame, y: pd.Series, **kwargs) -> ValidationPlan:
    return plan_validation(X, y, HpoRequest(algo_name="ridge", **kwargs))


# ──────────────────────────────────────────────────────────────────────────────
#  plan_validation
# ──────────────────────────────────────────────────────────────────────────────


def test_plan_validation_repairs_without_mutating_input() -> None:
    """Datetime columns are expanded in the plan; the input frame is untouched."""
    X, y = _numeric_frame()
    X["when"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(np.arange(len(X)), "D")
    request = HpoRequest(algo_name="Ridge", val_method="k_fold", n_folds=3)

    plan = assert_pure(plan_validation, X, y, request)

    assert "when" in X.columns
    assert "when" not in plan.X.columns
    assert any(col.startswith("when_") for col in plan.X.columns)
    assert plan.algo == "ridge"
    assert plan.method == "k_fold"
    assert plan.n_folds == 3
    assert plan.fs_mode == FeatureSelectionMode.disabled
    assert plan.is_isotonic is False
    assert plan.n_samples_eff == 20


def test_plan_validation_auto_hold_out_uses_the_decision_size() -> None:
    """'auto' → train_test_split: the integer test size of the decision."""
    X, y = _numeric_frame(n=30)
    plan = assert_pure(plan_validation, X, y, HpoRequest(algo_name="ridge"))

    assert plan.resolved_via_auto is True
    if plan.method == "train_test_split":
        assert plan.split_test_size == int(plan.auto_decision["test_size"])
    else:
        assert plan.split_test_size == plan.test_size


def test_plan_validation_rejects_invalid_feature_selection_cfg() -> None:
    X, y = _numeric_frame()
    request = HpoRequest(algo_name="ridge", feature_selection_cfg={"mode": "bogus"})
    with pytest.raises(HyperoptError, match="Invalid feature_selection_cfg"):
        plan_validation(X, y, request)


def test_normalize_feature_selection_cfg() -> None:
    assert assert_pure(_normalize_feature_selection_cfg, None) == FeatureSelectionCfg()
    cfg = assert_pure(_normalize_feature_selection_cfg, {"mode": "always"})
    assert cfg.mode == FeatureSelectionMode.always
    with pytest.raises(TypeError):
        _normalize_feature_selection_cfg(42)


# ──────────────────────────────────────────────────────────────────────────────
#  plan_preprocessing and the trial pipeline
# ──────────────────────────────────────────────────────────────────────────────


def test_plan_preprocessing_drops_uninformative_columns_in_the_plan() -> None:
    X, _ = _frame()
    request = HpoRequest(algo_name="ridge")

    plan = assert_pure(plan_preprocessing, X, request)

    assert plan.dropped_features == ["const"]
    assert list(plan.X.columns) == ["x1", "x2", "city"]
    assert "const" in X.columns  # the input is not changed
    assert plan.categorical_features == ["city"]
    assert plan.numerical_features == ["x1", "x2"]
    assert plan.preprocessor is not None
    # one-hot city (3) + 2 numeric (no missing values → no indicators).
    assert plan.n_features_eff == 5
    assert plan.n_features_size == 3
    assert plan.n_numeric_eff == 2
    assert plan.isotonic_max_feature_index == 1


def test_plan_preprocessing_reconciles_explicit_column_lists() -> None:
    X, _ = _frame()
    request = HpoRequest(
        algo_name="ridge",
        categorical_features=["city"],
        numerical_features=["x1", "x2", "const"],
    )

    plan = assert_pure(plan_preprocessing, X, request)

    assert plan.numerical_features == ["x1", "x2"]
    assert request.numerical_features == ["x1", "x2", "const"]


def test_plan_preprocessing_ndarray_without_lists() -> None:
    X = np.arange(20, dtype=float).reshape(10, 2)
    plan = assert_pure(plan_preprocessing, X, HpoRequest(algo_name="ridge"))

    assert plan.preprocessor is None
    assert plan.n_numeric_eff is None
    assert plan.n_features_size == plan.n_features_eff == 1
    assert plan.isotonic_max_feature_index == 1


def test_plan_preprocessing_unknown_algorithm_has_no_preprocessor() -> None:
    X, _ = _numeric_frame()
    plan = assert_pure(plan_preprocessing, X, HpoRequest(algo_name="fake_algo"))

    assert plan.preprocessor is None
    assert plan.n_features_eff == plan.n_features_size == 2


def test_estimate_n_features_after_preprocessing() -> None:
    X, _ = _frame()
    n = assert_pure(
        _estimate_n_features_after_preprocessing,
        X,
        preprocessor=None,
        categorical_features=["city"],
        numerical_features=["x1", "x2"],
        encoding=EncodingSettings(encoding="one_hot"),
        algo="ridge",
    )
    # one-hot city (3) + 2 numeric (no missing values → no indicators).
    assert n == 5


def test_isotonic_max_feature_index_and_missing_indicator() -> None:
    assert assert_pure(_isotonic_max_feature_index, 3, True, 9, 5) == 2
    assert assert_pure(_isotonic_max_feature_index, None, True, 9, 5) == 8
    assert assert_pure(_isotonic_max_feature_index, None, False, 9, 5) == 4
    assert assert_pure(_missing_indicator_added, "ridge") is True
    assert assert_pure(_missing_indicator_added, "isotonic_regression") is False


def test_assemble_estimator_step_order() -> None:
    X, _ = _frame()
    preprocessing = plan_preprocessing(X, HpoRequest(algo_name="ridge"))
    model = Ridge()

    pipeline = assert_pure(
        assemble_estimator,
        model,
        preprocessing,
        is_isotonic=False,
        fs_cfg=FeatureSelectionCfg(),
        oversampling=OversamplingSettings(enable=True, multiplier=1.5),
        random_state=3,
        apply_fs=True,
    )

    assert [name for name, _ in pipeline.steps] == [
        "preprocessor",
        "feature_selector",
        "sampler",
        "model",
    ]
    assert pipeline.named_steps["model"] is model
    assert pipeline.named_steps["sampler"].multiplier == 1.5


def test_assemble_estimator_bare_model_and_isotonic() -> None:
    plan = PreprocessingPlan(
        X=None,
        dropped_features=[],
        categorical_features=None,
        numerical_features=None,
        preprocessor=None,
        n_features_eff=1,
        n_numeric_eff=4,
        n_features_size=1,
        isotonic_max_feature_index=3,
    )
    model = Ridge()
    bare = assemble_estimator(
        model,
        plan,
        is_isotonic=False,
        fs_cfg=FeatureSelectionCfg(),
        oversampling=OversamplingSettings(),
        random_state=0,
    )
    assert bare is model
    isotonic = assemble_estimator(
        model,
        plan,
        is_isotonic=True,
        fs_cfg=FeatureSelectionCfg(),
        oversampling=OversamplingSettings(),
        random_state=0,
        feature_index=2,
    )
    adapter = isotonic.named_steps["isotonic_selector"]
    assert (adapter.feature_index, adapter.n_numeric) == (2, 4)


def test_resolve_apply_fs() -> None:
    trial = FixedTrial({"use_feature_selection": True})
    assert assert_pure(_resolve_apply_fs, trial, FeatureSelectionMode.auto, False)
    assert not _resolve_apply_fs(trial, FeatureSelectionMode.auto, True)
    assert _resolve_apply_fs(trial, FeatureSelectionMode.always, False)
    assert not _resolve_apply_fs(trial, FeatureSelectionMode.disabled, False)


def test_fs_transformer_factory_seed() -> None:
    cfg = FeatureSelectionCfg()
    assert assert_pure(_fs_transformer_factory, cfg, None).random_state == 42
    assert _fs_transformer_factory(cfg, 7).random_state == 7


def test_resolve_isotonic_feature_name_does_not_fit_the_preprocessor() -> None:
    X, y = _frame()
    preprocessing = plan_preprocessing(X, HpoRequest(algo_name="isotonic_regression"))

    name = assert_pure(
        _resolve_isotonic_feature_name,
        preprocessing.preprocessor,
        preprocessing.n_numeric_eff,
        preprocessing.X,
        y,
        1,
    )

    assert name == "num__x2"
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessing.preprocessor)
    assert _resolve_isotonic_feature_name(None, 2, X, y, 0) is None


# ──────────────────────────────────────────────────────────────────────────────
#  plan_search_space and the search-space helpers
# ──────────────────────────────────────────────────────────────────────────────


def _preprocessing_counts(n_features: int = 5, n_numeric: int | None = 3) -> PreprocessingPlan:
    return PreprocessingPlan(
        X=None,
        dropped_features=[],
        categorical_features=None,
        numerical_features=None,
        preprocessor=None,
        n_features_eff=n_features,
        n_numeric_eff=n_numeric,
        n_features_size=n_features,
        isotonic_max_feature_index=max(0, (n_numeric or n_features) - 1),
    )


def test_plan_search_space_applies_clipping_and_guard() -> None:
    _, y = _numeric_frame(n=20)
    user = {"max_depth": SearchSpaceEntry.model_validate([2, 30, "int"])}
    plan = assert_pure(
        plan_search_space,
        "decision_tree",
        y,
        space_override=user,
        limit_hyperparameters=False,
        user_hyperparameter_keys=None,
        complexity_guard=ComplexityGuardConfig(),
        n_samples_eff=15,
        preprocessing=_preprocessing_counts(),
    )

    assert plan.user_hyperparameter_keys == ["max_depth"]
    assert set(build_search_space("decision_tree", None)) <= set(plan.space_dict)
    assert plan.guard_records, "the guard narrows a tree space on 15 rows"
    assert user["max_depth"].high == 30  # the override is not changed
    params = plan.space_fn(FixedTrial({k: _low(v) for k, v in plan.space_dict.items()}))
    assert set(params) == set(plan.space_dict)


def _low(entry: object) -> object:
    if isinstance(entry, SearchSpaceEntry):
        if entry.dist_type == "categorical":
            return entry.config.options[0]
        return int(entry.low) if entry.dist_type == "int" else float(entry.low)
    return entry


def test_plan_search_space_svr_epsilon_and_isotonic_feature_index() -> None:
    _, y = _numeric_frame()
    svr = assert_pure(
        plan_search_space,
        "svr",
        y,
        space_override=None,
        limit_hyperparameters=False,
        user_hyperparameter_keys=[],
        complexity_guard=ComplexityGuardConfig(mode="off"),
        n_samples_eff=20,
        preprocessing=_preprocessing_counts(),
    )
    assert svr.space_dict["epsilon"].dist_type == "float_log"
    assert svr.guard_records == []

    isotonic = plan_search_space(
        "isotonic_regression",
        y,
        space_override=None,
        limit_hyperparameters=False,
        user_hyperparameter_keys=[],
        complexity_guard=ComplexityGuardConfig(),
        n_samples_eff=20,
        preprocessing=_preprocessing_counts(n_numeric=3),
    )
    entry = isotonic.space_dict["feature_index"]
    assert (entry.low, entry.high) == (0, 2)


def test_plan_search_space_callable_and_missing_space() -> None:
    def space_fn(trial):
        return {"alpha": trial.suggest_float("alpha", 0.1, 1.0)}

    plan = plan_search_space(
        "ridge",
        None,
        space_override=space_fn,
        limit_hyperparameters=False,
        user_hyperparameter_keys=None,
        complexity_guard=ComplexityGuardConfig(),
        n_samples_eff=10,
        preprocessing=_preprocessing_counts(),
    )
    assert plan.space_fn is space_fn and plan.space_dict == {}
    with pytest.raises(HyperoptError, match="No search space for algorithm"):
        plan_search_space(
            "fake_algo",
            None,
            space_override=None,
            limit_hyperparameters=False,
            user_hyperparameter_keys=None,
            complexity_guard=ComplexityGuardConfig(),
            n_samples_eff=10,
            preprocessing=_preprocessing_counts(),
        )


def test_log_complexity_guard(caplog: pytest.LogCaptureFixture) -> None:
    records = [
        {
            "algo": "decision_tree",
            "param": "max_depth",
            "before": [2, 30],
            "after": [2, 4],
            "n": 15,
            "p": 5,
            "p_size": 5,
            "size_mode": "small",
            "leaf_min": 2,
            "level": "A",
            "applied": True,
            "reason": "small data",
        }
    ]
    with caplog.at_level(logging.INFO):
        assert_pure(log_complexity_guard, records, ComplexityGuardConfig())
    assert "[complexity_guard] decision_tree max_depth 2–30 → 2–4" in caplog.text


def test_normalize_space_dict_and_enqueueable_params(caplog) -> None:
    space = assert_pure(_normalize_space_dict, "svr", {"C": [1, 50], "max_iter": 10})
    assert space["C"].dist_type == "float_log" and space["max_iter"] == 10

    kernel = SearchSpaceEntry.model_validate([["rbf", "linear"], "categorical"])
    kept = assert_pure(
        _enqueueable_initial_params,
        {"C": 2.0, "kernel": "poly", "gamma": "scale"},
        {"C": space["C"], "kernel": kernel, "gamma": space["C"]},
        "svr",
    )
    assert kept == {"C": 2.0}


def test_canonical_distributions() -> None:
    def space_fn(trial):
        return {"alpha": trial.suggest_float("alpha", 0.1, 1.0, log=True)}

    dists = assert_pure(
        _canonical_distributions,
        space_fn,
        fs_mode=FeatureSelectionMode.auto,
        is_isotonic=False,
        algo="ridge",
    )
    assert set(dists) == {"alpha", "use_feature_selection"}

    def broken(trial):
        raise RuntimeError("cannot replay")

    assert (
        _canonical_distributions(
            broken, fs_mode=FeatureSelectionMode.disabled, is_isotonic=False, algo="x"
        )
        is None
    )


def test_natural_domain_bounds_and_guard_boundary() -> None:
    entry = SearchSpaceEntry.model_validate([1, 30, "int"])
    bounds = assert_pure(
        _natural_domain_bounds,
        "n_neighbors",
        entry,
        n_samples_eff=20,
        n_features_eff=5,
        isotonic_max_feature_index=2,
    )
    assert bounds == (1.0, 19.0)
    records = [
        {
            "algo": "decision_tree",
            "param": "max_depth",
            "before": [2, 30],
            "after": [2, 4],
            "applied": True,
        }
    ]
    assert assert_pure(_is_guard_boundary, "decision_tree", records, "max_depth", 4.0, 1e-6)
    assert not _is_guard_boundary("decision_tree", records, "max_depth", 2.0, 1e-6)


def test_warn_if_optimum_at_boundary(caplog) -> None:
    search = SearchSpacePlan(
        space_fn=lambda trial: {},
        space_dict={"alpha": SearchSpaceEntry.model_validate([0.1, 1.0, "float"])},
        guard_records=[],
        user_hyperparameter_keys=[],
    )
    with caplog.at_level(logging.WARNING):
        assert_pure(
            _warn_if_optimum_at_boundary,
            {"alpha": 1.0},
            search,
            algo="ridge",
            n_samples_eff=50,
            preprocessing=_preprocessing_counts(),
        )
    assert "may lie beyond the range" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
#  plan_pruning
# ──────────────────────────────────────────────────────────────────────────────


def test_plan_pruning() -> None:
    cfg = {"enable": True, "strategy": "median", "min_steps": 2}
    plan = assert_pure(plan_pruning, cfg, "k_fold")
    assert plan.active is True
    assert isinstance(plan.pruner, optuna.pruners.MedianPruner)

    hold_out = assert_pure(plan_pruning, cfg, "train_test_split")
    assert hold_out.active is False and hold_out.pruner is not None
    assert plan_pruning(None, "k_fold") == PruningPlan(pruner=None, active=False)
    with pytest.raises(HyperoptError):
        plan_pruning({"enable": True, "strategy": "bogus"}, "k_fold")


# ──────────────────────────────────────────────────────────────────────────────
#  score_trial and the OOF matrix
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        ("neg_root_mean_squared_error", {"val_method": "k_fold", "n_folds": 3}),
        ("neg_mean_absolute_error", {"val_method": "k_fold", "n_folds": 3}),
        ("neg_mean_absolute_error", {"val_method": "train_test_split"}),
        ("r2", {"val_method": "loo"}),
    ],
)
@pytest.mark.parametrize("collect_oof", [False, True])
def test_score_trial_is_pure(metric: str, kwargs: dict, collect_oof: bool) -> None:
    """The estimator is not fitted, the data is not changed, results repeat."""
    X, y = _numeric_frame(n=24)
    validation = _validation(X, y, **kwargs)
    estimator = Ridge(alpha=0.5)
    from configurable_automl_engine.training_engine.metrics import (
        is_non_additive_metric,
    )

    result = assert_pure(
        score_trial,
        estimator,
        X,
        y,
        trial=FixedTrial({}),
        validation=validation,
        scorer=_build_scorer(metric),
        pruning_active=False,
        pool_metric=is_non_additive_metric(metric),
        random_state=0,
        collect_oof=collect_oof,
    )

    assert np.isfinite(result.value)
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)
    if collect_oof:
        assert result.oof is not None and result.oof.shape == (len(X),)
        assert np.isfinite(result.oof).any()
    else:
        assert result.oof is None


def test_score_trial_non_finite_value_becomes_worst_score() -> None:
    X, y = _numeric_frame(n=24)
    validation = _validation(X, y, val_method="k_fold", n_folds=3)

    def nan_scorer(estimator, X_te, y_te):
        return float("nan")

    result = score_trial(
        Ridge(),
        X,
        y,
        trial=FixedTrial({}),
        validation=validation,
        scorer=nan_scorer,
        pruning_active=False,
        pool_metric=False,
        random_state=0,
    )
    assert result.value == HPO_WORST_SCORE


def _frozen(number: int, state: TrialState) -> FrozenTrial:
    return FrozenTrial(
        number=number,
        state=state,
        value=1.0 if state == TrialState.COMPLETE else None,
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
    )


def test_phase_oof_matrix() -> None:
    trials = [
        _frozen(0, TrialState.COMPLETE),
        _frozen(1, TrialState.PRUNED),
        _frozen(2, TrialState.COMPLETE),
    ]
    oof = {0: np.array([1.0, 2.0]), 1: np.array([0.0, 0.0]), 2: np.array([3.0, 4.0])}

    matrix = assert_pure(_phase_oof_matrix, trials, oof)

    np.testing.assert_array_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])
    assert _phase_oof_matrix(trials, {}) is None


# ──────────────────────────────────────────────────────────────────────────────
#  The request helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_as_complexity_guard_config() -> None:
    assert assert_pure(as_complexity_guard_config, None) == ComplexityGuardConfig()
    cfg = ComplexityGuardConfig(mode="warn")
    assert as_complexity_guard_config(cfg) is cfg
    assert assert_pure(as_complexity_guard_config, {"mode": "off"}).mode == "off"


def test_request_defaults_match_optimize() -> None:
    request = HpoRequest(algo_name="ridge")
    assert request.policies == InputPolicies()
    assert request.oversampling.sampler_params() == {
        "multiplier": 1.0,
        "algorithm": "random",
        "target_bins": 8,
        "add_noise": False,
        "noise_level": 0.01,
    }
    assert request.encoding.high_cardinality_threshold == 50


# ──────────────────────────────────────────────────────────────────────────────
#  The engine side: the request builder and the result mapping
# ──────────────────────────────────────────────────────────────────────────────


def test_build_hpo_request_is_pure() -> None:
    from configurable_automl_engine.training_engine.config_parser import AlgoCfg
    from configurable_automl_engine.training_engine.hpo_phases import (
        _build_hpo_request,
    )
    from tests.hpo_spec_factory import make_phase_spec

    algo_cfg = AlgoCfg(hyperparameters={"alpha": [0.1, 1.0]})
    request = assert_pure(
        _build_hpo_request,
        "ridge",
        algo_cfg,
        make_phase_spec(),
        initial_params={"alpha": 0.5},
    )
    assert request.space_overrides["ridge"] == algo_cfg.hyperparameters
    assert request.space_overrides["ridge"] is not algo_cfg.hyperparameters


def test_to_phase_result_is_pure() -> None:
    from configurable_automl_engine.hpo.types import OptimizeResult
    from configurable_automl_engine.training_engine.hpo_phases import (
        _to_phase_result,
    )

    result = OptimizeResult(
        model=None,
        params={"alpha": 0.5},
        score=0.9,
        trials=[],
        complexity_guard=None,
        n_trials_evaluated=4,
    )
    phase = assert_pure(_to_phase_result, result, "ridge")
    assert (phase.score, phase.params, phase.n_trials_evaluated) == (0.9, {"alpha": 0.5}, 4)
    assert _to_phase_result(None, "ridge") is None


# ──────────────────────────────────────────────────────────────────────────────
#  The runner keeps a few grouped results
# ──────────────────────────────────────────────────────────────────────────────


def test_runner_state_is_small() -> None:
    """``_OptimizeRunner`` holds at most 40 attributes after a full run."""
    from configurable_automl_engine.hpo.runner import _OptimizeRunner

    X, y = _numeric_frame()
    runner = _OptimizeRunner(
        X, y, HpoRequest(algo_name="ridge", n_trials=2, collect_oof=True)
    )
    result = runner.run()

    assert result is not None and result.model is not None
    assert len(vars(runner)) <= 40, sorted(vars(runner))
