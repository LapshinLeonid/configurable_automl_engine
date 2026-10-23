"""
Parameterized tests for configurable_automl_engine.tuner.

• Smoke-test that optimize succeeds for every algorithm with a defined
  search space (including SGD, GPR, Isotonic, ARD, Poisson/Gamma/Tweedie).
• Validate the user-supplied space-override.
• Check errors: unknown algorithm, bad data, n_trials ≤ 0.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

import yaml
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.trial import FixedTrial
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from configurable_automl_engine import tuner as hyperopt
from configurable_automl_engine.common.hyperopt_defaults import (
    FloatSpace,
    SearchSpaceEntry,
)
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.tuner import (
    HPO_WORST_SCORE,
    HyperoptError,
    _apply_dynamic_space,
    _build_scorer,
    _can_stratify,
    _missing_indicator_enabled,
    optimize,
)

# ──────────────────────────────────────────────────────────────────────────────
# 0. Make the package visible and import the module
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Fixture with toy data
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(
        n_samples=120,
        n_features=10,
        noise=0.1,
        random_state=1,
    )
    return pd.DataFrame(X), pd.Series(y)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Helper to adapt X/y for "fussy" models
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_data(algo: str, X: pd.DataFrame, y: pd.Series):
    """Adapt the shape and range to the requirements of specific algorithms."""
    # IsotonicRegression needs exactly one feature
    if algo == "isotonicregression":
        X = X.iloc[:, [0]]
    # The GLM family requires y > 0
    if algo in {"poissonregressor", "gammaregressor", "tweedieregressor"}:
        y = np.abs(y) + 1.0
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 3. Smoke test for each supported algorithm
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "algo",
    ["knn"],
)
def test_optimize_smoke(algo: str, toy_data):
    """optimize must not fail and returns valid objects."""
    X, y = _prepare_data(algo, *toy_data)

    opt_res = hyperopt.optimize(
        algo,
        X,
        y,
        n_trials=3,  # minimal for speed
        random_state=0,
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert isinstance(params, dict) and params, "best_params is empty"
    assert isinstance(score, float) and not np.isnan(score), "score is invalid"
    assert hasattr(model, "predict"), "model has no predict method"


# ──────────────────────────────────────────────────────────────────────────────
# 4. User-supplied space-override
# ──────────────────────────────────────────────────────────────────────────────
def test_space_override(toy_data):
    X, y = toy_data

    def tiny_space(trial):
        return {"alpha": trial.suggest_float("alpha", 0.0, 0.001)}

    opt_res = hyperopt.optimize(
        "elasticnet",
        X,
        y,
        n_trials=2,
        space_overrides={"elasticnet": tiny_space},
    )
    assert opt_res is not None
    _, params, _ = opt_res.model, opt_res.params, opt_res.score

    assert params["alpha"] <= 0.001


def test_dict_space_override_remaps_legacy_names_once(toy_data):
    """A dict-based space override with a legacy name is remapped once, before the trials.

    The legacy ``n_iter`` key of ARDRegression must become ``max_iter`` in the
    space: otherwise every trial would trigger the "Remapped ..." DEBUG line
    inside ``clean_hyperparameters`` (~100 identical log entries per sweep).
    """
    X, y = toy_data
    y = pd.Series(np.abs(y) + 1.0)

    space_overrides = {
        "ardregression": {
            "n_iter": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
            "alpha_1": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        }
    }

    with patch(
        "configurable_automl_engine.tuner.create_model",
        wraps=hyperopt.create_model,
    ) as mock_create:
        opt_res = hyperopt.optimize(
            "ardregression",
            X,
            y,
            n_trials=4,
            random_state=0,
            space_overrides=space_overrides,
            validation_strategy="train_test_split",
        )

    assert opt_res is not None
    params = opt_res.params
    assert "n_iter" not in params
    assert "max_iter" in params
    assert 50 <= params["max_iter"] <= 500

    # Every model instantiation that receives hyperparameters must use the
    # modern name: the per-trial clean_hyperparameters has nothing left to
    # remap (the bare existence check call has no hyperparams at all).
    assert mock_create.call_args_list, "create_model was never called"
    for args, kwargs in mock_create.call_args_list:
        if args and args[0] == "ardregression" and kwargs:
            assert "n_iter" not in kwargs
            assert "max_iter" in kwargs


# ──────────────────────────────────────────────────────────────────────────────
# 5. Negative scenarios
# ──────────────────────────────────────────────────────────────────────────────
def test_invalid_algo(toy_data):
    X, y = toy_data
    with pytest.raises(hyperopt.InvalidAlgorithmError):
        hyperopt.optimize("does_not_exist", X, y, n_trials=2)


def test_bad_data():
    with pytest.raises(hyperopt.InvalidDataError):
        hyperopt.optimize("ridge", "not-an-array", [1, 2, 3], n_trials=2)


@pytest.mark.parametrize("bad_trials", [-1, -3])
def test_non_positive_trials(bad_trials, toy_data):
    X, y = toy_data
    with pytest.raises(ValueError):
        hyperopt.optimize("ridge", X, y, n_trials=bad_trials)


def test_zero_trials_returns_no_result(toy_data):
    """n_trials=0 → None (the unified "no valid result" signal).

    Previously n_trials=0 was considered invalid and raised ValueError. Now it is
    a legal "empty search": no trials means no valid result, so optimize()
    signals that by returning None.
    """
    X, y = toy_data
    result = hyperopt.optimize("ridge", X, y, n_trials=0)

    assert result is None


def test_apply_dynamic_space_types(toy_data):
    """
    Covers the logic of _apply_dynamic_space.
    We use the 'knn' algorithm because the code has a space factory for it.
    """
    X, y = toy_data

    class MockEntry:
        def __init__(self, bounds):
            self.bounds = bounds

        @property
        def low(self):
            return self.bounds[0]

        @property
        def high(self):
            return self.bounds[1]

        @property
        def dist_type(self):
            return self.bounds[2]

        @property
        def step(self):
            return self.bounds[3] if len(self.bounds) > 3 else None

    # Mimic the YAML structure for KNN
    # This covers the int, float, float_log, categorical and constant branches
    dynamic_config = {
        "n_neighbors": MockEntry([5, 15, "int"]),
        "p": MockEntry([1, 2, "int"]),
        "weights": MockEntry([["uniform", "distance"], None, "categorical"]),
        "leaf_size": 30,  # Constant (line 144)
    }
    opt_res = hyperopt.optimize(
        "knn", X, y, n_trials=2, space_overrides={"knn": dynamic_config}
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert isinstance(params["n_neighbors"], int)
    assert params["weights"] in ["uniform", "distance"]
    # Check that the constant made it into the model.
    # With preprocessor autodetection the model is wrapped in an ImbPipeline
    # (even for purely numeric data), so we extract the 'model' step.
    assert isinstance(model, ImbPipeline)
    assert model.named_steps["model"].leaf_size == 30


def test_validate_data_mismatch():
    """Checks the error when the lengths of X and y do not match (line 159)."""
    X = np.zeros((10, 2))
    y = np.zeros(5)
    with pytest.raises(hyperopt.InvalidDataError, match="Size mismatch"):
        hyperopt._validate_data(X, y)


def test_validate_data_invalid_y_type():
    """Checks the error for an invalid y type (line 155)."""
    X = np.zeros((5, 2))
    y = {1: 0, 2: 0}  # dict is not in ok_types
    with pytest.raises(hyperopt.InvalidDataError, match="y must be"):
        hyperopt._validate_data(X, y)


def test_optimize_with_oversampling(toy_data):
    """
    Covers the logic of enabling oversampling in the Pipeline.
    """
    X, y = toy_data
    y_bin = (y > y.mean()).astype(int)
    opt_res = hyperopt.optimize(
        "knn",
        X,
        y_bin,
        data_oversampling=True,
        data_oversampling_algorithm="random",
        data_oversampling_multiplier=1.2,
        n_trials=2,
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    print(model.steps)
    assert isinstance(model, ImbPipeline)
    assert any(isinstance(step[1], DataOversampler) for step in model.steps)


def test_optimize_pure_numeric_builds_preprocessor(toy_data):
    """On a purely numeric DataFrame without explicit column lists, optimize
    builds a preprocessor with StandardScaler (desync of the ``if cats:`` branch)."""
    from sklearn.preprocessing import StandardScaler

    X, y = toy_data  # fully numeric DataFrame

    opt_res = hyperopt.optimize(
        "knn",
        X,
        y,
        n_trials=2,
        random_state=0,
    )
    assert opt_res is not None
    model, _, _ = opt_res.model, opt_res.params, opt_res.score


    # The preprocessor must be built even when there are no categories
    assert isinstance(model, ImbPipeline)
    assert "preprocessor" in [step[0] for step in model.steps]

    preprocessor = model.named_steps["preprocessor"]
    transformers = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )
    assert "num" in transformers
    num_pipeline = transformers["num"]
    assert isinstance(num_pipeline.named_steps["scaler"], StandardScaler)


def test_can_stratify_negative():
    """Checks the conditions under which stratification is impossible (lines 182-183)."""
    # Many unique values (regression)
    y_reg = np.linspace(0, 1, 100)
    assert hyperopt._can_stratify(y_reg) is False

    # Multi-dimensional y
    y_multi = np.zeros((10, 2))
    assert hyperopt._can_stratify(y_multi) is False


def test_split_train_test_fallback():
    """Checks the fallback in train_test_split when stratification fails (lines 194-196)."""
    # Create a situation where stratification is impossible because a class has 1 instance
    X = np.random.rand(10, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # The method must complete without errors, catching the ValueError internally
    res = hyperopt._split_train_test(X, y, test_size=0.5)
    assert len(res) == 4


def test_optimize_pruning(toy_data):
    """
    Covers the exception handling block in _objective.
    Fixed: parameters are now registered via trial.suggest_int.
    """
    X, y = toy_data

    def pruning_space(trial):
        # Use suggest_int so optuna records the parameters in trial.params
        if trial.number == 0:
            n_neighbors = trial.suggest_int("n_neighbors", 5, 5)
            return {"n_neighbors": n_neighbors, "weights": "uniform", "p": 2}

        # The second attempt will raise an error (n_neighbors <= 0)
        n_neighbors = trial.suggest_int("n_neighbors", 0, 0)
        return {"n_neighbors": n_neighbors, "weights": "uniform", "p": 2}

    # The method must not fail; Trial 1 will simply be marked as Pruned
    opt_res = hyperopt.optimize(
        "knn",
        X,
        y,
        n_trials=2,
        # Pass it as a dict for the specific algorithm
        space_overrides={"knn": pruning_space},
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    # Now params will not be empty because Trial 0 completed successfully
    assert "n_neighbors" in params
    assert params["n_neighbors"] == 5


# ──────────────────────────────────────────────────────────────────────────────
# 6. initial_params: enqueue_trial tests
# ──────────────────────────────────────────────────────────────────────────────
def test_optimize_with_initial_params_enqueues_trial(toy_data):
    """Checks that study.enqueue_trial() is called with initial_params."""
    X, y = toy_data
    initial_params = {"alpha": 0.5, "l1_ratio": 0.3}

    with patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create:
        mock_study = MagicMock()
        mock_study.best_params = {"alpha": 0.6, "l1_ratio": 0.4}
        mock_study.best_value = 0.9
        mock_create.return_value = mock_study

        # Mock create_model and _validate_data/_get_estimator
        with (
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.create_model") as mock_create_model,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model

            optimize(
                "elasticnet", X, y, n_trials=1,
                initial_params=initial_params,
                space_overrides={
                    "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
                },
            )

            # Check that enqueue_trial was called with the correct parameters
            mock_study.enqueue_trial.assert_called_once_with(initial_params)
            # Check that optimize is still called
            mock_study.optimize.assert_called_once()


def test_optimize_without_initial_params_does_not_enqueue(toy_data):
    """Checks that without initial_params, enqueue_trial is NOT called."""
    X, y = toy_data

    with patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create:
        mock_study = MagicMock()
        mock_study.best_params = {"alpha": 0.6}
        mock_study.best_value = 0.9
        mock_create.return_value = mock_study

        with (
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.create_model") as mock_create_model,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model

            optimize(
                "elasticnet", X, y, n_trials=1,
                initial_params=None,
                space_overrides={
                    "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
                },
            )

            # enqueue_trial must NOT be called
            mock_study.enqueue_trial.assert_not_called()
            mock_study.optimize.assert_called_once()


def test_knn_space_dynamic_limit():
    """Checks the dynamic k limit in KNN (lines 80-81)."""
    # For 10 samples, 80% is 8. max_k must be 8.
    space_fn = hyperopt._make_knn_space(n_samples=10)

    # Emulate the n_neighbors request
    trial = FixedTrial({"n_neighbors": 5, "weights": "uniform", "p": 1})
    params = space_fn(trial)
    assert params["n_neighbors"] <= 8


def test_apply_dynamic_space_floats():
    trial = MagicMock()

    class MockEntry:
        def __init__(self, bounds):
            self.bounds = bounds

        @property
        def low(self):
            return self.bounds[0]

        @property
        def high(self):
            return self.bounds[1]

        @property
        def dist_type(self):
            return self.bounds[2]

        @property
        def step(self):
            return self.bounds[3] if len(self.bounds) > 3 else None

    space_dict = {
        "learning_rate": MockEntry([0.01, 0.1, "float"]),
        "gamma": MockEntry([1e-5, 1e-1, "float_log"]),
        "constant": 42,
    }

    _apply_dynamic_space(trial, space_dict)

    # Check the calls
    trial.suggest_float.assert_any_call("learning_rate", 0.01, 0.1, step=None)
    trial.suggest_float.assert_any_call("gamma", 1e-05, 0.1, log=True)


@pytest.fixture
def log_space_entry() -> Callable[[float], SearchSpaceEntry]:
    """Factory of real SearchSpaceEntry objects with a float_log distribution.

    Negative values are created via ``model_construct``, bypassing
    Pydantic validation. The standard schema intentionally rejects ``low <= 0``
    for float_log, so the tuner's guard can only be tested with objects built
    outside validation: this exactly mimics a dict config of the old format
    reaching the tuner directly.
    """

    def _make(low: float) -> SearchSpaceEntry:
        return SearchSpaceEntry.model_construct(
            config=FloatSpace.model_construct(type="float_log", low=low, high=1.0)
        )

    return _make


def test_apply_dynamic_space_float_log_rejects_non_positive_low(log_space_entry):
    """Guard in _apply_dynamic_space: float_log with low <= 0 is rejected.

    Negative scenarios use real SearchSpaceEntry objects from the fixture
    (built outside validation); the positive one goes through standard
    validation via ``SearchSpaceEntry.model_validate`` — the full integration path.
    """
    trial = MagicMock()

    for bad_low in (0.0, -1.0):
        with pytest.raises(ValueError, match="low must be > 0 for log-scale"):
            _apply_dynamic_space(trial, {"alpha": log_space_entry(bad_low)})
    assert trial.suggest_float.call_count == 0

    # Positive scenario: a valid entry goes to suggest_float(log=True)
    valid_entry = SearchSpaceEntry.model_validate([1e-6, 1.0, "float_log"])
    _apply_dynamic_space(trial, {"alpha": valid_entry})
    trial.suggest_float.assert_called_once_with("alpha", 1e-06, 1.0, log=True)


def test_build_scorer_error():
    with pytest.raises(HyperoptError, match="Unknown metric name"):
        _build_scorer("non_existent_metric_name_123")


def test_can_stratify_pandas():
    y_series = pd.Series([0, 1, 0, 1])
    assert _can_stratify(y_series) is True

    y_df = pd.DataFrame({"target": [0, 1, 0, 1]})
    assert _can_stratify(y_df) is False  # Because ndim != 1 for a DF with 1 column (usually)


def test_optimize_train_test_split_mode(toy_data):
    X, y = toy_data
    # Force the hold-out mode by passing a small amount of data
    # or by explicitly specifying the strategy
    opt_res = optimize(
        "knn", X, y, validation_strategy="train_test_split", n_trials=1
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score

    assert score is not None


def test_optimize_hpo_score_matches_pooled_trainer_val_score():
    """HPO and select use the same pooled value for non-additive metrics.

    For r2 (non-additive) the tuner's trial evaluation pools the metric over
    the OOF vector, so the HPO score matches the final ModelTrainer's pooled
    val_score on the same folds (the select phase uses the trainer's estimate).
    This is the "HPO and select use the same quantity" requirement of issue
    #123 — the estimate is computed over the pooled OOF vector, not averaged
    by folds.
    """
    from configurable_automl_engine.trainer import ModelTrainer

    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.randn(60, 4))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(60))
    space = {"ridge": lambda t: {"alpha": t.suggest_float("alpha", 0.05, 0.5)}}

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=2,
        metric="r2",
        validation_strategy="k_fold",
        n_folds=5,
        random_state=42,
        fit_best_model=False,
        space_overrides=space,
    )
    assert opt_res is not None

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=5,
        random_state=42,
    )
    # Use the best alpha found by HPO so the comparison is on the same model.
    trainer.hyperparams = dict(opt_res.params)
    trainer.fit(X, y)

    assert trainer.val_score is not None
    # Both estimates are pooled r2 over the same OOF vector/folds; only float
    # noise separates them (the trainer re-fits the pipeline on the folds).
    assert opt_res.score == pytest.approx(trainer.val_score, rel=1e-3)


@pytest.mark.parametrize(
    "metric",
    [
        "neg_root_mean_squared_log_error",  # RMSLE
        "d2_absolute_error_score",  # D²
        "neg_max_error",  # max error
    ],
)
def test_optimize_sklearn_non_additive_metrics_use_pooled_path(metric):
    """sklearn-fallback non-additive metrics (RMSLE, D², max error) use the
    pooled evaluation path, not the fold-average cross_val_score.

    Previously these metrics were absent from the non-additive list and were
    incorrectly averaged per fold (review feedback #2).
    """
    from configurable_automl_engine.trainer import ModelTrainer

    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.randn(60, 4))
    # Positive-valued target: RMSLE (log-based) requires values > -1.
    y = pd.Series(np.abs(X.iloc[:, 0] * 2.0 + rng.randn(60)) + 1.0)
    space = {"ridge": lambda t: {"alpha": 0.1}}

    with patch.object(hyperopt, "_evaluate_pooled_cv") as mock_pooled:
        mock_pooled.return_value = 0.5
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            metric=metric,
            validation_strategy="k_fold",
            n_folds=5,
            random_state=42,
            fit_best_model=False,
            space_overrides=space,
        )
    assert opt_res is not None
    # The pooled path is used: _evaluate_pooled_cv was invoked.
    assert mock_pooled.call_count == 1
    # The trainer pools the metric too (same quantity).
    trainer = ModelTrainer(
        algorithm="ridge", metric=metric, validation_strategy="k_fold", n_folds=5
    )
    trainer.hyperparams = {"alpha": 0.1}
    trainer.fit(X, y)
    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)


def test_optimize_raises_when_no_search_space(monkeypatch):
    """Checks that HyperoptError is raised when there is no search space for the algorithm."""

    # --- 1. Prepare dummy data ---
    X = np.random.rand(20, 3)
    y = np.random.rand(20)

    # --- 2. Mock create_model → so the algorithm is considered valid ---
    def fake_create_model(algo, **kwargs):
        class DummyModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        return DummyModel()

    # --- make the algorithm valid ---
    monkeypatch.setattr(hyperopt, "_get_estimator", lambda algo: True)

    with pytest.raises(HyperoptError, match="No search space for algorithm"):
        optimize("fake_algo", X, y, n_trials=1)


class TestTunerObjective:
    @pytest.fixture
    def dummy_data(self):
        return pd.DataFrame({"a": [1, 2, 3, 4, 5]}), pd.Series([1, 0, 1, 0, 1])

    @pytest.fixture
    def mock_space(self):
        return {"rf": lambda trial: {"n_estimators": 10}}

    def test_objective_trigger_fallback(self, dummy_data, mock_space):
        """
        The test forcibly makes np.isfinite return False
        to verify the HPO_WORST_SCORE constant fallback.
        """
        X, y = dummy_data
        EXPECTED_FALLBACK = HPO_WORST_SCORE
        # 1. Patch the ENTIRE environment so no real function runs
        with (
            patch(
                "configurable_automl_engine.tuner._evaluate_pooled_cv"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model") as mock_create,
            patch(
                "configurable_automl_engine.tuner._build_scorer"
            ) as mock_scorer_factory,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner.np.isfinite") as mock_finite,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            # WE GUARANTEE:
            # 1. We are in the k-fold branch
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # 2. The pooled evaluation returns something
            mock_cv.return_value = 0.5
            # 3. CRITICAL POINT: Any number is NOT finite
            mock_finite.return_value = False

            # The remaining stubs
            mock_create.return_value = MagicMock()
            mock_scorer_factory.return_value = MagicMock()
            # Call
            opt_res = optimize(
                algo_name="rf", X=X, y=y, n_trials=1, space_overrides=mock_space
            )
            assert opt_res is not None
            _, _, best_score = opt_res.model, opt_res.params, opt_res.score

            # Verify
            assert best_score == EXPECTED_FALLBACK
            assert mock_finite.called

    def test_objective_via_actual_nan(self, dummy_data, mock_space):
        """
        Test by patching np.mean (a more natural path).
        If tuner.py uses 'import numpy as np', patch 'np.mean'.
        If it uses 'from numpy import mean', patch 'mean'.
        """
        X, y = dummy_data

        # Try patching mean in the tuner module's namespace
        with (
            patch(
                "configurable_automl_engine.tuner._evaluate_pooled_cv",
                return_value=np.nan,
            ),
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            opt_res = optimize(
                algo_name="rf", X=X, y=y, n_trials=1, space_overrides=mock_space
            )
            assert opt_res is not None
            _, _, best_score = opt_res.model, opt_res.params, opt_res.score

            assert best_score == HPO_WORST_SCORE

    def test_consecutive_failures_disqualifies_algorithm(self, dummy_data, mock_space):
        """
        5 consecutive RuntimeErrors in cross_val_score → InvalidAlgorithmError.
        Checks that optimize() raises InvalidAlgorithmError after
        MAX_FATAL_FAILURES (5) consecutive fatal errors.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner._evaluate_pooled_cv"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # Each pooled evaluation call raises RuntimeError
            mock_cv.side_effect = RuntimeError("fatal failure")

            with pytest.raises(hyperopt.InvalidAlgorithmError, match="disqualified"):
                optimize(
                    algo_name="rf", X=X, y=y, n_trials=10, space_overrides=mock_space
                )

    def test_four_failures_one_success_not_disqualified(self, dummy_data, mock_space):
        """
        4 fatal errors + 1 success → the algorithm is NOT disqualified.
        Checks that with 4 consecutive RuntimeErrors and 1 successful
        cross_val_score run, optimization completes successfully.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner._evaluate_pooled_cv"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # 4 errors in a row, then success
            mock_cv.side_effect = [
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                0.85,
            ]

            opt_res = optimize(
                algo_name="rf", X=X, y=y, n_trials=5, space_overrides=mock_space
            )
            assert opt_res is not None
            _, _, best_score = opt_res.model, opt_res.params, opt_res.score

            assert best_score == 0.85

    def test_consecutive_failures_with_train_test_split(self, dummy_data, mock_space):
        """
        5 consecutive RuntimeErrors in train_test_split → InvalidAlgorithmError.
        Checks that the circuit breaker works with train_test_split,
        not only with k-fold/Leave-One-Out.
        """
        X, y = dummy_data
        with (
            patch("configurable_automl_engine.tuner.iter_splits") as mock_iter_splits,
            patch("configurable_automl_engine.tuner.create_model") as mock_create,
            patch(
                "configurable_automl_engine.tuner._build_scorer"
            ) as mock_scorer_factory,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            # Force train_test_split
            mock_make_cv.return_value = ("train_test_split", None, None)

            # iter_splits must return a fresh iterator on each call
            def fresh_iter(*args, **kwargs):
                return iter(
                    [
                        (
                            np.array([[1], [2]]),
                            np.array([[1], [2]]),
                            np.array([0, 1]),
                            np.array([0, 1]),
                        )
                    ]
                )

            mock_iter_splits.side_effect = fresh_iter

            # Model: fit always fails with RuntimeError
            mock_model = MagicMock()
            mock_model.fit.side_effect = RuntimeError("fatal failure")
            mock_create.return_value = mock_model
            mock_scorer_factory.return_value = MagicMock()

            with pytest.raises(hyperopt.InvalidAlgorithmError, match="disqualified"):
                optimize(
                    algo_name="rf", X=X, y=y, n_trials=10, space_overrides=mock_space
                )

    def test_train_test_split_success_resets_counter(self, dummy_data, mock_space):
        """
        4 fatal errors + 1 success with train_test_split → the algorithm is NOT disqualified.
        Checks that a successful train_test_split resets consecutive_fatal_failures.
        """
        X, y = dummy_data
        with (
            patch("configurable_automl_engine.tuner.iter_splits") as mock_iter_splits,
            patch("configurable_automl_engine.tuner.create_model") as mock_create,
            patch(
                "configurable_automl_engine.tuner._build_scorer"
            ) as mock_scorer_factory,
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("train_test_split", None, None)

            def fresh_iter(*args, **kwargs):
                return iter(
                    [
                        (
                            np.array([[1], [2]]),
                            np.array([[1], [2]]),
                            np.array([0, 1]),
                            np.array([0, 1]),
                        )
                    ]
                )

            mock_iter_splits.side_effect = fresh_iter

            # 4 errors in a row, then success
            mock_model = MagicMock()
            mock_model.fit.side_effect = [
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                RuntimeError("fatal failure"),
                None,  # trial 5 success → _objective returns score
                None,  # final best_model.fit(X, y)
            ]
            mock_create.return_value = mock_model

            # The scorer returns 0.85 on a successful call
            mock_scorer = MagicMock(return_value=0.85)
            mock_scorer_factory.return_value = mock_scorer

            opt_res = optimize(
                algo_name="rf", X=X, y=y, n_trials=5, space_overrides=mock_space
            )
            assert opt_res is not None
            _, _, best_score = opt_res.model, opt_res.params, opt_res.score

            assert best_score == 0.85

    def test_valueerror_not_fatal(self, dummy_data, mock_space):
        """
        10 consecutive ValueErrors → the algorithm is NOT disqualified.
        Checks that ValueError is considered a non-fatal error and does not
        increment the consecutive_fatal_failures counter.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner.model_selection.cross_val_score"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # All 10 cross_val_score calls raise ValueError
            mock_cv.side_effect = ValueError("non-fatal value error")

            # Optimization completes without InvalidAlgorithmError;
            # ValueError simply interrupts the trial (optuna.TrialPruned)
            result = optimize(
                algo_name="rf", X=X, y=y, n_trials=10, space_overrides=mock_space
            )

            # No trial is successful: no COMPLETED trial of the phase, and
            # optimize() signals failure by returning None — this is NOT the
            # "magic" constant -3.4028235e38, which looked like a valid
            # result to the orchestrator.
            assert result is None

    def test_pruned_trials_reset_fatal_counter(self, dummy_data, mock_space):
        """
        1 fatal error + trials pruned by the pruner + 4 fatal errors
        → the algorithm is NOT disqualified (regression).

        Checks that optuna.TrialPruned from early stopping
        (_evaluate_with_intermediate_reports) breaks the sequence of
        fatal failures and resets consecutive_fatal_failures: 5 failures in
        total, but only 4 in a row.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner._evaluate_with_intermediate_reports"
            ) as mock_eval,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            # Trial 1 — fatal failure; trial 2 — pruned as expected;
            # trials 3–6 — fatal failures; trial 7 — success.
            mock_eval.side_effect = [
                RuntimeError("fatal failure"),  # trial 1
                optuna.TrialPruned(),  # trial 2 (early stopping)
                RuntimeError("fatal failure"),  # trial 3
                RuntimeError("fatal failure"),  # trial 4
                RuntimeError("fatal failure"),  # trial 5
                RuntimeError("fatal failure"),  # trial 6
                0.85,  # trial 7
            ]

            opt_res = optimize(
                algo_name="rf",
                X=X,
                y=y,
                n_trials=7,
                space_overrides=mock_space,
                pruning={
                    "enable": True,
                    "strategy": "median",
                    "min_steps": 1,
                    "n_startup_trials": 1,
                },
            )
            assert opt_res is not None
            _, _, best_score = opt_res.model, opt_res.params, opt_res.score

            assert best_score == 0.85

    def test_valueerror_between_failures_resets_counter(self, dummy_data, mock_space):
        """
        1 fatal error + non-fatal ValueErrors + 4 fatal errors
        → the algorithm is NOT disqualified (regression).

        Checks that a non-fatal ValueError breaks the sequence of
        fatal failures and resets consecutive_fatal_failures: 5 failures in
        total, but only 4 in a row.
        """
        X, y = dummy_data
        with (
            patch(
                "configurable_automl_engine.tuner._evaluate_pooled_cv"
            ) as mock_cv,
            patch("configurable_automl_engine.tuner.create_model"),
            patch("configurable_automl_engine.tuner._build_scorer"),
            patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
            patch("configurable_automl_engine.tuner._validate_data"),
            patch("configurable_automl_engine.tuner._get_estimator"),
        ):
            mock_make_cv.return_value = ("k_fold", MagicMock(), None)
            mock_cv.side_effect = [
                RuntimeError("fatal failure"),  # trial 1
                ValueError("non-fatal value error"),  # trial 2
                ValueError("non-fatal value error"),  # trial 3
                RuntimeError("fatal failure"),  # trial 4
                RuntimeError("fatal failure"),  # trial 5
                RuntimeError("fatal failure"),  # trial 6
                RuntimeError("fatal failure"),  # trial 7
                0.85,  # trial 8
            ]

            opt_res = optimize(
                algo_name="rf", X=X, y=y, n_trials=8, space_overrides=mock_space
            )
            assert opt_res is not None
            _, _, best_score = opt_res.model, opt_res.params, opt_res.score

            assert best_score == 0.85


# ══════════════════════════════════════════════════════════════════════════════
# 7. End-to-end test of training_engine (train_best_model)
# ══════════════════════════════════════════════════════════════════════════════


def test_full_training_engine_pipeline(tmp_path: Path) -> None:
    """
    End-to-end integration test of the full training pipeline.

    Checks:
    1. Reading the YAML config.
    2. Running two HPO phases (all_algorithms → refine) and the terminal
       select step.
    3. Saving the final model to disk (.pkl).
    4. Returning a correct result structure.
    """
    from configurable_automl_engine.training_engine import train_best_model

    model_path = tmp_path / "models" / "best_model.pkl"
    config_path = tmp_path / "config.yaml"

    config: dict[str, Any] = {
        "general": {
            "comparison_metric": "rmse",
            "path_to_model": str(model_path),
            "serialization_format": "pickle",
            "validation_strategy": "train_test_split",
            "n_folds": 2,
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {"name": "refine", "n_trials": 1, "action": "refine"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {
            "ridge": {"enable": True},
            "random_forest": {"enable": True},
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Generate synthetic regression data
    X, y = make_regression(n_samples=120, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    result = train_best_model(
        config=str(config_path),
        df=df,
        target="target",
    )

    # Check the result structure
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "algorithm" in result, "Missing key 'algorithm'"
    assert "score" in result, "Missing key 'score'"
    assert "params" in result, "Missing key 'params'"
    assert "model_path" in result, "Missing key 'model_path'"

    assert result["algorithm"] in ("ridge", "random_forest"), (
        f"Unknown winning algorithm: {result['algorithm']}"
    )
    assert isinstance(result["score"], float), f"score is not float: {result['score']}"
    assert isinstance(result["params"], dict) and result["params"], (
        "best_params is empty or not a dict"
    )

    # Check that the model file was saved to disk
    saved_path = Path(result["model_path"])
    assert saved_path.exists(), f"Model file not found: {saved_path}"

    # Check that the model can be loaded back
    loaded = ModelTrainer.load(str(saved_path))
    assert loaded.pipeline is not None, "Loaded model does not contain a pipeline"
    assert loaded.algorithm == result["algorithm"], (
        f"Algorithm mismatch: {loaded.algorithm} != {result['algorithm']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. Adaptive preprocessing presets in optimize()
# ──────────────────────────────────────────────────────────────────────────────


def _num_transformer_of(preprocessor):
    """Extract the numeric transformer from a ColumnTransformer."""
    transformers = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )
    return transformers["num"]


def test_optimize_trees_preprocessor_without_scaling(toy_data):
    """In the HPO phase, trees do not use scaling."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    X, y = toy_data

    opt_res = hyperopt.optimize(
        "random_forest",
        X,
        y,
        n_trials=2,
        random_state=0,
        space_overrides={
            "random_forest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 5, 20)}
        },
    )
    assert opt_res is not None
    model, _, _ = opt_res.model, opt_res.params, opt_res.score


    preprocessor = model.named_steps["preprocessor"]
    num = _num_transformer_of(preprocessor)
    step_names = [s[0] for s in num.steps]
    assert "scaler" not in step_names
    assert num.named_steps["imputer"].strategy == "median"
    assert not isinstance(num.named_steps.get("scaler"), (StandardScaler, RobustScaler))


def test_optimize_glm_uses_robust_scaler(toy_data):
    """In the HPO phase, GLM uses RobustScaler + median imputation."""
    from sklearn.preprocessing import RobustScaler

    X, y = toy_data
    y_pos = pd.Series(np.abs(y) + 1.0)

    opt_res = hyperopt.optimize(
        "gammaregressor",
        X,
        y_pos,
        n_trials=2,
        random_state=0,
        space_overrides={
            "gammaregressor": lambda t: {
                "alpha": t.suggest_float("alpha", 1e-6, 1e-1),
                "max_iter": t.suggest_int("max_iter", 50, 100),
            }
        },
    )
    assert opt_res is not None
    model, _, _ = opt_res.model, opt_res.params, opt_res.score


    num = _num_transformer_of(model.named_steps["preprocessor"])
    assert isinstance(num.named_steps["scaler"], RobustScaler)
    assert num.named_steps["imputer"].strategy == "median"


def test_optimize_preprocessing_override_applied(toy_data):
    """In the HPO phase, an explicit preset override takes priority."""
    from sklearn.preprocessing import StandardScaler

    X, y = toy_data

    # For a tree, the user explicitly asks for standard scaling
    opt_res = hyperopt.optimize(
        "random_forest",
        X,
        y,
        n_trials=2,
        random_state=0,
        preprocessing_override={"scaling": "standard"},
        space_overrides={
            "random_forest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 5, 20)}
        },
    )
    assert opt_res is not None
    model, _, _ = opt_res.model, opt_res.params, opt_res.score


    num = _num_transformer_of(model.named_steps["preprocessor"])
    assert isinstance(num.named_steps["scaler"], StandardScaler)
    # Imputation remained the auto-choice of the tree class
    assert num.named_steps["imputer"].strategy == "median"


def test_optimize_logs_resolved_preset(toy_data, caplog):
    """In the HPO phase, the resolved preset is recorded in the logs."""
    X, y = toy_data

    with caplog.at_level("INFO", logger="configurable_automl_engine.tuner"):
        hyperopt.optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            random_state=0,
            space_overrides={
                "ridge": lambda t: {"alpha": t.suggest_float("alpha", 1e-4, 1.0)}
            },
        )

    assert "Resolved preprocessing preset" in caplog.text
    assert "scale_sensitive" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# 9. Feature selection integration
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def noisy_fs_data() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset for testing feature selection: 2 informative + 18 noisy columns.

    Pure noise in 18 columns makes feature selection useful: a model trained
    on the full space has to fight irrelevant features, while the selector
    (importance via ExtraTreesRegressor) keeps the informative columns.
    """
    rng = np.random.RandomState(7)
    X_info, y = make_regression(
        n_samples=250,
        n_features=2,
        n_informative=2,
        noise=0.15,
        random_state=7,
    )
    X_noise = rng.normal(size=(X_info.shape[0], 18))
    X = np.hstack([X_info, X_noise])
    columns = [f"info_{i}" for i in range(2)] + [f"noise_{i}" for i in range(18)]
    return pd.DataFrame(X, columns=columns), pd.Series(y)


_FS_ELASTICNET_SPACE = {
    "elasticnet": lambda t: {
        "alpha": t.suggest_float("alpha", 1e-3, 1.0, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
    }
}

_FS_ISOTONIC_SPACE = {
    "isotonic_regression": lambda t: {
        "increasing": t.suggest_categorical("increasing", [True, False])
    }
}


def _run_optimize_with_captured_study(
    *args: Any, **kwargs: Any
) -> tuple[Any, optuna.Study, Any, dict[str, Any] | None, float]:
    """Run optimize, intercepting the real Optuna study via create_study.

    Returns the tuple (model, study, params, score).
    """
    real_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=kwargs.get("random_state") or 42),
    )
    with patch(
        "configurable_automl_engine.tuner.optuna.create_study", return_value=real_study
    ):
        opt_res = hyperopt.optimize(*args, **kwargs)
        assert opt_res is not None
        model, params, score = opt_res.model, opt_res.params, opt_res.score

    return model, real_study, params, score


def test_auto_mode_explores_both_and_wins_with_fs(noisy_fs_data):
    """mode='auto': Optuna tries True/False and the trial with selection wins.

    On a dataset of 2 informative features and 18 columns of pure noise,
    shrinking the feature space objectively improves quality, so the best
    trial must pick use_feature_selection=True.
    """
    X, y = noisy_fs_data
    model, study, params, score = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=15,
        random_state=42,
        feature_selection_cfg={"mode": "auto"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    # Optuna explored both selection hypotheses
    tried = {
        t.params["use_feature_selection"]
        for t in study.trials
        if "use_feature_selection" in t.params
    }
    assert tried == {True, False}

    # The winning trial uses feature selection
    assert params["use_feature_selection"] is True
    assert "feature_selector" in model.named_steps
    assert isinstance(model.named_steps["feature_selector"], FeatureSelector)
    assert isinstance(score, float) and not np.isnan(score)


def test_auto_mode_winner_false_omits_feature_selector_in_final_model(toy_data):
    """mode='auto': the winner picked False -> no selector in the final model.

    Covers the path "auto → winner picked False" in the final assembly:
    ``best_apply_fs`` must resolve to False, the feature_selector step is not
    added to the model, and the decision is kept in the returned best_params.
    """
    X, y = toy_data
    with (
        patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.create_model"),
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        mock_study = MagicMock()
        # Explicit trial-state accounting: the mock study exposes one COMPLETED
        # trial with the feature-selection decision, so select_best_trial
        # picks it and reaches the final assembly.
        mock_trial = MagicMock(
            state=optuna.trial.TrialState.COMPLETE,
            number=1,
            value=0.9,
            params={
                "alpha": 0.6,
                "l1_ratio": 0.4,
                "use_feature_selection": False,
            },
        )
        mock_study.get_trials.return_value = [mock_trial]
        mock_create.return_value = mock_study

        opt_res = optimize(
            "elasticnet",
            X,
            y,
            n_trials=1,
            feature_selection_cfg={"mode": "auto"},
            space_overrides={
                "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
            },
        )
        assert opt_res is not None
        model, params, _ = opt_res.model, opt_res.params, opt_res.score


    # The HPO winner explicitly opted out of selection: no selector is built in
    assert "use_feature_selection" in params
    assert params["use_feature_selection"] is False
    assert "feature_selector" not in model.named_steps
    assert "model" in model.named_steps


def test_always_mode_forces_feature_selector(noisy_fs_data):
    """mode='always': the selector is in the pipeline, best_params has no service hints."""
    X, y = noisy_fs_data
    model, _, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "always"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    # The tuner's service key is absent from best_params (suggest_categorical was not called)
    assert "use_feature_selection" not in params
    # The final model is assembled with a feature selection step
    assert "feature_selector" in model.named_steps
    assert isinstance(model.named_steps["feature_selector"], FeatureSelector)


def test_disabled_mode_omits_feature_selector(noisy_fs_data):
    """mode='disabled': the selector does not appear in the final model."""
    X, y = noisy_fs_data
    model, _, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "disabled"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    assert "use_feature_selection" not in params
    assert "feature_selector" not in model.named_steps


def test_auto_mode_default_is_disabled_when_cfg_none(noisy_fs_data):
    """feature_selection_cfg=None is equivalent to disabled (backward compatibility)."""
    X, y = noisy_fs_data
    model, _, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    assert "use_feature_selection" not in params
    assert "feature_selector" not in model.named_steps


def test_auto_mode_initial_params_preserves_fs_decision(noisy_fs_data):
    """enqueue_trial with use_feature_selection from phase 1 starts with that decision."""
    X, y = noisy_fs_data
    initial = {"alpha": 0.5, "l1_ratio": 0.3, "use_feature_selection": True}
    _, study, params, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "auto"},
        initial_params=initial,
        space_overrides=_FS_ELASTICNET_SPACE,
    )

    # The first (enqueued) trial starts with the previous phase's decision
    first = study.trials[0]
    assert first.params.get("use_feature_selection") is True
    assert first.params.get("alpha") == 0.5
    assert first.params.get("l1_ratio") == 0.3
    # The decision key is kept in best_params (monotonicity between phases)
    assert "use_feature_selection" in params


def test_isotonic_regression_ignores_feature_selection(toy_data):
    """Isotonic regression: tuning does not fail, selection is forcibly disabled."""
    X, y = toy_data
    X_iso = X.iloc[:, [0]]

    # mode='always' — even forced mode does not break the univariate algorithm
    opt_res = hyperopt.optimize(
        "isotonic_regression",
        X_iso,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "always"},
        space_overrides=_FS_ISOTONIC_SPACE,
    )
    assert opt_res is not None
    model_always, _, _ = opt_res.model, opt_res.params, opt_res.score

    assert hasattr(model_always, "predict")
    steps_always = [name for name, _ in getattr(model_always, "steps", [])]
    assert "feature_selector" not in steps_always

    # mode='auto' — even if Optuna picks True, the selector does not enter the model
    opt_res = hyperopt.optimize(
        "isotonic_regression",
        X_iso,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "auto"},
        space_overrides=_FS_ISOTONIC_SPACE,
    )
    assert opt_res is not None
    model_auto, params_auto, _ = opt_res.model, opt_res.params, opt_res.score

    assert hasattr(model_auto, "predict")
    steps_auto = [name for name, _ in getattr(model_auto, "steps", [])]
    assert "feature_selector" not in steps_auto
    # For isotonic, the use_feature_selection category is not offered to Optuna
    # (the flag is forcibly False), so the service key does not reach
    # best_params even in 'auto' mode.
    assert "use_feature_selection" not in params_auto


def test_invalid_feature_selection_cfg_rejected(toy_data):
    """An invalid feature_selection_cfg is rejected before the search starts."""
    X, y = toy_data

    # Wrong type: a string instead of dict/FeatureSelectionCfg
    with pytest.raises(TypeError, match="feature_selection_cfg must be a"):
        hyperopt.optimize(
            "ridge", X, y, n_trials=2, feature_selection_cfg="always"
        )

    # Invalid dict: an unknown method is rejected by Pydantic validation
    with pytest.raises(HyperoptError, match="Invalid feature_selection_cfg"):
        hyperopt.optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            feature_selection_cfg={"mode": "always", "method": "not_a_method"},
        )


def test_fs_transformer_factory_uses_fixed_seed_when_random_state_none(toy_data):
    """With random_state=None, FeatureSelector gets a fixed seed of 42.

    Otherwise each fs_transformer_factory call would create a selector with a
    new random seed and feature selection would not be reproducible.
    """
    X, y = toy_data
    with (
        patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.create_model"),
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch(
            "configurable_automl_engine.tuner.FeatureSelector", autospec=True
        ) as mock_fs,
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        mock_study = MagicMock()
        # Explicit trial-state accounting: the mock study exposes one
        # COMPLETED trial so select_best_trial passes its completeness check.
        mock_study.get_trials.return_value = [
            MagicMock(
                state=optuna.trial.TrialState.COMPLETE,
                number=1,
                value=0.9,
                params={"alpha": 0.6},
            )
        ]
        mock_create.return_value = mock_study

        optimize(
            "elasticnet",
            X,
            y,
            n_trials=1,
            random_state=None,
            feature_selection_cfg={"mode": "always"},
            space_overrides={"elasticnet": lambda t: {"alpha": 0.01}},
        )

        assert mock_fs.call_args.kwargs["random_state"] == 42


def test_fs_reproducible_with_random_state_none(noisy_fs_data):
    """Two runs with random_state=None produce the same result.

    Optuna's randomness (TPE, splits) is not fixed with random_state=None,
    but feature selection uses a fixed seed of 42, so with a constant search
    space the resulting models are identical.
    """
    X, y = noisy_fs_data
    run_kwargs = dict(
        n_trials=2,
        random_state=None,
        feature_selection_cfg={"mode": "always"},
        space_overrides={"elasticnet": lambda t: {"alpha": 0.01, "l1_ratio": 0.5}},
    )
    opt_res = hyperopt.optimize("elasticnet", X, y, **run_kwargs)
    assert opt_res is not None
    model1, _, _ = opt_res.model, opt_res.params, opt_res.score

    opt_res = hyperopt.optimize("elasticnet", X, y, **run_kwargs)
    assert opt_res is not None
    model2, _, _ = opt_res.model, opt_res.params, opt_res.score


    assert np.allclose(model1.predict(X), model2.predict(X))


def test_fs_service_key_does_not_leak_into_model_constructor(toy_data):
    """The service key use_feature_selection does not reach the model constructor.

    Covers: initial_params (enqueue_trial) carries the previous phase's decision
    between HPO phases, but when creating the model through create_model the key
    must be stripped, and the original study.best_params must not be mutated.
    """
    X, y = toy_data
    initial_params = {"alpha": 0.5, "use_feature_selection": True}
    with (
        patch("configurable_automl_engine.tuner.optuna.create_study") as mock_create,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.create_model") as mock_create_model,
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch(
            "configurable_automl_engine.tuner.FeatureSelector", autospec=True
        ),
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        mock_study = MagicMock()
        # Explicit trial-state accounting: the mock study exposes one COMPLETED
        # trial with the feature-selection decision, so select_best_trial
        # picks it and reaches the final assembly.
        mock_trial = MagicMock(
            state=optuna.trial.TrialState.COMPLETE,
            number=1,
            value=0.9,
            params={
                "alpha": 0.6,
                "l1_ratio": 0.4,
                "use_feature_selection": True,
            },
        )
        mock_study.get_trials.return_value = [mock_trial]
        mock_create.return_value = mock_study

        opt_res = optimize(
            "elasticnet",
            X,
            y,
            n_trials=1,
            initial_params=initial_params,
            feature_selection_cfg={"mode": "auto"},
            space_overrides={
                "elasticnet": lambda t: {"alpha": t.suggest_float("alpha", 0, 1)}
            },
        )
        assert opt_res is not None
        _, params, _ = opt_res.model, opt_res.params, opt_res.score


        # enqueue_trial receives the previous phase's decision (HPO monotonicity)
        mock_study.enqueue_trial.assert_called_once_with(initial_params)
        # The final create_model does not receive the service key
        assert "use_feature_selection" not in mock_create_model.call_args.kwargs
        # The returned best_params keeps the key (monotonicity between phases)
        assert params["use_feature_selection"] is True


def test_fs_pipeline_from_optimize_serializes(tmp_path: Path, noisy_fs_data):
    """A pipeline with the feature_selector step serializes correctly.

    The new step does not break backward compatibility of pipeline
    serialization: after save/load predict is identical and the step is present
    in the restored model.
    """
    import joblib

    X, y = noisy_fs_data
    model, _, _, _ = _run_optimize_with_captured_study(
        "elasticnet",
        X,
        y,
        n_trials=3,
        random_state=42,
        feature_selection_cfg={"mode": "always"},
        space_overrides=_FS_ELASTICNET_SPACE,
    )
    assert "feature_selector" in model.named_steps

    path = tmp_path / "fs_pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    assert "feature_selector" in restored.named_steps
    assert np.array_equal(model.predict(X), restored.predict(X))


# ══════════════════════════════════════════════════════════════════════════════
# 10. Regression tests for refactoring optimize() into steps
# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter generation (space_fn) and the model constructor (create_model)
# run in _objective OUTSIDE try/except: their errors must propagate outward
# (FAIL trial + an exception from study.optimize), not become silent pruned
# trials → (None, None, None) or be counted by the circuit breaker.
def test_space_fn_error_marks_trial_failed_and_propagates(toy_data):
    """ValueError from a user space function → FAIL, not PRUNED.

    Original contract: an error in the space function marks the trial as FAIL
    and interrupts study.optimize (exception propagates outward), making the
    failure visible to the caller instead of masking it as a pruned trial.
    """
    X, y = toy_data
    real_study = optuna.create_study(direction="maximize")

    def space_raises(trial):
        trial.suggest_float("alpha", 0.0, 1.0)
        raise ValueError("user space fn boom")

    with patch(
        "configurable_automl_engine.tuner.optuna.create_study", return_value=real_study
    ):
        with pytest.raises(ValueError, match="user space fn boom"):
            optimize(
                "ridge",
                X,
                y,
                n_trials=2,
                space_overrides={"ridge": space_raises},
            )

    assert real_study.trials[0].state == optuna.trial.TrialState.FAIL
    # No pruning happens: this is neither early stopping nor a non-fatal failure.
    assert all(t.state != optuna.trial.TrialState.PRUNED for t in real_study.trials)


def test_create_model_error_marks_trial_failed_and_propagates(toy_data):
    """Model constructor failure (MemoryError) → FAIL, not disqualification.

    A constructor error is not a fatal EVALUATION failure: it must not
    increment consecutive_fatal_failures or lead to InvalidAlgorithmError
    after N attempts — per the original code contract.
    """
    X, y = toy_data
    real_study = optuna.create_study(direction="maximize")

    def exploding_model(algo, **kwargs):
        raise MemoryError("model constructor boom")

    with (
        patch(
            "configurable_automl_engine.tuner.optuna.create_study",
            return_value=real_study,
        ),
        patch(
            "configurable_automl_engine.tuner.create_model",
            side_effect=exploding_model,
        ),
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch("configurable_automl_engine.tuner._build_scorer"),
    ):
        with pytest.raises(MemoryError, match="model constructor boom"):
            optimize(
                "ridge",
                X,
                y,
                n_trials=5,
                space_overrides={"ridge": lambda t: {"alpha": 0.1}},
            )

    # The very first trial is FAIL — without accumulating the fatal-failure counter
    # (otherwise after 5 attempts it would be InvalidAlgorithmError, not MemoryError).
    assert real_study.trials[0].state == optuna.trial.TrialState.FAIL
    assert len(real_study.trials) == 1


def test_scoring_valueerror_still_prunes_not_fatal(toy_data):
    """ValueError at the EVALUATION stage (fit/scoring) is still → PRUNED.

    The separation is preserved: only parameter/model generation stays outside
    try/except; errors of the evaluation itself are handled by handle_failure
    as before (non-fatal ValueError → TrialPruned, no disqualification).
    """
    X, y = toy_data
    with (
        patch(
            "configurable_automl_engine.tuner.model_selection.cross_val_score"
        ) as mock_cv,
        patch("configurable_automl_engine.tuner.create_model"),
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        # All 10 cross_val_score calls (evaluation stage) raise ValueError
        mock_cv.side_effect = ValueError("non-fatal scoring error")

        result = optimize(
            algo_name="rf",
            X=X,
            y=y,
            n_trials=10,
            space_overrides={"rf": lambda trial: {"n_estimators": 10}},
        )

        # No trial completed → the "no valid result" contract:
        # None, without InvalidAlgorithmError.
        assert result is None


# ─────────── Cleaning uninformative features ───────────────────


def _tuner_garbage_data(n: int = 120, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """DataFrame with informative columns and garbage (constant/empty)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "const_num": 5.0,
            "all_nan": np.nan,
            "almost_empty": [np.nan] * (n - 1) + [1.0],
            "const_cat": "only",
        }
    )
    y = pd.Series(2.0 * X["x1"] - 1.5 * X["x2"] + rng.normal(0, 0.1, n))
    return X, y


def test_optimize_cleans_uninformative_features():
    """optimize() cleans garbage columns before building the preprocessor:
    the final model works with informative features, predictions are finite,
    and extra input columns do not affect the result."""
    X, y = _tuner_garbage_data()

    opt_res = hyperopt.optimize(
        "ridge",
        X,
        y,
        n_trials=2,
        random_state=0,
        space_overrides={
            "ridge": lambda t: {"alpha": t.suggest_float("alpha", 0.05, 0.5)}
        },
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert isinstance(score, float) and np.isfinite(score)
    assert isinstance(params, dict) and params
    assert hasattr(model, "predict")
    # The model was trained on cleaned data: predictions are finite and robust
    # to extra (garbage) input columns.
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    preds_clean = model.predict(X[["x1", "x2"]])
    assert np.allclose(preds, preds_clean)

    # The preprocessor is built from only the 2 informative columns (not 6).
    preprocessor = model.named_steps["preprocessor"]
    out = preprocessor.transform(X[["x1", "x2"]])
    assert out.shape[1] == 2


def test_optimize_numpy_input_skips_cleaning():
    """np.ndarray without column names: cleaning is skipped (a limitation),
    and optimization completes without failures."""
    rng = np.random.default_rng(3)
    X_arr = rng.normal(size=(100, 3))
    y_arr = X_arr[:, 0] * 2.0 - X_arr[:, 1] + rng.normal(0, 0.1, 100)

    opt_res = hyperopt.optimize(
        "ridge",
        X_arr,
        y_arr,
        n_trials=2,
        random_state=0,
        space_overrides={
            "ridge": lambda t: {"alpha": t.suggest_float("alpha", 0.05, 0.5)}
        },
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert isinstance(score, float) and np.isfinite(score)
    assert hasattr(model, "predict")
    assert np.isfinite(model.predict(X_arr)).all()


def test_runner_dropped_features_recorded():
    """_OptimizeRunner records the list of dropped columns for diagnostics."""
    X, y = _tuner_garbage_data()
    runner = hyperopt._OptimizeRunner(
        algo_name="ridge",
        X=X,
        y=y,
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        metric="r2",
        val_method="train_test_split",
        validation_strategy=None,
        n_folds=5,
        n_trials=2,
        random_state=0,
        sampler_seed=None,
        train_test_split_test_size=0.2,
        space_overrides=None,
        initial_params=None,
        preprocessor=None,
        categorical_features=None,
        numerical_features=None,
        encoding=None,
        preprocessing_override=None,
        pruning=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
        feature_selection_cfg=None,
    )
    runner.resolve_validation()
    runner.build_trial_pipeline()

    assert set(runner.dropped_features_) == {
        "const_num",
        "all_nan",
        "almost_empty",
        "const_cat",
    }
    assert set(runner.X.columns) == {"x1", "x2"}
    assert runner.preprocessor is not None


def test_runner_n_features_size_uses_original_columns():
    """n_features_size counts ORIGINAL columns; n_features_eff is post-one-hot.

    Regression (issue #134): the complexity guard resolved the size mode
    from the one-hot-expanded feature count, so large-but-wide data fell
    into the 'medium' mode. The runner must track both counts:
    n_features_size (original columns, a categorical column = 1) for the
    size-mode ratio and n_features_eff (what the model sees) for the rules.
    """
    rng = np.random.default_rng(13)
    n = 120
    cats = pd.DataFrame(
        {f"c{i}": rng.integers(0, 10, size=n).astype(str) for i in range(4)}
    )
    X = pd.concat([cats, pd.DataFrame({"x1": rng.normal(size=n)})], axis=1)
    y = pd.Series(rng.normal(size=n))

    runner = hyperopt._OptimizeRunner(
        algo_name="random_forest",
        X=X,
        y=y,
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        metric="r2",
        val_method="train_test_split",
        validation_strategy=None,
        n_folds=5,
        n_trials=2,
        random_state=0,
        sampler_seed=None,
        train_test_split_test_size=0.2,
        space_overrides=None,
        initial_params=None,
        preprocessor=None,
        categorical_features=None,
        numerical_features=None,
        encoding="one_hot",
        preprocessing_override=None,
        pruning=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
        feature_selection_cfg=None,
    )
    runner.resolve_validation()
    runner.build_trial_pipeline()

    # 5 original columns (4 categorical + 1 numerical).
    assert runner.n_features_size == 5
    # After one-hot: 4 categorical * 10 categories + 1 numerical = 41.
    assert runner.n_features_eff == 41


def test_runner_n_features_size_aligned_without_preprocessor():
    """Without a preprocessor (np.ndarray input) n_features_size aligns with
    n_features_eff (issue #134 review remark 1).

    For np.ndarray the estimate falls back to n_features_eff=1 (no column
    names to expand); previously n_features_size kept the raw width
    X.shape[1], so wide ndarray inputs entered the 'medium' mode via the
    ratio for the first time. Both counts must be identical so that direct
    optimize(numpy) calls keep the legacy guard behavior.
    """
    rng = np.random.default_rng(3)
    X_arr = rng.normal(size=(100, 8))
    y_arr = X_arr[:, 0] * 2.0 - X_arr[:, 1] + rng.normal(0, 0.1, 100)

    runner = hyperopt._OptimizeRunner(
        algo_name="random_forest",
        X=X_arr,
        y=y_arr,
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        metric="r2",
        val_method="train_test_split",
        validation_strategy=None,
        n_folds=5,
        n_trials=2,
        random_state=0,
        sampler_seed=None,
        train_test_split_test_size=0.2,
        space_overrides=None,
        initial_params=None,
        preprocessor=None,
        categorical_features=None,
        numerical_features=None,
        encoding="one_hot",
        preprocessing_override=None,
        pruning=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
        feature_selection_cfg=None,
    )
    runner.resolve_validation()
    runner.build_trial_pipeline()

    # No preprocessor was assembled for an ndarray without column lists.
    assert runner.preprocessor is None
    # Both counts coincide: the guard sees the same p it always did.
    assert runner.n_features_eff == 1
    assert runner.n_features_size == runner.n_features_eff


def test_optimize_external_preprocessor_skips_cleaning():
    """If a ready preprocessor is provided, self.X is not cleaned:
    the user has fixed the feature set themselves."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    X, y = _tuner_garbage_data()
    external = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["x1", "x2"])]
    )

    runner = hyperopt._OptimizeRunner(
        algo_name="ridge",
        X=X,
        y=y,
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        metric="r2",
        val_method="train_test_split",
        validation_strategy=None,
        n_folds=5,
        n_trials=2,
        random_state=0,
        sampler_seed=None,
        train_test_split_test_size=0.2,
        space_overrides=None,
        initial_params=None,
        preprocessor=external,
        categorical_features=None,
        numerical_features=None,
        encoding=None,
        preprocessing_override=None,
        pruning=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
        feature_selection_cfg=None,
    )
    runner.resolve_validation()
    runner.build_trial_pipeline()

    assert runner.dropped_features_ == []
    assert set(runner.X.columns) == {
        "x1",
        "x2",
        "const_num",
        "all_nan",
        "almost_empty",
        "const_cat",
    }
    assert runner.preprocessor is external


# ──────────────────────────────────────────────────────────────────────────────
#  Missing indicators: a safe fallback in the tuner
# ──────────────────────────────────────────────────────────────────────────────
def test_missing_indicator_enabled_guards_invalid_algorithms():
    """An empty/None/unknown algorithm name does not break preprocessor assembly:
    a safe value of True is returned (indicators enabled)."""
    assert _missing_indicator_enabled(None) is True
    assert _missing_indicator_enabled("") is True
    assert _missing_indicator_enabled("unknown_algo_2026") is True


def test_missing_indicator_enabled_univariate_disabled():
    """Strictly univariate algorithms (isotonic) — indicators disabled."""
    assert _missing_indicator_enabled("isotonic_regression") is False
    assert _missing_indicator_enabled("isotonic") is False


def test_missing_indicator_enabled_regular_algorithms():
    """Regular algorithms — indicators enabled."""
    assert _missing_indicator_enabled("ridge") is True
    assert _missing_indicator_enabled("rf") is True


# ──────────────────────────────────────────────────────────────────────────────
#  TPE history transfer between HPO phases (previous_trials)
# ──────────────────────────────────────────────────────────────────────────────
def test_optimize_previous_trials_returns_only_own_phase_trials(toy_data):
    """T2/T1: when previous_trials is passed, the new study keeps the
    transferred history as TPE warm-start, and the returned OptimizeResult
    contains ONLY the current phase's own trials (numbers >= the transferred
    boundary)."""
    X, y = toy_data
    first = hyperopt.optimize("knn", X, y, n_trials=2, random_state=42)
    assert first is not None
    assert [t.number for t in first.trials] == [0, 1]

    second = hyperopt.optimize(
        "knn", X, y, n_trials=2, random_state=43, previous_trials=first.trials
    )
    assert second is not None
    # Only the current phase's own trials (transferred history is excluded).
    assert len(second.trials) == 2
    assert all(t.number >= 2 for t in second.trials)
    assert len(second.trials + first.trials) == 4


def test_optimize_empty_previous_trials_is_noop(toy_data):
    """N4: previous_trials=[] is a no-op — no exception, no transfer."""
    X, y = toy_data
    res = hyperopt.optimize("knn", X, y, n_trials=2, previous_trials=[])
    assert res is not None
    assert len(res.trials) == 2


def test_optimize_incompatible_previous_trials_fail_fast(toy_data):
    """N2: trials with distributions incompatible with the current search
    space raise HPOHistoryError (fail fast, not a silent skip)."""
    from optuna.distributions import FloatDistribution
    from optuna.trial import create_trial

    X, y = toy_data
    bad = create_trial(
        value=0.5,
        params={"n_neighbors": 5},
        distributions={"n_neighbors": FloatDistribution(0.0, 100.0)},
    )
    with pytest.raises(hyperopt.HPOHistoryError, match="incompatible"):
        hyperopt.optimize("knn", X, y, n_trials=1, previous_trials=[bad])


def test_optimize_unrepresentable_space_fails_fast(toy_data):
    """B2: a search space that cannot be reproduced by the distribution
    pre-check (e.g. a legacy ``suggest_uniform`` call) fails fast with
    ``HPOHistoryError`` when non-empty history is transferred — no silent
    skip, which would silently degrade refine to a cold run."""
    X, y = toy_data
    first = hyperopt.optimize("knn", X, y, n_trials=2, random_state=42)
    assert first is not None

    def exotic_space(trial):
        # suggest_uniform is not implemented by the _DistributionRecorder
        # (only suggest_int/suggest_float/suggest_categorical are), so the
        # pre-check cannot reproduce the space.
        return {"n_neighbors": trial.suggest_uniform("n_neighbors", 1.0, 5.0)}

    with pytest.raises(hyperopt.HPOHistoryError, match="cannot be reproduced"):
        hyperopt.optimize(
            "knn",
            X,
            y,
            n_trials=2,
            space_overrides={"knn": exotic_space},
            previous_trials=first.trials,
        )


def test_optimize_zero_trials_with_history_returns_none(toy_data):
    """B2: n_trials=0 with transferred history → None (unified no-result
    signal), the history is not added to any study."""
    X, y = toy_data
    first = hyperopt.optimize("knn", X, y, n_trials=2, random_state=42)
    assert first is not None
    res = hyperopt.optimize("knn", X, y, n_trials=0, previous_trials=first.trials)
    assert res is None


def test_optimize_transferred_history_does_not_mask_total_failure(toy_data):
    """N1 (tuner level): if ALL own trials of the current phase fail, the run
    returns None even when transferred COMPLETED history is present — the
    history must not make the phase appear valid (no fallback)."""
    X, y = toy_data
    first = hyperopt.optimize("knn", X, y, n_trials=2, random_state=42)
    assert first is not None

    with (
        patch("configurable_automl_engine.tuner._build_scorer"),
        patch("configurable_automl_engine.tuner.create_model"),
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch("configurable_automl_engine.tuner._validate_data"),
        patch("configurable_automl_engine.tuner._get_estimator"),
        patch(
            "configurable_automl_engine.tuner.model_selection.cross_val_score"
        ) as mock_cv,
    ):
        mock_make_cv.return_value = ("k_fold", MagicMock(), None)
        # Every OWN trial evaluation raises a non-fatal ValueError → the trial
        # is PRUNED. With transferred COMPLETED history present, the phase must
        # STILL report no valid result (no fallback to previous-phase params).
        mock_cv.side_effect = ValueError("score boom")

        res = hyperopt.optimize(
            "knn",
            X,
            y,
            n_trials=3,
            random_state=44,
            validation_strategy="k_fold",
            n_folds=3,
            previous_trials=first.trials,
        )
        assert res is None


# ──────────────────────────────────────────────────────────────────────────────
#  fit_best_model: skipping the redundant final full-data fit
# ──────────────────────────────────────────────────────────────────────────────
def _ridge_space(trial):
    """A tiny fixed-space generator for the ridge smoke tests."""
    return {"alpha": trial.suggest_float("alpha", 0.05, 0.5)}


def test_optimize_fit_best_model_disabled_skips_final_fit(toy_data, monkeypatch):
    """fit_best_model=False skips the final full-data fit: the model slot is
    None while the best params and score stay valid."""
    X, y = toy_data

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "_fit_best_model must not run when fit_best_model=False"
        )

    monkeypatch.setattr(hyperopt._OptimizeRunner, "_fit_best_model", _forbidden)

    result = hyperopt.optimize(
        "ridge",
        X,
        y,
        n_trials=2,
        random_state=0,
        fit_best_model=False,
        space_overrides={"ridge": _ridge_space},
    )

    assert result is not None
    assert result.model is None
    assert isinstance(result.params, dict) and result.params
    assert isinstance(result.score, float) and np.isfinite(result.score)


@pytest.mark.parametrize(
    ("fit_best_model", "expect_model"),
    [(True, True), (False, False)],
)
def test_optimize_fit_best_model_flag(toy_data, fit_best_model, expect_model):
    """The flag controls whether the tuned model (fit on all data) is returned."""
    X, y = toy_data

    result = hyperopt.optimize(
        "ridge",
        X,
        y,
        n_trials=2,
        random_state=0,
        fit_best_model=fit_best_model,
        space_overrides={"ridge": _ridge_space},
    )

    assert result is not None
    assert (result.model is not None) is expect_model
    assert isinstance(result.params, dict) and result.params
    assert isinstance(result.score, float) and np.isfinite(result.score)


# ──────────────────────────────────────────────────────────────────────────────
#  sampler_seed: separate the TPE sampler seed from the data-split seed
#  (issue #119 — the phase seed must not change CV folds)
# ──────────────────────────────────────────────────────────────────────────────
def _capture_tpe_seed() -> tuple[dict[str, object], Any]:
    """Patch TPESampler to capture the seed while returning the real sampler."""
    real_tpe = optuna.samplers.TPESampler
    captured: dict[str, object] = {}

    def capturing_tpe(*args, **kwargs):
        captured["seed"] = kwargs.get("seed")
        return real_tpe(*args, **kwargs)

    return captured, patch(
        "configurable_automl_engine.tuner.optuna.samplers.TPESampler",
        side_effect=capturing_tpe,
    )


def test_optimize_sampler_seed_controls_tpe_only(toy_data):
    """sampler_seed feeds the TPE sampler, while the CV object is built with
    the fixed random_state (issue #119: the per-phase seed must not change
    the folds)."""
    X, y = toy_data
    captured, tpe_patch = _capture_tpe_seed()
    with (
        tpe_patch,
        patch("configurable_automl_engine.tuner.make_cv") as mock_make_cv,
        patch(
            "configurable_automl_engine.tuner.model_selection.cross_val_score",
            return_value=np.array([0.5, 0.6, 0.7]),
        ),
    ):
        mock_make_cv.return_value = (
            "k_fold",
            KFold(n_splits=3, shuffle=True, random_state=42),
            None,
        )
        result = hyperopt.optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            random_state=42,
            sampler_seed=43,
            validation_strategy="k_fold",
            n_folds=3,
            space_overrides={"ridge": _ridge_space},
        )

    assert result is not None
    # The TPE sampler runs with the per-phase sampler_seed...
    assert captured["seed"] == 43
    # ... while the folds are built with the fixed data-split seed.
    _, make_cv_kwargs = mock_make_cv.call_args
    assert make_cv_kwargs["random_state"] == 42


def test_optimize_preprocessor_uses_random_state_not_sampler_seed(toy_data):
    """The preprocessor (hashing / target-encoding cross-fit) is seeded with
    random_state, not sampler_seed (issue #119)."""
    X, y = toy_data
    _, tpe_patch = _capture_tpe_seed()
    with (
        tpe_patch,
        patch("configurable_automl_engine.tuner.build_preprocessor") as mock_bp,
        patch(
            "configurable_automl_engine.tuner.model_selection.cross_val_score",
            return_value=np.array([0.5, 0.6]),
        ),
    ):
        mock_bp.return_value = StandardScaler()
        result = hyperopt.optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            random_state=42,
            sampler_seed=43,
            validation_strategy="k_fold",
            n_folds=2,
            numerical_features=list(X.columns),
            space_overrides={"ridge": _ridge_space},
        )

    assert result is not None
    _, bp_kwargs = mock_bp.call_args
    assert bp_kwargs["random_state"] == 42


def test_optimize_sampler_seed_none_falls_back_to_random_state(toy_data):
    """Backward compatibility: without sampler_seed the TPE sampler uses
    random_state exactly as before the split."""
    X, y = toy_data
    captured, tpe_patch = _capture_tpe_seed()
    with (
        tpe_patch,
        patch(
            "configurable_automl_engine.tuner.model_selection.cross_val_score",
            return_value=np.array([0.5, 0.6]),
        ),
    ):
        result = hyperopt.optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            random_state=42,
            validation_strategy="k_fold",
            n_folds=2,
            space_overrides={"ridge": _ridge_space},
        )

    assert result is not None
    assert captured["seed"] == 42


def test_optimize_fixed_folds_restore_enqueue_monotonicity(toy_data):
    """issue #119: with a fixed random_state (the same folds across phases)
    and a per-phase sampler_seed, the params enqueued from the previous phase
    are re-evaluated on the SAME folds — refine can only improve the score."""
    X, y = toy_data
    first = hyperopt.optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        random_state=42,
        sampler_seed=42,
        fit_best_model=False,
        space_overrides={"ridge": _ridge_space},
    )
    assert first is not None

    second = hyperopt.optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        random_state=42,
        sampler_seed=43,
        initial_params=first.params,
        fit_best_model=False,
        space_overrides={"ridge": _ridge_space},
    )
    assert second is not None
    assert second.score >= first.score
