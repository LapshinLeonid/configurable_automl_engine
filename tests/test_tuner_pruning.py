"""
Tests of early stopping (pruning) in configurable_automl_engine.tuner.

They cover the key scenarios of the feature:
• publishing intermediate results and pruning hopeless trials;
• a real MedianPruner/HyperbandPruner prunes some trials, which is recorded
  in the logs;
• best_model/best_params/best_score are selected only from completed trials
  (the metric maximization rule);
• with pruning disabled the behavior is identical to the current one
  (trial.report is not called, cross_val_score is unchanged);
• pruned trials are not treated as fatal failures; when all trials are
  pruned, the "no results" scenario is preserved;
• hold-out (train_test_split) without natural steps: the pruner is not
  applied and the behavior is safe;
• the fact/count of pruned trials is recorded in the logs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from configurable_automl_engine import tuner
from configurable_automl_engine.training_engine.metrics import get_scorer_object
from configurable_automl_engine.tuner import (
    HPO_WORST_SCORE,
    HyperoptError,
    _build_pruner,
    _evaluate_pooled_cv,
    _evaluate_with_intermediate_reports,
    _normalize_pruning_config,
    optimize,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper objects: a deterministic "model" and "scorer"
# ──────────────────────────────────────────────────────────────────────────────
class FakeQualityModel(BaseEstimator, RegressorMixin):
    """Model whose predict returns the constant from the ``quality`` hyperparameter.

    Allows deterministic control over trial quality: the score of each fold
    equals ``quality``, so "bad" trials are guaranteed to be worse than "good"
    ones at every intermediate step.
    """

    def __init__(self, quality: float = 0.5, **kwargs: object) -> None:
        self.quality = quality

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.full(len(X), self.quality)


def _quality_scorer_factory(name: str):
    """Scorer factory: returns the model's mean prediction."""
    del name
    return lambda est, X, y: float(np.mean(est.predict(X)))


def _make_quality_space(good: float = 0.9, bad: float = 0.05):
    """Search space: even trials are good, odd trials are deliberately bad."""

    def space_fn(trial):
        trial.suggest_float("quality", 0.0, 1.0)
        quality = good if trial.number % 2 == 0 else bad
        return {"quality": quality}

    return space_fn


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((120, 5)))
    y = pd.Series(rng.random(120))
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests of the helpers
# ──────────────────────────────────────────────────────────────────────────────
def test_normalize_pruning_config_disabled():
    """enable=False or None → no pruner configuration."""
    assert _normalize_pruning_config(None) is None
    assert _normalize_pruning_config({}) is None
    assert _normalize_pruning_config({"enable": False}) is None


def test_normalize_pruning_config_defaults():
    """Missing fields are filled with default values."""
    cfg = _normalize_pruning_config({"enable": True})
    assert cfg == {
        "strategy": "median",
        "min_steps": 1,
        "n_startup_trials": 5,
        "reduction_factor": 3,
    }


def test_build_pruner_median():
    pruner = _build_pruner({"strategy": "median", "min_steps": 2, "n_startup_trials": 3})
    assert isinstance(pruner, MedianPruner)
    assert pruner._n_warmup_steps == 2
    assert pruner._n_startup_trials == 3


def test_build_pruner_hyperband():
    pruner = _build_pruner(
        {"strategy": "hyperband", "min_steps": 1, "reduction_factor": 4}
    )
    assert isinstance(pruner, HyperbandPruner)
    assert pruner._min_resource == 1
    assert pruner._reduction_factor == 4


def test_build_pruner_unknown_strategy_raises():
    with pytest.raises(HyperoptError, match="Unknown pruning strategy"):
        _build_pruner({"strategy": "quantile", "min_steps": 1})


def test_build_pruner_invalid_min_steps_raises():
    with pytest.raises(HyperoptError, match="min_steps"):
        _build_pruner({"strategy": "median", "min_steps": 0})


def test_build_pruner_invalid_n_startup_trials_raises():
    with pytest.raises(HyperoptError, match="n_startup_trials"):
        _build_pruner({"strategy": "median", "min_steps": 1, "n_startup_trials": 0})


def test_build_pruner_invalid_reduction_factor_raises():
    with pytest.raises(HyperoptError, match="reduction_factor"):
        _build_pruner({"strategy": "hyperband", "min_steps": 1, "reduction_factor": 1})


def test_evaluate_with_intermediate_reports_publishes_and_prunes():
    """Intermediate results are published; should_prune=True → TrialPruned."""
    trial = MagicMock()
    trial.number = 0
    trial.report = MagicMock()
    trial.should_prune = MagicMock(side_effect=[False, True])

    model = FakeQualityModel(quality=0.7)
    scorer = lambda est, X, y: float(np.mean(est.predict(X)))

    X = pd.DataFrame(np.random.rand(30, 3))
    y = pd.Series(np.random.rand(30))

    with pytest.raises(optuna.TrialPruned):
        _evaluate_with_intermediate_reports(
            trial,
            model,
            X,
            y,
            method="k_fold",
            n_folds=3,
            test_size=0.2,
            random_state=42,
            scorer=scorer,
        )

    # The cumulative mean is published on each fold (steps starting from 1).
    assert trial.report.call_count == 2
    steps = [call.kwargs["step"] for call in trial.report.call_args_list]
    assert steps == [1, 2]
    values = [call.args[0] for call in trial.report.call_args_list]
    assert all(v == pytest.approx(0.7) for v in values)


def test_evaluate_with_intermediate_reports_returns_mean():
    """Without pruning, the mean over all folds is returned."""
    trial = MagicMock()
    trial.number = 0
    trial.should_prune = MagicMock(return_value=False)

    model = FakeQualityModel(quality=0.7)
    scorer = lambda est, X, y: float(np.mean(est.predict(X)))

    X = pd.DataFrame(np.random.rand(30, 3))
    y = pd.Series(np.random.rand(30))

    score = _evaluate_with_intermediate_reports(
        trial,
        model,
        X,
        y,
        method="k_fold",
        n_folds=3,
        test_size=0.2,
        random_state=42,
        scorer=scorer,
    )
    assert score == pytest.approx(0.7)
    assert trial.report.call_count == 3


def test_evaluate_pooled_cv_returns_pooled_value():
    """_evaluate_pooled_cv returns the metric over the whole OOF vector.

    For a non-additive metric (r2) the pooled value differs from the mean of
    the per-fold values; the per-fold scorer is NOT invoked on the common
    path (no log noise for LOO+R²).
    """
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(120, 4))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(120))

    class ConstantPred(BaseEstimator, RegressorMixin):
        def __init__(self, offset: float = 0.5):
            self.offset = offset

        def fit(self, Xtr, ytr):
            return self

        def predict(self, Xte):
            return np.full(len(Xte), float(np.mean(y)) + self.offset)

    scorer = get_scorer_object("r2")

    class CountingScorer:
        """Proxy exposing the raw scorer attributes; counts __call__ invocations."""

        def __init__(self, inner):
            self.__dict__["_inner"] = inner
            self.calls = 0

        def __call__(self, estimator, Xte, yte):
            self.calls += 1
            return self._inner(estimator, Xte, yte)

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def __setattr__(self, key, value):
            if key == "calls":
                object.__setattr__(self, key, value)
            else:
                setattr(self._inner, key, value)

    counting = CountingScorer(scorer)
    pooled = _evaluate_pooled_cv(
        ConstantPred(),
        X,
        y,
        method="k_fold",
        n_folds=4,
        test_size=0.2,
        random_state=42,
        scorer=counting,
    )
    # The per-fold scorer is never invoked on the common path (lazy fallback).
    assert counting.calls == 0
    expected = float(scorer._sign * scorer._score_func(y.to_numpy(), np.full(len(y), float(np.mean(y)) + 0.5)))
    assert pooled == pytest.approx(expected, rel=1e-6)


def test_evaluate_pooled_cv_masks_non_finite_predictions():
    """_evaluate_pooled_cv masks non-finite predictions of individual folds."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 3))
    y = pd.Series(rng.randn(40))

    class NaNOnFirstFold(BaseEstimator, RegressorMixin):
        # Class-level counter: it survives clone() (which rebuilds fresh
        # instances per fold), so only the first fold's predictions are NaN.
        _folds_done = 0

        def __init__(self):
            pass

        def fit(self, Xtr, ytr):
            return self

        def predict(self, Xte):
            NaNOnFirstFold._folds_done += 1
            if NaNOnFirstFold._folds_done == 1:
                return np.full(len(Xte), np.nan)
            return np.full(len(Xte), 0.0)

    scorer = get_scorer_object("rmse")
    pooled = _evaluate_pooled_cv(
        NaNOnFirstFold(),
        X,
        y,
        method="k_fold",
        n_folds=4,
        test_size=0.2,
        random_state=42,
        scorer=scorer,
    )
    # The first fold is masked out; the pooled RMSE is over the remaining rows.
    assert np.isfinite(pooled)
    # Recompute: the surviving rows are those whose fold prediction is 0.0.
    # Re-run the splits to identify the first fold's rows.
    first_fold_rows = set()
    for i, (X_tr, X_te, y_tr, y_te, test_idx) in enumerate(
        tuner.iter_splits(
            X, y, method="k_fold", n_folds=4, test_size=0.2, random_state=42,
            include_indices=True,
        )
    ):
        if i == 0:
            first_fold_rows = set(np.asarray(test_idx))
            break
    valid = ~np.asarray([i in first_fold_rows for i in range(len(y))])
    expected = -float(np.sqrt(np.mean(y.to_numpy()[valid] ** 2)))
    assert pooled == pytest.approx(expected, rel=1e-6)


def test_evaluate_pooled_cv_fallback_on_unpoolable_scorer():
    """_evaluate_pooled_cv falls back to the per-fold mean when the scorer
    cannot be pooled (no raw metric function)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 3))
    y = pd.Series(rng.randn(40))

    class FakeModel(BaseEstimator, RegressorMixin):
        def fit(self, Xtr, ytr):
            return self

        def predict(self, Xte):
            return np.full(len(Xte), 0.7)

    class LambdaScorer:
        """A scorer without _score_func: pooling is impossible."""

        def __call__(self, estimator, Xte, yte):
            return 0.7

    pooled = _evaluate_pooled_cv(
        FakeModel(),
        X,
        y,
        method="k_fold",
        n_folds=4,
        test_size=0.2,
        random_state=42,
        scorer=LambdaScorer(),
    )
    assert pooled == pytest.approx(0.7)


def test_evaluate_with_intermediate_reports_pooled_masks_nan_steps():
    """LOO + r2 + pruning: NaN intermediate values are NOT published.

    On the first LOO fold the pooled r2 over a single pair is NaN (r2 is
    undefined on <2 samples). Such steps must be skipped in trial.report —
    Optuna pruners cannot compare NaN. Once the accumulated rows admit a
    finite pooled r2 (from the 2nd pair on), the steps are published.
    """
    trial = MagicMock()
    trial.number = 0
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=False)

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(20, 3))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(20))

    class LinearModel(BaseEstimator, RegressorMixin):
        def fit(self, Xtr, ytr):
            import numpy.linalg as la

            Xt = np.column_stack([np.ones(len(Xtr)), np.asarray(Xtr)])
            self.coef_, _, _, _ = la.lstsq(Xt, np.asarray(ytr), rcond=None)
            return self

        def predict(self, Xte):
            return np.column_stack([np.ones(len(Xte)), np.asarray(Xte)]) @ self.coef_

    scorer = get_scorer_object("r2")
    _evaluate_with_intermediate_reports(
        trial,
        LinearModel(),
        X,
        y,
        method="loo",
        n_folds=5,
        test_size=0.2,
        random_state=42,
        scorer=scorer,
        pool_metric=True,
    )
    # The first 1-row LOO step has an undefined pooled r2 (NaN) and must NOT
    # be published; the remaining steps (>= 2 accumulated pairs) are finite.
    steps = [call.kwargs["step"] for call in trial.report.call_args_list]
    assert 1 not in steps
    assert len(steps) == 19  # 20 LOO folds minus the first NaN step
    assert steps == list(range(2, 21))
    values = [call.args[0] for call in trial.report.call_args_list]
    assert all(np.isfinite(v) for v in values)
    # should_prune is still consulted on every fold.
    assert trial.should_prune.call_count == 20


def test_evaluate_with_intermediate_reports_pooled_publishes_finite_steps():
    """k_fold + r2 + pruning: finite pooled steps are published per fold."""
    trial = MagicMock()
    trial.number = 0
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=False)

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(120, 4))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(120))

    class LinearModel(BaseEstimator, RegressorMixin):
        def fit(self, Xtr, ytr):
            import numpy.linalg as la

            Xt = np.column_stack([np.ones(len(Xtr)), np.asarray(Xtr)])
            self.coef_, _, _, _ = la.lstsq(Xt, np.asarray(ytr), rcond=None)
            return self

        def predict(self, Xte):
            return np.column_stack([np.ones(len(Xte)), np.asarray(Xte)]) @ self.coef_

    scorer = get_scorer_object("r2")
    _evaluate_with_intermediate_reports(
        trial,
        LinearModel(),
        X,
        y,
        method="k_fold",
        n_folds=4,
        test_size=0.2,
        random_state=42,
        scorer=scorer,
        pool_metric=True,
    )
    # 4 folds → 4 published steps (all finite for k_fold).
    assert trial.report.call_count == 4
    steps = [call.kwargs["step"] for call in trial.report.call_args_list]
    assert steps == [1, 2, 3, 4]
    values = [call.args[0] for call in trial.report.call_args_list]
    assert all(np.isfinite(v) for v in values)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end tests of optimize() with real Optuna pruners
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def quality_patches():
    """Patches are active for the duration of the test: a deterministic model and scorer."""
    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
    ):
        yield


def test_optimize_pruning_median_prunes_bad_trials(
    toy_data, quality_patches, caplog
):
    """MedianPruner prunes the deliberately bad trials before all folds complete.

    Even trials are good (quality=0.9), odd trials are deliberately bad (0.05).
    After the first completed trial, every bad trial is pruned on the very first
    fold. The best result is selected only from completed trials, and the fact
    of pruning is recorded in the logs.
    """
    X, y = toy_data

    with caplog.at_level(logging.INFO):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=4,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )
        assert opt_res is not None
        model, params, score = opt_res.model, opt_res.params, opt_res.score


    # The best trial completed fully and has maximum quality.
    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None

    # The fact and the count of prunings are recorded in the logs.
    assert "Early stopping: pruned 2 of 4 trials" in caplog.text
    assert "Early stopping enabled: strategy=median" in caplog.text


def test_optimize_pruning_hyperband_prunes_bad_trials(
    toy_data, quality_patches, caplog
):
    """HyperbandPruner also prunes some of the trials."""
    X, y = toy_data

    with caplog.at_level(logging.INFO):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=8,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "hyperband",
                "min_steps": 1,
                "reduction_factor": 2,
            },
        )
        assert opt_res is not None
        model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None
    assert "Early stopping: pruned" in caplog.text
    assert "strategy=hyperband" in caplog.text


def test_optimize_pruning_all_trials_pruned_returns_no_results(
    toy_data, quality_patches, mocker, caplog
):
    """All trials are pruned → the "no results" scenario.

    Previously optimize() returned a "magic" constant -3.4028235e38, which
    looked like a valid result to the orchestrator. Now a complete failure is
    signaled by returning None throughout.

    In addition, the decision is made by explicitly counting the trial states
    (n_completed == 0) rather than by catching a ValueError from
    study.best_params; the state counters are logged for observability.
    """
    X, y = toy_data

    # Force every trial to be pruned at the very first step.
    mocker.patch.object(
        tuner.optuna.trial.Trial, "should_prune", return_value=True
    )

    with caplog.at_level(logging.INFO):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=4,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )
        assert opt_res is None
    # The state-counter log: 4 pruned, 0 completed, 0 failed.
    assert "Trial states for algorithm 'ridge'" in caplog.text
    assert "completed=0, pruned=4, failed=0" in caplog.text
    assert "no completed trials" in caplog.text


def test_optimize_pruning_single_trial_pruned_returns_no_results(
    toy_data, quality_patches, mocker
):
    """n_trials=1 and the single trial is pruned → (None, None, None).

    The "all pruned" edge case with the minimum possible number of trials.
    """
    X, y = toy_data

    mocker.patch.object(
        tuner.optuna.trial.Trial, "should_prune", return_value=True
    )

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=1,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )
    assert opt_res is None


def _make_study_with_states(
    states: list[optuna.trial.TrialState],
) -> optuna.Study:
    """Build a real Optuna study with trials in the given states.

    The ``catch`` option in ``study.optimize`` is not used by the tuner, so
    "live" FAIL trials (an exception from the objective) propagate outward and
    interrupt the run. To verify the explicit state counting, we fill a real
    study with pre-created trials and replace ``study.optimize`` with a no-op —
    get_trials() will return exactly these states.
    """
    study = optuna.create_study(direction="maximize")
    for state in states:
        study.add_trial(
            optuna.trial.create_trial(state=state, params={}, distributions={})
        )
    return study


def test_optimize_all_trials_failed_returns_no_results(
    toy_data, quality_patches, mocker
):
    """All trials FAILED → (None, None, None).

    Trials in the FAILED state appear in Optuna when the objective raises an
    exception that was not converted into optuna.TrialPruned. The explicit
    state counting (n_completed == 0) must disqualify the algorithm in this
    case too, without relying on a ValueError from study.best_params.
    """
    X, y = toy_data
    study = _make_study_with_states([optuna.trial.TrialState.FAIL] * 3)
    mocker.patch.object(tuner.optuna, "create_study", return_value=study)
    mocker.patch.object(study, "optimize")

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=3,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )
    assert opt_res is None


def test_optimize_pruned_and_failed_mix_returns_no_results(
    toy_data, quality_patches, mocker
):
    """A mix of PRUNED + FAILED without COMPLETED → (None, None, None).

    There are no completed trials at all — the algorithm is disqualified even
    if some trials were pruned by the pruner and some failed.
    """
    X, y = toy_data
    study = _make_study_with_states(
        [
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ]
    )
    mocker.patch.object(tuner.optuna, "create_study", return_value=study)
    mocker.patch.object(study, "optimize")

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )
    assert opt_res is None


def test_optimize_pruning_nonfinite_score_uses_worst_score(
    toy_data, quality_patches
):
    """A non-finite avg_score in the pruning branch → the trial gets HPO_WORST_SCORE.

    Covers the ``if not np.isfinite(avg_score)`` branch inside the
    ``_evaluate_with_intermediate_reports`` path: all trials complete with the
    "worst score" but are not pruned, so optimize() returns HPO_WORST_SCORE
    with valid parameters — and it is the valid_results filter in the
    orchestrator that discards such a result.
    """
    X, y = toy_data

    def fake_evaluate(
        trial,
        estimator,
        X,
        y,
        *,
        method,
        n_folds,
        test_size,
        random_state,
        scorer,
        pool_metric=False,
    ):
        del trial, estimator, X, y, method, n_folds, test_size, random_state, scorer, pool_metric
        return np.nan

    with patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )
        assert opt_res is not None
        model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == HPO_WORST_SCORE
    assert params is not None
    assert model is not None


def test_optimize_pruning_not_applied_for_train_test_split(
    toy_data, quality_patches, caplog
):
    """Hold-out without steps: the pruner is not applied, all trials complete."""
    X, y = toy_data

    with caplog.at_level(logging.WARNING):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=4,
            validation_strategy="train_test_split",
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )
        assert opt_res is not None
        _model, params, score = opt_res.model, opt_res.params, opt_res.score


    # All trials completed — the best one was found by the usual maximization rule.
    assert score == pytest.approx(0.9)
    assert params is not None
    assert "has no intermediate steps — pruning will not be applied" in caplog.text


def test_optimize_pruning_loo_works(toy_data, quality_patches):
    """LOO, as a strategy with natural steps, works with pruning."""
    X, y = toy_data

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="loo",
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None


def test_optimize_pruning_with_initial_params_refine(
    toy_data, quality_patches
):
    """refine (initial_params + enqueue_trial) is compatible with pruning."""
    X, y = toy_data

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=3,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        initial_params={"quality": 0.9},
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )
    assert opt_res is not None
    _model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == pytest.approx(0.9)
    assert params is not None


def test_optimize_pruning_two_folds_edge_case(toy_data, quality_patches):
    """A small number of folds (2) works with pruning (edge scenario)."""
    X, y = toy_data

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="k_fold",
        n_folds=2,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == pytest.approx(0.9)
    assert params is not None
    assert model is not None


# ──────────────────────────────────────────────────────────────────────────────
# Backward compatibility
# ──────────────────────────────────────────────────────────────────────────────
def test_optimize_without_pruning_never_reports(toy_data):
    """Without the pruning block, no intermediate results are published — the
    behavior is identical to the current one (every trial runs to completion)."""
    X, y = toy_data

    # Mock the environment so we reliably hit the cross_val_score branch
    # (as in the old code), and control the trial.report calls.
    report_mock = MagicMock()
    with (
        patch.object(
            tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)
        ),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(
            tuner.model_selection,
            "cross_val_score",
            lambda est, Xt, yt, cv=None, scoring=None, n_jobs=1: [0.5, 0.5, 0.5],
        ),
        patch.object(tuner.optuna.trial.Trial, "report", report_mock),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
        )

    report_mock.assert_not_called()


def test_optimize_disabled_pruning_uses_nop_pruner(toy_data):
    """With pruning disabled the study is created with NopPruner — full trial
    execution is guaranteed even with random report() calls."""
    X, y = toy_data

    with (
        patch.object(
            tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)
        ),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(
            tuner.optuna, "create_study", wraps=tuner.optuna.create_study
        ) as mock_create_study,
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
        )

    assert isinstance(mock_create_study.call_args.kwargs["pruner"], NopPruner)


def test_optimize_enabled_pruning_uses_configured_pruner(toy_data):
    """With pruning enabled, the configured pruner is attached to the study."""
    X, y = toy_data

    with (
        patch.object(
            tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)
        ),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(
            tuner.optuna, "create_study", wraps=tuner.optuna.create_study
        ) as mock_create_study,
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    pruner = mock_create_study.call_args.kwargs["pruner"]
    assert isinstance(pruner, MedianPruner)
    assert pruner._n_warmup_steps == 1


def test_optimize_pruned_trials_not_fatal(toy_data, quality_patches, caplog):
    """Pruned trials do not increase the fatal-failure counter.

    Many consecutively pruned trials do not disqualify the algorithm.
    """
    X, y = toy_data
    n_trials = 12

    opt_res = optimize(
        "ridge",
        X,
        y,
        n_trials=n_trials,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 1,
        },
    )
    assert opt_res is not None
    _model, _params, score = opt_res.model, opt_res.params, opt_res.score


    # The optimization completed without InvalidAlgorithmError.
    assert score == pytest.approx(0.9)
    assert "disqualified" not in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end scenario through training_engine (config -> train_best_model)
# ──────────────────────────────────────────────────────────────────────────────
def test_train_best_model_with_pruning_e2e(tmp_path: Path) -> None:
    """End-to-end scenario: a config with general.pruning.enable=true goes
    through train_best_model without errors."""
    from pathlib import Path as _Path

    from sklearn.datasets import make_regression

    from configurable_automl_engine.training_engine import train_best_model

    model_path = tmp_path / "models" / "best_model.pkl"
    config = {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "validation_strategy": "k_fold",
            "n_folds": 2,
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "pruning": {
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    X, y = make_regression(
        n_samples=120, n_features=5, noise=0.1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    result = train_best_model(config=config, df=df, target="target")

    assert result["algorithm"] == "ridge"
    assert isinstance(result["score"], float)
    assert result["params"]
    assert _Path(result["model_path"]).exists()


# ──────────────────────────────────────────────────────────────────────────────
# auto + pruning: fold-count synchronization
# ──────────────────────────────────────────────────────────────────────────────
def test_optimize_auto_kfold_pruning_passes_resolved_k():
    """auto→kfold (k=10) + pruning: the resolved k is passed to
    _evaluate_with_intermediate_reports, not the original n_folds.

    This test fails on the old code: before the fix, the early-stopping branch
    received n_folds=5 (the default) instead of auto_decision['k']=10, and
    iter_splits fell back to the default fold count (desynchronized from
    cross_val_score(cv=cv_obj)).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((400, 5)))
    y = pd.Series(rng.random(400))

    captured = {}

    def fake_make_cv(
        n_samples, *, val_method, n_folds, random_state, test_size, n_features=None
    ):
        # auto resolves to kfold with k=10 (the decision returned by make_cv).
        return (
            "k_fold",
            KFold(n_splits=10, shuffle=True, random_state=random_state),
            {"method": "kfold", "k": 10, "average_test_size": 40.0},
        )

    def fake_evaluate(
        trial,
        estimator,
        X,
        y,
        *,
        method,
        n_folds,
        test_size,
        random_state,
        scorer,
        pool_metric=False,
    ):
        captured["method"] = method
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "make_cv", fake_make_cv),
        patch.object(tuner, "get_effective_train_size", lambda *a, **k: 360),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert captured["method"] == "k_fold"
    assert captured["n_folds"] == 10


def test_optimize_auto_kfold_pruning_integration_seven_reports():
    """Integration: N=150/P=5 → auto resolves to k=7; the pruning branch runs
    exactly 7 folds and publishes 7 intermediate steps.

    On the old code there would be 5 steps (the default n_folds), i.e. the
    score would diverge from the non-pruning cross_val_score(cv=KFold(7))
    branch.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((150, 5)))
    y = pd.Series(rng.random(150))

    # The heuristic for this data must select k-fold with k=7.
    from configurable_automl_engine.common.validation_utils import (
        choose_validation_method,
    )

    decision = choose_validation_method(150, 5)
    assert decision == {"method": "kfold", "k": 7, "average_test_size": 21.4}

    report_steps: list[int] = []

    def fake_report(self, value, step):
        report_steps.append(step)

    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner.optuna.trial.Trial, "report", fake_report),
    ):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 5,
            },
        )
        assert opt_res is not None
        _model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == pytest.approx(0.9)
    assert params is not None
    # Exactly k=7 intermediate reports (steps 1..7) — one per fold.
    assert report_steps == list(range(1, 8))


def test_optimize_auto_pruning_parity_with_cross_val():
    """Parity: auto→kfold uses the same fold count (k) with and without
    pruning — the scores do not diverge between the branches."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((150, 5)))  # auto -> k=7
    y = pd.Series(rng.random(150))

    iter_splits_calls: list[tuple[str | None, int | None]] = []
    real_iter_splits = tuner.iter_splits

    def spy_iter_splits(*args, **kwargs):
        iter_splits_calls.append((kwargs.get("method"), kwargs.get("n_folds")))
        yield from real_iter_splits(*args, **kwargs)

    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "iter_splits", spy_iter_splits),
    ):
        # Without pruning: evaluation via iter_splits(method='k_fold',
        # n_folds=k) in the pooled path (r2 is non-additive).
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
        )
        # With pruning: evaluation via iter_splits(method='k_fold', n_folds=k).
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 5,
            },
        )

    # Both branches evaluate on the same folds (k=7): the scores do not
    # diverge between the pruning and non-pruning paths.
    assert iter_splits_calls == [("k_fold", 7), ("k_fold", 7)]


def test_optimize_auto_no_features_fallback_clamps_n_folds():
    """Negative scenario: auto without P (fallback, decision=None) with n_folds=1.

    make_cv clamps k = max(2, n_folds) via resolve_auto_no_features_fallback;
    the pruning branch must repeat the same clamping, otherwise
    KFold(n_splits=1) inside iter_splits would crash (a latent failure).
    """
    # Zero features — P is unavailable, the make_cv fallback branch kicks in.
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((400, 0)))
    y = pd.Series(rng.random(400))

    captured = {}

    def fake_evaluate(
        trial,
        estimator,
        X,
        y,
        *,
        method,
        n_folds,
        test_size,
        random_state,
        scorer,
        pool_metric=False,
    ):
        captured["method"] = method
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=1,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    # Fallback: method='k_fold' (N is enough), the fold count is clamped to 2.
    assert captured["method"] == "k_fold"
    assert captured["n_folds"] == 2


def test_optimize_auto_kfold_low_confidence_k2():
    """Edge scenario: auto→kfold with k=2 (low confidence) — k is passed
    as is (the max(2, k) clamping does not change the value)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((31, 16)))  # auto -> kfold k=2
    y = pd.Series(rng.random(31))

    captured = {}

    def fake_make_cv(
        n_samples, *, val_method, n_folds, random_state, test_size, n_features=None
    ):
        return (
            "k_fold",
            KFold(n_splits=2, shuffle=True, random_state=random_state),
            {"method": "kfold", "k": 2, "average_test_size": 15.5},
        )

    def fake_evaluate(
        trial,
        estimator,
        X,
        y,
        *,
        method,
        n_folds,
        test_size,
        random_state,
        scorer,
        pool_metric=False,
    ):
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "make_cv", fake_make_cv),
        patch.object(tuner, "get_effective_train_size", lambda *a, **k: 16),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert captured["n_folds"] == 2


def test_optimize_explicit_kfold_pruning_keeps_original_n_folds():
    """Explicit k_fold without regression: the pruning branch receives the
    original n_folds unchanged (no effective clamping is applied)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((120, 5)))
    y = pd.Series(rng.random(120))

    captured = {}

    def fake_evaluate(
        trial,
        estimator,
        X,
        y,
        *,
        method,
        n_folds,
        test_size,
        random_state,
        scorer,
        pool_metric=False,
    ):
        captured["method"] = method
        captured["n_folds"] = n_folds
        return 0.9

    with (
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_evaluate_with_intermediate_reports", fake_evaluate),
    ):
        optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )

    assert captured["method"] == "k_fold"
    assert captured["n_folds"] == 3


def test_optimize_auto_loo_with_pruning_works():
    """auto→LOO (N=10/P=3) + pruning: a strategy with natural steps works;
    the fold count for k-fold is not used."""
    X = pd.DataFrame(np.random.rand(10, 3))
    y = pd.Series(np.random.rand(10))

    report_steps: list[int] = []

    def fake_report(self, value, step):
        report_steps.append(step)

    with (
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
        patch.object(tuner.optuna.trial.Trial, "report", fake_report),
    ):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 5,
            },
        )
        assert opt_res is not None
        _model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == pytest.approx(0.9)
    assert params is not None
    # LOO: one step per object (10 objects → 10 reports).
    assert report_steps == list(range(1, 11))


def test_optimize_auto_train_test_split_pruning_not_applied(caplog):
    """auto→train_test_split: there are no natural steps — the pruner is not
    applied and all trials run to completion (documented behavior).

    auto resolves to train_test_split only on large data (N >= 50000).
    To avoid building a 50k-row DataFrame, make_cv is patched with the
    train_test_split decision (the heuristic itself is covered in
    test_validation_auto).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((1000, 5)))
    y = pd.Series(rng.random(1000))

    def fake_make_cv(
        n_samples, *, val_method, n_folds, random_state, test_size, n_features=None
    ):
        # The real heuristic would make this decision at N >= 50000.
        return (
            "train_test_split",
            None,
            {
                "method": "train_test_split",
                "test_size": 500,
                "train_size": 49500,
            },
        )

    with (
        caplog.at_level(logging.WARNING),
        patch.object(tuner, "make_cv", fake_make_cv),
        patch.object(tuner, "get_effective_train_size", lambda *a, **k: 49500),
        patch.object(tuner, "create_model", lambda algo, **kw: FakeQualityModel(**kw)),
        patch.object(tuner, "_build_scorer", _quality_scorer_factory),
    ):
        opt_res = optimize(
            "ridge",
            X,
            y,
            n_trials=1,
            validation_strategy="auto",
            n_folds=5,
            random_state=42,
            space_overrides={"ridge": _make_quality_space()},
            pruning={
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        )
        assert opt_res is not None
        _model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert score == pytest.approx(0.9)
    assert params is not None
    assert "has no intermediate steps — pruning will not be applied" in caplog.text


def test_train_best_model_auto_pruning_e2e(tmp_path: Path) -> None:
    """End-to-end scenario: validation_strategy='auto' + pruning.enable=true
    goes through train_best_model; the fold count comes from the auto decision
    (N=150/P=5 → k=7), not from the default n_folds=5."""
    from pathlib import Path as _Path

    from sklearn.datasets import make_regression

    from configurable_automl_engine.training_engine import train_best_model

    model_path = tmp_path / "models" / "best_model.pkl"
    config = {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "validation_strategy": "auto",
            "n_folds": 2,
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "pruning": {
                "enable": True,
                "strategy": "median",
                "min_steps": 1,
                "n_startup_trials": 1,
            },
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    X, y = make_regression(
        n_samples=150, n_features=5, noise=0.1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    result = train_best_model(config=config, df=df, target="target")

    assert result["algorithm"] == "ridge"
    assert isinstance(result["score"], float)
    assert result["params"]
    assert _Path(result["model_path"]).exists()


def test_optimize_pruning_with_transferred_history(toy_data, quality_patches):
    """Pruning + transferred TPE history interplay: the phase runs with the
    pruner on the transferred study, keeps the transferred trials as warm-start
    history, and returns ONLY its own (post-transfer) trials."""
    X, y = toy_data

    first = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=42,
        space_overrides={"ridge": _make_quality_space()},
    )
    assert first is not None
    assert len(first.trials) == 4

    second = optimize(
        "ridge",
        X,
        y,
        n_trials=4,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=43,
        space_overrides={"ridge": _make_quality_space()},
        pruning={
            "enable": True,
            "strategy": "median",
            "min_steps": 1,
            "n_startup_trials": 2,
        },
        previous_trials=first.trials,
    )
    assert second is not None
    # Only the current phase's own trials (numbers >= the transferred count).
    assert len(second.trials) == 4
    assert all(t.number >= 4 for t in second.trials)
