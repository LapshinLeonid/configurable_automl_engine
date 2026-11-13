"""Repeated (multi-split) trial scoring of the HPO runner.

The plan carries the deterministic repeat seeds and the aggregation policy;
``score_trial`` aggregates the per-repeat scores (median by default) while a
single non-finite repeat is penalized instead of failing the trial. The OOF
vectors of the repeats are stacked along the configuration axis for BBC-CV.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from optuna.trial import FixedTrial, FrozenTrial, TrialState
from sklearn import model_selection
from sklearn.linear_model import Ridge

from configurable_automl_engine.common.metrics import bbc_cv_score, get_scorer_object
from configurable_automl_engine.hpo import evaluation
from configurable_automl_engine.hpo.evaluation import _phase_oof_matrix, score_trial
from configurable_automl_engine.hpo.plans import ValidationPlan
from configurable_automl_engine.hpo.repeats import run_repeat_scores
from configurable_automl_engine.hpo.request import HpoRequest
from configurable_automl_engine.hpo.runner import plan_validation
from configurable_automl_engine.hpo.types import (
    HPO_WORST_SCORE,
    WORST_SCORE_THRESHOLD,
)


def _data(n: int = 48) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2.0 * X["a"] - X["b"] + rng.normal(scale=0.05, size=n), name="y")
    return X, y


def _plan(**kwargs) -> ValidationPlan:
    X, y = _data()
    return plan_validation(X, y, HpoRequest(algo_name="ridge", **kwargs))


def test_plan_validation_builds_repeat_seeds() -> None:
    plan = _plan(val_method="k_fold", n_folds=4, n_repeats=3, random_state=7)

    assert plan.n_repeats == 3
    assert plan.repeat_seeds == (7, 8, 9)
    assert plan.split_aggregation == "median"


def test_plan_validation_loo_ignores_repeats(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        plan = _plan(val_method="loo", n_repeats=3, random_state=7)

    assert plan.n_repeats == 1
    assert plan.repeat_seeds == (7,)
    assert "n_repeats=3 is meaningless" in caplog.text


def test_plan_validation_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="Unknown split_aggregation"):
        _plan(val_method="k_fold", split_aggregation="sum")


def test_engine_request_passes_repeat_settings() -> None:
    from configurable_automl_engine.training_engine.config_parser import AlgoCfg
    from configurable_automl_engine.training_engine.hpo_phases import (
        _build_hpo_request,
    )
    from tests.hpo_spec_factory import make_phase_spec

    spec = make_phase_spec(n_repeats=4, split_aggregation="mean")
    request = _build_hpo_request("ridge", AlgoCfg(), spec)

    assert request.n_repeats == 4
    assert request.split_aggregation == "mean"


def test_score_trial_keeps_one_oof_vector_per_repeat() -> None:
    X, y = _data()
    plan = _plan(val_method="k_fold", n_folds=4, n_repeats=3, random_state=11)

    result = score_trial(
        Ridge(),
        X,
        y,
        trial=FixedTrial({}),
        validation=plan,
        scorer=get_scorer_object("r2"),
        pruning_active=False,
        pool_metric=False,
        random_state=plan.repeat_seeds[0],
        collect_oof=True,
    )

    assert np.isfinite(result.value)
    assert result.oof is not None
    assert result.oof.shape == (3, len(X))
    assert np.isfinite(result.oof).all()
    assert len(result.repeat_scores) == 3
    assert result.repeat_mean is not None
    assert result.repeat_std is not None


def test_score_trial_is_deterministic_for_repeats() -> None:
    X, y = _data()
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=2, random_state=5)
    kwargs = {
        "trial": FixedTrial({}),
        "validation": plan,
        "scorer": get_scorer_object("neg_mean_absolute_error"),
        "pruning_active": False,
        "pool_metric": False,
        "random_state": 5,
        "collect_oof": True,
    }
    first = score_trial(Ridge(alpha=0.7), X, y, **kwargs)
    second = score_trial(Ridge(alpha=0.7), X, y, **kwargs)

    assert first.value == second.value
    np.testing.assert_array_equal(first.oof, second.oof)


def test_run_repeat_scores_median_and_mean() -> None:
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=3)
    values = {0: 0.1, 1: 0.9, 2: 0.5}

    common = {
        "validation": plan,
        "n_repeats": 3,
        "random_state": 11,
        "oof_out": None,
        "all_failed_message": "all failed",
    }
    median = run_repeat_scores(
        lambda r, _seed, _oof: values[r],
        split_aggregation="median",
        **common,
    )
    mean = run_repeat_scores(
        lambda r, _seed, _oof: values[r],
        split_aggregation="mean",
        **common,
    )

    assert median == pytest.approx(0.5)
    assert mean == pytest.approx(0.5)


def test_non_finite_repeat_is_penalized_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=3)
    values = {0: 0.8, 1: float("nan"), 2: 0.6}

    mask: list[bool] = []
    with caplog.at_level(logging.WARNING):
        value = run_repeat_scores(
            lambda r, _seed, _oof: values[r],
            validation=plan,
            n_repeats=3,
            random_state=11,
            oof_out=None,
            split_aggregation="median",
            all_failed_message="all failed",
            mask_out=mask,
        )

    assert value == pytest.approx(0.6)
    assert mask == [True, False, True]
    assert "non-finite score" in caplog.text


def test_all_non_finite_repeats_give_worst_score() -> None:
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=2)
    value = run_repeat_scores(
        lambda _r, _seed, _oof: float("nan"),
        validation=plan,
        n_repeats=2,
        random_state=11,
        oof_out=None,
        split_aggregation="median",
        all_failed_message="all failed",
    )

    assert value == HPO_WORST_SCORE


def test_one_failed_repeat_is_penalized_all_failed_raise() -> None:
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=2)

    def fail_on_first(repeat_index: int, _seed, _oof) -> float:
        if repeat_index == 0:
            raise ValueError("all folds failed")
        return 0.4

    mask: list[bool] = []
    scores: list[float] = []
    value = run_repeat_scores(
        fail_on_first,
        validation=plan,
        n_repeats=2,
        random_state=11,
        oof_out=None,
        split_aggregation="median",
        all_failed_message="all failed",
        scores_out=scores,
        mask_out=mask,
    )
    assert value != HPO_WORST_SCORE
    assert mask == [False, True]
    assert scores[1] == pytest.approx(0.4)

    def always_fail(_repeat_index: int, _seed, _oof) -> float:
        raise ValueError("all folds failed")

    with pytest.raises(ValueError, match="all failed"):
        run_repeat_scores(
            always_fail,
            validation=plan,
            n_repeats=2,
            random_state=11,
            oof_out=None,
            split_aggregation="median",
            all_failed_message="all failed",
        )


def _frozen(number: int) -> FrozenTrial:
    return FrozenTrial(
        number=number,
        state=TrialState.COMPLETE,
        value=1.0,
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
    )


def test_phase_oof_matrix_stacks_repeats() -> None:
    trials = [_frozen(0), _frozen(1)]
    oof = {
        0: np.array([[1.0, 2.0], [3.0, 4.0]]),
        1: np.array([5.0, 6.0]),
    }

    matrix = _phase_oof_matrix(trials, oof)

    np.testing.assert_array_equal(
        matrix,
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )


def test_phase_oof_matrix_enforces_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "BBC_MAX_OOF_CELLS", 3)
    trials = [_frozen(0)]
    oof = {0: np.array([[1.0, 2.0], [3.0, 4.0]])}

    assert _phase_oof_matrix(trials, oof) is None


def test_failed_repeat_oof_row_is_dropped_and_bbc_computes() -> None:
    """One failed repeat must not leave an all-NaN OOF row in BBC-CV."""
    X, y = _data(60)
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=2, random_state=5)
    real_scorer = get_scorer_object("neg_mean_absolute_error")
    calls = {"n": 0}

    def flaky_scorer(estimator, X_te, y_te):
        calls["n"] += 1
        if calls["n"] > plan.n_folds:
            return float("nan")
        return real_scorer(estimator, X_te, y_te)

    result = score_trial(
        Ridge(),
        X,
        y,
        trial=FixedTrial({}),
        validation=plan,
        scorer=flaky_scorer,
        pruning_active=False,
        pool_metric=False,
        random_state=5,
        collect_oof=True,
    )

    # The aggregate is above the worst-score threshold (one finite repeat out
    # of two), so the runner keeps the OOF matrix — it must contain only the
    # successful repeat, not the all-NaN row of the failed one.
    assert result.value > WORST_SCORE_THRESHOLD
    assert result.oof is not None
    assert result.oof.shape == (1, len(X))
    assert np.isfinite(result.oof).all()
    matrix = _phase_oof_matrix([_frozen(0)], {0: result.oof})
    assert matrix is not None
    score = bbc_cv_score(
        get_scorer_object("r2"), y.to_numpy(dtype=float), matrix, random_state=0
    )
    assert np.isfinite(score)


def test_repeated_holdout_scores_each_repeat() -> None:
    X, y = _data(60)
    plan = _plan(val_method="train_test_split", n_repeats=3, random_state=11)

    result = score_trial(
        Ridge(),
        X,
        y,
        trial=FixedTrial({}),
        validation=plan,
        scorer=get_scorer_object("neg_mean_absolute_error"),
        pruning_active=False,
        pool_metric=False,
        random_state=11,
        collect_oof=True,
    )

    assert np.isfinite(result.value)
    assert result.oof is not None and result.oof.shape == (3, len(X))
    assert len(result.repeat_scores) == 3
    assert all(np.isfinite(row).any() for row in result.oof)


def test_repeated_pooled_path_keeps_all_oof_rows() -> None:
    X, y = _data()
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=2, random_state=7)

    result = score_trial(
        Ridge(),
        X,
        y,
        trial=FixedTrial({}),
        validation=plan,
        scorer=get_scorer_object("r2"),
        pruning_active=False,
        pool_metric=True,
        random_state=7,
        collect_oof=True,
    )

    assert np.isfinite(result.value)
    assert result.oof is not None and result.oof.shape == (2, len(X))
    assert np.isfinite(result.oof).all()


def test_repeated_pruning_path_reports_steps_per_repeat() -> None:
    X, y = _data()
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=2, random_state=9)
    trial = MagicMock()
    trial.should_prune.return_value = False

    result = score_trial(
        Ridge(),
        X,
        y,
        trial=trial,
        validation=plan,
        scorer=get_scorer_object("neg_mean_absolute_error"),
        pruning_active=True,
        pool_metric=False,
        random_state=9,
        collect_oof=True,
    )

    assert np.isfinite(result.value)
    assert result.oof is not None and result.oof.shape == (2, len(X))
    steps = [call.kwargs["step"] for call in trial.report.call_args_list]
    assert steps == [1, 2, 3, 4, 5, 6]


def test_one_repeat_matches_the_classic_fold_mean() -> None:
    X, y = _data()
    plan = _plan(val_method="k_fold", n_folds=3, n_repeats=1, random_state=42)
    scorer = get_scorer_object("neg_mean_absolute_error")

    result = score_trial(
        Ridge(),
        X,
        y,
        trial=FixedTrial({}),
        validation=plan,
        scorer=scorer,
        pruning_active=False,
        pool_metric=False,
        random_state=42,
        collect_oof=False,
    )
    expected = float(
        np.mean(
            model_selection.cross_val_score(
                Ridge(), X, y, cv=plan.cv, scoring=scorer, n_jobs=1
            )
        )
    )

    assert result.value == pytest.approx(expected)


def test_repeat_diagnostics_reach_trial_user_attrs() -> None:
    from configurable_automl_engine.tuner import optimize

    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.05 * rng.normal(size=48)

    result = optimize(
        "ridge",
        X,
        y,
        n_trials=1,
        val_method="k_fold",
        n_folds=3,
        n_repeats=2,
        random_state=42,
        fit_best_model=False,
    )

    assert result is not None
    attrs = result.trials[0].user_attrs
    assert len(attrs["repeat_scores"]) == 2
    assert attrs["repeat_mean"] == pytest.approx(np.mean(attrs["repeat_scores"]))
    assert attrs["repeat_std"] == pytest.approx(np.std(attrs["repeat_scores"]))
