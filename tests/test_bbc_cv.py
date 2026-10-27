"""Tests for the BBC-CV (bootstrap bias-corrected CV) estimate.

Coverage:
    1. ``metrics._columnwise_raw_values`` — the vectorized per-column metric
       equals ``pooled_scorer_value`` for every comparison metric (including
       non-finite predictions and an all-NaN column) and for a generic scorer.
    2. ``metrics.bbc_cv_score`` — input validation, determinism, removal of
       the best-of-N selection bias on pure-noise configurations, duplicate
       and uncovered-row handling.
    3. ``tuner.optimize(collect_oof=True)`` — one OOF row per completed trial
       on every evaluation path (pooled k_fold, additive k_fold,
       train_test_split, pruning), each row reproducing its trial value.
    4. The engine: ``result["bbc_score"]`` without an outer hold-out, absent
       with a hold-out or over the memory budget, the OOF pool collected from
       every algorithm (threads parallel mode).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer, median_absolute_error

from configurable_automl_engine import tuner as hyperopt
from configurable_automl_engine.hpo import evaluation as hpo_evaluation
from configurable_automl_engine.training_engine import component
import configurable_automl_engine.training_engine.hpo_phases as hpo_phases_mod
from configurable_automl_engine.training_engine.component import (
    execute_phases,
    train_best_model,
)
from configurable_automl_engine.training_engine.data_prep import prepare_dataset
from configurable_automl_engine.training_engine.artifact import (
    _bbc_score_in_user_metric,
)
from configurable_automl_engine.training_engine.config_parser import Config
from configurable_automl_engine.training_engine.metrics import (
    _columnwise_raw_values,
    bbc_cv_score,
    get_scorer_object,
    pooled_scorer_value,
)

COMPARISON_METRICS = ["rmse", "neg_root_mean_squared_error", "mse", "mae", "r2", "nrmse"]


# ──────────────────────────────────────────────────────────────────────────────
#  1. _columnwise_raw_values
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("metric", COMPARISON_METRICS)
def test_columnwise_values_match_pooled_scorer(metric):
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    preds = y[:, None] + rng.normal(size=(50, 6))
    preds[3, 1] = np.nan
    preds[7, 4] = np.inf
    scorer = get_scorer_object(metric)
    values = _columnwise_raw_values(scorer, y, preds)
    expected = [pooled_scorer_value(scorer, y, preds[:, c]) for c in range(6)]
    np.testing.assert_allclose(values, expected, rtol=1e-12)


@pytest.mark.parametrize("metric", COMPARISON_METRICS)
def test_columnwise_values_all_nan_column_is_nan(metric):
    y = np.arange(10, dtype=float)
    preds = np.column_stack([y + 1.0, np.full(10, np.nan)])
    values = _columnwise_raw_values(get_scorer_object(metric), y, preds)
    assert np.isfinite(values[0])
    assert np.isnan(values[1])


def test_columnwise_values_constant_target_edge_cases():
    """R² on a constant target follows sklearn (1.0 exact, 0.0 otherwise);
    NRMSE on a constant target is +inf (raw -inf)."""
    y = np.full(5, 2.0)
    preds = np.column_stack([y, y + 1.0])
    r2 = _columnwise_raw_values(get_scorer_object("r2"), y, preds)
    np.testing.assert_allclose(r2, [1.0, 0.0])
    nrmse = _columnwise_raw_values(get_scorer_object("nrmse"), y, preds)
    assert np.all(np.isneginf(nrmse))


def test_columnwise_values_generic_scorer_falls_back_to_loop():
    rng = np.random.default_rng(1)
    y = rng.normal(size=30)
    preds = y[:, None] + rng.normal(size=(30, 3))
    scorer = make_scorer(median_absolute_error, greater_is_better=False)
    values = _columnwise_raw_values(scorer, y, preds)
    expected = [-median_absolute_error(y, preds[:, c]) for c in range(3)]
    np.testing.assert_allclose(values, expected)


# ──────────────────────────────────────────────────────────────────────────────
#  2. bbc_cv_score
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("y", "oof", "kwargs"),
    [
        (np.arange(5.0), np.zeros((2, 4)), {}),  # misaligned rows
        (np.arange(5.0), np.zeros(5), {}),  # 1-D matrix
        (np.arange(5.0), np.zeros((0, 5)), {}),  # no configuration
        (np.arange(5.0), np.zeros((1, 5)), {"n_bootstrap": 0}),
        (np.array([1.0, np.nan]), np.zeros((1, 2)), {}),  # < 2 usable rows
    ],
)
def test_bbc_cv_score_rejects_invalid_input(y, oof, kwargs):
    with pytest.raises(ValueError):
        bbc_cv_score(get_scorer_object("rmse"), y, oof, **kwargs)


def test_bbc_cv_score_is_deterministic_with_seed():
    rng = np.random.default_rng(2)
    y = rng.normal(size=40)
    oof = y[None, :] + rng.normal(size=(5, 40))
    scorer = get_scorer_object("r2")
    first = bbc_cv_score(scorer, y, oof, random_state=7)
    assert first == bbc_cv_score(scorer, y, oof, random_state=7)
    assert first != bbc_cv_score(scorer, y, oof, random_state=8)


def test_bbc_cv_score_removes_selection_bias_on_noise():
    """200 pure-noise configurations: the best in-sample R² is clearly
    positive (selection bias), the BBC-CV estimate is not."""
    rng = np.random.default_rng(3)
    y = rng.normal(size=60)
    oof = 0.3 * rng.normal(size=(200, 60))
    scorer = get_scorer_object("r2")
    best_of_n = max(pooled_scorer_value(scorer, y, row) for row in oof)
    bbc = bbc_cv_score(scorer, y, oof, random_state=0)
    assert best_of_n > 0.05
    assert bbc < 0.0
    assert bbc < best_of_n


def test_bbc_cv_score_single_configuration_close_to_its_score():
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    oof = (y + 0.5 * rng.normal(size=200))[None, :]
    scorer = get_scorer_object("rmse")
    full = pooled_scorer_value(scorer, y, oof[0])
    bbc = bbc_cv_score(scorer, y, oof, random_state=0)
    assert bbc == pytest.approx(full, rel=0.05)


@pytest.mark.parametrize(
    ("metric", "base", "to_metric"),
    [
        ("r2", "mse", lambda err, y: 1.0 - err / np.var(y)),
        ("nrmse", "rmse", lambda err, y: -(err / (y.max() - y.min()))),
    ],
)
def test_bbc_cv_score_normalized_metrics_use_full_sample_normalizer(
    metric, base, to_metric
):
    """R² / NRMSE: the bootstrap runs on MSE / RMSE and is normalized by the
    variance / range of all rows (no small-subset bias of the normalizer)."""
    rng = np.random.default_rng(6)
    y = rng.normal(size=50)
    oof = y[None, :] + rng.normal(size=(8, 50))
    err = -bbc_cv_score(get_scorer_object(base), y, oof, random_state=3)
    value = bbc_cv_score(get_scorer_object(metric), y, oof, random_state=3)
    assert value == pytest.approx(to_metric(err, y), rel=1e-12)


def test_bbc_cv_score_duplicates_and_uncovered_rows_do_not_change_result():
    rng = np.random.default_rng(5)
    y = rng.normal(size=30)
    oof = y[None, :] + rng.normal(size=(4, 30))
    scorer = get_scorer_object("mae")
    base = bbc_cv_score(scorer, y, oof, random_state=1)
    # A duplicated configuration is collapsed.
    assert bbc_cv_score(scorer, y, np.vstack([oof, oof[:2]]), random_state=1) == base
    # A row no configuration predicted (train part of train_test_split) and a
    # row with a NaN target are dropped.
    y_ext = np.concatenate([y, [1.0, np.nan]])
    oof_ext = np.hstack([oof, np.full((4, 1), np.nan), np.ones((4, 1))])
    assert bbc_cv_score(scorer, y_ext, oof_ext, random_state=1) == base


# ──────────────────────────────────────────────────────────────────────────────
#  3. tuner.optimize(collect_oof=True)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def reg_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(n_samples=120, n_features=6, noise=10.0, random_state=0)
    return pd.DataFrame(X), pd.Series(y)


def _ridge_space(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}


def _completed_values(res) -> list[float]:
    return [t.value for t in res.trials if t.state.name == "COMPLETE"]


@pytest.mark.parametrize(
    ("metric", "method", "pruning"),
    [
        ("r2", "k_fold", None),  # pooled path
        ("neg_mean_absolute_error", "k_fold", None),  # additive cross_validate path
        ("r2", "train_test_split", None),
        (
            "r2",
            "k_fold",
            {"enable": True, "strategy": "median", "min_steps": 1, "n_startup_trials": 1},
        ),
        (
            "neg_mean_absolute_error",
            "k_fold",
            {"enable": True, "strategy": "median", "min_steps": 1, "n_startup_trials": 1},
        ),
    ],
)
def test_optimize_collects_one_oof_row_per_completed_trial(
    reg_data, metric, method, pruning
):
    X, y = reg_data
    res = hyperopt.optimize(
        "ridge",
        X,
        y,
        metric=metric,
        val_method=method,
        n_folds=5,  # 120 rows -> equal folds: the MAE fold mean equals the pooled MAE
        n_trials=6,
        random_state=0,
        space_overrides={"ridge": _ridge_space},
        pruning=pruning,
        fit_best_model=False,
        collect_oof=True,
    )
    assert res is not None
    values = _completed_values(res)
    oof = res.oof_predictions
    assert oof is not None
    assert oof.shape == (len(values), len(X))
    scorer = hpo_evaluation._build_scorer(metric)
    for value, row in zip(values, oof, strict=True):
        covered = np.isfinite(row)
        if method == "train_test_split":
            assert 0 < covered.sum() < len(X)
        else:
            assert covered.all()
        assert pooled_scorer_value(scorer, y.to_numpy(), row) == pytest.approx(
            value, rel=1e-9
        )


def test_optimize_collect_oof_does_not_change_scores(reg_data):
    X, y = reg_data
    kwargs = dict(
        metric="neg_mean_absolute_error",
        val_method="k_fold",
        n_folds=5,
        n_trials=4,
        random_state=0,
        space_overrides={"ridge": _ridge_space},
        fit_best_model=False,
    )
    off = hyperopt.optimize("ridge", X, y, **kwargs)
    on = hyperopt.optimize("ridge", X, y, collect_oof=True, **kwargs)
    assert off.oof_predictions is None
    assert _completed_values(off) == pytest.approx(_completed_values(on), rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
#  4. Engine
# ──────────────────────────────────────────────────────────────────────────────


def _noisy_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    df["target"] = df["f0"] + rng.randn(n) * 2.0
    return df


def _engine_cfg(tmp_path, *, holdout="auto", parallel=False, metric="r2") -> dict:
    general: dict = {
        "comparison_metric": metric,
        "validation_strategy": "k_fold",
        "n_folds": 4,
        "holdout_fraction": holdout,
        "path_to_model": str(tmp_path / "model.pkl"),
        "phases": [
            {"name": "search", "n_trials": 4, "action": "all_algorithms"},
            {
                "name": "final",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
    }
    if parallel:
        general["parallel_strategy"] = "algorithms"
        general["parallel_mode"] = "threads"
    return {
        "general": general,
        "algorithms": {
            "ridge": {"enable": True},
            "lasso": {"enable": True},
        },
    }


@pytest.mark.parametrize("metric", ["r2", "rmse", "mae"])
def test_bbc_score_reported_without_holdout(tmp_path, metric):
    df = _noisy_df(60)
    result = train_best_model(
        config=_engine_cfg(tmp_path, metric=metric), df=df, target="target"
    )
    assert "holdout_score" not in result
    assert np.isfinite(result["bbc_score"])
    # The BBC-CV estimate corrects the optimism of the best-trial score.
    if metric == "r2":
        assert result["bbc_score"] <= result["score"]
    else:
        assert result["bbc_score"] >= result["score"]


def test_bbc_score_absent_with_holdout(tmp_path):
    df = _noisy_df(150)
    result = train_best_model(config=_engine_cfg(tmp_path), df=df, target="target")
    assert "holdout_score" in result
    assert "bbc_score" not in result


def test_bbc_score_skipped_over_memory_budget(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(hpo_phases_mod, "BBC_MAX_OOF_CELLS", 10)
    df = _noisy_df(60)
    with caplog.at_level(logging.WARNING):
        result = train_best_model(config=_engine_cfg(tmp_path), df=df, target="target")
    assert "bbc_score" not in result
    assert any("BBC-CV estimate skipped" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("parallel", [False, True])
def test_oof_pool_collects_every_algorithm(tmp_path, parallel):
    df = _noisy_df(60)
    cfg = Config.model_validate(_engine_cfg(tmp_path, holdout=None, parallel=parallel))
    prepared = prepare_dataset(cfg, df, "target")
    ctx = execute_phases(cfg, prepared)
    assert len(ctx.oof_pool) == 2
    for matrix in ctx.oof_pool:
        assert matrix.shape[1] == len(prepared.y)
        assert 1 <= matrix.shape[0] <= 4


def test_bbc_score_helper_ignores_misaligned_and_empty(caplog):
    y = pd.Series(np.arange(20, dtype=float))
    good = (y.to_numpy() + 1.0)[None, :]
    with caplog.at_level(logging.WARNING):
        value = _bbc_score_in_user_metric("rmse", [good, np.zeros((1, 5))], y, 0)
    assert value == pytest.approx(1.0)
    assert any("not aligned" in r.getMessage() for r in caplog.records)
    assert _bbc_score_in_user_metric("rmse", [np.zeros((1, 5))], y, 0) is None
