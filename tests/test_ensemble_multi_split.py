"""Multi-split behavior of the experimental top-K ensemble.

Covered contracts:

* the ensemble selection consumes the leaderboard aggregate already produced
  by the HPO multi-split scoring; no second, idempotent re-aggregation is
  applied (and ``n_repeats = 1`` stays the classic single-split behavior);
* the winner-vs-ensemble comparison scores both on the same repeats and the
  same finite OOF rows, then aggregates the per-repeat RMSE values;
* the comparison reports the number of contributing repeats, resolves the
  winner/tie verdict consistently and is rendered by the Markdown report;
* the engine path is deterministic for a fixed seed and the comparison values
  match the member OOF vectors.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunStatus,
)
from configurable_automl_engine.experiment.ensemble import (
    compute_weights,
    winner_vs_ensemble_comparison,
)
from configurable_automl_engine.experiment.reporting import MarkdownReporter
from configurable_automl_engine.experiment.result import ExperimentResult
from configurable_automl_engine.training_engine.ensemble_step import (
    top_k_from_leaderboard,
)
from configurable_automl_engine.training_engine.state import (
    PhaseContext,
    PhaseResult,
)

UTC = dt.UTC


def _context(
    records: dict[str, PhaseResult], winner: str | None = None
) -> PhaseContext:
    ctx = PhaseContext()
    ctx.leaderboard_snapshot = records
    ctx.winner_algo = winner
    return ctx


# ─────────────────── selection uses the leaderboard aggregate ─────────────────── #


def test_top_k_consumes_the_leaderboard_aggregate() -> None:
    """The published leaderboard score is already the multi-split aggregate.

    The HPO runner stores ``best_trial.value`` — the per-repeat aggregate — so
    the ensemble ranks and weighs the same aggregate, not a single split. A
    lower published error wins and drives the inverse-error weights.
    """
    snapshot = {
        "worse": PhaseResult(score=-0.20, params={}),
        "better": PhaseResult(score=-0.10, params={}),
    }
    selected, scores = top_k_from_leaderboard(_context(snapshot), "rmse", 1)
    assert selected == ["better"]
    assert scores == {"worse": pytest.approx(0.20), "better": pytest.approx(0.10)}

    weights = compute_weights({name: scores[name] for name in selected}, "rmse")
    assert weights["better"] == pytest.approx(1.0)


def test_tied_scores_keep_configuration_order() -> None:
    """Equal leaderboard scores preserve the deterministic insertion order."""
    snapshot = {
        "first": PhaseResult(score=-0.10, params={}),
        "second": PhaseResult(score=-0.10, params={}),
    }
    selected, scores = top_k_from_leaderboard(_context(snapshot), "rmse", 2)
    assert selected == ["first", "second"]
    assert scores["first"] == pytest.approx(scores["second"])


def test_selection_is_deterministic() -> None:
    snapshot = {
        "a": PhaseResult(score=-0.11, params={}),
        "b": PhaseResult(score=-0.09, params={}),
        "c": PhaseResult(score=-0.10, params={}),
    }
    first = top_k_from_leaderboard(_context(snapshot), "rmse", 2)
    second = top_k_from_leaderboard(_context(snapshot), "rmse", 2)
    assert first == second


# ─────────────────── winner vs ensemble comparison ─────────────────── #


def test_comparison_scores_both_on_same_repeats_and_rows() -> None:
    """The RMSE of both models uses the same finite rows of every repeat."""
    y = np.array([0.0, 1.0, 2.0, 3.0])
    member_a = np.array([0.0, 1.0, 2.0, 3.0])
    member_b = np.array([0.0, 2.0, 2.0, 4.0])
    # A missing member prediction excludes the row from BOTH estimates.
    member_b_repeat_2 = np.array([0.0, np.nan, 2.0, 4.0])
    winner = np.array([0.0, 1.5, 2.0, 3.5])
    explanation = winner_vs_ensemble_comparison(
        y_true=y,
        member_predictions_by_repeat=[
            {"a": member_a, "b": member_b},
            {"a": member_a, "b": member_b},
            {"a": member_a, "b": member_b_repeat_2},
        ],
        winner_predictions_by_repeat=[winner, winner, winner],
        weights={"a": 0.75, "b": 0.25},
        winner_algorithm="a",
        split_aggregation="median",
    )
    assert explanation is not None
    assert explanation["metric"] == "rmse"
    assert explanation["n_repeats"] == 3
    assert explanation["n_repeats_used"] == 3
    assert explanation["split_aggregation"] == "median"
    assert explanation["winner_algorithm"] == "a"

    ensemble = 0.75 * member_a + 0.25 * member_b
    expected_first = float(np.sqrt(np.mean((y - ensemble) ** 2)))
    expected_winner = float(np.sqrt(np.mean((y - winner) ** 2)))
    mask = np.array([True, False, True, True])
    expected_third_ensemble = float(np.sqrt(np.mean((y[mask] - ensemble[mask]) ** 2)))
    expected_third_winner = float(np.sqrt(np.mean((y[mask] - winner[mask]) ** 2)))
    assert explanation["ensemble_repeat_rmse"] == pytest.approx(
        [expected_first, expected_first, expected_third_ensemble]
    )
    assert explanation["winner_repeat_rmse"] == pytest.approx(
        [expected_winner, expected_winner, expected_third_winner]
    )
    assert explanation["ensemble_rmse"] == pytest.approx(
        float(np.median([expected_first, expected_first, expected_third_ensemble]))
    )
    assert explanation["winner_rmse"] == pytest.approx(
        float(np.median([expected_winner, expected_winner, expected_third_winner]))
    )
    assert explanation["delta_rmse"] == pytest.approx(
        explanation["ensemble_rmse"] - explanation["winner_rmse"]
    )
    assert explanation["better"] in {"winner", "ensemble", "tie"}


def test_comparison_counts_only_contributing_repeats() -> None:
    """A repeat without a common finite row is dropped from the aggregate."""
    y = np.array([0.0, 1.0, 2.0])
    prediction = np.array([0.0, 1.0, 2.0])
    missing = np.array([np.nan, np.nan, np.nan])
    explanation = winner_vs_ensemble_comparison(
        y_true=y,
        member_predictions_by_repeat=[
            {"a": prediction},
            {"a": missing},
            {"a": prediction},
        ],
        winner_predictions_by_repeat=[prediction, prediction, prediction],
        weights={"a": 1.0},
        winner_algorithm="a",
    )
    assert explanation is not None
    assert explanation["n_repeats"] == 3
    assert explanation["n_repeats_used"] == 2
    assert explanation["ensemble_repeat_rmse"] == pytest.approx([0.0, None, 0.0])
    assert explanation["ensemble_rmse"] == pytest.approx(0.0)


def test_comparison_reports_tie_and_respects_mean_aggregation() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = np.array([0.0, 1.0, 2.0, 3.0])
    explanation = winner_vs_ensemble_comparison(
        y_true=y,
        member_predictions_by_repeat=[{"a": prediction}],
        winner_predictions_by_repeat=[prediction],
        weights={"a": 1.0},
        winner_algorithm="a",
        split_aggregation="mean",
    )
    assert explanation is not None
    assert explanation["ensemble_rmse"] == pytest.approx(0.0)
    assert explanation["winner_rmse"] == pytest.approx(0.0)
    assert explanation["better"] == "tie"
    assert explanation["winner_better"] is False


def test_comparison_returns_none_when_repeats_are_missing() -> None:
    y = np.array([0.0, 1.0])
    assert (
        winner_vs_ensemble_comparison(
            y_true=y,
            member_predictions_by_repeat=[],
            winner_predictions_by_repeat=[],
            weights={},
            winner_algorithm="a",
        )
        is None
    )
    assert (
        winner_vs_ensemble_comparison(
            y_true=y,
            member_predictions_by_repeat=[{"a": np.array([0.0, 1.0])}],
            winner_predictions_by_repeat=[],
            weights={"a": 1.0},
            winner_algorithm="a",
        )
        is None
    )


def test_markdown_report_renders_comparison() -> None:
    section = {
        "enabled": True,
        "strategy": "weighted_average",
        "weight_method": "score_weighted",
        "k_requested": 2,
        "k_used": 2,
        "metric_user": "rmse",
        "metric_direction": "minimize",
        "members": [
            {"rank": 1, "algorithm": "a", "score": 1.0, "weight": 0.6},
            {"rank": 2, "algorithm": "b", "score": 3.0, "weight": 0.4},
        ],
        "comparison": {
            "metric": "rmse",
            "n_repeats": 3,
            "n_repeats_used": 2,
            "split_aggregation": "median",
            "winner_algorithm": "a",
            "winner_rmse": 1.25,
            "ensemble_rmse": 0.95,
            "delta_rmse": -0.30,
            "better": "ensemble",
            "winner_better": False,
            "winner_repeat_rmse": [1.2, 1.3, None],
            "ensemble_repeat_rmse": [0.9, 1.0, None],
        },
        "artifact_path": "/tmp/model_ensemble.pkl",
    }
    result = ExperimentResult(
        run_id=uuid.uuid4(),
        started_at=dt.datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=dt.datetime(2026, 1, 1, second=30, tzinfo=UTC),
        duration_seconds=30.0,
        status=RunStatus.completed,
        config={},
        resolved_validation="k_fold",
        resolved_n_folds=5,
        resolved_test_size=0.2,
        metric_user="rmse",
        metric_sklearn="neg_root_mean_squared_error",
        metric_direction=MetricDirection.minimize,
        runs=[],
        winner="a",
        winner_score=1.25,
        model_path="/tmp/model.pkl",
        ensemble=section,
    )
    markdown = str(MarkdownReporter(include_config=False).report(result))
    assert "**Winner vs ensemble (aggregated OOF).**" in markdown
    assert "RMSE: winner 1.250, ensemble 0.950" in markdown
    assert "ensemble is better" in markdown
    assert "Repeats: 2/3" in markdown
    assert "aggregation: median" in markdown


# ─────────────────── engine end-to-end ─────────────────── #


def _engine_config(directory: Any) -> dict[str, Any]:
    return {
        "general": {
            "comparison_metric": "rmse",
            "random_state": 42,
            "n_repeats": 2,
            "split_aggregation": "median",
            "validation_strategy": "train_test_split",
            "holdout_fraction": None,
            "path_to_model": str(directory / "model.pkl"),
            "ensemble": {"enable": True, "k": 2},
            "phases": [
                {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            },
            "elasticnet": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.8]},
            },
        },
        "oversampling": {"enable": False},
    }


def _small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [float(i % 5) for i in range(40)],
            "f2": [i % 7 * 0.5 for i in range(40)],
            "target": [10.0 - 0.2 * i for i in range(40)],
        }
    )


def test_engine_comparison_values_and_determinism(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine comparison matches the member OOF vectors and is repeatable."""
    from configurable_automl_engine import AutoMLExperiment
    from configurable_automl_engine.training_engine import ensemble_step

    frame = _small_frame()
    captured: dict[str, Any] = {}
    original_fit = ensemble_step._fit_finalist

    def capturing_fit(*args: Any, **kwargs: Any) -> Any:
        trainer = original_fit(*args, **kwargs)
        captured[args[0]] = trainer
        return trainer

    monkeypatch.setattr(ensemble_step, "_fit_finalist", capturing_fit)
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first = AutoMLExperiment(_engine_config(first_dir), frame, target="target").run()

    section = first.ensemble
    assert section is not None
    comparison = section["comparison"]
    assert comparison["n_repeats"] == 2
    assert comparison["n_repeats_used"] == 2
    assert comparison["winner_rmse"] == pytest.approx(first.winner_score, rel=1e-9)

    weights = {m["algorithm"]: m["weight"] for m in section["members"]}
    y = frame["target"].to_numpy(dtype=float)
    expected_ensemble: list[float] = []
    expected_winner: list[float] = []
    for repeat_index in range(2):
        vectors = {
            name: np.asarray(
                trainer.oof_predictions_by_repeat_[repeat_index], dtype=float
            )
            for name, trainer in captured.items()
        }
        mask = np.isfinite(y)
        for vector in vectors.values():
            mask &= np.isfinite(vector)
        ensemble_prediction = np.zeros(int(mask.sum()))
        for name, vector in vectors.items():
            ensemble_prediction += weights[name] * vector[mask]
        expected_ensemble.append(
            float(np.sqrt(np.mean((y[mask] - ensemble_prediction) ** 2)))
        )
        winner_vector = vectors[first.winner]
        expected_winner.append(
            float(np.sqrt(np.mean((y[mask] - winner_vector[mask]) ** 2)))
        )
    assert comparison["ensemble_repeat_rmse"] == pytest.approx(expected_ensemble)
    assert comparison["winner_repeat_rmse"] == pytest.approx(expected_winner)
    assert comparison["ensemble_rmse"] == pytest.approx(
        float(np.median(expected_ensemble))
    )
    assert comparison["winner_rmse"] == pytest.approx(float(np.median(expected_winner)))

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = AutoMLExperiment(_engine_config(second_dir), frame, target="target").run()
    assert second.ensemble is not None
    assert second.ensemble["comparison"] == comparison
    assert [(m["algorithm"], m["weight"]) for m in second.ensemble["members"]] == [
        (m["algorithm"], m["weight"]) for m in section["members"]
    ]
