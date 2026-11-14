"""Multi-split (repeated) validation scoring of the trainer and finalists.

The OOF channel of ``score_on_validation_splits`` runs over all repeated
seeds, the primary/additional estimates are combined by the configured
aggregation policy, and the finalist candidate receives the aggregated
``RMSE_oof`` with its spread.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from configurable_automl_engine import ModelTrainer, trainer_scoring
from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.trainer_artifact import _collect_artifact_metadata
from configurable_automl_engine.trainer_params import (
    _validate_n_repeats,
    _validate_split_aggregation,
)
from configurable_automl_engine.trainer_scoring import score_on_validation_splits
from configurable_automl_engine.trainer_state import ValidationScores
from configurable_automl_engine.training_engine import artifact as artifact_mod
from configurable_automl_engine.training_engine.artifact import (
    _build_trainer,
    _oof_score_in_user_metric,
)
from configurable_automl_engine.training_engine.component import train_best_model
from configurable_automl_engine.training_engine.config_parser import Config, GeneralCfg
from configurable_automl_engine.training_engine.finalists import (
    _fit_finalist,
    _to_candidate,
)


def _frame(n_samples: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_samples, 3)), columns=["a", "b", "c"])
    y = pd.Series(2.0 * X["a"] - 0.5 * X["b"] + rng.normal(size=n_samples))
    return X, y


def _make_steps(feature_selection_active: bool, X_for_estimate: Any) -> list[Any]:
    return [("scaler", StandardScaler()), ("model", Ridge(alpha=0.5))]


def _score(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int | None = None,
    split_aggregation: str = "median",
    strategy: str = "k_fold",
    n_folds: int = 4,
    test_size: float = 0.2,
    metric: str = "rmse",
) -> ValidationScores:
    return score_on_validation_splits(
        X,
        y,
        _make_steps,
        feature_selection_active=False,
        metric=metric,
        additional_metrics=["mae", "r2"],
        validation_strategy=strategy,
        n_folds=n_folds,
        test_size=test_size,
        random_state=42,
        algorithm="ridge",
        n_repeats=n_repeats,
        split_aggregation=split_aggregation,
    )


def test_n_repeats_one_is_the_classic_single_split() -> None:
    X, y = _frame()
    classic = _score(X, y)
    explicit = _score(X, y, n_repeats=1)

    assert classic.val_score == explicit.val_score
    assert classic.oof_score == explicit.oof_score
    assert classic.oof_coverage == explicit.oof_coverage
    assert classic.additional_scores == explicit.additional_scores
    assert np.array_equal(classic.oof_predictions, explicit.oof_predictions)
    assert explicit.repeat_scores is None
    assert explicit.score_mean is None
    assert explicit.score_median is None
    assert explicit.score_std is None
    assert explicit.oof_repeat_scores is None
    assert explicit.oof_predictions_by_repeat is None


@pytest.mark.parametrize("aggregation", ["median", "mean"])
def test_repeat_aggregation_and_spread(aggregation: str) -> None:
    X, y = _frame()
    scores = _score(X, y, n_repeats=4, split_aggregation=aggregation)

    assert scores.repeat_scores is not None
    assert len(scores.repeat_scores) == 4
    finite = [value for value in scores.repeat_scores if np.isfinite(value)]
    assert finite
    assert scores.score_mean == pytest.approx(float(np.mean(finite)))
    assert scores.score_median == pytest.approx(float(np.median(finite)))
    assert scores.score_std == pytest.approx(float(np.std(finite)))

    expected = (
        float(np.mean(finite)) if aggregation == "mean" else float(np.median(finite))
    )
    assert scores.val_score == pytest.approx(expected)

    assert scores.oof_repeat_scores is not None
    assert len(scores.oof_repeat_scores) == 4
    oof_finite = [value for value in scores.oof_repeat_scores if np.isfinite(value)]
    expected_oof = (
        float(np.mean(oof_finite))
        if aggregation == "mean"
        else float(np.median(oof_finite))
    )
    assert scores.oof_score == pytest.approx(expected_oof)
    assert scores.oof_score_std == pytest.approx(float(np.std(oof_finite)))
    assert scores.oof_predictions_by_repeat is not None
    assert len(scores.oof_predictions_by_repeat) == 4
    assert set(scores.additional_scores) == {"mae", "r2"}


def test_repeat_scoring_is_deterministic() -> None:
    X, y = _frame()
    first = _score(X, y, n_repeats=3)
    second = _score(X, y, n_repeats=3)

    assert first.repeat_scores == second.repeat_scores
    assert first.oof_repeat_scores == second.oof_repeat_scores
    assert first.oof_score == second.oof_score
    assert first.score_mean == second.score_mean
    assert first.score_median == second.score_median
    assert first.score_std == second.score_std
    for left, right in zip(
        first.oof_predictions_by_repeat or [],
        second.oof_predictions_by_repeat or [],
        strict=True,
    ):
        assert np.array_equal(left, right)


def test_holdout_coverage_is_computed_over_the_concatenated_repeats(
    caplog: pytest.LogCaptureFixture,
) -> None:
    X, y = _frame()
    with caplog.at_level(logging.WARNING):
        scores = _score(
            X,
            y,
            n_repeats=3,
            strategy="train_test_split",
            test_size=0.25,
        )

    assert scores.oof_coverage == pytest.approx(0.25)
    assert len(scores.oof_predictions_by_repeat or []) == 3
    assert "only the validation part" in caplog.text


def test_failed_first_repeat_exposes_a_valid_oof_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _frame()
    original = trainer_scoring._collect_fold_oof
    calls = 0

    def fail_first_repeat(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        if calls <= 4:  # the four folds of repeat 0
            return False
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_scoring, "_collect_fold_oof", fail_first_repeat)

    scores = _score(X, y, n_repeats=2)

    assert scores.oof_repeat_scores is not None
    assert scores.oof_repeat_scores[0] == float("inf")
    assert np.isfinite(scores.oof_repeat_scores[1])
    assert scores.oof_score == pytest.approx(scores.oof_repeat_scores[1])
    # The failed repeat contributes no OOF cells to the concatenated coverage.
    assert scores.oof_coverage == pytest.approx(0.5)
    assert scores.repeat_scores is not None
    assert np.isnan(scores.repeat_scores[0])
    assert scores.val_score == pytest.approx(scores.oof_score)
    # The exposed vector comes from a repeat with a valid OOF score.
    assert scores.oof_predictions_by_repeat is not None
    assert scores.oof_predictions is not None
    assert np.isfinite(scores.oof_predictions).any()
    assert np.array_equal(scores.oof_predictions, scores.oof_predictions_by_repeat[1])


class _NaNPredictor(BaseEstimator, RegressorMixin):
    """A regressor returning one NaN on the second predict call."""

    def __init__(self, call_counter: list[int]) -> None:
        self.call_counter = call_counter

    def fit(self, X: Any, y: Any) -> _NaNPredictor:
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        self.call_counter[0] += 1
        predictions = np.full(len(X), self.mean_)
        if self.call_counter[0] == 2:
            predictions[0] = np.nan
        return predictions


def test_real_nan_prediction_invalidates_only_its_repeat(
    caplog: pytest.LogCaptureFixture,
) -> None:
    X, y = _frame()
    call_counter = [0]

    def make_steps(feature_selection_active: bool, X_for_estimate: Any) -> list[Any]:
        return [("model", _NaNPredictor(call_counter))]

    with caplog.at_level(logging.WARNING):
        scores = score_on_validation_splits(
            X,
            y,
            make_steps,
            feature_selection_active=False,
            metric="rmse",
            additional_metrics=[],
            validation_strategy="k_fold",
            n_folds=4,
            test_size=0.2,
            random_state=42,
            algorithm="ridge",
            n_repeats=2,
        )

    assert "non-finite predictions" in caplog.text
    assert scores.oof_repeat_scores is not None
    assert scores.oof_repeat_scores[0] == float("inf")
    assert np.isfinite(scores.oof_repeat_scores[1])
    assert scores.oof_score == pytest.approx(scores.oof_repeat_scores[1])
    # Three folds of repeat 0 were collected before the NaN fold failed, but
    # the whole repeat is excluded from every aggregate.
    assert scores.oof_coverage == pytest.approx(0.875)
    assert scores.repeat_scores is not None
    assert np.isnan(scores.repeat_scores[0])
    assert scores.val_score == pytest.approx(scores.oof_score)


def test_representative_oof_vector_prefers_repeat_closest_to_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _frame()
    original = trainer_scoring._score_repeat
    calls = 0

    def outlier_first_repeat(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        outcome = original(*args, **kwargs)
        index = calls
        calls += 1
        if index == 0:
            return replace(outcome, oof_pred=outcome.oof_pred + 100.0)
        return outcome

    monkeypatch.setattr(trainer_scoring, "_score_repeat", outlier_first_repeat)

    scores = _score(X, y, n_repeats=3)

    assert scores.oof_repeat_scores is not None
    assert scores.oof_score is not None
    expected_index = min(
        range(len(scores.oof_repeat_scores)),
        key=lambda i: abs(scores.oof_repeat_scores[i] - scores.oof_score),
    )
    assert expected_index != 0
    assert scores.oof_predictions_by_repeat is not None
    assert np.array_equal(
        scores.oof_predictions, scores.oof_predictions_by_repeat[expected_index]
    )


def test_result_oof_score_uses_aggregated_rmse() -> None:
    X, y = _frame(n_samples=60)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="neg_root_mean_squared_error",
        random_state=42,
        validation_strategy="k_fold",
        n_folds=3,
        n_repeats=3,
        split_aggregation="mean",
    ).fit(X, y)

    result_oof = _oof_score_in_user_metric("rmse", trainer, y)

    assert trainer.oof_score_ is not None
    assert result_oof == pytest.approx(trainer.oof_score_)
    assert _oof_score_in_user_metric(
        "neg_root_mean_squared_error", trainer, y
    ) == pytest.approx(trainer.oof_score_)


def test_end_to_end_result_oof_score_matches_score(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
general:
  comparison_metric: rmse
  validation_strategy: k_fold
  n_folds: 3
  n_repeats: 3
  split_aggregation: mean
  path_to_model: 'm.pkl'
  phases:
    - name: Search
      n_trials: 1
      action: all_algorithms
    - name: FinalSelect
      n_trials: 1
      action: select
      strategy: cv_best
      output: winner
algorithms:
  ridge:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      alpha: [0.1, 1.0]
""",
        encoding="utf-8",
    )
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "d"])
    df["target"] = 2.0 * df["a"] - 0.5 * df["b"] + rng.normal(size=60)
    model_path = tmp_path / "m.pkl"

    result = train_best_model(
        cfg_file, df, target="target", model_path_override=model_path
    )
    loaded = ModelTrainer.load(model_path)

    assert result["oof_score"] == pytest.approx(result["score"])
    assert result["oof_score"] == pytest.approx(loaded.oof_score_)


def test_holdout_path_disables_repeat_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _frame(n_samples=24)
    cfg = Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            n_repeats=3,
            split_aggregation="mean",
            n_folds=3,
        ),
        algorithms={"ridge": {"enable": True}},
    )
    captured: dict[str, ModelTrainer] = {}
    real_build = _build_trainer

    def capture(*args: Any, **kwargs: Any) -> ModelTrainer:
        trainer = real_build(*args, **kwargs)
        captured["trainer"] = trainer
        return trainer

    monkeypatch.setattr(artifact_mod, "_build_trainer", capture)
    prepared = SimpleNamespace(
        X=X,
        y=y,
        holdout_X=X.iloc[:6],
        holdout_y=y.iloc[:6],
        metric_user="rmse",
        resolved_validation="k_fold",
        resolved_n_folds=3,
        resolved_test_size=0.2,
        random_state=42,
        datetime_expansion={},
    )

    score = artifact_mod._compute_holdout_score(
        cfg=cfg,
        winner_cfg=cfg.algorithms.ridge,
        prepared=prepared,
        winner_algo="ridge",
        final_params={"alpha": 0.5},
    )

    assert np.isfinite(score)
    assert captured["trainer"].n_repeats == 1
    assert captured["trainer"].split_aggregation == "median"


def test_artifact_config_hash_includes_repeat_settings() -> None:
    hashes = {
        _collect_artifact_metadata(
            ModelTrainer(
                algorithm="ridge",
                metric="rmse",
                n_repeats=n_repeats,
                split_aggregation=aggregation,
            ).settings
        )["config_hash"]
        for aggregation, n_repeats in [
            ("median", 1),
            ("median", 2),
            ("mean", 2),
        ]
    }

    assert len(hashes) == 3


def test_all_repeats_without_oof_give_no_oof_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _frame()
    monkeypatch.setattr(
        trainer_scoring, "_collect_fold_oof", lambda *args, **kwargs: False
    )

    scores = _score(X, y, n_repeats=2)

    assert scores.oof_score is None
    assert scores.oof_score_std is None
    assert scores.oof_repeat_scores == [float("inf"), float("inf")]
    assert scores.oof_coverage == 0.0
    assert scores.repeat_scores is not None
    assert all(np.isnan(value) for value in scores.repeat_scores)
    assert np.isnan(scores.val_score)

    class _Trainer:
        oof_score_ = scores.oof_score
        oof_repeat_scores_ = scores.oof_repeat_scores
        oof_predictions_by_repeat_ = scores.oof_predictions_by_repeat
        oof_predictions_ = scores.oof_predictions
        split_aggregation = "median"

    assert _oof_score_in_user_metric("rmse", _Trainer(), y) is None
    assert (
        _oof_score_in_user_metric("neg_root_mean_squared_error", _Trainer(), y) is None
    )


def test_all_repeats_failed_raise_the_primary_failure() -> None:
    X, y = _frame()

    def broken_steps(feature_selection_active: bool, X_for_estimate: Any) -> list[Any]:
        raise ValueError("cannot build")

    with pytest.raises(TrainingError, match="Validation scoring failed for all folds"):
        score_on_validation_splits(
            X,
            y,
            broken_steps,
            feature_selection_active=False,
            metric="rmse",
            additional_metrics=[],
            validation_strategy="k_fold",
            n_folds=3,
            test_size=0.2,
            random_state=42,
            algorithm="ridge",
            n_repeats=2,
        )


def test_repeats_may_exceed_folds() -> None:
    X, y = _frame()
    scores = _score(X, y, n_repeats=6, n_folds=3)

    assert scores.repeat_scores is not None
    assert len(scores.repeat_scores) == 6
    assert scores.oof_repeat_scores is not None
    assert len(scores.oof_repeat_scores) == 6


def test_large_repeat_count_with_small_frame() -> None:
    X, y = _frame(n_samples=12)
    scores = _score(X, y, n_repeats=10, n_folds=3)

    assert scores.repeat_scores is not None
    assert len(scores.repeat_scores) == 10
    assert scores.oof_predictions_by_repeat is not None
    assert len(scores.oof_predictions_by_repeat) == 10
    assert scores.oof_coverage == pytest.approx(1.0)


@pytest.mark.parametrize("aggregation", ["median", "mean"])
def test_model_trainer_end_to_end_repeat_settings(aggregation: str) -> None:
    X, y = _frame(n_samples=30)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="rmse",
        random_state=42,
        validation_strategy="k_fold",
        n_folds=3,
        n_repeats=3,
        split_aggregation=aggregation,
    ).fit(X, y)

    assert trainer.n_repeats == 3
    assert trainer.split_aggregation == aggregation
    assert trainer.repeat_scores_ is not None
    assert len(trainer.repeat_scores_) == 3
    expected = (
        float(np.mean(trainer.repeat_scores_))
        if aggregation == "mean"
        else float(np.median(trainer.repeat_scores_))
    )
    assert trainer.val_score == pytest.approx(expected)
    assert trainer.score_mean == pytest.approx(float(np.mean(trainer.repeat_scores_)))
    assert trainer.score_median == pytest.approx(
        float(np.median(trainer.repeat_scores_))
    )
    assert trainer.score_std == pytest.approx(float(np.std(trainer.repeat_scores_)))
    assert trainer.oof_repeat_scores_ is not None
    expected_oof = (
        float(np.mean(trainer.oof_repeat_scores_))
        if aggregation == "mean"
        else float(np.median(trainer.oof_repeat_scores_))
    )
    assert trainer.oof_score_ == pytest.approx(expected_oof)
    assert trainer.oof_score_std_ == pytest.approx(
        float(np.std(trainer.oof_repeat_scores_))
    )


def test_general_repeat_settings_reach_build_trainer_and_finalist() -> None:
    cfg = Config(
        general=GeneralCfg(
            comparison_metric="rmse",
            n_repeats=2,
            split_aggregation="mean",
        ),
        algorithms={"ridge": {"enable": True}},
    )
    algo_cfg = cfg.algorithms.ridge
    built = _build_trainer("ridge", algo_cfg, {"alpha": 0.5}, cfg)
    assert built.n_repeats == 2
    assert built.split_aggregation == "mean"

    X, y = _frame(n_samples=20)
    finalist = _fit_finalist(
        "ridge",
        algo_cfg,
        X,
        y,
        {"alpha": 0.5},
        cfg,
        "r2",
        validation_strategy="k_fold",
        n_folds=3,
        test_size=0.2,
        random_state=42,
    )
    assert finalist.n_repeats == 2
    assert finalist.split_aggregation == "mean"
    assert finalist.repeat_scores_ is not None
    assert len(finalist.repeat_scores_) == 2


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_validate_n_repeats_rejects_invalid_values(bad: Any) -> None:
    with pytest.raises(TrainingError, match="n_repeats must be a positive integer"):
        _validate_n_repeats(bad)


def test_validate_split_aggregation_rejects_unknown_value() -> None:
    assert _validate_split_aggregation("mean") == "mean"
    with pytest.raises(TrainingError, match="split_aggregation must be one of"):
        _validate_split_aggregation("max")


def test_small_sample_falls_back_to_holdout_per_repeat() -> None:
    X, y = _frame(n_samples=8)
    scores = _score(X, y, n_repeats=2, n_folds=5, test_size=0.25)

    assert scores.repeat_scores is not None
    assert len(scores.repeat_scores) == 2
    assert 0.0 < scores.oof_coverage < 1.0


def test_to_candidate_receives_the_aggregated_oof_spread() -> None:
    X, y = _frame(n_samples=12)

    class _Trainer:
        pipeline = "fitted-pipeline"
        oof_score_ = 0.25
        oof_score_std_ = 0.05
        oof_predictions_ = np.full(len(y), 0.0)

        def predict(self, X: Any) -> np.ndarray:
            return np.zeros(len(X))

    candidate = _to_candidate("ridge", _Trainer(), {"alpha": 0.5}, 0.1, X, y)

    assert candidate.rmse_oof == pytest.approx(0.25)
    assert candidate.rmse_oof_std == pytest.approx(0.05)


def test_validation_scores_artifact_drops_repeat_vectors() -> None:
    X, y = _frame()
    scores = _score(X, y, n_repeats=2)

    restored = pickle.loads(pickle.dumps(scores))

    assert restored.repeat_scores == scores.repeat_scores
    assert restored.oof_repeat_scores == scores.oof_repeat_scores
    assert restored.oof_score_std == scores.oof_score_std
    assert restored.oof_predictions_by_repeat is None


def test_validation_scores_unpickle_legacy_state() -> None:
    scores = ValidationScores(
        val_score=1.0,
        additional_scores={"mae": 0.5},
        oof_predictions=np.array([1.0, 2.0]),
        oof_score=0.3,
        oof_coverage=1.0,
        repeat_scores=[1.0],
        score_mean=1.0,
        score_median=1.0,
        score_std=0.0,
        oof_repeat_scores=[0.3],
        oof_score_std=0.0,
        oof_predictions_by_repeat=[np.array([1.0, 2.0])],
    )
    legacy_state = [
        scores.val_score,
        scores.additional_scores,
        scores.oof_predictions,
        scores.oof_score,
        scores.oof_coverage,
    ]

    class _Legacy:
        def __reduce__(self):
            return (ValidationScores.__new__, (ValidationScores,), legacy_state)

    restored = pickle.loads(pickle.dumps(_Legacy()))

    assert restored.val_score == scores.val_score
    assert restored.repeat_scores is None
    assert restored.score_mean is None
    assert restored.oof_predictions_by_repeat is None
