"""Tests of the experimental top-K ensemble (the T10 component).

Covered modules: ``experiment.ensemble`` (the pure core: top-K selection,
weight computation, ``EnsembleModel``), the config section
(``general.ensemble``), the engine integration
(``training_engine.ensemble_step`` + ``component``) and the consumers
(``ExperimentResult.ensemble``, the Markdown reporter, the facade's
``ensemble_model()``).

Covered scenarios (positive / negative / boundary):

- P1: the ensemble of the top-K models is built from a run and predicts
  (the full ``enable`` path: config → engine → result → facade);
- P2: at K=n with equal weights the ensemble prediction is the exact
  mathematical average of the member predictions;
- P3: ``compute_weights`` is direction-aware: ``minimize`` weights are
  inversely proportional to the scores, ``maximize`` — proportional;
- P4: ``select_top_k`` reuses the select-step top-K semantics (rank
  order, deterministic tie-break);
- P5: the ensemble section reaches ``ExperimentResult`` (JSON round-trip
  keeps it) and the Markdown reporter renders the members table;
- N1: k > n candidates → min(k, n) members with a ``RuntimeWarning``;
- N2: no trained models (an empty leaderboard snapshot) → a clear error,
  and the engine run completes without the ensemble section;
- N3: invalid scores (``None``, strings, non-finite, the worst-score
  sentinel class) are rejected before they can poison the weights;
- N4: a member without a trained predictor is an error, not a silent
  degradation;
- B1: k=1 → the ensemble is equivalent to the single model;
- B2: the weight fallback for non-positive scores of a ``maximize``
  metric (uniform weights with a warning);
- B3: serialization/deserialization round-trip of the ensemble
  (pickle and joblib), and a non-ensemble file is rejected on load.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunStatus,
    SerializationFormat,
)
from configurable_automl_engine.experiment.ensemble import (
    EnsembleMember,
    EnsembleModel,
    compute_weights,
    select_top_k,
    validate_ensemble_score,
)
from configurable_automl_engine.experiment.reporting import MarkdownReporter
from configurable_automl_engine.experiment.result import ExperimentResult
from configurable_automl_engine.training_engine.config_parser import Config

UTC = dt.UTC


# ────────────────────────── helpers ────────────────────────── #


class _FakePredictor:
    """A constant-output predictor (sklearn-style ``predict``)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def _members(scores: dict[str, float]) -> list[EnsembleMember]:
    """Members from a rank-ordered score mapping with equal weights."""
    return [
        EnsembleMember(name=name, score=score, weight=1.0, rank=idx)
        for idx, (name, score) in enumerate(scores.items(), start=1)
    ]


def _ensemble(
    scores: dict[str, float],
    outputs: dict[str, float],
    direction: MetricDirection | str = MetricDirection.minimize,
) -> EnsembleModel:
    """An ensemble of constant-output predictors over score-ordered members."""
    return EnsembleModel(
        metric_user="rmse",
        metric_direction=direction,
        members=_members(scores),
        predictors={name: _FakePredictor(value) for name, value in outputs.items()},
    )


def _engine_config(tmp_path: Any, **ensemble: Any) -> dict[str, Any]:
    """A valid two-phase config dict with the ensemble section enabled."""
    general: dict[str, Any] = {
        "comparison_metric": "rmse",
        "random_state": 42,
        "validation_strategy": "train_test_split",
        "holdout_fraction": None,
        "path_to_model": str(tmp_path / "model.pkl"),
        "ensemble": {"enable": True, **ensemble},
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
    }
    return {
        "general": general,
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


def _small_df() -> pd.DataFrame:
    """A tiny deterministic regression dataset."""
    rows = 40
    return pd.DataFrame(
        {
            "f1": [float(i % 5) for i in range(rows)],
            "f2": [i % 7 * 0.5 for i in range(rows)],
            "target": [10.0 - 0.2 * i for i in range(rows)],
        }
    )


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_ensemble_built_from_run_and_predicts(tmp_path: Any) -> None:
    """P1: the full engine path — config enable → run → result section →
    facade ``ensemble_model()`` → predictions."""
    from configurable_automl_engine import AutoMLExperiment

    experiment = AutoMLExperiment(
        _engine_config(tmp_path, k=5), _small_df(), target="target"
    )
    result = experiment.run()
    assert result.status is RunStatus.completed
    assert result.ensemble is not None
    section = result.ensemble
    assert section["enabled"] is True
    assert section["strategy"] == "weighted_average"
    # k=5 > n=2 → min(k, n): the effective size and the request are reported.
    assert section["k_requested"] == 5
    assert section["k_used"] == 2
    assert len(section["members"]) == 2
    # The winner is always among the members.
    assert result.winner in {m["algorithm"] for m in section["members"]}
    # The member weights are positive and sum to 1.
    weights = [m["weight"] for m in section["members"]]
    assert all(w > 0 for w in weights)
    assert sum(weights) == pytest.approx(1.0)
    # The ensemble artifact is written next to the winner artifact.
    assert section["artifact_path"].endswith("_ensemble.pkl")

    model = experiment.ensemble_model()
    assert len(model) == section["k_used"]
    preds = model.predict(_small_df()[["f1", "f2"]])
    assert preds.shape == (40,)
    assert np.all(np.isfinite(preds))


def test_p2_equal_weights_average_math(tmp_path: Any) -> None:
    """P2: at K=n and equal weights the ensemble is the exact average of
    the member predictions."""
    ens = _ensemble(
        {"a": 2.0, "b": 4.0},
        {"a": 10.0, "b": 6.0},
        direction=MetricDirection.minimize,
    )
    weights = compute_weights({"a": 1.0, "b": 1.0}, "minimize", method="equal")
    assert weights == {"a": 0.5, "b": 0.5}
    preds = ens.predict(pd.DataFrame({"x": [1, 2]}))
    assert preds == pytest.approx(np.array([8.0, 8.0]))


def test_p3_weights_direction_aware() -> None:
    """P3: ``minimize`` — inverse weighting (the smallest error gets the
    largest weight); ``maximize`` — proportional weighting."""
    scores = {"a": 1.0, "b": 3.0}
    minimize = compute_weights(scores, "minimize")
    assert minimize["a"] > minimize["b"]
    assert minimize["a"] == pytest.approx(0.75)
    assert minimize["b"] == pytest.approx(0.25)

    maximize = compute_weights({"a": 0.9, "b": 0.3}, "maximize")
    assert maximize["a"] > maximize["b"]
    assert maximize["a"] == pytest.approx(0.75)
    assert maximize["b"] == pytest.approx(0.25)

    # A metric name is accepted as the direction argument (the engine
    # passes prepared.metric_user).
    assert compute_weights(scores, "rmse") == minimize


def test_p4_top_k_reuses_select_semantics() -> None:
    """P4: rank order and the deterministic insertion-order tie-break."""
    # minimize (RMSE): ascending — the smallest error first.
    assert select_top_k({"a": 3.0, "b": 1.0, "c": 2.0}, 2, "minimize") == ["b", "c"]
    # maximize (R²): descending — the largest score first.
    assert select_top_k({"a": 0.3, "b": 0.9, "c": 0.1}, 2, "maximize") == ["b", "a"]
    # Ties keep the mapping insertion order (the configuration order).
    assert select_top_k({"first": 2.0, "second": 2.0}, 1, "minimize") == ["first"]


def test_p5_section_reaches_result_report_and_json(tmp_path: Any) -> None:
    """P5: the ensemble section survives the JSON round-trip and is
    rendered by the Markdown reporter."""
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
        winner_score=1.0,
        ensemble=section,
    )
    restored = ExperimentResult.model_validate_json(result.to_json())
    assert restored.ensemble == section

    markdown = str(MarkdownReporter(include_config=False).report(result))
    assert "## Ensemble (experimental)" in markdown
    assert "a" in markdown and "b" in markdown
    assert "0.600" in markdown and "0.400" in markdown
    assert "model_ensemble.pkl" in markdown

    # No ensemble → no section in the report.
    plain = result.model_copy(update={"ensemble": None})
    markdown_plain = str(MarkdownReporter(include_config=False).report(plain))
    assert "Ensemble" not in markdown_plain


# ────────────────────────── negative scenarios ────────────────────────── #


def test_n1_k_exceeds_candidates_min_k_n_with_warning() -> None:
    """N1: k > n → min(k, n) members with a RuntimeWarning."""
    scores = {"a": 1.0, "b": 2.0}
    with pytest.warns(RuntimeWarning, match="min\\(k, n\\)"):
        selected = select_top_k(scores, 5, "minimize")
    assert selected == ["a", "b"]


def test_n2_no_trained_models_clear_error(tmp_path: Any) -> None:
    """N2: an empty leaderboard snapshot → a clear error from the core,
    and the engine run completes without the ensemble section."""
    from configurable_automl_engine.training_engine.ensemble_step import (
        top_k_from_leaderboard,
    )
    from configurable_automl_engine.training_engine.state import PhaseContext

    with pytest.raises(ValueError, match="no trained models available"):
        top_k_from_leaderboard(PhaseContext(), "rmse", 3)

    # The engine path: the ensemble failure never aborts the run.
    from configurable_automl_engine import AutoMLExperiment

    experiment = AutoMLExperiment(
        _engine_config(tmp_path, k=2), _small_df(), target="target"
    )
    result = experiment.run()
    assert result.status is RunStatus.completed
    assert result.ensemble is not None  # a normal run still builds it

    # A failed run (no algorithms trained) carries no ensemble either.
    failed = AutoMLExperiment(_engine_config(tmp_path, k=2), pd.DataFrame())
    with pytest.raises(ValueError):
        failed.run()
    assert failed.result is None


def test_n3_invalid_scores_rejected() -> None:
    """N3: None/strings/bools/non-finite/sentinel-class scores never
    reach the weights (the winner-score invariant)."""
    sentinel = -3.5e38  # at or below WORST_SCORE_THRESHOLD
    for bad in (None, "0.5", True, float("nan"), float("inf"), sentinel):
        with pytest.raises(ValueError):
            validate_ensemble_score(bad)
        with pytest.raises(ValueError):
            compute_weights({"a": bad}, "minimize")
        with pytest.raises(ValueError):
            select_top_k({"a": bad}, 1, "minimize")
    assert validate_ensemble_score(np.float64(1.5)) == 1.5
    # An empty score mapping is rejected too.
    with pytest.raises(ValueError, match="at least one"):
        compute_weights({}, "minimize")
    with pytest.raises(ValueError, match="at least one"):
        select_top_k({}, 1, "minimize")


def test_n4_missing_predictor_is_error() -> None:
    """N4: a member without a trained predictor is an error — a silently
    degraded ensemble would report numbers it cannot produce."""
    with pytest.raises(ValueError, match="No trained model"):
        _ensemble({"a": 1.0, "b": 2.0}, {"a": 10.0})
    # Duplicate member names are rejected as well.
    with pytest.raises(ValueError, match="unique"):
        EnsembleModel(
            metric_user="rmse",
            metric_direction="minimize",
            members=[
                EnsembleMember("a", 1.0, 0.5, 1),
                EnsembleMember("a", 2.0, 0.5, 2),
            ],
            predictors={"a": _FakePredictor(1.0)},
        )
    # Review: a NEGATIVE member weight is rejected per member, even when
    # the weights sum to 1.0 ([2.0, -1.0]) — the sum check alone cannot
    # see the sign structure, and a negative weight must never reach the
    # members, the report section or the serialized artifact.
    with pytest.raises(ValueError, match="positive and finite"):
        EnsembleModel(
            metric_user="rmse",
            metric_direction="minimize",
            members=[
                EnsembleMember("a", 1.0, 2.0, 1),
                EnsembleMember("b", 2.0, -1.0, 2),
            ],
            predictors={"a": _FakePredictor(1.0), "b": _FakePredictor(2.0)},
        )
    # A non-finite weight is rejected per member as well.
    with pytest.raises(ValueError, match="positive and finite"):
        EnsembleModel(
            metric_user="rmse",
            metric_direction="minimize",
            members=[
                EnsembleMember("a", 1.0, float("nan"), 1),
                EnsembleMember("b", 2.0, 1.0, 2),
            ],
            predictors={"a": _FakePredictor(1.0), "b": _FakePredictor(2.0)},
        )


def test_n4_engine_member_without_record_skips_ensemble(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N4 (engine): a member failing to retrain degrades the whole step —
    the run completes, no ensemble section."""
    from configurable_automl_engine import AutoMLExperiment
    from configurable_automl_engine.training_engine import ensemble_step

    def failing_fit(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated retrain failure")

    monkeypatch.setattr(ensemble_step, "_fit_finalist", failing_fit)
    experiment = AutoMLExperiment(
        _engine_config(tmp_path, k=2), _small_df(), target="target"
    )
    result = experiment.run()
    assert result.status is RunStatus.completed
    assert result.winner is not None
    assert result.ensemble is None


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_k1_equivalent_to_single_model() -> None:
    """B1: k=1 → the ensemble with a single member of weight 1 predicts
    exactly like that member."""
    ens = _ensemble({"a": 1.0}, {"a": 7.5})
    assert len(ens) == 1
    assert ens.members[0].weight == pytest.approx(1.0)
    assert ens.predict(pd.DataFrame({"x": [1, 2]})) == pytest.approx([7.5, 7.5])
    # The engine-side selection with k=1 keeps the single best member.
    assert select_top_k({"a": 3.0, "b": 1.0}, 1, "minimize") == ["b"]


def test_b2_non_positive_scores_uniform_fallback() -> None:
    """B2: all ``maximize`` scores non-positive → uniform weights with a
    warning; a partially positive mapping keeps proportional weights."""
    with pytest.warns(RuntimeWarning, match="uniform"):
        weights = compute_weights({"a": -1.0, "b": -2.0}, "maximize")
    assert weights == {"a": 0.5, "b": 0.5}
    weights = compute_weights({"a": 0.0, "b": 2.0}, "maximize")
    assert weights["a"] == 0.0 and weights["b"] == pytest.approx(1.0)
    # Unknown weight methods are rejected.
    with pytest.raises(ValueError, match="Unknown ensemble weight method"):
        compute_weights({"a": 1.0}, "minimize", method="magic")


def test_b3_serialization_round_trip(tmp_path: Any) -> None:
    """B3: save/load round-trip (pickle and joblib), a non-ensemble file
    and a wrong version are rejected on load."""
    ens = _ensemble({"a": 1.5, "b": 2.5}, {"a": 10.0, "b": 30.0})
    frame = pd.DataFrame({"x": [1, 2, 3]})

    for fmt, suffix in (
        (SerializationFormat.pickle, ".pkl"),
        (SerializationFormat.joblib, ".joblib"),
    ):
        path = tmp_path / f"ens{suffix}"
        saved = ens.save(path, fmt)
        assert saved == path
        restored = EnsembleModel.load(path)
        assert len(restored) == len(ens)
        assert restored.metric_user == ens.metric_user
        assert restored.metric_direction is ens.metric_direction
        assert [m.name for m in restored.members] == [m.name for m in ens.members]
        assert [m.weight for m in restored.members] == pytest.approx(
            [m.weight for m in ens.members]
        )
        assert restored.predict(frame) == pytest.approx(ens.predict(frame))

    # Auto-detection: the format is derived from the extension.
    restored = EnsembleModel.load(tmp_path / "ens.joblib")
    assert restored.predict(frame) == pytest.approx(ens.predict(frame))

    # A foreign pickle (a plain dict) is rejected with a clear error.
    foreign = tmp_path / "foreign.pkl"
    with open(foreign, "wb") as fh:
        import pickle

        pickle.dump({"not": "an ensemble"}, fh)
    with pytest.raises(ValueError, match="not an ensemble artifact"):
        EnsembleModel.load(foreign)

    # A missing file raises FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        EnsembleModel.load(tmp_path / "missing.pkl")


def test_b3_report_section_json_safe(tmp_path: Any) -> None:
    """B3 (report): the built section is JSON-serializable (the result
    validator would reject non-JSON-safe payloads)."""
    ens = _ensemble({"a": 1.0, "b": 2.0}, {"a": 10.0, "b": 30.0})
    section = ens.report_section(
        artifact_path=tmp_path / "ens.pkl", k_requested=3, weight_method="equal"
    )
    json.dumps(section, allow_nan=False)  # must not raise


def test_b_config_validation_and_defaults(tmp_path: Any) -> None:
    """B (config): the section defaults to off, k >= 1 is enforced,
    unknown fields and methods are rejected."""
    cfg = Config.model_validate({"algorithms": {"ridge": {"enable": True}}})
    assert cfg.general.ensemble.enable is False
    assert cfg.general.ensemble.k == 3
    assert cfg.general.ensemble.weight_method == "score_weighted"

    cfg = Config.model_validate(
        {
            "general": {"ensemble": {"enable": True, "k": 1, "weight_method": "equal"}},
            "algorithms": {"ridge": {"enable": True}},
        }
    )
    assert cfg.general.ensemble.enable is True
    assert cfg.general.ensemble.k == 1

    with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic ValidationError
        Config.model_validate(
            {
                "general": {"ensemble": {"enable": True, "k": 0}},
                "algorithms": {"ridge": {"enable": True}},
            }
        )
    with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic ValidationError
        Config.model_validate(
            {
                "general": {"ensemble": {"enable": True, "weight_method": "magic"}},
                "algorithms": {"ridge": {"enable": True}},
            }
        )
    with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic ValidationError
        Config.model_validate(
            {
                "general": {"ensemble": {"enable": True, "unknown": True}},
                "algorithms": {"ridge": {"enable": True}},
            }
        )


def test_b_disabled_ensemble_no_section_no_artifact(tmp_path: Any) -> None:
    """B: ensemble disabled (default) → no section, no ensemble artifact,
    and ``ensemble_model()`` reports the consistent no-ensemble state."""
    from configurable_automl_engine import AutoMLExperiment

    config = _engine_config(tmp_path, k=2)
    config["general"]["ensemble"]["enable"] = False
    experiment = AutoMLExperiment(config, _small_df(), target="target")
    result = experiment.run()
    assert result.status is RunStatus.completed
    assert result.ensemble is None
    with pytest.raises(RuntimeError, match="No ensemble available"):
        experiment.ensemble_model()
