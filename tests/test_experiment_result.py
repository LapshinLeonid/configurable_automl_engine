"""Tests of the experiment result data model (``experiment`` package).

Covered scenarios (positive / negative / boundary):

- P1: a full successful-run result (phases, per-algorithm records, winner,
  timings) builds and validates;
- P2: dict/JSON round-trip keeps every field;
- P3: numpy scalars (np.float32/np.int64) are accepted and normalized
  to plain floats;
- N1: garbage score types (None, string, complex) are rejected;
- N2: non-finite and sentinel-class scores are rejected (the
  winner-score invariant);
- N3: missing required fields (run_id) raise a ValidationError;
- B1: an empty record list serializes correctly (a failed experiment);
- B2: exactly one record;
- B3: score values at the sentinel threshold boundary;
- B4: special characters/unicode in the label and algorithm names;
- B5: very long parameter values do not break serialization.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import uuid
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from configurable_automl_engine.common.definitions import (
    WORST_SCORE_THRESHOLD,
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment import (
    AlgorithmRun,
    ExperimentResult,
    MetricDirection as MetricDirectionReexported,
    RunPhaseAction as RunPhaseActionReexported,
    RunStatus as RunStatusReexported,
    RunStepStatus as RunStepStatusReexported,
)
from configurable_automl_engine.hpo.types import (
    HPO_WORST_SCORE,
    WORST_SCORE_THRESHOLD as HPO_WORST_SCORE_THRESHOLD,
)

UTC = dt.UTC


def _ts(seconds: float = 0.0) -> dt.datetime:
    """A fixed UTC timestamp shifted by ``seconds``."""
    return dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + dt.timedelta(seconds=seconds)


def _run_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default keyword arguments of a completed ``AlgorithmRun`` record."""
    values: dict[str, Any] = {
        "algorithm": "ridge",
        "phase_name": "search",
        "phase_action": "all_algorithms",
        "status": RunStepStatus.completed,
        "score_raw": -0.5,
        "score": 0.5,
        "params": {"alpha": 0.1},
        "n_trials_requested": 10,
        "n_trials_evaluated": 10,
        "n_trials_completed": 9,
        "n_trials_pruned": 1,
        "n_trials_failed": 0,
        "duration_seconds": 12.5,
    }
    values.update(overrides)
    return values


def _result_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default keyword arguments of a completed ``ExperimentResult``."""
    values: dict[str, Any] = {
        "run_id": uuid.uuid4(),
        "label": "baseline run",
        "engine_version": "0.0.0",
        "started_at": _ts(),
        "finished_at": _ts(90.0),
        "duration_seconds": 90.0,
        "status": RunStatus.completed,
        "config": {"general": {"comparison_metric": "rmse", "random_state": 42}},
        "resolved_validation": "k_fold",
        "resolved_n_folds": 5,
        "resolved_test_size": 0.2,
        "random_state": 42,
        "metric_user": "rmse",
        "metric_sklearn": "neg_root_mean_squared_error",
        "metric_direction": MetricDirection.minimize,
        "runs": [
            AlgorithmRun(**_run_kwargs()),
            AlgorithmRun(
                **_run_kwargs(
                    algorithm="random_forest",
                    score_raw=-0.7,
                    score=0.7,
                    phase_name="final_select",
                    phase_action="select",
                )
            ),
        ],
        "winner": "random_forest",
        "winner_score": 0.7,
        "winner_params": {"n_estimators": 100},
        "model_path": "models/best.pkl",
        "additional_metrics": {"mae": 1.25},
        "disqualified_algorithms": {"svr": "HPO failed"},
        "oof_score": 0.68,
        "holdout_score": 0.65,
        "bbc_score": 0.66,
        "n_trials_evaluated": 10,
    }
    values.update(overrides)
    return values


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_full_successful_run_builds_and_validates() -> None:
    """P1: a full result with phases, records, winner and timings validates."""
    result = ExperimentResult(**_result_kwargs())
    assert result.status is RunStatus.completed
    assert result.winner == "random_forest"
    assert result.winner_score == 0.7
    assert len(result.runs) == 2
    first, second = result.runs
    assert first.phase_name == "search"
    assert first.phase_action == "all_algorithms"
    assert first.score_raw == -0.5
    assert first.score == 0.5
    assert second.phase_action == "select"
    assert result.duration_seconds == 90.0
    assert result.metric_direction is MetricDirection.minimize
    assert result.disqualified_algorithms == {"svr": "HPO failed"}


def test_p1_user_semantics_of_neg_metric() -> None:
    """The neg_-metric raw score maps to the positive user value."""
    run = AlgorithmRun(**_run_kwargs(score_raw=-0.5, score=0.5))
    assert run.score == 0.5  # RMSE in user semantics is positive


def test_p2_json_round_trip_keeps_all_fields() -> None:
    """P2: to_json → from_json restores an equal model."""
    result = ExperimentResult(**_result_kwargs())
    restored = ExperimentResult.from_json(result.to_json())
    assert restored == result
    payload = json.loads(result.to_json())
    assert payload["run_id"] == str(result.run_id)
    assert payload["status"] == "completed"
    assert payload["runs"][0]["status"] == "completed"
    assert payload["metric_direction"] == "minimize"


def test_p2_model_dump_json_mode_is_json_safe() -> None:
    """``model_dump(mode='json')`` produces a JSON-serializable dict."""
    result = ExperimentResult(**_result_kwargs())
    payload = result.model_dump(mode="json")
    json.dumps(payload, allow_nan=False)  # must not raise
    assert isinstance(payload["run_id"], str)
    assert isinstance(payload["started_at"], str)


def test_p3_numpy_scalars_are_normalized_to_float() -> None:
    """P3: np.float32/np.float64/np.int64 score values are accepted."""
    run = AlgorithmRun(
        **_run_kwargs(
            score_raw=np.float32(-0.5),
            score=np.float64(0.5),
            params={"max_depth": int(np.int64(4))},
        )
    )
    assert type(run.score_raw) is float
    assert type(run.score) is float
    assert run.score_raw == -0.5
    result = ExperimentResult(
        **_result_kwargs(
            winner_score=np.float64(0.7),
            additional_metrics={"mae": np.float32(1.25)},
        )
    )
    assert type(result.winner_score) is float
    assert result.additional_metrics == {"mae": 1.25}


def test_p3_numpy_scalars_in_params_are_rejected() -> None:
    """Params are plain-JSON containers: numpy scalars are not accepted.

    The stdlib ``json`` cannot encode np.float32/np.int64, so such values
    would break ``to_json`` far from the error site; the engine passes its
    data through ``model_dump(mode='json')`` first, which normalizes them.
    """
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(params={"max_depth": np.int64(4)}))
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(params={"lr": np.float32(0.1)}))


def test_p3_int_scores_are_normalized_to_float() -> None:
    """Plain ints are accepted as scores (numpy-cast metrics may return them)."""
    run = AlgorithmRun(**_run_kwargs(score_raw=-1, score=1))
    assert type(run.score_raw) is float


# ────────────────────────── negative scenarios ────────────────────────── #


@pytest.mark.parametrize("bad", [None, "0.5", "abc", complex(1, 2), [0.5], {"s": 1}])
def test_n1_garbage_score_types_are_rejected(bad: Any) -> None:
    """N1: None, strings, complex numbers and containers are rejected."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(score_raw=bad))
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(score=bad))


def test_n1_bool_is_not_a_score() -> None:
    """bool is an int subclass but is not a valid score value."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(score_raw=True))


def test_n1_garbage_additional_metrics_are_rejected() -> None:
    """The additional-metrics map enforces the same score rules."""
    base = _result_kwargs()
    for bad in ({"mae": None}, {"mae": "high"}, {"mae": True}, {"mae": float("nan")}):
        with pytest.raises(ValidationError):
            ExperimentResult(**{**base, "additional_metrics": bad})


@pytest.mark.parametrize(
    "bad",
    [
        HPO_WORST_SCORE,
        WORST_SCORE_THRESHOLD,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_n2_nonfinite_and_sentinel_scores_are_rejected(bad: float) -> None:
    """N2: the winner-score invariant holds for every score field."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(score_raw=bad))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(winner_score=bad))


def test_n2_sentinel_rejected_on_every_report_score() -> None:
    """The sentinel class is rejected on oof/holdout/bbc scores too."""
    base = _result_kwargs()
    for field in ("oof_score", "holdout_score", "bbc_score"):
        with pytest.raises(ValidationError):
            ExperimentResult(**{**base, field: HPO_WORST_SCORE})


def test_n2_score_pair_must_be_complete() -> None:
    """score_raw without score (and vice versa) is an inconsistent record."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(score=None))
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(score_raw=None))


def test_n3_missing_required_fields_raise() -> None:
    """N3: required fields (run_id, metric, timings) cannot be omitted."""
    base = _result_kwargs()
    for field in ("run_id", "started_at", "finished_at", "metric_user", "config"):
        data = {k: v for k, v in base.items() if k != field}
        with pytest.raises(ValidationError):
            ExperimentResult(**data)


def test_n3_nil_run_id_is_rejected() -> None:
    """The nil UUID is not a valid run identifier."""
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(run_id=uuid.UUID(int=0)))


def test_n3_extra_fields_are_rejected() -> None:
    """Unknown fields fail fast (extra='forbid')."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(unexpected=1))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(unexpected=1))


def test_n3_legacy_phase_action_is_rejected() -> None:
    """The removed 'refine_winner' action cannot appear in a record."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(phase_action="refine_winner"))


def test_n3_resolved_validation_must_be_concrete() -> None:
    """The resolved strategy must not stay 'auto'."""
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(resolved_validation="auto"))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(resolved_validation="gossip_protocol"))


def test_n3_completed_requires_winner_and_score() -> None:
    """A completed experiment without a winner/score is inconsistent."""
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(winner=None))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(winner_score=None))


def test_n3_failed_requires_error_message() -> None:
    """A failed experiment must carry the failure message."""
    data = _result_kwargs(status=RunStatus.failed, winner=None, winner_score=None)
    with pytest.raises(ValidationError):
        ExperimentResult(**{**data, "error": None})
    result = ExperimentResult(**{**data, "error": "all algorithms failed"})
    assert result.status is RunStatus.failed
    assert result.winner is None


def test_n3_direction_must_match_metric_semantics() -> None:
    """A minimize-metric cannot be declared as maximize (and vice versa)."""
    with pytest.raises(ValidationError):
        ExperimentResult(
            **_result_kwargs(
                metric_user="r2",
                metric_sklearn="r2",
                metric_direction=MetricDirection.minimize,
            )
        )
    result = ExperimentResult(
        **_result_kwargs(
            metric_user="r2",
            metric_sklearn="r2",
            metric_direction=MetricDirection.maximize,
            winner_score=0.9,
        )
    )
    assert result.metric_direction is MetricDirection.maximize


def test_n3_duration_and_time_consistency() -> None:
    """finished_at < started_at and a too-small duration are rejected."""
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(finished_at=_ts(-10.0), duration_seconds=0.0))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(duration_seconds=1.0))


def test_p1_duration_tolerance_for_independent_clocks() -> None:
    """A small duration skew is allowed: the stamps and the duration may
    come from different clocks (e.g. monotonic vs datetime.now(UTC))."""
    # 30 ms skew — within the tolerance.
    result = ExperimentResult(
        **_result_kwargs(
            started_at=_ts(), finished_at=_ts(30.0), duration_seconds=29.97
        )
    )
    assert result.duration_seconds == 29.97
    # A 100 ms deficit — beyond the tolerance.
    with pytest.raises(ValidationError):
        ExperimentResult(
            **_result_kwargs(
                started_at=_ts(), finished_at=_ts(30.0), duration_seconds=29.9
            )
        )


def test_n3_naive_datetimes_are_rejected() -> None:
    """Timestamps without a timezone are rejected (consumers compare them).

    A naive datetime would raise TypeError at the consumer side against an
    aware one, so the model rejects it instead of storing the mixed forms.
    """
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(started_at=dt.datetime(2026, 1, 1, 12, 0)))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(finished_at=dt.datetime(2026, 1, 1, 12, 0)))


def test_p1_aware_datetimes_are_normalized_to_utc() -> None:
    """Non-UTC aware timestamps are stored normalized to UTC."""
    result = ExperimentResult(
        **_result_kwargs(
            started_at=dt.datetime(
                2026, 1, 1, 15, 0, tzinfo=dt.timezone(dt.timedelta(hours=3))
            ),
            finished_at=dt.datetime(
                2026, 1, 1, 15, 30, tzinfo=dt.timezone(dt.timedelta(hours=3))
            ),
            duration_seconds=1800.0,
        )
    )
    assert result.started_at.utcoffset() == dt.timedelta(0)
    assert result.finished_at.utcoffset() == dt.timedelta(0)
    assert result.started_at == _ts()
    assert result.finished_at == _ts(1800.0)
    # The normalization is stable across the JSON round-trip.
    assert ExperimentResult.from_json(result.to_json()).started_at == _ts()


def test_n3_time_comparison_across_input_timezones() -> None:
    """started/finished given in different timezones compare consistently."""
    with pytest.raises(ValidationError):
        ExperimentResult(
            **_result_kwargs(
                started_at=dt.datetime(
                    2026, 1, 1, 15, 0, tzinfo=dt.timezone(dt.timedelta(hours=3))
                ),
                finished_at=dt.datetime(2026, 1, 1, 11, 0, tzinfo=dt.UTC),
                duration_seconds=0.0,
            )
        )


def test_n3_non_json_safe_config_is_rejected() -> None:
    """The config snapshot must be a JSON-serializable mapping."""
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(config={"a": object()}))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(config={"a": float("nan")}))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(config="not a mapping"))


def test_n3_ensemble_section_without_model_path_is_rejected() -> None:
    """An ensemble section cannot exist without the winner artifact path.

    The ensemble artifact is saved next to the winner artifact and the
    section records that path; the engine degrades to "no section"
    when the result carries no ``model_path``. A section on a result
    without one therefore cannot come from the engine assembly — the
    consistency check rejects it at the result boundary instead of
    letting an ensemble without a reloadable artifact reach the
    consumers (the facade ``ensemble_model()`` reads the artifact path
    from the section).
    """
    ensemble_section = {
        "members": [
            {
                "algorithm": "random_forest",
                "score": 0.7,
                "weight": 1.0,
                "rank": 1,
            }
        ],
        "k_requested": 3,
        "effective_size": 1,
        "weight_method": "score_weighted",
        "artifact_path": "models/best_ensemble.pkl",
    }
    with pytest.raises(ValidationError, match="model_path"):
        ExperimentResult(**_result_kwargs(model_path=None, ensemble=ensemble_section))
    with pytest.raises(ValidationError, match="model_path"):
        ExperimentResult(**_result_kwargs(model_path="", ensemble=ensemble_section))
    # With the winner artifact recorded the section is accepted.
    result = ExperimentResult(
        **_result_kwargs(model_path="models/best.pkl", ensemble=ensemble_section)
    )
    assert result.ensemble is not None
    assert result.ensemble["artifact_path"] == "models/best_ensemble.pkl"


def test_n3_model_path_without_ensemble_section_is_accepted() -> None:
    """The inverse direction is fine: a winner artifact without an
    ensemble section is the default (the feature is off or the step
    degraded to no section)."""
    result = ExperimentResult(**_result_kwargs(model_path="models/best.pkl"))
    assert result.ensemble is None


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_empty_record_list_serializes_correctly() -> None:
    """B1: a failed experiment with zero records round-trips."""
    result = ExperimentResult(
        **_result_kwargs(
            status=RunStatus.failed,
            winner=None,
            winner_score=None,
            winner_params={},
            model_path=None,
            disqualified_algorithms={"ridge": "HPO failed", "svr": "HPO failed"},
            runs=[],
            error="no algorithm produced a valid result",
        )
    )
    assert result.runs == []
    restored = ExperimentResult.from_json(result.to_json())
    assert restored == result
    assert restored.runs == []


def test_b2_exactly_one_record() -> None:
    """B2: a single-algorithm result is valid."""
    result = ExperimentResult(**_result_kwargs(runs=[AlgorithmRun(**_run_kwargs())]))
    assert len(result.runs) == 1


def test_b3_score_just_above_threshold_is_accepted() -> None:
    """B3: the smallest valid score is the next float above the threshold."""
    just_above = float(np.nextafter(WORST_SCORE_THRESHOLD, 0.0))
    assert just_above > WORST_SCORE_THRESHOLD
    run = AlgorithmRun(**_run_kwargs(score_raw=just_above, score=just_above))
    assert run.score_raw == just_above
    result = ExperimentResult(**_result_kwargs(winner_score=just_above))
    assert result.winner_score == just_above


def test_b3_score_at_threshold_is_rejected() -> None:
    """B3: a score exactly at the threshold belongs to the sentinel class."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(score_raw=WORST_SCORE_THRESHOLD))
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(winner_score=WORST_SCORE_THRESHOLD))


def test_b3_definitions_threshold_is_single_source_of_truth() -> None:
    """``hpo/types`` re-exports the canonical layer-0 constant (no copy)."""
    assert HPO_WORST_SCORE_THRESHOLD is WORST_SCORE_THRESHOLD
    assert HPO_WORST_SCORE_THRESHOLD == float(np.finfo(np.float32).min)


def test_n3_json_safety_of_free_form_containers() -> None:
    """Non-finite floats and non-serializable objects in free-form
    containers are rejected at validation time.

    NaN/±inf silently degrade to ``null`` in ``model_dump(mode='json')``
    (silent data loss), and non-serializable objects surface only later at
    ``to_json`` time — both fail fast here instead.
    """
    base = _result_kwargs()
    bad_payloads = [
        {"winner_params": {"x": float("nan")}},
        {"winner_params": {"x": float("inf")}},
        {"winner_params": {"x": object()}},
        {"sanity_gate": {"circuits": {"a": float("nan")}}},
        {"complexity_guard": {"tightened": float("inf")}},
    ]
    for payload in bad_payloads:
        with pytest.raises(ValidationError):
            ExperimentResult(**{**base, **payload})
    # Non-finite values inside a run's params are rejected too.
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(params={"x": float("nan")}))
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(params={"x": object()}))
    # Plain JSON-safe values (including None sections) are accepted.
    result = ExperimentResult(
        **_result_kwargs(
            winner_params={"alpha": 0.5, "nested": {"list": [1, "a", None]}},
            sanity_gate={"circuit_a": {"verdict": "ok"}},
            complexity_guard=None,
        )
    )
    restored = ExperimentResult.from_json(result.to_json())
    assert restored.winner_params == result.winner_params
    assert restored.sanity_gate == result.sanity_gate


def test_n3_resolved_test_size_domain() -> None:
    """resolved_test_size accepts only a fraction in (0, 1) or rows ≥ 1.

    These are the two shapes the engine's validation resolution produces;
    NaN/±inf, non-positive values, bools and fractions ≥ 1 cannot come
    from it and are rejected.
    """
    base = _result_kwargs()
    for bad in (-0.5, 0, 0.0, True, float("nan"), float("inf"), 5.0, -100):
        with pytest.raises(ValidationError):
            ExperimentResult(**{**base, "resolved_test_size": bad})
    for good, expected in [(0.2, 0.2), (0.99, 0.99), (150, 150), (1, 1)]:
        result = ExperimentResult(**{**base, "resolved_test_size": good})
        assert result.resolved_test_size == expected


def test_p3_numpy_scalars_for_resolved_test_size() -> None:
    """numpy real scalars are accepted and normalized like the score fields."""
    base = _result_kwargs()
    result = ExperimentResult(**{**base, "resolved_test_size": np.int64(150)})
    assert result.resolved_test_size == 150
    assert type(result.resolved_test_size) is int
    result = ExperimentResult(**{**base, "resolved_test_size": np.float32(0.2)})
    assert type(result.resolved_test_size) is float
    assert result.resolved_test_size == pytest.approx(0.2, rel=1e-6)
    for bad in (np.float64(5.0), np.int64(0), np.float64(float("nan"))):
        with pytest.raises(ValidationError):
            ExperimentResult(**{**base, "resolved_test_size": bad})


def test_n3_custom_metric_result_round_trips_without_the_registry(
    tmp_path,
) -> None:
    """A custom-metric result deserializes in a process without the registry.

    The model is a cross-process contract: the metric direction of a custom
    metric is carried in the payload (the per-process scorer registry must
    not be consulted during deserialization). The regression runs a fresh
    interpreter without the custom metric registered and restores the
    payload there (the producer runs in this process).
    """
    from configurable_automl_engine.common.custom_metrics import (
        CustomMetricSpec,
        register_custom_metric,
    )

    register_custom_metric(
        "my_custom_metric",
        CustomMetricSpec(path="builtins:sum", kind="metric", greater_is_better=True),
    )
    try:
        result = ExperimentResult(
            **_result_kwargs(
                metric_user="my_custom_metric",
                metric_sklearn="my_custom_metric",
                metric_direction=MetricDirection.maximize,
            )
        )
    finally:
        # Leave the registry as it was for the other tests.
        from configurable_automl_engine.common import custom_metrics

        with custom_metrics._registry_lock:
            custom_metrics._registry.pop("my_custom_metric", None)

    payload_path = tmp_path / "result.json"
    payload_path.write_text(result.to_json(), encoding="utf-8")

    script = (
        "import sys\n"
        "from configurable_automl_engine.experiment import ExperimentResult\n"
        "from configurable_automl_engine.common.custom_metrics import lookup_custom_metric\n"
        "assert lookup_custom_metric('my_custom_metric') is None, 'registry must be empty'\n"
        "result = ExperimentResult.from_json(open(sys.argv[1], encoding='utf-8').read())\n"
        "assert result.metric_user == 'my_custom_metric'\n"
        "assert result.metric_direction.value == 'maximize'\n"
        "print('restored')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(payload_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "restored" in proc.stdout


def test_n3_builtin_metric_direction_is_still_cross_checked() -> None:
    """Built-in/sklearn metric names keep their static direction check."""
    with pytest.raises(ValidationError):
        ExperimentResult(
            **_result_kwargs(
                metric_user="r2",
                metric_sklearn="r2",
                metric_direction=MetricDirection.minimize,
            )
        )
    result = ExperimentResult(
        **_result_kwargs(
            metric_user="r2",
            metric_sklearn="r2",
            metric_direction=MetricDirection.maximize,
        )
    )
    assert result.metric_direction is MetricDirection.maximize


def test_b4_unicode_and_special_characters_are_preserved() -> None:
    """B4: unicode/special characters in label and names survive the round-trip."""
    label = 'эксперимент #1 ✓ <tag> & "quotes"'
    algo = "méthod_α λ-регрессия"
    result = ExperimentResult(
        **_result_kwargs(
            label=label,
            runs=[AlgorithmRun(**_run_kwargs(algorithm=algo))],
            winner=algo,
        )
    )
    restored = ExperimentResult.from_json(result.to_json())
    assert restored.label == label
    assert restored.runs[0].algorithm == algo
    assert restored.winner == algo


def test_b4_empty_label_and_empty_names_are_rejected() -> None:
    """An empty algorithm/phase name is rejected; the label may be None."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(algorithm="  "))
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(phase_name=""))
    result = ExperimentResult(**_result_kwargs(label=None))
    assert result.label is None


def test_b5_very_long_parameters_serialize() -> None:
    """B5: large parameter payloads do not break the JSON round-trip."""
    long_value = "x" * 100_000
    big_params = {"mask": long_value, "grid": list(range(500))}
    result = ExperimentResult(
        **_result_kwargs(
            runs=[AlgorithmRun(**_run_kwargs(params=big_params))],
            winner_params=big_params,
        )
    )
    restored = ExperimentResult.from_json(result.to_json())
    assert restored.winner_params == big_params
    assert restored.runs[0].params == big_params
    assert len(result.to_json()) > 100_000


# ────────────────────────── package re-exports ────────────────────────── #


def test_statuses_are_reexported_from_the_package() -> None:
    """The package re-exports the status enums and they are identical."""
    assert RunStatusReexported is RunStatus
    assert RunStepStatusReexported is RunStepStatus
    assert MetricDirectionReexported is MetricDirection
    assert RunPhaseActionReexported is RunPhaseAction
    assert set(RunStatus) == {"completed", "failed"}
    assert set(RunStepStatus) == {"completed", "failed", "disqualified"}
    assert set(RunPhaseAction) == {"all_algorithms", "refine", "select"}


def test_record_duration_must_be_non_negative() -> None:
    """A negative record duration is rejected."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(duration_seconds=-1.0))


def test_n3_negative_trial_counters_are_rejected() -> None:
    """Trial counters cannot be negative (consistent with duration/ge rules)."""
    for field in (
        "n_trials_requested",
        "n_trials_evaluated",
        "n_trials_completed",
        "n_trials_pruned",
        "n_trials_failed",
    ):
        with pytest.raises(ValidationError):
            AlgorithmRun(**_run_kwargs(**{field: -3}))
    # Zero is a valid count.
    run = AlgorithmRun(
        **_run_kwargs(
            n_trials_requested=0,
            n_trials_evaluated=0,
            n_trials_completed=0,
            n_trials_pruned=0,
            n_trials_failed=0,
        )
    )
    assert run.n_trials_evaluated == 0


def test_n3_phase_action_is_an_enum() -> None:
    """phase_action validates against the RunPhaseAction enum."""
    for action in RunPhaseAction:
        run = AlgorithmRun(**_run_kwargs(phase_action=action))
        assert run.phase_action is action
        # The str-Enum keeps the string semantics for consumers.
        assert run.phase_action == action.value
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(phase_action="refine_winner"))
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(phase_action="nonsense"))


def test_n3_failed_and_disqualified_records_require_error() -> None:
    """A failed/disqualified record must carry its reason in 'error'."""
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(status=RunStepStatus.failed, error=None))
    with pytest.raises(ValidationError):
        AlgorithmRun(**_run_kwargs(status=RunStepStatus.disqualified, error=None))
    disqualified = AlgorithmRun(
        **_run_kwargs(
            status=RunStepStatus.disqualified,
            error="sanity gate: duplicate predictions",
            score_raw=None,
            score=None,
        )
    )
    assert disqualified.error == "sanity gate: duplicate predictions"
    # A completed record needs no error message.
    assert AlgorithmRun(**_run_kwargs(error=None)).error is None


def test_n3_metric_sklearn_must_match_the_user_metric() -> None:
    """metric_sklearn must be the sklearn mapping of metric_user."""
    with pytest.raises(ValidationError):
        ExperimentResult(**_result_kwargs(metric_sklearn="neg_mean_absolute_error"))
    # The consistent pair is accepted.
    result = ExperimentResult(**_result_kwargs())
    assert result.metric_sklearn == "neg_root_mean_squared_error"
    # An alias-free metric maps to itself (lowercased).
    result = ExperimentResult(
        **_result_kwargs(
            metric_user="r2",
            metric_sklearn="r2",
            metric_direction=MetricDirection.maximize,
        )
    )
    assert result.metric_sklearn == "r2"
