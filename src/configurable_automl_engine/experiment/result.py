"""Central data model of an experiment result (the T1 foundation).

``ExperimentResult`` is the contract between the training engine, the
experiment store and the reporters (including a future MLflow reporter):
reporters consume only this model and never reach into the engine. The
model covers everything the engine reports today (the dict assembled by
``training_engine/artifact.py::persist_artifact``) plus the per-algorithm /
per-phase detail (``AlgorithmRun``): raw optimizer scores, user-semantic
scores, best parameters and the phase each record belongs to.

Key invariants:

* the model stores only serializable data: no ``ModelTrainer`` and no live
  sklearn objects — just artifact paths and metrics;
* every score value passes the winner-score invariant
  (``is_valid_winner_score``): ``None``, non-numeric values, non-finite
  values and the worst-score sentinel class (scores at or below
  ``WORST_SCORE_THRESHOLD``) are rejected at validation time;
* ``model_dump(mode="json")`` (and therefore ``to_json``) is JSON-safe:
  numpy scalars, ``Path``, UUID, enums and datetimes are normalized;
* records keep a deterministic order: the caller sorts them by phase name
  and configuration order (independent of parallel-worker completion).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from typing import Annotated, Any

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    field_validator,
    model_validator,
)

from configurable_automl_engine.common.definitions import (
    WORST_SCORE_THRESHOLD,
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
    ValidationStrategy,
)
from configurable_automl_engine.common.metrics import (
    reserved_metric_names,
    to_sklearn_name,
    user_direction,
)

__all__ = [
    "AlgorithmRun",
    "ExperimentResult",
    "MetricDirection",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
]


def _check_score_value(value: float) -> float:
    """Reject non-finite and sentinel-class score values.

    A valid score is finite and strictly above the worst-score sentinel
    threshold (``WORST_SCORE_THRESHOLD``, the float32 minimum): the same
    rule the engine applies to winner scores. NaN/±inf and sentinel-class
    values signal the absence of a real metric and must never reach the
    result.
    """
    if not math.isfinite(value):
        raise ValueError(f"Score must be finite, got {value!r}")
    if value <= WORST_SCORE_THRESHOLD:
        raise ValueError(
            f"Score {value!r} falls into the worst-score sentinel class "
            f"(<= {WORST_SCORE_THRESHOLD!r}) and must never reach the result"
        )
    return value


#: A score field type: numeric (plain int/float or a numpy real scalar,
#: normalized to ``float``) that passes the finite + sentinel check.
#: ``StrictFloat`` accepts the numpy real scalars (np.float32/np.float64/
#: np.int64) but rejects strings and bool.
ScoreValue = Annotated[StrictFloat, AfterValidator(_check_score_value)]
ScoreValueOrNone = ScoreValue | None

#: The allowed skew between ``duration_seconds`` and
#: ``finished_at - started_at``: the two may be computed by different
#: clocks (e.g. time.monotonic() vs datetime.now(UTC)), so a strict
#: microsecond match would reject legitimate results.
_DURATION_TOLERANCE_SECONDS = 0.05


def _check_score_validity(score: Any) -> Any:
    """Backward-compatible helper: validate one raw score value.

    Applies the same rules as the ``ScoreValue`` type; used for values that
    arrive inside containers (``additional_metrics``).
    """
    if isinstance(score, bool) or not isinstance(
        score, (int, float, np.floating, np.integer)
    ):
        # ValueError (not TypeError): the value flows through a Pydantic
        # validator, where a TypeError would surface as an unrelated
        # "error converting" message instead of a validation error.
        raise ValueError(  # noqa: TRY004
            f"Score must be a real number, got {type(score).__name__}: {score!r}"
        )
    return _check_score_value(float(score))


def _check_json_safe(value: Any, where: str) -> None:
    """Raise ``ValueError`` unless ``value`` survives a JSON round-trip.

    Free-form containers (parameters, report sections) must be JSON-safe:
    a non-finite float silently degrades to ``null`` in
    ``model_dump(mode="json")`` (silent data loss for the consumers), and a
    non-serializable object (including numpy scalars, which the stdlib
    ``json`` cannot encode) surfaces only later, at ``to_json`` time, far
    from the error site. Checking at validation time fails fast instead:
    callers pass engine data through ``Config.model_dump(mode="json")``
    first, which already normalizes numpy types.
    """
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"{where} must be JSON-serializable without NaN/±inf "
            f"(plain int/float/str/bool/None, list, dict; run numpy values "
            f"through model_dump(mode='json') first): {err}"
        ) from err


class _ResultModel(BaseModel):
    """Base of the result models: strict, no extra fields, JSON-safe."""

    model_config = ConfigDict(extra="forbid")


class AlgorithmRun(_ResultModel):
    """One algorithm's record of one pipeline phase.

    Attributes:
        algorithm: The algorithm name (as configured in ``algorithms``).
        phase_name: The name of the phase the record belongs to
            (``HPOPhaseCfg.name``).
        phase_action: The phase action: 'all_algorithms', 'refine' or
            'select' (the legacy 'refine_winner' is rejected by the config
            schema).
        status: The record outcome: 'completed', 'failed' or
            'disqualified'.
        score_raw: The best raw optimizer score of the phase (optimizer
            semantics, "higher is better"); ``None`` when the algorithm
            produced no valid score.
        score: The score in user semantics (``to_user_value``); ``None``
            when there is no valid raw score.
        params: The best found hyperparameters of the phase (a
            JSON-serializable mapping; tuples are stored as tuples but
            become lists after a JSON round-trip).
        n_trials_requested: The trial budget of the phase.
        n_trials_evaluated: The number of trials actually evaluated by the
            algorithm's HPO run (own COMPLETED trials).
        n_trials_completed: The trials that ran to completion.
        n_trials_pruned: The trials stopped early by the pruner.
        n_trials_failed: The trials that failed with an error.
        duration_seconds: The wall-clock duration of the record, in
            seconds.
        error: The failure/disqualification reason; required for
            ``status='failed'`` and ``status='disqualified'`` (the
            circuit-breaker message).
    """

    algorithm: str
    phase_name: str
    phase_action: RunPhaseAction
    status: RunStepStatus
    score_raw: ScoreValueOrNone = None
    score: ScoreValueOrNone = None
    params: dict[str, Any] = Field(default_factory=dict)
    n_trials_requested: int | None = Field(default=None, ge=0)
    n_trials_evaluated: int = Field(default=0, ge=0)
    n_trials_completed: int = Field(default=0, ge=0)
    n_trials_pruned: int = Field(default=0, ge=0)
    n_trials_failed: int = Field(default=0, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    error: str | None = None

    @field_validator("algorithm", "phase_name")
    @classmethod
    def _name_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("params")
    @classmethod
    def _params_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        _check_json_safe(value, "params")
        return value

    @model_validator(mode="after")
    def _consistency(self) -> AlgorithmRun:
        """Cross-field consistency of the record.

        The raw score and the user score appear (or vanish) together: the
        user value is derived from the raw value by ``to_user_value``, so
        one without the other would be an inconsistent record. A failed or
        disqualified record must carry its reason in ``error``.
        """
        if (self.score_raw is None) != (self.score is None):
            raise ValueError(
                "score_raw and score must be set together "
                f"(got score_raw={self.score_raw!r}, score={self.score!r})"
            )
        if self.status is not RunStepStatus.completed and not (
            self.error and self.error.strip()
        ):
            raise ValueError(
                f"A {self.status.value!r} record must carry its reason in 'error'"
            )
        return self


class ExperimentResult(_ResultModel):
    """The full result of one experiment run.

    The single data structure reporters (experiment store, reporting, a
    future MLflow reporter) consume; they never reach into the engine.

    Attributes:
        run_id: The unique identifier of the run (UUID).
        label: An optional human-readable label of the run.
        engine_version: The version of the engine that produced the result.
        started_at: The run start time (timezone-aware, stored normalized
            to UTC; naive datetimes are rejected).
        finished_at: The run finish time (timezone-aware, stored
            normalized to UTC; naive datetimes are rejected).
        duration_seconds: The total wall-clock duration of the run.
        status: The experiment outcome: 'completed' or 'failed'.
        config: A serializable snapshot of the run configuration
            (``Config.model_dump(mode="json")``), not a live ``Config``.
        resolved_validation: The resolved validation strategy ('auto' is
            already expanded by ``prepare_dataset``).
        resolved_n_folds: The number of folds (relevant for 'k_fold').
        resolved_test_size: The hold-out size: a fraction in (0, 1) or an
            integer number of test rows (≥ 1), the two shapes the
            validation resolution produces; anything else is rejected.
        random_state: The engine seed of the run (``general.random_state``).
        metric_user: The user-facing comparison metric name.
        metric_sklearn: The metric name in sklearn semantics; validated to
            be the sklearn mapping of ``metric_user`` (a registry-independent
            alias-map lookup).
        metric_direction: The metric direction in user semantics
            ('minimize' for errors, 'maximize' for score metrics); for
            custom metrics it is carried in the payload and not re-resolved
            from a per-process registry — deserialization needs no engine
            config. For built-in/sklearn names the declared direction is
            cross-checked against their fixed semantics.
        runs: The per-algorithm / per-phase records, in a deterministic
            order (phase name, then configuration order).
        winner: The winner of the terminal select step (``output: winner``).
            Not required to name one of the ``runs`` records: the select
            step may report the winner under its own name. Normally
            ``None`` for a failed experiment (all algorithms failed), but
            not forbidden: a run may fail after a winner was already
            determined (e.g. the final artifact save crashed) — in that
            case the winner is still reported, together with its score and
            the failure message in ``error``.
        winner_score: The final score of the winner in user semantics.
        winner_params: The winner's best hyperparameters (a
            JSON-serializable mapping; tuples become lists after a JSON
            round-trip).
        model_path: The path of the saved model artifact.
        additional_metrics: The informational metrics of the final model
            (name → value in the metric's own user semantics).
        disqualified_algorithms: The circuit-breaker map (name → reason).
        sanity_gate: The report sections of the ``sanity_gate`` select steps
            (a JSON-serializable mapping).
        complexity_guard: The aggregated report section of the adaptive
            complexity guard (a JSON-serializable mapping).
        ensemble: The report section of the experimental top-K ensemble
            (``general.ensemble``): the members (rank, algorithm, score,
            weight), the effective size, the weight method and the
            ensemble artifact path (a JSON-serializable mapping; ``None``
            when the ensemble was not built).
        oof_score: The comparison metric on the final pipeline's pooled OOF
            predictions (optimistic, not an independent estimate).
        holdout_score: The honest out-of-sample estimate on the reserved
            outer hold-out.
        bbc_score: The bias-corrected (BBC-CV) estimate of the selection
            procedure.
        n_trials_evaluated: The number of trials actually evaluated by the
            winner's HPO phase.
        error: The failure message for ``status='failed'``.
    """

    run_id: uuid.UUID
    label: str | None = None
    engine_version: str | None = None
    started_at: dt.datetime
    finished_at: dt.datetime
    duration_seconds: float = Field(ge=0.0)
    status: RunStatus

    config: dict[str, Any]

    resolved_validation: str
    resolved_n_folds: int = Field(ge=1)
    resolved_test_size: float | int
    random_state: int | None = None

    metric_user: str
    metric_sklearn: str
    metric_direction: MetricDirection

    runs: list[AlgorithmRun] = Field(default_factory=list)

    winner: str | None = None
    winner_score: ScoreValueOrNone = None
    winner_params: dict[str, Any] = Field(default_factory=dict)
    model_path: str | None = None
    additional_metrics: dict[str, float] = Field(default_factory=dict)
    disqualified_algorithms: dict[str, str] = Field(default_factory=dict)
    sanity_gate: dict[str, Any] | None = None
    complexity_guard: dict[str, Any] | None = None
    ensemble: dict[str, Any] | None = None
    oof_score: ScoreValueOrNone = None
    holdout_score: ScoreValueOrNone = None
    bbc_score: ScoreValueOrNone = None
    n_trials_evaluated: int | None = Field(default=None, ge=0)
    error: str | None = None

    @field_validator("run_id")
    @classmethod
    def _run_id_not_nil(cls, value: uuid.UUID) -> uuid.UUID:
        if value == uuid.UUID(int=0):
            raise ValueError("run_id must not be the nil UUID")
        return value

    @field_validator("started_at", "finished_at")
    @classmethod
    def _aware_datetime_to_utc(cls, value: dt.datetime) -> dt.datetime:
        """Require a timezone-aware timestamp and normalize it to UTC.

        Consumers (the experiment store, reporters) subtract and compare
        the timestamps directly; a naive datetime would raise ``TypeError``
        against an aware one, so the naive form is rejected and every
        stored timestamp is normalized to UTC.
        """
        if value.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware (e.g. datetime.now(dt.UTC))"
            )
        return value.astimezone(dt.UTC)

    @field_validator("config")
    @classmethod
    def _config_is_json_safe_mapping(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            # ValueError (not TypeError): the value flows through a Pydantic
            # validator, where a TypeError would surface as an unrelated
            # "error converting" message instead of a validation error.
            raise ValueError(  # noqa: TRY004
                "config must be a mapping (a JSON-serializable snapshot)"
            )
        _check_json_safe(value, "config")
        return value

    @field_validator("winner_params")
    @classmethod
    def _winner_params_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        _check_json_safe(value, "winner_params")
        return value

    @field_validator("sanity_gate", "complexity_guard", "ensemble")
    @classmethod
    def _report_sections_json_safe(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if value is not None:
            _check_json_safe(value, "report section")
        return value

    @field_validator("resolved_validation")
    @classmethod
    def _resolved_validation_known(cls, value: str) -> str:
        allowed = {s.value for s in ValidationStrategy} - {"auto"}
        if value not in allowed:
            raise ValueError(
                f"resolved_validation must be one of {sorted(allowed)}, got {value!r}"
            )
        return value

    @field_validator("resolved_test_size", mode="before")
    @classmethod
    def _resolved_test_size_valid(cls, value: Any) -> float | int:
        """Validate the resolved hold-out size.

        The engine resolves 'auto' into one of two shapes: a hold-out
        fraction in (0, 1) for an explicit ``train_test_split`` strategy, or
        an integer number of test rows (≥ 1) for ``auto`` →
        ``train_test_split``. Anything else (NaN/±inf, non-positive values,
        bool, fractions ≥ 1) cannot come from the resolution and is
        rejected. numpy real scalars are accepted and normalized like the
        score fields (``np.int64`` rows / ``np.float32`` fractions).
        """
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            # ValueError (not TypeError): the value flows through a Pydantic
            # validator, where a TypeError would surface as an unrelated
            # "error converting" message instead of a validation error.
            raise ValueError(  # noqa: TRY004
                f"resolved_test_size must be a fraction in (0, 1) or an "
                f"integer row count ≥ 1, got {value!r}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"resolved_test_size must be finite, got {value!r}")
        if isinstance(value, (int, np.integer)):
            if int(value) < 1:
                raise ValueError(
                    f"resolved_test_size as a row count must be ≥ 1, got {value!r}"
                )
            return int(value)
        if not 0.0 < float(value) < 1.0:
            raise ValueError(
                f"resolved_test_size as a fraction must be in (0, 1), got {value!r}"
            )
        return float(value)

    @field_validator("metric_user")
    @classmethod
    def _metric_user_known(cls, value: str) -> str:
        to_sklearn_name(value)  # registry-independent: never raises, normalizes case
        return value

    @model_validator(mode="after")
    def _metric_sklearn_matches_mapping(self) -> ExperimentResult:
        """The sklearn name must be the mapping of the user metric name.

        ``to_sklearn_name`` is the single conversion point and is
        registry-independent (a pure alias-map lookup with a lowercase
        fallback), so the check holds in every consumer process. A diverging
        ``metric_sklearn`` would mislabel the metric for the consumers.
        """
        expected = to_sklearn_name(self.metric_user)
        if self.metric_sklearn != expected:
            raise ValueError(
                f"metric_sklearn {self.metric_sklearn!r} does not match the "
                f"sklearn name of the metric '{self.metric_user}' "
                f"(expected {expected!r})"
            )
        return self

    @field_validator("additional_metrics", mode="before")
    @classmethod
    def _validate_additional_metrics(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            # ValueError (not TypeError): the value flows through a Pydantic
            # validator, where a TypeError would surface as an unrelated
            # "error converting" message instead of a validation error.
            raise ValueError(  # noqa: TRY004
                "additional_metrics must be a mapping of metric name → value"
            )
        return {name: _check_score_validity(raw) for name, raw in value.items()}

    @model_validator(mode="after")
    def _consistency(self) -> ExperimentResult:
        """Cross-field consistency of the result.

        A completed experiment has a winner; a failed one may not (all
        algorithms failed). ``winner_score`` follows ``winner``. For a
        metric whose semantics the model can know statically (a built-in or
        sklearn-registry name — never a per-process custom metric), the
        declared direction must match that semantics; a custom metric
        carries its direction in the payload itself and is not re-resolved,
        so deserialization never needs the producer's registry.
        """
        if self.metric_user.lower() not in reserved_metric_names():
            # A per-process custom metric: the live scorer registry must not
            # be consulted here (a consumer process does not have it).
            pass
        else:
            expected_direction = MetricDirection(user_direction(self.metric_user))
            if self.metric_direction is not expected_direction:
                raise ValueError(
                    f"metric_direction {self.metric_direction.value!r} does not "
                    f"match the metric '{self.metric_user}' "
                    f"(expected {expected_direction.value!r})"
                )
        if self.status is RunStatus.completed:
            if not self.winner:
                raise ValueError(
                    "A completed experiment must have a winner "
                    "(the terminal select step output)"
                )
            if self.winner_score is None:
                raise ValueError("A completed experiment must report the winner score")
        elif self.winner_score is not None and not self.winner:
            raise ValueError(
                "winner_score without a winner is not allowed: the score is "
                "the output of the terminal select step"
            )
        if (self.finished_at - self.started_at).total_seconds() < 0:
            raise ValueError("finished_at must not be earlier than started_at")
        if self.ensemble is not None and not (self.model_path or "").strip():
            # The ensemble artifact is saved next to the winner artifact
            # and its section records that path: a section without a
            # winner artifact path cannot come from the engine assembly
            # (the ensemble step degrades to "no section" when the
            # result carries no 'model_path'). Rejecting the section at
            # the result boundary turns a silent degradation — an
            # ensemble reported without a reloadable artifact — into an
            # explicit construction error.
            raise ValueError(
                "An ensemble section without 'model_path' is not allowed: "
                "the ensemble artifact is saved next to the winner artifact "
                "and the result must record that path"
            )
        expected_duration = (self.finished_at - self.started_at).total_seconds()
        # The duration and the timestamps may come from different clocks
        # (e.g. time.monotonic() vs datetime.now(UTC)); allow a small
        # tolerance for the skew instead of requiring a microsecond match.
        if self.duration_seconds + _DURATION_TOLERANCE_SECONDS < expected_duration:
            raise ValueError(
                "duration_seconds is smaller than finished_at - started_at "
                f"({self.duration_seconds!r} < {expected_duration!r})"
            )
        if self.status is RunStatus.failed and not (self.error and self.error.strip()):
            raise ValueError(
                "A failed experiment must carry a failure message in 'error'"
            )
        return self

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the result to a JSON string.

        Returns:
            str: The JSON representation (all values JSON-safe).
        """
        return self.model_dump_json(indent=indent, by_alias=False)

    @classmethod
    def from_json(cls, payload: str) -> ExperimentResult:
        """Restore the result from a JSON string (the ``to_json`` inverse).

        Returns:
            ExperimentResult: The restored model.
        """
        return cls.model_validate_json(payload)
