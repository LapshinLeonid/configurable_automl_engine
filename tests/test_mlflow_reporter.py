"""Tests of the optional MLflow reporter adapter
(``experiment.reporting.mlflow_reporter``).

The tests run without the ``mlflow`` package installed: the adapter's
``mlflow`` module is faked (a ``types.ModuleType`` with recording
stubs) and injected through ``monkeypatch`` of
``importlib.import_module`` + ``sys.modules``. The suite is skipped
when the package itself cannot be imported — never because a *test*
needs MLflow (the adapter, not the tests, owns the dependency).

Covered scenarios (positive / negative / boundary):

- P1: metrics, params, tags and artifacts land in the run;
- P2: the Engine → ExperimentResult → MLflowReporter chain works with
  no core changes (a full validated result is logged; the core modules
  import no mlflow anywhere — regression AST check over
  ``training_engine/``, ``hpo/``, ``tuner.py``);
- N1: mlflow not installed → a defined :class:`MLflowReporterError`
  naming the ``[mlflow]`` extra, the experiment continues (the error is
  catchable, not a bare ``ModuleNotFoundError``);
- N2: a tracking-server failure (the fake raises) → ``report`` returns
  ``None``, logs a warning, does not raise;
- B1: a NaN/±inf value in ``additional_metrics`` → skipped with a
  warning, the finite metrics are still logged;
- B2: the nested-run policy (``nested=True`` starts a run named after
  the experiment ``run_id``; ``nested=False`` without an active run →
  the defined error);
- B3: duplicate parameter names across phases → unique prefixed keys;
- B4: an empty leaderboard → logged as a placeholder artifact, no
  exception.
"""

from __future__ import annotations

import ast
import datetime as dt
import importlib
import json
import re
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

import configurable_automl_engine
from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment import AlgorithmRun, ExperimentResult
from configurable_automl_engine.experiment.reporting import (
    MLflowReporter,
    MLflowReporterError,
)

UTC = dt.UTC

PACKAGE_ROOT = Path(configurable_automl_engine.__file__).parent


# ──────────────────────────── fakes ───────────────────────────────────── #


class FakeMLflow:
    """A minimal recording stand-in for the ``mlflow`` module API.

    Records every ``log_metric`` / ``log_param`` / ``set_tag`` /
    ``log_text`` call and emulates the ``start_run`` context manager
    (including the nested-run bookkeeping the adapter relies on).
    """

    def __init__(self, *, active: bool = True, fail: Exception | None = None) -> None:
        self.metrics: dict[str, float] = {}
        self.params: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.started_runs: list[dict[str, Any]] = []
        self._active = active
        self._fail = fail

    # -- mlflow module API ------------------------------------------------ #

    def active_run(self) -> Any:
        """Emulate ``mlflow.active_run``."""
        if not self._active:
            return None
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="active-run"))

    def start_run(self, *, run_name: str | None = None, nested: bool = False) -> Any:
        """Emulate ``mlflow.start_run`` (the recording context manager)."""
        self.started_runs.append({"run_name": run_name, "nested": nested})
        if self._fail is not None:
            raise self._fail

        class _Run:
            """The ``ActiveRun`` context-manager stand-in."""

            def __init__(inner: Any, mlflow: FakeMLflow) -> None:
                inner._mlflow = mlflow

            def __enter__(inner: Any) -> Any:
                return types.SimpleNamespace(
                    info=types.SimpleNamespace(run_id="new-run")
                )

            def __exit__(inner: Any, *exc: object) -> bool:
                return False

        return _Run(self)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_text(self, text: str, artifact_file: str) -> None:
        self.texts[artifact_file] = text


@pytest.fixture()
def fake_mlflow() -> FakeMLflow:
    """The fake module injected as ``mlflow`` for one test."""
    fake = FakeMLflow()
    return fake


def _inject_mlflow(monkeypatch: pytest.MonkeyPatch, fake: FakeMLflow) -> None:
    """Make the adapter's lazy import resolve to ``fake``."""
    module = types.ModuleType("mlflow")
    for name in (
        "active_run",
        "start_run",
        "log_metric",
        "log_params",
        "set_tag",
        "log_text",
    ):
        # Late binding: a test may replace a method on ``fake`` after
        # the injection (e.g. to make log_params fail).
        setattr(
            module,
            name,
            lambda *args, _name=name, _fake=fake, **kwargs: getattr(_fake, _name)(
                *args, **kwargs
            ),
        )
    reporter_module = importlib.import_module(
        "configurable_automl_engine.experiment.reporting.mlflow_reporter"
    )
    monkeypatch.setattr(reporter_module.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(reporter_module, "is_installed", lambda name: True)


# ─────────────────────────── fixtures ─────────────────────────────────── #


def _ts(seconds: float = 0.0) -> dt.datetime:
    return dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + dt.timedelta(seconds=seconds)


def _run_kwargs(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "algorithm": "ridge",
        "phase_name": "search",
        "phase_action": RunPhaseAction.all_algorithms,
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


def _make_result(runs: list[dict[str, Any]], **overrides: Any) -> ExperimentResult:
    values: dict[str, Any] = {
        "run_id": uuid.uuid4(),
        "label": "mlflow run",
        "engine_version": "0.0.0",
        "started_at": _ts(),
        "finished_at": _ts(90.0),
        "duration_seconds": 90.0,
        "status": RunStatus.completed,
        "config": {},
        "resolved_validation": "k_fold",
        "resolved_n_folds": 5,
        "resolved_test_size": 0.2,
        "random_state": 42,
        "metric_user": "rmse",
        "metric_sklearn": "neg_root_mean_squared_error",
        "metric_direction": MetricDirection.minimize,
        "runs": [AlgorithmRun(**r) for r in runs],
        "winner": "ridge",
        "winner_score": 0.5,
        "winner_params": {"alpha": 0.1},
        "holdout_score": 0.55,
        "bbc_score": 0.53,
        "oof_score": 0.48,
    }
    values.update(overrides)
    return ExperimentResult(**values)


_COMPLETED_RUNS = [
    _run_kwargs(algorithm="ridge", score=0.774, score_raw=-0.774),
    _run_kwargs(algorithm="random_forest", score=0.842, score_raw=-0.842),
]


# ────────────────────────── positive scenarios ────────────────────────── #


class TestP1MetricsParamsArtifacts:
    """P1: metrics, params, tags and artifacts land in the run."""

    def test_metrics_logged(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS)

        artifact = MLflowReporter().report(result)

        assert artifact == "active-run"
        assert fake_mlflow.metrics["automl.winner_score_rmse"] == 0.5
        assert fake_mlflow.metrics["automl.holdout_score_rmse"] == 0.55
        assert fake_mlflow.metrics["automl.bbc_score_rmse"] == 0.53
        assert fake_mlflow.metrics["automl.oof_score_rmse"] == 0.48
        assert fake_mlflow.metrics["automl.duration_seconds"] == 90.0
        assert fake_mlflow.metrics["automl.score.ridge.search"] == 0.774
        assert fake_mlflow.metrics["automl.score.random_forest.search"] == 0.842

    def test_params_logged(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS)

        MLflowReporter().report(result)

        assert fake_mlflow.params["automl.param.alpha"] == "0.1"
        assert fake_mlflow.params["automl.resolved_validation"] == "k_fold"
        assert fake_mlflow.params["automl.resolved_n_folds"] == "5"
        assert fake_mlflow.params["automl.random_state"] == "42"

    def test_tags_logged(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS)

        MLflowReporter().report(result)

        assert fake_mlflow.tags["automl.run_id"] == str(result.run_id)
        assert fake_mlflow.tags["automl.status"] == "completed"
        assert fake_mlflow.tags["automl.metric"] == "rmse"
        assert fake_mlflow.tags["automl.metric_direction"] == "minimize"
        assert fake_mlflow.tags["automl.label"] == "mlflow run"

    def test_artifacts_logged(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS)

        MLflowReporter().report(result)

        assert set(fake_mlflow.texts) == {
            "experiment/leaderboard.txt",
            "experiment/report.json",
            "experiment/report.md",
        }
        leaderboard = fake_mlflow.texts["experiment/leaderboard.txt"]
        assert "ridge" in leaderboard and "random_forest" in leaderboard
        payload = json.loads(fake_mlflow.texts["experiment/report.json"])
        assert payload == result.model_dump(mode="json")
        assert (
            "# Experiment report: mlflow run"
            in fake_mlflow.texts["experiment/report.md"]
        )

    def test_artifacts_disabled(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)

        MLflowReporter(log_artifacts=False).report(_make_result(_COMPLETED_RUNS))

        assert fake_mlflow.texts == {}
        # Regression guard: log_artifacts=False must not swallow the
        # metrics and params — only the artifacts are switched off.
        assert fake_mlflow.metrics["automl.winner_score_rmse"] == 0.5
        assert fake_mlflow.params["automl.param.alpha"] == "0.1"


class TestP2EngineChain:
    """P2: the Engine → ExperimentResult → MLflowReporter chain, no core
    changes — and no mlflow import anywhere in the core."""

    def test_full_result_logged_without_core_changes(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        # A realistic full result (per-phase records, additional metrics,
        # honest scores) is logged end-to-end.
        result = _make_result(
            [
                _run_kwargs(algorithm="ridge", phase_name="search"),
                _run_kwargs(
                    algorithm="ridge",
                    phase_name="refine",
                    phase_action=RunPhaseAction.refine,
                    score=0.55,
                    score_raw=-0.55,
                    params={"alpha": 0.01},
                ),
                _run_kwargs(
                    algorithm="broken",
                    status=RunStepStatus.disqualified,
                    score=None,
                    score_raw=None,
                    error="circuit breaker",
                ),
            ],
            additional_metrics={"mae": 1.234},
        )

        artifact = MLflowReporter().report(result)

        assert artifact == "active-run"
        assert fake_mlflow.metrics["automl.winner_score_rmse"] == 0.5
        assert fake_mlflow.metrics["automl.additional.mae"] == 1.234
        assert fake_mlflow.metrics["automl.score.ridge.refine"] == 0.55
        # A disqualified record carries no score metric.
        assert not any(
            key.startswith("automl.score.broken") for key in fake_mlflow.metrics
        )
        # The refine phase's params are prefixed with algorithm:phase.
        assert fake_mlflow.params["automl.param.ridge__refine.alpha"] == "0.01"

    def test_core_never_imports_mlflow(self) -> None:
        """Regression: no ``import mlflow`` in the engine core.

        Only the adapter (and this test module) may mention mlflow
        imports; ``training_engine/``, ``hpo/`` and ``tuner.py`` must
        stay free of them.
        """
        forbidden = [
            PACKAGE_ROOT / "training_engine",
            PACKAGE_ROOT / "hpo",
            PACKAGE_ROOT / "tuner.py",
        ]
        offenders: list[str] = []
        for path in forbidden:
            files = [path] if path.is_file() else sorted(path.rglob("*.py"))
            for file in files:
                tree = ast.parse(file.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        offenders.extend(
                            f"{file}: import {alias.name}"
                            for alias in node.names
                            if alias.name == "mlflow"
                            or alias.name.startswith("mlflow.")
                        )
                    elif isinstance(node, ast.ImportFrom) and (
                        node.module == "mlflow"
                        or (node.module or "").startswith("mlflow.")
                    ):
                        offenders.append(f"{file}: from {node.module} import …")
        assert offenders == [], "\n".join(offenders)


# ────────────────────────── negative scenarios ────────────────────────── #


class TestN1MLflowMissing:
    """N1: mlflow not installed → a defined error, the experiment goes on."""

    def test_missing_package_raises_mlflow_reporter_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "configurable_automl_engine.experiment.reporting.mlflow_reporter"
            ".is_installed",
            lambda name: False,
        )

        with pytest.raises(MLflowReporterError, match=r"\[mlflow\] extra"):
            MLflowReporter().report(_make_result(_COMPLETED_RUNS))

    def test_error_is_not_module_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defined error is catchable (the experiment continues)."""
        monkeypatch.setattr(
            "configurable_automl_engine.experiment.reporting.mlflow_reporter"
            ".is_installed",
            lambda name: False,
        )

        try:
            MLflowReporter().report(_make_result(_COMPLETED_RUNS))
        except MLflowReporterError:
            pass  # caught: the caller keeps going, exactly the policy
        else:
            pytest.fail("MLflowReporterError was not raised")

    def test_no_mlflow_left_in_sys_modules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The missing dependency never pollutes ``sys.modules``."""
        monkeypatch.setattr(
            "configurable_automl_engine.experiment.reporting.mlflow_reporter"
            ".is_installed",
            lambda name: False,
        )
        monkeypatch.delitem(sys.modules, "mlflow", raising=False)

        with pytest.raises(MLflowReporterError):
            MLflowReporter().report(_make_result(_COMPLETED_RUNS))

        assert "mlflow" not in sys.modules


class TestN2TrackingServerFailure:
    """N2: a tracking-server failure → the experiment does not fall."""

    def test_failure_returns_none_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The failure hits inside the nested run (the same path a
        # tracking-server outage takes in the real client).
        fake = FakeMLflow(fail=ConnectionError("tracking server is down"))
        _inject_mlflow(monkeypatch, fake)

        with caplog.at_level("WARNING"):
            artifact = MLflowReporter(nested=True).report(_make_result(_COMPLETED_RUNS))

        assert artifact is None
        assert any(
            "continues without MLflow tracking" in record.message
            for record in caplog.records
        )

    def test_active_run_failure_returns_none_and_warns(
        self,
        fake_mlflow: FakeMLflow,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The same policy on the active-run path: a logging failure
        (e.g. an oversized param batch) does not raise."""
        _inject_mlflow(monkeypatch, fake_mlflow)

        def _fail_log_params(params: dict[str, str]) -> None:
            raise ConnectionError("tracking server is down")

        fake_mlflow.log_params = _fail_log_params  # type: ignore[method-assign]

        with caplog.at_level("WARNING"):
            artifact = MLflowReporter().report(_make_result(_COMPLETED_RUNS))

        assert artifact is None
        assert any(
            "continues without MLflow tracking" in record.message
            for record in caplog.records
        )

    def test_usage_error_still_raises(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A usage failure (no active run) raises instead of being
        swallowed by the failure policy."""
        fake = FakeMLflow(active=False)
        _inject_mlflow(monkeypatch, fake)

        with pytest.raises(MLflowReporterError, match="no active MLflow run"):
            MLflowReporter().report(_make_result(_COMPLETED_RUNS))


# ────────────────────────── boundary scenarios ────────────────────────── #


class TestB1NonFiniteMetrics:
    """B1: NaN/±inf in ``additional_metrics`` → skipped with a warning."""

    def test_nan_metric_skipped(
        self,
        fake_mlflow: FakeMLflow,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS, additional_metrics={"ok": 1.0})
        # The issue case: sklearn produced a NaN additional metric. The
        # model validation rejects a non-finite value (T1), so the NaN
        # is injected past validation — exactly the payload a hand-
        # assembled / partially-updated result could carry.
        object.__setattr__(
            result, "additional_metrics", {"ok": 1.0, "nan_metric": float("nan")}
        )

        with caplog.at_level("WARNING"):
            MLflowReporter().report(result)

        assert "automl.additional.nan_metric" not in fake_mlflow.metrics
        assert fake_mlflow.metrics["automl.additional.ok"] == 1.0
        assert any("non-finite" in record.message for record in caplog.records)

    def test_inf_metric_skipped_finite_kept(
        self,
        fake_mlflow: FakeMLflow,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS, additional_metrics={"ok": 1.5})
        object.__setattr__(
            result, "additional_metrics", {"ok": 1.5, "inf_metric": float("inf")}
        )

        with caplog.at_level("WARNING"):
            MLflowReporter().report(result)

        assert "automl.additional.inf_metric" not in fake_mlflow.metrics
        assert fake_mlflow.metrics["automl.additional.ok"] == 1.5
        # The experiment's own scores are untouched.
        assert fake_mlflow.metrics["automl.winner_score_rmse"] == 0.5


class TestB2RunPolicies:
    """B2: the nested-run policy."""

    def test_nested_true_starts_named_run(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS)

        MLflowReporter(nested=True).report(result)

        assert len(fake_mlflow.started_runs) == 1
        call = fake_mlflow.started_runs[0]
        assert call["nested"] is True
        assert call["run_name"] == str(result.run_id)

    def test_nested_false_logs_into_active_run(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)

        MLflowReporter(nested=False).report(_make_result(_COMPLETED_RUNS))

        assert fake_mlflow.started_runs == []  # no run of its own
        assert fake_mlflow.metrics["automl.winner_score_rmse"] == 0.5

    def test_nested_true_without_parent_is_top_level(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an active parent the nested run is promoted to
        top-level (MLflow's own semantics) — the report still works."""
        fake = FakeMLflow(active=False)
        _inject_mlflow(monkeypatch, fake)

        artifact = MLflowReporter(nested=True).report(_make_result(_COMPLETED_RUNS))

        assert artifact == "new-run"
        assert fake.metrics["automl.winner_score_rmse"] == 0.5


class TestB3DuplicateParamNames:
    """B3: duplicate parameter names across phases → unique keys."""

    def test_same_param_name_in_two_phases(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(
            [
                _run_kwargs(
                    algorithm="ridge",
                    phase_name="search",
                    params={"alpha": 0.1, "fit_intercept": True},
                ),
                _run_kwargs(
                    algorithm="ridge",
                    phase_name="refine",
                    phase_action=RunPhaseAction.refine,
                    params={"alpha": 0.01},
                ),
                _run_kwargs(
                    algorithm="forest",
                    phase_name="search",
                    params={"alpha": 2.0},
                ),
            ]
        )

        MLflowReporter().report(result)

        # Distinct keys per algorithm:phase — no silent overwrite.
        assert fake_mlflow.params["automl.param.ridge__search.alpha"] == "0.1"
        assert fake_mlflow.params["automl.param.ridge__refine.alpha"] == "0.01"
        assert fake_mlflow.params["automl.param.forest__search.alpha"] == "2.0"
        assert fake_mlflow.params["automl.param.ridge__search.fit_intercept"] == "true"

    def test_winner_params_logged_unprefixed(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)

        MLflowReporter().report(_make_result(_COMPLETED_RUNS))

        assert fake_mlflow.params["automl.param.alpha"] == "0.1"


class TestB4EmptyLeaderboard:
    """B4: an empty leaderboard → a placeholder artifact, no exception."""

    def test_failed_experiment_logs_placeholder(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(
            [],
            status=RunStatus.failed,
            error="all algorithms failed",
            winner=None,
            winner_score=None,
            holdout_score=None,
            bbc_score=None,
            oof_score=None,
        )

        artifact = MLflowReporter().report(result)

        assert artifact == "active-run"
        leaderboard = fake_mlflow.texts["experiment/leaderboard.txt"]
        # An empty leaderboard is logged as its placeholder summary —
        # a real artifact, not an exception.
        assert leaderboard  # non-empty text
        assert fake_mlflow.tags["automl.status"] == "failed"
        # No score metrics for a failed run (only the duration).
        assert fake_mlflow.metrics == {"automl.duration_seconds": 90.0}


class TestExtra:
    """Extra policies: sanitization of param strings, key stability."""

    def test_nested_param_dict_is_flattened(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(
            [
                _run_kwargs(
                    params={"model": {"alpha": 0.1, "depth": 5}},
                )
            ]
        )

        MLflowReporter().report(result)

        assert fake_mlflow.params["automl.param.ridge__search.model.alpha"] == "0.1"
        assert fake_mlflow.params["automl.param.ridge__search.model.depth"] == "5"

    def test_unicode_params_and_tags(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(
            [_run_kwargs(algorithm="модель-1", params={"альфа": 0.5})],
            label="Эксперимент «Титаник»",
        )

        MLflowReporter().report(result)

        assert fake_mlflow.params["automl.param.модель-1__search.альфа"] == "0.5"
        assert fake_mlflow.tags["automl.label"] == "Эксперимент «Титаник»"
        assert "модель-1" in fake_mlflow.texts["experiment/leaderboard.txt"]

    def test_result_without_scores(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A result without any honest/score values logs no score metrics
        (only the duration, which is always known)."""
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(
            [],
            status=RunStatus.failed,
            error="no winner",
            winner=None,
            winner_score=None,
            holdout_score=None,
            bbc_score=None,
            oof_score=None,
        )

        MLflowReporter().report(result)

        assert fake_mlflow.metrics == {"automl.duration_seconds": 90.0}

    def test_long_phase_name_key_is_clamped(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B5: a very long user-configured phase name is clamped.

        MLflow rejects an entity key over 250 characters
        (``MAX_ENTITY_KEY_LENGTH``); a 300-character phase name must not
        make ``log_params`` fail (which, per the failure policy, would
        degrade the whole report to ``None``) — the phase segment is cut
        to head + ellipsis + tail, and every generated key stays within
        the limit.
        """
        _inject_mlflow(monkeypatch, fake_mlflow)
        long_phase = "p" * 300
        result = _make_result(
            [
                _run_kwargs(
                    algorithm="ridge",
                    phase_name=long_phase,
                    params={"alpha": 0.1},
                )
            ]
        )

        artifact = MLflowReporter().report(result)

        assert artifact == "active-run"  # no tracking failure
        clamped_keys = [
            key for key in fake_mlflow.params if key.startswith("automl.param.ridge__")
        ]
        assert clamped_keys, "the clamped per-phase param key is missing"
        assert all(len(key) <= 250 for key in clamped_keys)
        assert "..." in clamped_keys[0]

    def test_long_flattened_param_key_is_clamped(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B5b: a long nested parameter sub-key is clamped too."""
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(
            [
                _run_kwargs(
                    params={"deep": {"k" * 300: 0.5}},
                )
            ]
        )

        artifact = MLflowReporter().report(result)

        assert artifact == "active-run"
        clamped_keys = [key for key in fake_mlflow.params if "deep." in key]
        assert clamped_keys
        assert all(len(key) <= 250 for key in clamped_keys)

    def test_long_additional_metric_name_is_clamped(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B5c: a long additional-metric name is clamped."""
        _inject_mlflow(monkeypatch, fake_mlflow)
        result = _make_result(_COMPLETED_RUNS, additional_metrics={"m" * 300: 1.0})

        artifact = MLflowReporter().report(result)

        assert artifact == "active-run"
        clamped = [
            key for key in fake_mlflow.metrics if key.startswith("automl.additional.")
        ]
        assert clamped
        assert len(clamped[0]) <= 250
        assert "..." in clamped[0]

    def test_normal_length_keys_are_not_clamped(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B5d: ordinary keys pass through unchanged (no ellipsis)."""
        _inject_mlflow(monkeypatch, fake_mlflow)

        MLflowReporter().report(_make_result(_COMPLETED_RUNS))

        assert "automl.param.alpha" in fake_mlflow.params
        assert "automl.param.ridge__search.alpha" in fake_mlflow.params
        assert not any("..." in key for key in fake_mlflow.params)

    def test_composed_key_over_limit_is_clamped(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B5e: the *composed* key is clamped to the 250-char limit.

        Every segment individually within the clamping budget can still
        sum over the limit (``automl.param.`` + a long algorithm name +
        ``__`` + a ~105-char phase + ``.`` + a ~105-char sub-key > 250):
        the final whole-key guard must catch it, otherwise ``log_params``
        fails and — per the failure policy — the whole report degrades
        to ``None``.
        """
        _inject_mlflow(monkeypatch, fake_mlflow)
        algorithm = "nearest_neighbors_regression"  # 30 chars
        phase = "p" * 105  # within the segment budget (no ellipsis)
        result = _make_result(
            [
                _run_kwargs(
                    algorithm=algorithm,
                    phase_name=phase,
                    params={"k" * 105: 0.5},
                )
            ]
        )

        artifact = MLflowReporter().report(result)

        # Without the whole-key guard log_params would raise here and
        # the report would degrade to None (the round-4 finding).
        assert artifact == "active-run"
        assert all(len(key) <= 250 for key in fake_mlflow.params)
        clamped = [
            key
            for key in fake_mlflow.params
            if key.startswith("automl.param.") and algorithm in key
        ]
        assert clamped
        # The composed key is tail-cut (no ellipsis marker by design).
        assert all(len(key) == 250 for key in clamped)

    def test_all_generated_keys_pass_mlflow_name_validation(
        self, fake_mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B5f: every generated key passes MLflow's name validation.

        MLflow validates every param/metric name against
        ``^[/\\w.\\- :]*$`` (``validate_param_and_metric_name``); the
        round-5 regression showed the unicode ellipsis ``…`` in a
        clamped key fails that regex and — via the failure policy —
        degrades the whole report to ``None``, which the fake-module
        tests alone cannot catch (``FakeMLflow`` does not validate
        names). This test runs every key of a clamped report through
        the same regex.
        """
        _inject_mlflow(monkeypatch, fake_mlflow)
        # Force every clamping path: long phase, long nested sub-key,
        # long additional-metric name, Cyrillic phase.
        result = _make_result(
            [
                _run_kwargs(
                    algorithm="ridge",
                    phase_name="ф" * 300,
                    params={"deep": {"k" * 300: 0.5}},
                ),
                _run_kwargs(
                    algorithm="ridge",
                    phase_name="p" * 105,
                    params={"k" * 105: 0.5},
                ),
            ],
            additional_metrics={"m" * 300: 1.0},
        )

        artifact = MLflowReporter().report(result)

        assert artifact == "active-run"
        name_pattern = re.compile(r"^[/\w.\- :]*$")
        invalid = {
            kind: [key for key in keys if not name_pattern.fullmatch(key)]
            for kind, keys in (
                ("params", fake_mlflow.params),
                ("metrics", fake_mlflow.metrics),
                ("tags", fake_mlflow.tags),
            )
        }
        assert not any(invalid.values()), invalid
