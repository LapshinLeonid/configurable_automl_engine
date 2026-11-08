"""Tests of the reporter framework (``experiment.reporting``).

Covered scenarios (positive / negative / boundary):

- P1: the JSON report parses back into a dict equivalent to the model
  dump (round-trip);
- P2: the Markdown report contains the leaderboard, the winner and the
  per-run details;
- P3: several reporters run in one composite call;
- N1: a non-serializable value smuggled into the result → a clear
  ``TypeError``, the destination file is not left partial;
- N2: an unwritable destination → a defined error;
- B1: an empty (failed) experiment → both reports are still produced;
- B2: special characters and unicode in names/parameters;
- B3: an existing report file is overwritten;
- B4: a very large result (thousands of records) — the JSON closes
  correctly;
- B5: numpy scalars / Path / Enum / pd.Timestamp in the result fields
  serialize to JSON types;
- B6: missing parent directories of the JSON destination are created;
- B7: the Markdown table survives pipe/newline characters in the names.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment import (
    AlgorithmRun,
    ExperimentResult,
)
from configurable_automl_engine.experiment.reporting import (
    CompositeReporter,
    JSONReporter,
    MarkdownReporter,
    Reporter,
    ReporterArtifact,
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


def _result_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default keyword arguments of a completed ``ExperimentResult``."""
    values: dict[str, Any] = {
        "run_id": uuid.uuid4(),
        "label": "reporter run",
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
        "runs": [],
        "winner": "ridge",
        "winner_score": 0.5,
        "winner_params": {"alpha": 0.1},
        "holdout_score": 0.55,
        "bbc_score": 0.53,
        "oof_score": 0.48,
    }
    values.update(overrides)
    return values


def _make_result(runs: list[dict[str, Any]], **overrides: Any) -> ExperimentResult:
    """Build an ``ExperimentResult`` from ``AlgorithmRun`` kwargs dicts."""
    values = _result_kwargs(runs=[AlgorithmRun(**r) for r in runs])
    values.update(overrides)
    return ExperimentResult(**values)


_COMPLETED_RUNS = [
    _run_kwargs(algorithm="Ridge", score=0.774, score_raw=-0.774),
    _run_kwargs(algorithm="Random Forest", score=0.842, score_raw=-0.842),
    _run_kwargs(algorithm="XGBoost", score=0.831, score_raw=-0.831),
]


# ────────────────────────── positive scenarios ────────────────────────── #


class TestP1JsonRoundTrip:
    """P1: the JSON report parses back into an equivalent model dump."""

    def test_report_writes_valid_json(self, tmp_path: Path) -> None:
        result = _make_result(_COMPLETED_RUNS)
        path = tmp_path / "report.json"

        artifact = JSONReporter(path).report(result)

        assert artifact == path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == result.model_dump(mode="json")

    def test_round_trip_restores_equal_model(self, tmp_path: Path) -> None:
        result = _make_result(_COMPLETED_RUNS)
        path = tmp_path / "report.json"

        artifact = JSONReporter(path).report(result)
        restored = ExperimentResult.from_json(
            Path(artifact).read_text(encoding="utf-8")
        )

        assert restored == result

    def test_json_is_utf8_with_ensure_ascii_false(self, tmp_path: Path) -> None:
        result = _make_result([], label="Отчёт «Эксперимент» №1")
        path = tmp_path / "report.json"

        JSONReporter(path).report(result)

        raw = path.read_bytes()
        assert "Отчёт «Эксперимент» №1".encode("utf-8") in raw
        assert b"\\u" not in raw  # no ASCII-escaped unicode


class TestP2MarkdownReport:
    """P2: the Markdown report contains leaderboard, winner, run details."""

    def test_report_sections(self) -> None:
        result = _make_result(_COMPLETED_RUNS)

        report = MarkdownReporter().report(result)

        assert "# Experiment report: reporter run" in report
        assert "## Leaderboard" in report
        assert "Winner: **ridge**" in report
        assert "## Run details" in report
        for algorithm in ("Ridge", "Random Forest", "XGBoost"):
            assert f"### {algorithm} — phase search (all_algorithms)" in report
        assert "| Rank | Algorithm | Score | Trials | Phase |" in report

    def test_honest_scores_shown_with_semantics(self) -> None:
        result = _make_result(_COMPLETED_RUNS)

        report = MarkdownReporter().report(result)

        assert "### Honest score estimates" in report
        assert "**0.550**" in report  # holdout
        assert "**0.530**" in report  # bbc
        assert "**0.480**" in report  # oof
        assert "optimistic" in report

    def test_winner_in_leaderboard_follows_metric_direction(self) -> None:
        result = _make_result(_COMPLETED_RUNS)

        report = MarkdownReporter().report(result)

        leaderboard_row = next(
            line for line in report.splitlines() if line.startswith("| 1 ")
        )
        assert "Ridge" in leaderboard_row  # smallest rmse → rank 1

    def test_failed_run_reports_status_and_reason(self) -> None:
        result = _make_result(
            [],
            status=RunStatus.failed,
            error="all algorithms failed",
        )

        report = MarkdownReporter().report(result)

        assert "**failed**" in report
        assert "## Failure" in report
        assert "all algorithms failed" in report

    def test_outcome_shows_winner_params(self) -> None:
        result = _make_result([], winner_params={"alpha": 0.1, "fit_intercept": True})

        report = MarkdownReporter().report(result)

        assert "- Winner params: `alpha=0.1, fit_intercept=True`" in report

    def test_metadata_shows_holdout_size_for_train_test_split(self) -> None:
        result = _make_result(
            [], resolved_validation="train_test_split", resolved_test_size=0.2
        )

        report = MarkdownReporter().report(result)

        assert "- Hold-out size: 0.20 of the data" in report

    def test_metadata_shows_row_count_holdout_size(self) -> None:
        result = _make_result(
            [], resolved_validation="train_test_split", resolved_test_size=100
        )

        report = MarkdownReporter().report(result)

        assert "- Hold-out size: 100 rows" in report

    def test_metadata_omits_holdout_size_for_k_fold(self) -> None:
        result = _make_result([])

        report = MarkdownReporter().report(result)

        assert "Hold-out size" not in report


class TestP3Composite:
    """P3: several reporters run in one composite call."""

    def test_composite_returns_all_artifacts(self, tmp_path: Path) -> None:
        result = _make_result(_COMPLETED_RUNS)
        json_path = tmp_path / "report.json"

        class _TextReporter(Reporter):
            def report(self, result: ExperimentResult) -> str:
                return f"text:{result.run_id}"

        composite = CompositeReporter([JSONReporter(json_path), _TextReporter()])

        artifacts = composite.report(result)

        assert artifacts == (json_path, f"text:{result.run_id}")
        assert json.loads(json_path.read_text(encoding="utf-8")) == (
            result.model_dump(mode="json")
        )

    def test_composite_invokes_in_registration_order(self) -> None:
        result = _make_result([])

        class _OrderReporter(Reporter):
            def __init__(self, name: str, calls: list[str]) -> None:
                self._name = name
                self._calls = calls

            def report(self, result: ExperimentResult) -> str:
                self._calls.append(self._name)
                return self._name

        calls: list[str] = []
        composite = CompositeReporter(
            [_OrderReporter("first", calls), _OrderReporter("second", calls)]
        )

        artifacts = composite.report(result)

        assert artifacts == ("first", "second")
        assert calls == ["first", "second"]

    def test_empty_composite_is_a_noop(self) -> None:
        result = _make_result([])

        assert CompositeReporter([]).report(result) == ()

    def test_composite_fail_fast(self) -> None:
        """N-2: the first reporter failure stops the composite.

        A failing first reporter propagates its exception and the second
        reporter is never invoked — a silent half-run of the sinks would
        be worse than an explicit error.
        """
        result = _make_result([])

        class _FailingReporter(Reporter[str]):
            def report(self, result: ExperimentResult) -> str:
                raise RuntimeError("sink is down")

        class _SpyReporter(Reporter[str]):
            def __init__(self) -> None:
                self.called = False

            def report(self, result: ExperimentResult) -> str:
                self.called = True
                return "ok"

        spy = _SpyReporter()
        composite = CompositeReporter([_FailingReporter(), spy])

        with pytest.raises(RuntimeError, match="sink is down"):
            composite.report(result)

        assert not spy.called

    def test_composite_passes_through_the_reported_failed_none_artifact(self) -> None:
        """The base-contract ``None`` artifact passes through the composite.

        A reporter with a documented degradation policy may return
        ``None`` instead of raising (see ``Reporter.report`` in
        ``experiment.reporting.base``; the MLflow adapter implements
        the concrete "reported-failed" policy: a tracking-server
        failure → ``None`` + warning). The composite treats the
        ``None`` artifact as a value, not as a failure: it stays in the
        tuple in the reporter's position and the remaining reporters
        still run.
        """
        result = _make_result([])

        class _DegradedReporter(Reporter[ReporterArtifact | None]):
            """The MLflow-policy stand-in: a ``None`` "reported-failed" artifact."""

            def report(self, result: ExperimentResult) -> ReporterArtifact | None:
                return None

        class _SpyReporter(Reporter[str]):
            def __init__(self) -> None:
                self.called = False

            def report(self, result: ExperimentResult) -> str:
                self.called = True
                return "ok"

        spy = _SpyReporter()
        composite = CompositeReporter([_DegradedReporter(), spy])

        artifacts = composite.report(result)

        assert artifacts == (None, "ok")
        assert spy.called


# ────────────────────────── negative scenarios ────────────────────────── #


class TestN1NonSerializableValue:
    """N1: a non-serializable value → a clear error, no partial file."""

    @pytest.fixture()
    def poisoned_result(self) -> Any:
        """A result carrying a function smuggled past the validation."""
        result = _make_result(_COMPLETED_RUNS)
        object.__setattr__(result, "label", lambda value: value)
        return result

    def test_typeerror_names_the_field(
        self, poisoned_result: Any, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="cannot be serialized"):
            JSONReporter(tmp_path / "report.json").report(poisoned_result)

    def test_no_partial_file_is_left(
        self, poisoned_result: Any, tmp_path: Path
    ) -> None:
        destination = tmp_path / "report.json"
        destination.write_text("previous content", encoding="utf-8")

        with pytest.raises(TypeError):
            JSONReporter(destination).report(poisoned_result)

        assert destination.read_text(encoding="utf-8") == "previous content"
        assert list(tmp_path.iterdir()) == [destination]  # no temp leftovers


class TestN2UnwritableDestination:
    """N2: an unwritable destination → a defined error.

    The destination's parent path cannot be created when one of its
    components is an existing *file*: ``mkdir`` raises
    ``NotADirectoryError`` (an ``OSError`` subclass) — the same defined
    failure a permission-denied directory produces, without relying on
    the process being non-root (a ``chmod``-based fixture silently does
    nothing under root).
    """

    def test_destination_under_a_file_raises_oserror(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(OSError):
            JSONReporter(blocker / "sub" / "report.json").report(
                _make_result(_COMPLETED_RUNS)
            )

    def test_failure_does_not_touch_the_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write failure leaves the existing destination file intact."""
        destination = tmp_path / "report.json"
        destination.write_text("previous content", encoding="utf-8")

        def _fail_write(self: JSONReporter, text: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(JSONReporter, "_atomic_write", _fail_write)

        with pytest.raises(OSError, match="disk full"):
            JSONReporter(destination).report(_make_result(_COMPLETED_RUNS))

        assert destination.read_text(encoding="utf-8") == "previous content"

    def test_write_failure_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B-1: a failure of the write itself leaves no orphan temp file.

        The temp path must be registered (and cleaned up in ``finally``)
        before any write, so a failing ``write()``/``close()`` — a disk
        full, an I/O error — does not leave a hidden partial
        ``.report.json.*.tmp`` next to the destination.
        """
        destination = tmp_path / "report.json"

        real_named_temporary = tempfile.NamedTemporaryFile

        def _failing_named_temporary(*args: Any, **kwargs: Any) -> Any:
            temp = real_named_temporary(*args, **kwargs)
            original_write = temp.write

            def _write_fail(data: str) -> int:
                raise OSError("disk full during write")

            temp.write = _write_fail  # type: ignore[method-assign]
            return temp

        monkeypatch.setattr(
            "configurable_automl_engine.experiment.reporting.json_reporter"
            ".tempfile.NamedTemporaryFile",
            _failing_named_temporary,
        )

        with pytest.raises(OSError, match="disk full during write"):
            JSONReporter(destination).report(_make_result(_COMPLETED_RUNS))

        assert list(tmp_path.iterdir()) == []  # no destination, no temp file


# ────────────────────────── boundary scenarios ────────────────────────── #


class TestB1EmptyExperiment:
    """B1: an empty (failed) experiment → both reports are produced."""

    def test_json_of_failed_experiment(self, tmp_path: Path) -> None:
        result = _make_result(
            [],
            status=RunStatus.failed,
            error="no data",
            winner=None,
            winner_score=None,
        )
        path = tmp_path / "report.json"

        JSONReporter(path).report(result)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == result.model_dump(mode="json")

    def test_markdown_of_failed_experiment(self) -> None:
        result = _make_result(
            [],
            status=RunStatus.failed,
            error="no data",
            winner=None,
            winner_score=None,
        )

        report = MarkdownReporter().report(result)

        assert "No winner (the experiment failed)." in report
        assert "_No ranked algorithms" in report
        assert "_No algorithm records" in report
        assert "## Failure" in report


class TestB2SpecialCharactersAndUnicode:
    """B2: special characters and unicode in names/parameters."""

    def test_markdown_table_survives_pipes_and_newlines(self) -> None:
        result = _make_result(
            [
                _run_kwargs(
                    algorithm="algo | with | pipes",
                    phase_name="phase\nwith newline",
                    params={"breaks": "a|b\nc"},
                )
            ]
        )

        report = MarkdownReporter().report(result)

        table_rows = [line for line in report.splitlines() if line.startswith("| ")]
        assert len(table_rows) == 2  # header + exactly one data row
        assert "algo \\| with \\| pipes" in table_rows[1]

    def test_markdown_survives_markdown_meta_characters(self) -> None:
        result = _make_result([_run_kwargs(algorithm="**bold** # heading *emph*")])

        report = MarkdownReporter().report(result)

        assert "**bold** # heading *emph*" in report

    def test_unicode_labels_round_trip_through_json(self, tmp_path: Path) -> None:
        label = "Эксперимент «Титаник» — 日本語 🚀"
        result = _make_result([_run_kwargs(algorithm="модель-1")], label=label)
        path = tmp_path / "report.json"

        JSONReporter(path).report(result)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["label"] == label
        assert payload["runs"][0]["algorithm"] == "модель-1"


class TestB3Overwrite:
    """B3: an existing report file is overwritten."""

    def test_existing_file_is_replaced(self, tmp_path: Path) -> None:
        destination = tmp_path / "report.json"
        destination.write_text("stale content", encoding="utf-8")

        JSONReporter(destination).report(_make_result(_COMPLETED_RUNS))

        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["label"] == "reporter run"


class TestB4LargeResult:
    """B4: thousands of records — the JSON closes correctly."""

    def test_thousands_of_records_serialize(self, tmp_path: Path) -> None:
        runs = [
            _run_kwargs(
                algorithm=f"algo-{i % 7}",
                phase_name=f"phase-{i % 3}",
                score=1.0 + i,
                score_raw=-(1.0 + i),
            )
            for i in range(3000)
        ]
        result = _make_result(runs)
        path = tmp_path / "report.json"

        JSONReporter(path).report(result)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["runs"]) == 3000


class TestB5NonPlainTypes:
    """B5: numpy scalars / Path / Enum / pd.Timestamp in the fields."""

    def test_numpy_scalars_normalize_to_json_types(self, tmp_path: Path) -> None:
        # np.float64 / np.bool_ are JSON-safe after the model's strict
        # float/bool normalization; np.int64 inside a free-form mapping
        # is rejected by the model itself, so the engine always dumps its
        # config before building the result.
        result = _make_result(
            [
                _run_kwargs(
                    params={"alpha": np.float64(0.1), "flag": True},
                    score=np.float32(0.5),
                    score_raw=np.float32(-0.5),
                    duration_seconds=np.float64(12.5),
                )
            ]
        )
        path = tmp_path / "report.json"

        JSONReporter(path).report(result)

        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload["runs"][0]["params"]
        assert params["alpha"] == 0.1
        assert params["flag"] is True
        assert payload["runs"][0]["score"] == pytest.approx(0.5, abs=1e-6)

    def test_path_enum_timestamp_in_config_snapshot(self, tmp_path: Path) -> None:
        """Path/Enum/Timestamp reach the JSON only through the config
        snapshot: the engine passes the config through
        ``model_dump(mode="json")`` (the same normalization the model
        applies), so a live ``Path`` / ``pd.Timestamp`` becomes its ISO /
        string form before the report is written."""
        config = {
            "artifacts_dir": str(Path("/tmp/runs/exp-1")),
            "validation": "k_fold",
            "last_touched": str(
                pd.Timestamp("2026-01-01T12:00:00+00:00").tz_localize(None).isoformat()
            ),
        }
        result = _make_result([], config=config)
        path = tmp_path / "report.json"

        JSONReporter(path).report(result)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["config"]["artifacts_dir"] == "/tmp/runs/exp-1"
        assert payload["config"]["validation"] == "k_fold"
        assert payload["config"]["last_touched"] == "2026-01-01T12:00:00"


class TestB6MissingDirectories:
    """B6: missing parent directories of the destination are created."""

    def test_nested_destination_is_created(self, tmp_path: Path) -> None:
        destination = tmp_path / "a" / "b" / "report.json"

        artifact = JSONReporter(destination).report(_make_result(_COMPLETED_RUNS))

        assert artifact == destination
        assert destination.exists()


class TestFilePermissions:
    """N-3: the report file gets the ordinary umask-based permissions.

    The atomic write goes through a private temp file, which
    ``NamedTemporaryFile`` creates as ``0o600``; the reporter chmods it
    to the ordinary file-creation mode (``0o666 & ~umask``) before the
    rename, so the artifact is not stuck with owner-only permissions.
    """

    def test_destination_permissions_follow_umask(self, tmp_path: Path) -> None:
        destination = tmp_path / "report.json"

        JSONReporter(destination).report(_make_result(_COMPLETED_RUNS))

        umask = os.umask(0)
        os.umask(umask)
        expected = 0o666 & ~umask
        actual = stat.S_IMODE(destination.stat().st_mode)
        assert actual == expected, (
            f"expected {oct(expected)} (umask {oct(umask)}), got {oct(actual)}"
        )

    def test_overwrite_keeps_destination_mode(self, tmp_path: Path) -> None:
        """An existing destination is replaced but its directory stays:
        the report policy is "last report wins" — the new file takes the
        umask-based mode of a fresh artifact (documented overwrite
        behaviour, no mode-preserving guarantee)."""
        destination = tmp_path / "report.json"
        destination.write_text("stale", encoding="utf-8")

        JSONReporter(destination).report(_make_result(_COMPLETED_RUNS))

        umask = os.umask(0)
        os.umask(umask)
        assert stat.S_IMODE(destination.stat().st_mode) == 0o666 & ~umask


class TestMarkdownExtra:
    """Extra Markdown policies: escaping helpers and config embedding."""

    def test_config_snapshot_embedded_and_collapsible(self) -> None:
        result = _make_result([], config={"general": {"random_state": 42}})

        report = MarkdownReporter(include_config=True).report(result)

        assert "<details>" in report
        assert '"random_state": 42' in report

    def test_config_snapshot_omitted_when_disabled(self) -> None:
        result = _make_result([], config={"general": {"random_state": 42}})

        report = MarkdownReporter(include_config=False).report(result)

        assert "<details>" not in report

    def test_code_span_with_backticks(self) -> None:
        from configurable_automl_engine.experiment.reporting.markdown_reporter import (
            _escape_code_span,
        )

        assert _escape_code_span("alpha=0.1") == "`alpha=0.1`"
        assert _escape_code_span("a`b") == "``a`b``"

    def test_disqualified_run_details(self) -> None:
        result = _make_result(
            [
                _run_kwargs(
                    algorithm="broken",
                    status=RunStepStatus.disqualified,
                    score=None,
                    score_raw=None,
                    error="circuit breaker: 5 consecutive failures",
                )
            ]
        )

        report = MarkdownReporter().report(result)

        assert "**disqualified**" in report
        assert "circuit breaker: 5 consecutive failures" in report


# ────────────── stored explain() payload section (issue #181) ────────────── #


class TestExplanationsSection:
    """The Markdown reporter renders the stored explain() payload."""

    def test_permutation_section_rendered(self) -> None:
        result = _make_result(
            _COMPLETED_RUNS,
            explanations={
                "method": "permutation",
                "importances": [
                    {"feature": "f1", "importance_mean": 0.25, "importance_std": 0.01},
                    {"feature": "f2", "importance_mean": -0.02},
                ],
                "baseline_score": 0.774,
                "metric_user": "rmse",
                "n_rows": 40,
                "n_features": 2,
                "data_semantics": "train",
            },
        )
        text = str(MarkdownReporter().report(result))
        assert "## Feature importance (permutation)" in text
        assert "| f1 " in text
        assert "| f2 " in text
        assert "Baseline:" in text

    def test_shap_baseline_is_mean_prediction(self) -> None:
        result = _make_result(
            _COMPLETED_RUNS,
            explanations={
                "method": "shap",
                "importances": [{"feature": "f1", "importance_mean": 0.1}],
                "baseline_prediction": 3.21,
            },
        )
        text = str(MarkdownReporter().report(result))
        assert "## Feature importance (shap)" in text
        assert "mean prediction" in text

    def test_no_explanations_no_section(self) -> None:
        text = str(MarkdownReporter().report(_make_result(_COMPLETED_RUNS)))
        assert "Feature importance" not in text
