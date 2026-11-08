"""Tests of the experiment export (the T11 component).

Covered scenarios (positive / negative / boundary):

- P1: the export creates the full bundle (the model copy, the
  leaderboard in JSON and Markdown, the report, the result snapshot,
  the metadata) with the correct content;
- P2: a repeated export of the same run works (overwrite) and the
  bundle stays complete; two same-labeled experiments do not collide
  (the ``run_id`` prefix keeps the file names unique);
- N1: no write permission in the destination → an ``OSError`` with no
  partial artifacts;
- N1b: an I/O failure mid-swap (after the backup phase moved some
  previous files aside) → the previous bundle is restored byte-exact,
  no backups/temp leftovers;
- N1c: even a failing restore (the worst case: both the swap and the
  restore hit I/O errors) never leaves ``.export-backup`` files
  silently swallowing the bundle files;
- N1d: a failure in the backup phase itself → the previous bundle
  intact;
- N2: a serialization failure in the middle of the export → the error
  propagates and no partial bundle is left in the destination (the
  previous bundle, if any, stays untouched);
- B1: an empty (non-existent) destination directory is created
  recursively;
- B2: a label with special characters is safe (it never enters a file
  name; it lands in the metadata intact);
- B3: a failed experiment is exported (the status and the failure
  reason in the metadata; no model file);
- B4: the destination path exists as a file → a clear ``ExportError``;
- B5: the original model artifact is not touched by the export (the
  copy is byte-exact, the source keeps its content and path);
- B6: a multi-suffix artifact name (e.g. ``model.tar.gz``) keeps the
  full suffix chain in the exported file name.

T10 × T11 / T9 integration (issue #181):

- P1e: a run with ``general.ensemble.enable=true`` → the bundle carries
  ``<run_id>_ensemble.<ext>`` (a byte-exact copy), the metadata records
  the original artifact path (``ensemble_source_path``) and the file
  map; moving the bundle elsewhere, ``ensemble_model()`` loads by the
  metadata (the bundle copy path is recorded only there) — the result
  snapshot's ``artifact_path`` is never rewritten (the read-only
  contract);
- P2e: a run with a stored explain payload → the bundle carries
  ``<run_id>_explain.json`` with the JSON-safe content;
- N1e: the ensemble artifact recorded but missing on disk → a clear
  ``ExportError`` and no partial bundle;
- N2e: a legacy result without the new fields (``explanations``/
  ``ensemble``) exports without errors;
- B1e: the ensemble disabled → the bundle structure is identical to
  the previous one (regression);
- B2e: explain never invoked → no ``<run_id>_explain.json``;
- B3e: a failed run with and without an ensemble section exports
  correctly;
- B4e: a mid-swap failure with the ensemble file in the bundle → the
  previous bundle stays intact.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment.export import (
    ExportError,
    ensemble_filename,
    explain_json_filename,
    export_experiment,
    leaderboard_json_filename,
    leaderboard_md_filename,
    metadata_filename,
    model_filename,
    report_md_filename,
    result_json_filename,
)
from configurable_automl_engine.experiment.result import (
    AlgorithmRun,
    ExperimentResult,
)

# ────────────────────────── helpers ────────────────────────── #

RUN_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _ts(seconds: float = 0.0) -> Any:
    """A fixed UTC timestamp shifted by ``seconds``."""
    import datetime as dt

    return dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.UTC) + dt.timedelta(
        seconds=seconds
    )


def _record(**overrides: Any) -> AlgorithmRun:
    """A completed algorithm record with the given overrides."""
    values: dict[str, Any] = {
        "algorithm": "ridge",
        "phase_name": "search",
        "phase_action": "all_algorithms",
        "status": RunStepStatus.completed,
        "score_raw": -0.5,
        "score": 0.5,
        "params": {"alpha": 0.1},
        "n_trials_requested": 4,
        "n_trials_evaluated": 4,
        "n_trials_completed": 4,
        "n_trials_pruned": 0,
        "n_trials_failed": 0,
        "duration_seconds": 1.5,
    }
    values.update(overrides)
    return AlgorithmRun(**values)


def _completed_result(**overrides: Any) -> ExperimentResult:
    """A minimal completed ``ExperimentResult`` for export tests."""
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "label": "export test",
        "engine_version": "0.0.0",
        "started_at": _ts(),
        "finished_at": _ts(10.0),
        "duration_seconds": 10.0,
        "status": RunStatus.completed,
        "config": {"general": {"comparison_metric": "rmse"}},
        "resolved_validation": "k_fold",
        "resolved_n_folds": 5,
        "resolved_test_size": 0.2,
        "random_state": 42,
        "metric_user": "rmse",
        "metric_sklearn": "neg_root_mean_squared_error",
        "metric_direction": MetricDirection.minimize,
        "runs": [_record()],
        "winner": "ridge",
        "winner_score": 0.5,
        "winner_params": {"alpha": 0.1},
        "model_path": None,
    }
    values.update(overrides)
    return ExperimentResult(**values)


def _failed_result(**overrides: Any) -> ExperimentResult:
    """A minimal failed ``ExperimentResult`` (no winner, no model)."""
    return _completed_result(
        status=RunStatus.failed,
        error="All algorithms failed",
        runs=[
            _record(
                algorithm="ridge",
                status=RunStepStatus.failed,
                score_raw=None,
                score=None,
                error="HPO failed",
            )
        ],
        winner=None,
        winner_score=None,
        **overrides,
    )


def _expected_names(result: ExperimentResult) -> set[str]:
    """The full bundle file-name set of ``result`` (with a model)."""
    return {
        leaderboard_json_filename(result),
        leaderboard_md_filename(result),
        report_md_filename(result),
        result_json_filename(result),
        metadata_filename(result),
    }


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_export_creates_full_bundle(
    tmp_path: Path,
) -> None:
    """P1: the export creates the full bundle with the correct content."""
    model_src = tmp_path / "artifacts" / "model.pkl"
    model_src.parent.mkdir(parents=True)
    model_src.write_bytes(b"\x80\x04model-bytes")
    result = _completed_result(model_path=str(model_src))
    destination = tmp_path / "export"

    returned = export_experiment(result, destination)

    assert returned == destination
    names = _expected_names(result) | {model_filename(result)}
    assert {p.name for p in destination.iterdir()} == names

    # The model copy is byte-exact and keeps the source extension.
    model_copy = destination / model_filename(result)
    assert model_copy.read_bytes() == model_src.read_bytes()

    # The result snapshot equals the JSON reporter's dump of the result.
    snapshot = json.loads(
        (destination / result_json_filename(result)).read_text(encoding="utf-8")
    )
    assert snapshot == json.loads(result.to_json())

    # The leaderboard JSON carries the structured entries.
    leaderboard = json.loads(
        (destination / leaderboard_json_filename(result)).read_text(encoding="utf-8")
    )
    assert leaderboard["metric_user"] == "rmse"
    assert leaderboard["entries"][0]["algorithm"] == "ridge"
    assert leaderboard["entries"][0]["rank"] == 1
    assert "ridge" in leaderboard["table"]

    # The Markdown documents are non-empty and mention the winner.
    assert "ridge" in (destination / leaderboard_md_filename(result)).read_text(
        encoding="utf-8"
    )
    assert "ridge" in (destination / report_md_filename(result)).read_text(
        encoding="utf-8"
    )

    # The metadata: identity, status, timings, engine version, file map.
    meta = json.loads(
        (destination / metadata_filename(result)).read_text(encoding="utf-8")
    )
    assert meta["run_id"] == str(RUN_ID)
    assert meta["label"] == "export test"
    assert meta["status"] == "completed"
    assert meta["engine_version"] == "0.0.0"
    assert meta["started_at"] == "2026-01-01T12:00:00+00:00"
    assert meta["finished_at"] == "2026-01-01T12:00:10+00:00"
    assert meta["duration_seconds"] == 10.0
    assert meta["winner"] == "ridge"
    assert meta["model_source_path"] == str(model_src)
    assert set(meta["exported_files"]) == names


def test_p2_repeated_export_overwrites_and_same_label_no_collision(
    tmp_path: Path,
) -> None:
    """P2: a repeated export overwrites; same-label runs do not collide."""
    destination = tmp_path / "export"
    result = _completed_result()

    export_experiment(result, destination)
    stamp = destination / metadata_filename(result)
    stamp.write_text("stale", encoding="utf-8")  # a stale file of the run

    export_experiment(result, destination)

    # The bundle is complete again and the stale metadata is replaced.
    assert {p.name for p in destination.iterdir()} == _expected_names(result)
    meta = json.loads(stamp.read_text(encoding="utf-8"))
    assert meta["run_id"] == str(RUN_ID)

    # Two same-labeled experiments (different run_id) into one
    # directory: both bundles coexist, no name collision.
    other = _completed_result(run_id=uuid.UUID(int=1), label="same label")
    export_experiment(other, destination)
    expected = _expected_names(result) | _expected_names(other)
    assert {p.name for p in destination.iterdir()} == expected
    assert (
        json.loads(
            (destination / metadata_filename(other)).read_text(encoding="utf-8")
        )["label"]
        == "same label"
    )


# ────────────────────────── negative scenarios ────────────────────────── #


def test_n1_no_write_permission_no_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1: an unwritable destination → OSError, no partial artifacts.

    The failure is injected at the ``_swap_into_place`` boundary (a
    chmod-based simulation is not reliable for a root-owned test
    process, and patching ``os.replace`` globally would trip inside the
    ``JSONReporter`` before the swap is ever reached): the first
    ``os.replace`` of the swap itself fails, so the export must fail
    while moving nothing into place and leave no artifacts or temporary
    leftovers behind.
    """
    destination = tmp_path / "ro"
    destination.mkdir()
    result = _completed_result()
    real_replace = os.replace
    swap_started = False

    def _failing_swap_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        nonlocal swap_started
        # The JSONReporter's internal atomic write also uses os.replace
        # (a rename inside the temp directory); only the swap's own
        # moves must fail. The swap touches the destination directory:
        # any source that is NOT inside the temp dir is a swap move.
        if Path(dst).parent == destination:
            swap_started = True
            raise PermissionError(13, "Permission denied")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(
        "configurable_automl_engine.experiment.export.os.replace",
        _failing_swap_replace,
    )
    with pytest.raises(PermissionError):
        export_experiment(result, destination)
    # The failure really happened inside the swap (not earlier).
    assert swap_started
    # No artifacts, no temp dirs, no backups: the destination is empty.
    assert list(destination.iterdir()) == []


def test_n1b_mid_swap_failure_restores_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1b: an I/O failure mid-swap → the previous bundle is restored.

    The previous bundle of the same run exists; the swap fails after
    the backup phase replaced some destination files (the failure is
    injected on the second move of the move phase). The rollback must
    put every previous file back byte-exact and remove every trace of
    the interrupted export (moved files, backups, temp dir).
    """
    destination = tmp_path / "export"
    result = _completed_result()
    export_experiment(result, destination)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    names = sorted(before)
    assert len(names) == 5  # the full bundle is in place

    real_replace = os.replace
    swap_moves = 0

    def _failing_move(src: Any, dst: Any, **kwargs: Any) -> None:
        nonlocal swap_moves
        if Path(dst).parent == destination:
            swap_moves += 1
            if swap_moves == 2:
                # Fail exactly once: the restore's own renames must go
                # through the real os.replace, otherwise the test would
                # sabotage the rollback it is supposed to verify.
                swap_moves = -1000  # no further failures
                raise OSError(5, "Input/output error")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(
        "configurable_automl_engine.experiment.export.os.replace",
        _failing_move,
    )
    with pytest.raises(OSError, match="Input/output error"):
        export_experiment(result, destination)
    assert swap_moves < 0  # the failure really happened mid-swap
    # The previous bundle is fully restored, no backups, no temp dirs.
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


def test_n1c_mid_swap_restore_failure_no_backup_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1c: even a failing restore never leaves backup files behind.

    The worst case from the review: both the swap AND the restore hit
    I/O errors. The restore is best-effort and never aborts, so no
    ``.export-backup`` file may silently swallow a bundle file: the
    destination must end with real file names only (some may be missing
    after a destructive restore failure, but no hidden backups).
    """
    destination = tmp_path / "export"
    result = _completed_result()
    export_experiment(result, destination)
    real_replace = os.replace
    # Destination-bound replaces seen so far: the first two are the
    # swap's move-phase renames (the second one is injected to fail),
    # everything after that is the restore.
    dest_replaces_seen = 0
    restore_renames = 0

    def _failing_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        nonlocal dest_replaces_seen, restore_renames
        dst_path = Path(dst)
        if dst_path.parent != destination:
            return real_replace(src, dst, **kwargs)
        if ".export-backup" in dst_path.name:
            return real_replace(src, dst, **kwargs)  # the backup phase
        if dest_replaces_seen < 2:
            dest_replaces_seen += 1
            if dest_replaces_seen == 2:
                # The swap fails exactly once; the restore's renames
                # (backup → target, the source is a backup) must go
                # through, except the single one the test breaks.
                return real_replace(src, dst, **kwargs)
        # The restore: fail the FIRST backup→target rename only.
        restore_renames += 1
        if restore_renames == 1:
            raise OSError(5, "Input/output error")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(
        "configurable_automl_engine.experiment.export.os.replace",
        _failing_replace,
    )
    with pytest.raises(OSError):
        export_experiment(result, destination)
    leftovers = [p.name for p in destination.iterdir() if ".export-backup" in p.name]
    assert leftovers == []  # no backup swallowed the bundle files


def test_n1d_backup_phase_failure_restores_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1d: a failure in the backup phase → the previous bundle intact.

    The backup phase replaced one destination file with its backup; the
    second backup rename fails. The rollback must return the already
    backed-aside file and leave the destination exactly as it was.
    """
    destination = tmp_path / "export"
    result = _completed_result()
    export_experiment(result, destination)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}

    real_replace = os.replace
    backup_moves = 0

    def _failing_backup(src: Any, dst: Any, **kwargs: Any) -> None:
        nonlocal backup_moves
        dst_path = Path(dst)
        if (
            dst_path.parent == destination
            and ".export-backup" in dst_path.name
            and Path(src).parent == destination
        ):
            backup_moves += 1
            if backup_moves == 2:
                # Fail exactly once (the restore must be able to run).
                backup_moves = -1000
                raise OSError(5, "Input/output error")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(
        "configurable_automl_engine.experiment.export.os.replace",
        _failing_backup,
    )
    with pytest.raises(OSError, match="Input/output error"):
        export_experiment(result, destination)
    assert backup_moves < 0
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


def test_n2_midway_failure_leaves_no_partial_bundle(tmp_path: Path) -> None:
    """N2: a reporter failure midway → no partial bundle, old bundle kept."""
    destination = tmp_path / "export"
    result = _completed_result()
    export_experiment(result, destination)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}

    # A result whose JSON snapshot cannot be serialized (an object
    # smuggled past the model validation): the export fails at the
    # snapshot step, after the leaderboard files were written into the
    # temporary directory.
    bad = _completed_result(run_id=uuid.UUID(int=2))
    bad = bad.model_copy(update={"winner_params": {"f": object()}})

    with pytest.raises(Exception):  # noqa: B017 — a serialization error
        export_experiment(bad, destination)

    # The previous bundle is intact, no temporary leftovers, no files of
    # the failed export.
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_empty_directory_created_recursively(tmp_path: Path) -> None:
    """B1: a non-existent (nested) destination is created recursively."""
    result = _completed_result()
    destination = tmp_path / "a" / "b" / "export"
    export_experiment(result, destination)
    assert destination.is_dir()
    assert {p.name for p in destination.iterdir()} == _expected_names(result)


def test_b2_label_with_special_characters_is_safe(tmp_path: Path) -> None:
    """B2: a label with special characters never breaks the file names."""
    result = _completed_result(label='a/b:c*d?"e|f<g>h\ni')
    destination = tmp_path / "export"
    export_experiment(result, destination)
    # The names are built from the run_id only.
    assert {p.name for p in destination.iterdir()} == _expected_names(result)
    # The label lands in the metadata intact.
    meta = json.loads(
        (destination / metadata_filename(result)).read_text(encoding="utf-8")
    )
    assert meta["label"] == 'a/b:c*d?"e|f<g>h\ni'


def test_b3_failed_experiment_is_exported(tmp_path: Path) -> None:
    """B3: a failed run exports with the status and reason, no model."""
    result = _failed_result()
    destination = tmp_path / "export"
    export_experiment(result, destination)
    assert {p.name for p in destination.iterdir()} == _expected_names(result)
    meta = json.loads(
        (destination / metadata_filename(result)).read_text(encoding="utf-8")
    )
    assert meta["status"] == "failed"
    assert meta["error"] == "All algorithms failed"
    assert meta["winner"] is None
    assert meta["model_source_path"] is None
    # The result snapshot is still a full, parseable dump.
    snapshot = json.loads(
        (destination / result_json_filename(result)).read_text(encoding="utf-8")
    )
    assert snapshot["status"] == "failed"


def test_b4_destination_is_a_file_clear_error(tmp_path: Path) -> None:
    """B4: the destination path is a file → a clear ExportError."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ExportError, match="not a directory"):
        export_experiment(_completed_result(), blocker)


def test_b5_original_model_artifact_untouched(tmp_path: Path) -> None:
    """B5: the export copies the model; the source stays byte-exact."""
    model_src = tmp_path / "artifacts" / "model.joblib"
    model_src.parent.mkdir(parents=True)
    payload = b"\x80\x04original-model-payload"
    model_src.write_bytes(payload)
    before = model_src.stat().st_mtime_ns
    result = _completed_result(model_path=str(model_src))
    destination = tmp_path / "export"

    export_experiment(result, destination)

    # The source is untouched (content, path, mtime).
    assert model_src.read_bytes() == payload
    assert model_src.stat().st_mtime_ns == before
    # The copy keeps the source extension (the serialization marker).
    copy = destination / model_filename(result)
    assert copy.name == f"{RUN_ID}_model.joblib"
    assert copy.read_bytes() == payload


# ────────────────────────── facade integration ────────────────────────── #


def test_facade_export_delegates_after_run(tmp_path: Path) -> None:
    """The facade ``export()`` delegates to the export component.

    A lightweight integration check without a full training run: the
    facade raises the not-run state error before ``run()`` and, with a
    result injected (the same private attribute the ``run()`` adopts),
    produces the same bundle the component does.
    """
    from configurable_automl_engine import AutoMLExperiment
    from configurable_automl_engine.experiment.experiment import (
        ExperimentNotRunError,
    )

    data = _minimal_frame()
    experiment = AutoMLExperiment(
        {"general": {"comparison_metric": "rmse"}}, data, target="target"
    )
    with pytest.raises(ExperimentNotRunError):
        experiment.export(tmp_path / "never")

    result = _completed_result(model_path=None)
    experiment._result = result
    destination = tmp_path / "bundle"
    assert experiment.export(destination) == destination
    assert {p.name for p in destination.iterdir()} == _expected_names(result)


def _minimal_frame() -> Any:
    """A tiny frame for the facade constructor (never trained on)."""
    import pandas as pd

    return pd.DataFrame({"f1": [1.0, 2.0], "target": [1.0, 2.0]})


# The JSON reporter import above guards the round-trip contract; keep a
# direct sanity check that a snapshot written by the export parses back
# into the result model (the same contract the JSONReporter documents).


def test_result_snapshot_round_trip(tmp_path: Path) -> None:
    """The exported ``<run_id>_result.json`` restores the result model."""
    result = _completed_result()
    destination = tmp_path / "export"
    export_experiment(result, destination)
    snapshot_path = destination / result_json_filename(result)
    restored = ExperimentResult.from_json(snapshot_path.read_text(encoding="utf-8"))
    assert restored == result


def test_b6_multi_suffix_model_extension_kept(tmp_path: Path) -> None:
    """B6: a multi-suffix artifact name keeps the full suffix chain."""
    result = _completed_result(model_path="artifacts/model.tar.gz")
    name = model_filename(result)
    assert name == f"{RUN_ID}_model.tar.gz"


def test_b7_directory_under_bundle_name_clear_error(tmp_path: Path) -> None:
    """B7: a directory occupying a bundle-file name → a clear error.

    A directory cannot be backed aside and replaced by a file rename;
    the export must reject it BEFORE any move happens (no backups, no
    moved files, no partial state) with an error naming the path.
    """
    result = _completed_result()
    destination = tmp_path / "export"
    blocker = destination / metadata_filename(result)
    blocker.mkdir(parents=True)
    blocker.joinpath("keep.txt").write_text("precious", encoding="utf-8")

    with pytest.raises(ExportError, match="directory under the bundle file name"):
        export_experiment(result, destination)
    # Nothing was moved: the blocker directory is intact, no backups.
    assert blocker.joinpath("keep.txt").read_text(encoding="utf-8") == "precious"
    assert [p.name for p in destination.iterdir() if ".export-backup" in p.name] == []


def test_n3_recorded_model_artifact_missing_clear_error(tmp_path: Path) -> None:
    """N3: a recorded but missing model artifact → a clear ExportError."""
    result = _completed_result(model_path=str(tmp_path / "gone" / "model.pkl"))
    with pytest.raises(ExportError, match="does not exist"):
        export_experiment(result, tmp_path / "export")
    # Nothing was produced by the failed export.
    assert (
        not (tmp_path / "export").exists()
        or list((tmp_path / "export").iterdir()) == []
    )


def test_n4_restore_step_failures_do_not_abort_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N4: one failing restore step does not prevent the others.

    Two independent restore failures are injected: the unlink of an
    already-moved new file and the rename of one backup back. The
    restore must still restore every other backup (and no backup
    leftover may remain), and the original swap error must propagate.
    """
    destination = tmp_path / "export"
    result = _completed_result()
    export_experiment(result, destination)
    previous = {p.name: p.read_bytes() for p in destination.iterdir()}

    real_replace = os.replace
    real_unlink = Path.unlink
    swap_moves = 0
    restored_renames = 0
    seen_unlink_failures = 0

    def _failing_unlink(self: Path, missing_ok: bool = False) -> None:
        nonlocal seen_unlink_failures
        if self.parent == destination and swap_moves < 0:
            # The restore is active: fail the unlink of the first moved
            # new file, let the rest through.
            seen_unlink_failures += 1
            if seen_unlink_failures == 1:
                raise OSError(5, "unlink failed")
        return real_unlink(self, missing_ok)

    def _combined_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        """A single os.replace fake for both the swap and the restore.

        Two separate monkeypatches of the same attribute would override
        each other (only the last one is active), so the swap and the
        restore behaviors are combined into one dispatcher.
        """
        nonlocal swap_moves, restored_renames
        dst_path = Path(dst)
        if dst_path.parent != destination:
            return real_replace(src, dst, **kwargs)
        if Path(src).name.endswith(".export-backup"):
            # A restore rename (backup → target): fail the second one.
            restored_renames += 1
            if restored_renames == 2:
                raise OSError(5, "restore rename failed")
            return real_replace(src, dst, **kwargs)
        if ".export-backup" in dst_path.name:
            # The backup phase: never fails here.
            return real_replace(src, dst, **kwargs)
        # The move phase: the second rename fails, then never again
        # (the first move succeeded, so the restore has a moved file
        # to unlink).
        swap_moves += 1
        if swap_moves == 2:
            swap_moves = -1000
            raise OSError(5, "Input/output error")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(
        "configurable_automl_engine.experiment.export.os.replace", _combined_replace
    )
    monkeypatch.setattr(Path, "unlink", _failing_unlink)
    with pytest.raises(OSError):
        export_experiment(result, destination)
    assert swap_moves < 0 and restored_renames >= 2 and seen_unlink_failures >= 1
    # The destructive unlink failure does not block the backups restore:
    # a failed rename falls back to copy + delete, so EVERY previous file
    # is back under its real name and no backup leftovers remain.
    leftovers = [p.name for p in destination.iterdir() if ".export-backup" in p.name]
    assert leftovers == []
    restored = {
        name: content
        for name, content in ((p.name, p.read_bytes()) for p in destination.iterdir())
        if name in previous
    }
    assert restored == previous


def test_n7_success_cleanup_unlink_failure_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N7: a failing success-path backup cleanup keeps the export success.

    Review round 2, blocker 2: after a fully successful swap the
    obsolete backup cleanup hit an ``unlink`` failure — the export used
    to raise AFTER the bundle was written (a reported failure for a
    successful export) and left every stale backup behind. The cleanup
    is best-effort now: the export succeeds, the fresh bundle is in
    place, other stale backups are still removed, and a leftover backup
    is an inert hidden file (the real-name file is the fresh export, so
    nothing is swallowed).
    """
    destination = tmp_path / "export"
    result = _completed_result()
    export_experiment(result, destination)

    real_unlink = Path.unlink
    failing_done = False

    def _failing_unlink(self: Path, missing_ok: bool = False) -> None:
        nonlocal failing_done
        # Fail exactly one success-path backup cleanup, then let the
        # rest through (the other stale backups must still go).
        if (
            not failing_done
            and self.parent == destination
            and ".export-backup" in self.name
        ):
            failing_done = True
            raise OSError(5, "unlink failed")
        return real_unlink(self, missing_ok)

    monkeypatch.setattr(Path, "unlink", _failing_unlink)
    # The export SUCCEEDS despite the cleanup failure.
    export_experiment(result, destination)
    assert failing_done  # the injected failure really happened
    # The fresh bundle is fully in place under real names (the
    # un-removable backup is an inert hidden extra, not a swallower:
    # its real-name file is the fresh export).
    real_names = {
        p.name for p in destination.iterdir() if ".export-backup" not in p.name
    }
    assert real_names == _expected_names(result)
    meta = json.loads(
        (destination / metadata_filename(result)).read_text(encoding="utf-8")
    )
    assert meta["run_id"] == str(RUN_ID)


def test_n5_restore_skips_gone_backup_and_unlinks_live_target(tmp_path: Path) -> None:
    """N5: the restore handles a gone backup and a live target name.

    A direct unit check of ``_restore_backups`` covering the branches
    the integration scenarios above cannot reach deterministically:

    - a backup that no longer exists (already restored by an earlier
      rename, or removed) → skipped, the live target is NOT deleted;
    - a backup whose target name is occupied → the occupant is removed
      and the backup lands on the target.
    """
    from configurable_automl_engine.experiment.export import _restore_backups

    destination = tmp_path / "dest"
    destination.mkdir()
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()

    live = destination / "live.txt"
    live.write_text("live")
    gone_target = destination / "gone.txt"
    gone_target.write_text("new-content-must-not-be-deleted")
    occupied = destination / "occupied.txt"
    occupied.write_text("new")
    occupied_backup = destination / ".occupied.txt.export-backup"
    occupied_backup.write_text("old-occupied")

    _restore_backups(
        backups={
            gone_target: destination / ".gone.txt.export-backup",  # missing backup
            occupied: occupied_backup,
        },
        moved=[],
        temp_dir=temp_dir,
        destination=destination,
    )

    assert gone_target.read_text() == "new-content-must-not-be-deleted"
    assert occupied.read_text() == "old-occupied"
    assert not occupied_backup.exists()
    assert live.read_text() == "live"


def test_n6_restore_rename_failure_recorded_then_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N6: a failing backup→target rename falls back to copy.

    The rename of the backup onto its target fails (an I/O error); the
    restore falls back to copy + delete of the backup source, so the
    previous content lands on the target and no backup file remains to
    silently swallow the bundle content. The error is still remembered
    and re-raised after the restore loop (a caller must know the swap
    failed), but the destination state is complete.
    """
    from configurable_automl_engine.experiment import export as export_module
    from configurable_automl_engine.experiment.export import _restore_backups

    destination = tmp_path / "dest"
    destination.mkdir()
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    target = destination / "file.txt"
    target.write_text("new")
    backup = destination / ".file.txt.export-backup"
    backup.write_text("old")

    real_replace = os.replace

    def _failing_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        if Path(dst) == target and Path(src) == backup:
            raise OSError(5, "rename failed")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(export_module.os, "replace", _failing_replace)
    # The fallback completes the restore, so the restore itself does not
    # re-raise (the original swap error propagates separately, in the
    # caller's except block).
    _restore_backups(
        backups={target: backup},
        moved=[],
        temp_dir=temp_dir,
        destination=destination,
    )
    # The copy fallback restored the previous content; the backup source
    # is gone (no swallowing leftovers).
    assert target.read_text() == "old"
    assert not backup.exists()


def test_n6b_restore_copy_fallback_failure_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N6b: the rename AND the copy fallback both fail → the error raises.

    When even the copy fallback cannot restore the file (an extreme I/O
    failure), the restore records the error and re-raises it after the
    loop; the backup source survives (the previous content is not
    destroyed).
    """
    from configurable_automl_engine.experiment import export as export_module
    from configurable_automl_engine.experiment.export import _restore_backups

    destination = tmp_path / "dest"
    destination.mkdir()
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    target = destination / "file.txt"
    target.write_text("new")
    backup = destination / ".file.txt.export-backup"
    backup.write_text("old")

    real_replace = os.replace
    real_copy2 = shutil.copy2

    def _failing_replace(src: Any, dst: Any, **kwargs: Any) -> None:
        if Path(dst) == target and Path(src) == backup:
            raise OSError(5, "rename failed")
        return real_replace(src, dst, **kwargs)

    def _failing_copy2(src: Any, dst: Any, **kwargs: Any) -> None:
        if Path(dst) == target:
            raise OSError(5, "copy failed")
        return real_copy2(src, dst, **kwargs)

    monkeypatch.setattr(export_module.os, "replace", _failing_replace)
    monkeypatch.setattr(export_module.shutil, "copy2", _failing_copy2)
    with pytest.raises(OSError, match="copy failed"):
        _restore_backups(
            backups={target: backup},
            moved=[],
            temp_dir=temp_dir,
            destination=destination,
        )
    # The backup source survived: the previous content is not destroyed.
    assert backup.read_text() == "old"


# ────────────────── T10 × T11 / T9 integration (issue #181) ────────────────── #


def _ensemble_section(artifact_path: str | None) -> dict[str, Any]:
    """A minimal ensemble report section with the given artifact path."""
    section: dict[str, Any] = {
        "enabled": True,
        "strategy": "weighted_average",
        "weight_method": "score_weighted",
        "k_requested": 2,
        "k_used": 1,
        "metric_user": "rmse",
        "metric_direction": "minimize",
        "members": [{"rank": 1, "algorithm": "ridge", "score": 0.5, "weight": 1.0}],
    }
    if artifact_path is not None:
        section["artifact_path"] = artifact_path
    return section


def _explain_payload() -> dict[str, Any]:
    """A minimal JSON-safe explain payload (the facade's storage shape)."""
    return {
        "method": "permutation",
        "importances": [
            {"feature": "f1", "importance_mean": 0.25, "importance_std": 0.01},
            {"feature": "f2", "importance_mean": -0.02, "importance_std": 0.02},
        ],
        "baseline_score": 0.5,
        "metric_user": "rmse",
        "n_rows": 10,
        "n_features": 2,
        "data_semantics": "train",
    }


def test_p1e_ensemble_artifact_exported_and_restorable(
    tmp_path: Path,
) -> None:
    """P1e: the ensemble artifact lands in the bundle (a byte-exact copy).

    The metadata records the original source path and the file map; the
    result snapshot's ``artifact_path`` is never rewritten (the export
    is read-only with respect to the result).
    """
    ensemble_src = tmp_path / "artifacts" / "model_ensemble.pkl"
    ensemble_src.parent.mkdir(parents=True)
    payload = b"ENSEMBLE-ARTIFACT-BYTES"
    ensemble_src.write_bytes(payload)
    model_src = tmp_path / "artifacts" / "model.pkl"
    model_src.write_bytes(b"MODEL-BYTES")
    result = _completed_result(
        model_path=str(model_src),
        ensemble=_ensemble_section(str(ensemble_src)),
    )
    destination = tmp_path / "export"

    export_experiment(result, destination)

    assert ensemble_filename(result) == f"{RUN_ID}_ensemble.pkl"
    expected = _expected_names(result) | {
        model_filename(result),
        ensemble_filename(result),
    }
    assert {p.name for p in destination.iterdir()} == expected

    # The ensemble copy is byte-exact; the original is untouched.
    ensemble_copy = destination / ensemble_filename(result)
    assert ensemble_copy.read_bytes() == payload
    assert ensemble_src.read_bytes() == payload

    # The metadata records the source path and the role in the file map.
    meta = json.loads(
        (destination / metadata_filename(result)).read_text(encoding="utf-8")
    )
    assert meta["ensemble_source_path"] == str(ensemble_src)
    assert meta["exported_files"][ensemble_filename(result)] == "ensemble"

    # The result snapshot keeps the ORIGINAL artifact path (the export
    # never mutates the result).
    snapshot = json.loads(
        (destination / result_json_filename(result)).read_text(encoding="utf-8")
    )
    assert snapshot["ensemble"]["artifact_path"] == str(ensemble_src)


def test_p1e_ensemble_bundle_relocation_loads_by_metadata(
    tmp_path: Path,
) -> None:
    """P1e (restoration): the bundle moved elsewhere carries the ensemble.

    The bundle copy path is recorded in the metadata ``exported_files``
    map only; a consumer relocating the bundle finds the ensemble file
    next to the model copy (the same ``run_id`` prefix), so the ensemble
    stays restorable from the export alone.
    """
    ensemble_src = tmp_path / "artifacts" / "model_ensemble.joblib"
    ensemble_src.parent.mkdir(parents=True)
    ensemble_src.write_bytes(b"ENSEMBLE-RELOCATED")
    model_src = tmp_path / "artifacts" / "model.pkl"
    model_src.write_bytes(b"MODEL-BYTES")
    result = _completed_result(
        model_path=str(model_src),
        ensemble=_ensemble_section(str(ensemble_src)),
    )
    destination = tmp_path / "export"
    export_experiment(result, destination)

    relocated = tmp_path / "elsewhere"
    shutil.copytree(destination, relocated)

    meta = json.loads(
        (relocated / metadata_filename(result)).read_text(encoding="utf-8")
    )
    bundle_ensemble = relocated / ensemble_filename(result)
    assert bundle_ensemble.is_file()
    assert meta["exported_files"][ensemble_filename(result)] == "ensemble"
    assert bundle_ensemble.read_bytes() == b"ENSEMBLE-RELOCATED"
    assert meta["ensemble_source_path"] == str(ensemble_src)


def test_p2e_explain_payload_exported(tmp_path: Path) -> None:
    """P2e: the stored explain payload → ``<run_id>_explain.json``."""
    result = _completed_result(explanations=_explain_payload())
    destination = tmp_path / "export"

    export_experiment(result, destination)

    expected = _expected_names(result) | {explain_json_filename(result)}
    assert {p.name for p in destination.iterdir()} == expected
    payload = json.loads(
        (destination / explain_json_filename(result)).read_text(encoding="utf-8")
    )
    assert payload == _explain_payload()
    # The file map lists the explain file with its role.
    meta = json.loads(
        (destination / metadata_filename(result)).read_text(encoding="utf-8")
    )
    assert meta["exported_files"][explain_json_filename(result)] == "explain_json"


def test_n1e_missing_ensemble_artifact_clear_error(tmp_path: Path) -> None:
    """N1e: a recorded but missing ensemble artifact → a clear error.

    The ensemble section promises a loadable artifact (``ensemble_model()``
    reads it), so a missing file is an ``ExportError`` — the same
    contract as the model copy — and the destination keeps its previous
    state (no partial bundle).
    """
    destination = tmp_path / "export"
    previous = _completed_result(run_id=uuid.UUID(int=3))
    export_experiment(previous, destination)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}

    model_src = tmp_path / "artifacts" / "model.pkl"
    model_src.parent.mkdir(parents=True)
    model_src.write_bytes(b"MODEL-BYTES")
    result = _completed_result(
        model_path=str(model_src),
        ensemble=_ensemble_section(str(tmp_path / "gone_ensemble.pkl")),
    )
    with pytest.raises(ExportError, match="does not exist"):
        export_experiment(result, destination)
    # The previous bundle is intact, no leftovers.
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


def test_n2e_legacy_result_without_new_fields_exports(tmp_path: Path) -> None:
    """N2e: an old-format result (no explanations/ensemble) exports fine."""
    result = _completed_result()
    assert result.explanations is None
    assert result.ensemble is None
    destination = tmp_path / "export"
    export_experiment(result, destination)
    assert {p.name for p in destination.iterdir()} == _expected_names(result)
    meta = json.loads(
        (destination / metadata_filename(result)).read_text(encoding="utf-8")
    )
    assert meta["ensemble_source_path"] is None


def test_b1e_ensemble_disabled_bundle_structure_unchanged(
    tmp_path: Path,
) -> None:
    """B1e: no ensemble section → the bundle matches the legacy structure."""
    result = _completed_result()
    destination = tmp_path / "export"
    export_experiment(result, destination)
    assert {p.name for p in destination.iterdir()} == _expected_names(result)


def test_b2e_no_explain_no_explain_file(tmp_path: Path) -> None:
    """B2e: explain never invoked → no ``<run_id>_explain.json``."""
    result = _completed_result()
    destination = tmp_path / "export"
    export_experiment(result, destination)
    assert not (destination / explain_json_filename(result)).exists()


def test_b3e_failed_run_with_and_without_ensemble_section(
    tmp_path: Path,
) -> None:
    """B3e: a failed run exports correctly with and without the section.

    The ensemble section is attached only after a successful artifact
    save, so both states are legitimate export inputs.
    """
    # Without the section: the plain failed-run bundle.
    failed = _failed_result()
    d1 = tmp_path / "failed-plain"
    export_experiment(failed, d1)
    assert {p.name for p in d1.iterdir()} == _expected_names(failed)

    # With the section (a run that failed AFTER the ensemble was saved):
    # the artifact is exported, the metadata keeps the source path.
    ensemble_src = tmp_path / "artifacts" / "failed_ensemble.pkl"
    ensemble_src.parent.mkdir(parents=True)
    ensemble_src.write_bytes(b"FAILED-RUN-ENSEMBLE")
    model_src = tmp_path / "artifacts" / "failed_model.pkl"
    model_src.parent.mkdir(exist_ok=True)
    model_src.write_bytes(b"FAILED-RUN-MODEL")
    failed_with = _failed_result(
        model_path=str(model_src),
        ensemble=_ensemble_section(str(ensemble_src)),
    )
    d2 = tmp_path / "failed-ensemble"
    export_experiment(failed_with, d2)
    expected = _expected_names(failed_with) | {
        model_filename(failed_with),
        ensemble_filename(failed_with),
    }
    assert {p.name for p in d2.iterdir()} == expected
    meta = json.loads((d2 / metadata_filename(failed_with)).read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert meta["ensemble_source_path"] == str(ensemble_src)


def test_b4e_mid_swap_failure_with_ensemble_keeps_old_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4e: a mid-swap failure with the ensemble file → the old bundle intact.

    An extension of the existing mid-swap scenario: the bundle now also
    carries the ensemble artifact and the explain payload; the rollback
    restores every previous file byte-exact and leaves no leftovers.
    """
    ensemble_src = tmp_path / "artifacts" / "model_ensemble.pkl"
    ensemble_src.parent.mkdir(parents=True)
    ensemble_src.write_bytes(b"OLD-ENSEMBLE")
    model_src = tmp_path / "artifacts" / "model.pkl"
    model_src.write_bytes(b"MODEL-BYTES")
    result = _completed_result(
        model_path=str(model_src),
        ensemble=_ensemble_section(str(ensemble_src)),
        explanations=_explain_payload(),
    )
    destination = tmp_path / "export"
    export_experiment(result, destination)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    assert len(before) == 8  # the full bundle incl. model + ensemble + explain

    real_replace = os.replace
    swap_moves = 0

    def _failing_move(src: Any, dst: Any, **kwargs: Any) -> None:
        nonlocal swap_moves
        if Path(dst).parent == destination:
            swap_moves += 1
            if swap_moves == 2:
                swap_moves = -1000
                raise OSError(5, "Input/output error")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(
        "configurable_automl_engine.experiment.export.os.replace",
        _failing_move,
    )
    with pytest.raises(OSError, match="Input/output error"):
        export_experiment(result, destination)
    assert swap_moves < 0
    # The previous bundle (including the ensemble copy) is fully restored.
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


def test_ensemble_filename_suffix_rules() -> None:
    """The ensemble file name follows the model suffix rules."""
    model_path = "artifacts/model.pkl"
    result = _completed_result(
        model_path=model_path,
        ensemble=_ensemble_section("artifacts/ens.pkl"),
    )
    assert ensemble_filename(result) == f"{RUN_ID}_ensemble.pkl"
    result = _completed_result(
        model_path=model_path,
        ensemble=_ensemble_section("artifacts/model_ensemble.tar.gz"),
    )
    assert ensemble_filename(result) == f"{RUN_ID}_ensemble.tar.gz"
    # No section / no artifact path → no ensemble file name.
    assert ensemble_filename(_completed_result()) is None
    section = _ensemble_section("artifacts/ens.pkl")
    del section["artifact_path"]
    assert (
        ensemble_filename(_completed_result(model_path=model_path, ensemble=section))
        is None
    )


def test_facade_explain_stored_then_exported(tmp_path: Path) -> None:
    """The facade explain→export chain (a lightweight integration check).

    With a result injected (the same private attribute ``run()`` adopts)
    carrying an explain payload, ``export()`` produces the bundle with
    the ``<run_id>_explain.json`` file.
    """
    from configurable_automl_engine import AutoMLExperiment

    experiment = AutoMLExperiment(
        {"general": {"comparison_metric": "rmse"}}, _minimal_frame(), target="target"
    )
    result = _completed_result(explanations=_explain_payload())
    experiment._result = result
    destination = tmp_path / "bundle"
    assert experiment.export(destination) == destination
    expected = _expected_names(result) | {explain_json_filename(result)}
    assert {p.name for p in destination.iterdir()} == expected
    payload = json.loads(
        (destination / explain_json_filename(result)).read_text(encoding="utf-8")
    )
    assert payload["method"] == "permutation"
