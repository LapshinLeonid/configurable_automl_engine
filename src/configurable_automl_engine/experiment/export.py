"""Export of the experiment artifact bundle into a directory (the T11
component).

:func:`export_experiment` (the engine behind
:meth:`AutoMLExperiment.export`) copies the consumed artifacts of one
finished run into a destination directory under a fixed, documented
layout. Everything the export produces comes from the
:class:`~configurable_automl_engine.experiment.result.ExperimentResult`
model plus the reporters of the T7 component — the exporter never
reaches into the training engine, so it works in any consumer process.

The bundle (the file names are the documented contract; every file is
prefixed with the ``run_id`` to avoid collisions between the exports of
different runs — including two same-labeled experiments exported into
one directory, and the repeated exports of one run):

* ``<run_id>_model.<ext>`` — the model artifact, **copied** from the
  path recorded in the result (``model_path``): the original file is
  never touched, moved or renamed (it may be referenced by the recorded
  path), the copy is byte-exact and keeps the original extension (the
  serialization format marker);
* ``<run_id>_leaderboard.json`` — the algorithm ranking table (the T6
  component): the rendered table plus the structured entries;
* ``<run_id>_leaderboard.md`` — the ranking table as plain text;
* ``<run_id>_report.md`` — the full Markdown report
  (:class:`~configurable_automl_engine.experiment.reporting.MarkdownReporter`,
  the T7 component);
* ``<run_id>_result.json`` — the full JSON snapshot of the result
  (:class:`~configurable_automl_engine.experiment.reporting.JSONReporter`,
  the T7 component);
* ``<run_id>_metadata.json`` — the run metadata: the run id, the label,
  the status (and the failure reason for a failed run), the timings,
  the engine version, the winner, the source path of the model artifact
  and the mapping of the exported files.

Documented contract:

* the destination directory does not exist → it is created recursively
  (with the parents);
* the destination path exists but is a file (or otherwise not a
  directory) → :class:`ExportError` with a clear message;
* an existing bundle of the same run is overwritten (the last export
  wins; unrelated files in the destination are not touched);
* atomicity: the bundle is assembled in a temporary directory inside
  the destination and moved into place at the end by a two-phase swap
  (the to-be-overwritten previous files are backed aside first, then
  the new files are renamed over the free names). A failure in the
  middle (e.g. a serialization error of a hand-built result, a missing
  model artifact) leaves the destination without a partial bundle —
  the previous bundle stays untouched (or no bundle appears at all on
  a fresh export) and the temporary leftovers are always removed. The
  restore after a mid-swap failure is best-effort and never aborts:
  one failed restoration does not prevent the others, so an I/O
  failure during the swap degrades the destination state but never
  leaves ``.export-backup`` files silently swallowing the bundle;
* no write permission in the destination → the ``OSError`` of the
  filesystem propagates before any artifact is produced (the temporary
  directory cannot even be created);
* the model artifact path may be absent (a failed run) → the model file
  is simply not exported; a path recorded in the result but missing on
  disk is an :class:`ExportError` naming the missing file;
* the run label (and any other metadata text) never enters a file name
  directly: the names are built from the ``run_id`` only, so special
  characters in the label cannot break the export;
* ``export()`` before ``run()`` is rejected by the facade with
  ``ExperimentNotRunError`` (the T5 state contract), not here — a
  failed experiment (``status='failed'``, no model artifact) is a
  legitimate export input: its status and failure reason land in the
  metadata file.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from configurable_automl_engine.experiment.leaderboard import build_leaderboard
from configurable_automl_engine.experiment.reporting import (
    JSONReporter,
    MarkdownReporter,
)
from configurable_automl_engine.experiment.result import ExperimentResult

__all__ = [
    "ExportError",
    "export_experiment",
    "leaderboard_json_filename",
    "leaderboard_md_filename",
    "metadata_filename",
    "model_filename",
    "report_md_filename",
    "result_json_filename",
]


class ExportError(RuntimeError):
    """The export of the experiment bundle failed.

    Raised when the destination exists but is not a directory, or the
    model artifact recorded in the result is missing on disk. The
    destination directory keeps its previous state on any failure: the
    bundle is never left half-written.
    """


# ─────────────────────────── file naming ─────────────────────────── #


def _run_prefix(result: ExperimentResult) -> str:
    """The canonical file-name prefix of a run: its ``run_id``."""
    return str(result.run_id)


def model_filename(result: ExperimentResult) -> str:
    """The exported model file name for ``result``.

    The extension is inherited from the recorded artifact path (the
    serialization format marker); a path without a suffix falls back to
    ``.pkl``. Multi-suffix names (e.g. ``model.tar.gz``) keep their full
    trailing suffix chain — the exported name stays loadable by the
    same auto-detection logic the recorded path relies on.
    """
    suffix = "".join(Path(result.model_path).suffixes) if result.model_path else ""
    return f"{_run_prefix(result)}_model{suffix or '.pkl'}"


def leaderboard_json_filename(result: ExperimentResult) -> str:
    """The exported leaderboard JSON file name for ``result``."""
    return f"{_run_prefix(result)}_leaderboard.json"


def leaderboard_md_filename(result: ExperimentResult) -> str:
    """The exported leaderboard Markdown file name for ``result``."""
    return f"{_run_prefix(result)}_leaderboard.md"


def report_md_filename(result: ExperimentResult) -> str:
    """The exported full report (Markdown) file name for ``result``."""
    return f"{_run_prefix(result)}_report.md"


def result_json_filename(result: ExperimentResult) -> str:
    """The exported result snapshot (JSON) file name for ``result``."""
    return f"{_run_prefix(result)}_result.json"


def metadata_filename(result: ExperimentResult) -> str:
    """The exported run metadata (JSON) file name for ``result``."""
    return f"{_run_prefix(result)}_metadata.json"


def _leaderboard_payload(result: ExperimentResult) -> tuple[dict[str, Any], str]:
    """The leaderboard payload and table text of ``result``.

    The payload keeps the rendered table text plus the structured
    entries, so a consumer can consume the ranking without importing
    the engine. The board is built once here for both the JSON payload
    and the Markdown table text; the full report (``MarkdownReporter``)
    rebuilds the board internally — that second build is the reporter's
    own contract and is not deduplicated here.

    Returns:
        tuple[dict[str, Any], str]: The JSON-safe leaderboard payload
            and the rendered table text.
    """
    board = build_leaderboard(result)
    payload = {
        "metric_user": board.metric_user,
        "metric_direction": board.metric_direction.value,
        "table": board.to_table(),
        "entries": [
            {
                "rank": entry.rank,
                "algorithm": entry.algorithm,
                "score": entry.score,
                "trials": entry.trials,
                "phase": entry.phase,
                "phase_action": entry.phase_action.value,
                "params": entry.params,
                "training_time_seconds": entry.training_time_seconds,
                "n_trials_failed": entry.n_trials_failed,
            }
            for entry in board.entries
        ],
    }
    return payload, board.to_table()


def _metadata_payload(
    result: ExperimentResult, exported_files: dict[str, str]
) -> dict[str, Any]:
    """The run metadata payload: identity, status, timings, file map."""
    return {
        "run_id": str(result.run_id),
        "label": result.label,
        "status": result.status.value,
        "error": result.error,
        "engine_version": result.engine_version,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "duration_seconds": result.duration_seconds,
        "metric_user": result.metric_user,
        "winner": result.winner,
        "winner_score": result.winner_score,
        "model_source_path": result.model_path,
        "exported_files": exported_files,
    }


def _build_report_artifacts(
    result: ExperimentResult, bundle_dir: Path
) -> dict[str, str]:
    """Produce the leaderboard and report artifacts into ``bundle_dir``.

    The two JSON payloads are rendered fully in memory first, so a
    serialization failure happens before any file is written; the two
    Markdown/JSON documents are written by their reporters.

    Returns:
        dict[str, str]: The mapping "bundle file name → file role" for
            the produced files.

    Raises:
        Exception: Whatever a producer raises (a serialization error of
            a hand-built result, ...); the caller removes the partial
            bundle directory, the destination keeps its previous state.
    """
    exported: dict[str, str] = {}
    # In-memory render first: the payloads may fail before any file is
    # written.
    leaderboard_payload, leaderboard_table = _leaderboard_payload(result)
    report_text = str(MarkdownReporter().report(result))
    names = {
        leaderboard_json_filename(result): (
            json.dumps(leaderboard_payload, ensure_ascii=False, indent=2),
            "leaderboard_json",
        ),
        leaderboard_md_filename(result): (
            leaderboard_table + "\n",
            "leaderboard_md",
        ),
        report_md_filename(result): (report_text, "report_md"),
    }
    for name, (text, role) in names.items():
        (bundle_dir / name).write_text(text, encoding="utf-8")
        exported[name] = role
    snapshot_name = result_json_filename(result)
    JSONReporter(bundle_dir / snapshot_name).report(result)
    exported[snapshot_name] = "result_json"
    return exported


def _copy_model_artifact(result: ExperimentResult, bundle_dir: Path) -> str | None:
    """Copy the recorded model artifact into ``bundle_dir``.

    The original artifact is never moved, renamed or otherwise touched:
    the export makes a byte-exact copy under the bundle file name.

    Returns:
        str | None: The exported file name, or ``None`` when the run
            recorded no model path (a failed run without an artifact).

    Raises:
        ExportError: The recorded artifact does not exist on disk (a
            clear error instead of a silent hole in the bundle).
        OSError: The copy failed (permissions, disk, ...).
    """
    if not result.model_path:
        return None
    source = Path(result.model_path)
    if not source.is_file():
        raise ExportError(
            "Cannot export the model artifact: the path recorded in the "
            f"result does not exist or is not a file: {source}"
        )
    name = model_filename(result)
    shutil.copyfile(source, bundle_dir / name)
    return name


def export_experiment(result: ExperimentResult, directory: str | Path) -> Path:
    """Export the artifact bundle of one finished run into ``directory``.

    The bundle is assembled in a temporary directory inside the
    destination and moved into place at the end: a failure never leaves
    a partial bundle in the destination (the previous bundle, if any,
    stays untouched).

    Args:
        result: The finished experiment result to export.
        directory: The destination directory. Missing directories are
            created recursively; an existing bundle of the same run is
            overwritten (the last export wins).

    Returns:
        Path: The destination directory (containing the exported
            bundle).

    Raises:
        ExportError: The destination exists but is not a directory, or
            the recorded model artifact is missing on disk.
        OSError: The destination is not writable (the ``OSError`` of the
            filesystem propagates before any artifact is produced).
    """
    destination = Path(directory)
    if destination.exists() and not destination.is_dir():
        raise ExportError(
            f"Export destination exists but is not a directory: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".export-{_run_prefix(result)}-", dir=destination)
    )
    try:
        # 1. The leaderboard / report artifacts (the in-memory payloads
        # first, so a serialization failure cannot leave partial files).
        exported = _build_report_artifacts(result, temp_dir)
        # 2. The model copy (may raise ExportError for a missing source).
        model_name = _copy_model_artifact(result, temp_dir)
        if model_name is not None:
            exported[model_name] = "model"
        # 3. The metadata file closes the bundle (it records the final
        # exported-file map, including its own name).
        metadata_name = metadata_filename(result)
        exported[metadata_name] = "metadata"
        (temp_dir / metadata_name).write_text(
            json.dumps(
                _metadata_payload(result, exported), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        # 4. Move the finished bundle into place: every file swap is an
        # atomic rename; a failure in the middle is rolled back, so the
        # destination never keeps a half state.
        _swap_into_place(temp_dir, destination)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def _swap_into_place(temp_dir: Path, destination: Path) -> None:
    """Move the finished ``temp_dir`` bundle into ``destination``.

    The destination directory itself stays (it may hold unrelated
    files); the bundle files replace the previous ones. Each file move
    is an ``os.replace`` (atomic on the same filesystem). The swap runs
    in two phases so a failure can never damage the previous bundle:

    * the backup phase moves every existing destination file that the
      swap will overwrite into a ``.<name>.export-backup`` sibling; the
      destination then holds no bundle file under the names being
      swapped (each individual rename is atomic, but between the two
      phases the destination keeps the non-overwritten part of its
      previous state);
    * the move phase renames every temp file over its target: these
      renames cannot fail against an existing target (the name is free
      after the backup phase), so the phase is a straight sequence of
      atomic renames; a failure of an individual rename (e.g. an I/O
      error) is handled by the best-effort restore below.

    The restore, in turn, never aborts on its own errors: a failed
    restore of one file does not prevent the restoration of the others
    (the first error is remembered and re-raised after the loop), so a
    mid-swap failure degrades the destination state but never leaves
    ``.export-backup`` leftovers silently swallowing the bundle files.

    The temporary directory is removed at the end whatever the outcome.

    Raises:
        OSError: A rename failed; the best-effort restore of the
            previous state ran first.
    """
    # Phase 1 — backup: move the to-be-overwritten destination files
    # aside. After this loop every target name is free, so phase 2 is a
    # sequence of plain renames onto free names. A destination path
    # occupied by a DIRECTORY (not a file) cannot be backed aside and
    # replaced by a file rename: it is rejected before anything is
    # moved, with a clear error naming the path.
    backups: dict[Path, Path] = {}
    moved: list[Path] = []
    for entry in sorted(temp_dir.iterdir()):
        target = destination / entry.name
        if target.is_dir() and not target.is_symlink():
            raise ExportError(
                f"Cannot export over '{target.name}': the destination "
                f"holds a directory under the bundle file name: {target}"
            )
    try:
        for entry in sorted(temp_dir.iterdir()):
            target = destination / entry.name
            if target.exists():
                backup = target.with_name(f".{target.name}.export-backup")
                os.replace(target, backup)
                backups[target] = backup
    except BaseException:
        _restore_backups(backups, moved, temp_dir, destination)
        raise
    # Phase 2 — move: rename the temp files over the (now free) target
    # names.
    try:
        for entry in sorted(temp_dir.iterdir()):
            target = destination / entry.name
            os.replace(entry, target)
            moved.append(target)
    except BaseException:
        _restore_backups(backups, moved, temp_dir, destination)
        raise
    # Success: the backups are provably obsolete now. The cleanup is
    # best-effort (mirroring `_restore_backups`): one failing unlink
    # must not turn a fully successful export into a reported failure,
    # and must not leave the other stale backups behind — a leftover
    # backup is an inert hidden file, not a bundle-file swallower (the
    # real-name file is the fresh export).
    for backup in backups.values():
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            pass
    shutil.rmtree(temp_dir, ignore_errors=True)


def _restore_backups(
    backups: dict[Path, Path],
    moved: list[Path],
    temp_dir: Path,
    destination: Path,
) -> None:
    """Best-effort restore of the destination after a failed swap.

    Undo order (each step per-file, never aborting the rest):

    1. remove the files the move phase already put into place (they
       must not survive a failed export);
    2. move every still-present backup back over its original name; a
       rename that fails (an I/O error, a cross-device edge) falls back
       to copy + delete of the backup source, so the previous file is
       restored even when the atomic rename is impossible and no
       ``.export-backup`` file is left silently swallowing the bundle
       content.

    Nothing else is deleted: a destination file that matches a temp
    entry name may be a previous bundle file just restored in step 2
    (the step-3 "remove leftover temp targets" of the first draft
    removed the freshly restored files and is the exact bug the review
    round 1 reproduced).

    The first error is remembered and re-raised after the loop: one
    failed restoration must not prevent the others. Because the restore
    runs inside the caller's ``except`` block, the re-raised restore
    error replaces the original swap error as the exception the caller
    sees; the original swap error is preserved as its ``__context__``
    (implicit chaining), so neither failure is lost. A fully successful
    restore re-raises nothing — the original swap error propagates
    unchanged.
    """
    first_error: BaseException | None = None
    for target in moved:
        try:
            target.unlink(missing_ok=True)
        except OSError as err:
            if first_error is None:
                first_error = err
    for target, backup in backups.items():
        if not backup.exists():
            # The backup is gone (a failed restore rename has already
            # moved this backup onto its target name, or the file was
            # removed by an earlier restore step): nothing to do, and
            # the target must NOT be deleted — it may already hold the
            # restored previous file.
            continue
        try:
            if target.exists():
                target.unlink()
            try:
                os.replace(backup, target)
            except OSError:
                # The atomic rename is impossible (an I/O error, a
                # cross-device edge): copy the content back and remove
                # the backup source, so the restore still completes and
                # no backup file swallows the previous bundle content.
                shutil.copy2(backup, target)
                backup.unlink()
        except OSError as err:
            if first_error is None:
                first_error = err
    shutil.rmtree(temp_dir, ignore_errors=True)
    if first_error is not None:
        raise first_error
