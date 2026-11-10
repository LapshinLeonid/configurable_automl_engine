"""Experiment store: the run history of the engine.

``ExperimentStore`` keeps the full ``ExperimentResult`` models produced by
the engine and serves them to the consumers (the facade and the reporters)
without re-running any training.

Policies (documented contract):

* registration is keyed by ``run_id``; registering a duplicate raises
  :class:`DuplicateRunError` unless ``force=True`` is passed (the forced
  registration replaces the stored result);
* the run label is taken from ``ExperimentResult.label`` and is *not*
  required to be unique: a label lookup returns every matching run;
* a missing ``run_id`` lookup raises ``KeyError``; a missing label lookup
  returns an empty list;
* with a ``path`` configured, every registration is appended to the file
  as one JSON line (JSONL) and the history is restored from the file at
  store creation. A damaged line (truncated write, garbage, or a record
  that no longer passes model validation after a schema change) is
  skipped with a :class:`RuntimeWarning` — never a startup crash. If the
  same ``run_id`` appears several times in the file, the last occurrence
  wins (the file is an append-only log);
* all operations are thread-safe (a single re-entrant lock): the engine
  may run experiments in parallel threads (``general.parallel_mode:
  threads``) and register results concurrently; no update is lost and
  the JSONL appends are serialized.
"""

from __future__ import annotations

import datetime as dt
import threading
import typing
import uuid
import warnings
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from configurable_automl_engine.common.definitions import RunStatus
from configurable_automl_engine.experiment.result import ExperimentResult

__all__ = [
    "DuplicateRunError",
    "ExperimentStore",
    "ExperimentStoreError",
    "RunSummary",
]


class ExperimentStoreError(Exception):
    """Base error of the experiment store."""


class DuplicateRunError(ExperimentStoreError):
    """A run with the same ``run_id`` is already registered.

    Raised by :meth:`ExperimentStore.add` when ``force=False`` (the
    default): a ``run_id`` identifies one experiment run, silently
    overwriting it would corrupt the history.
    """


class RunSummary(BaseModel):
    """Lightweight metadata of one registered run.

    The list endpoint returns summaries instead of the full
    ``ExperimentResult`` payloads: listing a large history (thousands of
    runs) stays cheap, and the full model is loaded only by an explicit
    ``get`` / ``find_by_label`` call.

    Attributes:
        run_id: The unique identifier of the run.
        label: The optional human-readable label of the run.
        status: The experiment outcome ('completed' or 'failed').
        started_at: The run start time (UTC).
        finished_at: The run finish time (UTC).
        duration_seconds: The wall-clock duration of the run.
        engine_version: The version of the engine that produced the run.
    """

    model_config = ConfigDict(frozen=True)

    run_id: uuid.UUID
    label: str | None
    status: RunStatus
    started_at: dt.datetime
    finished_at: dt.datetime
    duration_seconds: float
    engine_version: str | None

    @classmethod
    def from_result(cls, result: ExperimentResult) -> RunSummary:
        """Build the summary from a full ``ExperimentResult``.

        Args:
            result: The full result model of the run.

        Returns:
            RunSummary: The metadata-only view of the run.
        """
        return cls(
            run_id=result.run_id,
            label=result.label,
            status=result.status,
            started_at=result.started_at,
            finished_at=result.finished_at,
            duration_seconds=result.duration_seconds,
            engine_version=result.engine_version,
        )


#: Fields a run list can be sorted by.
SummaryOrderField = typing.Literal["started_at", "finished_at", "duration_seconds"]


class ExperimentStore:
    """Thread-safe history of experiment runs with an optional JSONL persist.

    The store keeps the full ``ExperimentResult`` models in memory (an
    ordered ``run_id`` → result mapping) and, when constructed with a
    ``path``, mirrors every registration to a JSONL file and restores the
    history from it at startup.

    Args:
        path: Optional path of the JSONL history file. Missing parent
            directories are created on the first write; when ``None`` the
            store is memory-only.
        restore: Restore the history from ``path`` at construction
            (default ``True``; with ``False`` the file is only appended
            to). Damaged lines are always skipped with a warning — see
            the module docstring for the full policy.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        restore: bool = True,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._results: dict[uuid.UUID, ExperimentResult] = {}
        self._file = None
        if self._path is not None and restore:
            self._restore()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")

    # ────────────────────────── registration ────────────────────────── #

    def add(self, result: ExperimentResult, *, force: bool = False) -> None:
        """Register a finished experiment run.

        Args:
            result: The full result model of the run (the label is taken
                from ``result.label``).
            force: Replace an already registered run with the same
                ``run_id`` instead of raising :class:`DuplicateRunError`.

        Raises:
            DuplicateRunError: The ``run_id`` is already registered and
                ``force`` is ``False``.
        """
        with self._lock:
            if result.run_id in self._results and not force:
                raise DuplicateRunError(
                    f"Run {result.run_id} is already registered "
                    "(pass force=True to replace it)"
                )
            self._results[result.run_id] = result
            self._append(result)

    # ──────────────────────────── lookups ───────────────────────────── #

    def get(self, run_id: uuid.UUID | str) -> ExperimentResult:
        """Return the full result of one run.

        Args:
            run_id: The run identifier (a ``UUID`` or its canonical string
                form).

        Returns:
            ExperimentResult: The full result model of the run.

        Raises:
            KeyError: No run with this ``run_id`` is registered.
        """
        key = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
        with self._lock:
            try:
                return self._results[key]
            except KeyError:
                raise KeyError(f"No run with run_id={key} is registered") from None

    def find_by_label(self, label: str) -> list[ExperimentResult]:
        """Return every run carrying the label, in registration order.

        Labels are not required to be unique, so the lookup returns a
        list; an unknown label yields an empty list.

        Args:
            label: The exact label to search for.

        Returns:
            list[ExperimentResult]: The matching full result models.
        """
        with self._lock:
            return [
                result for result in self._results.values() if result.label == label
            ]

    def list_runs(
        self,
        *,
        status: RunStatus | str | None = None,
        label: str | None = None,
        order_by: SummaryOrderField = "started_at",
        descending: bool = False,
    ) -> list[RunSummary]:
        """List the registered runs as lightweight metadata summaries.

        Args:
            status: Keep only runs with this outcome ('completed' or
                'failed'; a plain string is accepted); ``None`` keeps all.
            label: Keep only runs with this exact label; ``None`` keeps
                all.
            order_by: The summary field to sort by ('started_at',
                'finished_at' or 'duration_seconds'). The sort is stable:
                equal keys keep the registration order.
            descending: Sort from the largest key to the smallest.

        Returns:
            list[RunSummary]: The filtered and sorted run summaries.

        Raises:
            ValueError: ``status`` is not a valid ``RunStatus`` value.
        """
        status_filter = RunStatus(status) if status is not None else None
        with self._lock:
            summaries = [
                RunSummary.from_result(result)
                for result in self._results.values()
                if (status_filter is None or result.status is status_filter)
                and (label is None or result.label == label)
            ]
        summaries.sort(
            key=lambda summary: getattr(summary, order_by), reverse=descending
        )
        return summaries

    # ──────────────────────────── protocol ──────────────────────────── #

    def __len__(self) -> int:
        """The number of registered runs."""
        with self._lock:
            return len(self._results)

    def __contains__(self, run_id: object) -> bool:
        """Whether a run with this identifier is registered."""
        try:
            key = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
        except (TypeError, ValueError):
            return False
        with self._lock:
            return key in self._results

    def close(self) -> None:
        """Close the JSONL file handle (idempotent; a no-op without a path)."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> typing.Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the store (see :meth:`close`)."""
        self.close()

    # ─────────────────────────── internals ──────────────────────────── #

    def _append(self, result: ExperimentResult) -> None:
        """Append one registration to the JSONL file (the lock is held)."""
        if self._file is None:
            return
        self._file.write(result.to_json() + "\n")
        self._file.flush()

    def _restore(self) -> None:
        """Load the history from the JSONL file.

        Damaged lines (invalid JSON, or a record rejected by the model
        validation after a schema change) are skipped with a
        :class:`RuntimeWarning` — a damaged history must not prevent the
        engine from starting. For a duplicated ``run_id`` the last
        occurrence in the file wins.
        """
        if self._path is None or not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                result = self._parse_line(line, line_number)
                if result is not None:
                    self._results[result.run_id] = result

    def _parse_line(self, line: str, line_number: int) -> ExperimentResult | None:
        """Parse one JSONL line into a result model, or skip it.

        Returns:
            ExperimentResult | None: The parsed model, or ``None`` when
            the line is damaged (a warning is emitted in that case).
        """
        try:
            return ExperimentResult.model_validate_json(line)
        except Exception as err:  # noqa: BLE001 - any damage skips the line
            warnings.warn(
                f"{self._path}: skipping damaged history line {line_number}: {err}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
