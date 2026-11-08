"""JSONReporter: the full machine-readable snapshot of an experiment.

The reporter serializes the whole
:class:`~configurable_automl_engine.experiment.result.ExperimentResult`
(the config snapshot, the per-run records, the winner and the honest
scores) into one JSON file. The dump is produced by the pydantic model
itself (``model_dump(mode="json")``), so every value is JSON-safe by the
model's own invariant — the reporter adds no sanitization of its own
(numpy scalars, ``Path``, enums and datetimes are already normalized to
JSON types by the model; a hand-built payload that smuggles a
non-serializable object past the model validation fails with a
``TypeError`` naming the offending fields).

Policies (documented contract):

* atomic write: the JSON is rendered fully in memory first, written to a
  temporary file in the destination directory and renamed into place. A
  failure (including an unserializable value, which cannot happen for a
  validated model but may for a hand-built one) leaves the destination
  file untouched — no partial artifact;
* missing parent directories are created;
* an existing file is overwritten (the last report wins);
* the JSON is UTF-8 with ``ensure_ascii=False``: the report keeps the
  unicode (Cyrillic etc.) labels and names readable;
* the payload parses back (``json.loads``) into a dict equivalent to
  ``result.model_dump(mode="json")`` — the round-trip contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from configurable_automl_engine.experiment.reporting.base import (
    Reporter,
    ReporterArtifact,
)
from configurable_automl_engine.experiment.result import ExperimentResult

__all__ = ["JSONReporter"]


class JSONReporter(Reporter[ReporterArtifact]):
    """Write the full experiment snapshot to a JSON file.

    Args:
        path: Destination file path. Missing parent directories are
            created; an existing file is overwritten.
        indent: JSON indentation (``None`` — compact output, the
            default keeps the file small for large experiments).
    """

    def __init__(self, path: str | Path, *, indent: int | None = None) -> None:
        self._path = Path(path)
        self._indent = indent

    @property
    def path(self) -> Path:
        """The destination file path."""
        return self._path

    def report(self, result: ExperimentResult) -> ReporterArtifact:
        """Write the full result snapshot to the destination JSON file.

        The write is atomic: the payload is rendered fully first, then
        written to a temporary file next to the destination and renamed
        into place, so a failure never leaves a partial file behind.

        Args:
            result: The finished experiment result to snapshot.

        Returns:
            ReporterArtifact: The path of the written JSON file.

        Raises:
            TypeError: A value of the result cannot be serialized to
                JSON; the message names the offending top-level fields
                (only possible for values the model validation does not
                cover — e.g. an object injected past validation). The
                destination file stays untouched.
            OSError: The destination is not writable (e.g. a permission
                error); the destination file stays untouched.
        """
        try:
            payload = result.model_dump(mode="json")
        except (TypeError, ValueError) as err:
            # A validated result is JSON-safe by the model's own
            # invariant; this only fires for a hand-built result with an
            # object smuggled past the validation (e.g. a function in a
            # free-form field).
            raise TypeError(
                f"JSONReporter: the result {result.run_id} cannot be "
                f"serialized to JSON; the model validation should have "
                f"rejected it. Original error: {err}"
            ) from err
        try:
            text = json.dumps(
                payload,
                ensure_ascii=False,
                indent=self._indent,
                allow_nan=False,
            )
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"JSONReporter: the result {result.run_id} is not "
                f"JSON-serializable (offending fields: "
                f"{self._offending_fields(payload)}); the model validation "
                f"should have rejected it. Original error: {err}"
            ) from err
        self._atomic_write(text)
        return self._path

    @staticmethod
    def _offending_fields(payload: dict[str, Any]) -> str:
        """The top-level fields of ``payload`` that fail to serialize.

        A serialization error does not carry the field path, so the
        fields are probed individually to point at the offender(s).
        """
        bad = []
        for field, value in payload.items():
            try:
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError):
                bad.append(field)
        return ", ".join(bad) if bad else "<unknown>"

    def _atomic_write(self, text: str) -> None:
        """Write ``text`` to the destination atomically (tempfile + rename).

        The temporary file is created in the destination directory (the
        same filesystem, so the rename is atomic) and removed on any
        failure; an existing destination file is overwritten only after
        the full write succeeded. The temp path is registered before any
        write, so even a failure of the write itself (disk full, I/O
        error) leaves no orphaned temp file behind.

        The destination file gets the ordinary file-creation permissions
        (the process umask applied to ``0o666``): a private temp file is
        chmod'ed to that mode before the rename, so the report artifact
        is not stuck with the ``0o600`` of ``NamedTemporaryFile``.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        umask = os.umask(0)
        os.umask(umask)
        destination_mode = 0o666 & ~umask
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp:
                temp_path = temp.name  # registered before any write
                temp.write(text)
            os.chmod(temp_path, destination_mode)
            os.replace(temp_path, self._path)
            temp_path = None
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)
