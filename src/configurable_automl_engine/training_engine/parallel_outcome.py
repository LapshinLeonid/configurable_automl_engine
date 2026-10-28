"""The result of one task of ``run_parallel`` (``TaskOutcome``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["TaskError", "TaskOutcome", "TaskStatus"]

#: ``ok`` — the function returned a value; ``failed`` — it raised, or the
#: infrastructure failed before it could finish; ``timed_out`` — it ran out
#: of its task timeout or of the global timeout; ``crashed`` — its worker
#: process died (os._exit, a signal, a segfault, the OOM killer);
#: ``cancelled`` — it never started because the global timeout expired.
TaskStatus = Literal["ok", "failed", "timed_out", "crashed", "cancelled"]


@dataclass(frozen=True, slots=True)
class TaskError:
    """Why a task did not produce a value.

    Attributes:
        type (str): The exception type name (``"ValueError"``), or a short
            kind for failures without an exception (``"TaskTimeout"``,
            ``"WorkerCrash"``, ``"Cancelled"``).
        message (str): A human-readable description.
    """

    type: str
    message: str

    @classmethod
    def from_exception(cls, exc: BaseException) -> TaskError:
        """Build the error from an exception (type name and message)."""
        return cls(type(exc).__name__, str(exc))


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """The result of one task.

    Attributes:
        status (TaskStatus): How the task ended.
        value (Any): The returned value (``None`` unless ``status == "ok"``).
        error (TaskError | None): Why the task has no value (``None`` for
            ``ok``).
        attempts (int): How many times the task was submitted (a pool
            restart resubmits the unfinished tasks; 0 — never submitted).
        duration (float | None): The execution time in seconds of the last
            attempt, counted from the real start; ``None`` if it never
            started.
    """

    status: TaskStatus
    value: Any = None
    error: TaskError | None = None
    attempts: int = 1
    duration: float | None = None

    @property
    def ok(self) -> bool:
        """Whether the task returned a value."""
        return self.status == "ok"

    def reason(self) -> str:
        """A one-line reason of a non-``ok`` outcome (for logs and reports).

        Returns:
            str: For example ``"timed out after 600 s"``,
            ``"worker process crashed (exit code -9)"`` or
            ``"failed: ValueError: bad input"``; ``""`` for ``ok``.
        """
        if self.status == "ok":
            return ""
        message = self.error.message if self.error is not None else ""
        if self.status == "failed" and self.error is not None:
            return f"failed: {self.error.type}: {message}"
        return message or self.status.replace("_", " ")
