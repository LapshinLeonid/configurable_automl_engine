"""Forwarding of the worker log records to the parent process.

With the ``forkserver`` and ``spawn`` start methods a worker process does not
inherit the logging configuration of the parent (handlers, levels), so its
records would never reach the parent's log file. ``WorkerLogForwarding``
fixes that for every start method:

    - The parent takes a snapshot of its logging configuration (the levels,
      ``propagate`` / ``disabled`` flags and which loggers have handlers) and
      starts a listener thread on a multiprocessing queue.
    - Each worker runs ``_init_worker_logging`` as the pool initializer: it
      drops the handlers it may have inherited (``fork``), applies the
      snapshot, and puts a forwarding handler on every logger that has
      handlers in the parent. A record therefore stops at the same loggers as
      it would in the parent and is forwarded once per such logger.
    - The listener hands each record to the handlers of exactly the logger it
      was forwarded from (respecting the handler levels), so the parent's
      formatters, files and console output apply unchanged.

A worker killed in the middle of a write can leave the queue unreadable; the
listener then stops after a timeout instead of blocking the run.
"""

from __future__ import annotations

import importlib
import logging
import logging.handlers
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from types import TracebackType
from typing import Any, Self

__all__ = ["LoggingSnapshot", "WorkerLogForwarding", "snapshot_logging"]

logger = logging.getLogger(__name__)

#: The record attribute naming the logger whose handlers receive the record.
_TARGET_ATTR = "_forwarded_to"

#: How long ``stop()`` waits for the listener to drain the queue (seconds).
_LISTENER_STOP_TIMEOUT = 5.0


@dataclass(frozen=True, slots=True)
class _LoggerState:
    name: str  # "" — the root logger
    level: int
    propagate: bool
    disabled: bool
    has_handlers: bool


@dataclass(frozen=True, slots=True)
class LoggingSnapshot:
    """The parts of the parent's logging configuration a worker reproduces.

    Attributes:
        loggers (tuple[_LoggerState, ...]): The root logger and every logger
            whose level, flags or handlers differ from the defaults.
    """

    loggers: tuple[_LoggerState, ...]


def _logger(name: str) -> logging.Logger:
    return logging.getLogger() if name == "" else logging.getLogger(name)


def snapshot_logging() -> LoggingSnapshot:
    """Take the snapshot of the current process's logging configuration."""
    root = logging.getLogger()
    states = [_LoggerState("", root.level, True, root.disabled, bool(root.handlers))]
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if not isinstance(lg, logging.Logger):
            continue  # a PlaceHolder: no logger of that name exists yet
        if lg.level or not lg.propagate or lg.disabled or lg.handlers:
            states.append(
                _LoggerState(
                    name, lg.level, lg.propagate, lg.disabled, bool(lg.handlers)
                )
            )
    return LoggingSnapshot(tuple(states))


class _ForwardHandler(logging.handlers.QueueHandler):
    """Worker side: puts the record on the queue, tagged with its logger."""

    def __init__(self, queue: Any, target: str) -> None:
        super().__init__(queue)
        self._target = target

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # QueueHandler.prepare merges args and the traceback into the
        # message, so the record pickles without the original objects.
        prepared = super().prepare(record)
        setattr(prepared, _TARGET_ATTR, self._target)
        return prepared


def _init_worker_logging(
    queue: Any, snapshot: LoggingSnapshot, modules: Sequence[str] = ()
) -> None:
    """Pool initializer: route the worker's log records to the parent.

    Args:
        queue: The multiprocessing queue read by the parent's listener.
        snapshot (LoggingSnapshot): The parent's logging configuration.
        modules (Sequence[str]): Modules to import first, so the handlers
            they install at import time (e.g. optuna's) are replaced as well.
    """
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001, S112 — the task reports the error
            continue
    # The queue's feeder thread must not keep a dying worker alive.
    queue.cancel_join_thread()
    existing = [logging.getLogger()] + [
        lg
        for lg in logging.root.manager.loggerDict.values()
        if isinstance(lg, logging.Logger)
    ]
    for lg in existing:
        for handler in list(lg.handlers):
            lg.removeHandler(handler)  # inherited (fork) or import-time
    for state in snapshot.loggers:
        lg = _logger(state.name)
        lg.setLevel(state.level)
        lg.propagate = state.propagate
        lg.disabled = state.disabled
        if state.has_handlers:
            lg.addHandler(_ForwardHandler(queue, state.name))


class _Dispatch(logging.Handler):
    """Parent side: hands a forwarded record to its logger's own handlers."""

    def handle(self, record: logging.LogRecord) -> bool:
        target = _logger(getattr(record, _TARGET_ATTR, record.name))
        for handler in target.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)
        return True

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        self.handle(record)


class _Listener(logging.handlers.QueueListener):
    """A QueueListener whose ``stop`` cannot block the run forever."""

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self.enqueue_sentinel()
        thread.join(_LISTENER_STOP_TIMEOUT)
        if thread.is_alive():
            # A worker killed mid-write left the queue unreadable: give up
            # on the remaining records (the thread is a daemon).
            logger.warning("Worker log forwarding did not stop in time")
        self._thread = None


class WorkerLogForwarding:
    """Owns the queue and the listener that forward worker log records.

    Use as a context manager around the lifetime of the process pools; pass
    ``initializer`` / ``initargs`` to every pool it covers.

    Args:
        mp_context (BaseContext): The context of the worker processes.
        modules (Sequence[str]): Modules the workers import before their
            logging is configured (see ``_init_worker_logging``).
    """

    def __init__(self, mp_context: BaseContext, modules: Sequence[str] = ()) -> None:
        self._queue = mp_context.Queue()
        self._listener = _Listener(self._queue, _Dispatch())
        self._snapshot = snapshot_logging()
        self._modules = tuple(modules)
        self._started = False

    @property
    def initializer(self) -> Any:
        """The pool initializer (``_init_worker_logging``)."""
        return _init_worker_logging

    @property
    def initargs(self) -> tuple[Any, ...]:
        """The pool initializer arguments."""
        return (self._queue, self._snapshot, self._modules)

    def __enter__(self) -> Self:
        self._listener.start()
        self._started = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._started:
            self._listener.stop()
            self._started = False
        self._queue.close()
        self._queue.cancel_join_thread()
