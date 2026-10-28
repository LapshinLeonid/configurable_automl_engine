"""Adapter over ``concurrent.futures.ProcessPoolExecutor``.

The supervisor needs two things the public executor API does not give on
every supported Python version: the worker processes (to read the exit code
of a crashed worker) and a forced shutdown that does not wait for a hung
worker. Both live here. The only access to private executor internals is
``_worker_processes``; on Python 3.14+ the forced shutdown uses the public
``ProcessPoolExecutor.terminate_workers()`` / ``kill_workers()``.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any

__all__ = ["create_process_pool", "force_shutdown", "resolve_context"]

logger = logging.getLogger(__name__)

#: Python 3.14 added ProcessPoolExecutor.terminate_workers / kill_workers.
_HAS_PUBLIC_TERMINATE = hasattr(ProcessPoolExecutor, "terminate_workers")

#: How often a forced shutdown checks whether the workers have exited.
_EXIT_POLL = 0.05


def resolve_context(mp_context: str | BaseContext | None) -> BaseContext | None:
    """Turn a start-method name into a multiprocessing context.

    Args:
        mp_context (str | BaseContext | None): ``"fork"``, ``"forkserver"``,
            ``"spawn"``, a context object, or ``None`` (the multiprocessing
            default of the running interpreter).

    Returns:
        BaseContext | None: The context to pass to the executor.

    Raises:
        ValueError: If the start method is not available on this platform.
    """
    if mp_context is None or isinstance(mp_context, BaseContext):
        return mp_context
    return multiprocessing.get_context(mp_context)


def create_process_pool(
    max_workers: int | None, mp_context: BaseContext | None
) -> ProcessPoolExecutor:
    """Create a process pool (the workers start lazily, on submit)."""
    return ProcessPoolExecutor(max_workers, mp_context=mp_context)


def _worker_processes(pool: ProcessPoolExecutor) -> list[BaseProcess]:
    """The worker processes of the pool (empty once the pool is shut down).

    This is the only function that reads a private attribute of the
    executor: ``ProcessPoolExecutor`` keeps its workers in ``_processes``
    (a ``{pid: Process}`` dict, ``None`` after ``shutdown``) on every
    supported Python version, and there is no public accessor.
    """
    processes = getattr(pool, "_processes", None)
    return list(processes.values()) if processes else []


def force_shutdown(
    pool: ProcessPoolExecutor, grace_period: float
) -> dict[int, int | None]:
    """Stop the pool now: cancel queued tasks, terminate and reap the workers.

    The workers get SIGTERM at once (a hung worker never exits on its own);
    the ones still alive after ``grace_period`` seconds get SIGKILL.

    Args:
        pool (ProcessPoolExecutor): The pool to stop.
        grace_period (float): Seconds between SIGTERM and SIGKILL.

    Returns:
        dict[int, int | None]: The exit code of every worker the pool had,
        by PID (``None`` if a worker could not be reaped). A crashed worker
        keeps its own exit code; the workers this function terminated report
        ``-SIGTERM`` (or ``-SIGKILL``).
    """
    workers = _worker_processes(pool)
    public: Any = pool  # terminate_workers / kill_workers: Python 3.14+
    if _HAS_PUBLIC_TERMINATE and grace_period <= 0:
        public.kill_workers()
    elif _HAS_PUBLIC_TERMINATE:
        # terminate_workers() = shutdown(wait=False, cancel_futures=True)
        # followed by SIGTERM to every live worker. The shutdown drops the
        # executor's worker registry, so a later kill_workers() would find
        # nothing: the SIGKILL step below uses the process handles.
        public.terminate_workers()
    else:
        pool.shutdown(wait=False, cancel_futures=True)
        for w in workers:
            _signal(w, "terminate")
    stop_time = time.monotonic() + grace_period
    while time.monotonic() < stop_time and any(w.is_alive() for w in workers):
        time.sleep(_EXIT_POLL)
    for w in workers:
        if w.is_alive():
            logger.error(f"Worker {w.pid} resisted SIGTERM; sending SIGKILL")
            _signal(w, "kill")
    exit_codes: dict[int, int | None] = {}
    for w in workers:
        w.join(timeout=1.0)
        if w.pid is not None:
            exit_codes[w.pid] = w.exitcode
    return exit_codes


def _signal(worker: BaseProcess, action: str) -> None:
    try:
        getattr(worker, action)()
    except Exception:  # noqa: BLE001, S110
        pass  # already gone, already closed, or the OS refused
