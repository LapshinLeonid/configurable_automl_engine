"""Parallel execution of tasks in threads or in processes.

``run_parallel`` (public) and ``run_parallel_outcomes`` (internal, used by
the orchestrator) are thin wrappers over small parts:

    - ``parallel_transport``: ``PayloadTransport`` sends heavy
      ``pandas.DataFrame`` arguments through shared memory, the disk cache
      (``DiskPersistenceManager``) or pickle, and frees what it created.
    - ``parallel_deadlines``: the pure deadline policy (task timeout from the
      real start, queue budget, ``POLL_INTERVAL`` cap).
    - ``parallel_supervisor``: ``ProcessPoolSupervisor`` owns the process
      pool, restarts it on a hard timeout and on a crashed worker.
    - ``parallel_executor``: the adapter over ``ProcessPoolExecutor``.
    - ``parallel_outcome``: ``TaskOutcome``, the result of one task.

Usage example:
    results = run_parallel(
        func=my_heavy_function,
        args_seq=[(df1,), (df2,)],
        mode="processes",
        shared_args_indices=[0]
    )
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from multiprocessing.context import BaseContext
from typing import Any

from .parallel_deadlines import POLL_INTERVAL
from .parallel_executor import create_process_pool, resolve_context
from .parallel_outcome import TaskError, TaskOutcome
from .parallel_supervisor import (
    MAX_CRASH_ATTEMPTS,
    ProcessPoolSupervisor,
    is_infrastructure_error,
)
from .parallel_transport import DiskPersistenceManager, PayloadTransport

__all__ = [
    "MAX_CRASH_ATTEMPTS",
    "POLL_INTERVAL",
    "DiskPersistenceManager",
    "TaskOutcome",
    "run_parallel",
    "run_parallel_outcomes",
]

logger = logging.getLogger(__name__)

_MODES = ("threads", "processes")


def _normalize_tasks(
    args_seq: Iterable[Sequence[Any]] | None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None,
) -> tuple[list[Sequence[Any]], list[Mapping[str, Any]]]:
    """Materialize the task arguments.

    ``args_seq=None`` means one call without arguments; an explicitly empty
    ``args_seq`` means zero tasks.

    Raises:
        ValueError: If ``args_seq`` and ``kwargs_seq`` differ in length.
    """
    args_list: list[Sequence[Any]] = [()] if args_seq is None else list(args_seq)
    kwargs_list: list[Mapping[str, Any]] = (
        [{} for _ in args_list] if kwargs_seq is None else list(kwargs_seq)
    )
    if len(args_list) != len(kwargs_list):
        raise ValueError("args_seq and kwargs_seq must be of equal length")
    return args_list, kwargs_list


def _warn_threads_timeouts(timeout: float | None, task_timeout: float | None) -> None:
    # Config validation forbids timeouts in the threads mode, but it does not
    # protect direct run_parallel calls — they are ignored with a warning
    # rather than silently.
    if timeout is not None or task_timeout is not None:
        logger.warning(
            "Timeouts are ignored in threads mode: an "
            "already-running Python function cannot be interrupted, so "
            "a soft timeout does not save time and only discards the "
            "ready results of completed algorithms. "
            f"timeout={timeout!r}, task_timeout={task_timeout!r} are being "
            "ignored. Remove the timeouts or switch mode to 'processes' "
            "(where timeouts are hard and enforced)."
        )


def _run_threads(
    func: Callable[..., Any],
    args_list: Sequence[Sequence[Any]],
    kwargs_list: Sequence[Mapping[str, Any]],
    max_workers: int | None,
) -> list[TaskOutcome]:
    """Run the tasks in a thread pool, without timeouts.

    Raises:
        KeyboardInterrupt: If a task (or the user) interrupts the run.
    """
    started: list[float | None] = [None] * len(args_list)

    def call(idx: int) -> Any:
        started[idx] = time.time()
        return func(*args_list[idx], **kwargs_list[idx])

    outcomes: list[TaskOutcome] = []
    with ThreadPoolExecutor(max_workers) as pool:
        futures = [pool.submit(call, i) for i in range(len(args_list))]
        try:
            for _ in as_completed(futures):
                pass  # collect in the input order below
        except KeyboardInterrupt:
            logger.error("Interrupted by user")
            for fut in futures:
                fut.cancel()
            raise
        for idx, fut in enumerate(futures):
            t0 = started[idx]
            duration = time.time() - t0 if t0 is not None else None
            outcomes.append(_thread_outcome(idx, fut, duration))
    return outcomes


def _thread_outcome(idx: int, fut: Future[Any], duration: float | None) -> TaskOutcome:
    exc = fut.exception()
    if exc is None:
        return TaskOutcome("ok", fut.result(), duration=duration)
    if isinstance(exc, KeyboardInterrupt):
        logger.error("Interrupted by user")
        raise exc
    logger.error(f"Task {idx} failed: {exc}")
    return TaskOutcome("failed", None, TaskError.from_exception(exc), duration=duration)


def _fallback_to_threads(
    what: str,
    exc: BaseException,
    func: Callable[..., Any],
    args_list: Sequence[Sequence[Any]],
    kwargs_list: Sequence[Mapping[str, Any]],
    max_workers: int | None,
) -> list[TaskOutcome]:
    logger.warning(
        "%s failed before any task was submitted (%s); falling back to threads. "
        "The timeouts are dropped: a task running in a thread cannot be "
        "interrupted.",
        what,
        exc,
    )
    return _run_threads(func, args_list, kwargs_list, max_workers)


def _run_processes(
    func: Callable[..., Any],
    args_list: Sequence[Sequence[Any]],
    kwargs_list: Sequence[Mapping[str, Any]],
    *,
    max_workers: int | None,
    mp_context: BaseContext | None,
    timeout: float | None,
    task_timeout: float | None,
    shared_args_indices: list[int] | None,
    disk_args_indices: list[int] | None,
    shutdown_grace_period: float,
) -> list[TaskOutcome]:
    """Run the tasks in a supervised process pool.

    Threads are used instead only when nothing has run yet: the payload
    transport cannot be set up, or the process pool cannot be created.
    """
    with PayloadTransport(shared_args_indices, disk_args_indices) as transport:
        try:
            payloads = transport.prepare(args_list, kwargs_list)
            supervisor = ProcessPoolSupervisor(
                func,
                payloads,
                max_workers=max_workers,
                mp_context=mp_context,
                task_timeout=task_timeout,
                timeout=timeout,
                shutdown_grace_period=shutdown_grace_period,
                pool_factory=create_process_pool,
            )
            supervisor.start()
        except Exception as e:
            if not is_infrastructure_error(e):
                raise
            transport.close()
            return _fallback_to_threads(
                "Setting up the process pool",
                e,
                func,
                args_list,
                kwargs_list,
                max_workers,
            )
        return supervisor.run()


def run_parallel_outcomes(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None,
    mode: str = "threads",
    timeout: float | None = None,
    shared_args_indices: list[int] | None = None,
    disk_args_indices: list[int] | None = None,
    shutdown_grace_period: float = 5.0,
    task_timeout: float | None = None,
    mp_context: str | BaseContext | None = None,
) -> list[TaskOutcome]:
    """Run ``func`` over the tasks and return one ``TaskOutcome`` per task.

    The internal entry point behind :func:`run_parallel` (same arguments);
    the orchestrator uses it to report why a task has no value.

    Returns:
        list[TaskOutcome]: The outcomes in the input order.

    Raises:
        ValueError: If ``mode`` is unknown, or ``args_seq`` and
            ``kwargs_seq`` differ in length.
        KeyboardInterrupt: Re-raised after the workers and the shared
            resources are cleaned up.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be 'threads' or 'processes', got {mode!r}")
    args_list, kwargs_list = _normalize_tasks(args_seq, kwargs_seq)
    if not args_list:
        return []
    if mode == "threads":
        _warn_threads_timeouts(timeout, task_timeout)
        return _run_threads(func, args_list, kwargs_list, max_workers)
    return _run_processes(
        func,
        args_list,
        kwargs_list,
        max_workers=max_workers,
        mp_context=resolve_context(mp_context),
        timeout=timeout,
        task_timeout=task_timeout,
        shared_args_indices=shared_args_indices,
        disk_args_indices=disk_args_indices,
        shutdown_grace_period=shutdown_grace_period,
    )


def run_parallel(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None,
    mode: str = "threads",
    timeout: float | None = None,
    shared_args_indices: list[int] | None = None,
    disk_args_indices: list[int] | None = None,
    shutdown_grace_period: float = 5.0,
    task_timeout: float | None = None,  # per-task timeout
) -> list[Any]:
    """Run a function in parallel with memory and lifecycle management.

    Semantics:
        1. Tasks: ``args_seq=None`` runs ``func`` once without arguments; an
           empty ``args_seq`` runs nothing and returns ``[]``.
        2. Data transport (processes mode): DataFrame arguments go through
           shared memory or the disk cache instead of pickle, and every
           segment / temp file is freed on every exit path.
        3. Timeouts (processes mode, hard): ``task_timeout`` counts from the
           real start of a task in the worker; a task that never starts is
           bounded by ``submit + task_timeout + queue_budget`` with
           ``queue_budget = ceil(n_tasks / max_workers) * task_timeout``. A
           timed-out task's worker is terminated and the pool is restarted
           for the remaining tasks (a healthy task in flight at that moment
           is resubmitted from scratch). ``timeout`` bounds the whole batch.
           While tasks have not started, the wait is capped at
           ``POLL_INTERVAL``. In threads mode the timeouts are ignored with a
           warning: a running Python function cannot be interrupted.
        4. Crash recovery (processes mode): when a worker process dies
           (``os._exit``, a signal, the OOM killer), only its task is lost:
           the pool is restarted and the unfinished tasks are resubmitted. A
           task that crashes its worker ``MAX_CRASH_ATTEMPTS`` times is given
           up on, so the run always terminates.
        5. Fallback to threads: only when nothing has run yet — the process
           pool cannot be created, or the payload transport cannot be set
           up (a WARNING says the timeouts are dropped). An infrastructure
           failure after tasks were submitted fails the unfinished tasks and
           keeps the finished results; nothing is rerun.

    Args:
        func (Callable): The function to run (picklable in processes mode).
        args_seq (Iterable): Sequence of argument tuples.
        kwargs_seq (Iterable): Sequence of keyword-argument dicts.
        max_workers (int): Maximum number of workers.
        mode (str): Parallelism mode ("threads" or "processes").
        timeout (float | None): Global time limit in seconds for the whole
            batch (processes mode only). None — no limit.
        shared_args_indices (list[int]): DataFrame indices for Shared Memory.
            Data in shared memory is passed positionally and the index is
            NOT transferred: the worker always receives the frame with a
            RangeIndex(0, n) — worker code must not align X and y by index.
            An int64/uint64 column mixed with a float column is promoted to
            float64 for the worker (lossless up to 2**53 in absolute value).
        disk_args_indices (list[int]): DataFrame indices for the disk cache.
        shutdown_grace_period (float): Seconds a terminated worker gets
            between SIGTERM and SIGKILL.
        task_timeout (float | None): Per-task timeout in seconds, from the
            real start in the worker (processes mode only).

    Returns:
        list[Any]: The results in the input order. A task that failed, timed
            out, crashed its worker or was cancelled gives ``None``
            (``run_parallel_outcomes`` tells which).

    Raises:
        ValueError: If ``mode`` is unknown, or ``args_seq`` and
            ``kwargs_seq`` differ in length.
        KeyboardInterrupt: Re-raised to the caller after cleanup.
    """
    outcomes = run_parallel_outcomes(
        func,
        args_seq,
        kwargs_seq,
        max_workers=max_workers,
        mode=mode,
        timeout=timeout,
        shared_args_indices=shared_args_indices,
        disk_args_indices=disk_args_indices,
        shutdown_grace_period=shutdown_grace_period,
        task_timeout=task_timeout,
    )
    return [o.value if o.status == "ok" else None for o in outcomes]
