"""Process-pool supervisor of the processes mode of ``run_parallel``.

``ProcessPoolSupervisor`` owns the executor, the start signals and the
futures of one batch. It submits the tasks, collects the finished ones and
restarts the pool when needed:

    - **Hard timeout.** A task that runs past its deadline (see
      ``parallel_deadlines``) is marked ``timed_out``; the pool is stopped
      (its worker is terminated) and rebuilt, and the unfinished tasks are
      resubmitted. A healthy task that was in flight in another worker is
      resubmitted from scratch — its progress is lost, which is the price of
      a hard timeout.
    - **Crashed worker** (``BrokenProcessPool``: ``os._exit``, a signal, a
      segfault, the OOM killer). The task whose worker died is found by the
      PID in its start signal and the worker's exit code (the other workers
      are terminated by the pool and report ``-SIGTERM``). It is charged one
      crash; the pool is rebuilt and the unfinished tasks are resubmitted.
      A task charged ``MAX_CRASH_ATTEMPTS`` crashes is marked ``crashed`` and
      not retried, so the run always terminates.

An infrastructure failure after the tasks were submitted (for example, the
replacement pool cannot be created) marks the unfinished tasks ``failed``
and keeps the finished results; any other exception (a programming error,
``KeyboardInterrupt``) propagates after the pool is stopped. Nothing is
ever rerun in threads from here.
"""

from __future__ import annotations

import logging
import os
import pickle
import signal
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any

from .parallel_deadlines import next_wait, queue_budget, task_deadline
from .parallel_executor import create_process_pool, force_shutdown
from .parallel_outcome import TaskError, TaskOutcome, TaskStatus
from .parallel_transport import TaskPayload
from .parallel_worker import _StartSignal, _worker_proxy

__all__ = [
    "MAX_CRASH_ATTEMPTS",
    "ProcessPoolSupervisor",
    "is_infrastructure_error",
]

logger = logging.getLogger(__name__)

#: A task whose worker crashed this many times is marked ``crashed``.
MAX_CRASH_ATTEMPTS = 2

_INFRASTRUCTURE_ERRORS = (
    OSError,
    RuntimeError,
    MemoryError,
    EOFError,
    pickle.PickleError,
)
_PROGRAMMING_RUNTIME_ERRORS = (NotImplementedError, RecursionError)

PoolFactory = Callable[[int | None, BaseContext | None], ProcessPoolExecutor]


def is_infrastructure_error(exc: BaseException) -> bool:
    """Whether ``exc`` is an OS / process-pool failure, not a programming error.

    Infrastructure errors (``OSError``, ``RuntimeError`` — ``BrokenProcessPool``
    included —, ``MemoryError``, ``EOFError``, pickling errors) are reported as
    task failures; anything else (``TypeError``, ``AttributeError``, ...)
    is a bug and must propagate.
    """
    return isinstance(exc, _INFRASTRUCTURE_ERRORS) and not isinstance(
        exc, _PROGRAMMING_RUNTIME_ERRORS
    )


@dataclass(slots=True)
class _Running:
    """One submitted, unfinished attempt of a task."""

    idx: int
    signal: _StartSignal
    submit_t: float

    def started_at(self) -> float | None:
        started = self.signal.read()
        return started if started >= 0 else None

    def free(self) -> None:
        self.signal.close()
        self.signal.unlink()


class ProcessPoolSupervisor:
    """Runs one batch of tasks in a process pool (see the module docstring).

    Args:
        func (Callable[..., Any]): The target function (picklable).
        payloads (Sequence[TaskPayload]): One payload per task.
        max_workers (int | None): The number of worker processes (``None`` —
            the CPU count).
        mp_context (BaseContext | None): The multiprocessing context.
        task_timeout (float | None): The per-task timeout (from the real
            start).
        timeout (float | None): The global timeout of the batch.
        shutdown_grace_period (float): Seconds between SIGTERM and SIGKILL
            when the pool is stopped.
        pool_factory (PoolFactory): Creates the executors.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        payloads: Sequence[TaskPayload],
        *,
        max_workers: int | None = None,
        mp_context: BaseContext | None = None,
        task_timeout: float | None = None,
        timeout: float | None = None,
        shutdown_grace_period: float = 5.0,
        pool_factory: PoolFactory = create_process_pool,
    ) -> None:
        self._func = func
        self._payloads = list(payloads)
        self._max_workers = max_workers
        self._mp_context = mp_context
        self._task_timeout = task_timeout
        self._timeout = timeout
        self._grace = shutdown_grace_period
        self._factory = pool_factory
        n = len(self._payloads)
        self._outcomes: list[TaskOutcome | None] = [None] * n
        self._attempts = [0] * n
        self._crashes = [0] * n
        self._running: dict[Future[Any], _Running] = {}
        # Tasks not submitted yet because the pool broke while submitting
        # (a worker crashed meanwhile): the crash recovery submits them.
        self._unsubmitted: list[int] = []
        self._pool: ProcessPoolExecutor | None = None
        # True while the current pool holds a worker that runs an abandoned
        # (timed-out) task: the pool must then be stopped, not drained.
        self._abandoned = False
        self._budget = queue_budget(n, max_workers or os.cpu_count() or 1, task_timeout)
        self._global_deadline: float | None = None

    def start(self) -> None:
        """Create the process pool (an error here means nothing was submitted)."""
        self._pool = self._factory(self._max_workers, self._mp_context)

    def run(self) -> list[TaskOutcome]:
        """Submit every task and wait for all of them.

        Returns:
            list[TaskOutcome]: One outcome per task, in the task order.

        Raises:
            KeyboardInterrupt: If interrupted (the pool is stopped first).
            Exception: A programming error of the supervisor or the
                executor (anything that is not an infrastructure error).
        """
        if self._pool is None:
            self.start()
        if self._timeout is not None:
            self._global_deadline = time.time() + self._timeout
        completed = False
        try:
            try:
                self._submit(range(len(self._payloads)))
                while self._running or self._unsubmitted:
                    self._step()
            except Exception as e:
                if not is_infrastructure_error(e):
                    raise
                logger.error("Process pool failed after tasks were submitted: %s", e)
                self._fail_unfinished(e)
            completed = True
        finally:
            self._close(graceful=completed)
        return [
            o if o is not None else self._lost(i) for i, o in enumerate(self._outcomes)
        ]

    # ── the wait loop ──────────────────────────────────────────────────────

    def _step(self) -> None:
        if not self._running:
            # The pool broke before any submitted task was collected.
            self._stop_pool()
            self._resubmit_unfinished()
            return
        now = time.time()
        if self._global_deadline is not None and now >= self._global_deadline:
            self._expire_all()
            return
        timed_out = [f for f, r in self._running.items() if now >= self._deadline(r)]
        if timed_out:
            self._handle_timeouts(timed_out, now)
            return
        wait_s = next_wait(
            now,
            [self._deadline(r) for r in self._running.values()],
            self._global_deadline,
            self._task_timeout is not None
            and any(r.started_at() is None for r in self._running.values()),
        )
        done, _ = wait(list(self._running), timeout=wait_s, return_when=FIRST_COMPLETED)
        broken = False
        for fut in done:
            broken = self._collect(fut) or broken
        if broken:
            self._recover_from_crash()

    def _deadline(self, r: _Running) -> float:
        return task_deadline(
            r.submit_t, r.started_at(), self._task_timeout, self._budget
        )

    def _collect(self, fut: Future[Any]) -> bool:
        """Record a finished future; return True if its pool is broken."""
        exc = None if fut.cancelled() else fut.exception()
        if isinstance(exc, BrokenProcessPool):
            return True
        r = self._running.pop(fut)
        started = r.started_at()
        duration = time.time() - started if started is not None else None
        r.free()
        if fut.cancelled():
            self._set(r.idx, "cancelled", TaskError("Cancelled", "cancelled"), duration)
        elif exc is None:
            self._outcomes[r.idx] = TaskOutcome(
                "ok", fut.result(), attempts=self._attempts[r.idx], duration=duration
            )
        elif isinstance(exc, KeyboardInterrupt):
            logger.error("Interrupted by user")
            raise exc
        else:
            logger.error(f"Task {r.idx} failed: {exc}")
            self._set(r.idx, "failed", TaskError.from_exception(exc), duration)
        return False

    def _collect_finished(self) -> None:
        """Record every future that already has its own result or exception."""
        for fut in list(self._running):
            if fut.done() and not fut.cancelled():
                self._collect(fut)

    # ── timeouts ───────────────────────────────────────────────────────────

    def _handle_timeouts(self, timed_out: Iterable[Future[Any]], now: float) -> None:
        for fut in timed_out:
            if fut.done() and not self._collect(fut):
                continue  # it finished just in time
            r = self._running.pop(fut)
            if r.started_at() is not None:
                message = f"timed out after {self._task_timeout:g} s"
            else:
                message = f"did not start within {now - r.submit_t:.1f} s of the submit"
            logger.error(f"Task {r.idx} timed out")
            self._set(r.idx, "timed_out", TaskError("TaskTimeout", message), None)
            r.free()
            self._abandoned = True
        if self._running or self._unsubmitted:
            # The hung worker keeps running its task: stop the pool and
            # rebuild it for the tasks still waiting.
            self._restart()

    def _expire_all(self) -> None:
        message = f"the global timeout of {self._timeout:g} s expired"
        for fut, r in list(self._running.items()):
            if fut.done() and not self._collect(fut):
                continue
            self._running.pop(fut)
            if r.started_at() is not None:
                logger.error(f"Task {r.idx} timed out")
                self._set(r.idx, "timed_out", TaskError("TaskTimeout", message), None)
            else:
                logger.error(f"Task {r.idx} cancelled: {message}")
                error = TaskError(
                    "Cancelled", f"cancelled: {message} before it started"
                )
                self._set(r.idx, "cancelled", error, None)
            r.free()
            self._abandoned = True

    # ── restarts ───────────────────────────────────────────────────────────

    def _stop_pool(self) -> dict[int, int | None]:
        """Stop the current pool; return the exit codes of its workers."""
        pool, self._pool = self._pool, None
        if pool is None:
            return {}
        self._collect_finished()
        exit_codes = force_shutdown(pool, self._grace)
        self._collect_finished()
        return exit_codes

    def _restart(self) -> None:
        self._stop_pool()
        self._resubmit_unfinished()

    def _recover_from_crash(self) -> None:
        exit_codes = self._stop_pool()
        blamed = self._blame(exit_codes)
        for fut, r in list(self._running.items()):
            if r.idx not in blamed:
                continue
            self._crashes[r.idx] += 1
            code = blamed[r.idx]
            message = f"worker process crashed (exit code {code if code is not None else 'unknown'})"
            if self._crashes[r.idx] >= MAX_CRASH_ATTEMPTS:
                logger.error(f"Task {r.idx}: {message}; not retried")
                self._running.pop(fut)
                r.free()
                self._set(r.idx, "crashed", TaskError("WorkerCrash", message), None)
            else:
                logger.warning(f"Task {r.idx}: {message}; retrying")
        if self._running or self._unsubmitted:
            logger.warning(
                "A worker process crashed; restarting the pool for %d unfinished "
                "task(s)",
                len(self._running) + len(self._unsubmitted),
            )
            self._resubmit_unfinished()

    def _blame(self, exit_codes: dict[int, int | None]) -> dict[int, int | None]:
        """The tasks charged with the crash, with the exit code of their worker.

        The task that ran in a worker which died with its own exit code (not
        the ``-SIGTERM`` the pool sends to the other workers) is blamed. If
        no such worker is known, every started task is blamed; if none has
        started, every unfinished task is — each crash charges at least one
        task, so the number of restarts is bounded.
        """
        pids = {r.idx: r.signal.read_pid() for r in self._running.values()}
        started = {idx: pid for idx, pid in pids.items() if pid > 0}
        blamed = {
            idx: exit_codes[pid]
            for idx, pid in started.items()
            if exit_codes.get(pid) not in (None, -signal.SIGTERM)
        }
        if not blamed:
            blamed = {idx: exit_codes.get(pid) for idx, pid in started.items()}
        if not blamed:
            blamed = dict.fromkeys(pids)
        return blamed

    def _resubmit_unfinished(self) -> None:
        pending = sorted([r.idx for r in self._running.values()] + self._unsubmitted)
        self._unsubmitted = []
        for r in self._running.values():
            r.free()
        self._running.clear()
        self._abandoned = False
        self._pool = self._factory(self._max_workers, self._mp_context)
        # The resubmitted tasks get a fresh submit time: keeping the original
        # one would squeeze them against the queue deadline after several
        # restarts. The restarts are bounded (each one is caused by a timed
        # out or a crash-charged task), so the run still terminates.
        self._submit(pending)

    def _submit(self, indices: Iterable[int]) -> None:
        assert self._pool is not None
        queue = list(indices)
        for pos, idx in enumerate(queue):
            p = self._payloads[idx]
            sig = _StartSignal()
            try:
                fut = self._pool.submit(
                    _worker_proxy,
                    self._func,
                    p.args,
                    p.kwargs,
                    p.disk_indices,
                    p.shm_info,
                    sig.name,
                )
            except BrokenProcessPool:
                # A worker crashed while the batch was being submitted: the
                # futures already submitted report it, and the crash recovery
                # submits the rest to the replacement pool.
                sig.close()
                sig.unlink()
                self._unsubmitted.extend(queue[pos:])
                return
            except BaseException:
                sig.close()
                sig.unlink()
                raise
            self._attempts[idx] += 1
            self._running[fut] = _Running(idx, sig, time.time())

    # ── outcomes and cleanup ───────────────────────────────────────────────

    def _set(
        self,
        idx: int,
        status: TaskStatus,
        error: TaskError,
        duration: float | None,
    ) -> None:
        self._outcomes[idx] = TaskOutcome(
            status, None, error, attempts=self._attempts[idx], duration=duration
        )

    def _fail_unfinished(self, exc: BaseException) -> None:
        message = f"process pool failure: {exc}{self._reimport_hint()}"
        error = TaskError(type(exc).__name__, message)
        for idx, outcome in enumerate(self._outcomes):
            if outcome is None:
                logger.error(f"Task {idx} failed: {message}")
                self._outcomes[idx] = TaskOutcome(
                    "failed", None, error, attempts=self._attempts[idx]
                )

    def _reimport_hint(self) -> str:
        """A hint for the usual cause of a pool failing under forkserver/spawn."""
        method = self._mp_context.get_start_method() if self._mp_context else None
        if method not in ("forkserver", "spawn"):
            return ""
        return (
            f" (with the {method!r} start method the workers re-import the main "
            "script: guard its top-level code with "
            "`if __name__ == \"__main__\":`, or use the 'fork' start method)"
        )

    def _lost(self, idx: int) -> TaskOutcome:
        # Unreachable when run() returns normally; kept as a safe default.
        return TaskOutcome(
            "failed",
            None,
            TaskError("Lost", "the task has no recorded outcome"),
            attempts=self._attempts[idx],
        )

    def _close(self, graceful: bool) -> None:
        pool, self._pool = self._pool, None
        try:
            if pool is not None:
                if graceful and not self._running and not self._abandoned:
                    pool.shutdown(wait=True)
                else:
                    force_shutdown(pool, self._grace)
        finally:
            for r in self._running.values():
                r.free()
            self._running.clear()
