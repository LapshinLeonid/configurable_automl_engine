"""Module for parallel execution of tasks with memory management.
Provides tools for running functions in parallel (ThreadPool/ProcessPool)
with optimized transfer of heavy pandas.DataFrame objects through
Shared Memory and Disk Persistence (disk cache) mechanisms.
Main components:
    - SharedDataFrame (``common/shared_frame.py``): A container for placing
      data in the OS shared memory.
    - DiskPersistenceManager: A manager for temporary Parquet data storage.
    - run_parallel: A universal interface for running computations.
Usage example:
    results = run_parallel(
        func=my_heavy_function,
        args_seq=[(df1,), (df2,)],
        mode="processes",
        shared_args_indices=[0]
    )
"""

import concurrent.futures
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any

from .parallel_deadlines import POLL_INTERVAL, next_wait, queue_budget, task_deadline
from .parallel_transport import DiskPersistenceManager, PayloadTransport
from .parallel_worker import _StartSignal, _worker_proxy

__all__ = ["POLL_INTERVAL", "DiskPersistenceManager", "run_parallel"]

logger = logging.getLogger(__name__)


def _force_shutdown_processes(pool: Any, shutdown_grace_period: float = 5.0) -> None:
    """Forcefully shut down all processes in the pool.

    Used by the watchdog mechanism when hung tasks are detected
    (a C-level worker crash).

    Args:
        pool: ProcessPoolExecutor or ThreadPoolExecutor
        shutdown_grace_period: graceful shutdown wait time in seconds
    """
    workers = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)

    stop_time = time.time() + shutdown_grace_period
    while time.time() < stop_time and any(w.is_alive() for w in workers):
        time.sleep(0.1)

    for w in workers:
        if w.is_alive():
            try:
                logger.error(f"CRITICAL: Worker {w.pid} hung. Forcing SIGTERM.")
                w.terminate()
            except Exception:  # noqa: S110, BLE001
                pass

    time.sleep(0.2)

    for w in workers:
        if w.is_alive():
            try:
                logger.critical(
                    f"HARD KILL: Worker {w.pid} resisted SIGTERM. Sending SIGKILL."
                )
                w.kill()
            except Exception:  # noqa: S110, BLE001
                pass


def _pid_exists(pid: int) -> bool:
    """Check whether a process with the given PID is present in the OS.

    Uses a zero-signal probe (``os.kill(pid, 0)``): a successful call means
    the process exists, ``ProcessLookupError`` means it is gone, and
    ``PermissionError`` still means it exists (it is simply owned by another
    user). Used to distinguish an already-dead worker (nothing to terminate)
    from a stale/foreign PID.

    Args:
        pid (int): The PID to probe.

    Returns:
        bool: True if the process exists.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_worker_by_pid(
    pool: Any, pid: int, shutdown_grace_period: float = 5.0
) -> bool:
    """Terminate a specific worker process by its OS PID.

    Used by the hard-timeout mechanism: when a task exceeds ``task_timeout``,
    its result is discarded, but the worker process that executes it continues
    running the hung function. The PID of the worker is recorded in the task's
    ``_StartSignal`` at the moment the task really starts; terminating exactly
    that process frees the worker right away, without waiting the full graceful
    shutdown period for a process that will never exit on its own.

    Args:
        pool: The executor whose worker processes are inspected.
        pid: The OS PID of the worker process to terminate.
        shutdown_grace_period: Time in seconds allowed for the graceful exit
            before SIGKILL is sent.

    Returns:
        bool: True if the worker with this PID was found and terminated, or
            if it has already exited (a race: the process was reaped by the
            executor's management thread before the lookup) — in both cases
            the hung task is no longer running. False only if a live process
            with this PID exists but is not a worker of this pool.
    """
    for proc in list(getattr(pool, "_processes", {}).values()):
        if getattr(proc, "pid", None) != pid:
            continue
        try:
            proc.terminate()
        except Exception:  # noqa: S110, BLE001
            pass
        stop_time = time.time() + shutdown_grace_period
        while time.time() < stop_time and proc.is_alive():
            time.sleep(0.1)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:  # noqa: S110, BLE001
                pass
        return True
    # The worker is no longer registered in the pool: this can only happen
    # when it has already exited (the executor's management thread reaps
    # finished/crashed workers) — e.g. a racing task timeout. There is
    # nothing to terminate: the hung function is not running anymore, so the
    # outcome is treated as success. A live but unregistered PID (foreign
    # process / PID reuse) is reported as not handled.
    return not _pid_exists(pid)


def run_parallel(  # noqa: C901
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

        Parallelism logic:
            1. Mode: supports threads and processes.
            2. Data optimization: in processes mode, moves DataFrames into Shared Memory
               or to disk, eliminating Pickle serialization overhead.
            3. Lifecycle management: uses explicit Executor
               initialization and shutdown (with cancel_futures=True). This allows
               interrupting execution immediately on failures or timeouts.
            4. Global timer: The timeout parameter limits the total execution time
               of the entire sequence of tasks in processes mode. If the limit is exceeded,
               result collection stops and the remaining queued tasks are cancelled.
            5. Fault tolerance: if the initialization or execution of the process pool
               fails, the whole sequence is retried in "threads" mode
               with forced resource cleanup.
            6. Resource management: guarantees removal of memory segments and temporary
               files and the correct stop of all worker processes/threads.
            7. Timeout semantics: in "processes" mode the timeout is hard: when
               a task exceeds ``task_timeout``, the worker process executing it
               is terminated (identified by the PID recorded in the start
               signal) and the pool is restarted for the remaining tasks — a
               hung task no longer blocks the tasks queued behind it. While
               there are tasks that have not started yet, the wait loop polls
               the start signals with a short interval (POLL_INTERVAL), so the
               timeout of a task that has already started is noticed promptly
               and not at the queue deadline.
               Timeouts in "threads" mode are ignored: an already running Python
               function cannot be reliably interrupted, so a soft timeout did not
               save time and only forced the phase to wait for shutdown(wait=True)
               completion of all algorithms and then discard their ready results.
               If timeout / task_timeout are passed together with mode="threads"
               (including direct API calls bypassing config validation), a warning
               is emitted and the timeouts are reset to None — all tasks run to
               their natural completion.

    Args:
            func (Callable): The function to run.
            args_seq (Iterable): Sequence of argument tuples.
            kwargs_seq (Iterable): Sequence of keyword-argument dicts.
            max_workers (int): Maximum number of worker processes.
            mode (str): Parallelism mode ("threads" or "processes").
            timeout (int | float | None): Global time limit in seconds
                for running all tasks. Applies only in "processes" mode
                (hard timeout: hung workers are terminated forcibly).
                None — no global limit. In "threads" mode ignored
                with a warning.
            shared_args_indices (list[int]): DataFrame indices for Shared Memory.
                Data in shared memory is passed positionally and the index is
                NOT transferred: the worker always receives the frame with a
                RangeIndex(0, n), even when the original index differed (e.g.
                RangeIndex(100, 100+n)) — worker code must not align X and y
                by index. Note that an int64/uint64 column mixed
                with a float column is promoted to float64 for the worker,
                which is lossless as long as no value exceeds 2**53 in
                absolute value.
            disk_args_indices (list[int]): DataFrame indices for the disk cache.
            shutdown_grace_period (float): Time in seconds allowed for the graceful shutdown
                of workers before SIGTERM/SIGKILL is forcibly sent.
            task_timeout (float | None): Per-task (per-algorithm) timeout in seconds.
                If None — the global timeout is used. Counted from the moment the
                task actually starts in the worker (the timestamp is recorded by the
                worker through a side-channel), not from the moment of submit: while
                a task waits in the queue (for example, when there are more
                algorithms than max_workers), the queue does not "consume" its
                execution budget — after starting, the task gets up to task_timeout
                seconds of execution. So that a task that never starts (for example,
                due to a hung worker) does not hang forever, the deadline is bounded
                from above by submit_time + task_timeout + queue_budget, where
                queue_budget = ceil(number_of_tasks / max_workers) * task_timeout
                (for the whole batch drain). When a started task exceeds its
                deadline, the worker process executing it is terminated and the
                pool is restarted for the remaining tasks (they are resubmitted
                with a fresh submit deadline; the number of restarts is bounded
                by the number of timed-out tasks, so the run still terminates).
                A healthy task that is in-flight in another worker at the
                restart moment is resubmitted from scratch too — its progress
                is lost (the deliberate price of the hard timeout), so runtime
                can multiply when several timeouts occur in a row.
                Applies only in "processes" mode
                (hard timeout). In "threads" mode ignored with a warning.

            Returns:
                list[Any]: List of results. Tasks that did not finish in time
                    or raised exceptions (including InvalidAlgorithmError),
                    are replaced with None. User interruptions (KeyboardInterrupt)
                    are re-raised to the calling code.
    """
    pool = None

    args_seq = list(args_seq or [()])
    kwargs_seq = list(kwargs_seq or [{}] * len(args_seq))

    if len(args_seq) != len(kwargs_seq):
        raise ValueError("args_seq and kwargs_seq must be of equal length")

    # Timeouts in threads mode: an already running Python function cannot be
    # interrupted, so a soft timeout does not save time and only discarded the
    # ready results of completed algorithms after waiting for
    # shutdown(wait=True). Config validation already forbids timeouts in the
    # threads mode, but it does not protect against direct run_parallel calls —
    # therefore timeouts are ignored here with a warning rather than silently.
    if mode == "threads":
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
        timeout = None
        task_timeout = None

    # Start-signals of the actual task starts (only for processes)
    start_signals: list[_StartSignal] = []
    transport = (
        PayloadTransport(shared_args_indices, disk_args_indices)
        if mode == "processes"
        else PayloadTransport()
    )
    execution_tasks = [
        (p.args, p.kwargs, p.disk_indices, p.shm_info)
        for p in transport.prepare(args_seq, kwargs_seq)
    ]
    # Pre-allocate the list to preserve length and order
    results: list[Any] = [None] * len(execution_tasks)

    # 1. Determine the executor class
    executor_cls: Callable[[int | None], Executor] = ThreadPoolExecutor
    if mode == "processes":
        executor_cls = ProcessPoolExecutor

    start_time = time.time()

    try:
        pool = executor_cls(max_workers)

        future_to_idx = {}
        submit_time_by_future: dict[Future[Any], float] = {}
        signal_by_future: dict[Future[Any], _StartSignal | None] = {}

        def submit_one(
            pool_obj: Any,
            idx: int,
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
            disk_idx: list[int],
            shm_idx: dict[int, tuple[Any, ...]],
        ) -> None:
            """Submit one execution task and register its future.

            In processes mode a side-channel of the actual start is created for
            the task: the worker writes a timestamp and its PID into it, the
            parent binds the deadline to the timestamp and uses the PID to kill
            the worker on a hard timeout. ``future.running()`` is not suitable
            — ProcessPoolExecutor marks a task as running already when it
            enters the pool's internal queue.

            Args:
                pool_obj: The executor to submit to.
                idx: The task index (also the results[] slot).
                args: The positional arguments of the task.
                kwargs: The keyword arguments of the task.
                disk_idx: Indices of arguments restored from disk.
                shm_idx: Metadata of the arguments restored from shared memory.
            """
            if mode == "processes":
                sig = _StartSignal()
                start_signals.append(sig)
                fut = pool_obj.submit(
                    _worker_proxy, func, args, kwargs, disk_idx, shm_idx, sig.name
                )
                signal_by_future[fut] = sig
            else:
                fut = pool_obj.submit(func, *args, **kwargs)
                signal_by_future[fut] = None
            future_to_idx[fut] = idx
            submit_time_by_future[fut] = time.time()

        for i, (a, kw, d_idx, s_idx) in enumerate(execution_tasks):
            submit_one(pool, i, a, kw, d_idx, s_idx)

        def restart_pool(pool_obj: Any, hung_futures: list[Future[Any]]) -> Executor:
            """Kill the hung workers and restart the pool for the remaining tasks.

            A timed-out task does not free its worker: the result is discarded
            but the process keeps running the hung function to completion, so
            every task queued behind it never starts (with more tasks than
            workers, one hung algorithm silently drops the whole queue). For a
            hard timeout the worker processes executing the timed-out tasks are
            terminated first (identified by the PID recorded in the start
            signal), then the pool is forcibly shut down and rebuilt, and the
            remaining tasks are resubmitted to the fresh pool.

            Cost trade-off: the forced shutdown also kills the healthy tasks
            that happen to be in-flight in other workers at that moment. Their
            results are correct (they are resubmitted from scratch), but their
            progress is lost, so runtime can multiply when several timeouts
            occur in a row. This is the deliberate price of the hard per-task
            timeout: without it a single hung algorithm would block the whole
            queue.

            Args:
                pool_obj: The pool to kill.
                hung_futures: The futures of the timed-out tasks.

            Returns:
                Executor: The restarted pool.
            """
            hung_pids: set[int] = set()
            for fut in hung_futures:
                sig = signal_by_future.get(fut)
                if sig is not None:
                    pid = sig.read_pid()
                    if pid > 0:
                        hung_pids.add(pid)
            # All signals registered so far belong to the killed pool (both the
            # timed-out futures and the remaining ones): they can be freed right
            # away instead of accumulating one 16-byte segment per task until
            # the final cleanup.
            orphan_signals = [sig for sig in list(signal_by_future.values())]
            for pid in hung_pids:
                _terminate_worker_by_pid(pool_obj, pid, shutdown_grace_period)
            _force_shutdown_processes(pool_obj, shutdown_grace_period)
            for sig in orphan_signals:
                if sig is not None:
                    sig.close()
                    sig.unlink()
            new_pool = executor_cls(max_workers)
            for old_fut, idx in list(future_to_idx.items()):
                args, kwargs, disk_idx, shm_idx = execution_tasks[idx]
                submit_time_by_future.pop(old_fut, None)
                signal_by_future.pop(old_fut, None)
                future_to_idx.pop(old_fut)
                # The resubmitted tasks get a fresh submit deadline: keeping the
                # original one would squeeze them against the batch absolute
                # bound (submit + task_timeout + queue_budget) after several
                # restarts and could kill the last queued tasks before they
                # even run. The number of restarts is bounded by the number of
                # timed-out tasks, so the loop still terminates.
                submit_one(new_pool, idx, args, kwargs, disk_idx, shm_idx)
            return new_pool

        # Upper bound on the time a task may spend in the queue before the
        # absolute (non-rollable) deadline kicks in. It covers the whole batch
        # drain (ceil(n/max_workers) waves of task_timeout), so any task that
        # actually starts executing (within task_timeout) gets its full
        # execution budget, while a task that never starts (for example, due to
        # a hung worker) does not hang forever even with timeout=None.
        worker_count = max_workers or (os.cpu_count() or 1)
        budget = queue_budget(len(future_to_idx), worker_count, task_timeout)
        global_deadline = start_time + timeout if timeout is not None else None

        while future_to_idx:
            now = time.time()

            # Check: if the global timeout has expired — exit
            if global_deadline is not None and now >= global_deadline:
                for fut, idx in future_to_idx.items():
                    logger.error(f"Task {idx} timed out")
                    results[idx] = None
                future_to_idx.clear()
                break

            # Recompute the deadlines.
            # - A task that has already started (the worker recorded the
            #   timestamp): the deadline is counted from the real start
            #   (started + task_timeout), but no later than the absolute upper
            #   bound submit + task_timeout + queue_budget.
            # - A task still in the queue: the deadline is fixed
            #   (submit + task_timeout + queue_budget) and does not depend on
            #   the current time — the queue does not "consume" the execution
            #   budget, and the task itself does not "hang".
            deadline_by_future: dict[Future[Any], float] = {}
            for fut in list(future_to_idx):
                signal_obj = signal_by_future.get(fut)
                started = signal_obj.read() if signal_obj is not None else -1.0
                deadline_by_future[fut] = task_deadline(
                    submit_time_by_future.get(fut, start_time),
                    started if started >= 0 else None,
                    task_timeout,
                    budget,
                )

            # Check: if the per-task timeout (task_timeout) has expired.
            # In processes mode the timed-out task is discarded and its worker
            # process is killed, then the pool is rebuilt for the remaining
            # tasks — otherwise the hung process keeps occupying the worker
            # and the tasks queued behind it never start.
            timed_out_futures = [
                fut
                for fut, deadline in deadline_by_future.items()
                if fut in future_to_idx and now >= deadline
            ]
            if timed_out_futures:
                for fut in timed_out_futures:
                    idx = future_to_idx.pop(fut)
                    logger.error(f"Task {idx} timed out")
                    results[idx] = None
                if mode == "processes" and future_to_idx:
                    pool = restart_pool(pool, timed_out_futures)
                    continue
            if not future_to_idx:
                break

            # While a task has not started yet, the wait is capped at
            # POLL_INTERVAL (see parallel_deadlines).
            any_unstarted = task_timeout is not None and any(
                (sig := signal_by_future.get(fut)) is not None and sig.read() < 0
                for fut in future_to_idx
            )
            timeout_for_ac = next_wait(
                now,
                [deadline_by_future[fut] for fut in future_to_idx],
                global_deadline,
                any_unstarted,
            )

            try:
                if timeout_for_ac is not None:
                    for fut in as_completed(
                        future_to_idx.keys(), timeout=timeout_for_ac
                    ):
                        idx = future_to_idx.pop(fut)
                        try:
                            results[idx] = fut.result(timeout=0)
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Task {idx} failed: {e}")
                            results[idx] = None
                else:
                    for fut in as_completed(future_to_idx.keys()):
                        idx = future_to_idx.pop(fut)
                        try:
                            results[idx] = fut.result(timeout=0)
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Task {idx} failed: {e}")
                            results[idx] = None

            except concurrent.futures.TimeoutError:
                # as_completed did not finish any task within timeout_for_ac
                # Just keep waiting — this is not a C-level crash, merely slow tasks
                continue

            except KeyboardInterrupt:
                logger.error("Interrupted by user")
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error while waiting for tasks: {e}")
                future_to_idx.clear()
                break

    except Exception as e:
        if mode == "processes":
            logger.error("Error in process pool, falling back to threads: %s", e)

            transport.close()
            return run_parallel(
                func,
                args_seq,
                kwargs_seq,
                max_workers,
                mode="threads",
                # Timeouts are forbidden in threads mode, so they are not passed
                # to the fallback retry.
                timeout=None,
                task_timeout=None,
            )
        raise
    finally:
        if pool is not None:
            is_proc_executor = type(pool).__name__ == "ProcessPoolExecutor" or hasattr(
                pool, "_processes"
            )
            if mode == "processes" and is_proc_executor:
                _force_shutdown_processes(pool, shutdown_grace_period)
            else:
                # For ThreadPoolExecutor or if the mode is not "processes"
                pool.shutdown(wait=True, cancel_futures=True)
        transport.close()
        # Cleanup of the actual-start signals (created only in processes)
        for sig in start_signals:
            sig.close()
            sig.unlink()
    return results
