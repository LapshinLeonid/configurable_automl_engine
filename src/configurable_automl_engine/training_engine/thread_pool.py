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
import math
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.common.shared_frame import SharedDataFrame

logger = logging.getLogger(__name__)

# The wait loop polls the actual-start signals while there are tasks that
# have not started yet: a task can start at any moment and its real deadline
# (start + task_timeout) may then be much closer than the queue deadline
# (submit + task_timeout + queue_budget). as_completed must not sleep longer
# than this interval in that case, otherwise the timeout of an already-started
# task is noticed with a delay of the whole queue budget.
POLL_INTERVAL = 0.5

# Layout of the _StartSignal shared memory segment: a float64 start timestamp
# followed by an int64 worker PID. Both fields are written by the worker in a
# single assignment (a structured-record write), so the parent cannot observe
# a started timestamp with a still-empty PID. The PID is stored as int64, not
# float64, to avoid precision loss for large PIDs.
_START_SIGNAL_DTYPE = np.dtype([("start", np.float64), ("pid", np.int64)])


class DiskPersistenceManager:
    """Utility for temporarily saving a DataFrame to disk in Parquet format.

    Used as an alternative to Shared Memory when the data is too large for
    RAM or when strict typing via a disk cache is required.

    Attributes:
        tmp_dir (str | None): Path to the temporary directory (e.g. /dev/shm).
        created_files (list[str]): List of paths to the created temporary files.
    """

    def __init__(self, use_shm: bool = True):
        # Use /dev/shm on Linux if available, otherwise the standard temp dir
        # (a /dev/shm that cannot hold the file, e.g. the 64MB
        # Docker default, is detected by free_parquet_space before the Parquet
        # write; save_df then falls back to the system temp directory).
        self.tmp_dir = "/dev/shm" if use_shm and os.path.exists("/dev/shm") else None
        self.created_files: list[str] = []

    def free_parquet_space(self) -> int:
        """Free space (bytes) available on the Parquet temp file system.

        Used to decide whether the Parquet cache can fit the data without
        exhausting the file system: on a tmpfs /dev/shm (the
        Docker/Kubernetes default is 64MB) writing a large Parquet file can
        otherwise raise a catchable OSError (ENOSPC on write) — unlike the
        SIGBUS of the SHM path — but the pre-check keeps the disk cache on
        /dev/shm only when it safely fits.

        Returns:
            int: Free bytes on the file system of ``tmp_dir``, or 0 when the
            directory is not usable (e.g. /dev/shm is missing or a broken
            mount).
        """
        if self.tmp_dir is None:
            # The standard temp directory is always considered usable: the
            # system temp dir is not a tiny ramdisk, and even a full disk
            # raises a catchable OSError instead of crashing the process.
            return 1 << 62
        try:
            st = os.statvfs(self.tmp_dir)
        except (OSError, ValueError):
            return 0
        return st.f_bavail * st.f_frsize

    def save_df(self, df: pd.DataFrame) -> str:
        """Save a DataFrame to a temporary Parquet file.
        Saving logic:
        1. Location: the file is created in /dev/shm (RAM disk) to speed up operations
           on Linux, or in the system temporary folder.
        2. Format: Parquet is written via the pyarrow engine.
        3. Registration: the file path is added to the list for later cleanup.
        Args:
            df (pd.DataFrame): The dataset to save.
        Returns:
            str: Absolute path to the created temporary file.
        """
        # A conservative size estimate of the Parquet file: the in-memory
        # footprint of the DataFrame (deep=True accounts for object/string
        # columns with long values, which the old 8-bytes-per-element
        # heuristic underestimated) plus 1KB of Parquet metadata. Files
        # exceeding the free space on /dev/shm (a 64MB default in
        # Docker/Kubernetes) go to the system temp directory
        # instead: on a full tmpfs the write would either raise a catchable
        # ENOSPC (a warning and pickle fallback) or, for the SHM path, kill
        # the process with SIGBUS.
        if self.tmp_dir == "/dev/shm":
            approx = int(df.memory_usage(deep=True).sum()) + 1024
            if approx > self.free_parquet_space():
                logger.warning(
                    "Not enough free space in /dev/shm for the Parquet cache "
                    f"(~{approx} bytes needed, "
                    f"{self.free_parquet_space()} available); "
                    "writing the temporary file to the system temp directory "
                    "instead"
                )
                target_dir = None  # the standard system temporary directory
            else:
                target_dir = self.tmp_dir
        else:
            target_dir = self.tmp_dir

        fd, path = tempfile.mkstemp(suffix=".parquet", dir=target_dir)
        os.close(fd)
        # Register the path right after creation: if to_parquet fails halfway
        # (e.g. pyarrow rejects "dirty" data), the orphaned temp file must
        # still be removed by cleanup.
        self.created_files.append(path)
        df.to_parquet(path, engine="pyarrow")
        return path

    def cleanup(self) -> None:
        """Delete all created temporary files.
        Cleanup logic:
        1. Iteration: walks through the paths registered in created_files.
        2. Safety: suppresses file-missing errors if the file was deleted earlier.
        Returns:
            None
        """
        for path in self.created_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except (OSError, FileNotFoundError) as e:
                logger.warning(f"Failed to delete temp file {path}: {e}")


class _StartSignal:
    """Side-channel for recording the actual start of a task inside a worker.

    In processes mode the ``task_timeout`` deadline must be counted from the
    moment the task really starts executing in the worker, not from the moment
    it was submitted to the pool. ``future.running()`` is not suitable for
    this: ``ProcessPoolExecutor`` marks a task as running already when it is
    moved to the pool's internal queue (capacity ``max_workers+1``), i.e.
    before the actual start.

    ``_StartSignal`` is a shared memory segment (16 bytes: a float64 start
    timestamp and an int64 worker PID) into which the worker writes
    ``time.time()`` and ``os.getpid()`` right before calling the target
    function, while the parent reads them in the wait loop. The PID allows the
    parent to terminate exactly the worker process that executes a hung task
    (hard timeout). It is initialized with ``-1.0`` / ``0`` (meaning "not
    started yet"); real timestamps are always positive and PIDs always
    positive. It is removed by the owner (the parent) via ``unlink``.
    """

    def __init__(self) -> None:
        self._shm = shared_memory.SharedMemory(create=True, size=16)
        self._owner = True
        self.name = self._shm.name
        # A single structured view over the segment: the worker writes both
        # the start timestamp and its PID in one assignment (see
        # ``_mark_worker_start``), the parent reads them through typed fields.
        self._signal = np.ndarray((1,), dtype=_START_SIGNAL_DTYPE, buffer=self._shm.buf)
        self._signal[0] = (-1.0, 0)  # "not started yet"

    def mark_start(self, value: float) -> None:
        """Record the timestamp of the task's actual start."""
        self._signal[0]["start"] = value

    def mark_pid(self, pid: int) -> None:
        """Record the PID of the worker process executing the task."""
        self._signal[0]["pid"] = pid

    def read(self) -> float:
        """Return the start timestamp (``-1.0`` — the task has not started yet)."""
        return float(self._signal[0]["start"])

    def read_pid(self) -> int:
        """Return the worker PID (``0`` — the task has not started yet)."""
        return int(self._signal[0]["pid"])

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:  # noqa: S110, BLE001
            pass

    def unlink(self) -> None:
        if self._owner:
            try:
                self._shm.unlink()
            except (FileNotFoundError, OSError):
                pass
        self._owner = False


def _mark_worker_start(start_signal_name: str) -> None:
    """Record the actual start timestamp and worker PID in the shared-memory signal.

    Called by the worker right before executing the target function.

    Args:
        start_signal_name (str): Name of the shared memory segment of the signal.
    """
    try:
        shm = shared_memory.SharedMemory(name=start_signal_name)
        try:
            signal = np.ndarray((1,), dtype=_START_SIGNAL_DTYPE, buffer=shm.buf)
            # One assignment writes both fields: the parent can never observe
            # a started timestamp together with an empty PID.
            signal[0] = (time.time(), os.getpid())
        finally:
            shm.close()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Start-signal write failed: {e}")


def _worker_proxy(
    func: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    disk_indices: list[int] | None,
    shm_info: dict[int, tuple[Any, ...]],
    start_signal_name: str | None = None,
) -> Any:
    """Deserialize data and run the target function inside a worker.
    Execution logic:
    1. Shared Memory: finds SharedDataFrame objects by index and
       converts them back into DataFrames.
    2. Disk: reads Parquet files by the passed paths and restores DataFrames.
    3. Call: passes the restored data to the target function func.
    4. Cleanup: closes the local Shared Memory handles and forcibly
        clears the argument list to speed up the Garbage Collector.
    Args:
        func (Callable): Target function to run.
        args (Sequence): List of arguments (including proxy objects).
        kwargs (Mapping): Keyword arguments.
        disk_indices (list[int]): Indices of arguments saved to disk.
        shm_info (dict[int, tuple]): A dict where the key is the argument index
            and the value is a tuple of SHM metadata (name, shape, dtype, columns)
            used for restoration.
        start_signal_name (str | None): Name of the shared memory segment into
            which the worker writes the timestamp of the actual start.
    Returns:
        Any: Result of running the function func.
    """

    final_args = list(args)
    # Use a context manager to automatically close SHM descriptors
    # This frees the system resources right after the function call.
    shm_wrappers: list[SharedDataFrame] = []
    try:
        # 1. Restore from Shared Memory
        # Restore SHM from metadata (name, shape, dtype, columns)
        if shm_info:
            for idx, meta in shm_info.items():
                wrapper = SharedDataFrame(
                    name=meta[0], shape=meta[1], dtype=meta[2], columns=meta[3]
                )
                final_args[idx] = wrapper.to_df()
                shm_wrappers.append(wrapper)
        # 2. Load from disk
        if disk_indices:
            for idx in disk_indices:
                path = final_args[idx]
                if isinstance(path, str) and path.endswith(".parquet"):
                    final_args[idx] = pd.read_parquet(path)
        # 3. Record the actual start of the task (right before calling func)
        if start_signal_name is not None:
            _mark_worker_start(start_signal_name)
        return func(*final_args, **kwargs)
    finally:
        # Important: we close only the worker's local references (descriptors).
        # The SHM data itself stays alive
        # until the main process removes it via .unlink()
        for w in shm_wrappers:
            try:
                w.close()
            except Exception:  # noqa: S110, BLE001
                pass
        final_args.clear()  # Help the GC free references faster


def _perform_cleanup(
    shm_refs: list[SharedDataFrame] | None,
    persistence_manager: DiskPersistenceManager | None,
) -> None:
    """Helper function for safe cleanup with protection against system errors."""
    if shm_refs:
        for ref in shm_refs:
            # First close the descriptor
            try:
                ref.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"SHM close error (expected during forced shutdown): {e}")

            # Then try to destroy the segment in the OS
            try:
                ref.unlink()
            except (FileNotFoundError, OSError):
                # Ignore if it was already removed or there is no access
                pass
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Non-critical SHM unlink failure: {e}")

    if persistence_manager:
        try:
            persistence_manager.cleanup()
        except (PermissionError, OSError) as e:
            logger.error(f"Cleanup failed due to file locking/permissions: {e}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected persistence cleanup error: {e}")


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

    # Shared Memory preparation logic for processes
    shm_refs: list[SharedDataFrame] = []
    persistence_manager = DiskPersistenceManager()
    # Start-signals of the actual task starts (only for processes)
    start_signals: list[_StartSignal] = []

    if mode == "processes" and (shared_args_indices or disk_args_indices):
        try:
            task_payloads = []  # List of tuples
            # (args, kwargs, actual_shm_idx, actual_disk_idx)
            target_shm_indices = shared_args_indices or []
            target_disk_indices = disk_args_indices or []

            # One SHM segment per DataFrame object: the same df passed to several
            # tasks is placed into shared memory only once (saves memory and
            # avoids name-collision risks).
            shm_cache: dict[int, SharedDataFrame] = {}
            # The same applies to the disk fallback: one Parquet file per object.
            disk_cache: dict[int, str] = {}
            # DataFrames that are SHM-compatible but cannot be placed into shared
            # memory (not enough free space) are routed through the
            # disk cache: remembered so that the failed attempt is not repeated
            # for every task that shares the same object.
            shm_failed: set[int] = set()

            for args, kwargs in zip(args_seq, kwargs_seq):
                new_args = list(args)
                curr_shm_info = {}  # Index -> (name, shape, dtype, columns)
                for idx in set(target_shm_indices) | set(target_disk_indices):
                    if idx < len(new_args) and isinstance(new_args[idx], pd.DataFrame):
                        df_obj = new_args[idx]
                        if (
                            idx in target_shm_indices
                            and id(df_obj) not in shm_failed
                            and SharedDataFrame.is_compatible(df_obj)
                        ):
                            shm_wrapper = shm_cache.get(id(df_obj))
                            if shm_wrapper is None:
                                try:
                                    shm_wrapper = SharedDataFrame(df_obj)
                                except OSError as shm_err:
                                    # Not enough free space in the SHM file
                                    # system: creating the segment
                                    # here would later kill the process with
                                    # SIGBUS (Bus error, exit 135) when the
                                    # data pages are touched. Route this frame
                                    # through the disk cache instead — save_df
                                    # itself moves the file to the system temp
                                    # directory when /dev/shm cannot hold it.
                                    shm_failed.add(id(df_obj))
                                    logger.warning(
                                        "Shared memory transfer unavailable "
                                        "(%s); routing the dataframe through "
                                        "the disk cache instead",
                                        shm_err,
                                    )
                                    path = disk_cache.get(id(df_obj))
                                    if path is None:
                                        path = persistence_manager.save_df(df_obj)
                                        disk_cache[id(df_obj)] = path
                                    new_args[idx] = path
                                    continue
                                shm_cache[id(df_obj)] = shm_wrapper
                                shm_refs.append(shm_wrapper)
                            # Save the metadata for passing to the worker
                            curr_shm_info[idx] = (
                                shm_wrapper.name,
                                shm_wrapper.shape,
                                shm_wrapper.dtype,
                                shm_wrapper.columns,
                            )
                            new_args[idx] = None  # We do not pass the object itself
                        else:
                            path = disk_cache.get(id(df_obj))
                            if path is None:
                                path = persistence_manager.save_df(df_obj)
                                disk_cache[id(df_obj)] = path
                            new_args[idx] = path

                curr_disk = [
                    i
                    for i in (set(target_shm_indices) | set(target_disk_indices))
                    if i < len(new_args)
                    and isinstance(new_args[i], str)
                    and new_args[i].endswith(".parquet")
                ]

                task_payloads.append(
                    (tuple(new_args), kwargs, curr_disk, curr_shm_info)
                )

            # Override the iterable for execution
            execution_tasks = task_payloads
        except Exception as e:  # noqa: BLE001
            # SHM/parquet preparation failed (for example, pyarrow refuses to
            # write "dirty" data, or shared memory is exhausted). Free the
            # already-created segments and temp files, then pass the dataframes
            # as ordinary pickled objects (the pre-SHM behaviour) instead of
            # crashing run_parallel. The original disk indices are
            # kept: in _worker_proxy a non-string argument at a disk index is
            # passed through untouched, so DataFrames still arrive intact
            # (only real parquet paths are read from disk).
            logger.warning(
                "Shared memory / disk preparation failed (%s); "
                "transferring dataframes through pickle instead",
                e,
            )
            _perform_cleanup(shm_refs, persistence_manager)
            shm_refs = []
            execution_tasks = [
                (tuple(a), kw, disk_args_indices or [], {})
                for a, kw in zip(args_seq, kwargs_seq)
            ]
    else:
        # For threads or plain processes without SHM/Disk
        execution_tasks = [
            (tuple(a), kw, disk_args_indices or [], {})
            for a, kw in zip(args_seq, kwargs_seq)
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
        n_tasks = len(future_to_idx)
        queue_budget = (
            math.ceil(n_tasks / max(worker_count, 1)) * task_timeout
            if task_timeout is not None
            else float("inf")
        )

        while future_to_idx:
            now = time.time()
            elapsed = now - start_time

            # Check: if the global timeout has expired — exit
            effective_timeout = timeout
            if effective_timeout is not None and elapsed >= effective_timeout:
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
                task_t = task_timeout if task_timeout is not None else float("inf")
                signal_obj = signal_by_future.get(fut)
                started = signal_obj.read() if signal_obj is not None else None
                submit_t = submit_time_by_future.get(fut, start_time)
                if started is not None and started >= 0:
                    deadline_by_future[fut] = min(
                        started + task_t, submit_t + task_t + queue_budget
                    )
                elif task_timeout is not None:
                    deadline_by_future[fut] = submit_t + task_t + queue_budget
                else:
                    deadline_by_future[fut] = float("inf")

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

            # Compute the minimum remaining timeout among the remaining tasks
            remaining_by_future = {}
            for fut, deadline in deadline_by_future.items():
                if fut in future_to_idx:
                    rem = max(0.1, deadline - now)
                    remaining_by_future[fut] = rem
            min_remaining = (
                min(remaining_by_future.values()) if remaining_by_future else None
            )

            # If the global timeout is smaller — use it
            if effective_timeout is not None:
                global_remaining = max(0.1, effective_timeout - elapsed)
                if min_remaining is not None:
                    min_remaining = min(min_remaining, global_remaining)
                else:
                    min_remaining = global_remaining  # pragma: no cover

            # While a task has not started yet, cap the wait at POLL_INTERVAL:
            # the start signal is written asynchronously by the worker and is
            # not set on the first loop iterations, so the task's real deadline
            # (start + task_timeout) may appear at any moment and be much
            # closer than its queue deadline (submit + task_timeout +
            # queue_budget). Without the cap as_completed sleeps up to the
            # queue deadline and the timeout of an already-started task is
            # noticed with a delay of the whole queue budget.
            if task_timeout is not None:
                has_unstarted = False
                for fut in future_to_idx:
                    sig = signal_by_future.get(fut)
                    if sig is not None and sig.read() < 0:
                        has_unstarted = True
                        break
                if has_unstarted and min_remaining is not None:
                    min_remaining = min(min_remaining, POLL_INTERVAL)

            timeout_for_ac = (
                min_remaining
                if min_remaining is not None and min_remaining < float("inf")
                else None
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

            _perform_cleanup(shm_refs, persistence_manager)
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
                # Final SHM/Disk cleanup (only in processes mode)
                _perform_cleanup(shm_refs, persistence_manager)
            else:
                # For ThreadPoolExecutor or if the mode is not "processes"
                pool.shutdown(wait=True, cancel_futures=True)
        # Cleanup of the actual-start signals (created only in processes)
        for sig in start_signals:
            sig.close()
            sig.unlink()
    return results
