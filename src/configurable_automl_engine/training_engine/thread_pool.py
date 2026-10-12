"""Module for parallel execution of tasks with memory management.
Provides tools for running functions in parallel (ThreadPool/ProcessPool)
with optimized transfer of heavy pandas.DataFrame objects through
Shared Memory and Disk Persistence (disk cache) mechanisms.
Main components:
    - SharedDataFrame: A container for placing data in the OS shared memory.
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

logger = logging.getLogger(__name__)


class SharedDataFrame:
    """Wrapper for placing a DataFrame in shared memory.

    Allows passing large data volumes between processes without the overhead
    of serialization (pickle), using the POSIX Shared Memory mechanism.

    Attributes:
        name (str | None): Unique name of the shared memory segment in the OS.
        shm (shared_memory.SharedMemory): Object managing the memory segment.
        shared_array (np.ndarray): NumPy view of the data pointing to the SHM buffer
        (may be marked read-only after restoration).
        shape (tuple): Dimensionality of the original data array.
        dtype (np.dtype): Data type of the array elements.
        columns (list[str]): Column names for restoring the DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        name: str | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | Any = None,
        columns: list[str] | None = None,
    ) -> None:
        """Initialize a shared memory object for a DataFrame.
        Initialization logic:
        1. Create: If df is passed, a new Shared Memory segment is created,
           with an OS-generated unique name (name=None), into which the data
           is copied. Multiple SharedDataFrame objects can share one segment
           (see run_parallel: one segment per DataFrame object).
        2. Attach: If df is not passed, the object attaches to an existing
           segment by name using metadata (shape, dtype, columns).
        3. Metadata: The column name list is kept for the subsequent
           restoration of the DataFrame structure.
        Args:
            df (pd.DataFrame | None): Source DataFrame to place in SHM.
            name (str | None): Name of an existing memory segment.
            shape (tuple | None): Dimensionality of the data array.
            dtype (np.dtype | Any): Data type of the array elements.
            columns (list[str] | None): Column name list.
        Returns:
            None
        """

        self.name: str | None = None
        self._owner = df is not None
        self.columns: list[str] | None = None

        if df is not None:
            data = df.to_numpy()
            # name=None makes the OS generate a unique segment name, so
            # concurrent creations never collide (no FileExistsError).
            self.shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            self.name = self.shm.name
            self.shared_array = np.ndarray(
                data.shape, dtype=data.dtype, buffer=self.shm.buf
            )
            self.shared_array[:] = data[:]
            self.shape = data.shape
            self.dtype = data.dtype
            self.columns = df.columns.tolist()
        else:
            self.name = name
            self.shm = shared_memory.SharedMemory(name=name)
            actual_shape = shape if shape is not None else ()
            self.shared_array = np.ndarray(
                actual_shape, dtype=dtype, buffer=self.shm.buf
            )
            self.shape = self.shared_array.shape
            self.columns = columns

    @staticmethod
    def is_shared_array(X: Any) -> bool:
        """
        Check whether an object is a numpy array backed by SharedMemory.
        This is critical for avoiding repeated copying in ModelTrainer.
        """
        if not isinstance(X, np.ndarray):
            return False
        # Check whether the array buffer points to a shared memory segment
        # (in Python 3.8+ SharedMemory.buf returns a memoryview)
        return hasattr(X, "base") and isinstance(X.base, memoryview)

    @staticmethod
    def get_data_info(X: Any) -> tuple[int, list[str] | list[int]]:
        """
        Extract the number of features and their names (or indices) without copying.
        """
        if isinstance(X, pd.DataFrame):
            return X.shape[1], X.columns.tolist()
        elif isinstance(X, SharedDataFrame):
            shape = getattr(X, "shape", X.shared_array.shape)
            n_cols = shape[1] if len(shape) > 1 else 1
            cols = (
                X.columns if X.columns is not None else [str(i) for i in range(n_cols)]
            )
            return n_cols, cols
        elif isinstance(X, np.ndarray):
            n_cols = X.shape[1] if X.ndim > 1 else 1
            return n_cols, list(range(n_cols))
        return 0, []

    @staticmethod
    def is_compatible(df: Any) -> bool:
        """Check whether a DF can be placed into SHM
        (only simple types and a RangeIndex)."""
        """Check whether a DF can be placed into SHM 
        (an already shared array, or a DF with simple types and a RangeIndex)."""
        if SharedDataFrame.is_shared_array(df):
            return True
        if not isinstance(df, pd.DataFrame):
            return False

        # Check data types (whitelist: int, uint, float, bool)
        allowed_kinds = {"i", "u", "f", "b"}
        if not all(dt.kind in allowed_kinds for dt in df.dtypes):
            return False

        # Check the index: the current SHM implementation does not support complex indices.
        # If the index is not a RangeIndex, the object is considered incompatible with SHM
        # and is automatically redirected to DiskPersistenceManager.
        return isinstance(df.index, pd.RangeIndex)

    def to_df(self) -> pd.DataFrame:
        """Restore a pandas.DataFrame from shared memory.
        Restoration logic:
        1. Proxying: an np.ndarray is created with the writeable=False flag
        forcibly set to prevent
        unnecessary copying in pandas.
        2. Reconstruction: a new DataFrame is built from the array and the
           stored column name list.
        3. Isolation: the resulting DataFrame is an independent object in the
           memory of the current process.
        Returns:
            pd.DataFrame: The restored dataset.
        """

        # Set the writeable=False flag. This is critical: pandas often makes a
        # hidden copy if it "fears" that someone might modify the shared buffer.
        self.shared_array.setflags(write=False)

        # copy=False in the constructor along with a homogeneous numpy array
        # guarantees a DataFrame is created without allocating new memory for the data.
        return pd.DataFrame(self.shared_array, columns=self.columns, copy=False)

    def close(self) -> None:
        """Close access to the shared memory segment.
        Closing logic:
        1. Handle: closes the SHM access descriptor in the current process.
        2. Preservation: the data itself in the OS is not destroyed, allowing other
           processes to keep working with the segment.
        Returns:
            None
        """

        if hasattr(self, "shm"):
            self.shm.close()

    def unlink(self) -> None:
        """Destroy the shared memory segment in the operating system.
        Deletion logic:
        1. Ownership: the operation is only performed by the creating process (_owner=True).
        2. Release: marks the segment for deletion; the memory will be fully
           freed by the OS once all processes close their references to it.
        Returns:
            None
        """

        if self._owner and hasattr(self, "shm"):
            try:
                self.shm.unlink()
            except (FileNotFoundError, OSError):
                pass  # Already removed

    def get_view(self, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Return a view of the data.
        If columns are passed, returns a view restricted to those columns.
        """
        if columns is None:
            return self.to_df()

        # Important: use .loc to create a slice-view, not a copy
        return self.to_df().loc[:, columns]


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
        self.tmp_dir = "/dev/shm" if use_shm and os.path.exists("/dev/shm") else None
        self.created_files: list[str] = []

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

        fd, path = tempfile.mkstemp(suffix=".parquet", dir=self.tmp_dir)
        os.close(fd)
        df.to_parquet(path, engine="pyarrow")
        self.created_files.append(path)
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

    ``_StartSignal`` is a shared memory segment (8 bytes, float64) into which
    the worker writes ``time.time()`` right before calling the target
    function, while the parent reads it in the wait loop. It is initialized
    with ``-1.0`` (meaning "not started yet"); real timestamps are always
    positive. It is removed by the owner (the parent) via ``unlink``.
    """

    def __init__(self) -> None:
        self._shm = shared_memory.SharedMemory(create=True, size=8)
        self._owner = True
        self.name = self._shm.name
        self._buf = np.ndarray((1,), dtype=np.float64, buffer=self._shm.buf)
        self._buf[0] = -1.0

    def mark_start(self, value: float) -> None:
        """Record the timestamp of the task's actual start."""
        self._buf[0] = value

    def read(self) -> float:
        """Return the start timestamp (``-1.0`` — the task has not started yet)."""
        return float(self._buf[0])

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
    """Record the actual start timestamp of the task in the shared-memory signal.

    Called by the worker right before executing the target function.

    Args:
        start_signal_name (str): Name of the shared memory segment of the signal.
    """
    try:
        shm = shared_memory.SharedMemory(name=start_signal_name)
        try:
            buf = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
            buf[0] = time.time()
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
            7. Timeout semantics: in "processes" mode the timeout is hard: hung
               workers are forcibly terminated via SIGTERM/SIGKILL (watchdog).
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
                (for the whole batch drain). Applies only in "processes" mode
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
    shm_refs = []
    persistence_manager = DiskPersistenceManager()
    # Start-signals of the actual task starts (only for processes)
    start_signals: list[_StartSignal] = []

    if mode == "processes" and (shared_args_indices or disk_args_indices):
        task_payloads = []  # List of tuples
        # (args, kwargs, actual_shm_idx, actual_disk_idx)
        target_shm_indices = shared_args_indices or []
        target_disk_indices = disk_args_indices or []

        # One SHM segment per DataFrame object: the same df passed to several
        # tasks is placed into shared memory only once (saves memory and
        # avoids name-collision risks).
        shm_cache: dict[int, SharedDataFrame] = {}

        for args, kwargs in zip(args_seq, kwargs_seq):
            new_args = list(args)
            curr_shm_info = {}  # Index -> (name, shape, dtype, columns)
            for idx in set(target_shm_indices) | set(target_disk_indices):
                if idx < len(new_args) and isinstance(new_args[idx], pd.DataFrame):
                    if idx in target_shm_indices and SharedDataFrame.is_compatible(
                        new_args[idx]
                    ):
                        df_obj = new_args[idx]
                        shm_wrapper = shm_cache.get(id(df_obj))
                        if shm_wrapper is None:
                            shm_wrapper = SharedDataFrame(df_obj)
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
                        path = persistence_manager.save_df(new_args[idx])
                        new_args[idx] = path

            curr_disk = [
                i
                for i in (set(target_shm_indices) | set(target_disk_indices))
                if i < len(new_args)
                and isinstance(new_args[i], str)
                and new_args[i].endswith(".parquet")
            ]

            task_payloads.append((tuple(new_args), kwargs, curr_disk, curr_shm_info))

        # Override the iterable for execution
        execution_tasks = task_payloads
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
        try:
            executor_cls = ProcessPoolExecutor
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Could not initialize ProcessPoolExecutor:"
                f" {e}. Falling back to threads."
            )
            executor_cls = ThreadPoolExecutor

    start_time = time.time()

    try:
        pool = executor_cls(max_workers)

        future_to_idx = {}
        submit_time_by_future: dict[Future[Any], float] = {}
        signal_by_future: dict[Future[Any], _StartSignal | None] = {}
        for i, (a, kw, d_idx, s_idx) in enumerate(execution_tasks):
            if mode == "processes":
                # For each task create a side-channel of the actual start: the
                # worker writes a timestamp into it, the parent binds the
                # deadline to it. It is impossible to assume the task has
                # started based on future.running() — ProcessPoolExecutor
                # marks a task as running already when it enters the pool's
                # internal queue.
                sig = _StartSignal()
                start_signals.append(sig)
                fut = pool.submit(_worker_proxy, func, a, kw, d_idx, s_idx, sig.name)
                signal_by_future[fut] = sig
            else:
                fut = pool.submit(func, *a, **kw)
                signal_by_future[fut] = None
            future_to_idx[fut] = i
            submit_time_by_future[fut] = time.time()

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

            # Check: if the per-task timeout (task_timeout) has expired — exit
            for fut, deadline in deadline_by_future.items():
                if fut in future_to_idx and now >= deadline:
                    idx = future_to_idx.pop(fut)
                    logger.error(f"Task {idx} timed out")
                    results[idx] = None
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
