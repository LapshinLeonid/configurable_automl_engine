"""The worker side of the processes mode of ``run_parallel``.

- ``_StartSignal``: a 16-byte shared memory segment into which the worker
  writes the timestamp of the task's real start and its own PID.
- ``_worker_proxy``: the function the pool actually runs; it restores the
  frames sent through shared memory / the disk cache, records the start
  and calls the target function.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.common.shared_frame import SharedDataFrame

logger = logging.getLogger(__name__)

# Layout of the _StartSignal shared memory segment: a float64 start timestamp
# followed by an int64 worker PID. Both fields are written by the worker in a
# single assignment (a structured-record write), so the parent cannot observe
# a started timestamp with a still-empty PID. The PID is stored as int64, not
# float64, to avoid precision loss for large PIDs.
_START_SIGNAL_DTYPE = np.dtype([("start", np.float64), ("pid", np.int64)])


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
    function, while the parent reads them in the wait loop. The PID tells
    the parent which worker process ran the task (a crashed worker is matched
    to its task by it). It is initialized with ``-1.0`` / ``0`` (meaning "not
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
        """Close the parent's handle (the view is released first)."""
        # Drop the view over the segment first: a buffer with a live export
        # cannot be closed. A detached "not started" record replaces it.
        self._signal = np.zeros(1, dtype=_START_SIGNAL_DTYPE)
        self._signal[0] = (-1.0, 0)
        try:
            self._shm.close()
        except Exception:  # noqa: S110, BLE001
            pass

    def unlink(self) -> None:
        """Remove the segment from the OS (only once, by the owner)."""
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
            del signal
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
