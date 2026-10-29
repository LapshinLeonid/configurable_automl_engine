"""Payload transport of the processes mode of ``run_parallel``.

Heavy ``pandas.DataFrame`` arguments are not pickled into every task:
    1. Shared memory (``SharedDataFrame``): one segment per frame object,
       reused by every task that passes the same object. The worker restores
       the frame positionally, with a ``RangeIndex(0, n)``.
    2. Disk cache (``DiskPersistenceManager``): a Parquet temp file per frame
       object, for frames that shared memory cannot hold (an incompatible
       dtype mix, or not enough free space).
    3. Pickle: if neither can be prepared, the frames are passed as ordinary
       pickled arguments (with a warning).

``PayloadTransport`` owns every segment and temp file it creates and frees
them in ``close()``; used as a context manager it frees them on every exit
path, ``KeyboardInterrupt`` included.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import pandas as pd

from configurable_automl_engine.common.shared_frame import SharedDataFrame

__all__ = ["DiskPersistenceManager", "PayloadTransport", "TaskPayload"]

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class TaskPayload:
    """What one task sends to the worker (``_worker_proxy`` arguments).

    Attributes:
        args (tuple[Any, ...]): The positional arguments; a frame sent through
            shared memory is replaced with ``None``, a frame sent through the
            disk cache with the path of its Parquet file.
        kwargs (Mapping[str, Any]): The keyword arguments.
        disk_indices (list[int]): The argument indices the worker reads from
            Parquet (a non-path argument at such an index is passed through).
        shm_info (dict[int, tuple[Any, ...]]): Argument index → the shared
            memory metadata ``(name, shape, dtype, columns)``.
    """

    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    disk_indices: list[int]
    shm_info: dict[int, tuple[Any, ...]]


class PayloadTransport:
    """Prepares the task payloads and owns the segments and files it creates.

    Args:
        shared_indices (list[int] | None): The argument indices of the frames
            to send through shared memory.
        disk_indices (list[int] | None): The argument indices of the frames
            to send through the disk cache.
        persistence (DiskPersistenceManager | None): The disk cache; a new
            one by default.
    """

    def __init__(
        self,
        shared_indices: list[int] | None = None,
        disk_indices: list[int] | None = None,
        persistence: DiskPersistenceManager | None = None,
    ) -> None:
        self._shared = set(shared_indices or [])
        self._disk = set(disk_indices or [])
        self._disk_list = list(disk_indices or [])
        self._persistence = persistence or DiskPersistenceManager()
        self._segments: list[SharedDataFrame] = []
        # One segment / Parquet file per frame object, keyed by id(frame);
        # the frames themselves are kept alive by the caller's arguments.
        self._segment_by_id: dict[int, SharedDataFrame] = {}
        self._path_by_id: dict[int, str] = {}
        # Frames that cannot be placed into shared memory (not enough free
        # space): the failed attempt is not repeated for every task.
        self._shm_failed: set[int] = set()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def segment_names(self) -> list[str]:
        """The names of the shared memory segments owned by the transport."""
        return [s.name for s in self._segments if s.name is not None]

    @property
    def temp_files(self) -> list[str]:
        """The Parquet temp files owned by the transport."""
        return list(self._persistence.created_files)

    def prepare(
        self,
        args_seq: Sequence[Sequence[Any]],
        kwargs_seq: Sequence[Mapping[str, Any]],
    ) -> list[TaskPayload]:
        """Build the payload of every task.

        If shared memory and the disk cache cannot be prepared (for example,
        pyarrow refuses to write "dirty" data), the resources created so far
        are freed and the frames are passed through pickle with a warning.

        Args:
            args_seq (Sequence[Sequence[Any]]): The positional arguments of
                every task.
            kwargs_seq (Sequence[Mapping[str, Any]]): The keyword arguments of
                every task.

        Returns:
            list[TaskPayload]: One payload per task, in the task order.
        """
        if not (self._shared or self._disk):
            return self._plain(args_seq, kwargs_seq)
        try:
            return [
                self._encode_task(args, kwargs)
                for args, kwargs in zip(args_seq, kwargs_seq)
            ]
        except Exception as e:  # noqa: BLE001
            # The original disk indices are kept: in _worker_proxy a
            # non-string argument at a disk index is passed through untouched,
            # so the frames still arrive intact.
            logger.warning(
                "Shared memory / disk preparation failed (%s); "
                "transferring dataframes through pickle instead",
                e,
            )
            self.close()
            return self._plain(args_seq, kwargs_seq)

    def close(self) -> None:
        """Free every segment and temp file created so far (idempotent)."""
        _perform_cleanup(self._segments, self._persistence)
        self._segments = []
        self._segment_by_id = {}
        self._path_by_id = {}
        # A file that could not be removed stays registered (the warning is
        # already logged) so a later close() retries it.
        self._persistence.created_files = [
            p for p in self._persistence.created_files if os.path.exists(p)
        ]

    def _plain(
        self,
        args_seq: Sequence[Sequence[Any]],
        kwargs_seq: Sequence[Mapping[str, Any]],
    ) -> list[TaskPayload]:
        return [
            TaskPayload(tuple(a), kw, list(self._disk_list), {})
            for a, kw in zip(args_seq, kwargs_seq)
        ]

    def _encode_task(
        self, args: Sequence[Any], kwargs: Mapping[str, Any]
    ) -> TaskPayload:
        new_args = list(args)
        shm_info: dict[int, tuple[Any, ...]] = {}
        for idx in self._shared | self._disk:
            if idx < len(new_args) and isinstance(new_args[idx], pd.DataFrame):
                new_args[idx], meta = self._encode_frame(idx, new_args[idx])
                if meta is not None:
                    shm_info[idx] = meta
        disk = [
            i
            for i in self._shared | self._disk
            if i < len(new_args)
            and isinstance(new_args[i], str)
            and new_args[i].endswith(".parquet")
        ]
        return TaskPayload(tuple(new_args), kwargs, disk, shm_info)

    def _encode_frame(
        self, idx: int, df: pd.DataFrame
    ) -> tuple[Any, tuple[Any, ...] | None]:
        """Return the argument to send and the shared memory metadata."""
        if (
            idx in self._shared
            and id(df) not in self._shm_failed
            and SharedDataFrame.is_compatible(df)
        ):
            segment = self._segment(df)
            if segment is not None:
                meta = (segment.name, segment.shape, segment.dtype, segment.columns)
                return None, meta  # the frame itself is not sent
        return self._parquet_path(df), None

    def _segment(self, df: pd.DataFrame) -> SharedDataFrame | None:
        segment = self._segment_by_id.get(id(df))
        if segment is not None:
            return segment
        try:
            segment = SharedDataFrame(df)
        except OSError as shm_err:
            # Not enough free space in the SHM file system: creating the
            # segment would later kill the process with SIGBUS (Bus error,
            # exit 135) when the data pages are touched. Route this frame
            # through the disk cache instead — save_df itself moves the file
            # to the system temp directory when /dev/shm cannot hold it.
            self._shm_failed.add(id(df))
            logger.warning(
                "Shared memory transfer unavailable (%s); routing the "
                "dataframe through the disk cache instead",
                shm_err,
            )
            return None
        self._segment_by_id[id(df)] = segment
        self._segments.append(segment)
        return segment

    def _parquet_path(self, df: pd.DataFrame) -> str:
        path = self._path_by_id.get(id(df))
        if path is None:
            path = self._persistence.save_df(df)
            self._path_by_id[id(df)] = path
        return path
