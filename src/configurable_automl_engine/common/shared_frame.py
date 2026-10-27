"""Shared-memory transport of a ``pandas.DataFrame`` between processes.

:class:`SharedDataFrame` places the values of a DataFrame in an OS shared
memory segment (POSIX Shared Memory) so that worker processes can read them
without pickling. The helpers decide whether a frame can be shared at all
(a single NumPy dtype, integers exact in float64) and whether the shared
memory file system has room for the segment.

The process-pool machinery that uses it (``run_parallel``) lives in
``configurable_automl_engine.training_engine.thread_pool``.
"""

import os
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["SharedDataFrame"]

# Free-space safety margin for POSIX shared memory. A segment is
# created with ftruncate and pages are only touched later (by the np.ndarray
# copy at creation or by a worker's reads): on a tmpfs file system (Docker,
# Kubernetes and CI default /dev/shm to a 64MB ramdisk; desktop Linux usually
# gives it half of RAM) an ftruncate that exceeds the available space succeeds
# and the first touch of a page past the end of the backing store kills the
# process with SIGBUS (Bus error, exit code 135) — a signal, not a Python
# exception, so it cannot be caught or handled by the threads-mode fallback.
# The only reliable protection is a free-space check BEFORE the segment is
# created (os.statvfs on the file system backing the shared memory) with a
# conservative margin for the already-used portion of /dev/shm and for page
# granularity. When the check fails the DataFrame is routed through the disk
# cache (parquet) or pickle instead.
_SHM_FREE_SPACE_MARGIN = 0.3  # 30% of a segment's size must remain free


def _numpy_result_dtype(df: pd.DataFrame) -> np.dtype | None:
    """Return the dtype of an array that ``df.to_numpy()`` would produce.

    Computed WITHOUT materializing the array itself: for a frame with mixed
    column types ``to_numpy()`` copies the whole data into a new array, and
    ``SharedDataFrame.__init__`` copies it a second time right after, causing
    a transient doubling of memory in the main process on every HPO phase.

    Only plain numpy column dtypes are accepted: extension dtypes (Int64,
    Float64, boolean, string, category, ...) are rejected up front, because
    pandas may promote them to a different dtype than their plain numpy
    counterparts: e.g. ``Int64 + float64`` → object, while
    ``int64 + float64`` → float64. For plain numpy columns of the numeric
    kinds i/u/f/b pandas promotes to a common dtype exactly like numpy's
    ``np.result_type`` — except for bool: a bool column mixed with any
    non-bool column yields object (an array of raw pointers to
    parent-process Python objects), while ``np.result_type(bool, int64)``
    returns int64. The bool case is therefore handled explicitly, without
    relying on the internal pandas API ``pandas.core.dtypes``. Other plain
    numpy kinds (datetime64/timedelta64, M/m) are out of scope: mixing them
    with numeric columns, ``np.result_type`` raises or diverges from
    ``df.to_numpy()`` (which yields object) — the caller's rule 1 already
    rejects such frames before this helper is consulted.

    Args:
        df: The candidate DataFrame.

    Returns:
        The common plain numpy dtype of ``df.to_numpy()``, or None when the
        frame is not eligible (extension dtype present, or some dtypes).
    """
    try:
        dtypes = list(df.dtypes)
    except Exception:  # noqa: BLE001
        return None
    if not all(isinstance(dt, np.dtype) for dt in dtypes):
        return None
    if not dtypes:
        return np.dtype("float64")

    bool_mask = [dt.kind == "b" for dt in dtypes]
    if all(bool_mask):
        return np.dtype("bool")
    if any(bool_mask):
        # bool mixed with a non-bool column: df.to_numpy() gives an object
        # array, not a numeric one (unlike np.result_type) — the caller
        # rejects object arrays.
        return np.dtype("O")
    try:
        return np.result_type(*dtypes)
    except Exception:  # noqa: BLE001
        return None


def _ints_fit_float64(df: pd.DataFrame) -> bool:
    """Whether every int64/uint64 column of ``df`` fits into float64 exactly.

    A value can be converted to float64 without precision loss iff
    ``abs(x) <= 2**53``: ``2**53`` itself is representable exactly, ``2**53+1``
    is not. int32/uint32 → float64 is always lossless, and narrower integers
    promote to float32/float64 losslessly as well, so only 8-byte integer
    columns need a value check, and only when the frame actually promotes to
    float (the caller checks that the result dtype is float). Only
    int64/uint64 columns are inspected; non-8-byte and non-integer columns
    (including float columns that may carry NaN) are skipped, so no NaN check
    is needed.

    Args:
        df: The candidate DataFrame (plain numpy column types only).

    Returns:
        True when all int64/uint64 columns contain values representable in
        float64 without loss. Empty or absent such columns return True.
    """
    columns = df.select_dtypes(include=["int64", "uint64"])
    # Iterate positionally over the 8-byte integer columns only (the
    # optimized subset): with duplicate column names df[name] would return a
    # DataFrame and min()/max() would then produce Series (a ValueError in
    # the comparison below). The compatibility check must never crash — it
    # only decides SHM vs parquet.
    for i, dt in enumerate(columns.dtypes):
        col = columns.iloc[:, i]
        if len(col) == 0:
            continue
        if dt.kind == "u":
            # uint64 has no negative values: only the upper boundary matters.
            if int(col.max()) > 2**53:
                return False
        else:
            # Use Python int / separate min-max bounds: abs of
            # np.iinfo(np.int64).min overflows int64.
            low, high = col.min(), col.max()
            if high > 2**53 or low < -(2**53):
                return False
    return True


def _shm_backing_path() -> str | None:
    """Return the path of the file system backing POSIX shared memory.

    POSIX shared memory segments are anonymous tmpfs files mounted at
    /dev/shm on Linux (the shared_memory module's ``_make_filename`` only
    exposes the OS-generated random name; the directory comes from the
    shm_open(3) implementation). macOS lacks the directory-level API
    entirely, and Windows uses a different mechanism. For the free-space
    check the closest observable approximation is /dev/shm
    when it exists, otherwise None (no check possible).

    Returns:
        str | None: The backing directory (/dev/shm), or None when it
        cannot be determined.
    """
    # A mock /dev/shm in tests is a regular directory: still a valid target
    # for the check.
    if os.path.exists("/dev/shm"):
        return "/dev/shm"
    return None


def _shm_has_free_space(size: int) -> bool:
    """Whether an SHM segment of ``size`` bytes fits the free space.

    Checks ``os.statvfs`` on the file system backing the shared memory:
    a segment larger than the free space on a tmpfs would be
    created successfully by ftruncate but kill the process with SIGBUS on
    the first touch of a page past the end of the backing store (Bus error,
    exit code 135 — a signal, not a catchable exception). A conservative
    30% margin is kept free so that this segment never exhausts the tmpfs
    for the segments still to be created.

    Known limitation (TOCTOU): the statvfs check is not atomic with the
    subsequent segment creation — another process may consume the remaining
    free space between the check and the ftruncate, in which case SIGBUS is
    still theoretically possible. This is inherent to the POSIX shared
    memory API (there is no way to reserve space atomically); the 30% margin
    reduces the window but does not eliminate it. On a typical single-node
    run this is a best-effort guard against the 64MB /dev/shm default.

    When the backing path cannot be determined (non-Linux POSIX systems),
    no check is possible and True is returned: the existing behaviour (an
    attempt that may or may not fail with ENOMEM) is preserved there.

    Args:
        size: The requested segment size in bytes.

    Returns:
        bool: True when the segment safely fits (or the check is
        impossible), False when it must not be created.
    """
    path = _shm_backing_path()
    if path is None:
        return True
    try:
        st = os.statvfs(path)
    except (OSError, ValueError):
        # A broken/not-mounted /dev/shm (e.g. a container without the ramdisk
        # mounted) must not crash the caller: report "no room" so the data
        # is routed through the disk cache, which can then fall back to
        # pickle with a warning.
        return False
    free = st.f_bavail * st.f_frsize
    return free >= size * (1.0 + _SHM_FREE_SPACE_MARGIN)


class SharedDataFrame:
    """Wrapper for placing a DataFrame in shared memory.

    Allows passing large data volumes between processes without the overhead
    of serialization (pickle), using the POSIX Shared Memory mechanism.

    Index contract: the index is NOT transferred. The data is
    passed positionally, and the restored frame always gets a RangeIndex(0,
    n) — even when the original index was, e.g., RangeIndex(100, 100+n).
    Worker code (tuner, preprocessing, oversampling, feature_selection,
    validation) must not align X and y by index.

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
        # NOTE: the index is NOT transferred through shared
        # memory. to_df() always restores a RangeIndex(0, n): the data is
        # passed positionally, and worker code (tuner, preprocessing,
        # oversampling, feature_selection, validation) must not align X and y
        # by index.
        self.columns: list[str] | None = None

        if df is not None:
            data = df.to_numpy()
            # Free-space pre-check: on a tmpfs /dev/shm (Docker/
            # Kubernetes default 64MB) ftruncate succeeds even beyond the
            # available space and the first touch of the data then kills the
            # process with SIGBUS (Bus error, exit 135) — a signal that
            # cannot be caught or handled by the threads-mode fallback. The
            # check must run before the segment is created; on failure the
            # caller (run_parallel) routes the frame through the disk cache.
            if not _shm_has_free_space(data.nbytes):
                raise OSError(
                    "not enough free space in shared memory "
                    f"(need {data.nbytes} bytes, margin "
                    f"{_SHM_FREE_SPACE_MARGIN:.0%}); "
                    "use the disk cache or pickle instead"
                )
            # name=None makes the OS generate a unique segment name, so
            # concurrent creations never collide (no FileExistsError).
            self.shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            self.name = self.shm.name
            try:
                self.shared_array = np.ndarray(
                    data.shape, dtype=data.dtype, buffer=self.shm.buf
                )
                self.shared_array[:] = data[:]
            except Exception:
                # Creation failed partway: free the OS segment immediately so
                # it does not leak (a failure inside the
                # preparation must not leak SHM segments). self.shm exists
                # here: the failure is in the ndarray/copy step, which happens
                # strictly after the segment was created.
                if self.shm is not None:
                    self.shm.close()
                    try:
                        self.shm.unlink()
                    except (FileNotFoundError, OSError):
                        pass
                raise
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
        """Check whether a DF can be placed into SHM.

        A DataFrame is compatible when its data can be copied into shared
        memory as a single homogeneous numpy array of plain numpy types
        (int/uint/float/bool) without silently changing the values.

        The index is NOT transferred: to_df() restores RangeIndex(0, n).
        Data is passed positionally — worker code must not align X and y by
        index.

        The check never calls df.to_numpy() (it would copy the whole data
        just to learn the dtype): the result dtype is computed from the
        column dtypes instead. Consequently a frame may be reported
        compatible even when an actual to_numpy() call would raise (an
        exotic frame whose materialization fails): the exception then
        surfaces inside SharedDataFrame.__init__ and run_parallel's
        preparation fallback routes the frame through pickle.
        """
        if SharedDataFrame.is_shared_array(df):
            return True
        if not isinstance(df, pd.DataFrame):
            return False

        # Rule 1: every column must have a plain numpy dtype with a simple
        # numeric/bool kind (i/u/f/b). This rejects extension dtypes (Int64,
        # Float64, boolean, string, category, ...) even when df.to_numpy()
        # would promote them to a numeric array: extension dtypes carry
        # semantics (nullable values, categorical encoding) that is silently
        # lost in the SHM round-trip, and e.g. a numeric category promotes to
        # float64, which changes the astype(str) encoding ('10' → '10.0') and
        # can collapse big int ids into indistinguishable float64 values,
        # while the earlier parquet path preserved the exact int64.
        allowed_kinds = {"i", "u", "f", "b"}
        if not all(
            isinstance(dt, np.dtype) and dt.kind in allowed_kinds for dt in df.dtypes
        ):
            return False

        # Rule 2: compute the dtype df.to_numpy() would produce WITHOUT
        # materializing the array itself: for a mixed-type frame to_numpy()
        # makes a full in-memory copy, and SharedDataFrame.__init__ makes a
        # second one right after — a transient doubling of memory in the main
        # process on every phase. pandas promotes mixed plain
        # numpy columns to a common dtype exactly like numpy's result_type,
        # except that bool mixed with anything non-bool yields object (an
        # array of raw pointers to parent-process Python objects; copying
        # such pointers into shared memory crashes workers under
        # forkserver/spawn), so _numpy_result_dtype reports the
        # object dtype for that case and the frame is rejected below.
        array_dtype = _numpy_result_dtype(df)
        if array_dtype is None:
            return False
        if array_dtype.kind not in allowed_kinds:
            return False

        # Rule 3: reject precision-losing promotions: an int64/uint64 column
        # mixed with a float column promotes to float64, making integer
        # values above 2**53 indistinguishable (9007199254740993 and
        # 9007199254741001 both become 9007199254741000.0). Whether the loss
        # actually occurs depends on the VALUES, not the dtype: int64 is the
        # default dtype of integer columns in pandas, and a typical numeric
        # dataset (int + float features) only needs to go through parquet
        # when an 8-byte integer column really contains values beyond 2**53
        # (in practice almost only ID columns).
        if array_dtype.kind == "f" and not _ints_fit_float64(df):
            return False

        # The index itself is not transferred (see to_df): a non-RangeIndex
        # is rejected because restoring the frame with RangeIndex(0, n)
        # would silently change the label alignment of such an index.
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

        Index contract: the index is NOT transferred. The data
        is passed positionally, and the restored frame always gets a
        RangeIndex(0, n) — even when the original index was, e.g.,
        RangeIndex(100, 100+n). Worker code (tuner, preprocessing,
        oversampling, feature_selection, validation) must not align X and y
        by index.
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
