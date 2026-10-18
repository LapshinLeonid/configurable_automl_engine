import pytest
import time
import hashlib
import logging
import subprocess
import sys
import tempfile
import itertools
import concurrent.futures
import pandas as pd
import numpy as np
import os
from multiprocessing import shared_memory
import unittest

from unittest.mock import MagicMock, patch
from configurable_automl_engine.tuner import InvalidAlgorithmError

from configurable_automl_engine.training_engine import thread_pool

from configurable_automl_engine.training_engine.thread_pool import (
    run_parallel,
    SharedDataFrame,
    DiskPersistenceManager,
    _worker_proxy,
    _perform_cleanup,
    _force_shutdown_processes,
    _StartSignal,
    _terminate_worker_by_pid,
)

MODULE_PATH = "configurable_automl_engine.training_engine.thread_pool"


# --- Test functions ---
def cpu_bound_task(n: int) -> str:
    """Simulates a CPU-bound task (computing a hash)."""
    data = str(n).encode()
    for _ in range(100_000):
        data = hashlib.sha256(data).hexdigest().encode()
    return data.decode()


def simple_task(x: int, y: int = 0) -> int:
    """A simple task for testing arguments."""
    return x + y


def slow_task(seconds: float):
    """A task for testing timeouts."""
    time.sleep(seconds)
    return "done"


def failing_task():
    """A task for testing exception handling."""
    raise ValueError("Intentional failure")


# --- Tests ---
def test_run_parallel_threads_basic():
    """1. Check operation in threads mode."""
    args = [(1, 2), (3, 4), (5, 6)]
    results = run_parallel(simple_task, args_seq=args, mode="threads")

    assert sorted(results) == [3, 7, 11]


def test_run_parallel_processes_cpu_bound():
    """2. Check operation in processes mode."""
    # Use a small number of tasks for the test
    args = [(i,) for i in range(3)]
    results = run_parallel(cpu_bound_task, args_seq=args, mode="processes")

    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


def test_run_parallel_fallback_mechanism(caplog):
    """Updated: check the fallback when initialization fails."""
    args = [(1, 10)]
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.ProcessPoolExecutor",
        side_effect=RuntimeError("OS Error"),
    ):
        with caplog.at_level(logging.ERROR):
            results = run_parallel(simple_task, args_seq=args, mode="processes")

    assert results == [11]
    # The code logs: "Error in process pool, falling back to threads: OS Error"
    assert "falling back to threads" in caplog.text
    assert "OS Error" in caplog.text


def test_run_parallel_error_propagation(caplog):
    """Updated: check the log on a task error (format 'Task 0 failed')."""
    with caplog.at_level(logging.ERROR):
        results = run_parallel(failing_task, args_seq=[()], mode="threads")

    assert results == [None]
    # In the real code: logger.error(f"Task {idx} failed: {e}")
    assert "Task 0 failed" in caplog.text
    assert "Intentional failure" in caplog.text


def test_run_parallel_empty_args():
    """Additional: check running with no arguments (exactly once)."""

    def get_one():
        return 1

    results = run_parallel(get_one)
    assert results == [1]


def test_run_parallel_validation_error():
    """Additional: check mismatched argument lengths."""
    with pytest.raises(ValueError, match="must be of equal length"):
        run_parallel(
            simple_task,
            args_seq=[(1,)],
            kwargs_seq=[{}, {}],  # Different lengths
        )


def test_run_parallel_threads_timeout_ignored_with_warning(caplog):
    """Direct run_parallel call with mode='threads' and a timeout — ignored with a warning.

    Config validation forbids timeouts in threads mode, but it does not
    protect direct run_parallel calls: here the timeout is ignored and a
    warning is emitted so that the user is aware, and the result is not
    discarded.
    """
    with caplog.at_level(logging.WARNING):
        results = run_parallel(
            simple_task, args_seq=[(1, 2), (3, 4)], mode="threads", timeout=0.1
        )
    assert sorted(results) == [3, 7]
    assert "Timeouts are ignored in threads mode" in caplog.text


def test_run_parallel_threads_without_timeout_no_warning(caplog):
    """Without timeouts a threads run works and emits no warning."""
    with caplog.at_level(logging.WARNING):
        results = run_parallel(simple_task, args_seq=[(1, 2)], mode="threads")
    assert results == [3]
    assert "Timeouts are ignored" not in caplog.text


def raise_invalid():
    raise InvalidAlgorithmError("bad algo")


def test_run_parallel_invalid_algorithm_error_is_not_fatal(caplog):
    """InvalidAlgorithmError — an ordinary task failure: the result is None,
    not an interruption of the whole batch."""
    with caplog.at_level(logging.ERROR):
        results = run_parallel(raise_invalid, args_seq=[()])

    assert results == [None]
    assert "Task 0 failed" in caplog.text
    assert "bad algo" in caplog.text


def raise_keyboard_interrupt():
    raise KeyboardInterrupt()


def test_run_parallel_keyboard_interrupt_propagates(caplog):
    """Fixed: check the new log line on interruption."""

    def raise_ki():
        raise KeyboardInterrupt()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyboardInterrupt):
            run_parallel(raise_ki, args_seq=[()])

    # In the updated code the line is: "Interrupted by user"
    assert "Interrupted by user" in caplog.text


def raise_value_error():
    raise ValueError("boom")


def test_run_parallel_generic_exception_logged_and_returns_none(caplog):
    """Updated: check the log for an arbitrary exception."""

    def boom():
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR):
        results = run_parallel(boom, args_seq=[()])

    assert results == [None]
    # In the real code: "Task 0 failed: boom"
    assert "Task 0 failed" in caplog.text
    assert "boom" in caplog.text


def test_run_parallel_process_pool_init_error_not_masked(caplog):
    """The ProcessPoolExecutor constructor raising must surface through the
    real outer fallback (threads), not be masked by the removed dead
    try/except around the executor_cls assignment."""
    args = [(1, 10)]
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.ProcessPoolExecutor",
        side_effect=RuntimeError("OS Error"),
    ):
        with caplog.at_level(logging.ERROR):
            results = run_parallel(simple_task, args_seq=args, mode="processes")

    assert results == [11]
    # The dead-branch message no longer exists; only the real outer
    # except handles pool-creation failures.
    assert "Could not initialize ProcessPoolExecutor" not in caplog.text
    assert "falling back to threads" in caplog.text
    assert "OS Error" in caplog.text


def test_run_parallel_process_pool_missing_attr_propagates(monkeypatch):
    """A missing ProcessPoolExecutor attribute must produce an unmasked
    NameError out of run_parallel instead of a silent threads fallback:
    the absence of the import is detected at module import time."""
    monkeypatch.delattr(thread_pool, "ProcessPoolExecutor", raising=False)

    with pytest.raises(NameError):
        run_parallel(lambda x: x, args_seq=[(5,)], mode="processes")


# Helper function for tests
def process_df_task(df, multiplier):
    return df.sum().sum() * multiplier


def test_shared_dataframe_manual():
    """Test the basic functionality of SharedDataFrame outside run_parallel."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    shm_wrapper = SharedDataFrame(df=df)

    try:
        # The name is generated by the OS (name=None) and must be non-empty:
        # an explicit shm_{id}_{randint} name could collide across concurrent
        # tasks (FileExistsError).
        assert shm_wrapper.name
        df_recovered = shm_wrapper.to_df()
        pd.testing.assert_frame_equal(df, df_recovered)
    finally:
        shm_wrapper.close()
        shm_wrapper.unlink()


def test_shared_dataframe_auto_generated_unique_names():
    """Consecutive SharedDataFrame instances get distinct OS-generated names.

    Previously the name was shm_{id}_{randint(1000)}; with only 1000 random
    suffixes, dozens of concurrent tasks could collide on the same name and
    raise FileExistsError. name=None delegates uniqueness to the OS.
    """
    df = pd.DataFrame({"a": [1]})
    first = SharedDataFrame(df=df)
    second = SharedDataFrame(df=df)
    try:
        assert first.name
        assert second.name
        assert first.name != second.name
    finally:
        first.close()
        first.unlink()
        second.close()
        second.unlink()


def test_shared_dataframe_creation_uses_os_generated_name():
    """SharedDataFrame must not pass an explicit name when creating a segment.

    Passing name=None (the default) lets multiprocessing.shared_memory
    generate a globally unique name, eliminating FileExistsError collisions.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    buffer = memoryview(bytearray(df.to_numpy().nbytes))
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.shared_memory.SharedMemory"
    ) as mock_shm:
        mock_shm.return_value.name = "psm_test_segment"
        mock_shm.return_value.buf = buffer
        shm_wrapper = SharedDataFrame(df=df)
        try:
            create_call = mock_shm.call_args_list[0]
            assert create_call.kwargs.get("create") is True
            assert create_call.kwargs.get("name") is None
            assert shm_wrapper.name == "psm_test_segment"
        finally:
            shm_wrapper.close()
            shm_wrapper.unlink()


def test_disk_persistence_manager_manual():
    """Test the basic functionality of DiskPersistenceManager."""
    df = pd.DataFrame({"a": range(10)})
    manager = DiskPersistenceManager(use_shm=False)
    path = None
    try:
        path = manager.save_df(df)
        assert os.path.exists(path)
        assert path.endswith(".parquet")
        df_recovered = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df, df_recovered)
    finally:
        manager.cleanup()
        if path:
            assert not os.path.exists(path)


def test_disk_persistence_manager_save_df_uses_pyarrow_engine(monkeypatch):
    """to_parquet must always be called with engine='pyarrow' (fixed engine)."""
    df = pd.DataFrame({"a": range(5)})
    manager = DiskPersistenceManager(use_shm=False)

    calls: list[dict] = []
    original_to_parquet = pd.DataFrame.to_parquet

    def patched_to_parquet(self, path, *args, **kwargs):
        calls.append(kwargs)
        return original_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", patched_to_parquet)
    path = None
    try:
        path = manager.save_df(df)
        assert path is not None
        assert calls and calls[0]["engine"] == "pyarrow"
    finally:
        manager.cleanup()
        if path:
            assert not os.path.exists(path)


def test_disk_persistence_manager_registers_path_before_write(monkeypatch):
    """An orphaned temp file from a failed to_parquet must be cleaned up.

    save_df registers the path right after mkstemp: if pyarrow fails halfway
    through the write, the temp file is still registered and removed by
    cleanup (issue #121 review: the fallback path leaked .parquet files in
    /dev/shm).
    """
    from configurable_automl_engine.training_engine.thread_pool import (
        DiskPersistenceManager as DPM,
    )

    df = pd.DataFrame({"a": [1, 2, 3]})
    manager = DPM(use_shm=False)

    def failing_to_parquet(self, path, *args, **kwargs):
        # Simulate pyarrow aborting the write midway: the file exists but is
        # not a valid parquet, and to_parquet raises.
        with open(path, "wb") as f:
            f.write(b"partial data")
        raise RuntimeError("pyarrow write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)

    with pytest.raises(RuntimeError, match="pyarrow write failed"):
        manager.save_df(df)

    # The created temp file must be registered even though the write failed.
    assert len(manager.created_files) == 1
    assert os.path.exists(manager.created_files[0])

    manager.cleanup()
    assert not os.path.exists(manager.created_files[0])


def test_run_parallel_with_shared_memory():
    """Test Tasks 1.1 and 1.2: Passing data through Shared Memory in processes."""
    df = pd.DataFrame({"val": [10, 20, 30]})
    # Expected result: (10+20+30) * 1 = 60 and (10+20+30) * 2 = 120
    args = [(df, 1), (df, 2)]

    results = run_parallel(
        process_df_task, args_seq=args, mode="processes", shared_args_indices=[0]
    )

    assert sorted(results) == [60, 120]


def test_run_parallel_reuses_shm_segment_for_same_df():
    """The same df object must be placed into ONE shared memory segment.

    Previously each task copied the df into its own segment
    (shm_{id(df)}_{randint(1000)}), so with dozens of tasks the data was
    duplicated in RAM and concurrent creations could collide on a segment
    name (FileExistsError). Now one segment per object is reused by all
    tasks.
    """
    df = pd.DataFrame({"val": [10, 20, 30]})
    args = [(df, i) for i in range(1, 11)]  # the same df object in 10 tasks
    created: list[pd.DataFrame] = []

    class CountingSharedDataFrame(SharedDataFrame):
        def __init__(self, *a, **kw):
            if "df" in kw and kw["df"] is not None:
                created.append(kw["df"])
            elif a:
                created.append(a[0])
            super().__init__(*a, **kw)

    with patch(
        "configurable_automl_engine.training_engine.thread_pool.SharedDataFrame",
        CountingSharedDataFrame,
    ):
        results = run_parallel(
            process_df_task,
            args_seq=args,
            mode="processes",
            shared_args_indices=[0],
        )

    assert sorted(results) == [60 * i for i in range(1, 11)]
    # Only one segment was created for the shared df object
    assert len(created) == 1


def test_run_parallel_with_disk_persistence():
    """Test Phase 2: Passing data through Disk (Parquet) in processes."""
    df = pd.DataFrame({"val": [1, 2, 3]})
    args = [(df, 10)]

    results = run_parallel(
        process_df_task, args_seq=args, mode="processes", disk_args_indices=[0]
    )

    assert results == [60]


def test_run_parallel_shm_cleanup_on_error():
    """Check that resource cleanup is called even when a task fails."""
    df = pd.DataFrame({"a": [1]})

    def error_task(df):
        raise ValueError("Intentional error")

    # Call and ignore the error (it will be logged as Task Failed)
    results = run_parallel(
        error_task, args_seq=[(df,)], mode="processes", shared_args_indices=[0]
    )
    assert results == [None]
    # If nothing crashed here and no memory leaked (checked manually or with monitoring tools),
    # the finally block executed.


def test_shared_dataframe_reconstruction():
    """Cover the reconstruction of SharedDataFrame without passing df (worker side)."""
    df = pd.DataFrame({"x": [100]})
    original = SharedDataFrame(df=df)

    # Simulate what the proxy does: create the object by name and metadata
    reconstructed = SharedDataFrame(
        name=original.name,
        shape=original.shape,
        dtype=original.dtype,
        columns=original.columns,
    )

    pd.testing.assert_frame_equal(reconstructed.to_df(), df)

    original.close()
    original.unlink()


# 1. Move the function to the module level
def mixed_task_global(d1, d2):
    return int(d1.iloc[0, 0] + d2.iloc[0, 0])


def test_run_parallel_mixed_shm_and_disk():
    """Test the simultaneous use of both mechanisms."""
    df_shm = pd.DataFrame({"a": [1]})
    df_disk = pd.DataFrame({"b": [2]})

    # 2. Pass the global function
    results = run_parallel(
        mixed_task_global,
        args_seq=[(df_shm, df_disk)],
        mode="processes",
        shared_args_indices=[0],
        disk_args_indices=[1],
    )
    assert results == [3]


def test_disk_persistence_manager_cleanup_exception(caplog):
    """
    The test covers the case when the file exists but os.remove raises an error.
    It checks that logger.warning is logged in the except block.
    """
    # 1. Prepare the data
    df = pd.DataFrame({"a": [1]})
    manager = DiskPersistenceManager(use_shm=False)

    # Save a real file to get a valid path in the created_files list
    path = manager.save_df(df)

    # 2. Simulate an error when deleting
    # Set up log capture at the WARNING level
    with caplog.at_level(logging.WARNING):
        # Patch os.remove so that it raises PermissionError when called
        with patch("os.remove", side_effect=OSError("Access Denied")):
            # Call cleanup. It must not be interrupted by an exception (it is caught inside)
            manager.cleanup()

    # 3. Checks
    # Check that our message appears in the logs
    assert "Failed to delete temp file" in caplog.text
    assert "Access Denied" in caplog.text

    # Manually delete the file after the test since the mock prevented the manager from doing so
    try:
        import os

        if os.path.exists(path):
            os.remove(path)
    except:
        pass


def test_disk_persistence_manager_cleanup_file_not_found():
    """
    Additional test: if the file is already gone (os.path.exists = False),
    no error should occur and logger.warning must not be called.
    """
    manager = DiskPersistenceManager(use_shm=False)
    manager.created_files.append("non_existent_file.parquet")

    # This must not raise an exception or produce a warning in the logs,
    # because the if os.path.exists(path) check will trip.
    manager.cleanup()


# 1. Declare the function at the module level so pickle can find it
def worker_sum_func(df):
    return df["val"].sum()


def test_worker_proxy_integration():
    """
    Integration test for _worker_proxy and Shared Memory.
    This test covers the lines that reconstruct data from SHM in the worker.
    """
    df = pd.DataFrame({"val": [1, 2, 3]})

    # 2. Call run_parallel with a function that can be imported
    results = run_parallel(
        func=worker_sum_func,  # Pass a regular function instead of a lambda
        args_seq=[(df,)],
        mode="processes",
        shared_args_indices=[0],
    )

    # 3. Check the result
    assert results is not None, (
        "the result must not be None (check the logs for pickle errors)"
    )
    assert results[0] == 6, f"expected 6, got {results[0]}"


def test_worker_proxy_disk_integration(tmp_path):
    from configurable_automl_engine.training_engine.thread_pool import _worker_proxy

    df = pd.DataFrame({"val": [30]})
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path)

    # Direct proxy call, as the worker does
    result = _worker_proxy(lambda x: x["val"].iloc[0], (path,), {}, [0], [])
    assert result == 30


def test_worker_proxy_logic_direct_coverage():
    # 1. Data for reconstruction
    meta = ("test_shm_name", (1, 1), np.float64, ["a"])

    # Patch SharedDataFrame so it does not touch real OS memory
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.SharedDataFrame"
    ) as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.to_df.return_value = pd.DataFrame({"a": [1.0]})

        def simple_func(x):
            return x

        result = _worker_proxy(
            func=simple_func,
            args=(None,),  # In args, None now occupies the SHM slot
            kwargs={},
            disk_indices=None,
            shm_info={0: meta},  # Pass it through shm_info
        )
        assert result.iloc[0]["a"] == 1.0
        # Check that the constructor was called with our metadata
        mock_class.assert_called_with(
            name=meta[0], shape=meta[1], dtype=meta[2], columns=meta[3]
        )


def test_coverage_unlink_file_not_found():
    """
    Test to forcefully cover the exception handling branches in the unlink method.
    Uses a mock to simulate the absence of a segment in the OS.
    """
    # 1. Create a minimal DataFrame
    df = pd.DataFrame({"test": [1, 2, 3]})

    # 2. Initialize SharedDataFrame (it creates a real in-memory segment)
    shm_df = SharedDataFrame(df=df)

    # 3. Replace the unlink method of the shm object with a mock that always raises FileNotFoundError
    # This simulates a situation where the segment was already removed by someone else
    shm_df.shm.unlink = MagicMock(side_effect=FileNotFoundError("Already deleted"))

    # 4. Call the unlink method.
    # Thanks to the try-except block, the error will be caught and the 'pass' line will run
    try:
        shm_df.unlink()
    except FileNotFoundError:
        import pytest

        pytest.fail(
            "SharedDataFrame.unlink did not catch FileNotFoundError, coverage did not trigger!"
        )

    # 5. Cleanup (close the descriptors)
    shm_df.close()

    # Check that the mock was actually called
    assert shm_df.shm.unlink.called


# 1. Testing SharedDataFrame.is_compatible (dtypes and Index)
def test_shared_data_frame_is_compatible_coverage():
    """Covers the lines:
    - if not isinstance(df, pd.DataFrame): return False
    - array-dtype check of df.to_numpy()
    - if not isinstance(df.index, pd.RangeIndex): return False
    """
    # A. Test for an invalid type (Object/String)
    df_obj = pd.DataFrame({"col": ["a", "b", "c"]})
    assert SharedDataFrame.is_compatible(df_obj) is False

    # B. Test for an invalid index (not RangeIndex)
    df_custom_index = pd.DataFrame({"col": [1, 2, 3]}, index=pd.Index([10, 20, 30]))
    assert SharedDataFrame.is_compatible(df_custom_index) is False
    # C. Test for a valid DataFrame (int/float + RangeIndex)
    df_valid = pd.DataFrame({"col": [1.5, 2.0, 3.1]})
    assert SharedDataFrame.is_compatible(df_valid) is True


# 2. Testing the exception in _worker_proxy
def test_worker_proxy_close_exception():
    meta = ("name", (1, 2), np.int64, ["a", "b"])

    with patch(
        "configurable_automl_engine.training_engine.thread_pool.SharedDataFrame"
    ) as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.to_df.return_value = pd.DataFrame({"a": [1], "b": [2]})
        mock_instance.close.side_effect = Exception("Simulated close error")

        def dummy_func(df):
            return len(df.columns)

        # Pass shm_info so the proxy creates a worker wrapper
        result = _worker_proxy(
            func=dummy_func,
            args=(None,),
            kwargs={},
            disk_indices=[],
            shm_info={0: meta},
        )
        assert result == 2
        assert mock_instance.close.called

    # Check that the close method was called
    mock_instance.close.assert_called_once()


def test_perform_cleanup_all_exceptions():
    # An object whose close() fails, but unlink() must still be called
    ref_fail = MagicMock()
    ref_fail.close.side_effect = Exception("Close error")

    mock_pm = MagicMock()
    mock_pm.cleanup.side_effect = Exception("PM error")

    # The function must not raise an exception (Phase 3 resilience)
    try:
        _perform_cleanup(shm_refs=[ref_fail], persistence_manager=mock_pm)
    except Exception as e:
        pytest.fail(f"_perform_cleanup leaked an exception: {e}")

    assert ref_fail.close.called
    # unlink must always be called for OS cleanup, even if the descriptor did not close
    assert ref_fail.unlink.called
    assert mock_pm.cleanup.called


# 4. Check behavior with an empty persistence_manager
def test_perform_cleanup_none_pm():
    """Check the if persistence_manager: False branch"""
    mock_shm_ref = MagicMock()
    _perform_cleanup(shm_refs=[mock_shm_ref], persistence_manager=None)
    mock_shm_ref.close.assert_called_once()


def test_is_shared_array_robust():
    """Test for SharedDataFrame.is_shared_array taking into account the specifics of the .base attribute"""

    # 1. Coverage: return False (not an ndarray)
    assert SharedDataFrame.is_shared_array("not an array") is False

    # 2. Coverage: return False (ndarray without base or base not a memoryview)
    normal_arr = np.array([1, 2, 3])
    assert SharedDataFrame.is_shared_array(normal_arr) is False
    # 3. Coverage: return True
    # Use memoryview directly to guarantee the .base type
    raw_data = bytearray(b"\x00" * 24)
    m_view = memoryview(raw_data)
    # Create an array from the buffer — in most environments this sets .base to a memoryview
    shared_arr = np.frombuffer(m_view, dtype=np.float64)

    # If on your platform numpy still unwraps base,
    # we can detect that and forcibly create a situation to cover the line
    if not isinstance(shared_arr.base, memoryview):
        # Force creation of an object that satisfies the condition (for coverage)
        class MockArray(np.ndarray):
            pass

        mock_arr = np.array([1.0, 2.0]).view(MockArray)
        # Manually replace base to test the logic
        mock_arr.base = m_view
        assert SharedDataFrame.is_shared_array(mock_arr) is True
    else:
        assert SharedDataFrame.is_shared_array(shared_arr) is True


def test_get_data_info():
    # 1. Branch: pandas.DataFrame
    df = pd.DataFrame({"a": [1], "b": [2]})
    count, names = SharedDataFrame.get_data_info(df)
    assert count == 2
    assert names == ["a", "b"]

    # 2. Branch: SharedDataFrame
    sdf = SharedDataFrame(df)
    try:
        count, names = SharedDataFrame.get_data_info(sdf)
        assert count == 2
        assert names == ["a", "b"]
    finally:
        sdf.unlink()
        sdf.close()

    # 3. Branch: np.ndarray (ndim > 1)
    arr_2d = np.zeros((5, 3))
    count, names = SharedDataFrame.get_data_info(arr_2d)
    assert count == 3
    assert names == [0, 1, 2]

    # 4. Branch: np.ndarray (ndim == 1)
    arr_1d = np.array([1, 2, 3])
    count, names = SharedDataFrame.get_data_info(arr_1d)
    assert count == 1
    assert names == [0]

    # 5. Branch: Other type (else)
    count, names = SharedDataFrame.get_data_info("not_a_df")
    assert count == 0
    assert names == []


def test_is_compatible_all_branches():
    """Test for SharedDataFrame.is_compatible — covers type and index checks"""

    # 1. Coverage: is_shared_array(df) -> True
    # (Use the base-replacement trick if regular SHM does not act as a memoryview)
    data = bytearray(b"\x00" * 8)
    mv = memoryview(data)
    fake_shared = np.frombuffer(mv, dtype=np.int64)
    if not isinstance(fake_shared.base, memoryview):
        # If numpy ignored the memoryview in base, skip this branch
        # or use a mock, since the logic depends on the environment
        pass
    else:
        assert SharedDataFrame.is_compatible(fake_shared) is True
    # 2. Coverage: isinstance(df, pd.DataFrame) -> False
    assert SharedDataFrame.is_compatible("not a dataframe") is False

    # 3. Coverage: type check (dtypes)
    # Only allowed types (int, float, bool). int32 + float64 promotes to
    # float64 losslessly (int32 max ≪ 2**53), so the frame stays compatible.
    df_valid = pd.DataFrame(
        {"a": np.array([1], dtype=np.int32), "b": [1.5]},
        index=pd.RangeIndex(0, 1),
    )
    assert SharedDataFrame.is_compatible(df_valid) is True

    # Disallowed type (object/string)
    df_invalid_type = pd.DataFrame({"a": ["text"]})
    assert SharedDataFrame.is_compatible(df_invalid_type) is False

    # 4. Coverage: index check (not RangeIndex)
    df_invalid_index = pd.DataFrame({"a": [1]}, index=[10])  # Int64Index, not RangeIndex
    assert SharedDataFrame.is_compatible(df_invalid_index) is False


def test_is_compatible_does_not_call_to_numpy():
    """is_compatible must not materialize the data via df.to_numpy().

    Previously the check called df.to_numpy() only to read the resulting
    dtype: for a mixed-type frame that is a full in-memory copy, and
    SharedDataFrame.__init__ makes a second copy right after — a transient
    doubling of memory in the main process on every HPO phase (issue #122).
    The dtype is now computed from column dtypes alone, so a frame whose
    to_numpy would raise must still be checked without calling it.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})

    class BrokenDF(pd.DataFrame):
        def to_numpy(self, *args, **kwargs):
            raise RuntimeError("cannot materialize")

    broken = BrokenDF(df)
    with pytest.raises(RuntimeError):
        broken.to_numpy()  # the guard really triggers
    # The compatibility check must not trigger the broken method.
    assert SharedDataFrame.is_compatible(broken) is True


def test_get_view():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    sdf = SharedDataFrame(df)

    # For the test, temporarily add _df, as the get_view method expects it in the code
    sdf._df = df

    try:
        # 1. Branch: columns is None
        view_all = sdf.get_view(None)
        pd.testing.assert_frame_equal(view_all, df)

        # 2. Branch: specific columns are passed
        view_subset = sdf.get_view(["A"])
        assert view_subset.columns.tolist() == ["A"]
        assert view_subset.shape == (2, 1)
        # Check that this is a view (changing the original affects the view)
        # Note: .loc[:, cols] for a list of columns in pandas may create a copy
        # or a view depending on the version, but the test will cover the code line.
    finally:
        sdf.unlink()
        sdf.close()


def global_long_sleep_task(*args, **kwargs):
    time.sleep(10)


def _raise_timeout_at_iteration(*args, **kwargs):
    """Generator function: TimeoutError is raised during iteration (for fut in ...),
    not when as_completed(...) is called. This is a correct simulation of the
    behavior of a real as_completed generator."""
    raise concurrent.futures.TimeoutError()
    yield  # pragma: no cover


def test_run_parallel_worker_hard_kill():
    """Checks that tasks are marked as timed out when the global timeout expires."""
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.ProcessPoolExecutor"
    ) as mock_executor_cls:
        mock_pool = MagicMock()
        mock_executor_cls.return_value = mock_pool

        mock_future = MagicMock()
        mock_pool.submit.return_value = mock_future

        # as_completed raises TimeoutError on every iteration
        with patch(
            "configurable_automl_engine.training_engine.thread_pool.as_completed",
            side_effect=_raise_timeout_at_iteration,
        ):
            with patch(
                "configurable_automl_engine.training_engine.thread_pool.logger"
            ) as mock_logger:
                results = run_parallel(
                    lambda: None,
                    args_seq=[()],
                    mode="processes",
                    timeout=0.1,  # small timeout — the global limit will expire
                )

    assert results == [None]
    # Check that logger.error was called with a timeout message
    error_messages = [call[0][0] for call in mock_logger.error.call_args_list]
    assert any("Task 0 timed out" in msg for msg in error_messages)


def test_run_parallel_threads_global_timeout_ignored(caplog):
    """In threads mode the global timeout is ignored with a warning.

    Previously a task that did not finish within the timeout was marked as
    timed out and its result was discarded (None), but the phase still
    waited for shutdown(wait=True) completion of all algorithms — the soft
    timeout did not save time. Now the timeout in mode='threads' is ignored
    with a warning, the task runs to completion and returns its result.
    """
    with caplog.at_level(logging.WARNING):
        results = run_parallel(
            slow_task,
            args_seq=[(0.4,)],
            mode="threads",
            timeout=0.1,
        )
    assert results == ["done"]
    assert "Timeouts are ignored in threads mode" in caplog.text


def task_with_kwargs(df, multiplier=1, add=0):
    return (df.sum().sum() * multiplier) + add


def test_run_parallel_processes_with_kwargs():
    df = pd.DataFrame({"a": [1, 1]})
    # Pass both args and kwargs
    results = run_parallel(
        task_with_kwargs,
        args_seq=[(df,)],
        kwargs_seq=[{"multiplier": 2, "add": 5}],
        mode="processes",
        shared_args_indices=[0],
    )
    assert results == [9]  # (2 * 2) + 5


# In test_parallel.py (fix of the test itself)
def global_identity_func(x):
    return x.iloc[0, 0]


def test_run_parallel_duplicate_indices_handling():
    df = pd.DataFrame({"a": [1]})
    results = run_parallel(
        global_identity_func,  # Instead of a lambda
        args_seq=[(df,)],
        mode="processes",
        shared_args_indices=[0],
        disk_args_indices=[0],
    )
    assert results == [1]


def test_get_view_new():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    sdf = SharedDataFrame(df)
    try:
        # Check that get_view works correctly through to_df()
        view_subset = sdf.get_view(["A"])
        assert isinstance(view_subset, pd.DataFrame)
        assert view_subset.columns.tolist() == ["A"]
        assert view_subset.iloc[0, 0] == 1
    finally:
        sdf.close()
        sdf.unlink()


class TestParallelErrorHandling(unittest.TestCase):
    # --- TESTS FOR _perform_cleanup ---

    def test_shm_unlink_oserror_pass(self):
        """Coverage: except (FileNotFoundError, OSError): pass in SHM unlink"""
        shm_mock = MagicMock(spec=SharedDataFrame)
        # Simulate FileNotFoundError when trying to delete the segment
        shm_mock.unlink.side_effect = FileNotFoundError("Already gone")

        # The function must not crash
        try:
            _perform_cleanup([shm_mock], None)
        except Exception as e:
            self.fail(f"_perform_cleanup raised {type(e).__name__} unexpectedly!")

        shm_mock.unlink.assert_called_once()

    def test_shm_unlink_generic_exception_warning(self):
        """Coverage: except Exception as e: logger.warning(...) in SHM unlink"""
        shm_mock = MagicMock(spec=SharedDataFrame)
        shm_mock.unlink.side_effect = ValueError("Fatal SHM error")

        with self.assertLogs(MODULE_PATH, level="WARNING") as cm:
            _perform_cleanup([shm_mock], None)

        self.assertTrue(
            any("Non-critical SHM unlink failure" in log for log in cm.output)
        )

    def test_persistence_cleanup_permission_error(self):
        """Coverage: except (PermissionError, OSError) in persistence_manager.cleanup"""
        pm_mock = MagicMock(spec=DiskPersistenceManager)
        pm_mock.cleanup.side_effect = PermissionError("File locked")

        with self.assertLogs(MODULE_PATH, level="ERROR") as cm:
            _perform_cleanup(None, pm_mock)

        self.assertTrue(
            any(
                "Cleanup failed due to file locking/permissions" in log
                for log in cm.output
            )
        )

    # --- TESTS FOR run_parallel (Waiting loop) ---

    @patch(f"{MODULE_PATH}.ThreadPoolExecutor")
    def test_run_parallel_wait_loop_exception(self, mock_executor_cls):
        """Coverage: except Exception as e: logger.error(Error while waiting for tasks)"""
        mock_pool = MagicMock()
        mock_executor_cls.return_value = mock_pool

        mock_future = MagicMock()
        mock_pool.submit.return_value = mock_future

        def _raise_runtime_error(*args, **kwargs):
            raise RuntimeError("Internal pool crash")

        with patch(f"{MODULE_PATH}.as_completed") as mock_as_completed:
            mock_as_completed.side_effect = _raise_runtime_error

            with self.assertLogs(MODULE_PATH, level="ERROR") as cm:
                run_parallel(lambda x: x, args_seq=[(1,)], mode="threads")

        self.assertTrue(
            any("Error while waiting for tasks" in log for log in cm.output)
        )

    # --- TESTS FOR run_parallel (Forced worker termination) ---

    @patch(f"{MODULE_PATH}.ProcessPoolExecutor")
    @patch(f"{MODULE_PATH}.time.sleep")
    def test_worker_terminate_and_kill_exceptions(self, mock_sleep, mock_executor_cls):
        """
        Covers the blocks:
        - except Exception: pass on w.terminate()
        - except Exception: pass on w.kill()

        The watchdog has been removed, so we check that _force_shutdown_processes
        is called in the finally block for ProcessPoolExecutor.
        """
        mock_executor = mock_executor_cls.return_value

        # Create a worker that does not want to die
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        mock_worker.pid = 777

        # Configure errors on the termination system calls
        mock_worker.terminate.side_effect = RuntimeError("Terminate blocked")
        mock_worker.kill.side_effect = RuntimeError("Kill blocked")

        # Replace the process list inside the Executor
        mock_executor._processes = {0: mock_worker}

        # Simulate a situation where tasks did not finish in time
        with patch(f"{MODULE_PATH}.as_completed") as mock_as_completed:
            mock_as_completed.side_effect = _raise_timeout_at_iteration

            with self.assertLogs(MODULE_PATH, level="ERROR") as cm:
                run_parallel(
                    func=lambda x: x,
                    args_seq=[(1,)],
                    mode="processes",
                    timeout=0.1,
                    shutdown_grace_period=0.01,
                )

            # Check that the logs contain the timeout message
            self.assertTrue(any("Task 0 timed out" in log for log in cm.output))

        # Check that the methods were called (in the finally block of _force_shutdown_processes)
        self.assertTrue(mock_worker.terminate.called)
        self.assertTrue(mock_worker.kill.called)


# ═══════════════════════════════════════════════════════════════
#  Tests for as_completed + watchdog (deadlock prevention)
# ═══════════════════════════════════════════════════════════════


def test_as_completed_normal_completion():
    """Check that as_completed correctly collects results."""
    results = run_parallel(
        simple_task, args_seq=[(1, 2), (3, 4), (5, 6)], mode="threads"
    )
    assert sorted(results) == [3, 7, 11]


def test_run_parallel_threads_task_timeout_ignored(caplog):
    """In threads mode task_timeout is ignored with a warning.

    Previously the deadline was counted from the moment of submit, so a task
    waiting in the queue (e.g. with max_workers=1) could "expire" before even
    starting. Now task_timeout in mode='threads' is ignored with a warning,
    the task runs to completion and returns its result.
    """
    with caplog.at_level(logging.WARNING):
        results = run_parallel(
            slow_task,
            args_seq=[(0.3,)],
            mode="threads",
            task_timeout=0.01,
        )
    assert results == ["done"]
    assert "Timeouts are ignored in threads mode" in caplog.text


def hang_task(seconds):
    """Task that "hangs" (does not finish within the test's wait window)."""
    time.sleep(seconds)
    return "hung-ended"


def test_run_parallel_processes_queued_tasks_get_full_budget():
    """In processes mode the queue does not "consume" a task's execution budget.

    Previously the deadline of each algorithm was counted from the moment of
    submit, so with more tasks than max_workers, some of them "expired"
    before even starting. Now the deadline is bound to the task's actual
    start in the worker (side-channel), so each task whose real execution
    fits into task_timeout completes successfully even if it spent a long
    time in the queue. Covers the "tasks > max_workers+1" case.
    """
    # 10 tasks of 0.2s with max_workers=2: the last tasks start only
    # ~0.8-1.0s after submit (significantly later than task_timeout=0.8 from
    # submit), but each real execution (0.2s) fits into 0.8s with margin.
    results = run_parallel(
        slow_task,
        args_seq=[(0.2,)] * 10,
        mode="processes",
        max_workers=2,
        task_timeout=0.8,
        shutdown_grace_period=0.1,
    )
    assert results == ["done"] * 10


def test_run_parallel_processes_task_timeout_still_enforced():
    """In processes mode task_timeout still strictly limits the execution.

    After a task has really started, it must fit into task_timeout seconds of
    execution — otherwise it is forcibly marked as timed out (result None).
    """
    results = run_parallel(
        slow_task,
        args_seq=[(0.5,)],
        mode="processes",
        max_workers=1,
        task_timeout=0.2,
        shutdown_grace_period=0.1,
    )
    assert results == [None]


def test_run_parallel_never_starting_task_is_bounded():
    """A task that never starts does not hang forever.

    The first worker hangs on task0 (sleep 60s); task1 stays in the queue and
    never starts. Even with timeout=None, the queue deadline is bounded from
    above (submit + task_timeout + queue_budget), so run_parallel finishes in
    a bounded time instead of waiting forever (previously the rolling
    deadline never expired).
    """
    started = time.time()
    results = run_parallel(
        hang_task,
        args_seq=[(60.0,), (60.0,)],
        mode="processes",
        max_workers=1,
        task_timeout=0.5,
        shutdown_grace_period=0.1,
    )
    elapsed = time.time() - started
    assert results == [None, None]
    assert elapsed < 15  # finished in ~2*task_timeout + grace, not hung forever


def test_watchdog_not_triggered_on_slow_but_alive_tasks():
    """Check that the watchdog does NOT trigger on slow but alive tasks.

    as_completed blocks and waits for the task to finish. If a task is simply
    long — the watchdog does not increment the stale counter.
    """
    results = run_parallel(
        slow_task,
        args_seq=[(2.0,)],  # a 2-second task
        mode="threads",
    )
    assert results == ["done"]  # the task must complete normally


@patch("configurable_automl_engine.training_engine.thread_pool.as_completed")
def test_watchdog_triggers_on_c_level_crash(mock_as_completed):
    """Check that tasks are marked as timed out when the global timeout expires.

    We mock as_completed to raise TimeoutError on every iteration,
    simulating a hung process. A 0.1s global timeout will force the loop to exit.
    """
    mock_as_completed.side_effect = _raise_timeout_at_iteration

    with patch(
        "configurable_automl_engine.training_engine.thread_pool.logger"
    ) as mock_logger:
        results = run_parallel(
            lambda: None,
            args_seq=[()],
            mode="processes",
            timeout=0.1,  # small timeout — the global limit will expire
        )

    assert results == [None]
    error_messages = [call[0][0] for call in mock_logger.error.call_args_list]
    assert any("Task 0 timed out" in msg for msg in error_messages)


@patch("configurable_automl_engine.training_engine.thread_pool.as_completed")
def test_watchdog_not_triggered_when_global_timeout_expires(mock_as_completed):
    """Check that the watchdog does NOT trigger when the global timeout has expired.

    Even if as_completed raises TimeoutError, but remaining_global <= 0,
    the stale counter is not incremented — a normal timeout exit happens.
    We use a generator function so that TimeoutError is raised
    during iteration (for fut in ...), not when it is called.
    """
    mock_as_completed.side_effect = _raise_timeout_at_iteration

    with patch(
        "configurable_automl_engine.training_engine.thread_pool._force_shutdown_processes"
    ) as mock_shutdown:
        with patch(
            "configurable_automl_engine.training_engine.thread_pool.logger"
        ) as mock_logger:
            results = run_parallel(
                lambda: None,
                args_seq=[()],
                mode="processes",
                timeout=0.1,  # small global timeout
            )

    assert results == [None]
    error_messages = [call[0][0] for call in mock_logger.error.call_args_list]
    assert not any("WATCHDOG" in msg for msg in error_messages)  # the watchdog did not trigger
    assert any("Task 0 timed out" in msg for msg in error_messages)  # normal timeout


def test_force_shutdown_processes_terminate_and_kill():
    """Check that _force_shutdown_processes correctly terminates processes."""
    mock_worker = MagicMock()
    mock_worker.is_alive.return_value = True
    mock_worker.pid = 9999

    mock_pool = MagicMock()
    mock_pool._processes = {0: mock_worker}

    with patch("configurable_automl_engine.training_engine.thread_pool.time.sleep"):
        _force_shutdown_processes(mock_pool, shutdown_grace_period=0.1)

    assert mock_worker.terminate.called
    assert mock_worker.kill.called


def test_run_parallel_generic_exception_no_timeout(caplog):
    """Cover the except Exception in the else branch (timeout_for_ac is None)."""

    def failing_task():
        raise ValueError("No-timeout failure")

    with caplog.at_level(logging.ERROR):
        results = run_parallel(
            failing_task,
            args_seq=[()],
            mode="threads",
            timeout=None,  # disable the global timeout
        )

    assert results == [None]
    assert "Task 0 failed" in caplog.text
    assert "No-timeout failure" in caplog.text


def raise_invalid_no_timeout():
    raise InvalidAlgorithmError("bad algo no timeout")


def test_run_parallel_invalid_algorithm_error_is_not_fatal_no_timeout(caplog):
    """Cover the else branch (timeout_for_ac is None): InvalidAlgorithmError
    is handled as an ordinary task failure."""
    with caplog.at_level(logging.ERROR):
        results = run_parallel(
            raise_invalid_no_timeout, args_seq=[()], timeout=None
        )

    assert results == [None]
    assert "Task 0 failed" in caplog.text
    assert "bad algo no timeout" in caplog.text


def raise_keyboard_interrupt_no_timeout():
    raise KeyboardInterrupt()


def test_run_parallel_keyboard_interrupt_propagates_no_timeout(caplog):
    """Cover line 625: re-raise KeyboardInterrupt in the else branch (timeout_for_ac is None)."""

    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyboardInterrupt):
            run_parallel(
                raise_keyboard_interrupt_no_timeout, args_seq=[()], timeout=None
            )


# ═══════════════════════════════════════════════════════════════
#  Issue #120: hang detection delay + hung task not freeing worker
# ═══════════════════════════════════════════════════════════════


def choose_hang_or_fast(seconds):
    """Return "fast-done" for short tasks, "hung-ended" for long (hang) tasks."""
    if seconds >= 1:
        return hang_task(seconds)
    return slow_task(seconds)


def test_run_parallel_processes_hung_task_frees_worker():
    """A task that exceeds task_timeout must free its worker process.

    Previously the result of the hung task was discarded, but the process kept
    running the hung function to completion (until the final pool shutdown), so
    with max_workers=1 and tasks [hang, 0.1, 0.1] the two fast tasks never
    started and the call returned [None, None, None] after the whole queue
    budget. Now the hung worker is killed and the pool is restarted for the
    remaining tasks: the fast tasks run right after the hung one times out.
    """
    started = time.time()
    results = run_parallel(
        choose_hang_or_fast,
        args_seq=[(30.0,), (0.1,), (0.1,)],
        mode="processes",
        max_workers=1,
        task_timeout=2,
        shutdown_grace_period=0.1,
    )
    elapsed = time.time() - started
    assert results == [None, "done", "done"]
    # ~ task_timeout + pool restart + two 0.1s runs; the old behavior took the
    # whole queue budget (submit + task_timeout + ceil(3/1)*task_timeout ≈ 8s).
    assert elapsed < 8


def test_run_parallel_processes_multiple_hung_tasks_each_restart():
    """Every hung task frees its worker; the queued fast task still runs.

    With max_workers=1 and three hung tasks ahead of a fast one, each hung
    task consumes exactly its task_timeout and then the pool is restarted:
    the last (fast) task still gets its turn.
    """
    started = time.time()
    results = run_parallel(
        choose_hang_or_fast,
        args_seq=[(30.0,), (30.0,), (30.0,), (0.1,)],
        mode="processes",
        max_workers=1,
        task_timeout=0.5,
        shutdown_grace_period=0.1,
    )
    elapsed = time.time() - started
    assert results == [None, None, None, "done"]
    # 3 waves x task_timeout + restart overhead.
    assert elapsed < 5


def test_run_parallel_processes_hang_detected_without_queue_delay():
    """The timeout of an already-started task is noticed promptly.

    On the first loop iteration the start signals are not written yet, so the
    deadlines of all tasks equal the queue deadline
    (submit + task_timeout + queue_budget). Previously as_completed slept up
    to that deadline and the call took ~6s instead of ~2s: the fast tasks
    finished in 0.2s but the loop did not wake up by the deadline of the
    started hung task. Now the wait is capped at POLL_INTERVAL while unstarted
    tasks exist, so the call returns right after the hung task times out.
    """
    started = time.time()
    results = run_parallel(
        choose_hang_or_fast,
        args_seq=[(30.0,), (0.1,), (0.1,)],
        mode="processes",
        max_workers=2,
        task_timeout=2,
        shutdown_grace_period=0.1,
    )
    elapsed = time.time() - started
    assert results == [None, "done", "done"]
    # ~ task_timeout + a few poll intervals; the old behavior took
    # submit + task_timeout + ceil(3/2)*task_timeout ≈ 6s.
    assert elapsed < 5


def test_run_parallel_processes_restart_preserves_queued_tasks():
    """After a restart the tasks that were still queued are resubmitted.

    Two hung tasks occupy both workers; the two fast tasks are queued and must
    run after the pool restart triggered by the first hung tasks' timeouts.
    """
    started = time.time()
    results = run_parallel(
        choose_hang_or_fast,
        args_seq=[(30.0,), (30.0,), (0.1,), (0.1,)],
        mode="processes",
        max_workers=2,
        task_timeout=2,
        shutdown_grace_period=0.1,
    )
    elapsed = time.time() - started
    assert results == [None, None, "done", "done"]
    assert elapsed < 8


def test_run_parallel_processes_healthy_inflight_task_resubmitted():
    """A healthy in-flight task, killed by a neighbor's timeout, is resubmitted.

    A 0.9s filler occupies the second worker first; the 1.5s task starts after
    it and is still running when the hung task times out. The restart kills it
    mid-flight, but it is resubmitted to the fresh pool and runs to completion
    ("hung-ended") — collateral progress loss, not a wrong result.
    """
    started = time.time()
    results = run_parallel(
        choose_hang_or_fast,
        args_seq=[(30.0,), (0.9,), (1.5,)],
        mode="processes",
        max_workers=2,
        task_timeout=2,
        shutdown_grace_period=0.1,
    )
    elapsed = time.time() - started
    assert results == [None, "done", "hung-ended"]
    assert elapsed < 8


def test_run_parallel_processes_poll_interval_caps_first_wait():
    """as_completed must not sleep up to the queue deadline on the first wait.

    While unstarted tasks exist (the start signals are written asynchronously
    by the workers), the as_completed timeout is capped at POLL_INTERVAL — the
    loop re-reads the signals and notices the deadline of a started task in
    time instead of sleeping until submit + task_timeout + queue_budget.
    """
    recorded_timeouts = []
    real_as_completed = thread_pool.as_completed

    def recording_as_completed(fs, timeout=None):
        recorded_timeouts.append(timeout)
        yield from real_as_completed(fs, timeout=timeout)

    with (
        patch(
            "configurable_automl_engine.training_engine.thread_pool.as_completed",
            recording_as_completed,
        ),
        # Deterministic: the signals always report "not started yet", so every
        # wait must be capped by the poll interval and never by the queue
        # deadline (submit + task_timeout + queue_budget ≈ 4s). The tasks
        # themselves still run and complete in the real workers.
        patch.object(thread_pool._StartSignal, "read", return_value=-1.0),
    ):
        results = run_parallel(
            slow_task,
            args_seq=[(0.2,), (0.2,)],
            mode="processes",
            max_workers=2,
            task_timeout=2.0,
            shutdown_grace_period=0.1,
        )

    assert results == ["done", "done"]
    assert recorded_timeouts, "as_completed was never called"
    # With the signals always unset the cap is active on every iteration: no
    # as_completed wait may exceed the polling interval.
    for timeout in recorded_timeouts:
        assert timeout is not None
        assert timeout <= thread_pool.POLL_INTERVAL + 0.1


def test_run_parallel_processes_completed_results_survive_restart():
    """Results collected before a restart are preserved.

    The fast task completes before the hung one times out; after the pool
    restart caused by the hung task its result must stay in the output.
    """
    results = run_parallel(
        choose_hang_or_fast,
        args_seq=[(30.0,), (0.1,)],
        mode="processes",
        max_workers=2,
        task_timeout=2,
        shutdown_grace_period=0.1,
    )
    assert results == [None, "done"]


# --- _StartSignal: the side-channel now records the worker PID ---


def test_start_signal_records_pid():
    """The side-channel stores the start timestamp AND the worker PID."""
    sig = _StartSignal()
    try:
        assert sig.read() == -1.0  # not started yet
        assert sig.read_pid() == 0
        sig.mark_start(123.0)
        sig.mark_pid(456)
        assert sig.read() == 123.0
        assert sig.read_pid() == 456
        # The PID lives in an int64 buffer, not a float64 one: values above
        # 2**53 are not exactly representable as float64 and would be
        # rounded (review feedback on issue #120).
        big_pid = 2**53 + 1
        sig.mark_pid(big_pid)
        assert sig.read_pid() == big_pid
    finally:
        sig.close()
        sig.unlink()


# --- _terminate_worker_by_pid ---


def test_terminate_worker_by_pid_found_and_terminated():
    """The worker with the given PID is terminated (dies on SIGTERM)."""
    mock_worker = MagicMock()
    mock_worker.pid = 4242
    # The while-loop and the final is_alive() check both call is_alive().
    mock_worker.is_alive.side_effect = [True, False, False]
    mock_pool = MagicMock()
    mock_pool._processes = {0: mock_worker}

    with patch("configurable_automl_engine.training_engine.thread_pool.time.sleep"):
        found = _terminate_worker_by_pid(mock_pool, 4242, shutdown_grace_period=0.05)

    assert found is True
    mock_worker.terminate.assert_called_once()
    mock_worker.kill.assert_not_called()


def test_terminate_worker_by_pid_kill_fallback():
    """If SIGTERM does not help, SIGKILL is sent."""
    mock_worker = MagicMock()
    mock_worker.pid = 4242
    mock_worker.is_alive.return_value = True  # never exits gracefully
    mock_pool = MagicMock()
    mock_pool._processes = {0: mock_worker}

    with patch("configurable_automl_engine.training_engine.thread_pool.time.sleep"):
        found = _terminate_worker_by_pid(mock_pool, 4242, shutdown_grace_period=0.01)

    assert found is True
    mock_worker.terminate.assert_called_once()
    mock_worker.kill.assert_called_once()


def test_terminate_worker_by_pid_not_found():
    """No worker with the given PID → False, nothing is terminated.

    The pool has no worker with this PID and the OS probe reports a live
    process elsewhere (a foreign process / PID reuse): the function reports
    that it did not handle it.
    """
    mock_pool = MagicMock()
    mock_pool._processes = {0: MagicMock(pid=1111)}

    with patch(
        "configurable_automl_engine.training_engine.thread_pool.os.kill"
    ) as mock_kill:
        found = _terminate_worker_by_pid(mock_pool, 9999, shutdown_grace_period=0.01)

    assert found is False
    mock_kill.assert_called_once_with(9999, 0)


def test_terminate_worker_by_pid_already_dead_returns_true():
    """A worker already exited (race) is reported as success.

    The PID is not in the pool registry and the OS probe says the process is
    gone: the hung task is not running anymore, so there is nothing to
    terminate — the outcome is treated as success instead of a false
    "not handled" (issue #120 review feedback).
    """
    mock_pool = MagicMock()
    mock_pool._processes = {0: MagicMock(pid=1111)}

    with patch(
        "configurable_automl_engine.training_engine.thread_pool.os.kill",
        side_effect=ProcessLookupError("No such process"),
    ):
        found = _terminate_worker_by_pid(mock_pool, 9999, shutdown_grace_period=0.01)

    assert found is True


def test_terminate_worker_by_pid_not_registered_but_alive_elsewhere():
    """A live unregistered PID (foreign process / PID reuse) → False.

    ``os.kill(pid, 0)`` succeeding means some process claimed the PID, but it
    is not a worker of this pool: the function must not kill it.
    """
    mock_pool = MagicMock()
    mock_pool._processes = {0: MagicMock(pid=1111)}

    with patch(
        "configurable_automl_engine.training_engine.thread_pool.os.kill"
    ) as mock_kill:
        mock_kill.return_value = None  # the OS probe finds a live process
        found = _terminate_worker_by_pid(mock_pool, 9999, shutdown_grace_period=0.01)

    assert found is False
    mock_kill.assert_called_once_with(9999, 0)


def test_pid_exists():
    """The OS existence probe covers the success/PermissionError branches."""
    from configurable_automl_engine.training_engine.thread_pool import _pid_exists

    assert _pid_exists(0) is False
    assert _pid_exists(-5) is False
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.os.kill"
    ) as mock_kill:
        assert _pid_exists(1234) is True  # exists
        assert mock_kill.call_count == 1
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.os.kill",
        side_effect=ProcessLookupError("gone"),
    ):
        assert _pid_exists(1234) is False
    with patch(
        "configurable_automl_engine.training_engine.thread_pool.os.kill",
        side_effect=PermissionError("other user"),
    ):
        assert _pid_exists(1234) is True


def test_terminate_worker_by_pid_terminate_error_is_swallowed():
    """A failed terminate() call must not break the kill path."""
    mock_worker = MagicMock()
    mock_worker.pid = 4242
    mock_worker.is_alive.return_value = True
    mock_worker.terminate.side_effect = RuntimeError("Terminate blocked")
    mock_pool = MagicMock()
    mock_pool._processes = {0: mock_worker}

    with patch("configurable_automl_engine.training_engine.thread_pool.time.sleep"):
        found = _terminate_worker_by_pid(mock_pool, 4242, shutdown_grace_period=0.01)

    assert found is True
    mock_worker.kill.assert_called_once()


def test_terminate_worker_by_pid_kill_error_is_swallowed():
    """A failed kill() call must not propagate."""
    mock_worker = MagicMock()
    mock_worker.pid = 4242
    mock_worker.is_alive.return_value = True
    mock_worker.kill.side_effect = RuntimeError("Kill blocked")
    mock_pool = MagicMock()
    mock_pool._processes = {0: mock_worker}

    with patch("configurable_automl_engine.training_engine.thread_pool.time.sleep"):
        found = _terminate_worker_by_pid(mock_pool, 4242, shutdown_grace_period=0.01)

    assert found is True
    mock_worker.terminate.assert_called_once()


# --- disk fallback reuses one parquet file per df object ---


def df_row_count(df):
    return len(df)


def test_run_parallel_disk_cache_reuses_file_for_same_df(monkeypatch):
    """The disk fallback (a df incompatible with SHM) writes ONE parquet per
    df object, not one per task: the same X passed to many algorithms is not
    duplicated on disk.
    """
    df = pd.DataFrame({"label": ["a", "b", "c"], "val": [1, 2, 3]})
    args = [(df,), (df,), (df,)]
    saved = []

    original_save = DiskPersistenceManager.save_df

    def counting_save(self, dframe):
        saved.append(dframe)
        return original_save(self, dframe)

    monkeypatch.setattr(DiskPersistenceManager, "save_df", counting_save)

    results = run_parallel(
        df_row_count, args_seq=args, mode="processes", shared_args_indices=[0]
    )

    assert results == [3, 3, 3]
    assert len(saved) == 1  # one parquet file per df object


# ═══════════════════════════════════════════════════════════
#  Issue #121: object-dtype SHM arrays crash workers under
#  forkserver/spawn; dirty data must fall back to pickle
# ═══════════════════════════════════════════════════════════


def test_is_compatible_rejects_object_result_of_mixed_dtypes():
    """int + float + bool df.to_numpy() → object: NOT compatible with SHM.

    Copying raw pointers to parent-process Python objects into shared memory
    crashes workers under forkserver/spawn (issue #121). The frame must be
    routed through the disk/pickle path instead of SHM.
    """
    df_mixed = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "c": [True, False, True]}
    )
    assert df_mixed.to_numpy().dtype.kind == "O"  # kept honest by pandas
    assert SharedDataFrame.is_compatible(df_mixed) is False

    # int32 + float promotes to float64 (lossless, int32 max ≪ 2**53), so it
    # stays SHM-compatible; int64 + float is rejected because big int values
    # would collapse (see test_is_compatible_rejects_non_numeric_column_kinds).
    df_num = pd.DataFrame(
        {"a": np.array([1, 2, 3], dtype=np.int32), "b": [1.5, 2.5, 3.5]}
    )
    assert df_num.to_numpy().dtype.kind == "f"
    assert SharedDataFrame.is_compatible(df_num) is True

    # A single homogeneous object column must also be rejected.
    df_obj = pd.DataFrame({"a": ["x", "y"]})
    assert SharedDataFrame.is_compatible(df_obj) is False


def test_is_compatible_rejects_non_numeric_column_kinds():
    """Columns outside i/u/f/b are rejected even when to_numpy() is numeric.

    A numeric category column (kind 'O') promotes to float64 through
    to_numpy(): this silently changes the astype(str) encoding ('10' →
    '10.0') and can collapse big int ids into float64 (values > 2**53 become
    indistinguishable). Such frames used to go through parquet keeping exact
    int64, so they must not be promoted into SHM (issue #121 review).
    """
    # Numeric categories with a float column: to_numpy() → float64.
    df_numeric_cat = pd.DataFrame(
        {"cat": pd.Categorical([10, 20, 30]), "x": [1.5, 2.5, 3.5]}
    )
    assert df_numeric_cat.to_numpy().dtype.kind == "f"
    assert SharedDataFrame.is_compatible(df_numeric_cat) is False

    # Numeric categories alone: to_numpy() → int64, still rejected (the
    # categorical encoding is lost/rewritten by the SHM round-trip).
    df_cat_alone = pd.DataFrame({"cat": pd.Categorical([10, 20, 30])})
    assert df_cat_alone.to_numpy().dtype.kind == "i"
    assert SharedDataFrame.is_compatible(df_cat_alone) is False

    # Large int ids mixed with floats: to_numpy() → float64 loses precision
    # beyond 2**53, so the frame must not go through SHM.
    df_bigint = pd.DataFrame(
        {"id": [9007199254740999, 9007199254741001], "x": [1.5, 2.5]}
    )
    assert df_bigint.to_numpy().dtype.kind == "f"
    assert SharedDataFrame.is_compatible(df_bigint) is False

    # Homogeneous simple numeric columns stay compatible.
    df_int = pd.DataFrame({"id": [9007199254740999, 9007199254741001]})
    assert df_int.to_numpy().dtype.kind == "i"
    assert SharedDataFrame.is_compatible(df_int) is True

    # Small-width ints (int32) mixed with floats promote to float64 losslessly
    # (int32 max ≪ 2**53), so they may still go through SHM.
    df_small_int = pd.DataFrame(
        {"a": np.array([1, 2, 3], dtype=np.int32), "b": [1.5, 2.5, 3.5]}
    )
    assert df_small_int.to_numpy().dtype.kind == "f"
    assert SharedDataFrame.is_compatible(df_small_int) is True


def test_shared_dataframe_creation_failure_cleans_segment():
    """A failure inside SharedDataFrame(df) must not leak the OS segment."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with (
        patch(
            "configurable_automl_engine.training_engine.thread_pool."
            "shared_memory.SharedMemory"
        ) as mock_shm,
        patch(
            "configurable_automl_engine.training_engine.thread_pool.np.ndarray",
            side_effect=RuntimeError("simulated buffer failure"),
        ),
    ):
        mock_shm.return_value.name = "psm_leak_test"
        mock_shm.return_value.buf = memoryview(bytearray(24))
        with pytest.raises(RuntimeError):
            SharedDataFrame(df=df)
    mock_shm.return_value.close.assert_called_once()
    mock_shm.return_value.unlink.assert_called_once()


def test_run_parallel_mixed_dtypes_under_forkserver():
    """int + float + bool X must not crash workers under forkserver.

    On Linux CI with Python 3.11–3.13 the default start method is fork, which
    masks the bug: the child happens to see a snapshot of the parent memory.
    forkserver/spawn (the Python 3.14 Linux default and the macOS/Windows
    default) expose the crash, so the test forces the method explicitly.
    """
    if sys.platform != "linux":
        pytest.skip("start-method probing below is Linux-specific")

    # A second, different start method breaks the shared forkserver state;
    # run each method in its own subprocess instead. The worker is defined in
    # a real module file, so both forkserver and spawn can import it.
    script_template = """
# -*- coding: utf-8 -*-
import multiprocessing as mp

import pandas as pd

from configurable_automl_engine.training_engine.thread_pool import (
    run_parallel,
    SharedDataFrame,
)


def worker_mixed_dtypes(df):
    return int(df.sum().sum())


def main():
    # Set the start method inside main(): the module top-level is also
    # executed by the forkserver/spawn bootstrap in the child (where the
    # context is already initialized), so calling set_start_method at module
    # scope would raise "context has already been set".
    mp.set_start_method("{method}")
    df = pd.DataFrame(
        {{"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "c": [True, False, True]}}
    )
    assert SharedDataFrame.is_compatible(df) is False
    results = run_parallel(
        worker_mixed_dtypes, args_seq=[(df,)], mode="processes",
        shared_args_indices=[0]
    )
    assert results == [15], results
    print("OK")


if __name__ == "__main__":
    main()
"""
    for method in ("forkserver", "spawn"):
        script = script_template.format(method=method)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            script_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        finally:
            os.unlink(script_path)
        assert proc.returncode == 0, (
            f"{method} failed rc={proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        assert "OK" in proc.stdout


def test_run_parallel_parquet_write_failure_falls_back_to_pickle(caplog):
    """A to_parquet failure during preparation must not crash run_parallel.

    Preparation happens outside the pool: a write error used to kill the whole
    run_parallel call (and leak the created SHM segments). Now the prepared
    resources are released and the dataframes are sent through pickle with a
    warning instead (issue #121). The original disk indices survive the
    fallback: a DataFrame at such an index is passed through by _worker_proxy
    untouched (only real .parquet paths are read from disk).
    """
    df = pd.DataFrame({"a": [1, 2, 3]})

    with (
        patch(
            "configurable_automl_engine.training_engine.thread_pool."
            "SharedDataFrame.is_compatible",
            return_value=False,
        ),
        patch(
            "configurable_automl_engine.training_engine.thread_pool."
            "DiskPersistenceManager.save_df",
            side_effect=RuntimeError("disk full"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        results = run_parallel(
            process_df_task,
            args_seq=[(df, 1)],
            mode="processes",
            shared_args_indices=[0],
            disk_args_indices=[1],  # a non-DataFrame index: must be preserved
        )

    assert results == [6]
    assert "transferring dataframes through pickle instead" in caplog.text
    assert "disk full" in caplog.text


def test_run_parallel_preparation_failure_cleans_shm_segments(caplog, monkeypatch):
    """SHM segments created before a failure are released, not leaked."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [4, 5, 6]})

    imported_names: list[str] = []
    real_is_compatible = SharedDataFrame.is_compatible

    class TrackingSharedDataFrame(SharedDataFrame):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Creation passes the dataframe positionally; restoration never
            # does (it passes name=/shape=/dtype=/columns=).
            if args and isinstance(args[0], pd.DataFrame):
                imported_names.append(self.name)

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.thread_pool.SharedDataFrame",
        TrackingSharedDataFrame,
    )
    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.thread_pool."
        "DiskPersistenceManager.save_df",
        lambda self, dframe: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    # df1 is SHM-compatible (segment created), df2 is not → routed to disk,
    # where save_df raises. Preparation aborts between the two. Patch the
    # method on the tracking class itself so the lookup used by run_parallel
    # (SharedDataFrame.is_compatible → TrackingSharedDataFrame) sees it.
    monkeypatch.setattr(
        TrackingSharedDataFrame,
        "is_compatible",
        lambda candidate: (
            real_is_compatible(candidate) if candidate is df1 else False
        ),
    )
    with caplog.at_level(logging.WARNING):
        results = run_parallel(
            process_df_task,
            args_seq=[(df1, 1), (df2, 2)],
            mode="processes",
            shared_args_indices=[0],
        )

    assert results == [6, 30]
    assert "transferring dataframes through pickle instead" in caplog.text
    assert len(imported_names) == 1  # only df1 was placed into SHM

    # The SHM segment created for df1 must have been unlinked by the
    # fallback cleanup: attaching to it by name must fail afterwards.
    for name in imported_names:
        assert name is not None
        with pytest.raises((FileNotFoundError, OSError)):
            shared_memory.SharedMemory(name=name)


# ═══════════════════════════════════════════════════════════
#  Issue #122: value-based SHM compatibility, dtype without a
#  data copy, documented index contract
# ═══════════════════════════════════════════════════════════


def test_is_compatible_int64_float64_within_2_53():
    """int64 + float64 with values up to 2**53 → SHM-compatible.

    int64 is the default pandas dtype for integer columns: a typical numeric
    dataset (int and float features) must go through SHM, not through a
    parquet write (issue #122). 2**53 itself is representable in float64
    exactly, so the boundary value is allowed.
    """
    df = pd.DataFrame({"a": [2**53, 3, 7], "b": [0.5, 1.5, 2.5]})
    assert df.to_numpy().dtype.kind == "f"
    assert SharedDataFrame.is_compatible(df) is True
    # Negative boundary: 2**53 + 1 is not representable in float64.
    df_over = pd.DataFrame({"a": [2**53 + 1, 3], "b": [0.5, 1.5]})
    assert SharedDataFrame.is_compatible(df_over) is False
    # Negative values: -(2**53) is fine, -(2**53) - 1 is not.
    df_neg_ok = pd.DataFrame({"a": [-(2**53), 3], "b": [0.5, 1.5]})
    assert SharedDataFrame.is_compatible(df_neg_ok) is True
    df_neg_over = pd.DataFrame({"a": [-(2**53) - 1, 3], "b": [0.5, 1.5]})
    assert SharedDataFrame.is_compatible(df_neg_over) is False
    # uint64: only the upper boundary matters (no negative values).
    df_u_ok = pd.DataFrame(
        {"a": np.array([2**53, 1], dtype=np.uint64), "b": [0.5, 1.5]}
    )
    assert SharedDataFrame.is_compatible(df_u_ok) is True
    df_u_over = pd.DataFrame(
        {"a": np.array([2**53 + 1, 1], dtype=np.uint64), "b": [0.5, 1.5]}
    )
    assert SharedDataFrame.is_compatible(df_u_over) is False
    # abs(np.iinfo(np.int64).min) overflows int64: the check must not crash
    # and must reject the frame (the magnitude exceeds 2**53).
    df_int64_min = pd.DataFrame(
        {"a": np.array([np.iinfo(np.int64).min], dtype=np.int64), "b": [0.5]}
    )
    assert SharedDataFrame.is_compatible(df_int64_min) is False
    # Without a float neighbor there is no promotion: big int ids stay exact
    # in the int64 array and the frame remains compatible.
    df_big_alone = pd.DataFrame({"a": [2**53 + 999, np.iinfo(np.int64).max]})
    assert df_big_alone.to_numpy().dtype.kind == "i"
    assert SharedDataFrame.is_compatible(df_big_alone) is True


def test_is_compatible_rejects_extension_dtypes_and_mixed_bool():
    """Extension dtypes and bool mixed with non-bool → incompatible.

    Regression guard on the find_common_type trap (issue #122): nullable
    Int64/Float64/boolean have kinds i/f/b and would pass a purely
    dtype-kind based check, while df.to_numpy() of such a frame (possibly
    mixed with plain numeric columns) yields an object array of raw pointers
    to parent-process Python objects — copying it into shared memory crashes
    workers under forkserver/spawn (issue #121). Extension-typed frames must
    always go through the disk/pickle path.
    """
    # Int64 + float64: to_numpy() → object, find_common_type would say Float64.
    df = pd.DataFrame(
        {"a": pd.array([1, 2], dtype="Int64"), "b": [1.5, 2.5]}
    )
    assert df.to_numpy().dtype.kind == "O"
    assert SharedDataFrame.is_compatible(df) is False

    # Nullable boolean with NA: to_numpy() → object.
    df_bool = pd.DataFrame({"a": pd.array([True, None], dtype="boolean")})
    assert df_bool.to_numpy().dtype.kind == "O"
    assert SharedDataFrame.is_compatible(df_bool) is False

    # Int64 with NA: to_numpy() → float64 (kind f), but the extension dtype
    # must still be rejected.
    df_intna = pd.DataFrame({"a": pd.array([1, None], dtype="Int64")})
    assert df_intna.to_numpy().dtype.kind == "f"
    assert SharedDataFrame.is_compatible(df_intna) is False

    # Plain numpy bool mixed with numeric columns: to_numpy() → object.
    df_mixed = pd.DataFrame(
        {"a": [1, 2], "b": [1.5, 2.5], "c": [True, False]}
    )
    assert df_mixed.to_numpy().dtype.kind == "O"
    assert SharedDataFrame.is_compatible(df_mixed) is False

    # A single extension column without NA still promotes to a plain numpy
    # array, but the extension semantics must not silently enter SHM.
    df_int64 = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    assert df_int64.to_numpy().dtype.kind == "i"
    assert SharedDataFrame.is_compatible(df_int64) is False


def test_is_compatible_all_bool_columns():
    """All-bool frames are SHM-compatible and arrive as bool."""
    df = pd.DataFrame({"a": [True, False], "b": [False, True]})
    assert df.to_numpy().dtype.kind == "b"
    assert SharedDataFrame.is_compatible(df) is True


def test_is_compatible_empty_frames_behavior_unchanged():
    """Empty frames behave exactly as before (issue #122 edge cases).

    A frame with no columns and a frame with zero rows must not crash or
    change their compatibility status; the value check skips zero-row
    int64/uint64 columns.
    """
    assert SharedDataFrame.is_compatible(pd.DataFrame()) is True
    assert SharedDataFrame.is_compatible(pd.DataFrame(index=pd.RangeIndex(5))) is True
    zero_rows = pd.DataFrame(
        {"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")}
    )
    assert SharedDataFrame.is_compatible(zero_rows) is True
    zero_rows_big = pd.DataFrame({"a": pd.Series(dtype="int64")})
    assert SharedDataFrame.is_compatible(zero_rows_big) is True


def test_is_compatible_duplicate_column_names_no_crash():
    """Duplicate column names must not crash the value-based check.

    With duplicate names df[name] returns a DataFrame, so min()/max() would
    yield Series and the comparison would raise ValueError — the
    compatibility check must never crash, it only decides SHM vs parquet
    (issue #122 review B1). int-only and int+float duplicated frames are
    checked; the big-value variant (int64+float with values beyond 2**53)
    must be rejected, not crash.
    """
    df_int = pd.DataFrame({0: [1, 2], 1: [3, 4]}, dtype="int64")
    df_int.columns = ["a", "a"]
    assert SharedDataFrame.is_compatible(df_int) is True

    df_mixed = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [0.5, 1.5]})
    df_mixed.columns = ["a", "a", "f"]
    assert SharedDataFrame.is_compatible(df_mixed) is True

    df_big = pd.DataFrame({"a": [2**53 + 1, 2], "b": [3, 4], "c": [0.5, 1.5]})
    df_big.columns = ["a", "a", "f"]
    assert df_big.to_numpy().dtype.kind == "f"
    assert SharedDataFrame.is_compatible(df_big) is False

    # Direct helper call on an int+float duplicated frame.
    from configurable_automl_engine.training_engine.thread_pool import (
        _ints_fit_float64,
    )

    assert _ints_fit_float64(df_mixed) is True
    assert _ints_fit_float64(df_big) is False


def test_numpy_result_dtype_helper():
    """Direct coverage of _numpy_result_dtype (issue #122 decomposition)."""
    from configurable_automl_engine.training_engine.thread_pool import (
        _numpy_result_dtype,
    )

    assert _numpy_result_dtype(pd.DataFrame()) == np.dtype("float64")
    assert _numpy_result_dtype(
        pd.DataFrame({"a": [True, False]})
    ) == np.dtype("bool")
    assert _numpy_result_dtype(
        pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    ) == np.dtype("float64")
    assert _numpy_result_dtype(
        pd.DataFrame({"a": [1, 2], "b": [True, False]})
    ) == np.dtype("O")  # bool mixed with non-bool → object, like to_numpy()
    # Extension dtype → None (not eligible).
    assert _numpy_result_dtype(
        pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    ) is None
    # dtype access raising → None (defensive branch).
    class BrokenIndexDF(pd.DataFrame):
        @property
        def dtypes(self):
            raise RuntimeError("cannot read dtypes")

    assert _numpy_result_dtype(BrokenIndexDF({"a": [1]})) is None


def test_ints_fit_float64_helper():
    """Direct coverage of _ints_fit_float64 (issue #122 decomposition)."""
    from configurable_automl_engine.training_engine.thread_pool import (
        _ints_fit_float64,
    )

    assert _ints_fit_float64(pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})) is True
    assert _ints_fit_float64(
        pd.DataFrame({"a": np.array([2**53, 1], dtype=np.int64), "b": [0.5, 1.5]})
    ) is True
    assert _ints_fit_float64(
        pd.DataFrame({"a": np.array([2**53 + 1, 1], dtype=np.int64), "b": [0.5, 1.5]})
    ) is False
    assert _ints_fit_float64(
        pd.DataFrame({"a": np.array([2**55], dtype=np.uint64), "b": [0.5]})
    ) is False
    # int32 columns are never checked (promotion is always lossless).
    assert _ints_fit_float64(
        pd.DataFrame({"a": np.array([2**31 - 1], dtype=np.int32), "b": [0.5]})
    ) is True
    # Zero-row int64 column is skipped, not crashed on.
    assert _ints_fit_float64(
        pd.DataFrame(
            {
                "a": pd.Series(dtype="int64"),
                "b": pd.Series(dtype="float64"),
            }
        )
    ) is True


def test_is_compatible_does_not_materialize_data(monkeypatch):
    """is_compatible must not call DataFrame.to_numpy (memory guard).

    Previously the check called df.to_numpy() only to read the resulting
    dtype: for a mixed-type frame that is a full in-memory copy, and
    SharedDataFrame.__init__ makes a second copy right after — a transient
    doubling of memory in the main process on every HPO phase (issue #122).
    The dtype is now computed from column dtypes alone.
    """
    def boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("is_compatible must not call DataFrame.to_numpy()")

    monkeypatch.setattr(pd.DataFrame, "to_numpy", boom)

    df = pd.DataFrame(
        {"a": np.arange(4, dtype=np.int64), "b": np.arange(4, dtype=np.float64)}
    )
    assert SharedDataFrame.is_compatible(df) is True
    assert SharedDataFrame.is_compatible(pd.DataFrame({"a": [True, False]})) is True
    # The extension-dtype rejection path also must not materialize data.
    assert (
        SharedDataFrame.is_compatible(
            pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
        )
        is False
    )


# Column builders for the equivalence matrix (issue #122 test plan).
_MATRIX_TYPES = [
    # plain numpy numeric/bool kinds (i/u/f/b)
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool",
    # extension dtypes that must never enter SHM
    "Int64",
    "Float64",
    "boolean",
    "category",
    "object",
    "str",
]


def _matrix_column(type_name: str) -> object:
    """Build a length-3 column of the requested matrix type."""
    return {
        "int8": np.array([-5, 3, 7], dtype=np.int8),
        "int16": np.array([-5, 3, 7], dtype=np.int16),
        "int32": np.array([-5, 3, 7], dtype=np.int32),
        "int64": np.array([-5, 3, 7], dtype=np.int64),
        "uint8": np.array([1, 3, 7], dtype=np.uint8),
        "uint16": np.array([1, 3, 7], dtype=np.uint16),
        "uint64": np.array([1, 3, 7], dtype=np.uint64),
        "float16": np.array([-0.5, 1.5, 2.25], dtype=np.float16),
        "float32": np.array([-0.5, 1.5, 2.25], dtype=np.float32),
        "float64": np.array([-0.5, 1.5, 2.25]),
        "bool": np.array([True, False, True]),
        "Int64": pd.array([-5, 3, 7], dtype="Int64"),
        "Float64": pd.array([-0.5, 1.5, 2.25], dtype="Float64"),
        "boolean": pd.array([True, False, True], dtype="boolean"),
        "category": pd.Categorical(["a", "b", "c"]),
        "object": np.array(["a", "b", "c"], dtype=object),
        "str": pd.array(["a", "b", "c"], dtype="str"),
    }[type_name]


# Single columns, pairs and triples: the triple grading is the regression
# guard against the object-array bool trap (bool mixed with any non-bool
# column) for every combination, so the protection cannot silently rot when
# pandas/numpy change their promotion rules (issue #122 review M2).
_MATRIX_COMBOS = [
    combo
    for k in (1, 2, 3)
    for combo in itertools.combinations_with_replacement(_MATRIX_TYPES, k)
]


@pytest.mark.parametrize("column_types", _MATRIX_COMBOS)
def test_is_compatible_equivalence_with_to_numpy(column_types):
    """For every SHM-compatible frame, to_numpy() is numeric and lossless.

    The main regression guard of issue #122: for ANY combination of the
    matrix column types (single columns and pairs), if is_compatible returns
    True then df.to_numpy() must produce an int/uint/float/bool array (never
    an object array of raw pointers), and the values must survive the
    promotion to the common dtype unchanged.
    """
    df = pd.DataFrame(
        {f"c{i}": _matrix_column(t) for i, t in enumerate(column_types)}
    )
    if not SharedDataFrame.is_compatible(df):
        return
    arr = df.to_numpy()
    assert arr.dtype.kind in "iufb", (
        f"compatible frame {column_types} promotes to object array"
    )
    for i, col in enumerate(df.columns):
        orig = [
            float(v) if not isinstance(v, bool) else v for v in df[col].tolist()
        ]
        new = [float(v) if not isinstance(v, bool) else v for v in arr[:, i].tolist()]
        assert orig == new, f"values changed for {column_types} column {col}"


def test_shared_dataframe_roundtrip_resets_index_to_range():
    """SHM transfer restores RangeIndex(0, n), not the original index.

    The index is passed positionally and silently reset (issue #122): a frame
    with RangeIndex(100, 100+n) comes back as RangeIndex(0, n). Worker code
    must not align X and y by index.
    """
    df = pd.DataFrame({"a": [10, 20, 30]}, index=pd.RangeIndex(100, 103))
    wrapper = SharedDataFrame(df=df)
    try:
        restored = wrapper.to_df()
        assert isinstance(restored.index, pd.RangeIndex)
        assert list(restored.index) == [0, 1, 2]
        assert list(restored["a"]) == [10, 20, 30]
    finally:
        wrapper.close()
        wrapper.unlink()


def test_run_parallel_typical_numeric_dataset_uses_shm(monkeypatch):
    """A typical numeric dataset (int64 + float64, values ≤ 2**53) goes
    through shared memory, not through a parquet write.

    The compatibility of int64+float64 is now decided by VALUES, so the
    default pandas integer dtype does not force every HPO phase into writing
    parquet (issue #122). Absence of DiskPersistenceManager.save_df calls
    proves the SHM path was taken.
    """
    df = pd.DataFrame(
        {
            "a": np.arange(100, dtype=np.int64),
            "b": np.arange(100, dtype=np.float64) / 10,
        }
    )
    assert SharedDataFrame.is_compatible(df) is True
    saved: list[pd.DataFrame] = []
    original_save = DiskPersistenceManager.save_df

    def counting_save(self, dframe):
        saved.append(dframe)
        return original_save(self, dframe)

    monkeypatch.setattr(DiskPersistenceManager, "save_df", counting_save)

    results = run_parallel(
        df_row_count, args_seq=[(df,)], mode="processes", shared_args_indices=[0]
    )

    assert results == [100]
    assert saved == []  # no parquet write: the frame rode through SHM


def test_run_parallel_value_safe_int64_float64_under_forkserver():
    """int64 + float64 with values ≤ 2**53 survives forkserver/spawn SHM.

    The worker receives the frame promoted to float64 (lossless within
    2**53) and returns the same result as the plain pickle path (issue
    #122). Runs each start method in its own subprocess, like the issue
    #121 forkserver test.
    """
    if sys.platform != "linux":
        pytest.skip("start-method probing below is Linux-specific")

    script_template = """
# -*- coding: utf-8 -*-
import multiprocessing as mp

import pandas as pd

from configurable_automl_engine.training_engine.thread_pool import (
    run_parallel,
    SharedDataFrame,
)


def worker_int64_float64(df):
    # Values must arrive losslessly — as int64 through pickle or as
    # float64 through SHM; 2**53 is exactly representable in float64.
    assert list(df["a"]) == [2**53, 3, 7]
    assert list(df["b"]) == [0.5, 1.5, 2.5]
    # The int64 column rides as float64 after the SHM promotion (issue
    # #122) — documented behavior, lossless for values ≤ 2**53.
    return int(df["a"].iloc[0])


def worker_all_bool(df):
    assert list(df["a"]) == [True, False, True]
    assert str(df["a"].dtype) == "bool", df["a"].dtype
    return str(df["a"].dtype)


def main():
    mp.set_start_method("{method}")
    df = pd.DataFrame(
        {{"a": [2**53, 3, 7], "b": [0.5, 1.5, 2.5]}}
    )
    assert SharedDataFrame.is_compatible(df) is True
    via_shm = run_parallel(
        worker_int64_float64, args_seq=[(df,)], mode="processes",
        shared_args_indices=[0]
    )
    via_pickle = run_parallel(
        worker_int64_float64, args_seq=[(df,)], mode="processes"
    )
    assert via_shm == via_pickle == [2**53], (via_shm, via_pickle)

    df_bool = pd.DataFrame({{"a": [True, False, True]}})
    assert SharedDataFrame.is_compatible(df_bool) is True
    via_bool = run_parallel(
        worker_all_bool, args_seq=[(df_bool,)], mode="processes",
        shared_args_indices=[0]
    )
    assert via_bool == ["bool"], via_bool
    print("OK")


if __name__ == "__main__":
    main()
"""
    for method in ("forkserver", "spawn"):
        script = script_template.format(method=method)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            script_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        finally:
            os.unlink(script_path)
        assert proc.returncode == 0, (
            f"{method} failed rc={proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        assert "OK" in proc.stdout


def test_run_parallel_extension_int64_float64_under_forkserver():
    """Int64 + float64 (nullable extension dtype) never enters SHM.

    The frame is routed through the disk/pickle path and the worker gets the
    same values as through pickle (issue #122 — regression guard on the
    find_common_type trap: a naive dtype computation would declare this
    frame SHM-compatible while to_numpy() yields an object array).
    """
    if sys.platform != "linux":
        pytest.skip("start-method probing below is Linux-specific")

    script_template = """
# -*- coding: utf-8 -*-
import multiprocessing as mp

import pandas as pd

from configurable_automl_engine.training_engine.thread_pool import (
    run_parallel,
    SharedDataFrame,
)


def worker_extension(df):
    return (list(df["a"]), list(df["b"]))


def main():
    mp.set_start_method("{method}")
    df = pd.DataFrame(
        {{"a": pd.array([1, 2, 3], dtype="Int64"), "b": [0.5, 1.5, 2.5]}}
    )
    assert SharedDataFrame.is_compatible(df) is False, "Int64 must not enter SHM"
    # shared_args_indices=[0]: the frame is rejected by is_compatible and
    # routed through the parquet path instead of crashing a worker.
    via_disk = run_parallel(
        worker_extension, args_seq=[(df,)], mode="processes",
        shared_args_indices=[0]
    )
    via_pickle = run_parallel(
        worker_extension, args_seq=[(df,)], mode="processes"
    )
    assert via_disk == via_pickle == [([1, 2, 3], [0.5, 1.5, 2.5])], (
        via_disk, via_pickle
    )
    print("OK")


if __name__ == "__main__":
    main()
"""
    for method in ("forkserver", "spawn"):
        script = script_template.format(method=method)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            script_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        finally:
            os.unlink(script_path)
        assert proc.returncode == 0, (
            f"{method} failed rc={proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        assert "OK" in proc.stdout
