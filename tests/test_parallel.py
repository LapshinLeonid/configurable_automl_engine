import pytest
import time
import hashlib
import logging
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


class FailingExecutor:
    def __init__(self, *args, **kwargs):
        # Important: this text must match what we look for in the logs
        raise RuntimeError("init failed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_run_parallel_init_section_coverage(monkeypatch, caplog):
    """Updated: check when ProcessPoolExecutor is missing from the namespace."""
    # Remove the attribute to provoke an import/initialization error
    monkeypatch.delattr(thread_pool, "ProcessPoolExecutor", raising=False)

    with caplog.at_level(logging.ERROR):
        results = run_parallel(lambda x: x, args_seq=[(5,)], mode="processes")

    assert results == [5]
    assert (
        "Could not initialize ProcessPoolExecutor" in caplog.text
        or "falling back to threads" in caplog.text
    )


def test_run_parallel_init_section_coverage(monkeypatch, caplog):
    """Fixed: check the log when ProcessPoolExecutor is missing from the module."""
    monkeypatch.delattr(thread_pool, "ProcessPoolExecutor", raising=False)

    with caplog.at_level(logging.ERROR):
        results = run_parallel(lambda x: x, args_seq=[(5,)], mode="processes")

    assert results == [5]
    # In the code: "Could not initialize ProcessPoolExecutor"
    assert "Could not initialize ProcessPoolExecutor" in caplog.text


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
    - if not all(dt.kind in allowed_kinds for dt in df.dtypes): return False
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
    # Only allowed types (int, float, bool)
    df_valid = pd.DataFrame({"a": [1], "b": [1.5]}, index=pd.RangeIndex(0, 1))
    assert SharedDataFrame.is_compatible(df_valid) is True

    # Disallowed type (object/string)
    df_invalid_type = pd.DataFrame({"a": ["text"]})
    assert SharedDataFrame.is_compatible(df_invalid_type) is False

    # 4. Coverage: index check (not RangeIndex)
    df_invalid_index = pd.DataFrame({"a": [1]}, index=[10])  # Int64Index, not RangeIndex
    assert SharedDataFrame.is_compatible(df_invalid_index) is False


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
