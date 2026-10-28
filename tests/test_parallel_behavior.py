"""Behavioral tests of ``run_parallel`` with real worker processes.

The tests do not patch the internals of the wait loop: they run real tasks
(each sleeps at most about a second) and check what the caller sees — the
results, their order, the timeouts, the data transport and the absence of
leaked shared-memory segments and temp files.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from unittest.mock import patch

import pandas as pd
import pytest

from configurable_automl_engine.training_engine import parallel_executor
from configurable_automl_engine.training_engine.thread_pool import (
    MAX_CRASH_ATTEMPTS,
    run_parallel,
    run_parallel_outcomes,
)

from . import parallel_tasks as tasks

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the timing budgets assume POSIX process start"
)

#: The name through which the processes path creates its executor.
POOL_FACTORY_TARGET = f"{parallel_executor.__name__}.ProcessPoolExecutor"
_REAL_POOL = parallel_executor.ProcessPoolExecutor


def _leak_candidates() -> set[str]:
    """Shared-memory segments and Parquet temp files that exist right now."""
    found: set[str] = set()
    if os.path.isdir("/dev/shm"):
        found.update(glob.glob("/dev/shm/psm_*"))
        found.update(glob.glob("/dev/shm/*.parquet"))
    found.update(glob.glob(os.path.join(tempfile.gettempdir(), "*.parquet")))
    return found


@pytest.fixture
def no_leaks() -> Iterator[None]:
    """Fail the test if it leaves a segment or a temp file behind."""
    before = _leak_candidates()
    yield
    leaked = _leak_candidates() - before
    assert not leaked, f"leaked resources: {sorted(leaked)}"


@pytest.fixture
def replacement_pool_fails() -> Iterator[list[int]]:
    """The first process pool is real; creating any later pool fails."""
    created: list[int] = []

    def factory(*args, **kwargs):
        created.append(1)
        if len(created) > 1:
            raise RuntimeError("cannot create the replacement pool")
        return _REAL_POOL(*args, **kwargs)

    with patch(POOL_FACTORY_TARGET, side_effect=factory):
        yield created


# ── results and order ─────────────────────────────────────────────────────


def test_processes_results_keep_input_order(no_leaks):
    """Results follow the input order, not the completion order."""
    # Earlier tasks sleep longer, so they finish last.
    args = [(i, 0.05 * (6 - i)) for i in range(6)]
    results = run_parallel(
        tasks.sleep_then, args_seq=args, mode="processes", max_workers=2
    )
    assert results == list(range(6))


def test_processes_kwargs_are_passed(no_leaks):
    results = run_parallel(
        tasks.add,
        args_seq=[(1,), (2,)],
        kwargs_seq=[{"y": 10}, {"y": 20}],
        mode="processes",
        max_workers=2,
    )
    assert results == [11, 22]


def test_threads_results_keep_input_order():
    args = [(i, 0.02 * (4 - i)) for i in range(4)]
    assert run_parallel(tasks.sleep_then, args_seq=args, mode="threads") == [
        0,
        1,
        2,
        3,
    ]


def test_args_seq_none_calls_func_once():
    calls: list[int] = []

    def once() -> int:
        calls.append(1)
        return 7

    assert run_parallel(once) == [7]
    assert calls == [1]


def test_task_exception_becomes_none_and_others_survive(no_leaks):
    results = run_parallel(
        tasks.raise_value_error,
        args_seq=[("boom",)],
        mode="processes",
    )
    assert results == [None]


# ── timeouts ──────────────────────────────────────────────────────────────


def test_task_timeout_hung_task_is_none_others_complete(no_leaks):
    """A hard timeout drops only the hung task; the queue behind it runs."""
    started = time.time()
    results = run_parallel(
        tasks.sleep_then,
        args_seq=[("hung", 30.0), ("a", 0.1), ("b", 0.1)],
        mode="processes",
        max_workers=1,
        task_timeout=1.0,
        shutdown_grace_period=0.1,
    )
    assert results == [None, "a", "b"]
    assert time.time() - started < 8


def test_task_timeout_counts_from_the_real_start(no_leaks):
    """Queued tasks are not charged for the time they wait in the queue."""
    results = run_parallel(
        tasks.sleep_then,
        args_seq=[(i, 0.3) for i in range(6)],
        mode="processes",
        max_workers=2,
        task_timeout=0.8,
        shutdown_grace_period=0.1,
    )
    assert results == list(range(6))


def test_never_starting_task_is_bounded_by_the_queue_budget(no_leaks):
    started = time.time()
    results = run_parallel(
        tasks.sleep_then,
        args_seq=[("x", 30.0), ("y", 30.0)],
        mode="processes",
        max_workers=1,
        task_timeout=0.5,
        shutdown_grace_period=0.1,
    )
    assert results == [None, None]
    assert time.time() - started < 10


def test_global_timeout_stops_the_run(no_leaks):
    started = time.time()
    results = run_parallel(
        tasks.sleep_then,
        args_seq=[("a", 0.05), ("b", 30.0), ("c", 30.0)],
        mode="processes",
        max_workers=3,
        timeout=1.0,
        shutdown_grace_period=0.1,
    )
    assert results == ["a", None, None]
    assert time.time() - started < 6


def test_threads_timeouts_are_ignored_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        results = run_parallel(
            tasks.sleep_then,
            args_seq=[("done", 0.3)],
            mode="threads",
            timeout=0.05,
            task_timeout=0.05,
        )
    assert results == ["done"]
    assert "Timeouts are ignored in threads mode" in caplog.text


# ── transport ─────────────────────────────────────────────────────────────


def test_shared_memory_transport_is_positional(no_leaks):
    """A frame sent through shared memory arrives with a RangeIndex(0, n)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.RangeIndex(100, 103))
    results = run_parallel(
        tasks.frame_summary,
        args_seq=[(df, 1), (df, 2)],
        mode="processes",
        shared_args_indices=[0],
    )
    assert results == [([0, 1, 2], 6.0), ([0, 1, 2], 12.0)]


def test_disk_transport_round_trips_the_frame(no_leaks):
    df = pd.DataFrame({"label": ["x", "y"], "v": [1, 2]})
    results = run_parallel(
        tasks.frame_summary,
        args_seq=[(df, 3)],
        mode="processes",
        disk_args_indices=[0],
    )
    assert results == [([0, 1], 9.0)]


def test_shm_incompatible_frame_goes_through_disk(no_leaks):
    df = pd.DataFrame({"label": ["x", "y"], "v": [1, 2]})
    results = run_parallel(
        tasks.frame_summary,
        args_seq=[(df, 1)],
        mode="processes",
        shared_args_indices=[0],
    )
    assert results == [([0, 1], 3.0)]


# ── no leaks on any exit path ─────────────────────────────────────────────

_NUMERIC = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
_OBJECT = pd.DataFrame({"s": ["x", "y", "z"]})


@pytest.mark.parametrize("action", ["ok", "hang", "fail", "interrupt"])
def test_no_leaks_on_exit_path(action, no_leaks):
    """Shared-memory segments and Parquet files are freed on every exit path."""
    kwargs = dict(
        args_seq=[(_NUMERIC, action), (_OBJECT, "ok")],
        mode="processes",
        max_workers=2,
        shared_args_indices=[0],
        task_timeout=1.0,
        shutdown_grace_period=0.1,
    )
    if action == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            run_parallel(tasks.frame_then, **kwargs)
        return
    results = run_parallel(tasks.frame_then, **kwargs)
    assert results[1] == 3
    assert results[0] == (3 if action == "ok" else None)


# ── fixed defects: crash recovery, empty input, narrow fallback ───────────


def test_crashed_worker_loses_only_its_task(no_leaks):
    results = run_parallel(
        tasks.sleep_or_crash,
        args_seq=[(i, 1) for i in range(6)],
        mode="processes",
        max_workers=2,
    )
    assert results == [0, None, 20, 30, 40, 50]


def test_empty_args_seq_runs_nothing():
    calls: list[int] = []

    def record() -> int:
        calls.append(1)
        return 1

    assert run_parallel(record, args_seq=[]) == []
    assert calls == []


def test_failed_restart_after_timeout_does_not_rerun_in_threads(
    replacement_pool_fails, no_leaks
):
    """Hard timeout, then the replacement pool fails: the run stays bounded.

    The hung task must not be rerun in a thread (without a timeout), and the
    finished result must be kept.
    """
    started = time.time()
    results = run_parallel(
        tasks.sleep_then,
        args_seq=[("fast", 0.05), ("hung", 4.0), ("queued", 0.05)],
        mode="processes",
        max_workers=1,
        task_timeout=0.7,
        shutdown_grace_period=0.1,
    )
    elapsed = time.time() - started
    assert len(replacement_pool_fails) == 2
    assert results[0] == "fast"
    assert results[1] is None
    assert elapsed < 3.5


# ── outcomes ──────────────────────────────────────────────────────────────


def test_crash_outcome_names_the_crashed_task(no_leaks):
    outcomes = run_parallel_outcomes(
        tasks.sleep_or_crash,
        args_seq=[(i, 1) for i in range(6)],
        mode="processes",
        max_workers=2,
    )
    assert [o.status for o in outcomes] == ["ok", "crashed", "ok", "ok", "ok", "ok"]
    assert [o.value for o in outcomes] == [0, None, 20, 30, 40, 50]
    crashed = outcomes[1]
    assert crashed.attempts == MAX_CRASH_ATTEMPTS
    assert crashed.reason() == "worker process crashed (exit code 1)"


def test_task_crashing_every_attempt_is_given_up(no_leaks):
    started = time.time()
    outcomes = run_parallel_outcomes(
        tasks.always_crash, args_seq=[(), ()], mode="processes", max_workers=2
    )
    assert [o.status for o in outcomes] == ["crashed", "crashed"]
    assert all(o.attempts == MAX_CRASH_ATTEMPTS for o in outcomes)
    assert "exit code 3" in outcomes[0].reason()
    assert time.time() - started < 15


def test_no_leaks_on_crash(no_leaks):
    results = run_parallel(
        tasks.frame_then,
        args_seq=[(_NUMERIC, "crash"), (_OBJECT, "ok")],
        mode="processes",
        max_workers=2,
        shared_args_indices=[0],
    )
    assert results == [None, 3]


def test_outcome_statuses_and_reasons(no_leaks):
    outcomes = run_parallel_outcomes(
        tasks.sleep_then,
        args_seq=[("ok", 0.05), ("hung", 30.0)],
        mode="processes",
        max_workers=2,
        task_timeout=0.5,
        shutdown_grace_period=0.1,
    )
    assert outcomes[0].status == "ok" and outcomes[0].value == "ok"
    assert outcomes[0].duration is not None and outcomes[0].duration < 5
    assert outcomes[1].status == "timed_out"
    assert outcomes[1].reason() == "timed out after 0.5 s"

    failed = run_parallel_outcomes(
        tasks.raise_value_error, args_seq=[("bad input",)], mode="processes"
    )[0]
    assert failed.status == "failed"
    assert failed.reason() == "failed: ValueError: bad input"


def test_global_timeout_cancels_tasks_that_never_started(no_leaks):
    outcomes = run_parallel_outcomes(
        tasks.sleep_then,
        args_seq=[("a", 30.0), ("b", 30.0)],
        mode="processes",
        max_workers=1,
        timeout=0.8,
        shutdown_grace_period=0.1,
    )
    assert [o.status for o in outcomes] == ["timed_out", "cancelled"]
    assert "global timeout of 0.8 s" in outcomes[0].reason()


def test_threads_outcomes():
    outcomes = run_parallel_outcomes(
        tasks.raise_value_error, args_seq=[("x",)], mode="threads"
    )
    assert outcomes[0].status == "failed"
    assert outcomes[0].error is not None and outcomes[0].error.type == "ValueError"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        run_parallel(tasks.add, args_seq=[(1,)], mode="fibers")


# ── fallback policy ───────────────────────────────────────────────────────


def test_pool_creation_failure_falls_back_to_threads(caplog):
    with (
        patch(POOL_FACTORY_TARGET, side_effect=OSError("no semaphores")),
        caplog.at_level(logging.WARNING),
    ):
        results = run_parallel(
            tasks.add, args_seq=[(1, 2), (3, 4)], mode="processes", task_timeout=5
        )
    assert results == [3, 7]
    record = next(r for r in caplog.records if "falling back to threads" in r.message)
    assert record.levelno == logging.WARNING
    assert "timeouts are dropped" in record.message
    assert "no semaphores" in record.message


def test_pool_creation_programming_error_propagates():
    with (
        patch(POOL_FACTORY_TARGET, side_effect=TypeError("bad argument")),
        pytest.raises(TypeError, match="bad argument"),
    ):
        run_parallel(tasks.add, args_seq=[(1, 2)], mode="processes")


def test_failed_restart_marks_unfinished_tasks_failed(replacement_pool_fails, no_leaks):
    outcomes = run_parallel_outcomes(
        tasks.sleep_then,
        args_seq=[("fast", 0.05), ("hung", 4.0), ("queued", 0.05)],
        mode="processes",
        max_workers=1,
        task_timeout=0.7,
        shutdown_grace_period=0.1,
    )
    assert [o.status for o in outcomes] == ["ok", "timed_out", "failed"]
    assert "cannot create the replacement pool" in outcomes[2].reason()


def test_failed_restart_after_crash_keeps_finished_results(
    replacement_pool_fails, no_leaks
):
    outcomes = run_parallel_outcomes(
        tasks.sleep_or_crash,
        args_seq=[(0, 1, 0.01), (1, 1), (2, 1, 0.01)],
        mode="processes",
        max_workers=1,
    )
    assert outcomes[0].status == "ok" and outcomes[0].value == 0
    assert [o.status for o in outcomes[1:]] == ["failed", "failed"]


def test_programming_error_after_start_propagates_without_leaks(no_leaks):
    """An error raised after tasks started is not hidden behind a fallback."""
    created: list[int] = []

    def factory(*args, **kwargs):
        created.append(1)
        if len(created) > 1:
            raise AttributeError("bug in the restart")
        return _REAL_POOL(*args, **kwargs)

    with (
        patch(POOL_FACTORY_TARGET, side_effect=factory),
        pytest.raises(AttributeError, match="bug in the restart"),
    ):
        run_parallel(
            tasks.frame_then,
            args_seq=[(_NUMERIC, "hang"), (_NUMERIC, "ok")],
            mode="processes",
            max_workers=1,
            shared_args_indices=[0],
            task_timeout=0.5,
            shutdown_grace_period=0.1,
        )
