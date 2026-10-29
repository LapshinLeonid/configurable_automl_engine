"""Unit tests of the deadline policy and the payload transport of ``run_parallel``.

The deadline functions are pure: they are tested on plain numbers. The
transport is tested without worker processes.
"""

from __future__ import annotations

import math
import os
from multiprocessing import shared_memory

import pandas as pd
import pytest

from configurable_automl_engine.training_engine.parallel_deadlines import (
    MIN_WAIT,
    POLL_INTERVAL,
    next_wait,
    queue_budget,
    task_deadline,
)
from configurable_automl_engine.training_engine.parallel_transport import (
    DiskPersistenceManager,
    PayloadTransport,
)

INF = math.inf


# ── queue_budget ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("n_tasks", "workers", "task_timeout", "expected"),
    [
        (10, 2, 3.0, 15.0),  # 5 waves of 3 s
        (11, 2, 3.0, 18.0),  # a partial wave counts as a whole one
        (1, 4, 2.0, 2.0),
        (3, 0, 1.0, 3.0),  # workers below 1 count as 1
        (0, 2, 5.0, 0.0),
    ],
)
def test_queue_budget(n_tasks, workers, task_timeout, expected):
    assert queue_budget(n_tasks, workers, task_timeout) == expected


def test_queue_budget_without_task_timeout_is_unbounded():
    assert queue_budget(10, 2, None) == INF


# ── task_deadline ─────────────────────────────────────────────────────────


def test_task_deadline_without_timeout_is_unbounded():
    assert task_deadline(100.0, 105.0, None, INF) == INF
    assert task_deadline(100.0, None, None, INF) == INF


def test_task_deadline_unstarted_task_gets_the_queue_deadline():
    # submit + task_timeout + queue_budget
    assert task_deadline(100.0, None, 5.0, 20.0) == 125.0


def test_task_deadline_counts_from_the_real_start():
    # Started 10 s after the submit: 5 s of execution from the start.
    assert task_deadline(100.0, 110.0, 5.0, 20.0) == 115.0


def test_task_deadline_is_capped_by_the_queue_deadline():
    # A very late start cannot push the deadline past the queue deadline.
    assert task_deadline(100.0, 124.0, 5.0, 20.0) == 125.0


# ── next_wait ─────────────────────────────────────────────────────────────


def test_next_wait_nothing_bounded_waits_for_completion():
    assert next_wait(0.0, [INF, INF], None, any_unstarted=False) is None
    assert next_wait(0.0, [], None, any_unstarted=False) is None


def test_next_wait_uses_the_nearest_deadline():
    assert next_wait(10.0, [13.0, 12.0, INF], None, any_unstarted=False) == 2.0


def test_next_wait_due_deadline_gets_the_minimum_wait():
    assert next_wait(10.0, [9.0], None, any_unstarted=False) == MIN_WAIT


def test_next_wait_global_deadline_bounds_the_wait():
    assert next_wait(10.0, [20.0], 13.0, any_unstarted=False) == 3.0
    assert next_wait(10.0, [INF], 11.5, any_unstarted=False) == 1.5
    assert next_wait(10.0, [], 9.0, any_unstarted=False) == MIN_WAIT


def test_next_wait_is_capped_by_poll_interval_while_tasks_are_unstarted():
    assert next_wait(0.0, [30.0], None, any_unstarted=True) == POLL_INTERVAL
    # A shorter wait stays shorter than the poll interval.
    assert next_wait(0.0, [0.2], None, any_unstarted=True) == 0.2
    # The cap applies even when no deadline is finite.
    assert next_wait(0.0, [INF], None, any_unstarted=True) == POLL_INTERVAL


# ── PayloadTransport ──────────────────────────────────────────────────────


def _segment_exists(name: str) -> bool:
    try:
        seg = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    seg.close()
    return True


def test_transport_without_indices_passes_arguments_through():
    df = pd.DataFrame({"a": [1.0]})
    with PayloadTransport() as transport:
        payloads = transport.prepare([(df, 1)], [{"k": 2}])
        assert transport.segment_names == []
    assert payloads[0].args[0] is df
    assert payloads[0].kwargs == {"k": 2}
    assert payloads[0].shm_info == {}


def test_transport_one_segment_per_frame_object_freed_on_exit():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with PayloadTransport(shared_indices=[0]) as transport:
        payloads = transport.prepare([(df, 1), (df, 2)], [{}, {}])
        names = transport.segment_names
        assert len(names) == 1
        assert all(p.args[0] is None for p in payloads)
        assert {p.shm_info[0][0] for p in payloads} == set(names)
        assert _segment_exists(names[0])
    assert not _segment_exists(names[0])


def test_transport_disk_cache_one_file_per_frame_freed_on_exit():
    df = pd.DataFrame({"s": ["x", "y"]})  # not SHM-compatible
    with PayloadTransport(
        shared_indices=[0], persistence=DiskPersistenceManager(use_shm=False)
    ) as transport:
        payloads = transport.prepare([(df,), (df,)], [{}, {}])
        files = transport.temp_files
        assert len(files) == 1
        assert payloads[0].args[0] == files[0]
        assert payloads[0].disk_indices == [0]
        assert os.path.exists(files[0])
    assert not os.path.exists(files[0])


def test_transport_frees_resources_on_keyboard_interrupt(monkeypatch):
    df1 = pd.DataFrame({"a": [1.0]})
    df2 = pd.DataFrame({"s": ["x"]})

    def interrupt(self, dframe):
        raise KeyboardInterrupt

    transport = PayloadTransport(shared_indices=[0])
    monkeypatch.setattr(DiskPersistenceManager, "save_df", interrupt)
    with pytest.raises(KeyboardInterrupt), transport:
        transport.prepare([(df1,), (df2,)], [{}, {}])
    assert transport.segment_names == []


def test_transport_failed_preparation_falls_back_to_pickle(monkeypatch, caplog):
    df1 = pd.DataFrame({"a": [1.0]})
    df2 = pd.DataFrame({"s": ["x"]})

    def fail(self, dframe):
        raise RuntimeError("disk full")

    monkeypatch.setattr(DiskPersistenceManager, "save_df", fail)
    with PayloadTransport(shared_indices=[0]) as transport:
        payloads = transport.prepare([(df1,), (df2,)], [{}, {}])
        assert transport.segment_names == []  # the df1 segment was freed
    assert payloads[0].args[0] is df1
    assert payloads[1].args[0] is df2
    assert "transferring dataframes through pickle instead" in caplog.text
