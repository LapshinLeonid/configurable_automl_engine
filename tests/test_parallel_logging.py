"""Worker log records reach the parent's handlers with every start method.

With ``forkserver`` and ``spawn`` a worker does not inherit the parent's
logging configuration; ``run_parallel`` forwards the records through a queue.
"""

from __future__ import annotations

import logging
import multiprocessing
import os

import pytest

from configurable_automl_engine.training_engine.thread_pool import run_parallel

from . import parallel_tasks as tasks


class _Collect(logging.Handler):
    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def collected():
    """A handler on ``tests.parallel_tasks`` at INFO; ``.quiet`` at WARNING."""
    log = logging.getLogger("tests.parallel_tasks")
    quiet = logging.getLogger("tests.parallel_tasks.quiet")
    handler = _Collect()
    old = (log.level, log.propagate, quiet.level)
    log.setLevel(logging.INFO)
    log.propagate = False  # keep pytest's root handlers out of the count
    quiet.setLevel(logging.WARNING)
    log.addHandler(handler)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(old[0])
        log.propagate = old[1]
        quiet.setLevel(old[2])


@pytest.mark.parametrize("method", ["forkserver", "spawn", "fork"])
def test_worker_records_reach_the_parent_handlers(method, collected):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not available")
    pids = run_parallel(
        tasks.log_records,
        args_seq=[("a",), ("b",)],
        mode="processes",
        max_workers=2,
        mp_context=method,
    )
    assert all(isinstance(p, int) and p != os.getpid() for p in pids)
    messages = sorted(r.getMessage().splitlines()[0] for r in collected.records)
    # INFO and above arrive exactly once per task; DEBUG is below the parent's
    # level, and the WARNING-level ``quiet`` logger drops its INFO record.
    assert messages == sorted(
        f"{kind} {tag}" for tag in "ab" for kind in ("info", "warning", "exception")
    )
    assert {r.process for r in collected.records} == set(pids)
    exc = next(r for r in collected.records if r.getMessage().startswith("exception"))
    assert exc.levelno == logging.ERROR
    assert "ValueError: inside the worker" in exc.getMessage()


def test_handler_level_is_respected(collected):
    collected.setLevel(logging.WARNING)
    run_parallel(tasks.log_records, args_seq=[("a",)], mode="processes")
    assert sorted(r.levelname for r in collected.records) == ["ERROR", "WARNING"]
