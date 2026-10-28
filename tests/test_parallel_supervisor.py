"""Unit tests of ``ProcessPoolSupervisor`` with a fake, in-process pool.

The fake pool runs every submitted call synchronously, so the restart logic
can be driven deterministically (no worker processes, no timing).
"""

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

import pytest

from configurable_automl_engine.training_engine.parallel_supervisor import (
    ProcessPoolSupervisor,
    is_infrastructure_error,
)
from configurable_automl_engine.training_engine.parallel_transport import TaskPayload


def _double(x: int) -> int:
    return 2 * x


class _SyncPool:
    """Runs each call at submit time; ``break_at`` breaks the pool."""

    def __init__(self, break_at: int | None = None) -> None:
        self._processes: dict = {}
        self._break_at = break_at
        self._submitted = 0
        self.broken_futures: list[Future] = []

    def submit(self, fn, *args):
        if self._break_at is not None and self._submitted >= self._break_at:
            # The pool is broken: earlier futures fail, later submits raise.
            for fut in self.broken_futures:
                if not fut.done():
                    fut.set_exception(BrokenProcessPool("worker died"))
            raise BrokenProcessPool("A child process terminated abruptly")
        self._submitted += 1
        fut: Future = Future()
        if self._break_at is not None:
            self.broken_futures.append(fut)  # left pending until the break
        else:
            fut.set_result(fn(*args))
        return fut

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._processes = None

    def terminate_workers(self):
        self.shutdown(wait=False, cancel_futures=True)

    def kill_workers(self):
        self.shutdown(wait=False, cancel_futures=True)


def _payloads(n: int) -> list[TaskPayload]:
    return [TaskPayload((i,), {}, [], {}) for i in range(n)]


def test_pool_broken_while_submitting_resubmits_the_rest():
    """A crash during the submit loop is a crash, not an infrastructure failure."""
    pools = [_SyncPool(break_at=2), _SyncPool()]
    supervisor = ProcessPoolSupervisor(
        _double,
        _payloads(5),
        max_workers=1,
        shutdown_grace_period=0.01,
        pool_factory=lambda *_: pools.pop(0),
    )
    outcomes = supervisor.run()
    assert [o.status for o in outcomes] == ["ok"] * 5
    assert [o.value for o in outcomes] == [0, 2, 4, 6, 8]
    # The two tasks of the broken pool ran twice, the others once.
    assert [o.attempts for o in outcomes] == [2, 2, 1, 1, 1]


def test_replacement_pool_failure_fails_only_unfinished_tasks():
    def factory(*_):
        if not created:
            created.append(1)
            return _SyncPool(break_at=1)
        raise OSError("no more processes")

    created: list[int] = []
    outcomes = ProcessPoolSupervisor(
        _double, _payloads(3), shutdown_grace_period=0.01, pool_factory=factory
    ).run()
    assert [o.status for o in outcomes] == ["failed"] * 3
    assert "no more processes" in outcomes[0].reason()


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (OSError("x"), True),
        (BrokenProcessPool("x"), True),
        (RuntimeError("x"), True),
        (MemoryError(), True),
        (TypeError("x"), False),
        (AttributeError("x"), False),
        (NotImplementedError("x"), False),
        (RecursionError("x"), False),
        (KeyError("x"), False),
    ],
)
def test_is_infrastructure_error(exc, expected):
    assert is_infrastructure_error(exc) is expected
