"""Deadline policy of the processes mode of ``run_parallel``.

Pure functions on plain numbers (timestamps and durations in seconds): they
decide when a task has run out of time and how long the wait loop may sleep.
``float("inf")`` means "no deadline".

The rules:
    - The ``task_timeout`` of a task counts from its real start in the worker
      (``started_t``), not from the submit: the time a task waits in the queue
      does not consume its execution budget.
    - A task that never starts is still bounded: its deadline is at most
      ``submit_t + task_timeout + queue_budget``, where the queue budget covers
      the drain of the whole batch.
    - While some task has not started yet, the wait is capped at
      ``POLL_INTERVAL``: the task may start at any moment and its real
      deadline may then be much closer than its queue deadline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

__all__ = [
    "MIN_WAIT",
    "NO_DEADLINE",
    "POLL_INTERVAL",
    "next_wait",
    "queue_budget",
    "task_deadline",
]

#: The longest single wait while some task has not started yet (seconds).
POLL_INTERVAL = 0.5

#: The shortest single wait (seconds): a deadline that is already due is
#: still given a short wait, so a task finishing right now is collected.
MIN_WAIT = 0.1

NO_DEADLINE = math.inf


def queue_budget(n_tasks: int, workers: int, task_timeout: float | None) -> float:
    """The extra time a task may wait in the queue before it must start.

    It covers the drain of the whole batch: ``ceil(n_tasks / workers)`` waves
    of ``task_timeout`` each.

    Args:
        n_tasks (int): The number of tasks of the batch.
        workers (int): The number of worker processes (values below 1 count
            as 1).
        task_timeout (float | None): The per-task timeout; ``None`` — no
            timeout.

    Returns:
        float: The queue budget in seconds, ``inf`` without a task timeout.
    """
    if task_timeout is None:
        return NO_DEADLINE
    return math.ceil(n_tasks / max(workers, 1)) * task_timeout


def task_deadline(
    submit_t: float,
    started_t: float | None,
    task_timeout: float | None,
    queue_budget: float,
) -> float:
    """The absolute deadline of one task.

    Args:
        submit_t (float): When the task was submitted.
        started_t (float | None): When the task really started in the worker;
            ``None`` — it has not started yet.
        task_timeout (float | None): The per-task timeout; ``None`` — no
            timeout.
        queue_budget (float): The result of :func:`queue_budget`.

    Returns:
        float: ``min(started_t + task_timeout, submit_t + task_timeout +
        queue_budget)`` for a started task, the second term for a task that
        has not started, ``inf`` without a task timeout.
    """
    if task_timeout is None:
        return NO_DEADLINE
    queue_deadline = submit_t + task_timeout + queue_budget
    if started_t is None:
        return queue_deadline
    return min(started_t + task_timeout, queue_deadline)


def next_wait(
    now: float,
    deadlines: Iterable[float],
    global_deadline: float | None,
    any_unstarted: bool,
) -> float | None:
    """How long the wait loop may sleep before it must look again.

    Args:
        now (float): The current time.
        deadlines (Iterable[float]): The deadlines of the unfinished tasks.
        global_deadline (float | None): The deadline of the whole run;
            ``None`` — no global timeout.
        any_unstarted (bool): Whether some task with a finite deadline has
            not started yet (the wait is then capped at ``POLL_INTERVAL``).

    Returns:
        float | None: The wait in seconds (at least ``MIN_WAIT``), or ``None``
        to wait until a task finishes.
    """
    wait = min((max(MIN_WAIT, d - now) for d in deadlines), default=NO_DEADLINE)
    if global_deadline is not None:
        wait = min(wait, max(MIN_WAIT, global_deadline - now))
    if any_unstarted:
        wait = min(wait, POLL_INTERVAL)
    return None if math.isinf(wait) else wait
