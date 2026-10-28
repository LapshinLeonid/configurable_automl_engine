"""Module-level task functions for the processes-mode tests of ``run_parallel``.

The functions live in a real module (not in a test file) so that every start
method (``fork``, ``forkserver``, ``spawn``) can import them in the worker.
"""

from __future__ import annotations

import os
import time

import pandas as pd


def add(x: int, y: int = 0) -> int:
    """Return ``x + y``."""
    return x + y


def sleep_then(value: object, seconds: float) -> object:
    """Sleep ``seconds`` and return ``value``."""
    time.sleep(seconds)
    return value


def sleep_or_crash(i: int, crash_idx: int, seconds: float = 0.1) -> int:
    """Kill the worker process for task ``crash_idx``; otherwise return ``i * 10``."""
    if i == crash_idx:
        os._exit(1)
    time.sleep(seconds)
    return i * 10


def always_crash() -> None:
    """Kill the worker process on every attempt."""
    os._exit(3)


def raise_value_error(message: str = "boom") -> None:
    """Raise ``ValueError(message)``."""
    raise ValueError(message)


def raise_keyboard_interrupt() -> None:
    """Raise ``KeyboardInterrupt`` inside the worker."""
    raise KeyboardInterrupt


def frame_summary(df: pd.DataFrame, multiplier: int = 1) -> tuple[list, float]:
    """Return the index and the scaled sum of a frame (checks the transport)."""
    return list(df.index), float(df.select_dtypes("number").sum().sum()) * multiplier


def frame_then(df: pd.DataFrame, action: str) -> object:
    """Use the frame, then succeed, hang, crash or fail (leak checks)."""
    n = len(df)
    if action == "hang":
        time.sleep(30)
    elif action == "crash":
        os._exit(1)
    elif action == "fail":
        raise ValueError("task failed")
    elif action == "interrupt":
        raise KeyboardInterrupt
    return n
