"""Test helper: call a pure function twice and check that it behaves purely.

``assert_pure(fn, *args, **kwargs)`` snapshots every argument, calls ``fn``
twice and checks that

- the arguments are unchanged after the calls (``DataFrame`` / ``Series`` /
  ``ndarray`` arguments are compared with a copy taken before the call, other
  values with a deep copy);
- the two calls give equal results (dataclass plans compare with ``==``; the
  frames and arrays they hold are compared value by value).

It returns the result of the first call for further assertions.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import types
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator


_FUNCTIONS = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    functools.partial,
)


def _is_trial(value: Any) -> bool:
    return isinstance(value, (optuna.trial.BaseTrial,))


def _snapshot(value: Any) -> Any:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy(deep=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 — an uncopyable value is compared by identity
        return value


def _fitted(estimator: Any) -> list[str]:
    """The learned attributes of an estimator (sklearn convention: ``x_``)."""
    return sorted(
        k for k in vars(estimator) if k.endswith("_") and not k.startswith("_")
    )


def assert_same(actual: Any, expected: Any, path: str = "value") -> None:
    """Assert that two values are equal (frames, arrays and plans included)."""
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, obj=path)
        return
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, obj=path)
        return
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
        return
    if isinstance(expected, BaseEstimator):
        # Equal type, equal parameters and the same fitted state (the
        # learned ``*_`` attributes): fitting an estimator is a mutation.
        assert type(actual) is type(expected), path
        assert_same(
            actual.get_params(deep=False),
            expected.get_params(deep=False),
            f"{path}.params",
        )
        assert _fitted(actual) == _fitted(expected), f"{path}: fitted state"
        return
    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        assert type(actual) is type(expected), path
        for f in dataclasses.fields(expected):
            assert_same(
                getattr(actual, f.name), getattr(expected, f.name), f"{path}.{f.name}"
            )
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and list(actual) == list(expected), path
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}[{key!r}]")
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, e, f"{path}[{i}]")
        return
    if isinstance(expected, _FUNCTIONS):
        # Functions (e.g. a space function partial) are compared by repr.
        assert repr(actual) == repr(expected), path
        return
    if isinstance(expected, float) and np.isnan(expected):
        assert isinstance(actual, float) and np.isnan(actual), path
        return
    if type(expected).__eq__ is object.__eq__ and expected is not None:
        # Plain objects (CV splitters, pruners): equal type and state.
        assert type(actual) is type(expected), path
        state = getattr(expected, "__dict__", None)
        if state is not None:
            assert_same(vars(actual), state, f"{path}.__dict__")
        else:
            assert repr(actual) == repr(expected), path
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def assert_pure(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` twice; check unchanged inputs and equal results.

    An Optuna ``Trial`` argument is not compared: a trial records the
    suggestions made on it (that is how Optuna works, not a mutation of the
    data the function computes from).
    """
    before_args = [_snapshot(a) for a in args]
    before_kwargs = {k: _snapshot(v) for k, v in kwargs.items()}
    first = fn(*args, **kwargs)
    second = fn(*args, **kwargs)
    for i, (now, then) in enumerate(zip(args, before_args, strict=True)):
        if then is now or _is_trial(now):
            continue
        assert_same(now, then, f"argument {i}")
    for key, then in before_kwargs.items():
        if then is kwargs[key] or _is_trial(kwargs[key]):
            continue
        assert_same(kwargs[key], then, f"argument {key!r}")
    assert_same(second, first, "result")
    return first
