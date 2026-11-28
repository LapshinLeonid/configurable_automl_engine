"""Tests for the typed narrowing helpers in ``common.casts``."""

from __future__ import annotations

import numpy as np
import pytest

from configurable_automl_engine.common.casts import as_float, as_int, as_scorer


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (1.5, 1.5),
        ("2.25", 2.25),
        (np.float64(3.5), 3.5),
        (np.int64(4), 4.0),
    ],
)
def test_as_float_converts_numeric_values(value: object, expected: float) -> None:
    assert as_float(value) == expected
    assert isinstance(as_float(value), float)


@pytest.mark.parametrize("value", ["not-a-number", None, object()])
def test_as_float_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        as_float(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (1.9, 1),
        ("7", 7),
        (np.float64(3.5), 3),
        (np.int64(4), 4),
    ],
)
def test_as_int_converts_numeric_values(value: object, expected: int) -> None:
    assert as_int(value) == expected
    assert isinstance(as_int(value), int)


@pytest.mark.parametrize("value", ["not-an-int", None, object()])
def test_as_int_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        as_int(value)


def test_as_scorer_returns_the_same_object() -> None:
    def scorer(estimator: object, X: object, y: object) -> float:
        return 1.0

    assert as_scorer(scorer) is scorer
    assert as_scorer("rmse") == "rmse"
