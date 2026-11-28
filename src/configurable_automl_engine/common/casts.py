"""Narrowing helpers for values crossing untyped boundaries.

The engine talks to untyped libraries (scikit-learn, Optuna values, raw
configuration payloads), so a few explicit ``typing.cast`` calls are
unavoidable. These helpers give the recurring conversions a name and keep the
casts themselves in one place: callers describe the intended shape instead of
repeating an inline cast at every use site.

``as_float`` and ``as_int`` convert at runtime as well as narrowing the static
type — the numeric values arriving from Optuna trials and search-space models
may be numpy scalars, plain Python numbers or strings. ``as_scorer`` only
narrows the type: a scorer is a duck-typed callable and is never converted.
"""

from __future__ import annotations

from typing import cast

from configurable_automl_engine.common.types import Scorer

__all__ = ["as_float", "as_int", "as_scorer"]


def as_float(value: object) -> float:
    """Convert an untyped numeric value to ``float``.

    Args:
        value: A number, a numpy scalar or a numeric string.

    Returns:
        float: The value converted to a Python ``float``.

    Raises:
        TypeError: If the value cannot be converted to ``float``.
        ValueError: If a string value is not a valid number.
    """
    return float(cast(float | int | str, value))


def as_int(value: object) -> int:
    """Convert an untyped numeric value to ``int``.

    Args:
        value: A number, a numpy scalar or a numeric string.

    Returns:
        int: The value converted to a Python ``int``.

    Raises:
        TypeError: If the value cannot be converted to ``int``.
        ValueError: If a string value is not a valid integer.
    """
    return int(cast(float | int | str, value))


def as_scorer(value: object) -> Scorer:
    """Narrow an untyped value to the sklearn-style scorer type.

    Args:
        value: A scorer object (a ``make_scorer`` result, the engine's
            callable adapter or another duck-typed scorer).

    Returns:
        Scorer: The same object typed as a scorer.
    """
    return cast(Scorer, value)
