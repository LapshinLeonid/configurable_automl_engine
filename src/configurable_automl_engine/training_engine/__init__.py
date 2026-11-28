"""
training_engine package
~~~~~~~~~~~~~~~~~~~~~~~
AutoML orchestrator and more.
"""

from __future__ import annotations

from typing import Protocol, cast

from configurable_automl_engine.common.types import JSONDict

__all__ = ["train_best_model"]


class _TrainBestModel(Protocol):
    """The proxied component entry point."""

    def __call__(self, *args: object, **kwargs: object) -> JSONDict:
        """Train a model and return the result payload."""
        ...  # pragma: no cover - protocol definition


def train_best_model(*args: object, **kwargs: object) -> JSONDict:
    """Lazy proxy: import the component only when the function is actually called."""
    from .component import train_best_model as _tbm

    return cast(_TrainBestModel, _tbm)(*args, **kwargs)
