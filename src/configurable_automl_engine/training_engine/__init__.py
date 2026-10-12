"""
training_engine package
~~~~~~~~~~~~~~~~~~~~~~~
AutoML orchestrator and more.
"""

from typing import Any

__all__ = ["train_best_model"]


def train_best_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy proxy: import the component only when the function is actually called."""
    from .component import train_best_model as _tbm

    return _tbm(*args, **kwargs)
