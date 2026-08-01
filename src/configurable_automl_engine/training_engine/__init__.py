"""
training_engine package
~~~~~~~~~~~~~~~~~~~~~~~
AutoML orchestrator и прочее.
"""

from typing import Any

__all__ = ["train_best_model"]


def train_best_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy-proxy: импортируем component только когда реально зовут функцию."""
    from .component import train_best_model as _tbm

    return _tbm(*args, **kwargs)
