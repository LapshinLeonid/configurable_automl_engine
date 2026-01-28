"""
training_engine package
~~~~~~~~~~~~~~~~~~~~~~~
AutoML orchestrator и прочее.
"""
from typing import TYPE_CHECKING, Any

__all__ = ["train_best_model"]

if TYPE_CHECKING:        # ← только для type hints
    from .component import train_best_model as _tbm

def train_best_model(*args: Any, **kwargs: Any):   # noqa: D401
    """Lazy-proxy: импортируем component только когда реально зовут функцию."""
    from .component import train_best_model as _tbm
    return _tbm(*args, **kwargs)
