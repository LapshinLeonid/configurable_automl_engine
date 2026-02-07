"""
training_engine package
~~~~~~~~~~~~~~~~~~~~~~~
AutoML orchestrator и прочее.
"""
from typing import TYPE_CHECKING, Any, Dict

__all__ = ["train_best_model"]

if TYPE_CHECKING:        # ← только для type hints
    pass

def train_best_model(
        *args: Any, 
        **kwargs: Any
        ) -> Dict[str, Any]:   # noqa: D401
    """Lazy-proxy: импортируем component только когда реально зовут функцию."""
    from .component import train_best_model as _tbm
    return _tbm(*args, **kwargs)
