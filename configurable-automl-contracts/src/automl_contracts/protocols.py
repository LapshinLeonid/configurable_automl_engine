"""Public structural interfaces for custom metrics and scorers."""

from __future__ import annotations

from typing import Any, Protocol


class MetricFunction(Protocol):
    """A metric function called as ``fn(y_true, y_pred, **kwargs)``."""

    def __call__(
        self, y_true: Any, y_pred: Any, **kwargs: Any
    ) -> float:  # pragma: no cover - protocol definition
        ...


class ScorerCallable(Protocol):
    """An sklearn-style scorer called as ``scorer(estimator, X, y)``."""

    def __call__(
        self, estimator: Any, X: Any, y: Any
    ) -> float:  # pragma: no cover - protocol definition
        ...


__all__ = ["MetricFunction", "ScorerCallable"]
