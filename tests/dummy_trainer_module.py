"""A minimal custom ModelTrainer for the hold-out score tests.

The class records its constructor calls and predicts the train target mean —
enough to verify that ``_compute_holdout_score`` routes the hold-out
estimation through the winner's ``trainer_module`` (via ``_build_trainer``)
instead of a hardcoded built-in trainer.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np


class ModelTrainer:
    """A dummy trainer that always predicts the training-target mean.

    Attributes:
        instances (list[dict]): The constructor kwargs of every created
            instance (test observability).
        additional_scores (dict[str, float]): Required by the report
            assembly (always empty).
    """

    instances: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Record the constructor kwargs and store the hyperparameters."""
        ModelTrainer.instances.append(kwargs)
        self.hyperparams = dict(kwargs.get("hyperparams") or {})
        self.additional_scores: dict[str, float] = {}
        self.mean_: float | None = None

    def fit(self, X: Any, y: Any) -> ModelTrainer:
        """Store the training-target mean as the constant prediction."""
        self.mean_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the constant train-target mean for every row."""
        return np.full(len(X), self.mean_)

    def save(self, path: Any) -> None:
        """No-op save (the artifact path is not used in the test)."""
