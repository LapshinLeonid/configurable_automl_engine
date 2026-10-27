"""Test helpers: the trainer's preprocessor and validation scoring, by hand.

The pure ``fit()`` steps of ``ModelTrainer`` take explicit inputs; these
helpers feed them from a (fitted or fresh) trainer's settings, the way
``ModelTrainer.fit`` does.
"""

from __future__ import annotations

from typing import Any

from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.trainer_data import build_trainer_preprocessor
from configurable_automl_engine.trainer_scoring import score_on_validation_splits
from configurable_automl_engine.trainer_state import ValidationScores


def trainer_preprocessor(
    trainer: ModelTrainer,
    feature_names: list[str],
    *,
    categorical: list[str] | None = None,
    numerical: list[str] | None = None,
) -> Any:
    """The unfitted preprocessor ``fit()`` builds for these columns."""
    return build_trainer_preprocessor(
        trainer.algorithm,
        categorical if categorical is not None else trainer.categorical_features or [],
        numerical if numerical is not None else trainer.numerical_features or [],
        feature_names=list(feature_names),
        encoding=trainer.settings.encoding,
        override=trainer.settings.preprocessing_override,
        random_state=trainer.random_state,
    ).preprocessor


def score_with_trainer(
    trainer: ModelTrainer,
    X: Any,
    y: Any,
    preprocessor: Any,
    model: Any,
    *,
    feature_selection_active: bool = False,
) -> ValidationScores:
    """The validation estimate ``fit()`` computes with these components."""
    settings = trainer.settings
    return score_on_validation_splits(
        X,
        y,
        trainer._steps_factory(preprocessor, model),
        feature_selection_active=feature_selection_active,
        metric=settings.metric,
        additional_metrics=settings.additional_metrics,
        validation_strategy=settings.validation_strategy,
        n_folds=settings.n_folds,
        test_size=settings.test_size,
        random_state=settings.random_state,
        algorithm=settings.algorithm,
    )
