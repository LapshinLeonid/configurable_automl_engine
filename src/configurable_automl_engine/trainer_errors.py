"""The exception type of the single-model trainer.

Defines ``TrainingError``, raised by ``ModelTrainer`` and its helper modules
(``trainer_scoring``, ``trainer_feature_selection``, ``trainer_params``) for
errors related to the data or the training parameters. Its documented import
path is ``configurable_automl_engine.trainer.TrainingError`` (re-exported
there). The module must not import any other module of the package.
"""

from __future__ import annotations


class TrainingError(RuntimeError):
    """Exception raised for errors related to data or training parameters."""
