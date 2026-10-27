"""The exception type of the single-model trainer.

Defines ``TrainingError``, raised by ``ModelTrainer`` and its helper modules
(``trainer_scoring``, ``trainer_feature_selection``, ``trainer_params``) for
errors related to the data or the training parameters. The class keeps its
public identity ``configurable_automl_engine.trainer.TrainingError``. The
module must not import any other module of the package.
"""

from __future__ import annotations


class TrainingError(RuntimeError):
    """Exception raised for errors related to data or training parameters."""

    # The public identity of the class: it is documented and imported as
    # ``configurable_automl_engine.trainer.TrainingError`` (re-exported there),
    # and tracebacks and pickles keep that path.
    __module__ = "configurable_automl_engine.trainer"
