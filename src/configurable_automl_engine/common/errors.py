"""Exception types shared by the engines.

Defines ``TrainingError``, raised by ``ModelTrainer`` and its helper modules
(``trainer_scoring``, ``trainer_feature_selection``, ``trainer_params``) and
by ``IsotonicDataTransformer`` (``isotonic``, used by the trainer and by HPO)
for errors related to the data or the training parameters. Its documented
import path is ``configurable_automl_engine.trainer.TrainingError``
(re-exported there). The module must not import any other module of the
package.
"""

from __future__ import annotations


class TrainingError(RuntimeError):
    """Exception raised for errors related to data or training parameters."""
