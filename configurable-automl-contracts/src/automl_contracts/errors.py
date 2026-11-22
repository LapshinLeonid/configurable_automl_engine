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


class InvalidPredictionError(TrainingError, ValueError):
    """Exception raised when a model returns non-finite predictions on rows
    it was required to predict.

    Distinguishes two fundamentally different situations of the validation
    estimate:

    - a row did not participate in the validation by the design of the
      chosen split strategy (e.g. the train part of ``train_test_split`` has
      no out-of-fold prediction) — a normal, expected gap in the coverage;
    - the model was REQUIRED to predict a row but returned NaN or Inf —
      a quality defect of the model, not a coverage gap.

    Only the second situation raises this error: scoring silently dropping
    such rows would let a model that fails on most of the validation rows
    win with a good estimate computed on the few rows it happened to
    predict. The callers treat the trial/fold/candidate as invalid and apply
    the existing worst-score/penalty policy; the values are never replaced
    with zeros (that would also distort the metrics).
    """
