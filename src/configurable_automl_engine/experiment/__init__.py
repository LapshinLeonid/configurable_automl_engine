"""Experiment result models and run history (see :mod:`.result`).

The public import paths are :mod:`configurable_automl_engine.experiment.result`
and this package: ``ExperimentResult``, ``AlgorithmRun``, the statuses, and
the run history (``ExperimentStore``, the T4 component). The models are pure
data: the module imports nothing from the engine layers, so reporters can
consume results without importing the training engine.
"""

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment.result import (
    AlgorithmRun,
    ExperimentResult,
)
from configurable_automl_engine.experiment.store import (
    DuplicateRunError,
    ExperimentStore,
    ExperimentStoreError,
    RunSummary,
)

__all__ = [
    "AlgorithmRun",
    "DuplicateRunError",
    "ExperimentResult",
    "ExperimentStore",
    "ExperimentStoreError",
    "MetricDirection",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
    "RunSummary",
]
