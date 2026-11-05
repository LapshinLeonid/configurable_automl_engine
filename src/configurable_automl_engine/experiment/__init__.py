"""Experiment result models (see :mod:`.result`).

The public import paths are :mod:`configurable_automl_engine.experiment.result`
and this package: ``ExperimentResult``, ``AlgorithmRun`` and the statuses.
The models are pure data: the module imports nothing from the engine layers,
so reporters can consume results without importing the training engine.
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

__all__ = [
    "AlgorithmRun",
    "ExperimentResult",
    "MetricDirection",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
]
