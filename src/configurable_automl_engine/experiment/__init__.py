"""Experiment result models, run history, leaderboard, reporters and the
object facade.

The public import paths are :mod:`configurable_automl_engine.experiment.result`
and this package: ``ExperimentResult``, ``AlgorithmRun``, the statuses, the
run history (``ExperimentStore``, the T4 component), the algorithm
ranking table (``Leaderboard``), the reporter framework
(``Reporter``, ``JSONReporter``, ``MarkdownReporter``,
``CompositeReporter``), the optional MLflow adapter
(``MLflowReporter`` — imports ``mlflow`` lazily, only at report time)
and the object facade of an experiment run (``AutoMLExperiment``).
The models are pure data: the module imports nothing from the engine
layers, so reporters can consume results without importing the training
engine (the facade imports the engine lazily, inside its ``run()``).
"""

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment.experiment import (
    AutoMLExperiment,
    ExperimentNotRunError,
)
from configurable_automl_engine.experiment.leaderboard import (
    EMPTY_LEADERBOARD_MESSAGE,
    FAILED_SUMMARY_MESSAGE,
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard,
)
from configurable_automl_engine.experiment.reporting import (
    CompositeReporter,
    JSONReporter,
    MarkdownReporter,
    MLflowReporter,
    MLflowReporterError,
    Reporter,
    ReporterArtifact,
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
    "EMPTY_LEADERBOARD_MESSAGE",
    "FAILED_SUMMARY_MESSAGE",
    "AlgorithmRun",
    "AutoMLExperiment",
    "CompositeReporter",
    "DuplicateRunError",
    "ExperimentNotRunError",
    "ExperimentResult",
    "ExperimentStore",
    "ExperimentStoreError",
    "JSONReporter",
    "Leaderboard",
    "LeaderboardEntry",
    "MLflowReporter",
    "MLflowReporterError",
    "MarkdownReporter",
    "MetricDirection",
    "Reporter",
    "ReporterArtifact",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
    "RunSummary",
    "build_leaderboard",
]
