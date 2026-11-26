"""Experiment result models, run history, leaderboard and reporters.

The public import paths are :mod:`configurable_automl_engine.experiment.result`
and this package: ``ExperimentResult``, ``AlgorithmRun``, the statuses, the
run history (``ExperimentStore``), the algorithm ranking table
(``Leaderboard``), the reporter framework (``Reporter``, ``JSONReporter``,
``MarkdownReporter``, ``CompositeReporter``) and the optional MLflow adapter
(``MLflowReporter`` — imports ``mlflow`` lazily, only at report time).
The models are pure data: the module imports nothing from the engine
layers, so reporters can consume results without importing the training
engine.
"""

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment.ensemble import (
    EnsembleMember,
    EnsembleModel,
    EnsembleModelError,
    compute_weights,
    resolve_metric_direction,
    select_top_k,
    winner_vs_ensemble_comparison,
)
from configurable_automl_engine.experiment.export import (
    ExportError,
    export_experiment,
)
from configurable_automl_engine.experiment.leaderboard import (
    COMPLETED_SUMMARY_MESSAGE,
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
    "COMPLETED_SUMMARY_MESSAGE",
    "EMPTY_LEADERBOARD_MESSAGE",
    "FAILED_SUMMARY_MESSAGE",
    "AlgorithmRun",
    "CompositeReporter",
    "DuplicateRunError",
    "EnsembleMember",
    "EnsembleModel",
    "EnsembleModelError",
    "ExperimentResult",
    "ExperimentStore",
    "ExperimentStoreError",
    "ExportError",
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
    "compute_weights",
    "export_experiment",
    "resolve_metric_direction",
    "select_top_k",
    "winner_vs_ensemble_comparison",
]
