"""Experiment result models, run history, leaderboard, reporters, the
object facade and the model explainability helpers.

The public import paths are :mod:`configurable_automl_engine.experiment.result`
and this package: ``ExperimentResult``, ``AlgorithmRun``, the statuses, the
run history (``ExperimentStore``), the algorithm ranking table
(``Leaderboard``), the reporter framework (``Reporter``, ``JSONReporter``,
``MarkdownReporter``, ``CompositeReporter``), the optional MLflow adapter
(``MLflowReporter`` — imports ``mlflow`` lazily, only at report time), the
model explainability helpers (``explain_permutation_importance``,
``explain_shap`` — imports ``shap`` lazily, only at explanation time) and
the object facade of an experiment run (``AutoMLExperiment``).
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
from configurable_automl_engine.experiment.ensemble import (
    EnsembleMember,
    EnsembleModel,
    EnsembleModelError,
    compute_weights,
    resolve_metric_direction,
    select_top_k,
    winner_vs_ensemble_comparison,
)
from configurable_automl_engine.experiment.experiment import (
    AutoMLExperiment,
    ExperimentNotRunError,
)
from configurable_automl_engine.experiment.explain import (
    ExplainNotAvailableError,
    FeatureImportance,
    PermutationImportanceResult,
    ShapFeatureImportance,
    aggregate_to_original_features,
    explain_permutation_importance,
    explain_shap,
    render_importance_markdown,
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
    "AutoMLExperiment",
    "CompositeReporter",
    "DuplicateRunError",
    "EnsembleMember",
    "EnsembleModel",
    "EnsembleModelError",
    "ExperimentNotRunError",
    "ExperimentResult",
    "ExperimentStore",
    "ExperimentStoreError",
    "ExplainNotAvailableError",
    "ExportError",
    "FeatureImportance",
    "JSONReporter",
    "Leaderboard",
    "LeaderboardEntry",
    "MLflowReporter",
    "MLflowReporterError",
    "MarkdownReporter",
    "MetricDirection",
    "PermutationImportanceResult",
    "Reporter",
    "ReporterArtifact",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
    "RunSummary",
    "ShapFeatureImportance",
    "aggregate_to_original_features",
    "build_leaderboard",
    "compute_weights",
    "explain_permutation_importance",
    "explain_shap",
    "export_experiment",
    "render_importance_markdown",
    "resolve_metric_direction",
    "select_top_k",
    "winner_vs_ensemble_comparison",
]
