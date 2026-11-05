"""Experiment result models, run history and leaderboard (see
:mod:`.result`, :mod:`.store` and :mod:`.leaderboard`).

The public import paths are :mod:`configurable_automl_engine.experiment.result`
and this package: ``ExperimentResult``, ``AlgorithmRun``, the statuses, the
run history (``ExperimentStore``, the T4 component) and the algorithm
ranking table (``Leaderboard``). The models are pure data: the module
imports nothing from the engine layers, so reporters can consume results
without importing the training engine.
"""

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment.leaderboard import (
    EMPTY_LEADERBOARD_MESSAGE,
    FAILED_SUMMARY_MESSAGE,
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard,
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
    "DuplicateRunError",
    "ExperimentResult",
    "ExperimentStore",
    "ExperimentStoreError",
    "Leaderboard",
    "LeaderboardEntry",
    "MetricDirection",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
    "RunSummary",
    "build_leaderboard",
]
