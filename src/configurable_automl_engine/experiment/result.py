"""Engine-facing re-exports of the contract result models.

The definitions live in :mod:`automl_contracts.result`; the engine keeps the
old import path working through re-exports so reporters and stores do not
depend on the contract module layout.
"""

from __future__ import annotations

from automl_contracts.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from automl_contracts.result import (
    AlgorithmRun,
    ExperimentResult,
    ScoreValue,
    ScoreValueOrNone,
)

__all__ = [
    "AlgorithmRun",
    "ExperimentResult",
    "MetricDirection",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
    "ScoreValue",
    "ScoreValueOrNone",
]
