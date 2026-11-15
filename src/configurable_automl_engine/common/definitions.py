"""Engine-facing re-exports of the contract definitions.

The definitions themselves live in :mod:`automl_contracts.definitions`. Only
the runtime package mapping stays in the engine.
"""

from __future__ import annotations

from automl_contracts.definitions import (
    AVAILABLE_ALGORITHMS,
    WORST_SCORE_THRESHOLD,
    EncodingStrategy,
    MetricDirection,
    ParallelStrategy,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
    SerializationFormat,
    TargetTransform,
    ValidationStrategy,
)

#: Runtime metadata: the algorithm name → optional dependency package.
ALGO_PACKAGE_MAPPING: dict[str, str] = {
    "xgboosting": "xgboost",
}

__all__ = [
    "ALGO_PACKAGE_MAPPING",
    "AVAILABLE_ALGORITHMS",
    "WORST_SCORE_THRESHOLD",
    "EncodingStrategy",
    "MetricDirection",
    "ParallelStrategy",
    "RunPhaseAction",
    "RunStatus",
    "RunStepStatus",
    "SerializationFormat",
    "TargetTransform",
    "ValidationStrategy",
]
