"""Engine-facing re-exports of the contract ``general`` configuration models."""

from __future__ import annotations

from automl_contracts.config.general import (
    AUTO_HOLDOUT_FRACTION,
    AUTO_HOLDOUT_MIN_ROWS,
    ComparisonMetric,
    DatetimePolicy,
    GeneralCfg,
    InfPolicy,
    SplitAggregation,
)

__all__ = [
    "AUTO_HOLDOUT_FRACTION",
    "AUTO_HOLDOUT_MIN_ROWS",
    "ComparisonMetric",
    "DatetimePolicy",
    "GeneralCfg",
    "InfPolicy",
    "SplitAggregation",
]
