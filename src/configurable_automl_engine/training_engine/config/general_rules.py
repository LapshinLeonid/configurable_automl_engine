"""Runtime rules and re-exports of the contract cross-field rules.

Structural rules live in :mod:`automl_contracts.config.general_rules`; the
engine keeps the checks that depend on the environment (installed ``joblib``,
platform start methods and multiprocessing resources).
"""

from __future__ import annotations

from automl_contracts.config.general_rules import (
    MULTI_SPLIT_BUDGET_ERROR_THRESHOLD,
    MULTI_SPLIT_BUDGET_WARN_THRESHOLD,
    MULTI_SPLIT_SPLITS_PER_TRIAL_LIMIT,
    check_high_cardinality_encoding,
    check_metric_references,
    check_multi_split,
    check_pruning_consistency,
    deduplicate_additional_metrics,
    exclude_comparison_from_additional,
    validate_phase_pipeline,
)
from automl_contracts.config.phases import HPOPhaseCfg, PruningCfg

from configurable_automl_engine.common.metrics import check_custom_metric_name
from configurable_automl_engine.training_engine.config_runtime import (
    check_n_folds,
    check_start_method,
    check_threads_timeouts,
)

__all__ = [
    "MULTI_SPLIT_BUDGET_ERROR_THRESHOLD",
    "MULTI_SPLIT_BUDGET_WARN_THRESHOLD",
    "MULTI_SPLIT_SPLITS_PER_TRIAL_LIMIT",
    "HPOPhaseCfg",
    "PruningCfg",
    "check_custom_metric_name",
    "check_high_cardinality_encoding",
    "check_metric_references",
    "check_multi_split",
    "check_n_folds",
    "check_pruning_consistency",
    "check_start_method",
    "check_threads_timeouts",
    "deduplicate_additional_metrics",
    "exclude_comparison_from_additional",
    "validate_phase_pipeline",
]
