"""Engine-facing re-exports of the algorithm configuration models.

The models live in the contract; the engine keeps the runtime package mapping
and the helper functions used by ``validate_runtime_config``.
"""

from __future__ import annotations

from automl_contracts.config.algorithms import (
    ERR_REMOVED_PLUGIN_KEY,
    REMOVED_ALGORITHM_KEYS,
    AlgoCfg,
    AlgorithmsConfig,
    AlgorithmsConfigType,
    OversamplingAlgorithm,
    OversamplingCfg,
)

from configurable_automl_engine.common.definitions import ALGO_PACKAGE_MAPPING

__all__ = [
    "ALGO_PACKAGE_MAPPING",
    "ERR_REMOVED_PLUGIN_KEY",
    "REMOVED_ALGORITHM_KEYS",
    "AlgoCfg",
    "AlgorithmsConfig",
    "AlgorithmsConfigType",
    "OversamplingAlgorithm",
    "OversamplingCfg",
]
