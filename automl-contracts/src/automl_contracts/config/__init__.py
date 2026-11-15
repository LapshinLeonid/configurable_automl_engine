"""Public configuration models of the AutoML contract.

The section modules in this package are stable public import paths. The
``root`` and ``general_rules`` modules are implementation details of the
contract and are not part of the documented API.
"""

from .algorithms import (
    ERR_REMOVED_PLUGIN_KEY,
    AlgoCfg,
    AlgorithmsConfig,
    OversamplingAlgorithm,
    OversamplingCfg,
)
from .complexity_guard import (
    ComplexityGuardCfg,
    ComplexityGuardMode,
    ComplexityGuardSizeBasis,
    ComplexityGuardUserRangePolicy,
)
from .ensemble import EnsembleCfg, EnsembleWeightMethod
from .general import (
    AUTO_HOLDOUT_FRACTION,
    AUTO_HOLDOUT_MIN_ROWS,
    ComparisonMetric,
    DatetimePolicy,
    GeneralCfg,
    InfPolicy,
    SplitAggregation,
)
from .phases import (
    DEFAULT_PHASE_N_TRIALS,
    ERR_REMOVED_REFINE_WINNER,
    HPOPhaseAction,
    HPOPhaseCfg,
    PruningCfg,
    PruningStrategy,
    SelectOutput,
    SelectStrategy,
)
from .root import Config
from .sanity_gate import (
    ERR_REMOVED_SANITY_GATE_MODE,
    CorridorMode,
    GeneralizationGapMode,
    SanityGateCfg,
)

__all__ = [
    "AUTO_HOLDOUT_FRACTION",
    "AUTO_HOLDOUT_MIN_ROWS",
    "DEFAULT_PHASE_N_TRIALS",
    "ERR_REMOVED_PLUGIN_KEY",
    "ERR_REMOVED_REFINE_WINNER",
    "ERR_REMOVED_SANITY_GATE_MODE",
    "AlgoCfg",
    "AlgorithmsConfig",
    "ComparisonMetric",
    "ComplexityGuardCfg",
    "ComplexityGuardMode",
    "ComplexityGuardSizeBasis",
    "ComplexityGuardUserRangePolicy",
    "Config",
    "CorridorMode",
    "DatetimePolicy",
    "EnsembleCfg",
    "EnsembleWeightMethod",
    "GeneralCfg",
    "GeneralizationGapMode",
    "HPOPhaseAction",
    "HPOPhaseCfg",
    "InfPolicy",
    "OversamplingAlgorithm",
    "OversamplingCfg",
    "PruningCfg",
    "PruningStrategy",
    "SanityGateCfg",
    "SelectOutput",
    "SelectStrategy",
    "SplitAggregation",
]
