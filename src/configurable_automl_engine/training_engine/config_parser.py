"""
AutoML Config Engine: Validation and parsing of hierarchical YAML configs.
The module implements a strictly typed configuration object model
based on Pydantic v2, ensuring integrity checks of experiment parameters,
validation strategies, and hyperparameter search spaces
before the AutoML pipeline is launched.
Key features:
    1. Multi-Stage Pipeline: Configuring a sequence of steps
       (HPO phases 'all_algorithms' / 'refine' and explicit 'select' steps
       with cv_best / sanity_gate strategies) with phase cross-validation.
    2. Intelligent Validation Logic: Automatic consistency checks for
       settings, including the number of folds for the k-fold strategy
       and validation of numeric range boundaries.
    3. Flexible Search Space DSL: Support for a compact YAML syntax
       for defining search spaces (Categorical, Float, Int, Log)
       with automatic type coercion from list-like structures.
    4. Dependency Guard: A built-in mechanism that verifies whether the
       required third-party packages (XGBoost and others) are installed
       for all algorithms enabled in the config, at initialization time.
    5. Data Balancing Schema: Management of oversampling strategies
       (SMOTE, ADASYN) with field aliasing support
       for a cleaner YAML-file structure.

The schema lives in the :mod:`training_engine.config` package, one module
per config section; this module re-exports its names, so the import path
``training_engine.config_parser`` stays the same.
"""

from __future__ import annotations

# The names outside ``__all__`` that callers import from this module are
# re-exported too (marked ``noqa: F401``).
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.feature_selection import (
    FeatureSelectionCfg,
    FeatureSelectionMethod,
    FeatureSelectionMode,
)
from configurable_automl_engine.hpo.default_spaces import (
    SearchSpaceEntry,  # noqa: F401
)
from configurable_automl_engine.training_engine.config.algorithms import (
    ERR_REMOVED_PLUGIN_KEY,  # noqa: F401
    REMOVED_ALGORITHM_KEYS,  # noqa: F401
    AlgoCfg,
    AlgorithmsConfig,  # noqa: F401
    OversamplingAlgorithm,  # noqa: F401
    OversamplingCfg,  # noqa: F401
)
from configurable_automl_engine.training_engine.config.complexity_guard import (
    ComplexityGuardCfg,
    ComplexityGuardMode,
    ComplexityGuardSizeBasis,
    ComplexityGuardUserRangePolicy,
)
from configurable_automl_engine.training_engine.config.general import (
    AUTO_HOLDOUT_FRACTION,  # noqa: F401
    AUTO_HOLDOUT_MIN_ROWS,  # noqa: F401
    ComparisonMetric,  # noqa: F401
    DatetimePolicy,  # noqa: F401
    GeneralCfg,  # noqa: F401
    InfPolicy,  # noqa: F401
)
from configurable_automl_engine.training_engine.config.phases import (
    DEFAULT_PHASE_N_TRIALS,  # noqa: F401
    ERR_REMOVED_REFINE_WINNER,  # noqa: F401
    HPOPhaseAction,
    HPOPhaseCfg,
    PruningCfg,  # noqa: F401
    PruningStrategy,  # noqa: F401
    SelectOutput,
    SelectStrategy,
)
from configurable_automl_engine.training_engine.config.root import (
    Config,
    read_config,
)
from configurable_automl_engine.training_engine.config.sanity_gate import (
    ERR_REMOVED_SANITY_GATE_MODE,  # noqa: F401
    CorridorMode,
    GeneralizationGapMode,
    SanityGateCfg,
    SanityGateMode,  # noqa: F401
)

__all__ = [
    "AlgoCfg",
    "ComplexityGuardCfg",
    "ComplexityGuardMode",
    "ComplexityGuardSizeBasis",
    "ComplexityGuardUserRangePolicy",
    "Config",
    "CorridorMode",
    "FeatureSelectionCfg",
    "FeatureSelectionMethod",
    "FeatureSelectionMode",
    "GeneralizationGapMode",
    "HPOPhaseAction",
    "HPOPhaseCfg",
    "SanityGateCfg",
    "SelectOutput",
    "SelectStrategy",
    "ValidationStrategy",
    "read_config",
]
