"""Public enumerations, literals and constants of the AutoML contract.

The module intentionally depends only on the Python standard library. The
metric names, algorithm names and sentinel values are defined here so that
external consumers can use the contract without installing the engine or its
runtime dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class ValidationStrategy(str, Enum):
    """Allowed validation strategies for hyperparameter optimization."""

    train_test_split = "train_test_split"
    k_fold = "k_fold"
    loo = "loo"
    auto = "auto"


#: Categorical encoding strategies of the preprocessor.
EncodingStrategy = Literal["one_hot", "ordinal", "target", "frequency", "hashing"]


class SerializationFormat(str, Enum):
    """Supported model serialization formats."""

    pickle = "pickle"
    joblib = "joblib"


class TargetTransform(str, Enum):
    """Target transformations supported by the engine.

    ``none`` applies no transformation, ``log1p`` requires ``min(y) > -1``,
    ``boxcox`` requires strictly positive ``y``, ``yeo-johnson`` accepts any
    real target and ``auto`` lets the engine choose from the target
    skewness.
    """

    none = "none"
    log1p = "log1p"
    boxcox = "boxcox"
    yeo_johnson = "yeo-johnson"
    auto = "auto"


class ParallelStrategy(str, Enum):
    """Parallelization strategy of the engine."""

    ALGORITHMS = "algorithms"


class RunStatus(str, Enum):
    """Outcome status of a whole experiment run."""

    completed = "completed"
    failed = "failed"


class RunStepStatus(str, Enum):
    """Outcome status of one algorithm record inside a phase."""

    completed = "completed"
    failed = "failed"
    disqualified = "disqualified"


class RunPhaseAction(str, Enum):
    """Pipeline phase action of an ``AlgorithmRun`` record."""

    all_algorithms = "all_algorithms"
    refine = "refine"
    select = "select"


class MetricDirection(str, Enum):
    """Direction of the comparison metric in user semantics."""

    minimize = "minimize"
    maximize = "maximize"


#: The canonical, read-only algorithm names accepted in ``algorithms``.
#: The list type is preserved for backward compatibility with the engine API.
AVAILABLE_ALGORITHMS: list[str] = [
    "elasticnet",
    "sgdregressor",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "adaboost",
    "poissonregressor",
    "gammaregressor",
    "tweedieregressor",
    "gaussian_process_regression",
    "isotonic_regression",
    "nearest_neighbors_regression",
    "svr",
    "ardregression",
    "glm",
    "ridge",
    "lasso",
    "xgboosting",
]

#: The lower bound of the worst-score sentinel class.
#:
#: The value is the exact minimum of a float32 (formerly expressed through
#: ``numpy.finfo(numpy.float32).min``). Any score at or below this bound is
#: treated as a sentinel: real metrics of such magnitude are practically
#: impossible. Keeping the literal here removes numpy from the contract.
WORST_SCORE_THRESHOLD = -3.4028234663852886e38

__all__ = [
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
