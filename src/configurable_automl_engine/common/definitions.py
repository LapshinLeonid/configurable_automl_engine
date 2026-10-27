from enum import Enum
from typing import Literal


# ──────────────── validation enum ──────────────── #
class ValidationStrategy(str, Enum):
    """Allowed validation strategies for hyperopt."""

    train_test_split = "train_test_split"
    k_fold = "k_fold"
    loo = "loo"
    auto = "auto"


# ──────────────── categorical encoding ──────────────── #
#: Categorical encoding strategies of the preprocessor
#: (``preprocessing.SplitCategoricalEncoder``).
EncodingStrategy = Literal["one_hot", "ordinal", "target", "frequency", "hashing"]


# ──────────────── serialization enum ─────────────── #
class SerializationFormat(str, Enum):
    """
    Supported model serialization formats.
    Rationale: Provides typing and a single source of truth for choosing
    between the standard pickle and joblib optimized for heavy weights.
    """

    pickle = "pickle"
    joblib = "joblib"


ALGO_PACKAGE_MAPPING: dict[str, str] = {
    "xgboosting": "xgboost",
}


class ParallelStrategy(str, Enum):
    ALGORITHMS = "algorithms"
