from enum import Enum
from typing import Literal

import numpy as np


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


# ──────────────── target transform enum ──────────────── #
class TargetTransform(str, Enum):
    """Target transformations applied inside the model step of the pipeline.

    The transformation wraps the regressor in a
    ``TransformedTargetRegressor``: the model is trained on the transformed
    ``y`` and ``predict`` returns values in the original scale
    (``TargetTransformer.inverse_transform``, with clipping to the
    transformed train range).

    Requirements for ``y``:

    * ``none`` — no transformation; any numeric ``y``;
    * ``log1p`` — ``log(1 + y)``; requires ``min(y) > -1``
      (validated before training, see
      ``preprocessing.target_transform.validate_target_for_transform``);
    * ``boxcox`` — the Box-Cox power transformation; requires strictly
      positive ``y``: ``min(y) > 0`` (validated before training);
    * ``yeo-johnson`` — the Yeo-Johnson power transformation; any real
      ``y`` (no domain restrictions).
    """

    none = "none"
    log1p = "log1p"
    boxcox = "boxcox"
    yeo_johnson = "yeo-johnson"


ALGO_PACKAGE_MAPPING: dict[str, str] = {
    "xgboosting": "xgboost",
}


class ParallelStrategy(str, Enum):
    ALGORITHMS = "algorithms"


# ──────────────── experiment result ──────────────── #
class RunStatus(str, Enum):
    """Outcome status of a whole experiment run (``ExperimentResult.status``)."""

    completed = "completed"
    failed = "failed"


class RunStepStatus(str, Enum):
    """Outcome status of one algorithm record inside a phase."""

    completed = "completed"
    failed = "failed"
    disqualified = "disqualified"


class RunPhaseAction(str, Enum):
    """Pipeline phase action of an ``AlgorithmRun`` record.

    Mirrors ``HPOPhaseAction`` from the config schema
    (``training_engine/config/phases.py``); the legacy 'refine_winner' is
    rejected by the config schema and therefore absent here.
    """

    all_algorithms = "all_algorithms"
    refine = "refine"
    select = "select"


class MetricDirection(str, Enum):
    """Direction of the comparison metric in user semantics.

    ``minimize`` — error metrics (RMSE, MAE, NRMSE and all neg_ scorers
    inverted by ``to_user_value``): a smaller value is better.
    ``maximize`` — score metrics (R² and similar): a larger value is better.
    """

    minimize = "minimize"
    maximize = "maximize"


# ──────────────── worst-score sentinel ──────────────── #
# The lower bound of the "worst-score sentinel" class: the exact float32
# minimum (``np.finfo(np.float32).min``). Any score <= this bound is treated
# as a sentinel — real metrics of such magnitude are practically impossible.
# Comparing against this bound (instead of an exact match against
# HPO_WORST_SCORE via math.isclose) also catches the raw
# float(np.finfo(np.float32).min) value that numpy-cast metrics or custom
# tuners may return. The constant lives here, in the bottom import layer, so
# both the HPO package (``hpo/types.py`` re-exports it) and the experiment
# result model validate scores against one source of truth.
WORST_SCORE_THRESHOLD = float(np.finfo(np.float32).min)
