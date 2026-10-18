from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    TypeVar,
)

import numpy as np
from pydantic import BaseModel, Field, model_validator

T = TypeVar("T", str, int, float, bool)


# ───────────────── hyperopt ───────────────── #
class BaseDistribution(BaseModel):
    """Abstract base class for search distributions."""

    type: str


class CategoricalSpace(BaseDistribution, Generic[T]):
    """Distribution for categorical features: [options, 'categorical']."""

    type: Literal["categorical"]
    options: list[T]


class NumericSpace(BaseDistribution):
    """Base class for numeric ranges."""

    low: float
    high: float

    @model_validator(mode="after")
    def _validate_range(self) -> "NumericSpace":
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")
        return self


class FloatSpace(NumericSpace):
    """Distribution for floating-point numbers: [min, max, 'float', step?]."""

    type: Literal["float", "float_log"]
    step: float | None = None

    @staticmethod
    def validate_log_low(low: float) -> None:
        """Validate the lower bound for suitability on a logarithmic scale.

        Optuna requires a strictly positive ``low`` for
        ``trial.suggest_float(..., log=True)``.
        Args:
            low (float): Lower bound of the distribution.
        Raises:
            ValueError: If ``low <= 0``.
        """
        if low <= 0:
            raise ValueError(
                f"low must be > 0 for log-scale distributions. Got low={low}"
            )

    @model_validator(mode="after")
    def _validate_float_constraints(self) -> "FloatSpace":
        if self.type == "float_log" and self.step is not None:
            raise ValueError("The 'step' parameter is not supported for 'float_log'")
        if self.type == "float_log":
            self.validate_log_low(self.low)
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Step must be positive. Got {self.step}")
        return self


class IntSpace(NumericSpace):
    """Distribution for integer numbers: [min, max, 'int', step?]."""

    type: Literal["int"]
    low: int
    high: int
    step: int | None = None

    @model_validator(mode="after")
    def _validate_int_constraints(self) -> "IntSpace":
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Step must be positive. Got {self.step}")
        return self


class SearchSpaceEntry(BaseModel):
    """Description of the search space for an individual hyperparameter.

    A universal container supporting both categorical lists and numeric
    ranges. Implements transparent conversion logic from the shorthand
    YAML notation into typed distribution objects.
    Attributes:
        config (Union[CategoricalSpace, FloatSpace, IntSpace]): Validated
            object of a concrete distribution type.
    """

    config: Annotated[
        CategoricalSpace[Any] | FloatSpace | IntSpace, Field(discriminator="type")
    ]

    @model_validator(mode="before")
    @classmethod
    def _parse_list_to_dict(cls, data: Any) -> Any:
        """Convert a shorthand YAML list entry into a structured Pydantic dict.

        Parsing logic:
        1. If the input is not a list, it is returned "as is" for standard
            processing.
        2. Determines the distribution type by index [1] (categorical) or [2] (numeric).
        3. For numeric types, automatically infers the subtype (int/float)
            when it is not explicitly specified.
        4. Builds the internal dict with `type`, `low`, `high`, `step` keys for
           subsequent model discrimination.
        Args:
            data (Any): Original data from YAML (list or dict).
        Returns:
            Any: A dict ready to initialize a Pydantic model.
        Raises:
            ValueError: If the list contains fewer than 2 elements.
        """
        if not isinstance(data, list):
            return data
        # If this is a single value (string, number) — wrap it as categorical
        if len(data) == 1:
            # Single value → categorical distribution with one option
            return {"config": {"type": "categorical", "options": data}}
        if data[1] == "categorical":
            return {"config": {"type": "categorical", "options": data[0]}}
        if len(data) >= 3:
            dist_type = str(data[2])
            payload = {"type": dist_type, "low": data[0], "high": data[1]}
            if len(data) == 4:
                payload["step"] = data[3]
            return {"config": payload}
        # Infer type for 2-element numeric lists
        if isinstance(data[0], int) and isinstance(data[1], int):
            return {"config": {"type": "int", "low": data[0], "high": data[1]}}
        return {"config": {"type": "float", "low": data[0], "high": data[1]}}

    @property
    def low(self) -> Any:
        return getattr(self.config, "low", None)

    @property
    def high(self) -> Any:
        return getattr(self.config, "high", None)

    @property
    def dist_type(self) -> str:
        return self.config.type

    @property
    def step(self) -> int | float | None:
        return getattr(self.config, "step", None)

    @property
    def bounds(self) -> list[Any]:
        """Backward compatibility alias for the raw list structure."""
        if isinstance(self.config, CategoricalSpace):
            return [self.config.options, "categorical"]
        res = [self.low, self.high, self.dist_type]
        if self.step is not None:
            res.append(self.step)
        return res


# Common parameters for Poisson, Gamma and Tweedie regressors
GLM_COMMON = {
    "alpha": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
    "fit_intercept": SearchSpaceEntry.model_validate([[True, False], "categorical"]),
    "max_iter": SearchSpaceEntry.model_validate([50, 1000, "int", 50]),
}
DEFAULT_SPACES: dict[str, dict[str, SearchSpaceEntry]] = {
    "elasticnet": {
        "alpha": SearchSpaceEntry.model_validate([1e-4, 10.0, "float_log"]),
        "l1_ratio": SearchSpaceEntry.model_validate([0.0, 1.0, "float"]),
    },
    "lasso": {
        "alpha": SearchSpaceEntry.model_validate([1e-4, 10.0, "float_log"]),
    },
    "ridge": {
        "alpha": SearchSpaceEntry.model_validate([1e-4, 10.0, "float_log"]),
    },
    "decision_tree": {
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
    },
    "random_forest": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
        "bootstrap": SearchSpaceEntry.model_validate([[True, False], "categorical"]),
    },
    "extra_trees": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
    },
    "gradient_boosting": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.3, "float_log"]),
        "max_depth": SearchSpaceEntry.model_validate([2, 5, "int"]),
        "subsample": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
    },
    "svr": {
        "C": SearchSpaceEntry.model_validate([1e-2, 100.0, "float_log"]),
        "epsilon": SearchSpaceEntry.model_validate([1e-3, 1.0, "float_log"]),
        # Excluded 'sigmoid' (not strictly positive semi-definite, saturates
        # on large arguments) and 'poly' (prone to degeneracy on small
        # samples) — HPO keeps only the reliable rbf/linear kernels.
        "kernel": SearchSpaceEntry.model_validate([["rbf", "linear"], "categorical"]),
        "gamma": SearchSpaceEntry.model_validate([["scale", "auto"], "categorical"]),
    },
    "xgboosting": {
        "n_estimators": SearchSpaceEntry.model_validate([100, 800, "int", 100]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.3, "float_log"]),
        "max_depth": SearchSpaceEntry.model_validate([3, 10, "int"]),
        "subsample": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "colsample_bytree": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "gamma": SearchSpaceEntry.model_validate([0.0, 5.0, "float"]),
    },
    "sgdregressor": {
        "loss": SearchSpaceEntry.model_validate(
            [
                [
                    "squared_error",
                    "huber",
                    "epsilon_insensitive",
                    "squared_epsilon_insensitive",
                ],
                "categorical",
            ]
        ),
        "penalty": SearchSpaceEntry.model_validate(
            [["l2", "l1", "elasticnet"], "categorical"]
        ),
        "alpha": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "learning_rate": SearchSpaceEntry.model_validate(
            [["constant", "optimal", "invscaling", "adaptive"], "categorical"]
        ),
        "eta0": SearchSpaceEntry.model_validate([1e-4, 1e-1, "float_log"]),
        "l1_ratio": SearchSpaceEntry.model_validate([0.0, 1.0, "float"]),
        "max_iter": SearchSpaceEntry.model_validate([500, 5000, "int", 500]),
    },
    "adaboost": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 1.0, "float_log"]),
        "loss": SearchSpaceEntry.model_validate(
            [["linear", "square", "exponential"], "categorical"]
        ),
    },
    "poissonregressor": {**GLM_COMMON},
    "gammaregressor": {**GLM_COMMON},
    "tweedieregressor": {**GLM_COMMON},
    "glm": {**GLM_COMMON},
    "ardregression": {
        "max_iter": SearchSpaceEntry.model_validate([100, 1000, "int", 100]),
        "alpha_1": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "alpha_2": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "lambda_1": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "lambda_2": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
    },
    "nearest_neighbors_regression": {
        "n_neighbors": SearchSpaceEntry.model_validate([1, 50, "int"]),
        "weights": SearchSpaceEntry.model_validate(
            [["uniform", "distance"], "categorical"]
        ),
        "p": SearchSpaceEntry.model_validate([1, 2, "int"]),
    },
    "isotonic_regression": {
        "increasing": SearchSpaceEntry.model_validate([[True, False], "categorical"]),
    },
    "gaussian_process_regression": {},
}

# A projection of DEFAULT_SPACES keys. An informational summary of parameters
# with default search spaces. NOT used by configuration validation:
# the allowed hyperparameter set is determined by the estimator constructor
# signature (models.get_allowed_hyperparameters), so parameters outside this
# registry (e.g. min_samples_split) are also allowed if the estimator accepts them.
ALGO_HYPERPARAMETER_REGISTRY: dict[str, set[str]] = {
    algo: set(params.keys()) for algo, params in DEFAULT_SPACES.items()
}

DATA_DEPENDENT_CONSTRAINTS: dict[str, str] = {
    "n_neighbors": "n_samples_minus_one",
    "min_samples_leaf": "n_samples",
    "min_samples_split": "n_samples",
}

# ───────────────── adaptive epsilon for SVR ───────────────── #
# The physical meaning of epsilon in SVR is the absolute error in units of
# the target variable y within which the prediction error is ignored.
# A blind range [1e-3, 1.0] degrades on data with a small/large spread of y:
# with σ(y)=0.244, epsilon=0.1 is ~40% of the whole scale, while for prices
# in the millions epsilon=0.1 forces the model to fit every cent.
# Therefore the search bounds are scaled by the spread of y_train.
SVR_EPSILON_REL_MIN = 0.005
"""Lower bound of epsilon as a fraction of the spread of y: 0.005·spread (~0.5% "noise")."""

SVR_EPSILON_REL_MAX = 0.10
"""Upper bound of epsilon as a fraction of the spread of y: 0.10·spread (~10% "noise")."""

SVR_EPSILON_FALLBACK = (1e-4, 1e-2)
"""Fallback bounds for a constant/NaN target or zero spread."""

SVR_EPSILON_MIN_LOW = 1e-8
"""Positive minimum for the lower bound (invariant of ``FloatSpace.validate_log_low``)."""


def compute_svr_epsilon_bounds(
    y: Any,
    *,
    rel_min: float = SVR_EPSILON_REL_MIN,
    rel_max: float = SVR_EPSILON_REL_MAX,
    fallback: tuple[float, float] = SVR_EPSILON_FALLBACK,
    strategy: Literal["std", "iqr"] = "std",
    min_low: float = SVR_EPSILON_MIN_LOW,
) -> tuple[float, float]:
    """Compute adaptive search bounds for ``epsilon`` of an SVR from the spread of ``y``.

    ``epsilon`` in SVR is the absolute error in units of the target variable,
    so a universal blind range does not apply: the bounds are scaled by the
    spread of the training sample (by default σ(y)).

    Features:
    1. The computation runs once per ``optimize()`` call (outside
       ``space_fn``), otherwise each trial would have its own bounds and
       the Optuna TPE sampler would break.
    2. A constant ``y`` (σ = 0), NaN/non-finite values or an empty vector
       lead to fallback bounds without exceptions.
    3. The lower bound is clamped to the positive minimum ``min_low`` —
       an invariant of the logarithmic scale (``FloatSpace.validate_log_low``).
       For abnormally small spread the upper bound is additionally raised to
       preserve ``low < high``.

    Args:
        y (Any): Target vector (np.ndarray, pd.Series,
            pd.DataFrame or an iterable).
        rel_min (float): Multiplier of the lower bound. Defaults to 0.005.
        rel_max (float): Multiplier of the upper bound. Defaults to 0.10.
        fallback (tuple[float, float]): Bounds returned for zero or
            undefined spread. Defaults to (1e-4, 1e-2).
        strategy (Literal["std", "iqr"]): Spread estimation strategy.
            ``'std'`` — standard deviation (default);
            ``'iqr'`` — robust variant via the interquartile range
            (IQR / 1.349, the σ equivalent for a normal distribution),
            resistant to outliers.
        min_low (float): Positive minimum of the lower bound for the
            logarithmic scale. Must be strictly greater than zero.

    Returns:
        tuple[float, float]: Pair ``(low, high)`` of epsilon search bounds
            with the invariants ``low > 0`` and ``low < high``.

    Raises:
        ValueError: If ``strategy`` is not in ``{"std", "iqr"}`` or
            ``min_low <= 0``.
    """
    if strategy not in {"std", "iqr"}:
        raise ValueError(f"Unknown epsilon spread strategy: '{strategy}'")
    if min_low <= 0:
        raise ValueError(f"min_low must be > 0 for log-scale bounds. Got {min_low}")

    arr = np.asarray(y, dtype=float).reshape(-1)
    # Fallback behavior: constant y, NaN/±inf or an empty vector —
    # no exceptions.
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return fallback

    if strategy == "std":
        spread = float(np.std(arr))
    else:  # 'iqr'
        q1, q3 = np.percentile(arr, [25.0, 75.0])
        spread = float(q3 - q1) / 1.349

    if not np.isfinite(spread) or spread <= 0:
        return fallback

    low = rel_min * spread
    high = rel_max * spread
    # Clamp to the positive minimum for float_log (low > 0).
    low = max(low, min_low)
    # For abnormally small spread after clamping, low may exceed high — raise
    # high, keeping low < high (a valid range for suggest_float(log=True)).
    if low >= high:
        high = max(high, low * 2.0)
    return float(low), float(high)


def clip_search_space(space: dict[str, Any], n_samples: int) -> dict[str, Any]:
    """
    Adjust the search space bounds based on the sample size.

    Features:
    1. Avoids mutating global state through a deep copy (Pydantic model_copy).
    2. Guarantees that both low and high do not exceed the physical limit of the data.
    3. Preserves data types (int/float) according to the original configuration.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got {n_samples}")

    # Create a new dict so the original space is not mutated
    clipped_space = {}

    for param, entry in space.items():
        strategy = DATA_DEPENDENT_CONSTRAINTS.get(param)

        # If the parameter does not require clipping or is not a SearchSpaceEntry,
        # copy it as is
        if not strategy or not isinstance(entry, SearchSpaceEntry):
            clipped_space[param] = entry
            continue

        # 1. Compute the physical limit for this parameter
        raw_limit = n_samples - 1 if strategy == "n_samples_minus_one" else n_samples
        limit = float(max(1, raw_limit))  # Guarantee a minimum of 1

        # 2. Deep copy via Pydantic (protection against global state mutation)
        new_entry = entry.model_copy(deep=True)
        config = new_entry.config

        # 3. Clip numeric ranges
        if isinstance(config, (IntSpace, FloatSpace)):
            # First determine the new high (must not exceed limit)
            new_high = min(config.high, limit)

            # Then determine the new low (must not exceed the new high)
            # This solves the "low > limit" crash
            new_low = min(config.low, new_high)

            # 4. Update the config via model_copy to satisfy internal validation
            if isinstance(config, IntSpace):
                new_entry.config = config.model_copy(
                    update={"low": int(new_low), "high": int(new_high)}
                )
            else:
                new_entry.config = config.model_copy(
                    update={"low": new_low, "high": new_high}
                )

        clipped_space[param] = new_entry

    return clipped_space
