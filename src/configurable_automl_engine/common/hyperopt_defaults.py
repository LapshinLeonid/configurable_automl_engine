from decimal import Decimal
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    TypeVar,
)

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

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

    The internal ``explicit_type`` flag records whether the distribution type
    was written explicitly in the source config (``[low, high, "float"]``) or
    inferred from the boundary numbers. Inferred types are re-resolved
    against the parameter type (the default search space or the model
    signature) by :func:`normalize_hyperparameter_entry` — e.g. an
    integral-looking ``[1, 50]`` for a real-valued parameter stays a real
    range.
    """

    config: Annotated[
        CategoricalSpace[Any] | FloatSpace | IntSpace, Field(discriminator="type")
    ]
    explicit_type: Annotated[bool, SkipJsonSchema()] = Field(default=False)

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

        A dict-form entry (``{"config": {"type": "int", "low": 1, "high": 50}}``
        — the canonical ``model_dump()`` output or a handwritten config) whose
        config dict carries an explicit numeric ``type`` is marked as
        ``explicit_type=True``: the type was written explicitly, so
        :func:`normalize_hyperparameter_entry` must not re-resolve it. When
        the flag is already present in the data (a round trip of an inferred
        entry with ``explicit_type=False``), it is preserved as is.

        Args:
            data (Any): Original data from YAML (list or dict).
        Returns:
            Any: A dict ready to initialize a Pydantic model.
        Raises:
            ValueError: If the list contains fewer than 2 elements.
        """
        if not isinstance(data, list):
            if isinstance(data, dict):
                cfg = data.get("config")
                if (
                    isinstance(cfg, dict)
                    and cfg.get("type") in ("int", "float", "float_log")
                    and "explicit_type" not in data
                ):
                    data = {**data, "explicit_type": True}
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
            return {"config": payload, "explicit_type": True}
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


def align_high_to_step(low: float, high: float, step: float) -> float:
    """Align ``high`` to the step grid anchored at ``low``.

    Optuna computes the effective upper bound of a stepped distribution as
    ``low + floor((high - low) / step) * step`` using exact decimal
    arithmetic (``suggest_int``/``suggest_float`` with a step). A range whose
    ``(high - low)`` is not divisible by ``step`` is silently narrowed by
    Optuna, which skews the search away from the configured range. This
    helper applies the same alignment so that the bounds stored in a
    ``SearchSpaceEntry`` always describe the grid Optuna actually searches.

    Args:
        low: The lower bound (the grid anchor; itself stays on the grid).
        high: The upper bound to align.
        step: The grid step; must be positive.

    Returns:
        float: ``low + floor((high - low) / step) * step`` computed with
            exact decimal arithmetic (mirrors Optuna's internal adjustment).
    """
    d_high = Decimal(str(high))
    d_low = Decimal(str(low))
    d_step = Decimal(str(step))
    d_r = d_high - d_low
    d_aligned = (d_r // d_step) * d_step + d_low
    return float(d_aligned)


def align_entry_bounds(entry: SearchSpaceEntry) -> SearchSpaceEntry:
    """Return a deep copy of ``entry`` whose bounds are aligned to the step grid.

    After the bounds of a stepped numeric range are changed (clipping,
    the complexity guard), the upper bound may no longer lie on the step
    grid anchored at the lower bound: e.g. ``n_estimators`` 50–1000 with a
    step of 100 clipped to 50–300 yields ``(300 - 50) % 100 != 0`` and
    Optuna silently shifts the upper bound to 250. This helper re-aligns
    ``high`` to the grid (``high = low + floor((high - low) / step) * step``)
    exactly as Optuna does. When the aligned range becomes empty
    (``aligned_high < low``), the range collapses to the single grid value
    ``low`` — the parameter is fixed, which is the value Optuna would
    otherwise sample for such a degenerate range.

    Entries without a step, categorical entries and constant values are
    returned as a deep copy unchanged (no alignment applies). The original
    entry is never mutated.

    Args:
        entry: The ``SearchSpaceEntry`` (or any object) to align.

    Returns:
        SearchSpaceEntry: A deep copy with the upper bound aligned to the
            step grid (or unchanged for step-less/categorical entries).
    """
    config = entry.config
    if not isinstance(config, (IntSpace, FloatSpace)) or config.step is None:
        return entry.model_copy(deep=True)

    low = float(config.low)
    high = float(config.high)
    step = float(config.step)
    # An empty grid collapses to the anchor value: the range is fixed at the
    # single grid point (the only value Optuna could ever sample).
    aligned_high = max(float(align_high_to_step(low, high, step)), low)

    new_entry = entry.model_copy(deep=True)
    if isinstance(config, IntSpace):
        new_entry.config = config.model_copy(
            update={"low": int(low), "high": int(aligned_high)}
        )
    else:
        new_entry.config = config.model_copy(update={"low": low, "high": aligned_high})
    return new_entry


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
        # L2 penalty acts on the sum of squared residuals (not the mean), so
        # the useful alpha grows with n: on standardized features the optimum
        # is routinely 10–1000 (e.g. ~22 on sklearn's diabetes). The upper
        # bound of 10 used to clamp the search at the boundary.
        "alpha": SearchSpaceEntry.model_validate([1e-3, 1e3, "float_log"]),
    },
    "decision_tree": {
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
    },
    "random_forest": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
        # Fraction of features per split — the main decorrelation knob of a
        # forest. The complexity guard raises the lower bound to 1 / p.
        "max_features": SearchSpaceEntry.model_validate([0.1, 1.0, "float"]),
        "bootstrap": SearchSpaceEntry.model_validate([[True, False], "categorical"]),
    },
    "extra_trees": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
        "max_features": SearchSpaceEntry.model_validate([0.1, 1.0, "float"]),
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
        # A numeric RBF width instead of the "scale"/"auto" heuristics: on
        # standardized features (SVR is scale-sensitive) "scale" is ~1 / p,
        # which this range covers with several decades on both sides.
        # Ignored by the linear kernel.
        "gamma": SearchSpaceEntry.model_validate([1e-4, 10.0, "float_log"]),
    },
    "xgboosting": {
        "n_estimators": SearchSpaceEntry.model_validate([100, 800, "int", 100]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.3, "float_log"]),
        "max_depth": SearchSpaceEntry.model_validate([3, 10, "int"]),
        "min_child_weight": SearchSpaceEntry.model_validate([1, 10, "float"]),
        "subsample": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "colsample_bytree": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "gamma": SearchSpaceEntry.model_validate([0.0, 5.0, "float"]),
        "reg_lambda": SearchSpaceEntry.model_validate([1e-3, 100.0, "float_log"]),
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
    "gaussian_process_regression": {
        # The kernel family of the Gaussian process: translated into a
        # concrete sklearn kernel (amplitude * base + white noise) by
        # `models.create_model`. Kept as strings (not raw kernel objects)
        # so that Optuna can persist categorical trials, the TPE history
        # transfer between HPO phases works, and `trial.params` stays
        # JSON-serializable.
        "kernel_type": SearchSpaceEntry.model_validate(
            [["rbf", "matern", "rational_quadratic"], "categorical"]
        ),
    },
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
    4. Re-aligns the clipped upper bound to the step grid of a stepped range
       (e.g. an int step) so that the stored range is exactly the grid Optuna
       searches; an empty grid collapses to a single value (the parameter is
       fixed).
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

            # 5. Re-align the upper bound to the step grid (a clipped high may
            #    no longer lie on the grid; Optuna would silently narrow it).
            new_entry = align_entry_bounds(new_entry)

        clipped_space[param] = new_entry

    return clipped_space


# Scale parameters: real-valued regularization/learning hyperparameters whose
# magnitude spans orders of magnitude. When the user does not state the type
# explicitly, they are searched on a logarithmic scale by default.
SCALE_PARAMS: frozenset[str] = frozenset(
    {"alpha", "C", "epsilon", "gamma", "learning_rate"}
)


def _default_dist_type(algo_name: str, param_name: str) -> str | None:
    """Return the default distribution type of ``param_name`` for ``algo_name``.

    Consults :data:`DEFAULT_SPACES`. ``None`` when the parameter has no
    default entry or the default entry is categorical.
    """
    entry = DEFAULT_SPACES.get(algo_name, {}).get(param_name)
    if entry is None or entry.dist_type == "categorical":
        return None
    return entry.dist_type


def infer_param_numeric_type(algo_name: str, param_name: str) -> str | None:
    """Infer the numeric type of a model parameter.

    The type is taken from the parameter's entry in :data:`DEFAULT_SPACES`
    when present (``'int'``, ``'float'`` or ``'float_log'``); otherwise it
    falls back to the estimator constructor signature (the type of the
    parameter's default value). Returns ``None`` when the type cannot be
    determined (unknown algorithm, missing dependency, ``**kwargs``
    constructors, or a non-numeric default).

    Args:
        algo_name: Algorithm name.
        param_name: Hyperparameter name.

    Returns:
        ``'int'``, ``'float'``, ``'float_log'``, or ``None`` when unknown.
    """
    default = _default_dist_type(algo_name, param_name)
    if default in ("int", "float", "float_log"):
        return default
    # Lazy import: the estimator factory is only needed for parameters not
    # present in DEFAULT_SPACES, and this module stays lightweight otherwise.
    from configurable_automl_engine.models import get_param_numeric_type

    return get_param_numeric_type(algo_name, param_name)


def resolve_implicit_range_type(
    algo_name: str, param_name: str, low: float, high: float
) -> str | None:
    """Resolve the distribution type of an inferred (not explicit) range.

    The type is taken from the parameter's type — the default search space
    or the model signature — NOT from how the boundary numbers are written.
    A real-valued parameter stays real-valued even when both bounds look like
    integers (e.g. ``[1, 50]`` for ``C`` of an SVR).

    The logarithmic scale (``float_log``) is preferred when the parameter is
    searched on a log scale BY DEFAULT — the default space entry is
    ``float_log`` (e.g. ``alpha_1``/``lambda_1`` of an ARD) or the parameter
    is a scale parameter (``alpha``, ``C``, ``epsilon``, ``gamma``,
    ``learning_rate``) — and both bounds are positive. An explicit type
    (``[1, 50, "int"]``) is always preserved by the caller; with a
    non-positive bound the log scale is unavailable and a plain float range
    is used.

    Args:
        algo_name: Algorithm name.
        param_name: Hyperparameter name.
        low: Lower bound of the range.
        high: Upper bound of the range.

    Returns:
        The resolved distribution type (``'int'``, ``'float'`` or
        ``'float_log'``), or ``None`` when the type cannot be determined and
        the inference from the boundary numbers must be kept.
    """
    numeric_type = infer_param_numeric_type(algo_name, param_name)
    is_int_param = numeric_type == "int"
    wants_log = not is_int_param and (
        param_name in SCALE_PARAMS or numeric_type == "float_log"
    )
    if wants_log:
        # A real parameter with a logarithmic default (either a scale
        # parameter or a default 'float_log' entry from DEFAULT_SPACES):
        # prefer the logarithmic scale. If a bound is not strictly positive,
        # the log scale is unavailable and a plain float range is used.
        if low > 0 and high > 0:
            return "float_log"
        return "float"
    if numeric_type in ("int", "float"):
        return numeric_type
    # The parameter type is unknown — keep the inference from the numbers.
    return None


def normalize_hyperparameter_entry(
    algo_name: str, param_name: str, entry: SearchSpaceEntry
) -> SearchSpaceEntry:
    """Re-resolve the type of an inferred range against the parameter type.

    Only ranges whose type was inferred from the boundary numbers
    (``explicit_type=False``) are re-resolved; explicitly typed entries
    (``[1, 50, "int"]``) are returned unchanged. Categorical entries are
    passed through untouched.

    Args:
        algo_name: Algorithm name.
        param_name: Hyperparameter name.
        entry: The parsed ``SearchSpaceEntry``.

    Returns:
        The possibly re-typed ``SearchSpaceEntry``. The original entry is
        not mutated.
    """
    if entry.explicit_type or not isinstance(entry.config, (FloatSpace, IntSpace)):
        return entry
    low = float(entry.low)
    high = float(entry.high)
    resolved = resolve_implicit_range_type(algo_name, param_name, low, high)
    if resolved is None or resolved == entry.dist_type:
        return entry
    if resolved == "int":
        # An int range with fractional bounds is meaningless; silently
        # truncating [1.5, 2.5] to [1, 2] would change the user's range.
        # Keep the original (float) inference instead.
        if low != int(low) or high != int(high):
            return entry
        new_config: FloatSpace | IntSpace = IntSpace(
            type="int",
            low=int(low),
            high=int(high),
            step=int(entry.step) if entry.step is not None else None,
        )
    elif resolved == "float_log":
        # For a log-scale range a step is meaningless (rejected by the
        # FloatSpace validator), so it is dropped.
        new_config = FloatSpace(type="float_log", low=low, high=high)
    else:  # 'float'
        new_config = FloatSpace(
            type="float",
            low=low,
            high=high,
            step=float(entry.step) if entry.step is not None else None,
        )
    return SearchSpaceEntry.model_construct(
        config=new_config, explicit_type=entry.explicit_type
    )
