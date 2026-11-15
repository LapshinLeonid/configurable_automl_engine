"""Public search-space models of the configuration contract.

The distribution models and :class:`SearchSpaceEntry` describe the canonical
form of ``algorithms.<name>.hyperparameters``. The engine keeps the default
spaces, clipping and type inference in its runtime modules.
"""

from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    TypeVar,
)

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
            low: Lower bound of the distribution.
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
        config: Validated object of a concrete distribution type.
        explicit_type: Whether the distribution type was written explicitly in
            the source config rather than inferred from the boundary numbers.

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
        entry with ``explicit_type=False``), it is preserved as is."""
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
        """Return the entry in the shorthand list form.

        Categorical entries return ``[options, "categorical"]``; numeric
        entries return ``[low, high, type]`` plus ``step`` when it is set.
        """
        if isinstance(self.config, CategoricalSpace):
            return [self.config.options, "categorical"]
        res = [self.low, self.high, self.dist_type]
        if self.step is not None:
            res.append(self.step)
        return res
