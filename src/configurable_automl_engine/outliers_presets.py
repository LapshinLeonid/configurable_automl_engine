"""Outlier-handling specs of the numeric preprocessing (``OutlierSpec``).

Holds the discriminated union behind the ``outliers`` preset field: the
frozen per-method models (:class:`NoOutliers`, :class:`QuantileOutliers`,
:class:`IqrOutliers`), the string-shorthand normalization and the coercion
helpers (:func:`coerce_outliers`, :func:`format_outliers`). The specs are
re-exported by :mod:`configurable_automl_engine.preprocessing_presets`, so
the public import path stays the same. The module is a sibling of
``preprocessing_presets`` (not a member of the ``preprocessing`` package),
because the ``preprocessing`` package ``__init__`` imports the builder,
which imports ``preprocessing_presets`` — a member module would create an
import cycle.

The module follows the same pattern as ``ScalingSpec`` (a ``Tag`` +
:class:`~pydantic.Discriminator` union of frozen models with
``extra="forbid"``, a ``BeforeValidator`` normalizes the string shorthand),
so validation errors are precise: cross-method parameters are rejected in
the branch of their own method, out-of-range values with a clear message.
The clipping itself is performed by
:class:`~configurable_automl_engine.preprocessing.Winsorizer`; the module
must not import the other preprocessing modules.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    model_validator,
)

__all__ = [
    "IqrOutliers",
    "NoOutliers",
    "ObjectOutlierSpec",
    "OutlierSpec",
    "QuantileOutliers",
    "StringOutlierSpec",
    "coerce_outliers",
    "format_outliers",
]


def _coerce_outlier_spec(value: Any) -> Any:
    """Normalize a string shorthand into the object form of an outlier spec.

    Args:
        value: ``"none"`` / ``"quantile"`` / ``"iqr"`` (the string shorthand)
            or an already-object form (a dict or an :class:`OutlierSpec`
            member).

    Returns:
        The object form: ``{"method": <shorthand>}`` for a string, the input
        value otherwise.
    """
    if isinstance(value, str):
        return {"method": value}
    return value


def _discriminate_outlier_method(value: Any) -> Any:
    """Extract the ``method`` discriminator tag of an outlier spec value.

    Args:
        value: An outlier spec in any pre-coercion form (a dict, the object
            form or ``None``/garbage — anything non-dict/object yields
            ``None`` and is rejected by the union).

    Returns:
        The ``method`` tag (``'none'`` / ``'quantile'`` / ``'iqr'``) or
        ``None`` when it cannot be determined.
    """
    if isinstance(value, dict):
        return value.get("method")
    return getattr(value, "method", None)


def _outlier_tag(t: type[BaseModel]) -> str:
    """Return the discriminator tag of an outlier spec model class."""
    return t.model_fields["method"].default  # type: ignore[return-value]


class NoOutliers(BaseModel):
    """No outlier handling: numeric features are passed through as-is.

    The spec is a frozen value object: parameters cannot be mutated after
    creation, so module-level defaults stay safe to share.

    Attributes:
        method: The outlier handling discriminator (always ``"none"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["none"] = Field(
        default="none",
        description="No outlier handling is applied.",
    )


class QuantileOutliers(BaseModel):
    """Quantile (winsorized) clipping of numeric features.

    The bounds are learned from the training data (per CV fold, no leakage)
    by :class:`~configurable_automl_engine.preprocessing.Winsorizer` with
    ``method="quantile"``: every value outside
    ``[lower_quantile, upper_quantile]`` of the training column is clipped
    to the corresponding bound.

    The spec is a frozen value object: parameters cannot be mutated after
    creation, so module-level defaults stay safe to share.

    Attributes:
        method: The outlier handling discriminator (always ``"quantile"``).
        lower_quantile: The lower clipping quantile: ``0 <= lower < upper <= 1``.
        upper_quantile: The upper clipping quantile: ``0 <= lower < upper <= 1``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["quantile"] = Field(
        default="quantile",
        description=(
            "Quantile clipping: values outside the training quantiles are "
            "clipped to the bounds (Winsorizer)."
        ),
    )
    lower_quantile: float = Field(
        default=0.01,
        description="The lower clipping quantile: 0 <= lower < upper <= 1.",
    )
    upper_quantile: float = Field(
        default=0.99,
        description="The upper clipping quantile: 0 <= lower < upper <= 1.",
    )

    @model_validator(mode="after")
    def _validate_quantiles(self) -> QuantileOutliers:
        """Validate the quantile bounds (``0 <= lower < upper <= 1``).

        Returns:
            The validated model instance.

        Raises:
            ValueError: If a quantile is outside ``[0, 1]`` or the bounds
                are not strictly increasing.
        """
        if not (0.0 <= self.lower_quantile <= 1.0):
            raise ValueError(
                f"lower_quantile must be in [0, 1], got {self.lower_quantile}."
            )
        if not (0.0 <= self.upper_quantile <= 1.0):
            raise ValueError(
                f"upper_quantile must be in [0, 1], got {self.upper_quantile}."
            )
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                f"lower_quantile ({self.lower_quantile}) must be strictly "
                f"less than upper_quantile ({self.upper_quantile})."
            )
        return self


class IqrOutliers(BaseModel):
    """IQR-based clipping of numeric features.

    The bounds are learned from the training data (per CV fold, no leakage)
    by :class:`~configurable_automl_engine.preprocessing.Winsorizer` with
    ``method="iqr"``: every value outside
    ``[Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR]`` of the
    training column is clipped to the corresponding bound.

    The spec is a frozen value object: parameters cannot be mutated after
    creation, so module-level defaults stay safe to share.

    Attributes:
        method: The outlier handling discriminator (always ``"iqr"``).
        iqr_multiplier: The IQR multiplier: ``> 0``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["iqr"] = Field(
        default="iqr",
        description=(
            "IQR clipping: values outside Q1 - k * IQR .. Q3 + k * IQR are "
            "clipped to the bounds (Winsorizer)."
        ),
    )
    iqr_multiplier: float = Field(
        default=1.5,
        gt=0.0,
        description="The IQR multiplier of the clipping bounds (must be > 0).",
    )


#: The object form of an outlier-handling configuration: a union of the
#: per-method models discriminated explicitly by the ``method`` field
#: (``Tag`` + :class:`~pydantic.Discriminator`), so validation errors are
#: precise (no cross-method union noise) and the method parameters are
#: checked in the branch of their own method only.
ObjectOutlierSpec = Annotated[
    Annotated[NoOutliers, Tag(_outlier_tag(NoOutliers))]
    | Annotated[QuantileOutliers, Tag(_outlier_tag(QuantileOutliers))]
    | Annotated[IqrOutliers, Tag(_outlier_tag(IqrOutliers))],
    Discriminator(_discriminate_outlier_method),
]

#: String shorthand of an outlier-handling method (e.g. ``"quantile"``):
#: accepted wherever an :class:`OutlierSpec` is expected, and rendered as a
#: plain string (``enum``) alternative in the JSON schema. During validation
#: the shorthand is normalized to its object form by
#: :func:`_coerce_outlier_spec`, so this member effectively only
#: documents/renders the schema alternative.
StringOutlierSpec = Literal["none", "quantile", "iqr"]

OutlierSpec = Annotated[
    ObjectOutlierSpec | StringOutlierSpec,
    BeforeValidator(_coerce_outlier_spec),
    Field(
        description=(
            "Outlier handling for numeric features: a string shorthand "
            "('none' | 'quantile' | 'iqr') or an object "
            "({'method': 'none'} | {'method': 'quantile', "
            "'lower_quantile': float, 'upper_quantile': float} | "
            "{'method': 'iqr', 'iqr_multiplier': float})."
        ),
    ),
]

#: Validated accessor for a single :class:`OutlierSpec` value (used by
#: :func:`coerce_outliers` and the builder).
_OUTLIER_SPEC_ADAPTER: TypeAdapter[ObjectOutlierSpec] = TypeAdapter(ObjectOutlierSpec)

#: Module-level default of the ``outliers`` field: no outlier handling.
#: A frozen singleton, so the shared default cannot be mutated by a caller.
NO_OUTLIERS = NoOutliers()


def coerce_outliers(spec: OutlierSpec) -> ObjectOutlierSpec:
    """Coerce an outlier specification into its object form.

    Args:
        spec: An :class:`OutlierSpec` member, a string shorthand
            (``"none"`` / ``"quantile"`` / ``"iqr"``) or an object-form dict.

    Returns:
        The object form of the outlier spec.

    Raises:
        ValueError: If the value is not a valid outlier spec (delegated to
            Pydantic validation, a :class:`~pydantic.ValidationError`
            subclass).
    """
    coerced: Any = _coerce_outlier_spec(spec)
    if isinstance(coerced, (NoOutliers, QuantileOutliers, IqrOutliers)):
        return coerced
    obj: ObjectOutlierSpec = _OUTLIER_SPEC_ADAPTER.validate_python(coerced)
    return obj


def format_outliers(spec: OutlierSpec) -> str:
    """Format an outlier spec for the preset's log string.

    Args:
        spec: An outlier specification in any accepted form.

    Returns:
        ``'none'`` as-is; for the clipping methods the parameters are shown,
        e.g. ``quantile(0.01, 0.99)`` or ``iqr(k=1.5)``.
    """
    obj = coerce_outliers(spec)
    if isinstance(obj, QuantileOutliers):
        return f"quantile({obj.lower_quantile}, {obj.upper_quantile})"
    if isinstance(obj, IqrOutliers):
        return f"iqr(k={obj.iqr_multiplier})"
    return repr(obj.method)
