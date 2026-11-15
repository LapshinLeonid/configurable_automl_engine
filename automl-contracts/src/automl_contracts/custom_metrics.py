"""Structural declaration of user-defined metrics.

The model stores data only: the dotted import path, the calling convention,
the natural direction, additivity and static keyword arguments. Importing the
callable, checking its signature and resolving an sklearn scorer direction
belong to the engine and are deliberately not part of the contract.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Stable code of a malformed custom-metric name.
ERR_CUSTOM_NAME_INVALID = "custom_metric_invalid_name"

#: ``"metric"`` — ``fn(y_true, y_pred, **kwargs) -> float``;
#: ``"scorer"`` — an sklearn scorer ``(estimator, X, y) -> float``.
CustomMetricKind = Literal["metric", "scorer"]


class CustomMetricSpec(BaseModel):
    """The declaration of one custom metric or scorer.

    Attributes:
        path: Dotted import path of the callable (``"pkg.mod:attr"`` or
            ``"pkg.mod.attr"``).
        kind: ``"metric"`` for a ``(y_true, y_pred, **kwargs)`` function;
            ``"scorer"`` for an ``(estimator, X, y)`` scorer.
        greater_is_better: The natural direction of the returned value. It is
            required by the engine for ``kind="metric"`` and checked against
            the scorer direction for sklearn scorers.
        additive: ``True`` when the CV estimate is the mean of per-fold
            values; ``False`` when the pooled out-of-fold value is required.
        kwargs: Static keyword arguments of the metric function.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Dotted import path of the callable: 'package.module:attribute' "
            "(preferred) or 'package.module.attribute'."
        ),
    )
    kind: CustomMetricKind = Field(
        default="metric",
        description=(
            "'metric' — fn(y_true, y_pred, **kwargs) -> float; "
            "'scorer' — sklearn scorer(estimator, X, y) -> float."
        ),
    )
    greater_is_better: bool | None = Field(
        default=None,
        description=(
            "Natural direction of the returned value. Required for "
            "kind='metric'; for a make_scorer scorer it is taken from the "
            "scorer; a plain callable scorer defaults to true."
        ),
    )
    additive: bool = Field(
        default=False,
        description=(
            "true — the CV estimate is the mean of the per-fold values (only "
            "valid for per-sample means such as MAE); false (default) — the "
            "value over the pooled out-of-fold vector."
        ),
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Static keyword arguments passed to the metric function.",
    )

    @property
    def cache_key(self) -> tuple[str, str, bool | None, bool, str]:
        """A hashable identity of the spec (the scorer cache key)."""
        return (
            self.path,
            self.kind,
            self.greater_is_better,
            self.additive,
            json.dumps(self.kwargs, sort_keys=True, default=repr),
        )


__all__ = [
    "ERR_CUSTOM_NAME_INVALID",
    "CustomMetricKind",
    "CustomMetricSpec",
]
