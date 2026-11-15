"""Feature-selection configuration models of the contract."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────── feature selection config ──────────────────── #
class FeatureSelectionMode(str, Enum):
    """Operation modes of the feature space reduction mechanism."""

    disabled = "disabled"  # Feature selection is off (default behavior)
    always = "always"  # Feature selection is always enabled
    auto = "auto"  # Auto mode (Optuna checks whether selection improves the metric)


class FeatureSelectionMethod(str, Enum):
    """Supported algorithms for reducing the feature space."""

    importance = "importance"  # SelectFromModel based on ExtraTreesRegressor
    percentile = "percentile"  # SelectPercentile based on f_regression
    mutual_info = "mutual_info"  # SelectPercentile based on mutual_info_regression
    variance = "variance"  # VarianceThreshold (drops quasi-constant features)


class FeatureSelectionCfg(BaseModel):
    """Settings of the feature selection mechanism.

    Attributes:
        mode: Operation mode: 'disabled', 'always'
            or 'auto'.
        method: Selection algorithm: 'importance',
            'percentile', 'mutual_info' or 'variance'.
        percentile: Percentage of the best features to keep
            (0 < percentile <= 100) for the 'percentile' and 'mutual_info'
            methods.
        min_features: Guaranteed minimum number of features (>= 1),
            protects against an empty feature matrix.
        variance_threshold: Minimum feature variance (>= 0.0)
            for the 'variance' method.
        n_estimators: Number of trees in the ensemble (>= 10)
            for the 'importance' method.
    """

    model_config = ConfigDict(extra="forbid")

    mode: FeatureSelectionMode = Field(
        default=FeatureSelectionMode.disabled,
        description=(
            "Feature selection mode: 'disabled', 'always' or 'auto'. "
            "By default 'disabled' — feature selection is not applied."
        ),
    )
    method: FeatureSelectionMethod = Field(
        default=FeatureSelectionMethod.importance,
        description=(
            "Selection algorithm: 'importance' (tree-based), 'percentile' "
            "(f_regression), 'mutual_info' (mutual information) or "
            "'variance' (variance filter)."
        ),
    )
    percentile: float = Field(
        default=50.0,
        gt=0.0,
        le=100.0,
        description=(
            "Percentage of features to keep (for the 'percentile' and "
            "'mutual_info' methods)."
        ),
    )
    min_features: int = Field(
        default=2,
        ge=1,
        description=(
            "Lower bound on the number of features kept (protection against "
            "an empty feature matrix)."
        ),
    )
    variance_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum feature variance for the 'variance' method.",
    )
    n_estimators: int = Field(
        default=50,
        ge=10,
        description=(
            "Number of trees used to estimate feature importance in the "
            "'importance' method."
        ),
    )
