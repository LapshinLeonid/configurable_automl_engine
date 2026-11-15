"""The structural root configuration model of the contract.

``Config`` only validates the shape of the configuration. Runtime checks
(installed algorithm packages, estimator hyperparameter compatibility and
type normalization) belong to the engine and are applied at its public entry
points.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .algorithms import AlgorithmsConfigType, OversamplingCfg
from .general import GeneralCfg


# ─────────────────── root ──────────────────── #
class Config(BaseModel):
    """Root object of the AutoML configuration contract.

    Aggregates the configuration sections and validates their structure,
    field types, allowed values and pure cross-field rules. Checks that need
    the installed environment (algorithm packages, estimator signatures,
    runtime custom-metric callables, platform limits) are applied by the
    engine's ``validate_runtime_config`` function.

    Attributes:
        general: General experiment parameters.
        oversampling: Data preprocessing settings.
        algorithms: Registry of available and active algorithms.
    """

    model_config = ConfigDict(extra="forbid")

    general: GeneralCfg = Field(
        default_factory=GeneralCfg,
        description="General experiment and validation settings",
    )
    oversampling: OversamplingCfg = Field(
        default=OversamplingCfg(), description="Data balancing settings"
    )
    algorithms: AlgorithmsConfigType = Field(
        ...,
        description=(
            "Dictionary of algorithms, where the key is the algorithm name "
            "(for example, 'xgboosting', 'random_forest')"
        ),
    )

    @field_validator("algorithms")
    @classmethod
    def _must_have_enabled(cls, v: Any) -> Any:
        # Since 'v' is now an AlgorithmsConfig (object),
        # we get all its fields via .model_dump()
        enabled_algorithms = [
            algo
            for algo in v.model_dump().values()
            if algo is not None and algo.get("enable") is True
        ]

        if not enabled_algorithms:
            raise ValueError("At least one algorithm must be enabled (enable: true)")
        return v


__all__ = ["Config"]
