"""Engine-facing re-exports of the Sanity Gate configuration models."""

from __future__ import annotations

from typing import Literal

from automl_contracts.config.sanity_gate import (
    ERR_REMOVED_SANITY_GATE_MODE,
    CorridorMode,
    GeneralizationGapMode,
    SanityGateCfg,
)

#: Internal mode of the robust select step (not a configuration field).
SanityGateMode = Literal["off", "active"]

__all__ = [
    "ERR_REMOVED_SANITY_GATE_MODE",
    "CorridorMode",
    "GeneralizationGapMode",
    "SanityGateCfg",
    "SanityGateMode",
]
