"""Config schema of the experimental ensemble (``general.ensemble``).

Holds ``EnsembleCfg`` — the control section of the experimental T10
ensembling branch. The section is a *behavioral* flag with defaults
(ensemble building is off by default): the engine builds the weighted
average ensemble of the top-K leaderboard models and adds the report
section to the result when enabled. The module must not import the other
config modules.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Weight conversion method: ``"score_weighted"`` — direction-aware
#: weighting from the validation scores (inverse for error metrics,
#: proportional for score metrics); ``"equal"`` — the uniform baseline.
EnsembleWeightMethod = Literal["score_weighted", "equal"]


class EnsembleCfg(BaseModel):
    """Settings of the experimental top-K ensemble.

    The ensemble is built AFTER the terminal select step from the
    completed records of the run's leaderboard: the top-K algorithms are
    retrained on 100% of the data (exactly once each) and their
    predictions are averaged with the weights derived from the
    validation scores. The result of the ensemble is reported in the
    dedicated ``ensemble`` section of the ``ExperimentResult`` (the
    winner and the winner artifact are NOT affected: the ensemble is an
    additional experimental output, not a replacement of the terminal
    selection).

    Attributes:
        enable: Build the ensemble (off by default — the ensemble is an
            experimental feature outside the MVP pipeline).
        k: The requested ensemble size (the number of top leaderboard
            members, ≥ 1). When ``k`` exceeds the number of available
            completed algorithms, ``min(k, n)`` is used with a warning.
        weight_method: How the member weights are derived from the
            validation scores: ``"score_weighted"`` (default) —
            direction-aware conversion (inverse for error metrics,
            proportional for score metrics); ``"equal"`` — the uniform
            baseline.
    """

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=False,
        description=(
            "Build the experimental weighted-average ensemble of the top-K "
            "leaderboard models after the terminal selection (off by "
            "default). When enabled, the result carries the dedicated "
            "'ensemble' report section (members, scores, weights, the "
            "artifact path of the serialized ensemble) and the artifact "
            "file is written next to the winner model; the winner and the "
            "winner artifact themselves are not affected"
        ),
    )
    k: int = Field(
        default=3,
        ge=1,
        description=(
            "The requested ensemble size: the number of top leaderboard "
            "members to combine (>= 1). When k exceeds the number of "
            "completed algorithms, min(k, n) members are used with a "
            "warning; k=1 is the valid single-model boundary case"
        ),
    )
    weight_method: EnsembleWeightMethod = Field(
        default="score_weighted",
        description=(
            "How the member weights are derived from the validation "
            "scores: 'score_weighted' (default) — direction-aware "
            "conversion (inverse for error metrics like RMSE/MAE, "
            "proportional for score metrics like R2); 'equal' — the "
            "uniform weights baseline"
        ),
    )

    @model_validator(mode="after")
    def _check_cfg(self) -> Any:
        """Validate the section (a placeholder for future cross-rules)."""
        return self
