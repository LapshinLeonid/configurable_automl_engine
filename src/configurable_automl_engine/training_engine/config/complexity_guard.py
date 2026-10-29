"""Config schema of the adaptive complexity guard (``general.complexity_guard``).

Holds ``ComplexityGuardCfg`` and its enums (``ComplexityGuardMode``,
``ComplexityGuardUserRangePolicy``, ``ComplexityGuardSizeBasis``). The module
must not import the other config modules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ─────────────────── complexity guard ─────────────────── #
class ComplexityGuardMode(str, Enum):
    """Operation modes of the adaptive complexity guard."""

    off = "off"  # The guard is disabled; spaces are unchanged.
    warn = "warn"  # Restrictions are computed and logged, but NOT applied.
    apply = "apply"  # Restrictions are computed and applied.


class ComplexityGuardUserRangePolicy(str, Enum):
    """Policy for an empty intersection of a user range with the guard range."""

    user_wins = "user_wins"  # The explicit user range stays.
    engine_wins = "engine_wins"  # The guard restriction is applied.


class ComplexityGuardSizeBasis(str, Enum):
    """Basis of the row count ``n`` for the size mode."""

    rows = "rows"  # Count independent rows.
    groups = "groups"  # Reserved: with group validation repeats of one group
    # carry less information than independent rows.


class ComplexityGuardCfg(BaseModel):
    """Settings of the adaptive complexity guard for tree/boosting spaces.

    Extends the data-dependent clipping layer (``clip_search_space``): on
    small data the search ranges of tree/boosting hyperparameters are too
    wide, trials are spent on values that do not change the model, the TPE
    surface has flat regions and the search drifts toward overfit models.
    This section tightens the ranges depending on the number of rows (``n``)
    and the feature count.

    Two feature counts are used deliberately: the size-mode ratio ``n / p``
    uses the ORIGINAL column count (a categorical column counts as 1, before
    the one-hot expansion), so that large-but-wide data (one-hot expanded)
    is not pushed into a smaller size mode; the rules that depend on the
    feature count (``max_features``, ``colsample_bytree``) use ``p`` after
    preprocessing — what the model sees.

    Two independent levels of restrictions:
    * level A — lossless clipping: removes values that cannot change the
      model on data of the given size (``lossless``). For ``min_samples_leaf``
      the cap is a physical bound (larger leaves are unreachable on the sample
      size), not a claim about model quality;
    * level B — regularization: deliberately shifts the space toward simpler
      models on small data (``regularize``). For boosting this includes the
      ``n_estimators`` budget — the tree count together with
      ``learning_rate`` is the main complexity regulator of a boosting
      ensemble;
    * level "time" — time-only caps of bagging ensembles (``n_estimators``
      of random forest / extra trees): more trees only reduce the variance
      of the ensemble.

    Attributes:
        mode (ComplexityGuardMode): Operation mode: 'off' — disabled,
            'warn' — restrictions are computed and logged but NOT applied,
            'apply' — computed and applied.
        lossless (bool): Enable level A (lossless clipping).
        regularize (bool): Enable level B (regularization heuristic).
        user_range_policy (ComplexityGuardUserRangePolicy): Policy for an
            empty intersection of a user range (``algo.hyperparameters``)
            with the guard range: 'user_wins' (default) — the user range
            stays, 'engine_wins' — the guard restriction is applied.
        small_n_threshold (int): ``n < threshold`` marks the 'small' mode.
        small_np_ratio_threshold (float): ``n / p < threshold`` marks the
            'small' mode. ``p`` is the ORIGINAL column count (a categorical
            column counts as 1, before one-hot expansion).
        medium_n_threshold (int): ``n < threshold`` marks the 'medium' mode.
        medium_np_ratio_threshold (float): ``n / p < threshold`` marks the
            'medium' mode. ``p`` is the ORIGINAL column count (a categorical
            column counts as 1, before one-hot expansion).
        size_basis (ComplexityGuardSizeBasis): Basis of ``n``: 'rows' (default)
            — independent rows, 'groups' — reserved for group validation.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ComplexityGuardMode = Field(
        default=ComplexityGuardMode.apply,
        description=(
            "Operation mode of the adaptive complexity guard: 'apply' "
            "(default) — restrictions are computed and applied, 'warn' — "
            "computed and logged but NOT applied, 'off' — disabled (search "
            "spaces are unchanged)."
        ),
    )
    lossless: bool = Field(
        default=True,
        description=(
            "Enable level A (lossless clipping): removes values that cannot "
            "change the model on data of the given size (e.g. max_depth=25 "
            "on 15 rows is equivalent to max_depth=6). Safe. For "
            "min_samples_leaf the cap is a physical bound (larger leaves "
            "are unreachable on the sample size), not a claim about model "
            "quality."
        ),
    )
    regularize: bool = Field(
        default=True,
        description=(
            "Enable level B (regularization): deliberately shifts the space "
            "toward simpler models on small data. A heuristic."
        ),
    )
    user_range_policy: ComplexityGuardUserRangePolicy = Field(
        default=ComplexityGuardUserRangePolicy.user_wins,
        description=(
            "Policy for an empty intersection of a user range "
            "(algo.hyperparameters) with the guard range: 'user_wins' "
            "(default) — the explicit user range stays (a warning is "
            "logged), 'engine_wins' — the guard restriction is applied."
        ),
    )
    small_n_threshold: int = Field(
        default=50,
        ge=1,
        description=(
            "Row threshold of the 'small' mode: n < threshold marks the data as small."
        ),
    )
    small_np_ratio_threshold: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Ratio threshold of the 'small' mode: n / p < threshold marks "
            "the data as small (wide data with few rows per feature). "
            "p is the ORIGINAL column count (a categorical column counts "
            "as 1, before the one-hot expansion)."
        ),
    )
    medium_n_threshold: int = Field(
        default=300,
        ge=1,
        description=(
            "Row threshold of the 'medium' mode: n < threshold marks the "
            "data as medium."
        ),
    )
    medium_np_ratio_threshold: float = Field(
        default=20.0,
        gt=0.0,
        description=(
            "Ratio threshold of the 'medium' mode: n / p < threshold marks "
            "the data as medium. p is the ORIGINAL column count (a "
            "categorical column counts as 1, before the one-hot expansion)."
        ),
    )
    size_basis: ComplexityGuardSizeBasis = Field(
        default=ComplexityGuardSizeBasis.rows,
        description=(
            "Basis of the row count n for the size mode: 'rows' (default) — "
            "independent rows, 'groups' — reserved for group validation "
            "(repeats of one group carry less information than independent "
            "rows); only 'rows' is used today."
        ),
    )

    @model_validator(mode="after")
    def _check_thresholds(self) -> ComplexityGuardCfg:
        """Check the ordering of the size-mode thresholds.

        Returns:
            ComplexityGuardCfg: The validated settings object.

        Raises:
            ValueError: If ``small_n_threshold >= medium_n_threshold`` or the
                small ratio threshold is not below the medium one.
        """
        if self.small_n_threshold >= self.medium_n_threshold:
            raise ValueError(
                "general.complexity_guard.small_n_threshold "
                f"({self.small_n_threshold}) must be less than "
                f"medium_n_threshold ({self.medium_n_threshold})"
            )
        if self.small_np_ratio_threshold >= self.medium_np_ratio_threshold:
            raise ValueError(
                "general.complexity_guard.small_np_ratio_threshold "
                f"({self.small_np_ratio_threshold}) must be less than "
                f"medium_np_ratio_threshold ({self.medium_np_ratio_threshold})"
            )
        return self
