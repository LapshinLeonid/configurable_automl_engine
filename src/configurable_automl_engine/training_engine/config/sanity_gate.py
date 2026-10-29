"""Config schema of the Sanity Gate audit (``general.sanity_gate``).

Holds ``SanityGateCfg`` and the types it shares with the select steps
(``SanityGateMode``, ``CorridorMode``, ``GeneralizationGapMode``). The module
must not import the other config modules.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic_core import PydanticCustomError

ERR_REMOVED_SANITY_GATE_MODE = "removed_sanity_gate_mode"

#: Robust winner-selection mode for ``select_robust_winner``, used by the
#: pipeline's select steps: 'off' ranks candidates by CV score only (without
#: audit); 'active' applies disqualifications, with the winner being the best
#: RMSE_oof. Select steps with ``strategy: sanity_gate`` run in 'active' mode.
#: The 'warn_only' mode was removed along with the report keys
#: ``sanity_gate_warn_only``/``winner_if_active``. This is an internal mode of
#: the select step, NOT a configuration field: the ``general.sanity_gate.mode``
#: field was removed, and the strategy is set explicitly in action='select'
#: phases. The definition remains here (not exported in ``__all__``) because
#: the robust-selection mechanism (``training_engine/state.py`` and
#: ``training_engine/finalists.py``) imports this type directly; it is not
#: used in the config schema.
SanityGateMode = Literal["off", "active"]

#: Shape of the finalist-pool corridor: multiplicative / additive / auto
#: (chosen based on the metric type). Defined here so that ``SanityGateCfg``
#: and ``select_finalists`` share one type without a circular import.
CorridorMode = Literal["multiplicative", "additive", "auto"]

#: Circuit D mode for ``_check_generalization_gap``:
#: ``"warn"`` (default) — a circuit D violation (a finite ``gap >
#: max_generalization_gap`` or an infinite gap from division by zero:
#: ``RMSE_full ≈ 0`` / ``RMSE_oof ≈ 0``) goes to
#: ``soft_warnings`` (a soft warning, does not disqualify);
#: ``"reject"`` — strict disqualification (the old default behavior).
#: Reasons in both modes remain only for contract violations
#: (``y_pred_full = None``) and non-computability (no finite pairs).
#: The field name is NOT ``mode``: that name is reserved for the removed
#: ``sanity_gate.mode`` field (the ``_reject_removed_mode`` validator rejects
#: it with a migration error). The type is declared here and in
#: ``sanity_gate.py`` (standalone) without an import, to keep the modules
#: decoupled.
GeneralizationGapMode = Literal["warn", "reject"]


# ─────────────────── sanity gate / select-step thresholds ─────────── #
class SanityGateCfg(BaseModel):
    """Thresholds and settings of the Sanity Gate audit.

            The section is exclusively a **source of thresholds** for select steps
            with ``strategy: sanity_gate`` (finalist pool: corridor δ from the
            leader ∩ hard cap ``top_k_candidates``, optional soft family
            complementation; the audit thresholds of ``ModelSanityGate`` are passed
            to the gate constructor unchanged). There is no longer a global switch:
            the ``mode`` field was removed together with the ``warn_only`` semantics —
            selection happens ONLY via explicit ``select`` steps in
            ``general.phases``, and diagnostics are available in the gate section of
            each select step's report.

            Having the section without select steps with ``strategy: sanity_gate``
            is valid: the section is an optional source of thresholds, not a
            behavioral flag.

    Attributes:
                top_k_candidates (int): Hard cap on the finalist pool size (>= 1).
                corridor_delta (float): Relative corridor width δ (in (0, 1)).
                corridor_mode (CorridorMode): 'multiplicative' | 'additive' | 'auto'.
                enforce_family_diversity (bool): Enable soft family complementation.
                algorithm_families (dict[str, str] | None): Map of "algorithm → family".
                allowed_families (list[str] | None): Families taking part in the complementation.
                family_diversity_multiplier (float): Expanded corridor multiplier.
                min_prediction_diversity (float): Circuit A threshold (in (0, 1]);
                    computed on OOF predictions. Applied only in the
                    non-adaptive mode (adaptive_diversity=false).
                adaptive_diversity (bool): Adaptive mode for circuit A
                    (default True): comparison with the constant model —
                    zero prediction variance and models worse than the
                    constant prediction (R²_oof < 0) fail.
                min_unique_count (int): Absolute minimum nunique for circuit B (>= 1).
                min_unique_ratio (float): Lower bound for the fraction of unique values (in (0, 1]).
                    Applied as ``min_unique_ratio * N`` capped by the number of distinct
                    target values (a perfect model always reaches the cap).
                max_dead_feature_ratio (float): Circuit C threshold (in [0, 1]) — a signal
                    for linear models; standalone disqualification only when
                    ``dead_features_require_low_diversity=true`` and diversity is low.
                dead_features_require_low_diversity (bool): Combined rule
                    "many dead features AND low diversity".
                zero_coef_tolerance (float): Zero coefficient tolerance (>= 0).
                max_generalization_gap (float): Circuit D threshold (>= 1, default 1.5).
                generalization_gap_mode (GeneralizationGapMode): Circuit D mode
                    (default 'warn'): 'warn' — a circuit D violation (a finite gap
                    above the threshold or an infinite gap from division by zero)
                    goes to soft_warnings (a soft warning); 'reject' — a failure →
                    reasons → disqualification (the old strict behavior).
                    Reasons in both modes — only contract violations and
                    non-computability.
                check_permutation_sensitivity (bool): Enable the permutation audit of
                    feature sensitivity (circuit C, for non-linear models).
                permutation_max_rows (int): Row subsample used for permutations (>= 1).
                permutation_max_features (int): Cap on the checked columns (>= 1).
                permutation_repeats (int): Permutation repeats (>= 1).
                permutation_seed (int): Fixed permutation seed.
                permutation_tolerance (float): Feature "deadness" threshold (>= 0).
                audit_time_budget_seconds (float): Time budget for the audit phase
                    (>= 0; 0 means unlimited).
                check_diversity (bool): Enable circuit A.
                check_unique (bool): Enable circuit B.
                check_dead_features (bool): Enable circuit C.
                check_generalization_gap (bool): Enable circuit D.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_mode(cls, data: Any) -> Any:
        """Reject the removed ``mode`` field with a migration hint.

        The field was removed entirely along with the ``warn_only`` semantics
        (a deliberate design decision): selection happens only via an
        explicit select step. Silently ignoring it would hide the semantic
        change, so a config with ``sanity_gate.mode`` is rejected with a
        migration hint.

        Args:
            data: Raw input data of the section (a dict when parsed from YAML/JSON).

        Returns:
            Any: The data unchanged, if the ``mode`` key is absent.

        Raises:
            ValueError: If the data contains the ``mode`` key.
        """
        if isinstance(data, dict) and "mode" in data:
            # The stable error code (ERR_REMOVED_SANITY_GATE_MODE) lets tests rely
            # on error['type'] instead of the hint text.
            raise PydanticCustomError(
                ERR_REMOVED_SANITY_GATE_MODE,
                "general.sanity_gate.mode has been removed. "
                "Migration: remove sanity_gate.mode and add an explicit "
                "select phase, e.g. {action: 'select', strategy: "
                "'sanity_gate', output: 'winner'} as the last phase; "
                "gate diagnostics are available in the gate section of the "
                "select step report. The sanity_gate section remains the "
                "source of thresholds (corridor, top_k_candidates, audit "
                "limits, ...). See 'Migrating older configs' in API_REFERENCE.md.",
            )
        return data

    top_k_candidates: int = Field(
        default=3,
        ge=1,
        description="Hard cap on the finalist pool size (>= 1, default 3).",
    )
    corridor_delta: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
        description=(
            "Relative corridor width δ from the leader (strictly in (0, 1), default 0.15)."
        ),
    )
    corridor_mode: CorridorMode = Field(
        default="auto",
        description=(
            "Shape of the finalist pool corridor: 'multiplicative', 'additive' "
            "or 'auto' (chosen based on the metric type, default 'auto')."
        ),
    )
    enforce_family_diversity: bool = Field(
        default=False,
        description=(
            "Optional soft complementation of missing families into the finalist pool "
            "(default False — disabled)."
        ),
    )
    algorithm_families: dict[str, str] | None = Field(
        default=None,
        description=(
            "Map of 'algorithm → family' for the family complementation rule. "
            "None — algorithms without a classification are treated as their own family."
        ),
    )
    allowed_families: list[str] | None = Field(
        default=None,
        description=(
            "Families taking part in the family complementation rule (None — all are allowed)."
        ),
    )
    family_diversity_multiplier: float = Field(
        default=1.5,
        gt=0.0,
        description="Multiplier of the expanded corridor for family complementation (> 0).",
    )
    min_prediction_diversity: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description=(
            "Circuit A (diversity) threshold: Var(y_pred_oof)/Var(y) >= threshold; "
            "computed on OOF predictions (strictly in (0, 1], default 0.15). "
            "Applied only in the non-adaptive mode (adaptive_diversity=false)."
        ),
    )
    adaptive_diversity: bool = Field(
        default=True,
        description=(
            "Adaptive mode for circuit A (default True): comparison with a "
            "constant model — zero variance in predictions (a tie with the "
            "constant baseline) and models worse than the constant prediction "
            "(R²_oof < 0) fail; the absolute min_prediction_diversity "
            "threshold is not applied."
        ),
    )
    min_unique_count: int = Field(
        default=5,
        ge=1,
        description="Absolute lower bound for nunique in circuit B (>= 1).",
    )
    min_unique_ratio: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description=(
            "Lower bound for the fraction of unique predictions in circuit B (in (0, 1]). "
            "Applied as min_unique_ratio * N capped by the number of distinct target "
            "values (a perfect model always reaches the cap)."
        ),
    )
    max_dead_feature_ratio: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Threshold for the fraction of 'dead' features in circuit C (in [0, 1]) — for "
            "linear models this is a signal; standalone disqualification "
            "only when dead_features_require_low_diversity=true and diversity "
            "is low."
        ),
    )
    dead_features_require_low_diversity: bool = Field(
        default=True,
        description=(
            "Combined circuit C rule: disqualification only on the "
            "combination of 'many dead features AND low diversity'."
        ),
    )
    zero_coef_tolerance: float = Field(
        default=1e-12,
        ge=0.0,
        description="Coefficients with |coef| <= the threshold are considered zero (circuit C).",
    )
    max_generalization_gap: float = Field(
        default=1.5,
        ge=1.0,
        description="Circuit D threshold: RMSE_oof/RMSE_full > threshold — overfitting.",
    )
    generalization_gap_mode: GeneralizationGapMode = Field(
        default="warn",
        description=(
            "Circuit D (generalization gap) mode: 'warn' (default) — "
            "a circuit D violation (a finite gap > max_generalization_gap or "
            "an infinite gap from division by zero: RMSE_full≈0 / RMSE_oof≈0) "
            "is recorded in soft_warnings (does not disqualify); "
            "'reject' — a failure → reasons → disqualification (the old strict "
            "behavior). Reasons in both modes remain only for "
            "contract violations (y_pred_full=None) and non-computability (no "
            "finite pairs). The field is deliberately NOT named mode (a "
            "collision with the removed sanity_gate.mode field)."
        ),
    )
    check_permutation_sensitivity: bool = Field(
        default=True,
        description=(
            "Enable the permutation audit of feature sensitivity "
            "(circuit C for non-linear models; controls audit cost). "
            "Default is True; False — the permutation path is skipped, "
            "the linear coefficient-based path remains."
        ),
    )
    permutation_max_rows: int = Field(
        default=5000,
        ge=1,
        description="Row subsample for the permutation path of circuit C (>= 1, default 5000).",
    )
    permutation_max_features: int = Field(
        default=100,
        ge=1,
        description="Cap on the columns checked in the permutation path (>= 1, default 100).",
    )
    permutation_repeats: int = Field(
        default=3,
        ge=1,
        description="Number of permutation repeats per column (>= 1).",
    )
    permutation_seed: int = Field(
        default=42,
        description="Fixed permutation seed (for determinism).",
    )
    permutation_tolerance: float = Field(
        default=1e-3,
        ge=0.0,
        description="Relative 'deadness' threshold for a feature in permutations.",
    )
    audit_time_budget_seconds: float = Field(
        default=0,
        ge=0.0,
        description=(
            "Time budget for the Sanity Gate audit phase (in seconds). "
            "0 — unlimited (default)."
        ),
    )
    check_diversity: bool = Field(default=True, description="Enable circuit A.")
    check_unique: bool = Field(default=True, description="Enable circuit B.")
    check_dead_features: bool = Field(default=True, description="Enable circuit C.")
    check_generalization_gap: bool = Field(
        default=True, description="Enable circuit D."
    )
