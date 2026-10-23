"""
AutoML Config Engine: Validation and parsing of hierarchical YAML configs.
The module implements a strictly typed configuration object model
based on Pydantic v2, ensuring integrity checks of experiment parameters,
validation strategies, and hyperparameter search spaces
before the AutoML pipeline is launched.
Key features:
    1. Multi-Stage Pipeline: Configuring a sequence of steps
       (HPO phases 'all_algorithms' / 'refine' and explicit 'select' steps
       with cv_best / sanity_gate strategies) with phase cross-validation.
    2. Intelligent Validation Logic: Automatic consistency checks for
       settings, including the number of folds for the k-fold strategy
       and validation of numeric range boundaries.
    3. Flexible Search Space DSL: Support for a compact YAML syntax
       for defining search spaces (Categorical, Float, Int, Log)
       with automatic type coercion from list-like structures.
    4. Dependency Guard: A built-in mechanism that verifies whether the
       required third-party packages (XGBoost and others) are installed
       for all algorithms enabled in the config, at initialization time.
    5. Data Balancing Schema: Management of oversampling strategies
       (SMOTE, ADASYN) with field aliasing support
       for a cleaner YAML-file structure.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

# Stable migration error codes: tests may rely on ``error['type']`` from
# ValidationError instead of the hint text, which is subject to change.
ERR_REMOVED_REFINE_WINNER = "removed_refine_winner_action"
ERR_REMOVED_SANITY_GATE_MODE = "removed_sanity_gate_mode"

from configurable_automl_engine.common.definitions import (
    ALGO_PACKAGE_MAPPING,
    ParallelStrategy,
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.hyperopt_defaults import SearchSpaceEntry
from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    get_allowed_hyperparameters,
)
from configurable_automl_engine.preprocessing import EncodingStrategy
from configurable_automl_engine.preprocessing_presets import PreprocessingOverride
from configurable_automl_engine.training_engine.metrics import (
    AVAILABLE_METRICS,
    to_sklearn_name,
)

# Build the type on the fly. *AVAILABLE_METRICS unpacks the list into Literal arguments
ComparisonMetric = Literal[*AVAILABLE_METRICS]  # type: ignore


_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")

__all__ = [
    "AlgoCfg",
    "ComplexityGuardCfg",
    "ComplexityGuardMode",
    "ComplexityGuardSizeBasis",
    "ComplexityGuardUserRangePolicy",
    "Config",
    "CorridorMode",
    "FeatureSelectionCfg",
    "FeatureSelectionMethod",
    "FeatureSelectionMode",
    "GeneralizationGapMode",
    "HPOPhaseAction",
    "HPOPhaseCfg",
    "SanityGateCfg",
    "SelectOutput",
    "SelectStrategy",
    "ValidationStrategy",
    "read_config",
]

#: Robust winner-selection mode for ``select_robust_winner``, used by the
#: pipeline's select steps: 'off' ranks candidates by CV score only (without
#: audit); 'active' applies disqualifications, with the winner being the best
#: RMSE_oof. Select steps with ``strategy: sanity_gate`` run in 'active' mode.
#: The 'warn_only' mode was removed along with the report keys
#: ``sanity_gate_warn_only``/``winner_if_active``. This is an internal mode of
#: the select step, NOT a configuration field: the ``general.sanity_gate.mode``
#: field was removed, and the strategy is set explicitly in action='select'
#: phases. The definition remains here (not exported in ``__all__``) because
#: the robust-selection mechanism in ``training_engine/component.py`` imports
#: this type directly; it is not used in the config schema.
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

#: Pipeline phase actions: search across all algorithms, fine-tuning of the
#: whole input set (warm-start), and an explicit selection step. The legacy
#: 'refine_winner' value was removed — configs containing it are rejected by
#: validation with a migration hint.
HPOPhaseAction = Literal["all_algorithms", "refine", "select"]

#: Select step strategy: ranking by CV score or a Sanity Gate audit.
SelectStrategy = Literal["cv_best", "sanity_gate"]

#: What the select step returns: a single winner, the top-K candidates,
#: or the entire corridor pool of finalists.
SelectOutput = Literal["winner", "top_k", "pool"]


# ─────────────────── pruning (early stopping) ──────────────────── #
class PruningStrategy(str, Enum):
    """Supported early stopping (pruning) strategies for Optuna trials."""

    median = "median"
    hyperband = "hyperband"


class PruningCfg(BaseModel):
    """Early stopping (pruning) settings for trials in the HPO phase.

    Intermediate evaluation mechanism: when enabled, the tuner publishes a
    quality estimate after every natural validation step (for example, after
    each cross-validation fold), allowing the Optuna pruner to discard an
    obviously hopeless trial before the full evaluation finishes.
    For validation strategies without natural steps (hold-out /
    train_test_split) the pruner is not applied — the behavior is safe and
    documented (see API_REFERENCE.md).

    Attributes:
        enable (bool): Global flag enabling early stopping.
            Default is False — the feature is off and trials run to completion.
        strategy (PruningStrategy): Pruning strategy: 'median'
            (optuna.pruners.MedianPruner) or 'hyperband'
            (optuna.pruners.HyperbandPruner).
        min_steps (int): Minimum number of intermediate steps (for example,
            CV folds) before the first pruning decision. For 'median' this is
            n_warmup_steps, for 'hyperband' — min_resource.
        n_startup_trials (int): Number of completed trials after which pruning
            is enabled (only for 'median'; analogous to
            n_startup_trials in Optuna).
        reduction_factor (int): Budget reduction factor between Hyperband
            rounds (only for 'hyperband'; controls pruning aggressiveness).
    """

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=False,
        description=(
            "Flag enabling early stopping (pruning) of Optuna trials. "
            "Default is False — each trial runs to completion."
        ),
    )
    strategy: PruningStrategy = Field(
        default=PruningStrategy.median,
        description=(
            "Trial pruning strategy: 'median' (MedianPruner) "
            "or 'hyperband' (HyperbandPruner)"
        ),
    )
    min_steps: int = Field(
        default=1,
        ge=1,
        description=(
            "Minimum number of intermediate steps (for example, folds "
            "of cross-validation) before the first pruning decision. For 'median' "
            "maps to n_warmup_steps, for 'hyperband' — min_resource"
        ),
    )
    n_startup_trials: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of completed trials after which pruning is enabled "
            "(only for 'median'; analogous to n_startup_trials in Optuna)"
        ),
    )
    reduction_factor: int = Field(
        default=3,
        ge=2,
        description=(
            "Budget reduction factor between Hyperband rounds "
            "(only for 'hyperband'; pruning aggressiveness)"
        ),
    )


# ─────────────────── phases ──────────────────── #
class HPOPhaseCfg(BaseModel):
    """Configuration of a single pipeline step.

    The class describes the parameters of a specific step: an HPO phase
    (``all_algorithms`` / ``refine``) or an explicit selection step
    (``select``), enabling multi-stage search and selection strategies
    (for example: search across all models → audited select → fine-tuning
    finalists → final winner select).

    Attributes:
        name (str): Unique phase identifier.
        n_trials (int): Iteration (trial) limit for HPO phases
            (``all_algorithms`` / ``refine``). Ignored for ``select``,
            but the field remains mandatory for schema uniformity.
        action (str): Action type: 'all_algorithms' (search across all
            algorithms), 'refine' (fine-tune the whole input set
            with warm-start) or 'select' (selection step). The legacy value
            'refine_winner' was removed: configs containing it are rejected
            by validation with a migration hint.
        strategy (Literal['cv_best', 'sanity_gate'] | None): Select step
            strategy (only for action='select'): ranking by CV score
            ('cv_best') or a Sanity Gate audit ('sanity_gate').
        output (Literal['winner', 'top_k', 'pool'] | None): What the
            select step returns (only for action='select'): 'winner' — a
            single winner (terminal step), 'top_k' — the top-K candidates,
            'pool' — the entire corridor pool of finalists.
        k (int | None): Number of candidates (>= 1) for output='top_k'.
            Required only for output='top_k'; forbidden for
            output='winner'/'pool'.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="phase",
        description=(
            "User-defined name of the optimization phase"
            " (for example, 'coarse_search' or 'fine_tuning')"
        ),
    )
    n_trials: int = Field(
        ge=1,
        description=(
            "Number of iterations (trials) for HPO phases (all_algorithms / "
            "refine). Ignored for action='select'"
        ),
    )
    action: HPOPhaseAction = Field(
        default="all_algorithms",
        description=(
            "Phase action: 'all_algorithms' — search across all algorithms, "
            "'refine' — fine-tune the whole input set with warm-start, "
            "'select' — explicit selection step (strategy governs the "
            "strategy, output the output shape). The legacy 'refine_winner' "
            "is removed: use the pair 'select' (cv_best) + 'refine'"
        ),
    )
    strategy: SelectStrategy | None = Field(
        default=None,
        description=(
            "Select step strategy (only for action='select'): 'cv_best' — "
            "ranking by CV score, 'sanity_gate' — finalist pool + "
            "ModelSanityGate audit + ranking by RMSE_oof. Forbidden for "
            "all_algorithms/refine phases"
        ),
    )
    output: SelectOutput | None = Field(
        default=None,
        description=(
            "Output shape of the select step (only for action='select'): "
            "'winner' — a single winner (terminal step), "
            "'top_k' — the top-K candidates (requires the k parameter), "
            "'pool' — the entire corridor pool of finalists. Forbidden for "
            "all_algorithms/refine phases"
        ),
    )
    k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of candidates (>= 1) for output='top_k'; required "
            "only for output='top_k', forbidden for output='winner'/'pool'. "
            "Forbidden for all_algorithms/refine phases"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_refine_winner(cls, data: Any) -> Any:
        """Reject the legacy 'refine_winner' action with a migration hint.

        The value was deliberately removed without backward compatibility:
        silently ignoring it or translating it automatically would hide a
        semantic change (refine_winner narrowed the set to a single candidate,
        whereas the new select + refine pair does not). The error ships with a
        migration hint.

        Args:
            data: Raw input data of the phase (a dict when parsed from YAML/JSON).

        Returns:
            Any: The data unchanged, if no legacy value is present.

        Raises:
            ValueError: If action equals 'refine_winner'.
        """
        if isinstance(data, dict) and data.get("action") == "refine_winner":
            # The stable error code (ERR_REMOVED_REFINE_WINNER) lets tests rely
            # on error['type'] instead of the hint text.
            raise PydanticCustomError(
                ERR_REMOVED_REFINE_WINNER,
                "action 'refine_winner' has been removed. "
                "Migration: replace the 'refine_winner' phase with explicit "
                "'select' and 'refine' phases, e.g. "
                "all_algorithms -> select(strategy: cv_best, output: top_k, "
                "k: N) -> refine -> select(strategy: cv_best, output: winner); "
                "for a single-winner pipeline use "
                "all_algorithms -> select(strategy: cv_best, output: winner). "
                "See the migration table in README.md.",
            )
        return data

    @model_validator(mode="after")
    def _validate_select_fields(self) -> HPOPhaseCfg:
        """Validate the consistency of strategy/output/k with the phase action.

        Validation logic:
        1. For action='select' the strategy and output fields are required.
        2. For output='top_k' the k parameter is required (k >= 1 is enforced
           by the field constraint); for output='winner'/'pool' the k parameter
           is forbidden.
        3. For action='all_algorithms'/'refine' the strategy/output/k fields
           are forbidden (candidate narrowing is the job of a select step).

        Returns:
            HPOPhaseCfg: The validated phase object.

        Raises:
            ValueError: If the set of fields is inconsistent with the phase action.
        """
        if self.action == "select":
            missing = [
                field
                for field in ("strategy", "output")
                if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    f"action 'select' requires fields {missing} to be set "
                    "(strategy: 'cv_best' | 'sanity_gate', "
                    "output: 'winner' | 'top_k' | 'pool')"
                )
            if self.output == "top_k" and self.k is None:
                raise ValueError(
                    "output 'top_k' requires the integer parameter k (>= 1); "
                    "set k on the select phase or use output 'winner'/'pool'"
                )
            if self.output in ("winner", "pool") and self.k is not None:
                raise ValueError(
                    f"parameter k is not allowed for output '{self.output}' "
                    "(k is only valid for output 'top_k'); remove k or use "
                    "output 'top_k'"
                )
            return self

        forbidden = [
            field
            for field in ("strategy", "output", "k")
            if getattr(self, field) is not None
        ]
        if forbidden:
            raise ValueError(
                f"fields {forbidden} are only allowed for action 'select' "
                f"(got action '{self.action}'): candidate narrowing is the "
                "job of explicit select phases"
            )
        return self


# ─────────────────── feature selection ──────────────────── #
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
        mode (FeatureSelectionMode): Operation mode: 'disabled', 'always'
            or 'auto'.
        method (FeatureSelectionMethod): Selection algorithm: 'importance',
            'percentile', 'mutual_info' or 'variance'.
        percentile (float): Percentage of the best features to keep
            (0 < percentile <= 100) for the 'percentile' and 'mutual_info'
            methods.
        min_features (int): Guaranteed minimum number of features (>= 1),
            protects against an empty feature matrix.
        variance_threshold (float): Minimum feature variance (>= 0.0)
            for the 'variance' method.
        n_estimators (int): Number of trees in the ensemble (>= 10)
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
                    computed on OOF predictions.
                adaptive_diversity (bool): Adaptive mode for circuit A.
                min_unique_count (int): Absolute minimum nunique for circuit B (>= 1).
                min_unique_ratio (float): Lower bound for the fraction of unique values (in (0, 1]).
                    Applied as ``min_unique_ratio * N`` capped by the number of distinct
                    target values (a perfect model always reaches the cap; issue #125).
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
                "limits, ...). See the migration table in README.md.",
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
            "computed on OOF predictions (strictly in (0, 1], default 0.15)."
        ),
    )
    adaptive_diversity: bool = Field(
        default=False,
        description=(
            "Adaptive mode for circuit A: comparison with a constant model "
            "(zero variance in predictions — failure)."
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
            "values (a perfect model always reaches the cap; issue #125)."
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
      models on small data (``regularize``).
    * level "time" — time-only caps (``n_estimators``): more trees only
      reduce the variance of the ensemble.

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


# ─────────────────── general ─────────────────── #
class GeneralCfg(BaseModel):
    """General settings of the AutoML process, validation, and parallelism.

    The central control node of the experiment, responsible for choosing
    metrics, quality evaluation strategies (Cross-Validation / Hold-out),
    and distributing computational resources.
    Attributes:
        comparison_metric (ComparisonMetric): Main metric for ranking models.
        additional_metrics (List[ComparisonMetric]): Additional informational
            metrics, computed for the final model with the same validation method
            used for model comparison (hold-out/CV, not train score).
        path_to_model (Path): Filesystem path for exporting the model artifact.
        serialization_format (SerializationFormat): Save format (pickle/joblib).
        log_to_file (Path | None): File for writing the engine's run logs.
        phases (List[HPOPhaseCfg]): Sequence of pipeline steps:
            HPO phases ('all_algorithms' / 'refine') and explicit selection
            steps ('select'); the last phase must be a select with output: winner.
        validation_strategy (ValidationStrategy): Evaluation method (k-fold or hold-out).
        n_folds (int): Number of splits for cross-validation.
        categorical_encoding (Literal['one_hot', 'ordinal']): Encoding strategy
            for categorical features. Default is 'one_hot'.
        parallel_strategy (str): Parallelization level (by algorithms/folds).
        max_workers (int | None): Limit on threads or processes.
        parallel_mode (str): Technical execution mode ('threads' or 'processes').
        feature_selection (FeatureSelectionCfg): Feature space reduction
            settings. By default mode='disabled' — feature selection is not applied.
    """

    model_config = ConfigDict(extra="forbid")

    comparison_metric: ComparisonMetric = Field(
        default="r2",
        description=(
            "Metric for comparing models"
            " and choosing the best one (for example, r2, rmse, accuracy)"
        ),
    )
    additional_metrics: list[ComparisonMetric] = Field(
        default_factory=list,
        description=(
            "List of additional quality metrics, whose values are computed "
            "for the final trained model with the same validation method used for "
            "model comparison (hold-out scoring or the mean over CV folds), and "
            "are returned together with the training results (key 'additional_metrics'). "
            "The metrics are purely informational: they do not take part "
            "in hyperparameter optimization, model comparison, or winner selection. "
            "Valid values are the same as the set of comparison_metric values. "
            "Duplicates in the list are ignored (the value is counted once), and a "
            "match with the main comparison metric (including aliases, for example "
            "rmse ↔ neg_root_mean_squared_error) is excluded from the list — its value "
            "is already returned as 'score'. Both rules are applied during "
            "config validation, so the resulting list in "
            "general.additional_metrics may differ from the one originally passed "
            "by the user."
        ),
    )
    path_to_model: Path = Field(
        default=Path("model.pkl"),
        description="Path for saving/loading the resulting trained model",
    )
    serialization_format: SerializationFormat = Field(
        default=SerializationFormat.pickle,
        description="Model serialization format (pickle, joblib, etc.)",
    )
    log_to_file: Path | None = Field(
        default=None,
        description=(
            "Path to the log file. If not set, logs are printed to the console only"
        ),
    )
    phases: list[HPOPhaseCfg] = Field(
        ..., description="List of sequential hyperparameter optimization phases"
    )
    validation_strategy: ValidationStrategy = Field(
        default=ValidationStrategy.auto,
        description=(
            "Quality evaluation strategy:"
            "k-fold cross-validation, fixed hold-out,"
            "LOO, or automatic selection"
        ),
    )
    n_folds: int = Field(
        default=5,
        description=(
            "Number of blocks (folds) for cross-validation."
            "Used only when validation_strategy = 'k_fold'"
        ),
    )
    categorical_encoding: EncodingStrategy = Field(
        default="one_hot",
        description=(
            "Encoding strategy for categorical features: 'one_hot' (each "
            "category -> a separate binary column), 'ordinal' (each "
            "categorical column -> one numeric column with an imposed "
            "order), 'target' (target encoding with smoothing), 'frequency' "
            "(category frequency) or 'hashing' (deterministic hashing "
            "into a fixed number of binary columns). Ordinal encoding does "
            "not increase the number of columns and is useful for linear models, "
            "but it imposes an artificial order on categories. Target/frequency/"
            "hashing are effective for high-cardinality columns."
        ),
    )
    high_cardinality_threshold: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Cardinality threshold for the automatic mode (>= 0). Columns "
            "with a number of unique values strictly greater than the threshold "
            "are encoded with the high_cardinality_encoding strategy, the rest — "
            "with the categorical_encoding strategy. Must be set together with "
            "high_cardinality_encoding. None — the automatic mode is disabled."
        ),
    )
    high_cardinality_encoding: EncodingStrategy | None = Field(
        default=None,
        description=(
            "Encoding strategy for high-cardinality columns "
            "(above high_cardinality_threshold). Must be set together with "
            "high_cardinality_threshold. None — the mode is disabled."
        ),
    )
    hashing_n_components: int = Field(
        default=16,
        ge=1,
        description=(
            "Number of binary columns per categorical column for "
            "'hashing' encoding (>= 1). Controls the dimensionality of the "
            "output features: the growth in the number of features does not "
            "depend on the column cardinality."
        ),
    )
    target_encoding_smoothing: float = Field(
        default=20.0,
        ge=0,
        description=(
            "Smoothing parameter (m) of target encoding (>= 0). At m=0 "
            "the pure mean of the target variable per category is used; "
            "at large m rare categories approach the global mean."
        ),
    )
    target_encoding_fallback: float | None = Field(
        default=None,
        description=(
            "Fallback value of target encoding for categories not present "
            "in the training set. None — the global mean of the target variable."
        ),
    )
    parallel_strategy: ParallelStrategy = Field(
        default=next(iter(ParallelStrategy)),
        description=(
            "Parallelization strategy."
            "Currently only 'algorithms' is supported"
            " (each algorithm in its own thread/process)."
        ),
    )
    max_workers: int | None = Field(
        default=None,
        description=(
            "Maximum number of threads/processes."
            "If null, the number of CPU cores is used"
        ),
    )
    parallel_mode: Literal["threads", "processes"] = Field(
        default="threads",
        description=(
            "Concurrency mode: threads (for I/O-bound tasks)"
            " or processes (for CPU-intensive computations)"
        ),
    )
    phase_timeout: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Global timeout for the whole HPO phase (in seconds). "
            "If None — the default value of 3600 seconds (1 hour) is used. "
            "Supported only in parallel_mode='processes', where it is hard: "
            "hung workers are forcibly terminated. In parallel_mode='threads' "
            "timeouts are forbidden and are a config error: an already "
            "running Python function cannot be interrupted, so a soft timeout "
            "does not save time and only discards the ready results of "
            "completed algorithms."
        ),
    )
    task_timeout: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Timeout for a single task (algorithm) within a phase (in seconds). "
            "If None — phase_timeout or the global timeout is used. "
            "Counted from the moment the task actually starts, not from the "
            "moment it was submitted to the pool: while a task waits in the "
            "queue (more algorithms than max_workers), it does not expire "
            "before starting. Supported only in parallel_mode='processes', "
            "where it is hard. In parallel_mode='threads' timeouts are "
            "forbidden and are a config error: an already running Python "
            "function cannot be interrupted, so a soft timeout does not save "
            "time and only discards the ready results of completed algorithms."
        ),
    )
    pruning: PruningCfg = Field(
        default_factory=PruningCfg,
        description=(
            "Early stopping (pruning) settings for Optuna trials. "
            "Disabled by default: each trial runs to completion."
        ),
    )
    feature_selection: FeatureSelectionCfg = Field(
        default_factory=FeatureSelectionCfg,
        description=(
            "Feature space reduction settings. By default "
            "mode='disabled' — feature selection is not applied."
        ),
    )
    sanity_gate: SanityGateCfg = Field(
        default_factory=SanityGateCfg,
        description=(
            "Thresholds and settings of the Sanity Gate audit for select steps "
            "with strategy: sanity_gate. The section is an optional "
            "source of thresholds (corridor, top_k_candidates, audit limits), "
            "not a behavioral flag: selection is performed only by explicit "
            "select phases in general.phases. Having the section without "
            "sanity select steps is valid"
        ),
    )
    complexity_guard: ComplexityGuardCfg = Field(
        default_factory=ComplexityGuardCfg,
        description=(
            "Settings of the adaptive complexity guard for tree/boosting "
            "search spaces on small data (mode: 'off'/'warn'/'apply', "
            "lossless/regularize levels, user_range_policy, size-mode "
            "thresholds, size_basis). See API_REFERENCE.md."
        ),
    )

    @field_validator("additional_metrics")
    @classmethod
    def _deduplicate_additional_metrics(
        cls, v: list[ComparisonMetric]
    ) -> list[ComparisonMetric]:
        """Remove duplicates from the list of additional metrics.

        Documented behavior: the same metric specified several times is
        computed and returned exactly once (the order of the first
        occurrence is preserved). Deduplication is done using the
        lowercased string representation of the metric: ``ComparisonMetric``
        items are strings, so calling ``.lower()`` is type-safe and does not
        depend on an arbitrary element type.
        """
        seen: set[str] = set()
        result: list[ComparisonMetric] = []
        for name in v:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                result.append(name)
        return result

    @model_validator(mode="after")
    def _exclude_comparison_from_additional(self) -> GeneralCfg:
        """Exclude the main comparison metric from the additional metrics list.

        The value of the main comparison metric is already returned in the
        training results as ``score``, so duplicating it in
        ``additional_metrics`` is unnecessary. The comparison is done by the
        sklearn name of the metric, so aliases of the same metric
        (for example, 'rmse' and 'neg_root_mean_squared_error') count as a
        match.
        """
        comparison_sklearn = to_sklearn_name(self.comparison_metric)
        filtered = [
            m
            for m in self.additional_metrics
            if to_sklearn_name(m) != comparison_sklearn
        ]
        if len(filtered) != len(self.additional_metrics):
            logging.getLogger(__name__).warning(
                "additional_metrics contains the comparison metric '%s'; "
                "its value is already returned as 'score' and will not be "
                "duplicated in 'additional_metrics'.",
                self.comparison_metric,
            )
        self.additional_metrics = filtered
        return self

    @model_validator(mode="after")
    def _check_n_folds(self) -> GeneralCfg:
        """Check the logical consistency of validation and serialization settings.

        Validation logic:
        1. Checks that the `joblib` package is available when the corresponding format is selected.
        2. Ensures `n_folds` is a positive number for any strategy.
        3. Strict check for `k_fold`: the number of blocks must be at least 2.

        Returns:
            GeneralCfg: The validated settings object.

        Raises:
            ValueError: If the 'joblib' package is not installed or `n_folds` is
                invalid for the selected strategy.
        """
        if self.serialization_format == SerializationFormat.joblib and not is_installed(
            "joblib"
        ):
            raise ValueError(
                "serialization_format='joblib' requires"
                " the 'joblib' package to be installed"
            )
        if self.n_folds < 1:
            raise ValueError("`n_folds` must be at least 1")
        # Strict validation only for k-fold
        if self.validation_strategy == ValidationStrategy.k_fold and self.n_folds < 2:
            raise ValueError("n_folds must be ≥ 2 for k-fold validation")
        return self

    @model_validator(mode="after")
    def _check_pruning_consistency(self) -> GeneralCfg:
        """Check the consistency of early stopping settings with validation.

        Validation logic:
        1. For 'k_fold' the pruner must be able to fire before the evaluation
           finishes: min_steps cannot exceed n_folds.
        2. For 'train_test_split' there are no natural steps — early stopping
           is not applied; we emit a warning rather than an error (the
           behavior is documented and safe).

        Returns:
            GeneralCfg: The validated settings object.

        Raises:
            ValueError: If the pruning settings conflict with validation.
        """
        if not self.pruning.enable:
            return self

        if (
            self.validation_strategy == ValidationStrategy.k_fold
            and self.pruning.min_steps > self.n_folds
        ):
            raise ValueError(
                f"pruning.min_steps ({self.pruning.min_steps}) cannot be greater "
                f"than general.n_folds ({self.n_folds}) for k-fold validation: "
                "the pruner would never get enough intermediate steps. "
                "Decrease pruning.min_steps or increase general.n_folds."
            )

        if self.validation_strategy == ValidationStrategy.train_test_split:
            logging.getLogger(__name__).warning(
                "pruning.enable=true with validation_strategy='train_test_split': "
                "early stopping will NOT be applied because hold-out evaluation "
                "has no intermediate steps."
            )
        return self

    @model_validator(mode="after")
    def _validate_phase_pipeline(self) -> GeneralCfg:
        """Cross-validate the sequence of pipeline phases.

        The order of after-validators in Pydantic v2 is determined by their
        definition order, so this validator is deliberately declared BEFORE
        ``_check_high_cardinality_encoding``: pipeline structure errors (the
        root cause of a config) are guaranteed to be reported before errors
        in categorical column encoding.

        Validation logic (an empty phase list is not checked — the guard
        against missing phases stays in the ``train_best_model`` runtime):
        1. ``select`` cannot be the first phase: before any search there is
           nothing to audit/rank.
        2. Phases after a terminal select (``output: winner``) are forbidden:
           such a step yields a single winner, and any subsequent
           fine-tuning would hide the selection result.
        3. The last phase must be a ``select`` with ``output: winner``:
           the final artifact is built only from the result of the terminal
           selection step.

        Returns:
            GeneralCfg: The validated settings object.

        Raises:
            ValueError: If the phase sequence violates the pipeline rules.
        """
        phases = self.phases
        if not phases:
            return self

        if phases[0].action == "select":
            raise ValueError(
                f"phase '{phases[0].name}' (index 0): action 'select' cannot "
                "be the first phase — there is nothing to rank/audit yet. "
                "Start the pipeline with an 'all_algorithms' (or 'refine') "
                "phase."
            )

        for index, phase in enumerate(phases[:-1]):
            if phase.action == "select" and phase.output == "winner":
                raise ValueError(
                    f"phase '{phase.name}' (index {index}): select with "
                    "output 'winner' is terminal — no phases are allowed "
                    "after it. Move it to the last position or use "
                    "output 'top_k'/'pool' to keep the pipeline going."
                )

        last = phases[-1]
        if not (last.action == "select" and last.output == "winner"):
            raise ValueError(
                f"the last phase '{last.name}' (action '{last.action}') must "
                "be a 'select' phase with output 'winner': the final artifact "
                "is built from the result of the terminal select step. "
                "Migration: append "
                "{action: 'select', strategy: 'cv_best', output: 'winner'} "
                "as the last phase. See the migration table in README.md."
            )
        return self

    @model_validator(mode="after")
    def _check_high_cardinality_encoding(self) -> GeneralCfg:
        """Check the consistency of the high-cardinality encoding settings.

        Validation logic:
        1. ``high_cardinality_threshold`` and ``high_cardinality_encoding``
           must be set together (otherwise the behavior is ambiguous — the
           configuration is rejected before training starts).
        2. The values of ``categorical_encoding`` / ``high_cardinality_encoding``
           are validated by the ``EncodingStrategy`` type (an unsupported
           strategy is rejected during config parsing).

        Returns:
            GeneralCfg: The validated settings object.

        Raises:
            ValueError: If only one of the high-cardinality encoding
                parameters is set.
        """
        threshold = self.high_cardinality_threshold
        hc_encoding = self.high_cardinality_encoding
        if (threshold is None) != (hc_encoding is None):
            raise ValueError(
                "general.high_cardinality_threshold and "
                "general.high_cardinality_encoding must be set together: "
                f"got threshold={threshold!r}, encoding={hc_encoding!r}. "
                "Provide both parameters to enable automatic high-cardinality "
                "encoding, or omit both to disable it."
            )
        return self

    @model_validator(mode="after")
    def _check_threads_timeouts(self) -> GeneralCfg:
        """Forbid timeouts in threads mode.

        In ``parallel_mode='threads'`` timeouts are meaningless and harmful:
                an already running Python function cannot be interrupted, so a soft
                timeout does not save time — it only forces the phase to wait for
                ``shutdown(wait=True)`` completion of all algorithms and then
                discards their ready results. Therefore setting any of
                ``phase_timeout`` / ``task_timeout`` together with
                ``parallel_mode='threads'`` is a configuration error (a breaking
                change; backward compatibility is not preserved).

                Returns:
                    GeneralCfg: The validated settings object.

                Raises:
                    ValueError: If ``parallel_mode='threads'`` and at least one of the
                        timeouts ``phase_timeout`` / ``task_timeout`` is set.
        """
        if self.parallel_mode == "threads" and (
            self.phase_timeout is not None or self.task_timeout is not None
        ):
            offenders = []
            if self.phase_timeout is not None:
                offenders.append(f"general.phase_timeout={self.phase_timeout!r}")
            if self.task_timeout is not None:
                offenders.append(f"general.task_timeout={self.task_timeout!r}")
            raise ValueError(
                "timeouts are not supported in parallel_mode='threads': an "
                "already-running Python function cannot be "
                "interrupted, so a soft timeout does not save time and only "
                "discards the ready results of completed algorithms. "
                f"Offending settings: {', '.join(offenders)}. "
                "Remove the timeouts or switch general.parallel_mode to "
                "'processes' (where timeouts are hard and enforced)."
            )
        return self


# ──────────────── oversampling ──────────────── #
class OversamplingAlgorithm(str, Enum):
    random = "random"
    smote = "smote"
    adasyn = "adasyn"


class OversamplingCfg(BaseModel):
    """Class balancing and data synthesis parameters.

    Provides configuration of methods for resolving the target-variable imbalance.
    Uses an aliases mechanism to map flat YAML keys into a structured object.
    Attributes:
        enable (bool): Global flag enabling oversampling.
        multiplier (float): Target sample-increase factor.
        algorithm (OversamplingAlgorithm): Selected generation method
            (SMOTE, ADASYN, etc.).
    """

    # accept alias names
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    enable: bool = Field(
        default=False,
        alias="data_oversampling",
        description=("Flag enabling data balancing. Applies only to the training set"),
    )
    multiplier: float = Field(
        default=1.0,
        alias="data_oversampling_multiplier",
        ge=1.0,
        description="How many times to increase the number of minority-class examples",
    )
    algorithm: OversamplingAlgorithm = Field(
        default=OversamplingAlgorithm.random,
        alias="data_oversampling_algorithm",
        description="Algorithm for synthesizing new data (Random, SMOTE, ADASYN)",
    )

    @model_validator(mode="after")
    def _validate_oversampling_logic(self) -> OversamplingCfg:
        if self.enable and self.multiplier == 1.0:
            logging.getLogger(__name__).warning(
                "Oversampling multiplier = 1 ➜ class balance will not change."
            )
        return self


# ───────────────── algorithms ───────────────── #


class AlgoCfg(BaseModel):
    """Technical configuration of a specific ML algorithm in the pipeline.

    Stores the algorithm enablement settings, overridden search spaces,
    and paths to the trainer and tuner modules.
    Attributes:
        enable (bool): Flag for the algorithm's participation in the current experiment.
        limit_hyperparameters (bool): Reduced search-space mode.
        hyperparameters (Dict | None): Custom parameter bounds
            overriding the default values.
        tuner (str): Dotted path to the optimizer module.
        trainer_module (str): Dotted path to the model training implementation.
    """

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=True, description="Whether to use this algorithm in the AutoML pipeline"
    )
    limit_hyperparameters: bool = Field(
        default=False, description=("Limit the hyperparameters of the search space")
    )
    hyperparameters: dict[str, SearchSpaceEntry] | None = Field(
        default=None,
        description=(
            "Keys must correspond to the algorithm's allowed hyperparameters: "
            "the constructor parameters of the sklearn estimator (including legacy "
            "names from LEGACY_PARAM_MAPPINGS). The source of truth is "
            "models.get_allowed_hyperparameters."
        ),
    )
    preprocessing: PreprocessingOverride | None = Field(
        default=None,
        description=(
            "Explicit override of the feature preprocessing preset for this "
            "algorithm. Set partially or fully: 'imputation_strategy' "
            "('mean'|'median') and/or 'scaling' ('standard'|'robust'|'none'). "
            "Takes precedence over the automatic choice based on the algorithm "
            "class. None — automatic selection."
        ),
    )
    tuner: str | None = Field(
        default="configurable_automl_engine.tuner",
        description="Path to the tuner module for hyperparameter optimization",
    )
    trainer_module: str | None = Field(
        default="configurable_automl_engine.trainer",
        description=(
            "Dotted path to the module containing the `ModelTrainer` class"
            "(for example, 'configurable_automl_engine.trainer')."
        ),
    )

    def get_required_package(self, algo_name: str) -> str | None:
        """Determine the name of the external Python package required by the algorithm.

        Lookup logic:
        1. Consults the global `ALGO_PACKAGE_MAPPING`.
        2. Maps the internal algorithm name (for example, 'xgboost')
            to its PyPI name.

        Args:
            algo_name (str): Unique identifier of the algorithm.
        Returns:
            Optional[str]: Name of the package to install via pip, or None
                if no dependency is defined.
        """
        return ALGO_PACKAGE_MAPPING.get(algo_name)

    def get_unknown_hyperparameters(self, algo_name: str) -> list[str]:
        """Return the hyperparameters incompatible with the given algorithm.

        Checks the keys of `self.hyperparameters` against the allowed set of
        estimator constructor parameters (see
        `models.get_allowed_hyperparameters`), including legacy names from
        `LEGACY_PARAM_MAPPINGS`. If the allowed set cannot be determined
        (the algorithm is unknown, the dependency is not installed, or the
        constructor accepts **kwargs), the check is skipped (soft fallback).

        Args:
            algo_name (str): Unique identifier of the algorithm.
        Returns:
            List[str]: List of invalid keys. An empty list means everything is correct.
        """
        if self.hyperparameters is None:
            return []
        allowed = get_allowed_hyperparameters(algo_name)

        if allowed is None:
            return []

        return [k for k in self.hyperparameters if k not in allowed]

    @field_validator("tuner", "trainer_module")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if v is None:
            return v
        if not _DOTTED_PATH_RE.fullmatch(v):
            raise ValueError(
                f"'{v}' is not a valid dotted path "
                "(expecting 'package.module' or 'a.b.c.Class' format)"
            )
        return v


class _AlgorithmsConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


AlgorithmsConfig = create_model(
    "AlgorithmsConfig",
    __base__=_AlgorithmsConfigBase,
    **{name: (AlgoCfg | None, Field(default=None)) for name in AVAILABLE_ALGORITHMS},
)  # type: ignore[call-overload]

if TYPE_CHECKING:
    # use a static alias for mypy
    AlgorithmsConfigType = _AlgorithmsConfigBase
else:
    # runtime — dynamic create_model
    AlgorithmsConfigType = AlgorithmsConfig


# ─────────────────── root ──────────────────── #
class Config(BaseModel):
    """Root object of the whole AutoML configuration system.

    Aggregates all configuration sections and performs the final validation
    of the parameter graph integrity, including system dependency checks.
    Attributes:
        general (GeneralCfg): General experiment parameters.
        oversampling (OversamplingCfg): Data preprocessing settings.
        algorithms (AlgorithmsConfig): Registry of available and active algorithms.
    """

    model_config = ConfigDict(extra="forbid")

    general: GeneralCfg = Field(
        ..., description="General experiment and validation settings"
    )
    oversampling: OversamplingCfg = Field(
        default=OversamplingCfg(), description="Data balancing settings"
    )
    algorithms: AlgorithmsConfigType = Field(
        ...,
        description=(
            "Dictionary of algorithms, where the key is the algorithm name "
            "(for example, 'xgboost', 'random_forest')"
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

    @model_validator(mode="after")
    def _check_algorithm_dependencies(self) -> Config:
        # model_fields works through an instance too, but it is deprecated
        # (PydanticDeprecatedSince211, removal in V3.0) — use the class instead
        for name in type(self.algorithms).model_fields:
            # Get the attribute value (could be None if not provided in data)
            algo_cfg = getattr(self.algorithms, name)

            # Skip if the algorithm entry is missing (None)
            if algo_cfg is None:
                continue

            # Ensure it has an 'enable' attribute before checking it
            # If it's a Pydantic model, check attribute, otherwise get() as dict
            enabled = getattr(algo_cfg, "enable", False)

            if enabled:
                # Safely get the required package
                required_pkg = getattr(
                    algo_cfg, "get_required_package", lambda n: None
                )(name)

                if required_pkg and not is_installed(required_pkg):
                    raise ValueError(
                        f"Algorithm '{name}' is enabled, but the package "
                        f"'{required_pkg}' is not installed. "
                        f"Please run: pip install {required_pkg}"
                    )
        return self

    @model_validator(mode="after")
    def _check_hyperparameter_compatibility(self) -> Config:
        errors = []
        for name in type(self.algorithms).model_fields:
            algo_cfg = getattr(self.algorithms, name)
            if algo_cfg is None or not algo_cfg.enable:
                continue
            unknown = algo_cfg.get_unknown_hyperparameters(name)
            if unknown:
                allowed = sorted(get_allowed_hyperparameters(name) or set())
                errors.append(
                    f"Algorithm '{name}': unknown hyperparameters {unknown}. "
                    f"Allowed parameters: {allowed}"
                )
        if errors:
            raise ValueError("\n".join(errors))
        return self

    @model_validator(mode="after")
    def _normalize_hyperparameter_types(self) -> Config:
        """Re-resolve the type of inferred (not explicitly typed) ranges.

        The type of a range is taken from the parameter's type — the default
        search space or the estimator signature — not from how the boundary
        numbers are written. So ``[1, 50]`` for a real-valued parameter
        (e.g. ``C`` of an SVR, ``alpha`` of a Tweedie) stays a real range
        instead of silently becoming an integer range. A parameter whose
        default space entry is ``float_log`` (e.g. ``alpha_1``/``lambda_1``
        of an ARD regression), as well as scale parameters (``alpha``,
        ``C``, ``epsilon``, ``gamma``, ``learning_rate``), default to the
        logarithmic scale without an explicit type. Explicitly typed ranges
        (``[1, 50, "int"]``) are left untouched.
        """
        from configurable_automl_engine.common.hyperopt_defaults import (
            normalize_hyperparameter_entry,
        )

        for name in type(self.algorithms).model_fields:
            algo_cfg = getattr(self.algorithms, name)
            if algo_cfg is None or not algo_cfg.enable:
                continue
            hyperparams = algo_cfg.hyperparameters
            if not hyperparams:
                continue
            # Update the existing dict in place: the Config was built from the
            # input data, and other code may hold a reference to the same
            # hyperparameters mapping (e.g. through model_dump) — replacing
            # the dict would silently detach those references.
            for param, entry in list(hyperparams.items()):
                normalized = normalize_hyperparameter_entry(name, param, entry)
                if normalized is not entry:
                    hyperparams[param] = normalized
        return self


# ────────────────── API ────────────────────── #
def read_config(path: str | Path) -> Config:
    """Load, parse, and validate an experiment configuration file.

    How it works:
    1. Opens the file at the given path in UTF-8 encoding.
    2. Safely loads the YAML structure into a Python dictionary.
    3. Triggers cascading Pydantic validation
        to build a typed `Config` object.
    Args:
        path (str | Path): Path to a .yaml or .yml configuration file.
    Returns:
        Config: The root configuration object, ready for use in the AutoML engine.

    Raises:
        FileNotFoundError: If the file at the given path is not found.
        ValidationError: If the file structure does not match the schema or
            the parameter logic is violated.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
