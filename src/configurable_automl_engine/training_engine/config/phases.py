"""Config schema of the pipeline steps (``general.phases``) and of pruning.

Holds ``HPOPhaseCfg`` (one search or select step), the step types
(``HPOPhaseAction``, ``SelectStrategy``, ``SelectOutput``), the early
stopping settings (``PruningStrategy``, ``PruningCfg``) and the default
pipeline (``_default_phases``). The module must not import the other config
modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic_core import PydanticCustomError

# Stable migration error codes: tests may rely on ``error['type']`` from
# ValidationError instead of the hint text, which is subject to change.
ERR_REMOVED_REFINE_WINNER = "removed_refine_winner_action"

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
#: Default number of HPO trials of a phase when ``n_trials`` is omitted.
DEFAULT_PHASE_N_TRIALS = 50


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
            (``all_algorithms`` / ``refine``), default
            ``DEFAULT_PHASE_N_TRIALS``. Ignored for ``select``.
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
        default=DEFAULT_PHASE_N_TRIALS,
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
                "See 'Migrating older configs' in API_REFERENCE.md.",
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


# ─────────────────── general ─────────────────── #
def _default_phases() -> list[HPOPhaseCfg]:
    """Build the default pipeline used when ``general.phases`` is omitted.

    Returns:
        list[HPOPhaseCfg]: An ``all_algorithms`` HPO phase followed by a
        terminal ``select`` step picking the best cross-validation score.
    """
    return [
        HPOPhaseCfg(name="search", action="all_algorithms"),
        HPOPhaseCfg(
            name="final_select",
            action="select",
            strategy="cv_best",
            output="winner",
        ),
    ]
