"""Config schema of the ``general`` section.

Holds ``GeneralCfg`` — the fields only: its cross-field validators are thin
``@model_validator`` methods that call the rules in
:mod:`.general_rules` — and the types of its fields (``ComparisonMetric``,
``InfPolicy``, ``DatetimePolicy``, the ``"auto"`` hold-out constants). It
builds on the section modules (phases, Sanity Gate, complexity guard) and
:mod:`.general_rules`; within the config package only :mod:`.root` imports
it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from configurable_automl_engine.common.definitions import (
    ParallelStrategy,
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.metrics import (
    AVAILABLE_METRICS,
)
from configurable_automl_engine.feature_selection import (
    FeatureSelectionCfg,
)
from configurable_automl_engine.preprocessing import EncodingStrategy
from configurable_automl_engine.training_engine.config import (
    general_rules as rules,
)
from configurable_automl_engine.training_engine.config.complexity_guard import (
    ComplexityGuardCfg,
)
from configurable_automl_engine.training_engine.config.phases import (
    HPOPhaseCfg,
    PruningCfg,
    _default_phases,
)
from configurable_automl_engine.training_engine.config.sanity_gate import (
    SanityGateCfg,
)

# Build the type on the fly. *AVAILABLE_METRICS unpacks the list into Literal arguments
ComparisonMetric = Literal[*AVAILABLE_METRICS]  # type: ignore

#: ``general.holdout_fraction: "auto"`` reserves this fraction of the rows as
#: the outer hold-out when the input has at least ``AUTO_HOLDOUT_MIN_ROWS``
#: rows (so the hold-out has at least 20 rows); on smaller data no hold-out is
#: reserved and the honest estimate is the BBC-CV ``bbc_score``.
AUTO_HOLDOUT_FRACTION = 0.2
AUTO_HOLDOUT_MIN_ROWS = 100

#: Policy for ±inf values in numeric features: ``"as_missing"`` (default) —
#: replace with NaN and warn; ``"error"`` — reject with ValueError.
InfPolicy = Literal["as_missing", "error"]

#: Policy for datetime/timedelta columns: ``"expand"`` (default) — replace
#: with derived components (year/month/day/dayofweek, hour if non-zero) and
#: warn; ``"drop"`` — drop the columns and warn; ``"error"`` — reject with
#: ValueError.
DatetimePolicy = Literal["expand", "drop", "error"]


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
            Defaults to ``all_algorithms`` + ``select(cv_best, winner)``.
        validation_strategy (ValidationStrategy): Evaluation method
            ('train_test_split', 'k_fold', 'loo' or 'auto').
        n_folds (int): Number of splits for cross-validation.
        random_state (int | None): The engine seed driving data splits, model
            seeds and the TPE sampler base seed (``random_state +
            phase_ordinal``); recorded in the report. None — the run is not
            reproducible (a WARNING is logged).
        categorical_encoding (EncodingStrategy): Encoding strategy for
            categorical features ('one_hot', 'ordinal', 'target', 'frequency'
            or 'hashing'). Default is 'one_hot'.
        parallel_strategy (str): Parallelization level (only by algorithms).
        max_workers (int | None): Limit on threads or processes.
        parallel_mode (str): Technical execution mode ('threads' or 'processes').
        start_method (str | None): How the worker processes start in
            'processes' mode ('forkserver', 'spawn' or 'fork'); None — the
            engine default ('forkserver' where available, else 'spawn').
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
        default_factory=_default_phases,
        description=(
            "List of sequential pipeline steps. Default: one HPO phase over "
            "all algorithms followed by picking the best cross-validation "
            "score: [{action: all_algorithms, n_trials: 50}, "
            "{action: select, strategy: cv_best, output: winner}]"
        ),
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
    random_state: int | None = Field(
        default=42,
        description=(
            "The engine seed: drives the data splits (CV folds / hold-out), "
            "the model seeds (via the ModelTrainer and the model factory) and "
            "the TPE sampler base seed (random_state + hpo_phase_ordinal). "
            "It is recorded in the training report under the 'random_state' "
            "key. None — a non-deterministic run: the data splits are not "
            "fixed and stochastic models draw a fresh RNG at fit time; a "
            "WARNING is logged that the run is not reproducible."
        ),
    )
    holdout_fraction: (
        Annotated[float, Field(gt=0.0, lt=1.0)] | Literal["auto"] | None
    ) = Field(
        default="auto",
        description=(
            "Fraction of rows (in (0, 1)) reserved as an outer hold-out "
            "BEFORE hyperparameter optimization: these rows never participate "
            "in model selection and are used only to report the honest "
            'out-of-sample estimate result["holdout_score"] of the winner. '
            "The final model is still trained on ALL data (the hold-out rows "
            "are included in the final fit). 'auto' (default) — reserve "
            f"{AUTO_HOLDOUT_FRACTION:g} of the rows when the input has at "
            f"least {AUTO_HOLDOUT_MIN_ROWS} rows; on smaller data no hold-out "
            "is reserved. None — no outer hold-out is reserved and the key "
            'is not reported. Without a hold-out result["score"] and '
            'result["oof_score"] are computed on the folds that selected the '
            "algorithm and hyperparameters and are optimistic; the honest "
            'estimate is then the bias-corrected (BBC-CV) result["bbc_score"].'
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
        default=50,
        ge=0,
        description=(
            "Cardinality threshold for the automatic mode (>= 0). Columns "
            "with a number of unique values strictly greater than the threshold "
            "are encoded with the high_cardinality_encoding strategy (default "
            "'frequency'), the rest — with the categorical_encoding strategy. "
            "Must be set together with high_cardinality_encoding. The default "
            "50 enables the automatic high-cardinality mode out of the box; to "
            "disable it, set both this field and high_cardinality_encoding to "
            "null explicitly."
        ),
    )
    high_cardinality_encoding: EncodingStrategy | None = Field(
        default="frequency",
        description=(
            "Encoding strategy for high-cardinality columns (above "
            "high_cardinality_threshold; default 'frequency' — the relative "
            "category frequency, which does not expand the feature space). "
            "Must be set together with high_cardinality_threshold. The default "
            "'frequency' keeps the memory/CPU footprint low on high-cardinality "
            "columns; set this field and the threshold to null to disable the "
            "automatic mode."
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
    start_method: Literal["forkserver", "spawn", "fork"] | None = Field(
        default=None,
        description=(
            "How the worker processes start in parallel_mode='processes': "
            "'forkserver', 'spawn' or 'fork'. None — the engine default: "
            "'forkserver' where the platform has it (Linux, macOS), otherwise "
            "'spawn'. With 'forkserver' and 'spawn' the workers re-import the "
            "main script, so a script that calls train_best_model must guard "
            "its top-level code with `if __name__ == \"__main__\":`. 'fork' "
            "(POSIX only) does not re-import the script but copies the "
            "multi-threaded parent process, which can deadlock a worker. "
            "Setting it with parallel_mode='threads' is a config error."
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
    inf_policy: InfPolicy = Field(
        default="as_missing",
        description=(
            "Policy for infinite values (±inf) in numeric features: "
            "'as_missing' (default) — replace them with NaN and log a "
            "WARNING listing the columns and counts (missing values are "
            "imputed by the preprocessor); 'error' — reject the data with a "
            "ValueError naming the columns."
        ),
    )
    datetime_policy: DatetimePolicy = Field(
        default="expand",
        description=(
            "Policy for datetime/timedelta columns (not handled by any "
            "transformer and previously silently dropped): 'expand' (default) "
            "— replace each column with derived numeric components "
            "(year, month, day, dayofweek, and hour if it is non-zero) and log "
            "a WARNING listing the generated columns; 'drop' — remove the "
            "columns and log a WARNING; 'error' — reject the data with a "
            "ValueError naming the columns."
        ),
    )
    drop_id_like_columns: bool = Field(
        default=False,
        description=(
            "Whether ID-like columns (object/string with nunique/n >= 0.95, "
            "or a unique monotonic integer column) are dropped from the "
            "feature matrix. False (default) — such columns are only reported "
            "with a WARNING naming them; true — they are removed. ID-like "
            "columns are typically uninformative for regression and can slow "
            "training down dramatically."
        ),
    )

    @field_validator("additional_metrics")
    @classmethod
    def _deduplicate_additional_metrics(
        cls, v: list[ComparisonMetric]
    ) -> list[ComparisonMetric]:
        """Remove duplicate additional metrics (see ``general_rules``)."""
        return rules.deduplicate_additional_metrics(v)

    @model_validator(mode="after")
    def _exclude_comparison_from_additional(self) -> GeneralCfg:
        """Drop the comparison metric from the additional metrics."""
        self.additional_metrics = rules.exclude_comparison_from_additional(
            self.comparison_metric, self.additional_metrics
        )
        return self

    @model_validator(mode="after")
    def _check_n_folds(self) -> GeneralCfg:
        """Check the validation and serialization settings."""
        rules.check_n_folds(
            self.serialization_format, self.n_folds, self.validation_strategy
        )
        return self

    @model_validator(mode="after")
    def _check_pruning_consistency(self) -> GeneralCfg:
        """Check the early stopping settings against the validation."""
        rules.check_pruning_consistency(
            self.pruning, self.validation_strategy, self.n_folds
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
        """
        rules.validate_phase_pipeline(self.phases)
        return self

    @model_validator(mode="after")
    def _check_high_cardinality_encoding(self) -> GeneralCfg:
        """Check that the high-cardinality settings are set together."""
        rules.check_high_cardinality_encoding(
            self.high_cardinality_threshold, self.high_cardinality_encoding
        )
        return self

    @model_validator(mode="after")
    def _check_start_method(self) -> GeneralCfg:
        """Check that ``start_method`` applies and is available here."""
        rules.check_start_method(self.start_method, self.parallel_mode)
        return self

    @model_validator(mode="after")
    def _check_threads_timeouts(self) -> GeneralCfg:
        """Forbid timeouts in threads mode."""
        rules.check_threads_timeouts(
            self.parallel_mode, self.phase_timeout, self.task_timeout
        )
        return self
