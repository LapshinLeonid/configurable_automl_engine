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
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from configurable_automl_engine.common.custom_metrics import CustomMetricSpec
from configurable_automl_engine.common.definitions import (
    ParallelStrategy,
    SerializationFormat,
    TargetTransform,
    ValidationStrategy,
)
from configurable_automl_engine.common.metrics import (
    AVAILABLE_METRICS,
)
from configurable_automl_engine.feature_selection import (
    FeatureSelectionCfg,
)
from configurable_automl_engine.preprocessing import EncodingStrategy
from configurable_automl_engine.preprocessing_presets import PreprocessingOverride
from configurable_automl_engine.training_engine.config import (
    general_rules as rules,
)
from configurable_automl_engine.training_engine.config.complexity_guard import (
    ComplexityGuardCfg,
)
from configurable_automl_engine.training_engine.config.ensemble import (
    EnsembleCfg,
)
from configurable_automl_engine.training_engine.config.phases import (
    HPOPhaseCfg,
    PruningCfg,
    _default_phases,
)
from configurable_automl_engine.training_engine.config.sanity_gate import (
    SanityGateCfg,
)

#: A metric reference: a built-in metric name (``AVAILABLE_METRICS``) or the
#: name of a ``general.custom_metrics`` entry. Membership is a cross-field
#: rule (``general_rules.check_metric_references``).
ComparisonMetric = str

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
        comparison_metric (ComparisonMetric): Main metric for ranking models
            (a built-in name or a ``custom_metrics`` key).
        additional_metrics (List[ComparisonMetric]): Additional informational
            metrics, computed for the final model with the same validation method
            used for model comparison (hold-out/CV, not train score).
        custom_metrics (dict[str, CustomMetricSpec]): User metrics and scorers
            loaded by a dotted import path, keyed by the name the metric
            fields refer to.
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
        preprocessing (PreprocessingOverride | None): Run-level default
            override of the adaptive feature preprocessing preset, applied
            to every algorithm and merged per field with its per-algorithm
            block (fields the per-algorithm block leaves unset come from
            here).
            Precedence per field: class preset < general block <
            per-algorithm block. None — no run-level default.
        target_transform (TargetTransform): Transformation of the target
            ``y`` inside the ``model`` step of every trial pipeline, the
            final trainer and the hold-out trainer ('none' / 'log1p' /
            'boxcox' / 'yeo-johnson' / 'auto'); applied identically everywhere so
            the selected hyperparameters match the final model. GLM
            algorithms with a restricted loss domain are validated against
            the transformed target before HPO. The 'auto' mode decides the
            concrete method from the target skewness (see
            ``target_transform_skew_threshold``): with a reserved outer
            hold-out — once, before the HPO phases, on the train part of
            the split; without a hold-out — inside every cross-validation
            fold (each fit resolves its own method from its own y_train,
            so the validation-fold values cannot influence the choice).
        target_transform_skew_threshold (float): The ``|skew(y)|`` threshold
            of the ``general.target_transform: 'auto'`` decision: a target
            whose absolute Fisher skewness exceeds the threshold receives
            ``log1p`` (``min(y) > -1``) or ``yeo-johnson``, a more symmetric
            target keeps ``none``. Must be > 0.
    """

    model_config = ConfigDict(extra="forbid")

    custom_metrics: dict[str, CustomMetricSpec] = Field(
        default_factory=dict,
        description=(
            "User-defined metrics and scorers, keyed by the metric name that "
            "comparison_metric / additional_metrics refer to. A name must "
            "match ^[a-z][a-z0-9_]*$ and must not be a built-in or sklearn "
            "metric name. Each entry names a callable by its dotted import "
            "path ('package.module:attribute'); the callable is imported and "
            "its signature is checked at config validation time."
        ),
    )
    comparison_metric: ComparisonMetric = Field(
        default="r2",
        description=(
            "Metric for comparing models and choosing the best one: one of "
            + ", ".join(AVAILABLE_METRICS)
            + ", or the name of a custom_metrics entry"
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
            "Valid values are the same as for comparison_metric (built-in "
            "names and custom_metrics names). "
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
    ensemble: EnsembleCfg = Field(
        default_factory=EnsembleCfg,
        description=(
            "Experimental weighted-average ensemble of the top-K "
            "leaderboard models (disabled by default). When "
            "general.ensemble.enable is true, after the terminal "
            "selection the engine retrains the top-K completed "
            "algorithms on 100% of the data, averages their predictions "
            "with weights derived from the validation scores and reports "
            "the dedicated 'ensemble' section in the result. The winner "
            "and the winner artifact are NOT affected"
        ),
    )
    preprocessing: PreprocessingOverride | None = Field(
        default=None,
        description=(
            "Run-level default override of the adaptive feature preprocessing "
            "preset, applied to EVERY algorithm and merged per field with "
            "the algorithm's own block (algorithms.<name>.preprocessing): a "
            "field the algorithm block leaves null is taken from here. "
            "Reuses the same model as the per-algorithm block: "
            "'imputation_strategy' ('mean'|'median'), 'scaling' "
            "('standard'|'robust'|'quantile'|'none' or an object form) and "
            "'outliers' ('none'|'quantile'|'iqr' or an object form); "
            "fields left as null keep the automatic selection based on the "
            "algorithm class. Precedence per field: class preset < "
            "general.preprocessing < algorithms.<name>.preprocessing — a "
            "field set at a higher level replaces the whole field from the "
            "lower level (nested values are never merged). None — no "
            "run-level default. Unknown keys are rejected at config "
            "validation."
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
            "transformer): 'expand' (default) "
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
    target_transform: TargetTransform = Field(
        default=TargetTransform.none,
        description=(
            "Transformation of the target y inside the 'model' step of the "
            "pipeline, applied identically in the HPO trials, in the final "
            "trainer and in the hold-out trainer: 'none' (default) — no "
            "transformation; 'log1p' — log(1 + y) (requires min(y) > -1); "
            "'boxcox' — the Box-Cox power transformation (requires "
            "min(y) > 0); 'yeo-johnson' — the Yeo-Johnson power "
            "transformation (any real y); 'auto' — the engine decides "
            "from the target skewness: |skew(y)| > "
            "target_transform_skew_threshold selects 'log1p' when "
            "min(y) > -1, otherwise 'yeo-johnson'; a more symmetric target "
            "keeps 'none'. The decision point follows the estimation "
            "design: with a reserved outer hold-out the method is decided "
            "once before HPO on the train part of the split (the resolved "
            "concrete method — never 'auto' — is what the HPO, the final "
            "trainer, the artifact metadata and the ExperimentResult see). "
            "Without a reserved hold-out (holdout_fraction null, or 'auto' "
            "on data below the auto-holdout threshold) the decision moves "
            "INSIDE every cross-validation fold: each fold (and the final "
            "full fit) resolves its own method from its own y_train, so "
            "the validation-fold target values cannot influence the "
            "transformation they are scored with; the config and the "
            "artifact metadata then report the 'auto' placeholder, and the "
            "fitted pipelines carry the resolved method per fit. The "
            "hyperparameters are optimized for the model trained on the "
            "TRANSFORMED target, so the search matches the final model. A "
            "domain violation (min(y) outside the method's domain) fails "
            "once before the first trial with a concrete method, and "
            "per-fold with the fold-local resolution. GLM algorithms with "
            "a restricted loss domain (gammaregressor, poissonregressor, "
            "tweedieregressor/glm) are validated by the target-domain "
            "guard: against the TRANSFORMED target with a concrete "
            "method, and against the UNTRANSFORMED y with 'auto' (the "
            "fold-local resolution cannot assume a method; a GLM that "
            "fails on a transformed fold is penalized by the ordinary HPO "
            "failure policy). 'predict' returns values in the original "
            "scale (the inverse transformation clips to the train range)."
        ),
    )
    target_transform_skew_threshold: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "The |skew(y)| threshold of the 'auto' mode of "
            "general.target_transform: a target whose absolute Fisher "
            "skewness exceeds the threshold receives a transformation "
            "('log1p' when min(y) > -1, otherwise 'yeo-johnson'); a more "
            "symmetric target keeps 'none'. Must be > 0."
        ),
    )

    @field_validator("custom_metrics", mode="before")
    @classmethod
    def _validate_custom_metric_names(cls, v: Any) -> Any:
        """Check the custom metric names (see ``general_rules``)."""
        return rules.check_custom_metric_names(v)

    @field_validator("additional_metrics")
    @classmethod
    def _deduplicate_additional_metrics(
        cls, v: list[ComparisonMetric]
    ) -> list[ComparisonMetric]:
        """Remove duplicate additional metrics (see ``general_rules``)."""
        return rules.deduplicate_additional_metrics(v)

    @model_validator(mode="after")
    def _check_metric_references(self) -> GeneralCfg:
        """Check that every metric name is built-in or declared."""
        rules.check_metric_references(
            self.comparison_metric, self.additional_metrics, self.custom_metrics
        )
        return self

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
