# Architecture

The engine is a pipeline of steps over a few state objects, each with a single
owner:

* `hpo.runner._OptimizeRunner` — one HPO run of one algorithm (`tuner.optimize`);
* `trainer.ModelTrainer` — the training, validation and saving of one model;
* `training_engine.state.PhaseContext` — the state of the phase pipeline,
  owned by `training_engine/component.py`.

Only the owner mutates its state. The computational steps are pure functions:
they read their inputs and return a frozen result — a plan
(`hpo/plans.py`), a trainer state group (`trainer_state.py`) or an outcome
(`SelectOutcome`, `WorkerOutcome`, `TargetDomainOutcome` in
`training_engine/state.py`) — and the owner assigns it in one visible line.
`tests/test_ownership.py` checks the rule statically, and the pure steps have
direct unit tests (inputs unchanged, two calls give equal results). Every
module logs under its own name (`logging.getLogger(__name__)`).

## Import layers

The modules form four layers. Each layer imports only from itself and from
the layers below it — top-level, function-level and `TYPE_CHECKING` imports
alike — and there are no import cycles:

| Layer | Modules |
|---|---|
| 3 — orchestration | `training_engine/` (config schema, phases, selection, artifact saving, `run_parallel`) |
| 2 — engines | `hpo/`, `tuner.py`; `trainer.py`, `trainer_*.py`. HPO and the trainer modules do not import each other; `tuner.py` is the façade over `hpo` |
| 1 — ML building blocks | `models`, `preprocessing_presets`, `preprocessing`, `oversampling`, `validation`, `feature_selection`, `isotonic`, `experiment` |
| 0 — shared utilities | `common/` (and the generated `_version`) |

The package `__init__` is the public façade on top; it loads
`training_engine.component` (and with it Optuna) only on the first call of
`train_best_model`. The rules are `import-linter` contracts in
`pyproject.toml`, checked in CI by `lint-imports`; package imports inside
functions are not allowed (except the lazy `train_best_model` proxy in
`training_engine/__init__.py`). If a lower module needs something from above,
move that thing down.

Inside the `preprocessing` package the import direction is:
`builder -> features -> columns`, `encoders -> columns` and
`builder -> encoders -> columns`; `outliers` must not import any other
preprocessing module (it is a pure, self-contained transformer; its classes
are pickled inside saved models, so its module path is part of the artifact
format and the module must stay where it is).

The numerical pipeline assembled by `build_preprocessor` is
`[Winsorizer] -> SimpleImputer -> [scaler]`: the winsorization
(`outliers`, an `OutlierSpec` from `outliers_presets`) comes strictly before
the imputer, so the imputed value is not biased by the extreme values and the
missing-value indicator columns are not clipped. The preset that decides
whether the winsorization step exists, which imputation strategy and which
scaler are used, is resolved once per algorithm by
`resolve_preprocessing_preset` (`preprocessing_presets`) with the precedence
"class preset < run-level `general.preprocessing` < per-algorithm override",
and the same resolution is shared by the HPO trial pipeline and the final
trainer.

The step order shared by the HPO trial pipeline
(`hpo/trial_pipeline.py::assemble_estimator`) and the final `ModelTrainer`
pipeline is
`preprocessor → [isotonic_selector] → [feature_selector] → [sampler] →
model`, where for a non-`none` target transformation
(`HpoRequest.target_transform` / `general.target_transform`) the last
`model` step is the regressor wrapped into a `TransformedTargetRegressor`
(`wrap_with_target_transform`): the HPO trials, the final trainer and the
hold-out trainer all fit on the transformed `y`, so the selected
hyperparameters match the returned model. The target-domain guard of the
GLM algorithms (`hpo/target_domain.py`, applied once before the first HPO
phase) is checked against the transformed target when a transformation is
configured; a domain violation of the transformation itself fails once
before the first trial (`validate_target_for_transform`).

# Project Structure

```
├── benchmarks                              # Quality benchmarks of Sanity Gate, Complexity Guard, oversampling
├── scripts
│   └── compare_with_base.py                # Differential behavior check against the base commit (optional CI workflow: manual run or the `compare-with-base` PR label)
├── src                                     # Project source code root
│   └── configurable_automl_engine          # Main AutoML engine package
│       ├── common                          # Layer 0: shared utilities (imports nothing above common)
│       │   ├── custom_metrics.py           # CustomMetricSpec: user metrics/scorers by dotted path, scorer cache, process-local registry
│       │   ├── datetime_features.py        # Datetime/timedelta input columns: the datetime policy, expansion at fit and predict time
│       │   ├── definitions.py              # Constants, enums and literals (validation, encoding, serialization)
│       │   ├── dependency_utils.py         # Optional library and dependency checks
│       │   ├── errors.py                   # TrainingError
│       │   ├── log_setup.py                # Centralized logging setup (setup_logging)
│       │   ├── metrics.py                  # Evaluation metrics (built-in and registered custom), BBC-CV estimate
│       │   ├── serialization_utils.py      # Model/pipeline serialization logic
│       │   ├── settings.py                 # Setting groups shared by HPO and the trainer (encoding, oversampling, input policies)
│       │   ├── shared_frame.py             # SharedDataFrame: a DataFrame passed between processes through shared memory
│       │   └── validation_utils.py         # Validation strategy choice, target checks, validate_features, effective train size
│       ├── hpo                             # Layer 2: hyperparameter optimization engine behind tuner.optimize
│       │   ├── complexity_guard.py         # Adaptive tightening of tree/boosting search spaces on small data
│       │   ├── complexity_rules.py         # The complexity guard rules (Rule, _rule_* functions, rule tables)
│       │   ├── default_spaces.py           # Default search spaces, clipping, and constraints
│       │   ├── evaluation.py               # Trial scoring (score_trial): hold-out, fold-wise with pruning reports, pooled OOF
│       │   ├── plans.py                    # Frozen plans returned by the pure setup steps (validation, preprocessing, ...)
│       │   ├── pruning.py                  # Early stopping (pruning) plan
│       │   ├── request.py                  # HpoRequest: the settings of one HPO run
│       │   ├── runner.py                   # _OptimizeRunner: run loop, circuit breaker, best trial; plan_validation
│       │   ├── search_space.py             # Search space plan, complexity guard, boundary warnings
│       │   ├── target_domain.py            # Target-domain guard for GLMs (Poisson, Gamma, Tweedie)
│       │   ├── trial_pipeline.py           # Preprocessing plan and trial pipeline assembly
│       │   └── types.py                    # Exceptions, OptimizeResult, worst-score constants
│       ├── experiment                      # Layer 1: experiment result model, run history, leaderboard, reporters, object facade
│       │   ├── ensemble.py                 # Experimental top-K ensemble core: select_top_k, compute_weights, resolve_metric_direction (the shared public resolver), EnsembleModel (save/load, report section); EnsembleModelError is the module's base error
│       │   ├── experiment.py               # AutoMLExperiment: the object facade (run / leaderboard / best_model / ensemble_model / report / export / explain); explain() stores its JSON-safe payload on the run result (the last call wins)
│       │   ├── explain.py                  # Winner explainability: permutation importance, optional SHAP, Markdown rendering
│       │   ├── export.py                   # export_experiment: the artifact bundle of a run into a directory (atomic two-phase swap, <run_id>_* names; the model AND ensemble artifact copies, the stored explain payload, metadata with the artifact source paths)
│       │   ├── leaderboard.py              # Leaderboard: the algorithm ranking table built from an ExperimentResult
│       │   ├── reporting                   # Reporters: pure consumers of ExperimentResult (no engine, no mlflow imports)
│       │   │   ├── base.py                # Reporter (generic ABC), CompositeReporter (several reporters in one call)
│       │   │   ├── json_reporter.py       # JSONReporter: atomic full-snapshot JSON (model_dump, utf-8, overwrite)
│       │   │   ├── markdown_reporter.py   # MarkdownReporter: metadata, outcome, honest scores, leaderboard, ensemble, stored explain payload, run details
│       │   │   └── mlflow_reporter.py     # MLflowReporter (optional extra [mlflow]): the only component that knows about MLflow (lazy guarded import)
│       │   ├── result.py                   # ExperimentResult / AlgorithmRun: the central serializable result model (incl. the ensemble section and the stored explain payload)
│       │   └── store.py                    # ExperimentStore: thread-safe run history, JSONL persist, damaged-line skip
│       ├── training_engine                 # Layer 3: orchestration
│       │   ├── artifact.py                 # Trainer assembly, final fit, artifact saving and the result report
│       │   ├── component.py                # Main orchestrator: train_best_model, the phase loop; applies the step outcomes
│       │   ├── config                      # Config schema (Pydantic), one module per section
│       │   │   ├── algorithms.py          # algorithms / oversampling sections (AlgoCfg, OversamplingCfg)
│       │   │   ├── complexity_guard.py    # general.complexity_guard (ComplexityGuardCfg)
│       │   │   ├── ensemble.py            # general.ensemble (EnsembleCfg): the experimental top-K ensemble section
│       │   │   ├── general.py             # general section (GeneralCfg, fields only)
│       │   │   ├── general_rules.py       # Cross-field validation rules of the general section
│       │   │   ├── phases.py              # Pipeline steps (HPOPhaseCfg) and pruning (PruningCfg)
│       │   │   ├── root.py                # Root Config and read_config
│       │   │   └── sanity_gate.py         # general.sanity_gate (SanityGateCfg)
│       │   ├── config_parser.py            # Façade: re-exports the config schema (the public import path)
│       │   ├── data_prep.py                # Config loading, X/y preparation, hold-out, validation strategy
│       │   ├── ensemble_step.py            # The experimental top-K ensemble step: runs after the terminal select, never aborts the run; reuses experiment.ensemble.resolve_metric_direction (the single shared direction resolver)
│       │   ├── finalists.py                # Winner selection, finalists pool, training of the finalists
│       │   ├── hpo_phases.py               # HPO phase execution: the HPO request and the workers
│       │   ├── parallel_deadlines.py       # run_parallel: pure deadline policy (task timeout, queue budget)
│       │   ├── parallel_executor.py        # run_parallel: ProcessPoolExecutor adapter (forced shutdown, exit codes)
│       │   ├── parallel_logging.py         # run_parallel: forwards the workers' log records to the parent
│       │   ├── parallel_outcome.py         # run_parallel: TaskOutcome, the result of one task
│       │   ├── parallel_supervisor.py      # run_parallel: process-pool supervisor (timeout and crash restarts)
│       │   ├── parallel_transport.py       # run_parallel: shared memory / disk cache / pickle payload transport
│       │   ├── parallel_worker.py          # run_parallel: worker side (start signal, argument restore)
│       │   ├── robust_winner.py            # Robust winner: Sanity Gate audit of the finalists, fallback ranking
│       │   ├── run_result.py               # build_experiment_result: assembly of the ExperimentResult model; owner stamp (mark_result_owner)
│       │   ├── sanity_circuits.py          # Sanity Gate circuits A–D (diversity, unique, dead features, gap)
│       │   ├── sanity_gate.py              # ModelSanityGate: settings, check(), audit logging
│       │   ├── sanity_result.py            # SanityCheckResult, severity, input checks
│       │   ├── select_steps.py             # Select steps: cv_best and sanity_gate strategies (return a SelectOutcome)
│       │   ├── state.py                    # Shared pipeline dataclasses (PhaseResult, PhaseContext, step outcomes, ...)
│       │   └── thread_pool.py              # run_parallel / run_parallel_outcomes: threads runner, fallback policy
│       ├── feature_selection.py            # Layer 1: optional feature selection step and its config (FeatureSelectionCfg)
│       ├── isotonic.py                     # Layer 1: IsotonicDataTransformer, the isotonic regression data adapter
│       ├── models.py                       # Layer 1: model factory, hyperparameter cleaning
│       ├── oversampling                    # Layer 1: experimental target-aware oversampling
│       │   ├── columns.py                  # Categorical/indicator columns, Gaussian noise, dtype restoration
│       │   ├── experimental.py             # Experimental status flag and its warn-once WARNING
│       │   ├── sampler.py                  # DataOversampler and oversample()
│       │   ├── smoter.py                   # SMOTER-style generator for a regression target
│       │   └── targets.py                  # Binning of a regression target, sampling strategy, decoding
│       ├── preprocessing                   # Layer 1: feature preprocessing: winsorization (outliers), imputation, scaling, encoding
│       │   ├── builder.py                  # build_preprocessor: the single preprocessor entry point ([Winsorizer] -> SimpleImputer -> [scaler] for numeric columns)
│       │   ├── columns.py                  # ColumnNameSelector: column selection by name
│       │   ├── encoders.py                 # Target / frequency / hashing / split categorical encoders
│       │   ├── features.py                 # Feature type detection, uninformative columns, feature count estimate
│       │   ├── outliers.py                 # Winsorizer: standalone winsorization transformer (must not import the other preprocessing modules; its classes are pickled inside saved models, so the module path is part of the artifact format)
│       │   ├── quantile.py                 # _FoldSafeQuantileTransformer: QuantileTransformer with fold-size-aware n_quantiles clipping
│       │   └── target_transform.py         # TargetTransformer (log1p / boxcox / yeo-johnson), wrap_with_target_transform, the 'auto' resolution from the target skewness
│       ├── outliers_presets.py             # Layer 1: OutlierSpec specs (none / quantile / iqr), re-exported by preprocessing_presets (a sibling, not a member of the preprocessing package: the package __init__ imports the builder which imports preprocessing_presets — a member module would create an import cycle)
│       ├── preprocessing_presets.py        # Layer 1: per-algorithm preprocessing presets (imputation, scaling, outliers; precedence: class preset < general < algorithm)
│       ├── trainer.py                      # Layer 2: ModelTrainer: training, validation and saving of a single model
│       ├── trainer_artifact.py             # Layer 2: saved-artifact metadata and load-time version checks
│       ├── trainer_data.py                 # Layer 2: ModelTrainer input data: X/y, diagnostics, preprocessor, alignment
│       ├── trainer_feature_selection.py    # Layer 2: ModelTrainer feature selection decision and auto-check
│       ├── trainer_params.py               # Layer 2: ModelTrainer constructor argument validation
│       ├── trainer_quick.py                # Layer 2: the logic of the train_model() facade
│       ├── trainer_scoring.py              # Layer 2: ModelTrainer validation scoring and the OOF channel
│       ├── trainer_state.py                # Layer 2: ModelTrainer state groups (TrainerSettings, ...) and the artifact format version
│       ├── tuner.py                        # Layer 2: public HPO entry point optimize() (façade over hpo)
│       └── validation.py                   # Layer 1: high-level cross-validation strategies
├── tests                                   # Unit and integration test suites
├── agents.md                               # Instructions for AI agents in the CI pipeline
├── API_REFERENCE.md                        # User guide and API reference
├── config.schema.json                      # JSON Schema of the config (generated)
├── example.py                              # Quick start example
└── schema_generator.py                     # Generates config.schema.json from the Pydantic models
```
