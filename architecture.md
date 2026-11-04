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
| 1 — ML building blocks | `models`, `preprocessing_presets`, `preprocessing`, `oversampling`, `validation`, `feature_selection`, `isotonic` |
| 0 — shared utilities | `common/` (and the generated `_version`) |

The package `__init__` is the public façade on top; it loads
`training_engine.component` (and with it Optuna) only on the first call of
`train_best_model`. The rules are `import-linter` contracts in
`pyproject.toml`, checked in CI by `lint-imports`; package imports inside
functions are not allowed (except the lazy `train_best_model` proxy in
`training_engine/__init__.py`). If a lower module needs something from above,
move that thing down.

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
│       ├── training_engine                 # Layer 3: orchestration
│       │   ├── artifact.py                 # Trainer assembly, final fit, artifact saving and the result report
│       │   ├── component.py                # Main orchestrator: train_best_model, the phase loop; applies the step outcomes
│       │   ├── config                      # Config schema (Pydantic), one module per section
│       │   │   ├── algorithms.py           # algorithms / oversampling sections (AlgoCfg, OversamplingCfg)
│       │   │   ├── complexity_guard.py     # general.complexity_guard (ComplexityGuardCfg)
│       │   │   ├── general.py              # general section (GeneralCfg, fields only)
│       │   │   ├── general_rules.py        # Cross-field validation rules of the general section
│       │   │   ├── phases.py               # Pipeline steps (HPOPhaseCfg) and pruning (PruningCfg)
│       │   │   ├── root.py                 # Root Config and read_config
│       │   │   └── sanity_gate.py          # general.sanity_gate (SanityGateCfg)
│       │   ├── config_parser.py            # Façade: re-exports the config schema (the public import path)
│       │   ├── data_prep.py                # Config loading, X/y preparation, hold-out, validation strategy
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
│       ├── preprocessing                   # Layer 1: feature preprocessing: imputation, scaling, encoding
│       │   ├── builder.py                  # build_preprocessor: the single preprocessor entry point
│       │   ├── columns.py                  # ColumnNameSelector: column selection by name
│       │   ├── encoders.py                 # Target / frequency / hashing / split categorical encoders
│       │   └── features.py                 # Feature type detection, uninformative columns, feature count estimate
│       ├── preprocessing_presets.py        # Layer 1: per-algorithm preprocessing presets
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
