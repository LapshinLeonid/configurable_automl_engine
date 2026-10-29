![PyPI - Version](https://img.shields.io/pypi/v/configurable-automl-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/configurable-automl-engine)
![PyPI - License](https://img.shields.io/pypi/l/configurable-automl-engine)
![PyPI - Types](https://img.shields.io/pypi/types/configurable-automl-engine)
[![codecov](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine/branch/main/graph/badge.svg?token=TAUHARATUM)](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine)
[![CI](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml/badge.svg)](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Socket Badge](https://badge.socket.dev/pypi/package/configurable-automl-engine/0.4.0?artifact_id=tar-gz)](https://socket.dev/pypi/package/configurable-automl-engine/overview/0.4.0/tar-gz)


AutoML Engine is a configuration-driven automated machine learning library for
**tabular regression** in Python. You describe which regressors to try, how to
validate them and how the final model is chosen in a YAML file or a dict; the
engine tunes every algorithm with Optuna, picks a winner, reports an honest
estimate of its quality and saves the trained pipeline.

The library solves **regression only**: the target must be numeric.
Classification is not supported.

# Features

* **Configuration-Driven Architecture**: Fully controlled via YAML schemas and Python configuration classes (Pydantic-based) for reproducible experiments. The config is validated before training starts: unknown keys, unknown hyperparameter names, missing optional packages and contradictory settings are reported as errors.
* **Honest Quality Estimate**: Besides the (optimistic) validation score of the best trial, every run reports an estimate that is not biased by the search itself: `holdout_score` on rows reserved before tuning, or `bbc_score` (Bootstrap Bias Corrected CV) when no hold-out is reserved. See [Training results](API_REFERENCE.md#training-results).
* **Multi-Step Pipelines**: A run is a list of steps (`general.phases`): search steps (`all_algorithms`, `refine`) tune hyperparameters, select steps (`select`) narrow the candidates down to the final winner. See [How the engine works](API_REFERENCE.md#how-the-engine-works).
* **Sanity Gate**: An optional select strategy that protects the final winner selection from "degenerate" models: constant/flat predictions, step functions, unused features and overfit learners.
* **Flexible Validation Strategies**: Supports various splitting techniques including KFold, Leave-One-Out, and Train-Test Split, plus an `auto` strategy that automatically picks between them based on dataset size and dimensionality.
* **Dynamic Hyperparameter Optimization**: Integrated wrapper for Optuna to automate search space configuration and trial management. Every algorithm has a built-in search space, so `{}` is a valid algorithm config.
* **Extensible Model Factory**: 18 distinct scikit-learn/XGBoost regressors (17 without the optional `xgboost` package) under 19 registry names (`glm` is the same `TweedieRegressor` as `tweedieregressor`). Algorithms are referred to only by these names; short aliases (`rf`, `knn`, `xgboost`, …) are not supported.
* **Robust Preprocessing Pipeline**: Automated handling of scaling, missing value imputation, categorical encoding, infinite values and date/time columns, with per-algorithm preprocessing presets and optional feature selection.
* **Parallel Execution Across Algorithms**: Different algorithms are tuned at the same time in threads or processes (`parallel_strategy: "algorithms"`, the only strategy). Trials of one algorithm and the cross-validation folds inside a trial run sequentially (`n_jobs=1`), so a config with a single algorithm does not get faster with more workers. With `parallel_mode: "processes"` the workers start with `forkserver` (or `spawn`) and import the main script again: **guard the script's top-level code with `if __name__ == "__main__":`**, or set `general.start_method: "fork"` (see API_REFERENCE).
* **Seamless Serialization**: Robust I/O tools for saving and loading models, metadata, and preprocessing artifacts in joblib or pickle formats.
* **Dynamic Search Space Clipping**: Hyperparameter boundaries are automatically adjusted based on dataset size (e.g., `n_neighbors` capped at `n_samples - 1`).
* **Adaptive Complexity Guard**: Tightens the search ranges of trees and boosting on small data (lossless clipping + regularization), saving HPO budget and reducing overfitting on small samples. Disableable via `general.complexity_guard`.
* **Broken Algorithm Circuit Breaker**: Automatically disqualifies algorithms after consecutive fatal failures (`MemoryError`, `RuntimeError`, `InvalidDataError`), preventing wasted compute.
* **Early Stopping (Pruning)**: Optional Optuna-based pruning of unpromising trials (MedianPruner / HyperbandPruner) using intermediate per-fold scores, freeing HPO budget for more meaningful trials.
* **Phase/Task Timeouts (process mode only)**: A global timeout for a search step (`phase_timeout`) and a per-algorithm timeout (`task_timeout`), enforced by terminating hung worker processes; a crashed worker loses only its own algorithm, which is listed in `disqualified_algorithms` with the reason. Timeouts require `parallel_mode: "processes"`; with `parallel_mode: "threads"` (the default) setting either of them is a configuration error (`ValueError`), because a running Python thread cannot be stopped.
* **Signature-Aware Hyperparameter Names**: Legacy parameter names are automatically remapped (e.g., `n_iter` → `max_iter` for ARDRegression). Unknown hyperparameter names are **rejected** — at config validation and in a direct call of `create_model(...)` / `ModelTrainer` alike — so typos do not go unnoticed (the only exception is `xgboosting`, whose estimator accepts arbitrary `**kwargs`).
* ⚠️ **Experimental**: **Target-Aware Oversampling**: Optional data augmentation with random oversampling, SMOTE or ADASYN (via imbalanced-learn). Because the target is continuous, it is first split into quantile bins and the sampling strategy is computed on these bins, so rare ranges of the target get more synthetic rows. Oversampling is applied only to the training part of each fold. Optional Gaussian noise injection (`oversampling.add_noise` / `noise_level`) is applied to numeric features after resampling for regularization; binary indicator columns (one-hot, missing-value indicators, 0/1 flags — also after scaling) are left untouched.

# Dependencies

We recommend using the latest version of Python. AutoML Engine supports Python 3.11 and newer.

These distributions are essential for the core functionality and will be installed automatically:

* **NumPy** (>=2.4.2): Base package for numerical computing and array manipulation.
* **Scipy** (>=1.17.0): Used for advanced scientific computing and statistical functions.
* **Pandas** (>=3.0.0): Used for data structures and high-level manipulation of tabular datasets before model ingestion. 
* **PyArrow** (>=23.0.0): Provides a cross-language development platform for in-memory data, enabling efficient data exchange and high-performance integration with Pandas through the Arrow columnar format.
* **Scikit-learn** (>=1.8.0): The primary library for machine learning algorithms, preprocessing tools, and validation frameworks.
* **Imbalanced-learn** (>=0.14.1): Provides the oversampling algorithms (random, SMOTE, ADASYN) used by the experimental oversampling module; the engine applies them to quantile bins of the regression target.
* **Optuna** (>=4.7.0): Powers the engine to perform automated hyperparameter optimization searches.
* **Pydantic** (>=2.12.5): Data validation and settings management using Python type annotations.
* **PyYAML** (>=6.0.3): Implements the standard configuration schema, allowing the system to parse YAML files for model parameters and training setups.
* **Joblib** (>=1.5.3): Provides lightweight pipelining and model serialization (saving/loading).


## Optional Dependencies

These distributions will not be installed automatically. You can install them using the bracket syntax (e.g., `pip install "configurable-automl-engine[xgboost]"`).

| Extra | Installs | Adds |
|-------|----------|------|
| `xgboost` | `xgboost>=3.1.3` | The `xgboosting` algorithm. Without it a config that enables `xgboosting` is rejected at validation. |
| `logging` | `concurrent-log-handler>=0.9.25` | Process-safe rotating log files (`general.log_to_file`) when several processes write to the same log. Without it the standard `RotatingFileHandler` is used. |
| `all` | `xgboost` + `logging` | Both of the above. |

# Installation

To install, run:

    pip install configurable-automl-engine

With the optional XGBoost algorithm:

    pip install "configurable-automl-engine[xgboost]"

With all optional extras (`xgboost` and `logging`):

    pip install "configurable-automl-engine[all]"

Testing:

    python -c "import configurable_automl_engine; print('Success!')"

# Quick Start

The example can be run from [example.py](example.py). It needs only the basic
installation and takes about a minute on a laptop.

```python
from sklearn.datasets import load_diabetes

import configurable_automl_engine as caml

df = load_diabetes(as_frame=True).frame

config = {
    "general": {
        "comparison_metric": "r2",
        "path_to_model": "diabetes_model.pkl",
    },
    # {} = use the built-in search space of the algorithm
    "algorithms": {
        "ridge": {},
        "random_forest": {},
    },
}

results = caml.train_best_model(config=config, df=df, target="target")

print(f"Winner: {results['algorithm']}")
print(f"Hold-out R2 (honest estimate): {results['holdout_score']:.4f}")
```

What happens here:

* Every setting that is not written has a default: 50 tuning trials per
  algorithm, then the algorithm with the best validation score is the winner
  (see [How the engine works](API_REFERENCE.md#how-the-engine-works)).
* The dataset has 442 rows, so by default (`holdout_fraction: "auto"`, data
  with at least 100 rows) 20 % of the rows are reserved **before** tuning.
  `holdout_score` is the winner's R² on these untouched rows.
* The saved model in `diabetes_model.pkl` is trained on **all** rows.

📖 For detailed API documentation and configuration file structure, see [API Reference & Configuration Guide](API_REFERENCE.md).

# Contributing

Small improvements, fixes, reporting issues, requesting features are always appreciated. Use [GitHub issue tracker](https://github.com/LapshinLeonid/configurable_automl_engine/issues).

If you are considering larger contributions to the source code, please contact the author first.

If you contribute, please ensure your code:

* 100% covered by tests.
* Has 0 errors when checked by the ruff linter and passes `ruff format --check`.
* Has 0 errors when checked by the mypy static analyzer.
* All docstrings written in English.

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
module logs under its own name (`logging.getLogger(__name__)`). Breaking
changes are listed in [CHANGELOG.md](CHANGELOG.md).

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
│   └── compare_with_base.py                # Differential behavior check against the base commit (run in CI)
├── src                                     # Project source code root
│   └── configurable_automl_engine          # Main AutoML engine package
│       ├── common                          # Layer 0: shared utilities (imports nothing above common)
│       │   ├── datetime_features.py        # Datetime/timedelta input columns: the datetime policy, expansion at fit and predict time
│       │   ├── definitions.py              # Constants, enums and literals (validation, encoding, serialization)
│       │   ├── dependency_utils.py         # Optional library and dependency checks
│       │   ├── errors.py                   # TrainingError
│       │   ├── log_setup.py                # Centralized logging setup (setup_logging)
│       │   ├── metrics.py                  # Evaluation metrics, BBC-CV estimate
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
│       ├── trainer_quick.py                # Layer 2: the logic of the legacy train_model() facade
│       ├── trainer_scoring.py              # Layer 2: ModelTrainer validation scoring and the OOF channel
│       ├── trainer_state.py                # Layer 2: ModelTrainer state groups (TrainerSettings, ...) and the artifact format version
│       ├── tuner.py                        # Layer 2: public HPO entry point optimize() (façade over hpo)
│       └── validation.py                   # Layer 1: high-level cross-validation strategies
├── tests                                   # Unit and integration test suites
├── agents.md                               # Instructions for AI agents in the CI pipeline
├── API_REFERENCE.md                        # User guide and API reference
├── CHANGELOG.md                            # Release notes (breaking changes)
├── config.schema.json                      # JSON Schema of the config (generated)
├── example.py                              # Quick start example
└── schema_generator.py                     # Generates config.schema.json from the Pydantic models
```

# ⚠️ NeuroSlop Warning

This project utilizes Large Language Models (LLMs) to assist in development and maintenance. To ensure transparency regarding the origin of the codebase, please note the following:

* Original Core & Architecture: The fundamental architecture, core logic, and overall project conceptualization are 100% original and authored by the human creator.
* Automated Testing: The test suite is almost entirely LLM-generated. While these tests aim for high coverage and functional verification, they were synthesized based on the provided source code.
* Source Code Generation: Portions of non-critical source code were also LLM-generated. However, all generated code has undergone a manual code review by the author to the full extent of their technical expertise and competence to ensure quality and logic. Additionally, the code is fully compliant with modern development standards, showing no issues or warnings from ruff and mypy, and maintains 100% unit test coverage to ensure reliability and correctness.
* Commit History: All commit messages and titles have been generated by an LLM. This ensures a consistent (though automated) narrative of the project's evolution.
* Documentation: The project documentation, including parts of this README and inline comments, is partially LLM-generated. AI was used to expand on technical details and improve readability based on the original technical specifications.

While this project leverages neural synthesis for scaffolding, verification, and non-critical components, users and contributors should be aware that the intellectual core, architectural vision, and primary logic remain entirely human-made. The AI serves as an assistant, while the fundamental value and creative direction are the result of deliberate human engineering.

# Contact

If you'd like to contact the author, you can use Telegram @Lapshin_LA or email leonid.lapshin.a@gmail.com
