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
* **Honest Quality Estimate**: Besides the (optimistic) validation score of the best trial, every run reports an estimate that is not biased by the search itself: `holdout_score` on rows reserved before tuning, or `bbc_score` (Bootstrap Bias Corrected CV) when no hold-out is reserved. See [Which score to trust](#which-score-to-trust).
* **Multi-Step Pipelines**: A run is a list of steps (`general.phases`): search steps (`all_algorithms`, `refine`) tune hyperparameters, select steps (`select`) narrow the candidates down to the final winner. See [How a run works](#how-a-run-works).
* **Sanity Gate**: An optional select strategy that protects the final winner selection from "degenerate" models: constant/flat predictions, step functions, unused features and overfit learners.
* **Flexible Validation Strategies**: Supports various splitting techniques including KFold, Leave-One-Out, and Train-Test Split, plus an `auto` strategy that automatically picks between them based on dataset size and dimensionality.
* **Dynamic Hyperparameter Optimization**: Integrated wrapper for Optuna to automate search space configuration and trial management. Every algorithm has a built-in search space, so `{}` is a valid algorithm config.
* **Extensible Model Factory**: 18 distinct scikit-learn/XGBoost regressors (17 without the optional `xgboost` package) under 19 registry names (`glm` is the same `TweedieRegressor` as `tweedieregressor`). In the config the full names are required; short aliases (`rf`, `knn`, `xgboost`, …) work only in the Python API.
* **Robust Preprocessing Pipeline**: Automated handling of scaling, missing value imputation, categorical encoding, infinite values and date/time columns, with per-algorithm preprocessing presets and optional feature selection.
* **Parallel Execution Across Algorithms**: Different algorithms are tuned at the same time in threads or processes (`parallel_strategy: "algorithms"`, the only strategy). Trials of one algorithm and the cross-validation folds inside a trial run sequentially (`n_jobs=1`), so a config with a single algorithm does not get faster with more workers.
* **Seamless Serialization**: Robust I/O tools for saving and loading models, metadata, and preprocessing artifacts in joblib or pickle formats.
* **Dynamic Search Space Clipping**: Hyperparameter boundaries are automatically adjusted based on dataset size (e.g., `n_neighbors` capped at `n_samples - 1`).
* **Adaptive Complexity Guard**: Tightens the search ranges of trees and boosting on small data (lossless clipping + regularization), saving HPO budget and reducing overfitting on small samples. Disableable via `general.complexity_guard`.
* **Broken Algorithm Circuit Breaker**: Automatically disqualifies algorithms after consecutive fatal failures (`MemoryError`, `RuntimeError`, `InvalidDataError`), preventing wasted compute.
* **Early Stopping (Pruning)**: Optional Optuna-based pruning of unpromising trials (MedianPruner / HyperbandPruner) using intermediate per-fold scores, freeing HPO budget for more meaningful trials.
* **Phase/Task Timeouts (process mode only)**: A global timeout for a search step (`phase_timeout`) and a per-algorithm timeout (`task_timeout`), enforced by a watchdog that terminates hung workers. Timeouts require `parallel_mode: "processes"`; with `parallel_mode: "threads"` (the default) setting either of them is a configuration error (`ValueError`), because a running Python thread cannot be stopped.
* **Signature-Aware Hyperparameter Names**: Legacy parameter names are automatically remapped (e.g., `n_iter` → `max_iter` for ARDRegression). Unknown hyperparameter names in the config are **rejected** at config validation, so typos do not go unnoticed (the only exception is `xgboosting`, whose estimator accepts arbitrary `**kwargs`). Only a direct call of `configurable_automl_engine.models.create_model(...)` silently drops parameters the estimator does not accept.
* ⚠️ **Experimental**: **Target-Aware Oversampling**: Optional data augmentation with random oversampling, SMOTE or ADASYN (via imbalanced-learn). Because the target is continuous, it is first split into quantile bins and the sampling strategy is computed on these bins, so rare ranges of the target get more synthetic rows. Oversampling is applied only to the training part of each fold. Gaussian noise injection (`add_noise` / `noise_level`) is available only when `DataOversampler` is used directly, not from the config.

**Scale.** The engine targets small and medium tabular datasets that fit in
memory. Cross-validation inside a trial is not parallelized, and the
`bbc_score` estimate is skipped (with a WARNING) when the planned number of
stored out-of-fold predictions exceeds 4,000,000 values. There are no
performance benchmarks; the [benchmarks](benchmarks) directory measures the
quality effect of individual components (Sanity Gate, Complexity Guard,
oversampling).

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
| `xgboost` | `xgboost>=3.1.3` | The `xgboosting` algorithm (alias `xgboost`). Without it a config that enables `xgboosting` is rejected at validation. |
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
  (see [How a run works](#how-a-run-works)).
* The dataset has 442 rows, so by default (`holdout_fraction: "auto"`, data
  with at least 100 rows) 20 % of the rows are reserved **before** tuning.
  `holdout_score` is the winner's R² on these untouched rows.
* The saved model in `diabetes_model.pkl` is trained on **all** rows.

The same config as YAML (pass the file path as `config=`):

```yaml
general:
  comparison_metric: "r2"
  path_to_model: "diabetes_model.pkl"
algorithms:
  ridge: {}
  random_forest: {}
```

To add XGBoost, install the `xgboost` extra and add `xgboosting: {}` to
`algorithms`. To tune only selected parameters within your own ranges, use
`limit_hyperparameters: true` with a `hyperparameters` block — but keep in mind
that a narrow range can cut off the optimum (the tuner logs a WARNING when the
best trial lies on the boundary of a range). The built-in search spaces are a
better starting point.

📖 For detailed API documentation and configuration file structure, see [API Reference & Configuration Guide](API_REFERENCE.md).

# How a run works

A training run is a short pipeline of steps configured in `general.phases`.
When `phases` is omitted, the engine uses:

```yaml
general:
  phases:
    - {name: "search", action: "all_algorithms", n_trials: 50}
    - {name: "final_select", action: "select", strategy: "cv_best", output: "winner"}
```

**Search steps** tune hyperparameters with Optuna:

* `all_algorithms` — `n_trials` trials for every enabled algorithm, with
  different algorithms tuned in parallel.
* `refine` — `n_trials` more trials for each algorithm still in the race,
  starting from its best parameters so far; its score can only stay the same
  or improve.

**Select steps** (`action: "select"`) decide which algorithms stay in the race:

* `strategy: "cv_best"` — rank the candidates by validation score.
* `strategy: "sanity_gate"` — take the candidates close to the leader, train
  each on all rows and reject degenerate models (near-constant predictions,
  step functions, models that ignore most features, strong overfitting); the
  best candidate that passes the checks wins. Recommended for small or noisy
  data.
* `output` — what the step keeps: `winner` (one model; this ends the
  pipeline), `top_k` (the best `k`) or `pool` (every candidate close to the
  leader).

The pipeline must start with a search step and end with a select step with
`output: "winner"`. A typical "coarse search, then fine-tune the top 3"
pipeline:

```yaml
general:
  phases:
    - {name: "coarse", action: "all_algorithms", n_trials: 100}
    - {name: "top3",   action: "select", strategy: "sanity_gate", output: "top_k", k: 3}
    - {name: "fine",   action: "refine", n_trials: 200}
    - {name: "final",  action: "select", strategy: "sanity_gate", output: "winner"}
```

Details: [Pipeline steps](API_REFERENCE.md#pipeline-steps-phases) and
[Sanity Gate](API_REFERENCE.md#sanity-gate-robust-winner-selection).

# Which score to trust

`train_best_model` returns several numbers; only some of them are honest
estimates of how the model will do on new data:

| Key | Present | Trust it as a quality estimate? |
|-----|---------|---------------------------------|
| `score` | always | **No** — the validation score of the best of many trials (`n_trials_evaluated`), measured on the same folds that were used to choose it. Optimistic, especially on small data. |
| `oof_score` | when out-of-fold predictions exist | **No** — computed on the same folds as `score` and carries the same bias. A diagnostic. |
| `holdout_score` | a hold-out is reserved (by default: data with ≥ 100 rows) | **Yes** — the winner measured on rows reserved before tuning and never used for tuning or selection. |
| `bbc_score` | no hold-out is reserved (by default: data with < 100 rows, or `holdout_fraction: null`) | **Yes** — Bootstrap Bias Corrected CV over all evaluated trials; slightly conservative. Skipped over the memory budget (4,000,000 stored values). |

Use `holdout_score` when it is present, otherwise `bbc_score`. The hold-out
share is set by `general.holdout_fraction` (`"auto"`, a fraction in `(0, 1)`
or `null`). See [Training results](API_REFERENCE.md#training-results) for every
key of the result.

# Contributing

Small improvements, fixes, reporting issues, requesting features are always appreciated. Use [GitHub issue tracker](https://github.com/LapshinLeonid/configurable_automl_engine/issues).

If you are considering larger contributions to the source code, please contact the author first.

If you contribute, please ensure your code:

* 100% covered by tests.
* Has 0 errors when checked by the ruff linter and passes `ruff format --check`.
* Has 0 errors when checked by the mypy static analyzer.
* All docstrings written in English.

# Project Structure

```
├── benchmarks                              # Quality benchmarks of Sanity Gate, Complexity Guard, oversampling
├── src                                     # Project source code root
│   └── configurable_automl_engine          # Main AutoML engine package
│       ├── common                          # Shared utilities and helper functions
│       │   ├── complexity_guard.py         # Adaptive tightening of tree/boosting search spaces on small data
│       │   ├── definitions.py              # Constants, enums, and schema definitions
│       │   ├── dependency_utils.py         # Optional library and dependency checks
│       │   ├── hyperopt_defaults.py        # Default search spaces, clipping, and constraints
│       │   ├── serialization_utils.py      # Model/pipeline serialization logic
│       │   ├── target_domain.py            # Target-domain guard for GLMs (Poisson, Gamma, Tweedie)
│       │   └── validation_utils.py         # Low-level data validation and effective train size
│       ├── training_engine                 # Core orchestration and execution logic
│       │   ├── component.py                # Main orchestrator: train_best_model, phases, select steps, artifact saving
│       │   ├── config_parser.py            # Configuration parsing and validation
│       │   ├── logger.py                   # Centralized logging management
│       │   ├── metrics.py                  # Evaluation metrics, BBC-CV estimate
│       │   ├── sanity_gate.py              # Sanity Gate checks for degenerate models
│       │   └── thread_pool.py              # Multi-threading, parallel execution, watchdog
│       ├── feature_selection.py            # Optional feature selection step
│       ├── models.py                       # Model factory, aliases, hyperparameter cleaning
│       ├── oversampling.py                 # Experimental target-aware oversampling
│       ├── preprocessing.py                # Feature preprocessing: imputation, scaling, encoding
│       ├── preprocessing_presets.py        # Per-algorithm preprocessing presets
│       ├── trainer.py                      # ModelTrainer: training, validation and saving of a single model
│       ├── tuner.py                        # Hyperparameter optimization, circuit breaker
│       └── validation.py                   # High-level cross-validation strategies
├── tests                                   # Unit and integration test suites
├── agents.md                               # Instructions for AI agents in the CI pipeline
├── API_REFERENCE.md                        # User guide and API reference
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
