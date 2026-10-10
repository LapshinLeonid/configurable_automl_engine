![PyPI - Version](https://img.shields.io/pypi/v/configurable-automl-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/configurable-automl-engine)
![PyPI - License](https://img.shields.io/pypi/l/configurable-automl-engine)
![PyPI - Types](https://img.shields.io/pypi/types/configurable-automl-engine)
[![codecov](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine/branch/main/graph/badge.svg?token=TAUHARATUM)](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine)
[![CI](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml/badge.svg)](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Socket Badge](https://badge.socket.dev/pypi/package/configurable-automl-engine/0.2.0?artifact_id=tar-gz)](https://socket.dev/pypi/package/configurable-automl-engine/overview/0.2.0/tar-gz)


AutoML Engine is a configuration-driven automated machine learning library for Python. 
It provides a high-performance ecosystem for model selection and hyperparameter optimization, 
designed to scale from local experimentation to large-scale data processing. 

# Features

* **Configuration-Driven Architecture**: Fully controlled via YAML schemas and Python configuration classes (Pydantic-based) for reproducible experiments.
* **Flexible Validation Strategies**: Supports various splitting techniques including KFold, Leave-One-Out, and Train-Test Split, plus an `auto` strategy that automatically picks between them based on dataset size and dimensionality. 
* **Dynamic Hyperparameter Optimization**: Integrated wrapper for Optuna to automate search space configuration and trial management.
* **Extensible Model Factory**: Built-in support for 19+ regression algorithms with automatic hyperparameter cleaning.
* **Robust Preprocessing Pipeline**: Automated handling of scaling, missing value imputation, and categorical encoding.
* **Advanced Imbalance Handling**: Built-in oversampling module supporting SMOTE, ADASYN, and random oversampling, with optional Gaussian noise injection (`add_noise` flag / `noise_level`) applied to numeric features after resampling for regularization.
* **Parallel Execution**: Utilizes threading and multi-processing for faster hyperparameter searches and cross-validation loops.
* **Seamless Serialization**: Robust I/O tools for saving and loading models, metadata, and preprocessing artifacts in joblib or pickle formats.
* **Dynamic Search Space Clipping**: Hyperparameter boundaries are automatically adjusted based on dataset size (e.g., `n_neighbors` capped at `n_samples - 1`).
* **Broken Algorithm Circuit Breaker**: Automatically disqualifies algorithms after consecutive fatal failures (`MemoryError`, `RuntimeError`, `InvalidDataError`), preventing wasted compute.
* **Early Stopping (Pruning)**: Optional Optuna-based pruning of unpromising trials (MedianPruner / HyperbandPruner) using intermediate per-fold scores, freeing HPO budget for more meaningful trials.
* **Granular Phase/Task Timeouts**: Configurable global phase timeout and per-task timeout with a watchdog mechanism to prevent deadlocks. In `"threads"` mode timeouts are **soft**: an already running Python function cannot be killed.
* **Signature-Aware Hyperparameter Cleaning**: Legacy parameter names are automatically remapped (e.g., `n_iter` → `max_iter` for ARDRegression), and unknown parameters are safely dropped.
* **Sanity Gate**: Protects the final winner selection from
"degenerate" models: constant/flat predictions, step functions, unused
features and overfit learners.

# Dependencies

We recommend using the latest version of Python. AutoML Engine supports Python 3.11 and newer.

These distributions are essential for the core functionality and will be installed automatically:

* **NumPy** (>=2.4.2): Base package for numerical computing and array manipulation.
* **Scipy** (>=1.17.0): Used for advanced scientific computing and statistical functions.
* **Pandas** (>=3.0.0): Used for data structures and high-level manipulation of tabular datasets before model ingestion. 
* **PyArrow** (>=23.0.0): Provides a cross-language development platform for in-memory data, enabling efficient data exchange and high-performance integration with Pandas through the Arrow columnar format.
* **Scikit-learn** (>=1.8.0): The primary library for machine learning algorithms, preprocessing tools, and validation frameworks.
* **Imbalanced-learn** (>=0.14.1): Provides oversampling algorithms (like SMOTE) for handling datasets with skewed class distributions.
* **Optuna** (>=4.7.0): Powers the engine to perform automated hyperparameter optimization searches.
* **Pydantic** (>=2.12.5): Data validation and settings management using Python type annotations.
* **PyYAML** (>=6.0.3): Implements the standard configuration schema, allowing the system to parse YAML files for model parameters and training setups.
* **Joblib** (>=1.5.3): Provides lightweight pipelining and model serialization (saving/loading).


## Optional Dependencies

These distributions will not be installed automatically. You can install them using the bracket syntax (e.g., pip install "automl-engine[xgboost]").

* **XGBoost**: Adds support for high-performance gradient boosting models.

# Installation

To install, run:

    pip install configurable-automl-engine


For the full suite including all supported gradient boosting backends:

     pip install configurable-automl-engine[all]   

Testing:

    python -c "import configurable_automl_engine; print('Success!')"

# Quick Start

> **Note (epic #74):** the `select` steps are fully implemented (task R2).
> The `refine` action is declared in the config schema but its execution is
> not implemented yet (task R3) — configs containing `refine` phases fail
> with `NotImplementedError`. The example below includes a `refine` phase;
> remove it (or replace with another `all_algorithms` phase) to run today.
> See the pipeline section below.

The example can be run from [example.py](example.py). It demonstrates the
full target pipeline with a Sanity Gate select step; the shorter config
below is a minimal illustration of the same schema (`all_algorithms` →
terminal select) — the two examples are intentionally different.

    from sklearn.datasets import load_diabetes

    import configurable_automl_engine as caml

    data = load_diabetes(as_frame=True)
    df = data.frame

    config = {
        "general": {
            "comparison_metric": "r2",
            "phases": [
                {"n_trials": 100, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner"
                }
            ],
            "path_to_model": "diabetes_model.pkl"
        },
        "algorithms": {
            "random_forest": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"n_estimators": [10, 500], "max_depth": [3, 20]}
            },
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]}
            },
            "xgboosting": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {
                    "n_estimators": [100, 1000],
                    "max_depth": [3, 10],
                    "learning_rate": [0.01, 0.3],
                    "subsample": [0.5, 1.0]
                }
            },
        }
    }

    results = caml.train_best_model(config=config, df=df, target='target')

    print(f"Winner: {results['algorithm']}, Score: {results['score']:.4f}")

# Pipeline phases: `refine` / `select` (epic #74)

> **Note (epic #74):** the `select` action is fully implemented (task R2):
> both strategies (`cv_best` / `sanity_gate`) and all output forms
> (`winner` / `top_k` / `pool`). The `refine` action is declared in the
> config schema, but its execution is not implemented yet (task R3) —
> configs containing `refine` phases are rejected at runtime with
> `NotImplementedError`.

The HPO pipeline is assembled from explicit phases (`general.phases`):

* `action: "all_algorithms"` — `n_trials` HPO attempts for every enabled algorithm;
* `action: "refine"` — `n_trials` warm-started HPO attempts for **every** candidate of the incoming working set. As the first phase `refine` is accepted by the schema: the incoming working set is then the full set of enabled algorithms (no warm-start from a previous phase); the execution semantics are finalized by task R3.
* `action: "select"` — an explicit selection step with `strategy: "cv_best" | "sanity_gate"` and `output: "winner" | "top_k" | "pool"` (parameter `k >= 1` is required for, and only allowed with, `output: "top_k"`). The step runs synchronously (never inside `run_parallel`) and its output **replaces** the pipeline working set — subsequent `refine` / `select` steps operate on it:
  * `strategy: "cv_best"` — ranking by raw CV score (no audit, no gate section in the report). `output: "winner"` — the CV leader; `output: "top_k"` — the first `k` candidates of the ranking; `output: "pool"` — the corridor finalists pool (corridor δ ∩ `top_k_candidates`, thresholds from the `sanity_gate` section) ranked by CV score **without** the `ModelSanityGate` audit.
  * `strategy: "sanity_gate"` — the two-stage selection (T3+T4): finalists pool (corridor δ ∩ `top_k_candidates` from `general.sanity_gate`) → every finalist trained on **100% of the data exactly once** → `ModelSanityGate` audit → ranking of the candidates that passed the audit by `RMSE_oof` (formal fallback criterion when the whole pool is disqualified). `output: "winner"` — the robust winner (terminal step); `output: "top_k"` — the first `k` candidates of the `RMSE_oof` ranking; `output: "pool"` — every candidate that passed the audit (or the full fallback ranking when all are disqualified). Gate disqualifications are fed into the circuit breaker (`disqualified_algorithms`) on a par with HPO failures. A terminal select with `output: "winner"` reuses the already trained full-fit trainer of the winner — the winner is **not** retrained for the artifact. A repeated sanity select (e.g. after `refine`) trains its finalists from scratch — stale full-fit trainers are never reused.
  * Every `sanity_gate` select adds its own section to `result["sanity_gate"]` keyed by the phase name (pool, per-candidate audit, applied disqualifications, `fallback_used`, `winner_rmse_oof`, `overhead_seconds`); `cv_best` selects add none. The legacy report keys `disqualified_by_sanity_gate`, `sanity_gate_warn_only` and `sanity_gate_overhead_seconds` were removed together with the `warn_only` mode (task R2).

Validation rules: `select` cannot be the first phase; no phases are allowed after a terminal select (`output: "winner"`); the **last phase must be a `select` with `output: "winner"`**. The parameter `k` is deliberately not cross-checked against `sanity_gate.top_k_candidates` at the config level: the pool is capped by `top_k_candidates` first, and the exact `k`-vs-pool interplay is defined at execution time. The `sanity_gate` section remains a pure source of thresholds (corridor, `top_k_candidates`, audit limits) for select steps with `strategy: "sanity_gate"` — the global switch `general.sanity_gate.mode` was removed together with the deprecated action `refine_winner`.

Migration table for typical legacy configs:

| Legacy config | New config (after epic #74) |
|---------------|------------|
| `all_algorithms` → `refine_winner` | `all_algorithms` → `select(strategy: cv_best, output: top_k, k: N)` → `refine` → `select(strategy: cv_best, output: winner)` — or simply an extended `all_algorithms` + `select(strategy: cv_best, output: winner)` |
| Implicit gate at the end (`sanity_gate.mode: active`) | Explicit last phase `select(strategy: sanity_gate, output: winner)` |
| `sanity_gate.mode: warn_only` | `select(strategy: sanity_gate, ...)` step + read the gate section of the select-step report (`result["sanity_gate"][<phase name>]`) |

Configs with `refine_winner` or `sanity_gate.mode` are rejected at load with a migration hint (no silent translation).

# Contributing

Small improvements, fixes, reporting issues, requesting features are always appreciated. Use [GitHub issue tracker](https://github.com/LapshinLeonid/configurable_automl_engine/issues).

If you are considering larger contributions to the source code, please contact the author first.

If you contribute, please ensure your code:

* 100% covered by tests.
* Has 0 errors when checked by the ruff linter.
* Has 0 errors when checked by the mypy static analyzer.
* All docstrings written in English or Russian.

# Project Structure

```
├── src                                     # Project source code root
│   └── configurable_automl_engine          # Main AutoML engine package
│       ├── common                          # Shared utilities and helper functions
│       │   ├── definitions.py              # Constants, enums, and schema definitions
│       │   ├── dependency_utils.py         # Optional library and dependency checks
│       │   ├── hyperopt_defaults.py        # Default search spaces, clipping, and constraints
│       │   ├── serialization_utils.py      # Model/pipeline serialization logic
│       │   └── validation_utils.py         # Low-level data validation and effective train size
│       ├── training_engine                 # Core orchestration and execution logic
│       │   ├── component.py                # Pipeline building block base classes
│       │   ├── config_parser.py            # Configuration parsing and validation
│       │   ├── logger.py                   # Centralized logging management
│       │   ├── metrics.py                  # Evaluation metrics implementation
│       │   └── thread_pool.py              # Multi-threading, parallel execution, watchdog
│       ├── models.py                       # Model factory, aliases, hyperparameter cleaning
│       ├── oversampling.py                 # Imbalance handling and resampling
│       ├── trainer.py                      # Training process orchestrator
│       ├── tuner.py                        # Hyperparameter optimization, circuit breaker
│       └── validation.py                   # High-level cross-validation strategies
└── tests                                   # Unit and integration test suites
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
