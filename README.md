![PyPI - Version](https://img.shields.io/pypi/v/configurable-automl-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/configurable-automl-engine)
![PyPI - License](https://img.shields.io/pypi/l/configurable-automl-engine)
![PyPI - Types](https://img.shields.io/pypi/types/configurable-automl-engine)
[![codecov](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine/branch/main/graph/badge.svg?token=TAUHARATUM)](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine)
[![CI](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml/badge.svg)](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

AutoML Engine is a configuration-driven automated machine learning library for Python. 
It provides a high-performance ecosystem for model selection and hyperparameter optimization, 
designed to scale from local experimentation to large-scale data processing. 

# Features

* **Configuration-Driven Architecture**: Fully controlled via YAML schemas and Python configuration classes (Pydantic-based) for reproducible experiments.
* **Flexible Validation Strategies**: Supports various splitting techniques including KFold, Leave-One-Out, and Train-Test Split, plus an `auto` strategy that automatically picks between them based on dataset size and dimensionality.
* **Dynamic Hyperparameter Optimization**: Integrated wrapper for Optuna to automate search space configuration and trial management.
* **Extensible Model Factory**: Built-in support for 19+ regression algorithms with automatic hyperparameter cleaning.
* **Robust Preprocessing Pipeline**: Automated handling of scaling, encoding, and missing value imputation.
* **Advanced Imbalance Handling**: Built-in oversampling module supporting SMOTE, ADASYN, BorderlineSMOTE, and Random oversampling with noise injection.
* **Nested Validation Support**: Ability to perform complex nested cross-validation to ensure model generalizability.
* **Parallel Execution**: Utilizes threading and multi-processing for faster hyperparameter searches and cross-validation loops.
* **Seamless Serialization**: Robust I/O tools for saving and loading models, metadata, and preprocessing artifacts in joblib or pickle formats.
* **Dynamic Search Space Clipping**: Hyperparameter boundaries are automatically adjusted based on dataset size (e.g., `n_neighbors` capped at `n_samples - 1`).
* **Broken Algorithm Circuit Breaker**: Automatically disqualifies algorithms after consecutive fatal failures (`MemoryError`, `RuntimeError`, `InvalidDataError`), preventing wasted compute.
* **Granular Phase/Task Timeouts**: Configurable global phase timeout and per-task timeout with a watchdog mechanism to prevent deadlocks.
* **Signature-Aware Hyperparameter Cleaning**: Legacy parameter names are automatically remapped (e.g., `n_iter` → `max_iter` for ARDRegression), and unknown parameters are safely dropped.

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

The example can be run from [example.py](example.py).

    from sklearn.datasets import load_diabetes

    import configurable_automl_engine as caml

    data = load_diabetes(as_frame=True)
    df = data.frame

    config = {
        "general": {
            "comparison_metric": "r2",
            "phases": [
                {"n_trials": 100, "action": "all_algorithms"},
                {"n_trials": 200, "action": "refine_winner"}
            ],
            "path_to_model": "diabetes_model.joblib"
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

📖 For detailed API documentation and configuration file structure, see [API Reference & Configuration Guide](API_REFERENCE.md).

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
