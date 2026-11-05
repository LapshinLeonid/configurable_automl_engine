![PyPI - Version](https://img.shields.io/pypi/v/configurable-automl-engine)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/configurable-automl-engine)
![PyPI - License](https://img.shields.io/pypi/l/configurable-automl-engine)
![PyPI - Types](https://img.shields.io/pypi/types/configurable-automl-engine)
[![codecov](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine/branch/main/graph/badge.svg?token=TAUHARATUM)](https://codecov.io/gh/LapshinLeonid/configurable_automl_engine)
[![CI](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml/badge.svg)](https://github.com/LapshinLeonid/configurable_automl_engine/actions/workflows/CI.yaml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Socket Badge](https://badge.socket.dev/pypi/package/configurable-automl-engine/0.5.0?artifact_id=tar-gz)](https://socket.dev/pypi/package/configurable-automl-engine/overview/0.5.0/tar-gz)


AutoML Engine is a configuration-driven automated machine learning library for
**tabular regression** in Python. You describe which regressors to try, how to
validate them and how the final model is chosen in a YAML file or a dict; the
engine tunes every algorithm with Optuna, picks a winner, reports an honest
estimate of its quality and saves the trained pipeline.

The library solves **regression only**: the target must be numeric.
Classification is not supported.

# Features

* **Configuration-Driven Architecture** — a YAML/dict config validated before
  training starts: unknown keys, unknown hyperparameter names, missing optional
  packages and contradictory settings are errors. See [Configuration reference](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#configuration-reference).
* **Honest Quality Estimate** — the best trial's validation score is optimistic,
  because it was used to pick the winner. Every run also reports
  `holdout_score` on rows reserved before tuning, or `bbc_score` (Bootstrap
  Bias Corrected CV) when no hold-out is reserved.
  See [Training results](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#training-results).
* **Multi-Step Pipelines** — a run is a list of steps (`general.phases`): search
  steps tune hyperparameters, select steps narrow the candidates down to the
  winner. See [How the engine works](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#how-the-engine-works).
* **Sanity Gate** — an optional select strategy that rejects degenerate models
  (constant or step-like predictions, unused features, overfitting) before the
  winner is chosen. See [Sanity Gate](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#sanity-gate-robust-winner-selection).
* **Flexible Validation Strategies** — KFold, Leave-One-Out, Train-Test Split,
  plus an `auto` strategy that picks between them based on dataset size and
  dimensionality. See [`ModelTrainer`](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#modeltrainer).
* **Custom Metrics and Scorers** — plug in your own metric function or
  sklearn scorer by import path (`general.custom_metrics`, e.g.
  `path: "my_project.metrics:mape"`) and use it as `comparison_metric` or in
  `additional_metrics`; it is checked at config load and works in the
  processes mode. See "Custom metrics" in `API_REFERENCE.md`.
* **Dynamic Hyperparameter Optimization** — an Optuna wrapper with a built-in
  search space for every algorithm (`{}` is a valid algorithm config).
  See [`optimize`](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#optimizealgo_name-x-y).
* **Extensible Model Factory** — 18 scikit-learn/XGBoost regressors (17 without
  the optional `xgboost` package); `glm` and `tweedieregressor` are two names
  for the same `TweedieRegressor`. See [Algorithms section](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#algorithms-section).
* **Robust Preprocessing Pipeline** — scaling, missing-value imputation,
  outlier handling (quantile/IQR winsorization), categorical encoding,
  infinite values, date/time columns, per-algorithm
  presets and optional feature selection. See
  [Categorical encoding](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#categorical-encoding) and
  [preprocessing presets](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#adaptive-feature-preprocessing-preprocessing-presets).
* **Parallel Execution Across Algorithms** — algorithms are tuned at the same
  time in threads or processes (`parallel_strategy: "algorithms"`); a config
  with a single algorithm does not get faster with more workers. With
  `parallel_mode: "processes"` the workers start with `forkserver` (or
  `spawn`) and import the main script again: **guard the script's top-level
  code with `if __name__ == "__main__":`**, or set
  `general.start_method: "fork"`. See
  [`run_parallel`](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#run_parallelfunc-args_seq).
* **Seamless Serialization** — joblib or pickle formats for models, metadata
  and preprocessing artifacts. See [`ModelTrainer`](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#modeltrainer).
* **Dynamic Search Space Clipping** — hyperparameter boundaries are adjusted
  based on dataset size (e.g. `n_neighbors` capped at `n_samples - 1`). See
  [Default search spaces](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#algorithms-section).
* **Adaptive Complexity Guard** — tightens the search ranges of trees and
  boosting on small data; disable with `general.complexity_guard`. See
  [Complexity Guard configuration](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#complexity-guard-configuration-complexity_guard).
* **Broken Algorithm Circuit Breaker** — algorithms are disqualified after
  consecutive fatal failures (`MemoryError`, `RuntimeError`, `InvalidDataError`).
* **Early Stopping (Pruning)** — optional Optuna-based pruning of unpromising
  trials. See [Early stopping](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#early-stopping-pruning).
* **Phase/Task Timeouts (process mode only)** — a global timeout for a search
  step (`phase_timeout`) and a per-algorithm timeout (`task_timeout`); with
  `parallel_mode: "threads"` setting either of them is a configuration error.
  See [`run_parallel`](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#run_parallelfunc-args_seq).
* **Signature-Aware Hyperparameter Names** — legacy parameter names are
  remapped (e.g. `n_iter` → `max_iter` for ARDRegression); unknown names are
  rejected so typos do not go unnoticed (except `xgboosting`, which accepts
  arbitrary `**kwargs`).
* ⚠️ **Experimental**: **Target-Aware Oversampling** — random oversampling,
  SMOTE or ADASYN (requires `imbalanced-learn`), applied only to the training
  part of each fold. See [Oversampling section](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#oversampling-section)

# Dependencies

We recommend using the latest version of Python. AutoML Engine supports Python 3.11 and newer.

The required dependencies (installed automatically):

* NumPy (>=2.4.2)
* SciPy (>=1.17.0)
* Pandas (>=3.0.0)
* PyArrow (>=23.0.0)
* Scikit-learn (>=1.8.0)
* Imbalanced-learn (>=0.14.1)
* Optuna (>=4.7.0)
* Pydantic (>=2.12.5)
* PyYAML (>=6.0.3)
* Joblib (>=1.5.3)

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

The example can be run from [example.py](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/example.py). It needs only the basic
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
  (see [How the engine works](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md#how-the-engine-works)).
* The dataset has 442 rows, so by default (`holdout_fraction: "auto"`, data
  with at least 100 rows) 20 % of the rows are reserved **before** tuning.
  `holdout_score` is the winner's R² on these untouched rows.
* The saved model in `diabetes_model.pkl` is trained on **all** rows.

📖 For detailed API documentation and configuration file structure, see [API Reference & Configuration Guide](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/API_REFERENCE.md).

# Contributing

Small improvements, fixes, reporting issues, requesting features are always appreciated. Use [GitHub issue tracker](https://github.com/LapshinLeonid/configurable_automl_engine/issues).

If you are considering larger contributions to the source code, please contact the author first.

If you contribute, please ensure your code:

* 100% covered by tests.
* Has 0 errors when checked by the ruff linter and passes `ruff format --check`.
* Has 0 errors when checked by the mypy static analyzer.
* All docstrings written in English.

# Architecture

If you plan to change the code, read [architecture.md](https://github.com/LapshinLeonid/configurable_automl_engine/blob/v0.5.0/architecture.md) first: it describes how the modules are layered, which imports are allowed and how this is checked in CI.

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
