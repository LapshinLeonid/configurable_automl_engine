# User Guide & API Reference

This document starts with the shortest path to a trained model and then goes
deeper. You do not need to read past [Quick start](#quick-start) and
[How the engine works](#how-the-engine-works) to use the library: every other
setting has a sensible default.

**Contents**

1. [Quick start](#quick-start) — a complete example and the smallest working config.
2. [How the engine works](#how-the-engine-works) — the few terms used throughout this document.
3. [Recipes](#recipes) — copy-paste configs for common tasks.
4. [Configuration reference](#configuration-reference) — every option, grouped by section.
   * [`general`](#general-section) · [Pipeline steps (`phases`)](#pipeline-steps-phases) · [Sanity Gate](#sanity-gate-robust-winner-selection) · [`algorithms`](#algorithms-section) · [`oversampling`](#oversampling-section)
   * Advanced: [early stopping](#early-stopping-pruning) · [additional metrics](#additional-metrics-additional_metrics) · [feature selection](#feature-selection-feature_selection) · [complexity guard](#complexity-guard-configuration-complexity_guard) · [preprocessing presets](#adaptive-feature-preprocessing-preprocessing-presets) · [categorical encoding](#categorical-encoding)
5. [Training results](#training-results) — what `train_best_model` returns.
6. [Python API](#python-api) — functions and classes.
7. [Migrating older configs](#migrating-older-configs).

The machine-readable schema of the config is [config.schema.json](config.schema.json).
A few cross-field rules cannot be expressed in JSON Schema; they live in
[config_parser.py](/src/configurable_automl_engine/training_engine/config_parser.py)
and are listed under "Validation rules" in each section below.

# Quick start

The same example as in the README and [example.py](example.py):

```python
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
```

What this config does:

* `phases` — the pipeline: first 100 tuning attempts (trials) for every
  algorithm (`all_algorithms`), then a `select` step that keeps the algorithm
  with the best validation score (`cv_best`) as the single `winner`.
* `comparison_metric: "r2"` — "best" means the highest R².
* `limit_hyperparameters: true` + `hyperparameters` — only the listed
  parameters are tuned, within the given `[low, high]` ranges; all other
  parameters keep the estimator defaults.
* `path_to_model` — the winner is trained on all rows and saved there.
* `xgboosting` needs `pip install "configurable-automl-engine[xgboost]"`.

The same config as YAML (pass the file path as `config=`):

```yaml
general:
  comparison_metric: "r2"
  phases:
    - {n_trials: 100, action: "all_algorithms"}
    - {name: "final_select", n_trials: 1, action: "select", strategy: "cv_best", output: "winner"}
  path_to_model: "diabetes_model.pkl"
algorithms:
  random_forest:
    enable: true
    limit_hyperparameters: true
    hyperparameters: {n_estimators: [10, 500], max_depth: [3, 20]}
  ridge:
    enable: true
    limit_hyperparameters: true
    hyperparameters: {alpha: [0.1, 1.0]}
  xgboosting:
    enable: true
    limit_hyperparameters: true
    hyperparameters:
      n_estimators: [100, 1000]
      max_depth: [3, 10]
      learning_rate: [0.01, 0.3]
      subsample: [0.5, 1.0]
```

Every setting above has a default, so it can be shortened. The smallest valid
config lists only the algorithms — `general` falls back to R², 50 trials per
algorithm and the `cv_best` winner, and `{}` means "use the built-in search
space":

```yaml
algorithms:
  ridge: {}
  random_forest: {}
```

To use the Sanity Gate or a multi-step search, see [Recipes](#recipes).

# How the engine works

A training run is a short **pipeline** of steps, configured in
`general.phases`. There are two kinds of steps:

* **Search steps** tune hyperparameters with [Optuna](https://optuna.org/).
  Each tuning attempt is called a **trial**: the engine picks hyperparameters,
  trains the model, and measures its validation score.
  * `all_algorithms` — tune every enabled algorithm (`n_trials` attempts each).
  * `refine` — continue tuning only the algorithms that are still in the race,
    starting from their best parameters so far.
* **Select steps** (`action: "select"`) decide which algorithms stay in the race.
  * `strategy` — how candidates are ranked: `cv_best` (by validation score) or
    `sanity_gate` (validation score plus [extra checks](#sanity-gate-robust-winner-selection)
    that reject degenerate models).
  * `output` — what the step keeps: `winner` (one model — this ends the
    pipeline), `top_k` (the best `k`), or `pool` (every candidate close
    enough to the leader).

The pipeline always ends with a select step that keeps a single `winner`.
When `phases` is omitted, the engine uses:

```yaml
phases:
  - {name: "search", action: "all_algorithms", n_trials: 50}
  - {name: "final_select", action: "select", strategy: "cv_best", output: "winner"}
```

Other terms used in this document:

| Term | Meaning |
|------|---------|
| Validation score / CV score | The quality of a trial measured on rows the model was **not** trained on (cross-validation folds or a hold-out part). |
| OOF (out-of-fold) predictions | For every row, the prediction of the fold model that did not see that row. Used for honest quality estimates. |
| `RMSE_oof` | RMSE computed on the OOF predictions. |
| `RMSE_full` | RMSE of a model trained on all rows, measured on the same rows (in-sample). |
| Candidates | The algorithms still in the race at a given step. Search steps tune them, select steps narrow them down. |
| Disqualified algorithm | An algorithm removed from the race: it failed repeatedly during tuning, did not fit the target, or was rejected by the Sanity Gate. Reasons are reported in `result["disqualified_algorithms"]`. |

# Recipes

**More or fewer tuning attempts.** Write the pipeline explicitly and set
`n_trials`:

```yaml
general:
  phases:
    - {action: "all_algorithms", n_trials: 200}
    - {action: "select", strategy: "cv_best", output: "winner"}
```

**Reject degenerate models (recommended for small or noisy data).** Use the
Sanity Gate for the final choice. It double-checks the best candidates and
skips models that predict a near-constant value, ignore most features, or
overfit badly:

```yaml
general:
  phases:
    - {action: "all_algorithms", n_trials: 100}
    - {action: "select", strategy: "sanity_gate", output: "winner"}
```

**Coarse search, then fine-tune the top 3.**

```yaml
general:
  phases:
    - {name: "coarse",  action: "all_algorithms", n_trials: 100}
    - {name: "top3",    action: "select", strategy: "sanity_gate", output: "top_k", k: 3}
    - {name: "fine",    action: "refine", n_trials: 200}
    - {name: "final",   action: "select", strategy: "sanity_gate", output: "winner"}
```

**Search only the parameters you list.** By default your ranges are merged
into the built-in search space. With `limit_hyperparameters: true` only the
listed parameters are tuned; the rest keep the estimator defaults:

```yaml
algorithms:
  random_forest:
    limit_hyperparameters: true
    hyperparameters:
      n_estimators: [50, 300]
      max_depth: [3, 12]
```

**Fixed validation scheme.**

```yaml
general:
  validation_strategy: "k_fold"   # or "train_test_split", "loo", "auto" (default)
  n_folds: 5
```

**Hard time limits.** Timeouts are only supported with process-based
parallelism:

```yaml
general:
  parallel_mode: "processes"
  max_workers: 4
  phase_timeout: 1800    # seconds for a whole search step
  task_timeout: 600      # seconds for one algorithm
```

**An honest final estimate on untouched rows.** Reserve 20 % of the rows that
never take part in model selection; the result then contains `holdout_score`:

```yaml
general:
  holdout_fraction: 0.2
```

**Extra metrics in the report.**

```yaml
general:
  comparison_metric: "rmse"
  additional_metrics: ["r2", "mae"]
```

# Configuration reference

A config has three top-level sections. Only `algorithms` is required:

* `general` — what to optimize, how to validate, the pipeline, parallelism.
* `algorithms` — which models to try and their search spaces.
* `oversampling` — optional (experimental) data augmentation.

## General section

**Main options**

| Option | Default | Description |
|--------|---------|-------------|
| `comparison_metric` | `"r2"` | Metric that defines the best model. See [allowed values](#allowed-values-for-comparison_metric). |
| `phases` | search + `cv_best` winner | The pipeline. See [Pipeline steps](#pipeline-steps-phases). |
| `path_to_model` | `"model.pkl"` | Where the winning model is saved. |
| `serialization_format` | `"pickle"` | `"pickle"` or `"joblib"`. |
| `log_to_file` | `null` | Path of a log file. Without it logs go to the console only. |
| `additional_metrics` | `[]` | Extra metrics reported for the final model; they do not affect the choice. See [Additional metrics](#additional-metrics-additional_metrics). |

**Validation**

| Option | Default | Description |
|--------|---------|-------------|
| `validation_strategy` | `"auto"` | `"train_test_split"`, `"k_fold"`, `"loo"` (leave-one-out) or `"auto"`. `"auto"` picks between LOO, k-fold and a hold-out split from the number of rows and features. |
| `n_folds` | `5` | Number of folds; used only with `"k_fold"` (ignored with `"auto"`). |
| `holdout_fraction` | `null` | Fraction of rows, strictly in `(0, 1)`, reserved **before** tuning for an independent final estimate (`result["holdout_score"]`). The final model is still trained on all rows. |
| `random_state` | `42` | Seed of the data splits, the models and the tuner — the same seed gives the same result. `null` makes the run non-reproducible (a WARNING is logged). Recorded as `result["random_state"]`. |

**Parallelism and time limits**

| Option | Default | Description |
|--------|---------|-------------|
| `max_workers` | CPU count | Maximum number of threads/processes. |
| `parallel_mode` | `"threads"` | `"threads"` or `"processes"`. |
| `parallel_strategy` | `"algorithms"` | What runs in parallel. Currently only `"algorithms"` (different algorithms are tuned at the same time). |
| `phase_timeout` | `null` (3600 s) | Time limit of one search step, seconds (≥ 1). **Only with `parallel_mode: "processes"`**, where hung workers are terminated. In `"threads"` mode setting it is a config error: a running Python thread cannot be stopped, so the limit would only throw away finished results. |
| `task_timeout` | `null` (= `phase_timeout`) | Time limit of one algorithm, seconds (≥ 1). Only with `"processes"`, same reason. |

**Data handling**

| Option | Default | Description |
|--------|---------|-------------|
| `categorical_encoding` | `"one_hot"` | `"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`. See [Categorical encoding](#categorical-encoding). |
| `high_cardinality_threshold` | `50` | Columns with more distinct values than this use `high_cardinality_encoding` instead. Set both options to `null` to turn this off; they must be set together. |
| `high_cardinality_encoding` | `"frequency"` | Encoding of high-cardinality columns (same values as `categorical_encoding`). |
| `hashing_n_components` | `16` | Columns per categorical feature for `"hashing"` (≥ 1). |
| `target_encoding_smoothing` | `20.0` | Smoothing `m` of `"target"` encoding (≥ 0). |
| `target_encoding_fallback` | `null` | `"target"` encoding value for unseen categories; `null` = mean of the target. |
| `inf_policy` | `"as_missing"` | Infinite values in numeric features: `"as_missing"` turns them into missing values (imputed later) with a WARNING; `"error"` rejects the data. Applied at training and prediction time. |
| `datetime_policy` | `"expand"` | Date/time columns: `"expand"` replaces each column with `year`, `month`, `day`, `dayofweek` (and `hour` when it is not always zero); `"drop"` removes them; `"error"` rejects the data. A WARNING lists the affected columns. |
| `drop_id_like_columns` | `false` | ID-like columns (text columns where ≥ 95 % of values are unique, or unique monotonic integers) are only reported with a WARNING by default; `true` removes them. This is a heuristic: a monotonic counter such as a timestamp may be a useful feature. |

**Advanced blocks** (each has its own section below): `pruning`
([early stopping](#early-stopping-pruning)), `feature_selection`
([feature selection](#feature-selection-feature_selection)), `sanity_gate`
([Sanity Gate settings](#sanity-gate-configuration-sanity_gate)),
`complexity_guard` ([complexity guard](#complexity-guard-configuration-complexity_guard)).

### Allowed values for `comparison_metric`

* `"nrmse"`
* `"rmse"`
* `"neg_root_mean_squared_error"`
* `"mae"`
* `"mse"`
* `"r2"`

### Additional metrics (`additional_metrics`)

The optional `general.additional_metrics` parameter is a list of extra quality metrics whose values are computed for the **final trained model** (after the winner is selected) and returned together with the training results.

Rules and behavior:

* **Optional & backward compatible.** If the parameter is absent or empty, the system behaves exactly as before: no `additional_metrics` key is added to the results.
* **Allowed values** — the same set as for `comparison_metric` (see above). An unknown metric name fails config validation before training starts (`ValidationError` with the list of supported values).
* **Informational only.** Additional metrics do **not** participate in hyperparameter optimization, model comparison, or winner selection. The winner is determined solely by the main `comparison_metric`.
* **Duplicates.** If the same metric is listed several times, it is computed and returned exactly once (first-occurrence order is kept). Duplicates are removed at config validation time.
* **Overlap with the main metric.** If an additional metric coincides with the main comparison metric (including aliases of the same metric, e.g. `"rmse"` and `"neg_root_mean_squared_error"`), it is excluded from the list — its value is already returned once as `score`. A warning is logged.
* **Post-validation list.** Both rules above are applied at config validation time, so the resulting `general.additional_metrics` (as exposed on the validated config object and forwarded to the final `ModelTrainer`) may differ from the list the user provided: duplicates are removed and the comparison metric (with its aliases) is dropped. Only the metrics present in the final list are computed and returned.
* **Computation.** Each additional metric is evaluated for the final model with the **same validation method as the comparison metric**: on a hold-out split (`train_test_split`) or as the CV estimate (`k_fold`/`loo`) — never on the full training set. The CV estimate is the mean over folds for additive metrics (MAE and similar per-sample means) and the pooled value over the out-of-fold vector for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D², explained_variance, max error, median absolute error and their aliases), matching the `val_score` semantics. Error-type metrics (RMSE, MAE, MSE, NRMSE) are returned as positive natural values, score-type metrics (R²) as-is. Non-finite values (e.g., `inf` from NRMSE on a constant target) are returned as-is and do not affect training completion or artifact saving.
* **Failure tolerance.** If a single additional metric cannot be computed (scorer error), it is skipped with a warning and does not abort training, model selection, or artifact saving.

Example:

```yaml
general:
  comparison_metric: "rmse"
  additional_metrics: ["r2", "mae", "mse"]
```

### Early stopping (`pruning`)

The optional `pruning` block enables Optuna-based early stopping of unpromising trials during the HPO phase. When enabled, the tuner publishes an intermediate quality score after each natural validation step (e.g., after each cross-validation fold — the cumulative running mean), and the configured pruner may prune a hopeless trial before its full evaluation finishes. Pruned trials are **not** errors: they do not count as fatal failures, do not disqualify the algorithm, and do not stop the optimization. The best model is selected only from fully completed trials using the same metric-maximization rule as without pruning. If all trials are pruned, the standard "no successful trials" scenario applies (no new exceptions).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable` | `bool` | `false` | Master switch for early stopping. When `false` (default), every trial runs to completion — identical behavior to configurations without this block. |
| `strategy` | `string` | `"median"` | Pruning strategy: `"median"` (Optuna `MedianPruner`) or `"hyperband"` (Optuna `HyperbandPruner`). |
| `min_steps` | `int` | `1` | Minimum number of intermediate steps (e.g., CV folds) before the first pruning decision. For `"median"` it maps to `n_warmup_steps`; for `"hyperband"` it maps to `min_resource`. Must be ≥ 1. |
| `n_startup_trials` | `int` | `5` | Number of completed trials before pruning is enabled (`"median"` only; Optuna `n_startup_trials`). Must be ≥ 1. |
| `reduction_factor` | `int` | `3` | Budget reduction factor between Hyperband rounds (`"hyperband"` only; controls aggressiveness). Must be ≥ 2. |

Example:

```yaml
general:
  validation_strategy: "k_fold"
  n_folds: 5
  pruning:
    enable: true
    strategy: "median"
    min_steps: 2
    n_startup_trials: 5
  phases:
    - name: "search"
      n_trials: 50
      action: "all_algorithms"
    - name: "final_select"
      n_trials: 1
      action: "select"
      strategy: "cv_best"
      output: "winner"
```

Validation rules (rejected at config load with a clear message):

* Unknown `strategy` (anything other than `"median"` / `"hyperband"`).
* Non-positive `min_steps`, `n_startup_trials`, or `reduction_factor < 2`.
* Conflict with validation settings: `pruning.enable: true` + `validation_strategy: "k_fold"` + `min_steps > n_folds` (the pruner would never get enough intermediate steps).

Behavior by validation strategy:

* `k_fold` / `loo` — pruning works on natural per-fold (per-sample) steps.
* `auto` — pruning is applied when `auto` resolves to `k_fold` or `loo`; if it resolves to `train_test_split`, pruning is not applied (see below).
* `train_test_split` (hold-out) — there are no intermediate steps, so the pruner is **not** applied; the configuration is accepted with a warning, and every trial runs to completion. This behavior is safe and documented.

### Feature selection (`feature_selection`)

The optional `general.feature_selection` block adds a feature-selection step
in front of the model. It is off by default.

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `"disabled"` | `"disabled"` — no selection; `"always"` — every trial and the final model use it; `"auto"` — the tuner decides per trial whether selection helps (the decision is kept in the winner's `params` as `use_feature_selection`). Always off for `isotonic_regression`. |
| `method` | `"importance"` | `"importance"` (tree-based importance), `"percentile"` (F-test), `"mutual_info"` (mutual information) or `"variance"` (drops near-constant features). |
| `percentile` | `50.0` | Percentage of features kept by `"percentile"` / `"mutual_info"`, in `(0, 100]`. |
| `min_features` | `2` | Never keep fewer features than this (≥ 1). |
| `variance_threshold` | `0.0` | Minimum variance for `"variance"` (≥ 0). |
| `n_estimators` | `50` | Trees used by `"importance"` (≥ 10). |

```yaml
general:
  feature_selection:
    mode: "auto"
    method: "mutual_info"
    percentile: 30
```

### Complexity Guard configuration (`complexity_guard`)

The optional `general.complexity_guard` block is an **adaptive complexity
guard** for the search spaces of trees and boosting on small data. It extends
the existing data-dependent clipping layer (`clip_search_space` in
`common/hyperopt_defaults.py`): on small data the default ranges of
tree/boosting hyperparameters are too wide (e.g. `max_depth` 3–20 or 2–32 on
a 15-row training fold), trials are spent on values that do not change the
model, the TPE surface has flat regions and the search drifts toward overfit
models. The guard tightens the ranges depending on the number of rows (`n` —
the effective training-fold rows) and the feature count.

Two feature counts are used deliberately:

* The size-mode ratio `n / p` uses the **original column count** (a
  categorical column counts as 1, before the one-hot expansion): the
  one-hot expansion must not push large-but-wide data into a smaller size
  mode.
* The rules that depend on the feature count — `max_features` and
  `colsample_bytree` — use **`p` after preprocessing** (what the model
  sees, including one-hot columns and missing-value indicators).

Two independent levels of restrictions (each toggled separately):

* **Level A — lossless clipping** (`lossless: true`): removes values that
  cannot change the model on data of the given size. For example,
  `max_depth=25` on 15 rows is equivalent to `max_depth=6`. Safe. Note: for
  `min_samples_leaf` the level A cap (`≤ floor(n / 2)`) is a *physical*
  bound — larger leaves are unreachable on that sample size — but the
  remaining values below the cap still change the model, so the level is
  "lossless" only in the sense that no reachable leaf size is removed.
* **Level B — regularization** (`regularize: true`): deliberately shifts the
  space toward simpler models on small data. A heuristic. For boosting this
  includes the `n_estimators` budget (see below) — the tree count together
  with `learning_rate` is the main complexity regulator of a boosting
  ensemble.
* **Time-only caps** (`n_estimators` ≤ 300 in the small mode) apply to the
  **bagging** ensembles only (random forest, extra trees) and are marked
  explicitly as "for time", not "for regularization": bagging averages
  independent trees, so more trees only reduce the variance of the ensemble.

The size mode is determined by `n` and the ratio `n / p`, where `p` is the
**original column count** (a categorical column counts as 1, before the
one-hot expansion):

* `"small"`: `n < small_n_threshold` or `n / p < small_np_ratio_threshold`;
* `"medium"`: `n < medium_n_threshold` or `n / p < medium_np_ratio_threshold`
  (and not small);
* `"ordinary"`: everything else — the spaces are not changed.

For example, `n = 4000` rows and 50 original columns (250 one-hot expanded)
give `n / p = 80 >= 20` — the `"ordinary"` mode and unchanged spaces,
while the old `p`-after-preprocessing counting produced `16 < 20` and the
`"medium"` restrictions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `string` | `"apply"` | Operation mode: `"apply"` (default) — restrictions are computed and applied; `"warn"` — computed and logged but NOT applied (the spaces stay unchanged); `"off"` — the guard is disabled. |
| `lossless` | `bool` | `true` | Enable level A (lossless clipping). |
| `regularize` | `bool` | `true` | Enable level B (regularization heuristic). |
| `user_range_policy` | `string` | `"user_wins"` | Policy for an **empty** intersection of a user range (`algo.hyperparameters`) with the guard range: `"user_wins"` (default) — the explicit user range stays and a warning is logged (an explicit setting beats the heuristic); `"engine_wins"` — the guard restriction is applied. |
| `small_n_threshold` | `int` | `50` | Row threshold of the `"small"` mode (≥ 1). |
| `small_np_ratio_threshold` | `float` | `5.0` | Ratio threshold of the `"small"` mode (> 0). `p` is the original column count (a categorical column counts as 1, before one-hot). |
| `medium_n_threshold` | `int` | `300` | Row threshold of the `"medium"` mode (≥ 1). |
| `medium_np_ratio_threshold` | `float` | `20.0` | Ratio threshold of the `"medium"` mode (> 0). `p` is the original column count (a categorical column counts as 1, before one-hot). |
| `size_basis` | `string` | `"rows"` | Basis of `n`: `"rows"` (default) — independent rows; `"groups"` — reserved for group validation (repeats of one group carry less information than independent rows); only `"rows"` is used today. |

Rules of the `"small"` mode (with `n = 15`, `p = 5` as the example):

* **Decision tree / Random forest / Extra trees**
  * `min_samples_leaf`: level A — upper bound ≤ `floor(n / 2)`; level B —
    lower bound 2, upper bound ≤ `floor(n / 4)` (example: 2–3). On `n < 8`
    the range `[2, floor(n / 4)]` is empty and collapses to the lower bound
    2, capped by the physical limit `floor(n / 2)` — the leaf regularization
    does not vanish (it never collapses to `min_samples_leaf = 1`);
  * `max_depth`: level A — ≤ `floor(n / leaf_min) − 1`; level B —
    ≤ `ceil(log2(n / leaf_min))` (example: A gives ≤ 6, B gives 2–3);
  * `max_features` (a fraction): level A — lower bound ≥ `1 / p` (at least one
    feature);
  * `n_estimators` (RF / Extra trees): capped only for time (e.g. 100–300) —
    bagging averages independent trees, so more trees only reduce variance,
    not a regularization.
* **Gradient boosting / XGBoost**
  * `max_depth`: level A — as for trees; level B — 1–3;
  * `min_samples_leaf` (Gradient boosting): as for trees;
  * `min_child_weight` (XGBoost): level B — lower bound 2–3;
  * `subsample`: level B — lower bound ≥ 0.7;
  * `colsample_bytree`: level A — lower bound ≥ `1 / p`;
  * `reg_lambda` (XGBoost): level B — `[1, 100]` on the log scale;
  * `n_estimators`: level B — the upper bound is derived from the space's
    `learning_rate` upper bound: `cap = 60 / lr_high`. For a boosting ensemble
    the number of trees together with `learning_rate` is the main complexity
    regulator (every tree is fitted on the residual of the previous ones), so
    this is a *regularization* restriction: it is disabled together with
    `regularize`, unlike the RF/Extra trees time-only cap. The cap depends on
    the `learning_rate` upper bound so that the product `n_estimators ×
    learning_rate` never exceeds the budget 60 for any sampled pair — e.g. with
    the default lr range `[0.01, 0.3]` the cap is `60 / 0.3 = 200`, with
    `[0.01, 1.0]` it is `60`, with `[0.01, 0.1]` it is `600`. A strict
    per-trial budget (sampling one parameter conditionally on the other) is not
    supported by the guard's per-parameter rules; the upper-bound derivation
    above is the closest the space mechanism allows. When `learning_rate` is
    absent from the space, the default upper bound 0.3 is assumed.
    `learning_rate` itself is not restricted by the guard.

Rules of the `"medium"` mode — the same but softer:
`min_samples_leaf` lower bound 2; `max_depth` ≤ `ceil(log2(n / leaf_min)) + 2`;
XGBoost `max_depth` ≤ 5. In the `"ordinary"` mode the spaces are unchanged.

Interaction with user ranges (`limit_hyperparameters` + `hyperparameters`):

* In the default mode (`limit_hyperparameters: false`) the user ranges
  override and are merged into `DEFAULT_SPACES`, and the guard applies to the
  merged space.
* With `limit_hyperparameters: true` the search space contains ONLY the
  user-listed parameters: the guard restricts those ranges exactly as below,
  and no default-space parameter (e.g. SVR's adaptive `epsilon`) is added.
* The default spaces adapt automatically. The default XGBoost space
  (`DEFAULT_SPACES['xgboosting']`) includes `min_child_weight` (1–10, float)
  and `reg_lambda` (1e-3–100, log scale), so the level B rules of the guard
  fire on the defaults (both are tightened to `≥ 2` / `[1, 100]` on small
  data) and are reported in the log and the `complexity_guard` report — they
  are not restricted only when the user overrides them with a conflicting
  range and `user_range_policy: "user_wins"` keeps the override.
* A range the user set explicitly is intersected with the guard range; on an
  **empty** intersection the `user_range_policy` decides (by default the user
  range stays and a warning is logged). The strict "engine wins" mode is
  enabled via `user_range_policy: "engine_wins"`.
* The result is never empty: `low <= high`, the int/float type and the log
  scale are preserved; a range collapsed to one value fixes the parameter
  (logged). Integer-like parameters (leaf sizes, depths, tree counts) never
  go below their physical minimum of 1 — on extremely small `n` an empty
  range is fixed at 1, never at 0 (sklearn rejects 0 with
  `InvalidParameterError`; XGBoost silently overrides `max_depth=0`).

Transparency:

* Per (algorithm, phase, parameter) one INFO log record: "was → became" with
  the reason, e.g. `max_depth 3–20 → 2–3: n=15, p=5, mode=small, leaf_min=2`.
* The same records are aggregated into the `train_best_model` report under the
  `complexity_guard` key: `{"parameters": [ ... ]}` — **one record per
  restricted parameter** (the A/B rules of the same parameter are merged, and
  identical records of repeated phases are not duplicated) with `algo`,
  `param`, `before`, `after`, `levels` (the list of applied rule levels, e.g.
  `["A", "B"]` or `["B"]`), `size_mode`, `n`, `p`, `leaf_min`, `reason`,
  `applied`, `conflict`, `fixed`.

Note on `max_features`: the parameter accepts both a float (a fraction of the
features, clamped to `(0, 1]`) and an int (a plain feature count, physical
minimum 1). The guard treats them accordingly — an int user range is never
collapsed to a fixed `1`.

Note on `min_samples_leaf`: sklearn also accepts a float in `(0, 1)` as a
fraction of the sample size. The guard's rules are expressed in absolute rows,
so a fractional range is converted to rows (`value × n`) before a rule applies
and divided by `n` afterwards — a user fraction range stays in `(0, 1]` and is
never widened to a row count (a float `>= 1` is a row count and is treated as
integer-like).

Determinism and parallelism:

* The guard is computed from the same inputs (`n`, `p`, the config) in every
  HPO phase, so the TPE history transfer between phases
  (`_transfer_previous_trials`) stays valid — no `HPOHistoryError`.
* In the subprocesses the restrictions are identical: the inputs are computed
  deterministically and travel with the phase snapshot.

Validation rules (rejected at config load with a clear message):

* Unknown keys — the block has `extra="forbid"` (a typo is a hard error).
* `small_n_threshold` must be `< medium_n_threshold`.
* `small_np_ratio_threshold` must be `< medium_np_ratio_threshold`.
* Thresholds must be positive; `mode` must be one of `off`/`warn`/`apply`;
  `user_range_policy` — one of `user_wins`/`engine_wins`; `size_basis` — one
  of `rows`/`groups`.

Example:

```yaml
general:
  phases: [...]
  complexity_guard:
    mode: "apply"
    lossless: true
    regularize: true
    user_range_policy: "user_wins"
    small_n_threshold: 50
    small_np_ratio_threshold: 5.0
    medium_n_threshold: 300
    medium_np_ratio_threshold: 20.0
    size_basis: "rows"
```

## Pipeline steps (`phases`)

`general.phases` is a list of steps executed in order (see
[How the engine works](#how-the-engine-works) for the idea). Omit it to use
the default "search all algorithms, keep the best validation score" pipeline.

### Fields of a step

| Field | Applies to | Default | Description |
|-------|-----------|---------|-------------|
| `action` | all | `"all_algorithms"` | `"all_algorithms"`, `"refine"` or `"select"`. |
| `name` | all | `"phase"` | Your label for the step. Used in logs and as the key of the step's report in `result["sanity_gate"]`. |
| `n_trials` | `all_algorithms`, `refine` | `50` | Tuning attempts per algorithm (≥ 1). Ignored by `select`. |
| `strategy` | `select` only (required) | — | `"cv_best"` or `"sanity_gate"`. |
| `output` | `select` only (required) | — | `"winner"`, `"top_k"` or `"pool"`. |
| `k` | `output: "top_k"` only (required) | — | How many candidates to keep (≥ 1). |

### Search steps

* **`all_algorithms`** — runs `n_trials` tuning attempts for every enabled
  algorithm, in parallel across algorithms. All of them become candidates
  for the next step.
* **`refine`** — runs `n_trials` more attempts for **each** current candidate,
  starting from its best parameters so far (they are tried first, so the score
  of a candidate can only stay the same or improve). Refine does not drop
  candidates — it only updates their parameters and scores. If one algorithm
  fails, only that algorithm is dropped; if all of them fail, the step raises
  `RuntimeError` instead of silently falling back to the old parameters.
  `refine` may also be the first step: then it tunes all enabled algorithms
  from scratch.

All search steps evaluate models on the same data splits, so scores are
comparable between steps. Only the tuner's random seed changes from step to
step (`random_state + step number`) so that the first attempts are not repeated.

### Select steps

A select step ranks the current candidates and keeps some of them; the next
steps work only with what it kept.

| | `output: "winner"` | `output: "top_k"` | `output: "pool"` |
|---|---|---|---|
| `strategy: "cv_best"` — rank by validation score | The best candidate (ties: the order of `algorithms` in the config). Ends the pipeline. | The first `k` candidates. | Every candidate close to the leader (see [shortlist](#how-the-sanity-gate-works)), ranked by validation score, without the extra checks. |
| `strategy: "sanity_gate"` — validation score + checks | The best candidate that passes the [checks](#how-the-sanity-gate-works). Ends the pipeline; the model trained during the checks is reused as the final model. | The first `k` of the ranking: candidates that pass the checks (by `RMSE_oof`), then — only if fewer than `k` pass — the "least bad" rejected ones. | Every candidate that passes the checks (all finalists, "least bad" first, if none pass). |

* Select steps run one after another, never in parallel.
* A select step with nothing to rank raises `RuntimeError`.
* A repeated `sanity_gate` select (for example after `refine`) re-trains its
  candidates from scratch.

### Validation rules

The config is rejected at load time, with a message saying what to change, when:

* the first step is a `select` (there is nothing to rank yet);
* the last step is not a `select` with `output: "winner"`;
* any step follows a `select` with `output: "winner"`;
* `strategy` or `output` is missing on a `select`, or set on a search step;
* `k` is missing for `output: "top_k"`, or set for any other output;
* `action: "refine_winner"` is used (removed — see [Migrating older configs](#migrating-older-configs)).

`k` is not compared with `sanity_gate.top_k_candidates` at load time: if `k`
is larger than the shortlist, you simply get fewer candidates.

### Example

```yaml
general:
  phases:
    - name: "coarse"
      action: "all_algorithms"
      n_trials: 100
    - name: "gate_select"
      action: "select"
      strategy: "sanity_gate"
      output: "top_k"
      k: 3
    - name: "refine"
      action: "refine"
      n_trials: 200
    - name: "gate_final"
      action: "select"
      strategy: "sanity_gate"
      output: "winner"
```

## Sanity Gate: robust winner selection

The best validation score is not always the best model. On small or noisy
data the leader can be a model that predicts almost the same value for every
row, a step function, a model that ignores most features, or a model that
memorized the training data. The Sanity Gate catches these cases. It is used
by select steps with `strategy: "sanity_gate"`; its thresholds are set in the
optional `general.sanity_gate` block.

### How the Sanity Gate works

1. **Shortlist.** Take the candidates whose validation score is close to the
   leader's — within the relative margin `corridor_delta` (15 % by default) —
   and keep at most `top_k_candidates` (3 by default) of them, best first.
2. **Train each finalist once on all rows.** This also produces its OOF
   predictions and `RMSE_oof`; the same trained model is reused for the
   checks and the report. A finalist that fails to train is skipped (the
   reason is logged); if none can be trained, `RuntimeError` is raised.
3. **Run four checks** on each finalist. Letters A–D are the short names used
   in log messages and reports:

   | Check | Switch | What it rejects |
   |-------|--------|-----------------|
   | **A — prediction diversity** | `check_diversity` | Models no better than predicting the mean. By default (`adaptive_diversity: true`) a model fails when its OOF predictions are constant or its `R²_oof < 0`; models with a weak but real signal pass. With `adaptive_diversity: false` it fails when `Var(OOF predictions) / Var(y) < min_prediction_diversity`. |
   | **B — distinct predictions** | `check_unique` | Step-like models that output only a few distinct values: fewer than `max(min_unique_count, min_unique_ratio × N)` distinct OOF predictions. The threshold never exceeds the number of distinct target values, so a perfect model always passes. |
   | **C — unused features** | `check_dead_features` | Models that ignore most inputs: the share of features with a zero coefficient (linear models) or no effect when shuffled (other models) is above `max_dead_feature_ratio`. By default this rejects a model only together with low prediction diversity; otherwise it is a warning. |
   | **D — overfitting** | `check_generalization_gap` | A large gap between out-of-sample and in-sample error: `RMSE_oof / RMSE_full > max_generalization_gap`. By default (`generalization_gap_mode: "warn"`) this is only a warning; `"reject"` makes it a rejection. A perfect in-sample fit (infinite gap) is treated the same way; `RMSE_oof ≈ 0` additionally logs a WARNING about possible target leakage. |

   Checks A and B use the OOF predictions, check D uses both, check C
   inspects the model itself.
4. **Pick the winner**: among the finalists that pass, the one with the lowest
   `RMSE_oof`. The validation score and `RMSE_full` are not used for this
   ranking.

If **every** finalist is rejected, the run does not fail. The "least bad"
finalist wins — fewest failed checks, then the least severe ones (D is the
most severe, then A, B, C), then the lowest `RMSE_oof` — and a WARNING lists
all reasons.

Rejected algorithms are also added to `result["disqualified_algorithms"]`,
the same as algorithms that failed during tuning. Warnings never disqualify.

**How "close to the leader" is measured** (`corridor_mode`):

* `"multiplicative"` — within `δ` of the leader's value: for errors
  `score ≤ best × (1 + δ)`, for R²-like metrics `score ≥ best × (1 − δ)`.
* `"additive"` — within `δ` of the score range:
  `|score − best| ≤ δ × (max − min)` over all candidates. Works when scores can
  be zero or negative.
* `"auto"` (default) — multiplicative for error metrics, additive for R²-like
  metrics; also switches to additive when the multiplicative margin would not
  work (error ≈ 0, or a negative best R²). An unknown metric uses the additive
  form with a warning.

Candidates exactly on the boundary are included. Ties keep the order of the
`algorithms` section. The shortlist is never topped up beyond the margin,
except by the optional family rule: with `enforce_family_diversity: true`,
when all finalists belong to one algorithm family (linear, ensembles,
kernel/GPR), the best member of each missing family within a wider margin
(`family_diversity_multiplier × δ`) is added, up to `top_k_candidates`. Only
families from `allowed_families` take part.

### Sanity Gate configuration (`sanity_gate`)

The block only holds thresholds; it does nothing without a select step with
`strategy: "sanity_gate"`. Omitting it (or writing `{}`) uses the defaults.
Unknown keys are rejected, so a typo is an error.

**Shortlist**

| Field | Default | Description |
|-------|---------|-------------|
| `top_k_candidates` | `3` | Maximum number of finalists (≥ 1). Larger than the number of algorithms is fine. |
| `corridor_delta` | `0.15` | Margin `δ` from the leader, strictly between 0 and 1. To keep only the leader, use a tiny value such as `1e-9`. |
| `corridor_mode` | `"auto"` | `"multiplicative"`, `"additive"` or `"auto"`. |
| `enforce_family_diversity` | `false` | Add the best model of each missing algorithm family. |
| `algorithm_families` | `null` | Custom `{algorithm: family}` map for the family rule. |
| `allowed_families` | `null` | Families that may be added (`null` — all). |
| `family_diversity_multiplier` | `1.5` | Wider margin for the family rule (> 0). |

**Check A — prediction diversity**

| Field | Default | Description |
|-------|---------|-------------|
| `check_diversity` | `true` | Enable the check. |
| `adaptive_diversity` | `true` | Compare with the "predict the mean" baseline (see the table above). `false` uses the fixed threshold below. |
| `min_prediction_diversity` | `0.15` | Fixed threshold for `Var(OOF predictions) / Var(y)`, in `(0, 1]`; used only with `adaptive_diversity: false`. High values (such as `1.0` on noisy data) are accepted but reject good models. |

**Check B — distinct predictions**

| Field | Default | Description |
|-------|---------|-------------|
| `check_unique` | `true` | Enable the check. |
| `min_unique_count` | `5` | Minimum number of distinct predictions (≥ 1). |
| `min_unique_ratio` | `0.05` | Minimum share of distinct predictions, in `(0, 1]`. |

**Check C — unused features**

| Field | Default | Description |
|-------|---------|-------------|
| `check_dead_features` | `true` | Enable the check. |
| `max_dead_feature_ratio` | `0.4` | Maximum share of unused features, in `[0, 1]`. |
| `dead_features_require_low_diversity` | `true` | Reject only when there are many unused features **and** prediction diversity is low. |
| `zero_coef_tolerance` | `1e-12` | A linear coefficient with `abs(coef) <=` this value counts as zero (≥ 0). |
| `check_permutation_sensitivity` | `true` | Use the shuffle test for non-linear models. `false` skips it (the coefficient test for linear models remains). |
| `permutation_max_rows` | `5000` | Rows sampled for the shuffle test (≥ 1). |
| `permutation_max_features` | `100` | Maximum number of features shuffled (≥ 1). |
| `permutation_repeats` | `3` | Shuffles per feature (≥ 1). |
| `permutation_seed` | `42` | Seed of the shuffles. |
| `permutation_tolerance` | `1e-3` | Relative change below which a feature counts as unused (≥ 0). |

**Check D — overfitting**

| Field | Default | Description |
|-------|---------|-------------|
| `check_generalization_gap` | `true` | Enable the check. |
| `max_generalization_gap` | `1.5` | Maximum `RMSE_oof / RMSE_full` (≥ 1). A gap exactly equal to the threshold passes. |
| `generalization_gap_mode` | `"warn"` | `"warn"` — a large gap is reported in `soft_warnings` only; `"reject"` — it disqualifies the model. Missing predictions are always a rejection. |

**Time limit**

| Field | Default | Description |
|-------|---------|-------------|
| `audit_time_budget_seconds` | `0` | Total time for the checks, seconds; `0` — no limit. Finalists not checked in time get the reason "audit skipped: budget exceeded" and cannot win. |

**Validation rules** — rejected at load time: unknown keys; the removed
`mode` key (see [Migrating older configs](#migrating-older-configs));
unknown `corridor_mode` or `generalization_gap_mode`; values outside the
ranges listed above.

Example:

```yaml
general:
  phases:
    - {name: "search", action: "all_algorithms", n_trials: 50}
    - {name: "gate_select", action: "select", strategy: "sanity_gate", output: "winner"}
  sanity_gate:
    top_k_candidates: 3
    corridor_delta: 0.15
    max_generalization_gap: 1.5
    generalization_gap_mode: "reject"   # treat strong overfitting as a rejection
algorithms:
  ridge: {}
  random_forest:
    hyperparameters:
      n_estimators: [50, 300]
```

### Sanity Gate report

Every select step with `strategy: "sanity_gate"` adds a section to
`result["sanity_gate"]`, keyed by the step `name`. Steps with
`strategy: "cv_best"` add nothing, and without any Sanity Gate step the key
is absent.

| Key of `result["sanity_gate"][<step name>]` | Type | Description |
|-----|------|-------------|
| `strategy` | `str` | Always `"sanity_gate"`. |
| `generalization_gap_mode` | `str` | The check D mode used: `"warn"` or `"reject"`. |
| `pool` | `list[str]` | Finalists, best first. |
| `audit` | `dict[str, dict]` | Per finalist: `is_valid`, `reasons` (rejections), `soft_warnings`, `diversity_ratio`, `unique_ratio`, `dead_features_count`, `generalization_gap` (`null` when disabled or not computable), `generalization_gap_infinite`, `generalization_gap_mode`, `severity`. |
| `disqualified` | `dict[str, list[str]]` | Rejected finalists and their reasons (also copied to `result["disqualified_algorithms"]`). Warnings are not listed here. |
| `fallback_used` | `bool` | `true` when every finalist was rejected and the "least bad" one won. |
| `winner_rmse_oof` | `float \| null` | `RMSE_oof` of the winner. |
| `overhead_seconds` | `float` | Time spent on training the finalists and running the checks. |

### Things to know

* **Hold-out validation gives partial OOF predictions.** With
  `train_test_split` only the hold-out rows have OOF predictions, so checks A,
  B and `RMSE_oof` use those rows only. Prefer `k_fold` or `loo` for the most
  reliable checks.
* **Small data.** Check B relies on `min_unique_count` when `N` is small. A
  constant target skips check A (with a warning); check B still applies.
* **Weak signal.** With the default adaptive check A, an honest model with a
  low R² (for example 0.1) is not rejected; a model worse than predicting the
  mean (`R²_oof < 0`) is.
* **Correlated features.** The shuffle test can mark one of two correlated
  features as unused (the model gets the same information from the other).
  That is why check C only warns by default.
* **Cost.** `permutation_max_rows` / `permutation_max_features` bound the
  shuffle test; `check_permutation_sensitivity: false` turns it off.

## Algorithms section

The `algorithms` section is a dictionary where the key is the algorithm name, and the value is a set of configurations for that algorithm.

Supported algorithms:

* `"elasticnet"`
* `"sgdregressor"`
* `"decision_tree"`
* `"random_forest"`
* `"extra_trees"`
* `"gradient_boosting"`
* `"adaboost"`
* `"poissonregressor"`
* `"gammaregressor"`
* `"tweedieregressor"`
* `"gaussian_process_regression"`
* `"isotonic_regression"`
* `"nearest_neighbors_regression"`
* `"svr"`
* `"ardregression"`
* `"glm"`
* `"ridge"`
* `"lasso"`
* `"xgboosting"`

Short aliases are also supported: `"dt"`, `"rf"`, `"et"`, `"gb"`, `"ab"`, `"sgd"`, `"knn"`, `"gpr"`, `"ard"`, `"xgboost"`, and others.

**Target-domain guard for GLMs.** Before the first HPO phase the engine checks that the target fits the loss domain of the GLMs: `gammaregressor` requires `y > 0`, `poissonregressor` requires `y >= 0`, and `tweedieregressor` / `glm` follow their `power` (`power <= 0` — any `y`; `0 < power < 2` — `y >= 0`; `power >= 2` — `y > 0`; without `power` in the search space the estimator default `power=0` accepts any target). An algorithm whose domain does not fit the target is excluded **before any trial runs**: its reason lands in `result["disqualified_algorithms"]` and a single `WARNING` is logged. A Tweedie `power` search range that is only partly valid is clipped to its valid part instead (e.g. `[0, 3]` on a target with negative values becomes `[0, 0]`; on a target with zeros the upper bound is kept just below `2`; categorical options are filtered), with a `WARNING` describing the change.

Algorithm configuration consists of:

* `enable` — boolean flag, whether hyperparameter search is performed for the algorithm.
* `limit_hyperparameters` — (optional) boolean flag (default `false`). `true` — the search is restricted to ONLY the parameters listed in `hyperparameters`; every other parameter stays at the estimator default (no `DEFAULT_SPACES` entries are added). `false` — the user ranges override and are merged into the algorithm's default search space. `true` requires a non-empty `hyperparameters` block (rejected at config load otherwise).
* `hyperparameters` — (optional) hyperparameter value constraints, unique to each algorithm. Keys must match hyperparameters accepted by the estimator constructor (including legacy names from `LEGACY_PARAM_MAPPINGS`, e.g. `n_iter` → `max_iter` for ARDRegression). The allowed set is derived from the estimator signature via [`get_allowed_hyperparameters`](src/configurable_automl_engine/models.py), not from the default search space — so parameters outside `DEFAULT_SPACES` (e.g. `min_samples_split`, `max_features`) are valid as long as the estimator accepts them. Unknown names (typos) are rejected at configuration validation. Two documented exceptions for `isotonic_regression`: `feature_index` — an integer data-adapter hint (not an estimator constructor parameter) consumed by `IsotonicDataTransformer`; it selects the column fed to the strictly univariate isotonic model **relative to the numerical block** of the preprocessed matrix (one-hot dummies of categorical columns are not meaningful inputs, so searchable/selectable indices cover only the numerical features, `0..n_numeric−1`). When the engine path omits `feature_index`, the tuner merges a data-driven numeric range into the search space; an explicit user value is kept as is. `isotonic_feature` — an informational read-only key reported in the HPO result: the name (from the preprocessor's `get_feature_names_out`) of the numerical feature actually chosen. It is consumed by the engine (not passed to the estimator). The `IsotonicRegression` model itself defaults to `out_of_bounds='clip'` (unless set explicitly) so that the default hold-out validation never fails with NaN predictions.

A numeric range is written as `[low, high]` or with an explicit type: `[low, high, "float"]`, `[low, high, "float_log"]` or `[low, high, "int"]`. When the type is **not** written explicitly, it is taken from the **parameter's type** (the default search space or the estimator signature), not from how the boundary numbers look — so `[1, 50]` for a real-valued parameter (e.g. `C` of an SVR or `alpha` of a Tweedie) stays a real range instead of silently becoming an integer one. The **logarithmic scale** (`float_log`) is used by default when the parameter is searched on a log scale by default — a scale parameter (`alpha`, `C`, `epsilon`, `gamma`, `learning_rate`) or a parameter whose default search space entry is `float_log` (e.g. `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` of an ARD regression or `eta0` of an SGD regressor) — and both bounds are positive; an explicit type (`[1, 50, "int"]`) is always honored. An explicit `[low, high, "float_log"]` requires `low > 0` (the log scale needs strictly positive bounds).

When the best trial of an HPO phase lies at the boundary of a numeric search range, the tuner logs a warning — "the true optimum may lie beyond the range, widen it" — signalling that the search was cut off. The check covers dict-based numeric search spaces (YAML/`SearchSpaceEntry`, including the `space_overrides` dict form of a direct `optimize()` call); function-based spaces (custom `space_fn`) and categorical parameters are not checked.
* `preprocessing` — (optional) explicit override of the feature preprocessing preset for this algorithm. Takes priority over the automatic class-based selection. May override the whole preset or only some fields:

```yaml
algorithms:
  ridge:
    enable: true
    preprocessing:
      imputation_strategy: median   # optional: "mean" | "median"
      scaling: standard              # optional: "standard" | "robust" | "none"
```

Invalid values in the `preprocessing` block are rejected at configuration validation with a clear message.

## Adaptive Feature Preprocessing (preprocessing presets)

The engine automatically chooses the feature preprocessing strategy based on the selected regression algorithm. A **preprocessing preset** declaratively describes the numeric-feature handling rules: the missing-value imputation strategy and whether/how scaling is applied. Each supported algorithm is assigned to an **algorithm class** with common preprocessing requirements; membership is fixed in the default mapping table (see [`preprocessing_presets.py`](src/configurable_automl_engine/preprocessing_presets.py)) — adding a new algorithm only requires registering it there.

### Algorithm classes

| Class | Algorithms | Imputation | Scaling |
|-------|-----------|------------|---------|
| `scale_sensitive` — scale-sensitive models (linear, kernel, distance-based) | `elasticnet`, `sgdregressor`, `ridge`, `lasso`, `ardregression`, `svr`, `nearest_neighbors_regression`, `gaussian_process_regression` | `mean` (standard) | `standard` (always applied) |
| `trees` — trees and tree ensembles | `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `adaboost`, `xgboosting` | `median` (robust to outliers) | `none` (never applied) |
| `glm_skewed` — GLMs with skewed distributions | `poissonregressor`, `gammaregressor`, `tweedieregressor`, `glm` | `median` (skewness-aware, robust to outliers) | `robust` (RobustScaler) |
| `univariate` — specialized univariate algorithms | `isotonic_regression` | `median` | `none` |
| `default` — fallback for supported algorithms without an explicit registration | any supported algorithm missing from the mapping | `mean` | `standard` |

Unknown algorithm names (typos) are rejected with a `ValueError` listing the available algorithms, instead of silently falling back to the default preset.

### Composition of a preset

Each preset defines:

* `imputation_strategy` — missing-value imputation strategy for numeric features: `mean` (standard) or `median` (robust to outliers / skewed distributions).
* `scaling` — whether scaling is needed and its type: `standard` (`StandardScaler`), `robust` (`RobustScaler`, robust to outliers), or `none` (numeric features are passed to the model without scaling).

### Selection and override rules

* The preset is resolved automatically from the algorithm name (aliases are supported) before training starts; the decision is made once, so there is no measurable overhead. Algorithm names are validated against the supported registry — unknown names raise `ValueError`.
* The same preset is applied at every lifecycle stage — hyperparameter optimization (HPO), final training, and prediction — through the single resolution point used by both `tuner.optimize` and `ModelTrainer`.
* Explicit user override (the `preprocessing` block in the algorithm config, or the `preprocessing_override` argument of `ModelTrainer`/`tuner.optimize`) takes priority over the automatic selection. Fields left unset keep the automatically chosen values.
* The resolved preset is logged (INFO level) and stored on the trainer as `preprocessing_preset`, so the applied preprocessing can always be determined from logs.

## Oversampling section

The optional `oversampling` section may include:

* `enable` — (optional) enable oversampling.
* `multiplier` — (optional) factor to increase dataset size (minimum 1.0).
* `algorithm` — (optional) oversampling algorithm.
* `target_bins` — (optional, flat alias `data_oversampling_target_bins`) number of quantile bins for the regression oversampling strategy (default 8, minimum 2).

### Supported oversampling algorithms

* `"random"`
* `"smote"`
* `"adasyn"`

The engine is regression-only: any numeric target (float or integer — counts, whole-unit prices) is oversampled through the regression path regardless of integrality. The target is discretized into `target_bins` quantile bins, the bins are resampled as pseudo-classes, and the real target values are restored afterwards:

* `random` — every duplicated row keeps its source row's real target (`RandomOverSampler.sample_indices_`), never a bin midpoint;
* `smote` / `adasyn` — synthetic rows are generated SMOTER-style: for a synthetic sample between row `i` and a same-bin neighbor `n` with gap `u`, both the features (`x = x_i + u·(x_n − x_i)`) and the target (`y = y_i + u·(y_n − y_i)`) are interpolated, so synthetic `y` values lie between the two parents' values. `adasyn` uses density-weighted parent selection (rows whose feature-space neighborhoods are dominated by other bins are replicated more often), following the original ADASYN idea. Categorical columns follow the SMOTENC rule — a category that differs between the two parents keeps the first parent's value.

The classification path is kept only for standalone `DataOversampler` calls with an explicit `task="classification"` argument (the default is `"regression"`).

> ⚠️ **Experimental**: the nightly evidence gate (`benchmarks/run_oversampling_benchmark.py`) observed a median held-out R² degradation of `adasyn` on the count dataset of about 0.019 (> 0.01) relative to the no-oversampling baseline. The feature is therefore marked experimental: a WARNING is logged whenever the oversampler actually runs, and you should verify the effect of oversampling on your own data before relying on it. When the gate passes again, the experimental mark is removed.

Gaussian noise injection is **not a separate algorithm** — it is enabled with the `add_noise` flag of `DataOversampler` (intensity is controlled by `noise_level`, default `0.01`) and applies to numeric features **after** resampling, for any of the three algorithms. Integer columns are cast to float when `add_noise=True`, and categorical columns are excluded from the noise. Noise is not supported on sparse input (e.g., the output of the `hashing` encoder): combining `add_noise=True` with a sparse matrix raises `TypeError`.

# Training results

`train_best_model` returns a dict:

```python
{
    "algorithm": "ridge",          # the winner
    "score": 0.123,                # its validation score (comparison_metric)
    "metric": "rmse",              # general.comparison_metric
    "params": {...},               # the winner's hyperparameters
    "model_path": "model.pkl",     # where the trained model was saved
    "random_state": 42,            # seed of the run (None = not reproducible)
    "oof_score": 0.131,            # honest estimate of the final pipeline (see below)
    "n_trials_evaluated": 47,      # trials behind "score"
    # present only in some runs:
    "additional_metrics": {"r2": 0.98, "mae": 0.07},   # general.additional_metrics
    "holdout_score": 0.118,        # general.holdout_fraction
    "sanity_gate": {...},          # Sanity Gate select steps, by step name
    "disqualified_algorithms": {"svr": "reason ..."},
    "complexity_guard": {"parameters": [...]},
}
```

| Key | When present | Meaning |
|-----|--------------|---------|
| `algorithm`, `params` | always | The winning algorithm and its hyperparameters. |
| `score` | always | Validation score of the best trial, in the metric's natural form: RMSE/MAE/MSE/NRMSE are positive, R² is as is. |
| `metric` | always | The metric name from `general.comparison_metric`. |
| `model_path` | always | Path of the saved model. |
| `random_state` | always | Seed of the run; `None` for a non-reproducible run. |
| `oof_score` | when OOF predictions exist | The same metric computed on the final pipeline's out-of-fold predictions. See below. |
| `n_trials_evaluated` | when known | Completed trials of the winner in the step that produced `score`. |
| `additional_metrics` | `general.additional_metrics` is non-empty | `{metric: value}` for the final model, same form as `score`. |
| `holdout_score` | `general.holdout_fraction` is set | The winner's metric on the rows reserved before tuning. |
| `sanity_gate` | at least one Sanity Gate select step | Per-step report, see [Sanity Gate report](#sanity-gate-report). |
| `disqualified_algorithms` | something was disqualified | `{algorithm: reason}` — failures during tuning, a target that does not fit a GLM, or a Sanity Gate rejection. |
| `complexity_guard` | the guard changed a search range | One "before → after" record per changed parameter, see [Complexity guard](#complexity-guard-configuration-complexity_guard). |

**`score` vs `oof_score`.** `score` is the best of many trials measured on the
same folds, so it is slightly optimistic. `oof_score` re-measures the final
pipeline: every row is predicted by a model that did not see it. The
difference (`score − oof_score` for R²-like metrics, `oof_score − score` for
errors) is how optimistic `score` is; a positive value means the honest
estimate is worse. `n_trials_evaluated` is the "N" in "best of N". With
`train_test_split` only the hold-out rows have OOF predictions.

**`holdout_score`.** With `general.holdout_fraction` the reserved rows never
take part in tuning or selection. `holdout_score` is the winner (trained on
the remaining rows with its best parameters) measured on them. The saved
model is then trained on **all** rows, including the reserved ones.

# Python API

## `train_best_model(config, df, target, model_path_override)`

The main entry point for the AutoML pipeline. Validates data, runs multi-phase hyperparameter optimization, selects the best algorithm, and saves the final model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str`, `Path`, `Config`, or `dict` | Configuration — path to YAML/JSON file, `Config` object, or dictionary. |
| `df` | `pd.DataFrame` | Input dataset. |
| `target` | `str` or `None` | Name of the target column. Defaults to `"target"`. |
| `model_path_override` | `str`, `Path`, or `None` | Alternative path to save the model. |

**Returns:** `dict[str, Any]` — the winner, its scores and diagnostic reports. See [Training results](#training-results) for every key.

**Raises:** `TypeError` for unsupported config types; `RuntimeError` if no algorithm succeeds.

---

## `optimize(algo_name, X, y, ...)`

Runs hyperparameter optimization for a single algorithm using Optuna.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algo_name` | `str` | — | Algorithm name (e.g., `"random_forest"`). |
| `X` | `np.ndarray` or `pd.DataFrame` | — | Feature matrix. |
| `y` | `np.ndarray`, `pd.Series`, or `pd.DataFrame` | — | Target vector. |
| `data_oversampling` | `bool` | `False` | Enable oversampling. |
| `data_oversampling_multiplier` | `float` | `1.0` | Oversampling multiplier. |
| `data_oversampling_algorithm` | `str` | `"random"` | Oversampling algorithm (`"random"`, `"smote"`, `"adasyn"`). |
| `data_oversampling_target_bins` | `int` | `8` | Number of quantile bins for the regression oversampling strategy (minimum 2). |
| `metric` | `str` | `"r2"` | Metric to maximize. |
| `val_method` | `ValidationStrategy` or `str` | `"k_fold"` | Validation method (legacy parameter). |
| `validation_strategy` | `ValidationStrategy` or `str` | `"k_fold"` | Validation method (`"train_test_split"`, `"k_fold"`, `"loo"`, `"auto"`). |
| `train_test_split_test_size` | `float` | `0.2` | Test set size when using `train_test_split` validation. Ignored when using `"auto"`. |
| `n_folds` | `int` | `5` | Number of CV folds. |
| `n_trials` | `int` | `50` | Number of Optuna trials. |
| `random_state` | `int` or `None` | `42` | Random seed of the data splits and the preprocessor (CV folds, `train_test_split`, pruning folds, hashing / target-encoding cross-fit). |
| `sampler_seed` | `int` or `None` | `None` | Seed of the TPE sampler, separated from `random_state`: each HPO phase may use a different `sampler_seed` (e.g. `random_state + step number`) while the data splits stay fixed. `None` — the sampler falls back to `random_state` (backward-compatible). |
| `space_overrides` | `dict[str, Callable]` or `None` | `None` | Custom search space overrides. |
| `initial_params` | `dict[str, Any]` or `None` | `None` | Hyperparameters from the previous HPO phase, enqueued as the first trial (`enqueue_trial`) — the "monotonicity anchor" between phases. |
| `previous_trials` | `list[optuna.trial.FrozenTrial]` or `None` | `None` | Trials of the previous HPO phase to be transferred into the new study via `study.add_trials` (the TPE history warm-start). Transferred BEFORE `enqueue_trial`/`optimize`; they feed the TPE sampler but never count as the current phase's own trials. Empty list — no-op. Trials whose distributions are incompatible with the current search space raise `HPOHistoryError` (fail fast). |
| `preprocessing_override` | `dict` or `PreprocessingOverride` or `None` | `None` | Explicit override of the preprocessing preset. Partial (`{"scaling": "none"}`) or full overrides are supported; takes priority over automatic class-based selection. |
| `pruning` | `dict[str, Any]` or `None` | `None` | Early-stopping settings: `{"enable": bool, "strategy": "median"\|"hyperband", "min_steps": int, "n_startup_trials": int, "reduction_factor": int}`. When `None` or `enable: false`, trials run to completion (identical to previous behavior). |
| `feature_selection_cfg` | `FeatureSelectionCfg`, `dict`, or `None` | `None` | Feature selection config (method, percentile, `min_features`, etc.). `None` means `FeatureSelectionCfg()` (`mode='disabled'`). In `'always'` mode every trial pipeline (and the final model) contains the `feature_selector` step; in `'disabled'` it never does; in `'auto'` Optuna decides per trial via the `use_feature_selection` categorical (recorded in `best_params`). For `isotonic_regression` the flag is forcibly turned off. |
| `complexity_guard_cfg` | `ComplexityGuardConfig`, `dict`, or `None` | `None` | Settings of the adaptive complexity guard for tree/boosting search spaces (from `general.complexity_guard`). `None` — the guard is disabled (the exact legacy behavior for direct callers). When enabled, the dict-based search spaces of trees/boostings are tightened depending on `n` (effective training-fold rows) and the feature counts: the size mode uses the original column count (`p_size`, a categorical column counts as 1), the `max_features`/`colsample_bytree` rules use `p` (features after preprocessing). |
| `user_hyperparameter_keys` | `list[str]` or `None` | `None` | Names of parameters set explicitly by the user (`algo.hyperparameters`); the guard intersects its ranges with these user ranges and applies `user_range_policy` on an empty intersection. |
| `fit_best_model` | `bool` | `True` | Whether to train the final model on all provided data after the search. Set to `False` when only the best parameters and score are needed (e.g., the training engine's HPO phases — the final model is trained by its own trainer): the redundant full-data fit is skipped and the `model` field of the result is `None`. |

**Returns:** `OptimizeResult` (a dataclass with `model`, `params`, `score`, `trials`, `complexity_guard` and `n_trials_evaluated`) when the phase produced a valid result; `None` when **no trial of the current phase succeeded** (e.g., `n_trials=0`, or every trial is pruned via a non-fatal error). `model` is `None` when `fit_best_model=False` (the full-data fit is skipped); otherwise it is the trained model with the best parameters. `trials` always contains ONLY the current phase's own Optuna trials (transferred `previous_trials` are excluded), so the caller can warm-start the next phase without duplicating computations. `complexity_guard` is the list of the adaptive complexity guard records for the phase (`None` when the guard is disabled or produced no records). `n_trials_evaluated` is the number of trials actually evaluated by the phase (own COMPLETED trials; transferred history never counts) — the honest denominator of the best-trial `score` (the "best of N" maximum). `None` is the explicit failure signal — the orchestrator (`train_best_model`) then excludes the algorithm from candidates instead of treating the `HPO_WORST_SCORE` sentinel (`-3.4028235e38`) as a valid result; transferred history never masks a total failure of the current phase. The sentinel itself is returned as `score` only when trials complete but all produce a non-finite metric.

In `'auto'` mode `params` additionally contains the service key `use_feature_selection` (`True`/`False`) — the winner's decision about feature selection, which is preserved between multi-phase HPO runs (passed to the next phase via `initial_params`/`enqueue_trial`) and consumed by `train_best_model` for the final fit. The key is absent in `'always'`/`'disabled'` modes and for `isotonic_regression` (feature selection is forcibly turned off there). The key is never passed to the base model constructor.

**Raises:** `ValueError` for invalid `n_trials`; `HyperoptError` for missing search spaces; `HPOHistoryError` when `previous_trials` are incompatible with the current search space (fail fast); `InvalidAlgorithmError` after 5 consecutive fatal failures.

---

## `create_model(algorithm, **hyperparams)`

Creates a scikit-learn compatible regressor instance by algorithm name.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `algorithm` | `str` | Algorithm name or alias (e.g., `"rf"`, `"xgboost"`, `"knn"`). |
| `**hyperparams` | `Any` | Hyperparameters passed to the estimator constructor. |

**Returns:** `RegressorMixin` instance.

**Raises:** `TypeError` if `algorithm` is not a string; `ValueError` for unknown algorithms; `ImportError` for missing optional dependencies.

**Notes:** Automatically cleans and remaps hyperparameters via `clean_hyperparameters()`. Sets `max_iter=10000` for SVR. Sets `kernel=RBF(1.0)` for GaussianProcessRegressor by default. Injects a fixed `random_state=42` into stochastic estimators (trees, forests, boosting, randomized linear models, XGBoost) unless the caller set `random_state` explicitly — an explicit value always wins. An explicit `random_state=None` (a truly random fit) is respected and produces a WARNING that the fit is not reproducible.

---

## `ModelTrainer`

Orchestrator class for training, validation, and serialization of regression models.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | `str` | `"elasticnet"` | Algorithm name. |
| `hyperparams` | `dict` or `None` | `None` | Model hyperparameters. |
| `metric` | `str` | `"r2"` | Evaluation metric. |
| `random_state` | `int` or `None` | `42` | Random seed. |
| `data_oversampling` | `bool` | `False` | Enable oversampling. |
| `data_oversampling_multiplier` | `float` | `1.0` | Oversampling multiplier. |
| `data_oversampling_algorithm` | `str` | `"random"` | Oversampling algorithm (`"random"`, `"smote"`, `"adasyn"`). |
| `data_oversampling_target_bins` | `int` | `8` | Number of quantile bins for the regression oversampling strategy (minimum 2). |
| `serialization_format` | `SerializationFormat` | `pickle` | Save format. |
| `categorical_features` | `list[str]` or `None` | `None` | Categorical column names. |
| `numerical_features` | `list[str]` or `None` | `None` | Numerical column names. |
| `id_column` | `str` or `None` | `None` | ID column to exclude. |
| `encoding_strategy` | `str` | `"one_hot"` | Categorical encoding strategy: `"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`. |
| `additional_metrics` | `Iterable[str]` or `None` | `None` | Additional (informational) metrics computed for the trained model with the same validation method as the main metric (hold-out split or the CV estimate — the fold mean for additive metrics, the pooled OOF value for non-additive ones — never a train score). Any iterable of strings is accepted (list, tuple, set, generator); names are normalized to lowercase. Stored in `additional_scores` after `fit()`. |
| `validation_strategy` | `str` or `ValidationStrategy` | `"train_test_split"` | Validation strategy for the final model quality estimate: `"train_test_split"` (one hold-out split), `"k_fold"` (CV fold mean), `"loo"` or `"auto"`. The engine path always passes the already-resolved strategy. |
| `n_folds` | `int` | `5` | Number of folds for `"k_fold"` (`>= 2`). |
| `test_size` | `float` or `int` | `0.2` | Hold-out size: a fraction in `(0, 1)` or an integer number of rows (like sklearn `train_test_split`). |
| `preprocessing_override` | `dict` or `PreprocessingOverride` or `None` | `None` | Explicit override of the preprocessing preset. Partial or full overrides are supported; takes priority over the automatic class-based selection. |
| `high_cardinality_threshold` | `int` or `None` | `50` (engine config) / `None` (low-level API) | Cardinality threshold for the automatic high-cardinality mode (`>= 0`). Columns with more unique values than the threshold are encoded with `high_cardinality_encoding` (default `"frequency"`); the rest use `encoding_strategy`. The engine config enables the mode by default; set both parameters to `None` to disable it. |
| `high_cardinality_encoding` | `str` or `None` | `"frequency"` (engine config) / `None` (low-level API) | Encoding strategy for high-cardinality columns (`"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`). Must be set together with `high_cardinality_threshold`. |
| `hashing_n_components` | `int` | `16` | Number of binary columns per categorical column when `"hashing"` encoding is used (`>= 1`). Controls output dimensionality regardless of column cardinality. |
| `target_encoding_smoothing` | `float` | `20.0` | Smoothing parameter `m` of target encoding (`>= 0`). With `m=0` the raw per-category mean is used; larger values pull rare categories toward the global mean. |
| `target_encoding_fallback` | `float` or `None` | `None` | Fallback value for categories unseen in training under target encoding. `None` means the global target mean. |
| `inf_policy` | `str` | `"as_missing"` | Policy for infinite values (`±inf`) in numeric features during `fit()`: `"as_missing"` (default) — replace with NaN and log a WARNING listing the columns/counts; `"error"` — reject with `TrainingError` naming the columns. |
| `datetime_policy` | `str` | `"expand"` | Policy for datetime/timedelta columns during `fit()`: `"expand"` (default) — replace each column with derived numeric components (`year`, `month`, `day`, `dayofweek`, `hour` if non-zero) and log a WARNING listing the generated columns; `"drop"` — remove and warn; `"error"` — reject. |
| `drop_id_like_columns` | `bool` | `False` | Whether ID-like columns (object/string with `nunique/n >= 0.95`, or a unique monotonic integer column) are dropped during `fit()`. `False` (default) — only a WARNING naming the columns; `True` — they are removed. |
| `warn_id_like` | `bool` | `True` | Whether the ID-like WARNING is emitted during `fit()`. The engine path (via `prepare_dataset`) passes `False` so the diagnostics are not duplicated; standalone calls warn by default. |

**Key attributes:**

* `preprocessing_preset` — the resolved `PreprocessingPreset` (algorithm class, imputation strategy, scaling type) applied during `fit()`. Useful for observability and serialization.

**Key methods:**

* `fit(X, y)` — Train the model with full preprocessing pipeline. The target is validated like at the engine entry point: a non-numeric or NaN/±inf target, or a constant one, is rejected with `TrainingError`.
* `predict(X)` — Make predictions on new data. Accepts the raw schema the model was trained on: datetime/timedelta columns expanded at training time (`datetime_policy="expand"`) are expanded the same way first (object-dtype date strings are parsed with `pd.to_datetime`), and extra columns (e.g. dropped ID-like ones) are ignored.
* `register_datetime_expansion(mapping)` — Record a datetime expansion applied upstream of `fit()` (used by the engine, which expands the data in `prepare_dataset`), so that `predict` re-applies it to raw data.
* `save(path)` — Serialize to disk.
* `load(path)` — Load from disk (class method).

**Attributes after `fit()`:**

* `datetime_expansion_` — `dict[str, list[str]]`: source datetime/timedelta column → generated component columns; re-applied by `predict` (empty when nothing was expanded).
* `val_score` — value of the main metric computed on data the model has **not** been trained on: a hold-out split score (`train_test_split`) or the CV estimate (`k_fold`/`loo`) — the same validation method used by HPO to compare models. The CV estimate is the mean over folds for additive metrics and the pooled value over the out-of-fold vector for non-additive metrics (RMSE, MSE, R², nRMSE, RMSLE, D², explained_variance, max error, median absolute error and their aliases). It is **not** a train score. Error-type metrics are returned as positive natural values (sign-corrected). When the pooled value is impossible (a scorer without the sklearn raw metric attributes, or no finite OOF pairs), the engine falls back to the per-fold mean and logs a WARNING once per algorithm and phase (e.g. an HPO phase name or the final fit).
* `additional_scores` — `dict[str, float]` with values of the configured additional metrics computed with the same validation method as `val_score` (empty if none were set).

### Categorical encoding

Categorical columns are encoded by a single shared preprocessor used both during HPO and final training. Five strategies are available via the `general.categorical_encoding` config key (or the `encoding` argument of `tuner.optimize`):

* `one_hot` (default) — each category becomes a binary column via `OneHotEncoder` with `sparse_output=True` (sparse one-hot). Unknown categories on prediction are ignored. The sparse encoder output keeps the preprocessed matrix in CSR storage whenever the total density is below `sparse_threshold` — this prevents the dense `n × n` float blow-up on high-cardinality/ID-like columns.
* `ordinal` — each categorical column becomes a single numeric column via `OrdinalEncoder` (categories ordered alphabetically). Unknown categories map to `-1`, so prediction never fails. Does not expand the number of columns but imposes an artificial ordering on categories.
* `target` — target encoding (`TargetEncodingTransformer`): each category is replaced by the smoothed mean of the target for that category, `(n_cat * mean_cat + m * global_mean) / (n_cat + m)`, where `m = general.target_encoding_smoothing`. All statistics are computed on the training part only, so no information from validation/test data leaks into the encoding — including cross-validation, where the preprocessor is refitted on every fold. To also avoid within-train leakage (a row's own target inflating its encoded feature, critical for high-cardinality columns with few rows per category), `fit_transform` uses cross-fitting: each training row is encoded with statistics computed on a fold that excludes it, while the full-sample statistics are fixed on the encoder for subsequent `transform` on new data. Categories absent from the training data map to `general.target_encoding_fallback` (default: the global target mean). One output column per input column.
* `frequency` — frequency encoding (`FrequencyEncodingTransformer`): each category is replaced by its relative frequency in the training data. Deterministic; unknown categories map to `0.0`. One output column per input column.
* `hashing` — feature hashing (`HashingEncodingTransformer`): each category is deterministically hashed (blake2b, salt derived from the random seed) into `general.hashing_n_components` binary columns per input column. The output dimensionality is fixed and independent of column cardinality — unlike one-hot, high-cardinality columns do not cause feature explosion. Unknown categories are hashed the same way, so prediction never fails. Hash collisions (different categories landing in the same column) are a documented limitation.

#### Sparse output of `hashing` and sparse one-hot

The hashing transformer returns a `scipy.sparse.csr_matrix` (COO→CSR, `nnz = n_samples * n_columns`, values `1.0`) instead of a dense `float64` matrix — exactly like sklearn's standard `FeatureHasher`. The default one-hot is sparse too (`OneHotEncoder(sparse_output=True)`): a unique-per-row ID column no longer expands into a dense `n × n` float matrix per fold (≈3 GB at 20k rows before the fix). Building a dense `(n_rows, n_cols * n_components)` matrix would consume gigabytes of memory on real high-cardinality tables (OOM risk). `SplitCategoricalEncoder` is sparse-aware: parts are joined with `sp.hstack` when any encoder produces a sparse matrix.

Because `ColumnTransformer` uses the default `sparse_threshold=0.3`, its output becomes a `csr_matrix` whenever a sparse part is present and the total density is below the threshold; dense one-hot of a low-cardinality column stays dense (no behavior change). The models `gaussian_process_regression`, `isotonic_regression` and `ardregression` reject sparse input, so for them the preprocessor is automatically built with `force_dense_output=True` (`sparse_threshold=0.0`), returning a dense matrix (flag derived per algorithm by `models.requires_dense_input` in both `ModelTrainer` and `tuner.optimize`). `DataOversampler` handles sparse input as well: `random` oversampling stays sparse, while `smote`/`adasyn` convert the input to dense (they require dense distance matrices by nature).

The dimensionality and the storage format of the preprocessed matrix are logged at INFO level after each final fit: `preprocessed matrix: n × p (sparse/dense)` — the categorical expansion is always visible in the logs.

#### High-cardinality mode

The engine config enables the automatic high-cardinality mode **by default**: `general.high_cardinality_threshold=50` with `general.high_cardinality_encoding="frequency"`. Columns whose number of unique values strictly exceeds the threshold are encoded with `high_cardinality_encoding`, while the remaining categorical columns keep `categorical_encoding`. The split is computed at fit time from the training data only (via `SplitCategoricalEncoder`), so the decision never depends on validation/test observations. A threshold of `0` marks every non-empty column as high-cardinality; a threshold larger than every column's cardinality effectively disables the mode. To disable the automatic mode, set **both** `high_cardinality_threshold` and `high_cardinality_encoding` to `null`/`None` explicitly. The low-level `build_preprocessor`/`SplitCategoricalEncoder` API keeps the previous `None` defaults.

#### Edge cases

* Columns with a single unique category or fully missing columns are handled without failures for all strategies (missing values are imputed with the most frequent category before encoding).
* NaNs in categorical columns are converted to strings and treated as ordinary categories.
* Configurations without the new parameters keep the documented defaults (the automatic high-cardinality mode is on: threshold 50 + `"frequency"`); models saved before the introduction of these strategies keep their previous behavior (backward compatibility).
* `target`/`frequency`/`hashing` encode columns before oversampling (the preprocessor step precedes the sampler step in the pipeline), so SMOTE/ADASYN always receive already-encoded numeric features.

---

## `run_parallel(func, args_seq, ...)`

Executes a function in parallel across multiple workers with shared memory and disk persistence support.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | — | Function to execute. |
| `args_seq` | `Iterable[Sequence]` | `[()]` | Sequence of argument tuples. |
| `kwargs_seq` | `Iterable[Mapping]` | — | Sequence of keyword argument dicts. |
| `max_workers` | `int` or `None` | `None` | Max worker count (defaults to CPU count). |
| `mode` | `str` | `"threads"` | `"threads"` or `"processes"`. |
| `timeout` | `float` or `None` | `None` | Global timeout in seconds. Used only in `"processes"` mode (hard: hung workers are force-terminated). `None` — no limit. In `"threads"` mode it is ignored with a warning — a running Python function cannot be interrupted, so a soft timeout did not save time and only discarded ready results. |
| `task_timeout` | `float` or `None` | `None` | Per-task timeout in seconds. Used only in `"processes"` mode. In `"threads"` mode it is ignored with a warning: the deadline counted from submit made queued tasks "expire" before starting. |
| `shared_args_indices` | `list[int]` or `None` | `None` | Indices of DataFrame args to share via shared memory. Data is passed positionally and the index is NOT transferred (the worker always receives a `RangeIndex(0, n)`); worker code must not align `X`/`y` by index. When the free space of the shared-memory file system is insufficient (in Docker/Kubernetes `/dev/shm` defaults to 64 MB), the frame is automatically routed through the disk cache (Parquet, falling back to the system temp directory) or pickle instead of crashing the process with a Bus error (SIGBUS, exit 135). |
| `disk_args_indices` | `list[int]` or `None` | `None` | Indices of DataFrame args to persist to disk. |

**Returns:** `list[Any]` — results in the same order as inputs; failed tasks return `None`.

---

## `train_model(cfg_or_algo, metric_or_testsize, params_or_metric, ...)`

Legacy facade function for training a single model. Accepts either a config dictionary or positional arguments.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg_or_algo` | `dict` or `str` | Config dict or algorithm name. |
| `metric_or_testsize` | `str` or `float` | Metric name or test size. |
| `params_or_metric` | `dict` or `str` | Hyperparameters or metric name. |
| `X` | `Any` | Feature matrix. |
| `y` | `Any` | Target vector. |
| `enable_logging` | `bool` | Enable result logging. |
| `random_state` | `int` or `None` | Random seed. |
| `log_path` | `str`, `Path`, or `None` | Log file path. Used only when `enable_logging=True`: file logging is initialized via `setup_logging` (one file handler per process — the first path wins), and the training-result message and library logs are written to this file. In the config-dict branch the `log_path` key takes precedence over this argument, except when present with a `None` value (falls back to the argument). |
| `feature_selection_cfg` (keyword-only) | `FeatureSelectionCfg`, `dict`, or `None` | Feature selection config (method, percentile, `min_features`, etc.). In the config-dict branch the `feature_selection_cfg` key takes precedence; a key present with a `None` value is treated as unset and falls back to this argument; `None` means `FeatureSelectionCfg()` (`mode='disabled'`). |
| `feature_selection_active` (keyword-only) | `bool` or `None` | Explicit feature selection activity flag (takes precedence over the config mode). In the config-dict branch the `feature_selection_active` key takes precedence; a key present with a `None` value is treated as unset and falls back to this argument; non-bool values are rejected by `ModelTrainer`; `None` defers the decision to the config. |

**Returns:** `float` — validation metric value computed on hold-out data (or the CV estimate: the fold mean for additive metrics, the pooled OOF value for non-additive ones) with the `ModelTrainer` validation strategy — an honest out-of-sample estimate, not a train score.

---

## `DataOversampler`

Oversampling wrapper compatible with imbalanced-learn pipelines.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multiplier` | `float` | `1.0` | Dataset size multiplier. |
| `algorithm` | `str` | `"random"` | Algorithm (`"random"`, `"smote"`, `"adasyn"`). |
| `add_noise` | `bool` | `False` | Add Gaussian noise to numeric features. |
| `balance` | `bool` | `False` | Balance all classes to majority class size. |
| `random_state` | `int` or `None` | `42` | Random seed. |
| `noise_level` | `float` | `0.01` | Noise intensity relative to std. |
| `task` | `str` | `"regression"` | `"regression"` (default) — bin any numeric target into quantiles and use the regression path; `"classification"` — treat `y` as class labels. |
| `target_bins` | `int` | `8` | Number of quantile bins for the regression path (minimum 2). |

**Key methods:**

* `fit_resample(X, y)` — Resample with bypass of `_check_X_y` for categorical data support.
* `oversample(data, target)` — DataFrame-level oversampling interface.

# Migrating older configs

Removed options are rejected at load time with an error that explains the
replacement; nothing is translated silently.

| Old config | Replace with |
|------------|--------------|
| A pipeline whose last step is a search step (no final `select`) | Append `{action: "select", strategy: "cv_best", output: "winner"}` as the last step — or omit `phases` entirely to get the default pipeline. |
| `action: "refine_winner"` | `select` + `refine`: `all_algorithms → select(strategy: cv_best, output: top_k, k: N) → refine → select(strategy: cv_best, output: winner)`. For a single winner use `all_algorithms → select(strategy: cv_best, output: winner)`. |
| `general.sanity_gate.mode` (`off` / `warn_only` / `active`) | Remove the key. To use the Sanity Gate, add a select step with `strategy: "sanity_gate"`; without such a step the thresholds are not used. The gate report is always produced for those steps. |
| Report keys `disqualified_by_sanity_gate`, `sanity_gate_warn_only`, `sanity_gate_overhead_seconds`, `winner_if_active` | `result["sanity_gate"][<step name>]` (`disqualified`, `overhead_seconds`) and `result["disqualified_algorithms"]`. |
| `sanity_gate.corridor_delta: 0` | A tiny positive value such as `1e-9`, or a `cv_best` select. |
| `sanity_gate.permutation_max_rows: null` / `permutation_max_features: null` | An integer limit, or `check_permutation_sensitivity: false` to skip the shuffle test. |
| `sanity_gate.generalization_gap_mode` not set, expecting strict rejection | Set `generalization_gap_mode: "reject"` (the default is now `"warn"`). |
