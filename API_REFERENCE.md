# Configuration File Structure

The system uses a typed YAML or JSON config based on Pydantic.

Scheme: [config.schema.json](config.schema.json).

Some Pydantic validation rules cannot be described in the schema. See [config_parser.py](/src/configurable_automl_engine/training_engine/config_parser.py) for details.

The file must contain two sections: "general" and "algorithms". Additionally, it may include an optional "oversampling" section.

## General section

The `general` section may include the following attributes:

* `phases` — required section (array) of hyperparameter optimization phases.
* `comparison_metric` — (optional) accuracy metric for model comparison. Defaults to `"r2"` if not specified.
* `additional_metrics` — (optional) list of additional quality metrics computed for the final trained model and returned together with the training results (see [Additional metrics](#additional-metrics-additional_metrics)).
* `path_to_model` — (optional) path to save the best model.
* `serialization_format` — (optional) format for saving the model (`"pickle"` or `"joblib"`).
* `log_to_file` — (optional) path to the log file.
* `validation_strategy` — (optional) strategy for evaluating model accuracy (`"train_test_split"`, `"k_fold"`, `"loo"`, `"auto"`). With `"auto"` the engine automatically picks between LOO, k-fold and train-test split based on the number of observations and features.
* `n_folds` — (optional) number of folds for cross-validation; used only if `validation_strategy = "k_fold"`. Ignored when `validation_strategy = "auto"`.
* `max_workers` — (optional) maximum number of threads/processes. If not specified, the number of CPU cores is used.
* `parallel_mode` — (optional) parallelism mode (`"threads"` or `"processes"`). Defaults to `"threads"`.
* `parallel_strategy` — (optional) controls parallel execution strategy. Defaults to `"algorithms"`.
* `phase_timeout` — (optional) global timeout for the entire HPO phase in seconds (minimum 1.0). If `null`, defaults to 3600.
* `task_timeout` — (optional) per-task timeout in seconds (minimum 1.0). If `null`, uses `phase_timeout`.
* `pruning` — (optional) early-stopping (pruning) settings for Optuna trials. Disabled by default.
* `categorical_encoding` — (optional) categorical encoding strategy: `"one_hot"` (default), `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`. See [Categorical encoding](#categorical-encoding).
* `high_cardinality_threshold` — (optional, `>= 0`) cardinality threshold for the automatic high-cardinality mode. Must be set together with `high_cardinality_encoding`.
* `high_cardinality_encoding` — (optional) encoding strategy for high-cardinality columns (`"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`). Must be set together with `high_cardinality_threshold`.
* `hashing_n_components` — (optional, `>= 1`, default `16`) number of binary columns per categorical column when `"hashing"` encoding is used.
* `target_encoding_smoothing` — (optional, `>= 0`, default `20.0`) smoothing parameter `m` of target encoding.
* `target_encoding_fallback` — (optional, default `null`) fallback value of target encoding for categories unseen in training; `null` means the global target mean.

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
```

Validation rules (rejected at config load with a clear message):

* Unknown `strategy` (anything other than `"median"` / `"hyperband"`).
* Non-positive `min_steps`, `n_startup_trials`, or `reduction_factor < 2`.
* Conflict with validation settings: `pruning.enable: true` + `validation_strategy: "k_fold"` + `min_steps > n_folds` (the pruner would never get enough intermediate steps).

Behavior by validation strategy:

* `k_fold` / `loo` — pruning works on natural per-fold (per-sample) steps.
* `auto` — pruning is applied when `auto` resolves to `k_fold` or `loo`; if it resolves to `train_test_split`, pruning is not applied (see below).
* `train_test_split` (hold-out) — there are no intermediate steps, so the pruner is **not** applied; the configuration is accepted with a warning, and every trial runs to completion. This behavior is safe and documented.

### Structure of an optimization phase (`phases`)

* `n_trials` — number of iterations within this phase.
* `name` — (optional) user-defined name of the optimization phase.
* `action` — (optional) action for the phase, default is `"all_algorithms"`.

### Allowed actions for an optimization phase

* `"all_algorithms"` — for each algorithm, performs `n_trials` hyperparameter optimization attempts. The best algorithm is passed to the next phase.
* `"refine_winner"` — performs `n_trials` hyperparameter optimization attempts for the best algorithm from the previous phase.

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
* **Computation.** Each additional metric is evaluated for the final model with the **same validation method as the comparison metric** (issue #24): on a hold-out split (`train_test_split`) or as the mean over CV folds (`k_fold`/`loo`) — never on the full training set. Error-type metrics (RMSE, MAE, MSE, NRMSE) are returned as positive natural values, score-type metrics (R²) as-is. Non-finite values (e.g., `inf` from NRMSE on a constant target) are returned as-is and do not affect training completion or artifact saving.
* **Failure tolerance.** If a single additional metric cannot be computed (scorer error), it is skipped with a warning and does not abort training, model selection, or artifact saving.

Example:

```yaml
general:
  comparison_metric: "rmse"
  additional_metrics: ["r2", "mae", "mse"]
  phases:
    - name: "search"
      n_trials: 50
      action: "all_algorithms"
```

Result format (`train_best_model` return value):

```python
{
    "algorithm": "ridge",
    "score": 0.123,              # main comparison metric in natural (user-facing) semantics
    "metric": "rmse",            # user-facing name of the comparison metric
    "params": {...},
    "model_path": "model.pkl",
    "additional_metrics": {"r2": 0.98, "mae": 0.07, "mse": 0.009},  # only when non-empty
}
```

Metric semantics (issue #26): `score` always reflects the natural value of the
metric as the user knows it — positive RMSE/MAE/MSE/NRMSE, plain R². The
internal sklearn negative scorers (`neg_root_mean_squared_error` and other
`neg_*`) are an optimizer detail: their inverted values never leak into
`result["score"]`. `metric` carries the user-facing name of the comparison
metric from the configuration (`general.comparison_metric`).

The `additional_metrics` key is a dict mapping each configured metric name to its computed value for the final model (same natural semantics as `score`).

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

Algorithm configuration consists of:

* `enable` — boolean flag, whether hyperparameter search is performed for the algorithm.
* `limit_hyperparameters` — (optional) boolean flag to set limits for hyperparameter search.
* `hyperparameters` — (optional) hyperparameter value constraints, unique to each algorithm. Keys must match hyperparameters accepted by the estimator constructor (including legacy names from `LEGACY_PARAM_MAPPINGS`, e.g. `n_iter` → `max_iter` for ARDRegression). The allowed set is derived from the estimator signature via [`get_allowed_hyperparameters`](src/configurable_automl_engine/models.py), not from the default search space — so parameters outside `DEFAULT_SPACES` (e.g. `min_samples_split`, `max_features`) are valid as long as the estimator accepts them. Unknown names (typos) are rejected at configuration validation.
* `preprocessing` — (optional) explicit override of the feature preprocessing preset for this algorithm. Takes priority over the automatic class-based selection (FR-5). May override the whole preset or only some fields:

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

The engine automatically chooses the feature preprocessing strategy based on the selected regression algorithm (FR-1). A **preprocessing preset** declaratively describes the numeric-feature handling rules: the missing-value imputation strategy and whether/how scaling is applied. Each supported algorithm is assigned to an **algorithm class** with common preprocessing requirements; membership is fixed in the default mapping table (see [`preprocessing_presets.py`](src/configurable_automl_engine/preprocessing_presets.py)) — adding a new algorithm only requires registering it there (FR-6).

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
* The same preset is applied at every lifecycle stage — hyperparameter optimization (HPO), final training, and prediction — through the single resolution point used by both `tuner.optimize` and `ModelTrainer` (FR-4).
* Explicit user override (the `preprocessing` block in the algorithm config, or the `preprocessing_override` argument of `ModelTrainer`/`tuner.optimize`) takes priority over the automatic selection (FR-5). Fields left unset keep the automatically chosen values.
* The resolved preset is logged (INFO level) and stored on the trainer as `preprocessing_preset`, so the applied preprocessing can always be determined from logs (AC-9).

## Oversampling section

The optional `oversampling` section may include:

* `enable` — (optional) enable oversampling.
* `multiplier` — (optional) factor to increase dataset size (minimum 1.0).
* `algorithm` — (optional) oversampling algorithm.

### Supported oversampling algorithms

* `"random"`
* `"smote"`
* `"adasyn"`


# API Reference

## `train_best_model(config, df, target, model_path_override)`

The main entry point for the AutoML pipeline. Validates data, runs multi-phase hyperparameter optimization, selects the best algorithm, and saves the final model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str`, `Path`, `Config`, or `dict` | Configuration — path to YAML/JSON file, `Config` object, or dictionary. |
| `df` | `pd.DataFrame` | Input dataset. |
| `target` | `str` or `None` | Name of the target column. Defaults to `"target"`. |
| `model_path_override` | `str`, `Path`, or `None` | Alternative path to save the model. |

**Returns:** `dict[str, Any]` with keys `"algorithm"`, `"score"`, `"metric"`, `"params"`, `"model_path"`. `"score"` is the comparison metric in natural user-facing semantics (positive RMSE/MAE/MSE/NRMSE, plain R² — internal `neg_*` sklearn scorers are inverted back), `"metric"` is its user-facing name from `general.comparison_metric`. When `general.additional_metrics` is configured (non-empty after deduplication and exclusion of the comparison metric), an additional key `"additional_metrics"` maps each extra metric name to its value computed for the final model (same natural semantics).

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
| `data_oversampling_algorithm` | `str` | `"random"` | Oversampling algorithm. |
| `metric` | `str` | `"r2"` | Metric to maximize. |
| `val_method` | `ValidationStrategy` or `str` | `"k_fold"` | Validation method (legacy parameter). |
| `validation_strategy` | `ValidationStrategy` or `str` | `"k_fold"` | Validation method (`"train_test_split"`, `"k_fold"`, `"loo"`, `"auto"`). |
| `train_test_split_test_size` | `float` | `0.2` | Test set size when using `train_test_split` validation. Ignored when using `"auto"`. |
| `n_folds` | `int` | `5` | Number of CV folds. |
| `n_trials` | `int` | `50` | Number of Optuna trials. |
| `random_state` | `int` or `None` | `42` | Random seed. |
| `space_overrides` | `dict[str, Callable]` or `None` | `None` | Custom search space overrides. |
| `preprocessing_override` | `dict` or `PreprocessingOverride` or `None` | `None` | Explicit override of the preprocessing preset (FR-5). Partial (`{"scaling": "none"}`) or full overrides are supported; takes priority over automatic class-based selection. |
| `pruning` | `dict[str, Any]` or `None` | `None` | Early-stopping settings: `{"enable": bool, "strategy": "median"\|"hyperband", "min_steps": int, "n_startup_trials": int, "reduction_factor": int}`. When `None` or `enable: false`, trials run to completion (identical to previous behavior). |
| `feature_selection_cfg` | `FeatureSelectionCfg`, `dict`, or `None` | `None` | Feature selection config (method, percentile, `min_features`, etc.). `None` means `FeatureSelectionCfg()` (`mode='disabled'`). In `'always'` mode every trial pipeline (and the final model) contains the `feature_selector` step; in `'disabled'` it never does; in `'auto'` Optuna decides per trial via the `use_feature_selection` categorical (recorded in `best_params`). For `isotonic_regression` the flag is forcibly turned off. |

**Returns:** `tuple[Any | None, dict[str, Any] | None, float | None]` — `(best_model, best_params, best_score)`. When **no trial succeeds** (e.g., every trial is pruned via a non-fatal error) the function returns `(None, None, None)` as an explicit failure signal — the orchestrator (`train_best_model`) then excludes the algorithm from candidates instead of treating the `HPO_WORST_SCORE` sentinel (`-3.4028235e38`) as a valid result (issue #13). The sentinel itself is returned as `best_score` only when trials complete but all produce a non-finite metric.

In `'auto'` mode `best_params` additionally contains the service key `use_feature_selection` (`True`/`False`) — the winner's decision about feature selection, which is preserved between multi-phase HPO runs (passed to the next phase via `initial_params`/`enqueue_trial`) and consumed by `train_best_model` for the final fit. The key is absent in `'always'`/`'disabled'` modes and for `isotonic_regression` (feature selection is forcibly turned off there). The key is never passed to the base model constructor.

**Raises:** `ValueError` for invalid `n_trials`; `HyperoptError` for missing search spaces; `InvalidAlgorithmError` after 5 consecutive fatal failures.

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

**Notes:** Automatically cleans and remaps hyperparameters via `clean_hyperparameters()`. Sets `max_iter=10000` for SVR. Sets `kernel=RBF(1.0)` for GaussianProcessRegressor by default.

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
| `data_oversampling_algorithm` | `str` | `"random"` | Oversampling algorithm. |
| `serialization_format` | `SerializationFormat` | `pickle` | Save format. |
| `categorical_features` | `list[str]` or `None` | `None` | Categorical column names. |
| `numerical_features` | `list[str]` or `None` | `None` | Numerical column names. |
| `id_column` | `str` or `None` | `None` | ID column to exclude. |
| `encoding_strategy` | `str` | `"one_hot"` | Categorical encoding strategy: `"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`. |
| `additional_metrics` | `Iterable[str]` or `None` | `None` | Additional (informational) metrics computed for the trained model with the same validation method as the main metric (hold-out split or CV-fold mean, never a train score). Any iterable of strings is accepted (list, tuple, set, generator); names are normalized to lowercase. Stored in `additional_scores` after `fit()`. |
| `validation_strategy` | `str` or `ValidationStrategy` | `"train_test_split"` | Validation strategy for the final model quality estimate: `"train_test_split"` (one hold-out split), `"k_fold"` (CV fold mean), `"loo"` or `"auto"`. The engine path always passes the already-resolved strategy (D3, issue #24). |
| `n_folds` | `int` | `5` | Number of folds for `"k_fold"` (`>= 2`). |
| `test_size` | `float` or `int` | `0.2` | Hold-out size: a fraction in `(0, 1)` or an integer number of rows (like sklearn `train_test_split`). |
| `preprocessing_override` | `dict` or `PreprocessingOverride` or `None` | `None` | Explicit override of the preprocessing preset (FR-5). Partial or full overrides are supported; takes priority over the automatic class-based selection. |
| `high_cardinality_threshold` | `int` or `None` | `None` | Cardinality threshold for the automatic high-cardinality mode (`>= 0`). Columns with more unique values than the threshold are encoded with `high_cardinality_encoding`; the rest use `encoding_strategy`. Must be set together with `high_cardinality_encoding`. `None` disables the mode. |
| `high_cardinality_encoding` | `str` or `None` | `None` | Encoding strategy for high-cardinality columns (`"one_hot"`, `"ordinal"`, `"target"`, `"frequency"` or `"hashing"`). Must be set together with `high_cardinality_threshold`. |
| `hashing_n_components` | `int` | `16` | Number of binary columns per categorical column when `"hashing"` encoding is used (`>= 1`). Controls output dimensionality regardless of column cardinality. |
| `target_encoding_smoothing` | `float` | `20.0` | Smoothing parameter `m` of target encoding (`>= 0`). With `m=0` the raw per-category mean is used; larger values pull rare categories toward the global mean. |
| `target_encoding_fallback` | `float` or `None` | `None` | Fallback value for categories unseen in training under target encoding. `None` means the global target mean. |

**Key attributes:**

* `preprocessing_preset` — the resolved `PreprocessingPreset` (algorithm class, imputation strategy, scaling type) applied during `fit()`. Useful for observability and serialization.

**Key methods:**

* `fit(X, y)` — Train the model with full preprocessing pipeline.
* `predict(X)` — Make predictions on new data.
* `save(path)` — Serialize to disk.
* `load(path)` — Load from disk (class method).

**Attributes after `fit()`:**

* `val_score` — value of the main metric computed on data the model has **not** been trained on: a hold-out split score (`train_test_split`) or the mean over CV folds (`k_fold`/`loo`) — the same validation method used by HPO to compare models (issue #24). It is **not** a train score. Error-type metrics are returned as positive natural values (sign-corrected).
* `additional_scores` — `dict[str, float]` with values of the configured additional metrics computed with the same validation method as `val_score` (empty if none were set).

**Categorical encoding**

Categorical columns are encoded by a single shared preprocessor used both during HPO and final training. Five strategies are available via the `general.categorical_encoding` config key (or the `encoding` argument of `tuner.optimize`):

* `one_hot` (default) — each category becomes a binary column via `OneHotEncoder`. Unknown categories on prediction are ignored.
* `ordinal` — each categorical column becomes a single numeric column via `OrdinalEncoder` (categories ordered alphabetically). Unknown categories map to `-1`, so prediction never fails. Does not expand the number of columns but imposes an artificial ordering on categories.
* `target` — target encoding (`TargetEncodingTransformer`): each category is replaced by the smoothed mean of the target for that category, `(n_cat * mean_cat + m * global_mean) / (n_cat + m)`, where `m = general.target_encoding_smoothing`. All statistics are computed on the training part only (inside `fit`), so no information from validation/test data leaks into the encoding — including cross-validation, where the preprocessor is refitted on every fold. Categories absent from the training data map to `general.target_encoding_fallback` (default: the global target mean). One output column per input column.
* `frequency` — frequency encoding (`FrequencyEncodingTransformer`): each category is replaced by its relative frequency in the training data. Deterministic; unknown categories map to `0.0`. One output column per input column.
* `hashing` — feature hashing (`HashingEncodingTransformer`): each category is deterministically hashed (blake2b, salt derived from the random seed) into `general.hashing_n_components` binary columns per input column. The output dimensionality is fixed and independent of column cardinality — unlike one-hot, high-cardinality columns do not cause feature explosion. Unknown categories are hashed the same way, so prediction never fails. Hash collisions (different categories landing in the same column) are a documented limitation.

**Sparse output of `hashing`**

The hashing transformer returns a `scipy.sparse.csr_matrix` (COO→CSR, `nnz = n_samples * n_columns`, values `1.0`) instead of a dense `float64` matrix — exactly like sklearn's standard `FeatureHasher`. Building a dense `(n_rows, n_cols * n_components)` matrix would consume gigabytes of memory on real high-cardinality tables (OOM risk). `SplitCategoricalEncoder` is sparse-aware: parts are joined with `sp.hstack` when any encoder produces a sparse matrix.

Because `ColumnTransformer` uses the default `sparse_threshold=0.3`, its output becomes a `csr_matrix` whenever a sparse part is present. The models `gaussian_process_regression`, `isotonic_regression` and `ardregression` reject sparse input, so for them the preprocessor is automatically built with `force_dense_output=True` (`sparse_threshold=0.0`), returning a dense matrix (flag derived per algorithm by `models.requires_dense_input` in both `ModelTrainer` and `tuner.optimize`). `DataOversampler` handles sparse input as well: `random` oversampling stays sparse, while `smote`/`adasyn` convert the input to dense (they require dense distance matrices by nature).

**High-cardinality mode**

When `general.high_cardinality_threshold` and `general.high_cardinality_encoding` are set together, columns whose number of unique values strictly exceeds the threshold are encoded with `high_cardinality_encoding`, while the remaining categorical columns keep `categorical_encoding`. The split is computed at fit time from the training data only (via `SplitCategoricalEncoder`), so the decision never depends on validation/test observations. A threshold of `0` marks every non-empty column as high-cardinality; a threshold larger than every column's cardinality effectively disables the mode.

**Edge cases**

* Columns with a single unique category or fully missing columns are handled without failures for all strategies (missing values are imputed with the most frequent category before encoding).
* NaNs in categorical columns are converted to strings and treated as ordinary categories.
* Configurations without the new parameters and models saved before the introduction of these strategies keep their previous behavior (backward compatibility).
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
| `timeout` | `float` or `None` | `3600` | Global timeout in seconds. |
| `task_timeout` | `float` or `None` | `None` | Per-task timeout in seconds. |
| `shared_args_indices` | `list[int]` or `None` | `None` | Indices of DataFrame args to share via shared memory. |
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
| `log_path` | `str`, `Path`, or `None` | Log file path. |
| `feature_selection_cfg` (keyword-only) | `FeatureSelectionCfg`, `dict`, or `None` | Feature selection config (method, percentile, `min_features`, etc.). In the config-dict branch the `feature_selection_cfg` key takes precedence; a key present with a `None` value is treated as unset and falls back to this argument; `None` means `FeatureSelectionCfg()` (`mode='disabled'`). |
| `feature_selection_active` (keyword-only) | `bool` or `None` | Explicit feature selection activity flag (takes precedence over the config mode). In the config-dict branch the `feature_selection_active` key takes precedence; a key present with a `None` value is treated as unset and falls back to this argument; non-bool values are rejected by `ModelTrainer`; `None` defers the decision to the config. |

**Returns:** `float` — validation metric value computed on hold-out data (or CV-fold mean) with the `ModelTrainer` validation strategy — an honest out-of-sample estimate, not a train score (issue #24).

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

**Key methods:**

* `fit_resample(X, y)` — Resample with bypass of `_check_X_y` for categorical data support.
* `oversample(data, target)` — DataFrame-level oversampling interface.

---