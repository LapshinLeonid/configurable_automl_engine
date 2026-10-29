# Changelog

## 0.5.0 — Breaking changes

0.5.0 restructures the internals: the custom tuner/trainer plug-in is gone,
the HPO, trainer and selection steps are pure functions, and the mutable
state is regrouped. Results do not change: the same config, data and seed
give the same winner, parameters, scores and predictions as 0.4.x, in both
`threads` and `processes` mode.

### Config

- **The `algorithms.<name>.tuner` and `algorithms.<name>.trainer_module` keys
  are removed**, and with them the custom tuner/trainer modules. The engine
  always runs its own HPO (`tuner.optimize`) and its own `ModelTrainer`. A
  config that still sets either key is rejected with the error code
  `removed_plugin_module_key`, naming the key's path and telling you to delete
  it. See "Migrating older configs" in [API_REFERENCE.md](API_REFERENCE.md).

### `parallel_mode: "processes"` — the main script must be guarded

- **The worker processes now start with `forkserver`** (Linux, macOS; `spawn`
  on Windows) on every Python version. Before, Linux on Python 3.11–3.13 used
  `fork`. With `forkserver` and `spawn` the workers import the main script
  again, so **every top-level statement of the script runs a second time**. A
  script that calls `train_best_model` at the top level without
  `if __name__ == "__main__":` now fails: the workers start a nested run
  ("Safe importing of main module" in the `multiprocessing` documentation),
  and the HPO phase ends with "No algorithms produced valid scores"; the
  reason names the missing guard. Other top-level side effects repeat too,
  e.g. `logging.basicConfig(filename=..., filemode="w")` truncates the log
  file in the middle of the run. Such scripts worked on Linux up to Python
  3.13.
- **What to do:** put the script's code under
  `if __name__ == "__main__":` (see API_REFERENCE, "A script that uses
  `parallel_mode: \"processes\"`"), or set the new
  `general.start_method: "fork"` (POSIX only; `fork` copies the
  multi-threaded parent and can deadlock a worker). Jupyter notebooks and
  `threads` mode are not affected.
- New config key `general.start_method` (`"forkserver"`, `"spawn"`,
  `"fork"`, default `null` = the engine default). It is a config error with
  `parallel_mode: "threads"` or when the platform does not have the method.
- Patches made in the parent process (e.g. `unittest.mock.patch` in tests) no
  longer reach the workers unless `start_method: "fork"` is set.

### Saved models

- **Artifact format version 2.** `ModelTrainer` now keeps its state in a few
  frozen groups instead of flat attributes, and the artifact metadata records
  `artifact_format_version`. `ModelTrainer.load` refuses an artifact saved by
  engine 0.4.x or older with a `TrainingError` that names the engine and
  format versions: retrain the model with 0.5.0.

### `ModelTrainer`

- The validated constructor settings live in `trainer.settings`
  (`TrainerSettings`, frozen). The results of `fit()` are grouped in the same
  way (`trainer_state`: `PreparedFeatures`, `ValidationScores`,
  `FeatureSelectionResult`, ...).
- The documented attributes are still available, as read-only properties:
  `algorithm`, `hyperparams`, `metric`, `random_state`,
  `serialization_format`, `encoding_strategy`, `additional_metrics`,
  `feature_selection_cfg`, `validation_strategy`, `n_folds`, `test_size`,
  `categorical_features`, `numerical_features`, `feature_names`,
  `dropped_features_`, `val_score`, `additional_scores`, `oof_predictions_`,
  `oof_score_`, `oof_coverage_`, `feature_selection_active_`,
  `selected_features_mask_`. **Assigning to them raises `AttributeError`.**
  Pass the values to the constructor, or build changed settings with
  `dataclasses.replace(trainer.settings, ...)`.
- The other constructor arguments are no longer instance attributes; read
  them from the setting groups: `trainer.settings.encoding`
  (`high_cardinality_threshold`, `high_cardinality_encoding`,
  `hashing_n_components`, `target_encoding_smoothing`,
  `target_encoding_fallback`), `trainer.settings.oversampling` (`enable`,
  `multiplier`, `algorithm`, `target_bins`, `add_noise`, `noise_level` — the
  `data_oversampling*` arguments), `trainer.settings.policies` (`inf_policy`,
  `datetime_policy`, `drop_id_like_columns`) and `trainer.settings.id_column`
  / `preprocessing_override`.
- The constructor arguments and `fit` / `predict` / `save` / `load` are
  unchanged.

### Logging

- **Every module logs under its own name** (`logging.getLogger(__name__)`).
  Records that used to carry `configurable_automl_engine.tuner` now come from
  `configurable_automl_engine.hpo.<module>`; records that carried
  `configurable_automl_engine.training_engine.component` now come from the
  pipeline step module (`data_prep`, `hpo_phases`, `finalists`,
  `select_steps`, `artifact`); `ModelTrainer` helpers log as
  `configurable_automl_engine.trainer_data`, `trainer_scoring` and
  `trainer_feature_selection`. Handlers and levels set on the package logger
  `configurable_automl_engine` (as `setup_logging` does) are not affected;
  filters on the old names must be updated.

### Import paths

- `training_engine.component` exports only `execute_phases` and
  `train_best_model`. The names it re-exported in 0.4.x are imported from
  their own modules:
  - `training_engine.state`: `PhaseResult`, `PhaseContext`, `PreparedData`,
    `FinalistCandidate`, `RobustWinnerResult`;
  - `training_engine.data_prep`: `load_config`, `prepare_dataset`;
  - `training_engine.finalists`: `select_winner`, `select_finalists`,
    `select_robust_winner`, `is_valid_winner_score`,
    `FAMILY_DIVERSITY_MULTIPLIER_DEFAULT`;
  - `training_engine.artifact`: `persist_artifact`.
- The HPO exceptions (`HyperoptError`, `InvalidAlgorithmError`,
  `InvalidDataError`, `HPOHistoryError`) and `TrainingError` report the module
  that defines them (`configurable_automl_engine.hpo.types`,
  `configurable_automl_engine.common.errors`) in `__module__`, tracebacks
  and pickles. Their documented import paths (`configurable_automl_engine.tuner`,
  `configurable_automl_engine.trainer`) keep working.
- The setting groups `EncodingSettings`, `OversamplingSettings` and
  `InputPolicies` live in `configurable_automl_engine.common.settings`.

### Moved modules (internal import paths)

The modules now form import layers (`training_engine` → `hpo`/`tuner`/trainer
→ building blocks → `common`, see "Import layers" in [README.md](README.md)),
so several internal modules and names moved. Code that imported or patched
them must use the new paths; the documented import paths
(`configurable_automl_engine`, `tuner`, `trainer`, `training_engine.component`,
`training_engine.config_parser`, `training_engine.thread_pool.run_parallel`)
keep working.

| Old path | New path |
|---|---|
| `configurable_automl_engine.training_engine.metrics` | `configurable_automl_engine.common.metrics` |
| `configurable_automl_engine.training_engine.logger` | `configurable_automl_engine.common.log_setup` |
| `configurable_automl_engine.training_engine.thread_pool`: `SharedDataFrame`, `_numpy_result_dtype`, `_ints_fit_float64`, `_shm_backing_path`, `_shm_has_free_space` | `configurable_automl_engine.common.shared_frame` (`run_parallel` and `DiskPersistenceManager` stay in `thread_pool`) |
| `configurable_automl_engine.common.hyperopt_defaults` | `configurable_automl_engine.hpo.default_spaces` |
| `configurable_automl_engine.common.complexity_guard` | `configurable_automl_engine.hpo.complexity_guard` |
| `configurable_automl_engine.common.target_domain` | `configurable_automl_engine.hpo.target_domain` |
| `configurable_automl_engine.common.artifact_metadata` | `configurable_automl_engine.trainer_artifact` |
| `configurable_automl_engine.trainer_errors` | `configurable_automl_engine.common.errors` (`TrainingError`; still importable from `configurable_automl_engine.trainer`) |
| `configurable_automl_engine.trainer.IsotonicDataTransformer` | `configurable_automl_engine.isotonic.IsotonicDataTransformer` (still importable from `trainer`) |
| `configurable_automl_engine.training_engine.config_parser`: `FeatureSelectionCfg`, `FeatureSelectionMode`, `FeatureSelectionMethod` | `configurable_automl_engine.feature_selection` (still importable from `config_parser`) |
| `configurable_automl_engine.preprocessing.EncodingStrategy` | `configurable_automl_engine.common.definitions.EncodingStrategy` (still importable from `preprocessing`) |

- `configurable_automl_engine.common` is a regular package now (it has an
  `__init__.py`, which imports nothing).
- Saved models reference `IsotonicDataTransformer` and the
  `FeatureSelection*` config types by their new module paths. The artifact
  format version stays 2 (0.5.0 is not released yet); the old paths are
  still importable, so artifacts saved by earlier 0.5.0 development builds
  load as before.
- The own records of `setup_logging` (e.g. the first-path-wins warning) log
  as `configurable_automl_engine.common.log_setup` instead of
  `configurable_automl_engine.training_engine.logger`.

### Internal APIs (for code that patched or called private functions)

- One HPO run is described by an `HpoRequest` (`hpo.request`) and run by
  `hpo.runner.run_optimization(X, y, request)`; `tuner.optimize()` builds the
  request from its unchanged keyword arguments. The training engine calls
  `run_optimization` directly: `hpo_phases._run_hpo(*, X, y, request)`. The
  plug-in loaders (`_load_tuner`, `_load_module`, `_call_tuner`,
  `_supported_kwargs`, `_prepare_search_space`, ...) are removed.
- The HPO setup steps are pure functions that return frozen plans
  (`hpo.plans`: `ValidationPlan`, `PreprocessingPlan`, `SearchSpacePlan`,
  `PruningPlan`, `BestTrial`); `_OptimizeRunner(X, y, request)` keeps them.
  Trials are scored by `hpo.evaluation.score_trial`.
- `ModelTrainer.fit` runs the pure steps of `trainer_data`,
  `trainer_scoring` and `trainer_feature_selection` and assigns their
  results.
- The pipeline steps return outcomes that only `component` applies to the
  `PhaseContext`: the select steps return a `SelectOutcome`
  (`select_steps.select_step_outcome` replaces `_execute_select_phase`),
  `hpo_phases._consume_worker_results(results)` returns a `WorkerOutcome`, and
  `hpo_phases._apply_target_domain_guard(algorithms, y)` returns a
  `TargetDomainOutcome`.
- The last six functions above the complexity limit
  (`DataOversampler._fit_resample`, `validate_features`,
  `ModelSanityGate.check` and `_permutation_impl`, `iter_splits`,
  `create_model`) are split into small private steps; ruff `C901` ≤ 15 now
  holds for the whole package with no `# noqa: C901` exception. Behavior is
  unchanged: return values, random draws, log messages and exceptions are
  pinned by `tests/test_characterization_complex_functions.py`.

### `run_parallel` and the `processes` mode

`run_parallel` was split into small parts (`training_engine.parallel_*`
modules: payload transport, deadline policy, process-pool supervisor,
executor adapter, `TaskOutcome`). Its signature and its return value are
unchanged except for the new optional `mp_context` argument; the results of
healthy runs do not change.

**Fixed**

- **One crashed worker no longer loses the whole batch.** When a worker
  process dies (OOM killer, a segfault in a C extension, `os._exit`), only
  its task gets `None`: the pool is restarted and the other tasks are
  resubmitted. Before, every task of the batch (every algorithm of the HPO
  phase) was dropped. A task that crashes its worker twice is given up on, so
  the run always terminates.
- **`run_parallel(func, args_seq=[])` returns `[]`** and never calls `func`.
  Before, it called `func` once and returned one result for zero tasks.
  `args_seq=None` still means one call without arguments.
- **No unbounded hang after a failed pool restart.** If the replacement pool
  could not be created after a hard timeout, every task was rerun in threads
  without a timeout — the hung task included — so the run could hang forever
  (and the cleanup then failed with `AttributeError`). Now the unfinished
  tasks fail with the reason and the finished results are kept.

**Changed**

- **Fallback policy.** `processes` mode falls back to threads only if the
  process pool cannot be created or the payload transport cannot be set up,
  before any task is submitted, and logs a `WARNING` (it was an `ERROR`) that
  the timeouts are dropped. After tasks were submitted nothing is rerun in
  threads: an infrastructure failure fails the unfinished tasks, and a
  programming error (`TypeError`, `AttributeError`, ...) propagates instead
  of being hidden behind a fallback log line.
- **Worker logs reach the parent's log.** The log records of the worker
  processes are forwarded to the parent process (a `QueueHandler` installed
  by the pool initializer, a `QueueListener` in the parent) and handled by
  the parent's handlers with every start method, so `general.log_to_file`
  keeps the HPO records of the workers (phase results, complexity-guard
  records) and their warnings keep the configured format.
- **Explicit start method.** New `mp_context` argument (`"forkserver"`,
  `"spawn"`, `"fork"` or a context). The default is now `"forkserver"`
  where available (Linux, macOS) and `"spawn"` elsewhere, on every Python
  version — before, it was whatever the interpreter defaulted to (`fork` on
  Linux up to 3.13, `forkserver` on 3.14). Workers therefore import the task
  function's module, and patches made in the parent process do not reach
  them; pass `mp_context="fork"` (or set `general.start_method: "fork"`) to
  get the old Linux behavior. The fork server
  preloads numpy/pandas and the task function's module, so the workers start
  without importing them again.
- **Disqualification reasons for lost algorithms.** An algorithm whose HPO
  worker timed out, crashed or failed in `processes` mode is now listed in
  `disqualified_algorithms` with the reason (`"timed out after 600 s"`,
  `"worker process crashed (exit code -9)"`, `"failed: ValueError: ..."`)
  instead of silently missing from the phase. Like every disqualified
  algorithm, it does not run in the following phases.
- An unknown `mode` raises `ValueError` (before, it silently ran threads).
- Workers are stopped with SIGTERM right away (SIGKILL after
  `shutdown_grace_period`) instead of first waiting the grace period for a
  worker that is busy with a hung task; on Python 3.14+ through the public
  `ProcessPoolExecutor.terminate_workers()` / `kill_workers()`. A healthy run
  shuts the pool down gracefully.
- Internal: `_force_shutdown_processes`, `_terminate_worker_by_pid` and
  `_pid_exists` are removed; `_StartSignal` and `_worker_proxy` live in
  `training_engine.parallel_worker`, `DiskPersistenceManager` in
  `training_engine.parallel_transport` (still importable from
  `thread_pool`).

### Direct `optimize()` (#50)

- A partial `space_overrides` dict is merged with the algorithm's default
  space (the parameters it does not name keep their default ranges) instead
  of being searched alone; `limit_hyperparameters=True` searches only the
  named parameters.
- Every algorithm has a default space, so `optimize(algo, X, y)` without
  overrides works for all of them. kNN without overrides searches
  `n_neighbors` in 1–50, clipped to `n_samples − 1`.
- The range clipping and the adaptive complexity guard now apply to direct
  `optimize()` calls as well, exactly as in the training engine.
