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
