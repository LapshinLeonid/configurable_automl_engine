# Changelog

All notable changes to `configurable-automl-contracts` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The contract is versioned independently of `configurable-automl-engine`.

## [0.2.0] - 2026-11-24

Compatibility: MINOR (0.x MINOR). The two new `general` fields are optional
and have safe defaults, so the JSON Schema only gains optional properties; the
distribution rename is a breaking packaging change for consumers pinned to the
old name. The engine adopts the new fields and therefore raises its minimum
contract version to `0.2.0`.

### Added

- **MINOR (compatible API):** `general.bbc_n_bootstrap` (default 500) and
  `general.bbc_time_budget_seconds` (default 60, `0` disables the limit)
  configure the BBC-CV bootstrap resample count and its wall-clock budget.

### Changed

- **BREAKING (0.x MINOR):** the distribution is renamed from
  `automl-contracts` to `configurable-automl-contracts`. Migration: replace the
  old distribution name with the new one in requirements and lock files; the
  import package remains `automl_contracts`.

## [0.1.0] - 2026-11-15

### Added

- Initial public contract with configuration models, result DTOs, metric and
  algorithm names, enumerations, exceptions, protocols and the configuration
  JSON Schema.
- `py.typed` marker and the canonical
  `automl_contracts/schemas/config.schema.json` package data.
- Public structural protocols `MetricFunction` and `ScorerCallable`.
- Stable error codes `ERR_CUSTOM_NAME_INVALID`, `ERR_REMOVED_PLUGIN_KEY`,
  `ERR_REMOVED_REFINE_WINNER` and `ERR_REMOVED_SANITY_GATE_MODE`.

### Changed

- **BREAKING (0.x MINOR):** `CustomMetricSpec` performs structural validation
  only. Callable import, signature, direction and name-collision checks run at
  engine entry points instead of in the model constructor. Constructing the
  model with an invalid callable now succeeds; `train_best_model`,
  `AutoMLExperiment`, `optimize` and `ModelTrainer` still reject it before
  training.
- **BREAKING (0.x MINOR):** direct construction of `Config`, `GeneralCfg` and
  the re-exported engine paths performs structural validation only. Algorithm
  dependency checks, hyperparameter compatibility, custom-metric runtime
  checks, `check_start_method`, `check_threads_timeouts` and the joblib part of
  `check_n_folds` moved to the engine's `validate_runtime_config` and its
  public entry points.
- **BREAKING (0.x MINOR):** runtime validation errors raised by the engine's
  `read_config`, `load_config`, `train_best_model` and `AutoMLExperiment` are
  `ValueError` (and `pydantic_core.PydanticCustomError` for custom metrics)
  instead of `pydantic.ValidationError`. `ValidationError` remains for
  structural validation of contract models.
- **BREAKING (0.x MINOR):** reserved custom-metric names are no longer rejected
  by the direct `Config`/`GeneralCfg` constructor. The contract checks only the
  name format (`ERR_CUSTOM_NAME_INVALID`); the engine checks collisions with
  built-in and scikit-learn names (`ERR_CUSTOM_NAME_COLLISION`).
- **BREAKING (0.x MINOR):** `CustomMetricSpec.natural_greater_is_better` is not
  part of the contract. The engine exposes `natural_greater_is_better(spec)` in
  `configurable_automl_engine.common.custom_metrics`; code reading the property
  must switch to that function.
- **BREAKING (0.x MINOR):** `AlgoCfg.get_required_package` and
  `AlgoCfg.get_unknown_hyperparameters` moved out of the model; the engine
  provides the corresponding runtime helpers.
- Result models accept plain Python numbers; the engine normalizes NumPy
  scalars at its boundary.
- `ExperimentResult` resolves built-in metric directions from the static
  `BUILTIN_METRIC_DIRECTIONS` table and takes custom-metric directions from the
  payload `metric_direction`; it no longer resolves sklearn metrics through a
  process-local registry. The engine keeps the runtime direction and
  name-collision checks for arbitrary sklearn names.
- `WORST_SCORE_THRESHOLD` is defined as the float32 minimum literal and no
  longer requires NumPy.

### Deprecated

- `ScalingType` remains available as an alias in
  `automl_contracts.preprocessing`; prefer `StringScaling` or the concrete
  scaling models.

### Migration

- Import contract symbols from `automl_contracts` or its stable submodules.
  The old `configurable_automl_engine` paths remain as re-exports for the
  `0.x` contract line. See the migration table in the package README and in
  `API_REFERENCE.md`.
- Regenerate `config.schema.json` after any model change. The canonical file is
  `automl_contracts/schemas/config.schema.json`; the root `config.schema.json`
  is a generated compatibility mirror.
