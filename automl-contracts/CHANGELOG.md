# Changelog

All notable changes to `automl-contracts` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The contract is versioned independently of `configurable-automl-engine`.

## [Unreleased]

## [0.1.4] - 2026-11-19

### Changed

- The contract is published to public PyPI. Installation no longer requires a
  private registry, a personal access token or the `--no-deps` workaround, and
  the engine resolves the declared contract range automatically.

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
