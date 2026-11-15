# Changelog

All notable changes to `automl-contracts` are documented here. The format is
based on Keep a Changelog, and the project follows SemVer.

## [Unreleased]

## [0.1.0] - 2026-11-15

### Added

- Initial public contract with configuration models, result DTOs, metric and
  algorithm names, enumerations, exceptions, protocols and the configuration
  JSON Schema.
- `py.typed` marker and the canonical
  `automl_contracts/schemas/config.schema.json` package data.

### Changed

- `CustomMetricSpec` performs structural validation only; callable import,
  signature, direction and name-collision checks run at engine entry points.
- Result models accept plain Python numbers; the engine normalizes numpy
  scalars at its boundary.
- Direct configuration construction performs structural validation only;
  runtime checks are applied by the engine.

### Deprecated

- `ScalingType` remains available as an alias in
  `automl_contracts.preprocessing`; prefer `StringScaling`.
