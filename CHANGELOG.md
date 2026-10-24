# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Fixed

- Orphaned logger name `"training_engine"`: orchestrator logs are now emitted
  through the package hierarchy (`configurable_automl_engine.training_engine.component`),
  so phase starts, winner lines and "Model saved to …" reach
  `general.log_to_file` (previously they were lost or routed to the
  lastResort stderr handler).

  **Compatibility note:** if you filter log records by the logger name
  `"training_engine"`, update the filter to
  `configurable_automl_engine.training_engine.component` — the module now
  uses `logging.getLogger(__name__)` like the rest of the library.

- `setup_logging` no longer constructs a file handler when one is already
  attached (previously every call with a log path leaked an open file
  descriptor). Documented behavior: the first path wins; a WARNING is logged
  when a different path is supplied on a subsequent call.
