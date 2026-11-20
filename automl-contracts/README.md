# automl-contracts

`automl-contracts` is the standalone public data contract of the
[Configurable AutoML Engine](https://github.com/LapshinLeonid/configurable_automl_engine).
It describes the configuration format, the result DTOs, the enumerations and
value sets, the public exceptions, the constants, the structural protocols and
the JSON Schema that external consumers may rely on.

The distribution is intentionally small: it depends only on the Python
standard library, Pydantic v2 and `pydantic-core`. It does not install the
engine, scikit-learn, pandas, NumPy, Optuna, PyYAML or any other runtime
dependency of the engine.

The package is useful when you need to:

- validate or generate an engine configuration without training a model;
- build a UI, a configuration editor or a CI validator on top of the schema;
- type-check custom metrics and the objects exchanged with the engine;
- inspect result DTOs without importing the training stack;
- share one definition of the contract between several services.

## What is and is not a contract

The engine separates three levels of responsibility:

| Level | What belongs there | Where it lives |
|---|---|---|
| Public library contract | Configuration models and value sets, result DTOs, statuses, public exceptions, stable error codes, public constants, `Protocol` interfaces and the JSON Schema | `automl-contracts` |
| Configuration contract | The structure of a configuration, field types, defaults, required fields and structural cross-field rules | `automl-contracts` |
| Engine implementation | Training, preprocessing, HPO, winner selection, artifacts, reports, experiment storage, explainability, parallel execution, installed-package and environment checks | `configurable-automl-engine` |

The rule for moving a definition into this package is deliberately narrow:
**only definitions and pure Pydantic rules move**. Anything that needs
scikit-learn, pandas, NumPy, Optuna, PyYAML, joblib, the installed-package
registry or environment inspection stays in the engine.

In particular, the following are not part of this package: `ModelTrainer`,
`train_model`, `optimize`, `create_model`, `FeatureSelector`,
`DataOversampler`, `build_preprocessor`, `DEFAULT_SPACES`, `read_config`, HPO
runners, reporters, the experiment store, export helpers and explainability
helpers. They remain public engine APIs with their own compatibility rules.

Borderline DTOs (`RunSummary`, `LeaderboardEntry`, `FeatureImportance`,
`EnsembleMember`, `ReporterArtifact`) are engine-only while the contract is in
the `0.x` series. Adding any of them later is an additive public API change and
must be reflected in `CHANGELOG.md` and in the compatibility matrix.

The inclusion rules, the compatibility matrix and the maintainer process are
documented in this guide and in
[the API reference](https://github.com/LapshinLeonid/configurable_automl_engine/blob/main/API_REFERENCE.md#contracts-package-automl-contracts).

## Package contents and stable import paths

The importable module is `automl_contracts`. Its top-level namespace
re-exports the public symbols and fixes them in `__all__`; importing it must not
import the engine.

| Module | What you may import from it |
|---|---|
| `automl_contracts` | The supported top-level facade: configuration models, result DTOs, enums, constants, exceptions and protocols listed in `automl_contracts.__all__` |
| `automl_contracts.definitions` | `ValidationStrategy`, `EncodingStrategy`, `SerializationFormat`, `TargetTransform`, `ParallelStrategy`, `RunStatus`, `RunStepStatus`, `RunPhaseAction`, `MetricDirection`, `AVAILABLE_ALGORITHMS`, `WORST_SCORE_THRESHOLD` |
| `automl_contracts.errors` | `TrainingError`, `InvalidPredictionError` |
| `automl_contracts.metrics` | `AVAILABLE_METRICS`, `ALIAS_TO_SKLEARN`, `BUILTIN_METRIC_DIRECTIONS`; `to_sklearn_name` is an internal compatibility helper kept for the old engine path |
| `automl_contracts.custom_metrics` | `CustomMetricKind`, `CustomMetricSpec`, `ERR_CUSTOM_NAME_INVALID` |
| `automl_contracts.protocols` | `MetricFunction`, `ScorerCallable` |
| `automl_contracts.search_space` | `SearchSpaceEntry`, `BaseDistribution`, `CategoricalSpace`, `NumericSpace`, `FloatSpace`, `IntSpace` |
| `automl_contracts.feature_selection` | `FeatureSelectionMode`, `FeatureSelectionMethod`, `FeatureSelectionCfg` |
| `automl_contracts.preprocessing` | `ObjectScalingSpec`, `ScalingSpec`, scaling and outlier specifications, `PreprocessingPreset`, `PreprocessingOverride`, `AlgorithmClass`, `coerce_scaling`, `scaling_type`, `coerce_outliers`, `format_outliers`, `merge_preprocessing_overrides`, `NO_OUTLIERS` and the deprecated `ScalingType` alias |
| `automl_contracts.result` | `AlgorithmRun`, `ExperimentResult` |
| `automl_contracts.config` | `Config`, `GeneralCfg`, `HPOPhaseCfg`, `SanityGateCfg`, `ComplexityGuardCfg`, `EnsembleCfg`, `AlgoCfg`, `OversamplingCfg`, `AlgorithmsConfig` |
| `automl_contracts.config.general` | `GeneralCfg`, `ComparisonMetric`, `InfPolicy`, `DatetimePolicy`, `SplitAggregation`, `AUTO_HOLDOUT_FRACTION`, `AUTO_HOLDOUT_MIN_ROWS` |
| `automl_contracts.config.phases` | `HPOPhaseAction`, `SelectStrategy`, `SelectOutput`, `PruningCfg`, `PruningStrategy`, `HPOPhaseCfg`, `DEFAULT_PHASE_N_TRIALS`, `ERR_REMOVED_REFINE_WINNER` |
| `automl_contracts.config.sanity_gate` | `SanityGateCfg`, `CorridorMode`, `GeneralizationGapMode`, `ERR_REMOVED_SANITY_GATE_MODE` |
| `automl_contracts.config.complexity_guard` | `ComplexityGuardCfg`, `ComplexityGuardMode`, `ComplexityGuardUserRangePolicy`, `ComplexityGuardSizeBasis` |
| `automl_contracts.config.ensemble` | `EnsembleCfg`, `EnsembleWeightMethod` |
| `automl_contracts.config.algorithms` | `AlgoCfg`, `OversamplingCfg`, `OversamplingAlgorithm`, `AlgorithmsConfig`, `AlgorithmsConfigType`, `ERR_REMOVED_PLUGIN_KEY` |

`ALIAS_TO_SKLEARN` and `BUILTIN_METRIC_DIRECTIONS` are public module data used
to interpret metric names and result payloads. `to_sklearn_name` is exported
for compatibility but is an internal helper; new code should use the alias map
instead.

`automl_contracts.config.root` and `automl_contracts.config.general_rules` are
internal implementation modules. `automl_contracts/schemas/config.schema.json`
is published package data, not a Python module. `ScalingType` is kept only as a
deprecated alias in `automl_contracts.preprocessing`; prefer `StringScaling` or
the concrete scaling models.

The authoritative list of top-level names is `automl_contracts.__all__`.
`automl_contracts.__version__` is read from the installed distribution
metadata.

## Installation

`automl-contracts` is published on public PyPI, so the contract and its runtime
dependencies install in one step:

```bash
python -m pip install automl-contracts
```

The engine declares `automl-contracts>=0.1.0,<0.2.0`, so
`pip install configurable-automl-engine` installs a compatible contract
automatically. The contract release is published before the engine release that
depends on it. A clean environment needs no credentials, private index or
`--no-deps` workaround. Run `python -m pip check` after installation to confirm
that the requirements resolve.

### Index configuration and dependency-confusion risk

The project publishes and owns the `automl-contracts` name on PyPI, so the
documented installation resolves to the maintained distribution. Keep a single
trusted index and do not add an `--extra-index-url` that can serve another
distribution under the same name: pip gives an additional index no priority, so
a same-named package can win the resolution.

If an organization needs an internal mirror, mirror the PyPI wheel and sdist
with their hashes, pin the exact version in requirements or constraints, and
review every package that resolves from the additional index. Keep `pip.conf`,
`.pypirc`, Docker layers and CI logs free of credentials.

### Publication pipeline

A contract release is published to public PyPI by
`.github/workflows/publish-contracts.yaml`. The SourceCraft `sync-to-github`
workflow pushes the `contracts-v<version>` tag to GitHub, which starts the
release workflow. The workflow uses PyPI Trusted Publishing, so the repository
stores no long-lived PyPI token. It runs only for a `contracts-v<version>` tag
and refuses to publish when:

- the tag does not match the `contracts-v<version>` mask;
- the tag version differs from the static version in `pyproject.toml`;
- any release gate fails: contract boundaries, release configuration, Ruff,
  Mypy, import layers, contract tests, artifact metadata, the clean
  installation of the exact wheel or the consumer E2E on that wheel;
- the PyPI trusted publisher for the workflow is not configured or rejects the
  OIDC token.

The workflow checks out the tag, builds the wheel and sdist, and uploads exactly
the artifacts that passed the gates with `pypa/gh-action-pypi-publish`.
Published versions are immutable: an attempt to upload the same version again
fails instead of being skipped. After the upload the workflow installs
`automl-contracts==<version>` back from PyPI in two clean environments, once
from the wheel and once by building the sdist, and validates a reference
configuration through the public models, protocols and packaged JSON Schema,
without the engine.

The SourceCraft `publish-contracts` workflow additionally mirrors the released
artifacts to the organization registry for private consumers. It requires the
full `refs/tags/contracts-v<version>` reference to point at `HEAD` before the
private upload. Public PyPI is the canonical index documented here, and the
engine resolves the contract from it.

## Using the public API

### Configuration models

```python
from automl_contracts import Config, GeneralCfg, TargetTransform

config = Config.model_validate(
    {
        "general": {
            "comparison_metric": "r2",
            "target_transform": "log1p",
            "holdout_fraction": 0.2,
        },
        "algorithms": {
            "ridge": {},
            "random_forest": {"enable": True},
        },
    }
)

assert isinstance(config.general, GeneralCfg)
assert config.general.target_transform is TargetTransform.log1p
```

The direct contract constructor performs **structural** validation only:
unknown keys, invalid types, invalid phases and structural cross-field rules.
Runtime checks that need the engine's environment (installed algorithm
packages, estimator signatures, callable imports, thread/process limits) run
at the engine entry points, not in this package. An engine consumer should
still call `read_config`, `train_best_model` or `AutoMLExperiment` for the full
check. The internal `load_config` helper is not a public entry point.

### Custom metrics and protocols

```python
from automl_contracts import CustomMetricKind, CustomMetricSpec, MetricFunction


def mape(y_true, y_pred, **kwargs) -> float:
    """A custom metric with the engine calling convention."""
    ...


metric: MetricFunction = mape
kind: CustomMetricKind = "metric"
spec = CustomMetricSpec(
    path="my_project.metrics:mape",
    kind=kind,
    greater_is_better=False,
)

assert spec.cache_key[0] == "my_project.metrics:mape"
```

`CustomMetricSpec` is a data declaration. Creating it does not import the
callable; imports, signatures, direction conflicts and name collisions are
checked by the engine before training. `MetricFunction` and `ScorerCallable`
are structural `Protocol` definitions for static typing; they are not runtime
registries and are not `runtime_checkable`.

### Exceptions and result DTOs

```python
from automl_contracts import (
    AlgorithmRun,
    ExperimentResult,
    InvalidPredictionError,
    RunStatus,
    TrainingError,
)

try:
    ...
except InvalidPredictionError as exc:
    # Both a TrainingError and a ValueError: a model returned non-finite
    # predictions on rows it was required to predict.
    print(exc)
except TrainingError as exc:
    # The engine's data or training-parameter error.
    print(exc)
```

`ExperimentResult` and `AlgorithmRun` are serializable DTOs: the engine writes
them, the store and reporters read them. Their `model_dump(mode="json")`
output is JSON-safe. The contract accepts plain Python numbers; the engine
normalizes NumPy scalars at its boundary.

## JSON Schema without the engine

The canonical schema ships as package data and is available after installing
only `automl-contracts`:

```python
import json
from importlib.resources import files

schema_path = files("automl_contracts").joinpath("schemas/config.schema.json")
schema = json.loads(schema_path.read_text(encoding="utf-8"))

print(schema["title"])
```

You can also generate the schema object in memory:

```python
from automl_contracts import Config

schema = Config.model_json_schema()
```

The JSON file is a build artifact of the Pydantic models, not an independent
source of truth. In a repository checkout, regenerate the canonical file and
the root compatibility mirror after changing any model:

```bash
python automl-contracts/scripts/generate_schema.py
python schema_generator.py
```

A release candidate must pass a schema reproducibility check: regenerate the
schema in a temporary location and require it to match the checked-in canonical
file byte for byte. `tests/contracts/test_json_schema.py` performs this check
as part of `pytest tests/contracts` in CI. Do not edit `config.schema.json` by
hand.

## Versioning and compatibility

`automl-contracts` uses its own SemVer version, independent of the engine
version. The version is static in `automl-contracts/pyproject.toml`; contract
releases are tagged `contracts-v<version>`, while engine releases are tagged
`v<version>`. `setuptools-scm` is used by the engine only, so the two tag
families do not interfere.

The current line is `0.1.x`. Until `1.0.0`:

- potentially incompatible changes are released as a MINOR version and marked
  `BREAKING` in `CHANGELOG.md` with a migration note;
- PATCH is reserved for strictly compatible fixes;
- an incompatible change is never hidden in a PATCH release;
- every public API or schema change is recorded in
  [`automl-contracts/CHANGELOG.md`](https://github.com/LapshinLeonid/configurable_automl_engine/blob/main/automl-contracts/CHANGELOG.md).

From `1.0.0`, the usual SemVer rules apply:

| Change | Release |
|---|---|
| Compatible bug fix that does not change public behavior, types or schema semantics | PATCH |
| New public type, protocol, module or optional field with a safe default | MINOR |
| New value in an `Enum` or a closed `Literal` set | MINOR or MAJOR depending on consumer behavior; a consumer that rejects an unknown value makes it MAJOR |
| New required field | MAJOR |
| Removed or renamed public symbol, field, method or value | MAJOR |
| Changed type or semantics of a public field or method | MAJOR |
| Removed old engine import path | MAJOR |
| Description-only or field-order-only schema change | PATCH |
| Schema change that breaks producers or consumers | MAJOR |

Adding a value to an `Enum` or a closed `Literal` is a producer/consumer
decision, not a cosmetic one: old validators may reject a payload produced by a
newer library. Classify it by the actual compatibility impact and state the
impact in the changelog.

### Compatibility matrix

The engine declares the contract range in its own `pyproject.toml`:

```toml
dependencies = [
    "automl-contracts>=0.1.0,<0.2.0",
]
```

| `automl-contracts` | `configurable-automl-engine` | Status |
|---|---|---|
| `0.1.x` | Releases that declare `>=0.1.0,<0.2.0` | Supported; compatibility is part of the release gate |
| `<0.1.0` | Any | Not supported |
| `>=0.2.0,<1.0.0` | The current engine line | Requires a new engine release with an updated range |
| `>=1.0.0,<2.0.0` | A future engine that declares `>=1.0.0,<2.0.0` | Supported after the policy and range update |

The engine's range is deliberately narrow during `0.x`: a MINOR contract
release may contain an incompatible change, so the engine must not accept
`0.2.0` automatically. A combination outside the declared range is not
supported even if an import happens to work. Changing the range is a separate
change with its own changelog entry; the contract tests verify the declared
range.

When the engine and the contract change together, update both distributions in
one PR, bump the contract version by SemVer, update the engine's range and run
both build/test jobs before merging.

## Migration from engine imports

The engine keeps the old public import paths as re-exports of the same
objects. This is a transition period, not a second source of truth: there must
never be a duplicate class with the same name.

The table below covers the most common imports. The complete matrix, including
every submodule and the engine-only symbols, is in the
[API reference](https://github.com/LapshinLeonid/configurable_automl_engine/blob/main/API_REFERENCE.md#migration-from-engine-imports).

| Before | After |
|---|---|
| `from configurable_automl_engine import Config` | `from automl_contracts import Config` |
| `from configurable_automl_engine import AlgorithmRun, ExperimentResult` | `from automl_contracts import AlgorithmRun, ExperimentResult` |
| `from configurable_automl_engine import CustomMetricSpec` | `from automl_contracts import CustomMetricSpec` |
| `from configurable_automl_engine.common.definitions import ValidationStrategy, TargetTransform, MetricDirection, WORST_SCORE_THRESHOLD` | `from automl_contracts import ...` (or `automl_contracts.definitions`) |
| `from configurable_automl_engine.common.errors import TrainingError, InvalidPredictionError` | `from automl_contracts import TrainingError, InvalidPredictionError` (or `automl_contracts.errors`) |
| `from configurable_automl_engine.training_engine.config_parser import Config, GeneralCfg, HPOPhaseCfg` | `from automl_contracts import Config, GeneralCfg, HPOPhaseCfg` (or the matching `automl_contracts.config.*` module) |
| `from configurable_automl_engine.training_engine.config.general import GeneralCfg` | `from automl_contracts.config.general import GeneralCfg` |
| `from configurable_automl_engine.training_engine.config.ensemble import EnsembleCfg, EnsembleWeightMethod` | `from automl_contracts.config.ensemble import EnsembleCfg, EnsembleWeightMethod` |
| `from configurable_automl_engine.feature_selection import FeatureSelectionCfg` | `from automl_contracts.feature_selection import FeatureSelectionCfg` |
| `from configurable_automl_engine.preprocessing_presets import PreprocessingPreset` | `from automl_contracts.preprocessing import PreprocessingPreset` |
| `from configurable_automl_engine.outliers_presets import NoOutliers` | `from automl_contracts.preprocessing import NoOutliers` |
| `from configurable_automl_engine.hpo.default_spaces import SearchSpaceEntry, FloatSpace` | `from automl_contracts.search_space import SearchSpaceEntry, FloatSpace` |
| `from configurable_automl_engine.experiment.result import ExperimentResult` | `from automl_contracts.result import ExperimentResult` |
| `from configurable_automl_engine.models import AVAILABLE_ALGORITHMS` | `from automl_contracts import AVAILABLE_ALGORITHMS` |
| `from configurable_automl_engine.common.metrics import AVAILABLE_METRICS, to_sklearn_name` | `from automl_contracts.metrics import AVAILABLE_METRICS, to_sklearn_name` (the second name is an internal compatibility helper) |
| `from configurable_automl_engine.common.custom_metrics import CustomMetricSpec, CustomMetricKind` | `from automl_contracts.custom_metrics import CustomMetricSpec, CustomMetricKind` |
| `from configurable_automl_engine.trainer import TrainingError` | `from automl_contracts.errors import TrainingError` |
| `from configurable_automl_engine.trainer_params import TrainingError` | `from automl_contracts.errors import TrainingError` |
| `from configurable_automl_engine.tuner import WORST_SCORE_THRESHOLD` | `from automl_contracts import WORST_SCORE_THRESHOLD` |
| `from configurable_automl_engine.preprocessing import EncodingStrategy` | `from automl_contracts import EncodingStrategy` |

Engine-only APIs keep their old paths and do not move into the contract:
`read_config`, `train_best_model`, `AutoMLExperiment`, `ModelTrainer`,
`train_model`, `optimize`, `create_model`, `FeatureSelector`,
`DataOversampler`, reporters, `ExperimentStore`, `export_experiment`,
`explain` and `DEFAULT_SPACES`. The internal `load_config` helper is used by
the engine but is not a public entry point.

Re-export policy:

- all old paths in the compatibility matrix remain importable for the whole
  `0.x` contract line;
- re-exported classes, enums and exceptions are identical objects (`is`);
  type aliases and constants have equal values and types;
- `ScalingType` is a deprecated alias and is not part of the top-level
  `__all__`;
- removal of an old path requires a MAJOR release, a deprecation period and a
  changelog entry that names the replacement.

The exhaustive table, including submodule imports and engine-only paths, lives
in [the API reference](https://github.com/LapshinLeonid/configurable_automl_engine/blob/main/API_REFERENCE.md#migration-from-engine-imports).

## Changing the contract

A contract change and the engine change that consumes it are made in one pull
request. The mandatory order is:

1. Decide the compatibility class of the change (PATCH, MINOR or MAJOR) and
   write it down before editing the code.
2. Update the contract models, enums, protocols or exceptions.
3. Regenerate the canonical JSON Schema and the root compatibility mirror.
4. Bump the static version in `automl-contracts/pyproject.toml` when the
   public API or schema changes.
5. Update `automl-contracts/CHANGELOG.md` with the change, its compatibility
   class, the affected public API or schema and the migration instructions.
6. Update the inclusion rules and the compatibility matrix in
   `API_REFERENCE.md` when a boundary, a decision or a public path changes.
7. Update the consumer documentation and the migration table when an import
   path changes.
8. Update the engine's dependency range when the contract line changes.
9. Run the contract tests, the engine tests, `ruff`, `mypy`, `lint-imports`,
   the schema reproducibility check and both isolated builds.
10. Publish only after merge, from the `contracts-v*` tag, after the version
    check passes.

A documentation-only change does not bump the contract version. A public API
or schema change without a changelog entry, a compatibility-matrix update where
required, a regenerated schema and a version decision is incomplete and must
not be merged. The contract tests check the documented public surface, the
declared version range and the Pydantic declarations; they do not pin a
snapshot of that surface and do not read `CHANGELOG.md`. The review checklist
is what catches a missing changelog entry or an undocumented compatibility
change.

The maintainer process is described above. Installation and index guidance
are documented in the [installation section](#installation).
