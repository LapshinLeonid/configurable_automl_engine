# automl-contracts

Public data contract of the Configurable AutoML Engine.

The package contains the configuration models, result DTOs, enumerations,
exceptions, constants, structural protocols and the JSON Schema used by
external consumers. It depends only on the Python standard library and
Pydantic; the AutoML engine and its runtime dependencies are not required.

```python
from automl_contracts import Config, ExperimentResult

cfg = Config.model_validate(
    {
        "general": {},
        "algorithms": {"random_forest": {"enable": True}},
    }
)
```

The engine applies additional runtime checks (installed algorithm packages,
estimator hyperparameter signatures and custom-metric callables) at its public
entry points. The contract validates only structure and public data rules.

The JSON Schema is available as package data:

```python
from importlib.resources import files

schema = files("automl_contracts").joinpath("schemas/config.schema.json")
```
