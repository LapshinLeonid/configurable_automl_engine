"""Result DTOs, metric names and the JSON Schema of the contract."""

from __future__ import annotations

import datetime as dt
import inspect
import json
import uuid
from pathlib import Path

import automl_contracts
from automl_contracts import AVAILABLE_METRICS


def _result(**overrides):
    values = {
        "run_id": uuid.uuid4(),
        "started_at": dt.datetime.now(dt.UTC),
        "finished_at": dt.datetime.now(dt.UTC),
        "duration_seconds": 0.0,
        "status": "completed",
        "config": {},
        "resolved_validation": "k_fold",
        "resolved_n_folds": 3,
        "resolved_test_size": 0.2,
        "metric_user": "rmse",
        "metric_sklearn": "neg_root_mean_squared_error",
        "metric_direction": "minimize",
        "winner": "ridge",
        "winner_score": 1.5,
    }
    values.update(overrides)
    return automl_contracts.ExperimentResult(**values)


def test_result_round_trip_is_json_safe() -> None:
    result = _result()
    restored = automl_contracts.ExperimentResult.from_json(result.to_json())
    assert restored.model_dump() == result.model_dump()


def test_result_rejects_wrong_builtin_metric_direction() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="metric_direction"):
        _result(metric_direction="maximize")


def test_metric_names_and_directions() -> None:
    from automl_contracts.definitions import MetricDirection
    from automl_contracts.metrics import BUILTIN_METRIC_DIRECTIONS

    assert AVAILABLE_METRICS == [
        "nrmse",
        "rmse",
        "neg_root_mean_squared_error",
        "mae",
        "mse",
        "r2",
    ]
    assert BUILTIN_METRIC_DIRECTIONS["r2"] is MetricDirection.maximize
    assert BUILTIN_METRIC_DIRECTIONS["global_nrmse"] is MetricDirection.minimize


def _clean(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "description" and isinstance(item, str):
                value[key] = inspect.cleandoc(item).strip()
            else:
                _clean(item)
    elif isinstance(value, list):
        for item in value:
            _clean(item)


def test_saved_schema_matches_the_models() -> None:
    schema = automl_contracts.Config.model_json_schema()
    _clean(schema)
    path = (
        Path(__file__).resolve().parents[2]
        / "configurable-automl-contracts"
        / "src"
        / "automl_contracts"
        / "schemas"
        / "config.schema.json"
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == schema
    root_copy = Path(__file__).resolve().parents[2] / "config.schema.json"
    assert json.loads(root_copy.read_text(encoding="utf-8")) == schema
    assert root_copy.read_bytes() == path.read_bytes()
