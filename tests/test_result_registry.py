"""Tests of the explicit result hand-off API.

The engine publishes each assembled ``ExperimentResult`` through the result
registry; consumers read it only through the API and can request the result
owned by their private run token. Covered scenarios:

- P1: a published result is returned as the latest result;
- P2: an owner-scoped read returns the result only for the matching token;
- P3: publishing a new result replaces the latest one and its owner;
- P4: concurrent publishers never expose a foreign result through the
  owner-scoped read;
- B1: an empty registry returns ``None`` for both read forms;
- B2: clearing drops the published result;
- N1: the result registry does not expose a mutable module-level result
  variable, and ``component`` carries none;
- N2: the deprecated ``component.last_experiment_result`` read-only alias
  returns the same result as the explicit registry API.
"""

from __future__ import annotations

import datetime as dt
import threading
import uuid

import pytest

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunStatus,
)
from configurable_automl_engine.experiment.result import ExperimentResult
from configurable_automl_engine.training_engine import component
from configurable_automl_engine.training_engine.result_registry import (
    clear_experiment_result,
    get_experiment_result_owned_by,
    get_last_experiment_result,
    publish_experiment_result,
)
from configurable_automl_engine.training_engine.run_result import result_owned_by


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    """Clear the process-wide registry around every test."""
    clear_experiment_result()
    yield
    clear_experiment_result()


def _result() -> ExperimentResult:
    """A minimal constructed result; the registry does not inspect its fields."""
    now = dt.datetime.now(dt.UTC)
    return ExperimentResult.model_construct(
        run_id=uuid.uuid4(),
        started_at=now,
        finished_at=now,
        duration_seconds=0.0,
        status=RunStatus.completed,
        config={},
        resolved_validation="holdout",
        resolved_n_folds=5,
        resolved_test_size=0.2,
        metric_user="rmse",
        metric_sklearn="neg_root_mean_squared_error",
        metric_direction=MetricDirection.minimize,
    )


def test_p1_publish_and_read_latest_result() -> None:
    """P1: a published result is returned as the latest result."""
    clear_experiment_result()
    result = _result()
    publish_experiment_result(result)
    assert get_last_experiment_result() is result


def test_p2_owner_scoped_read_returns_only_own_result() -> None:
    """P2: the owner-scoped read matches the private run token."""
    clear_experiment_result()
    token = uuid.uuid4()
    result = _result()
    publish_experiment_result(result, run_token=token)

    assert get_experiment_result_owned_by(token) is result
    assert get_experiment_result_owned_by(uuid.uuid4()) is None
    assert result_owned_by(result, token)


def test_p3_new_publish_replaces_result_and_owner() -> None:
    """P3: the latest result and its owner are replaced together."""
    clear_experiment_result()
    first_token = uuid.uuid4()
    second_token = uuid.uuid4()
    first = _result()
    second = _result()
    publish_experiment_result(first, run_token=first_token)
    publish_experiment_result(second, run_token=second_token)

    assert get_last_experiment_result() is second
    assert get_experiment_result_owned_by(first_token) is None
    assert get_experiment_result_owned_by(second_token) is second


def test_p4_concurrent_publish_never_returns_foreign_result() -> None:
    """P4: concurrent publishers never expose a foreign result to a token owner."""
    workers = 8
    barrier = threading.Barrier(workers)
    results = [_result() for _ in range(workers)]
    tokens = [uuid.uuid4() for _ in range(workers)]
    observed: list[tuple[int, ExperimentResult | None]] = []
    errors: list[Exception] = []
    observed_lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            publish_experiment_result(results[index], run_token=tokens[index])
            seen = get_experiment_result_owned_by(tokens[index])
        except Exception as err:  # pragma: no cover - surfaced by the assertions
            with observed_lock:
                errors.append(err)
            return
        with observed_lock:
            observed.append((index, seen))

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(observed) == workers
    for index, seen in observed:
        assert seen is None or seen is results[index]

    latest = get_last_experiment_result()
    assert latest is not None
    owners = [index for index, result in enumerate(results) if latest is result]
    assert len(owners) == 1
    owner = owners[0]
    assert get_experiment_result_owned_by(tokens[owner]) is latest
    for index, token in enumerate(tokens):
        if index != owner:
            assert get_experiment_result_owned_by(token) is None


def test_b1_empty_registry_returns_none() -> None:
    """B1: both read forms return ``None`` with no published result."""
    clear_experiment_result()
    assert get_last_experiment_result() is None
    assert get_experiment_result_owned_by(uuid.uuid4()) is None


def test_b2_clear_drops_published_result() -> None:
    """B2: clearing removes the published result and its owner."""
    token = uuid.uuid4()
    publish_experiment_result(_result(), run_token=token)
    clear_experiment_result()
    assert get_last_experiment_result() is None
    assert get_experiment_result_owned_by(token) is None


def test_n1_no_mutable_result_global_is_exposed() -> None:
    """N1: the result hand-off state is hidden behind the explicit API."""
    import configurable_automl_engine.training_engine.result_registry as registry

    assert "last_experiment_result" not in component.__dict__
    assert not hasattr(registry, "last_experiment_result")


def test_n2_deprecated_component_alias_reads_registry() -> None:
    """N2: the deprecated component alias returns the explicit registry result."""
    result = _result()
    publish_experiment_result(result)
    with pytest.warns(DeprecationWarning, match="get_last_experiment_result"):
        assert component.last_experiment_result is result
