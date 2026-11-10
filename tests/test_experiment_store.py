"""Tests of the experiment store — the run history.

Covered scenarios (positive / negative / boundary):

- P1: add/get/list work with a full ``ExperimentResult``;
- P2: JSONL persist and restore after the store is recreated;
- P3: status filtering and time sorting;
- N1: duplicate ``run_id`` → ``DuplicateRunError`` (with ``force=True``
  the registration is replaced);
- N2: a damaged JSONL line is skipped with a warning, the other records
  survive;
- N2b: a record that fails the model validation is skipped the same way;
- N3: a missing ``run_id`` raises ``KeyError``;
- B1: an empty store lists nothing and the label lookup is empty;
- B2: thousands of runs — ``list_runs`` does not degrade;
- B3: concurrent registration from several threads;
- B4: a label with special characters/unicode;
- B5: a fully failed run is stored and restored.
"""

from __future__ import annotations

import datetime as dt
import threading
import uuid
from typing import Any

import pytest

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment import (
    AlgorithmRun,
    DuplicateRunError,
    ExperimentResult,
    ExperimentStore,
    RunSummary,
)

UTC = dt.UTC


def _ts(seconds: float = 0.0) -> dt.datetime:
    """A fixed UTC timestamp shifted by ``seconds``."""
    return dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + dt.timedelta(seconds=seconds)


def _run_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default keyword arguments of a completed ``AlgorithmRun`` record."""
    values: dict[str, Any] = {
        "algorithm": "ridge",
        "phase_name": "search",
        "phase_action": "all_algorithms",
        "status": RunStepStatus.completed,
        "score_raw": -0.5,
        "score": 0.5,
        "params": {"alpha": 0.1},
        "n_trials_requested": 10,
        "n_trials_evaluated": 10,
        "n_trials_completed": 10,
        "duration_seconds": 12.5,
    }
    values.update(overrides)
    return values


def _result_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default keyword arguments of a completed ``ExperimentResult``."""
    values: dict[str, Any] = {
        "run_id": uuid.uuid4(),
        "label": "baseline run",
        "engine_version": "0.0.0",
        "started_at": _ts(),
        "finished_at": _ts(90.0),
        "duration_seconds": 90.0,
        "status": RunStatus.completed,
        "config": {"general": {"comparison_metric": "rmse", "random_state": 42}},
        "resolved_validation": "k_fold",
        "resolved_n_folds": 5,
        "resolved_test_size": 0.2,
        "random_state": 42,
        "metric_user": "rmse",
        "metric_sklearn": "neg_root_mean_squared_error",
        "metric_direction": MetricDirection.minimize,
        "runs": [AlgorithmRun(**_run_kwargs())],
        "winner": "ridge",
        "winner_score": 0.5,
        "winner_params": {"alpha": 0.1},
        "model_path": "models/best.pkl",
        "additional_metrics": {"mae": 1.25},
        "oof_score": 0.48,
        "n_trials_evaluated": 10,
    }
    values.update(overrides)
    return values


def _make_result(**overrides: Any) -> ExperimentResult:
    """Build a valid ``ExperimentResult`` with the default kwargs."""
    return ExperimentResult(**_result_kwargs(**overrides))


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_add_get_list_with_full_result() -> None:
    """P1: add/get/list work with a full ExperimentResult."""
    store = ExperimentStore()
    result = _make_result()
    store.add(result)

    assert len(store) == 1
    assert result.run_id in store

    fetched = store.get(result.run_id)
    assert fetched == result
    # Also fetchable by the canonical string form of the UUID.
    assert store.get(str(result.run_id)) == result

    summaries = store.list_runs()
    assert len(summaries) == 1
    summary = summaries[0]
    assert isinstance(summary, RunSummary)
    assert summary.run_id == result.run_id
    assert summary.label == result.label
    assert summary.status is RunStatus.completed
    assert summary.started_at == result.started_at
    assert summary.finished_at == result.finished_at
    assert summary.duration_seconds == result.duration_seconds
    assert summary.engine_version == "0.0.0"


def test_p2_jsonl_persist_and_restore(tmp_path) -> None:
    """P2: persist to JSONL and restore after the store is recreated."""
    path = tmp_path / "history.jsonl"
    first = _make_result(label="first")
    second = _make_result(label="second")

    with ExperimentStore(path) as store:
        store.add(first)
        store.add(second)
    assert path.exists()
    assert path.read_text(encoding="utf-8").count("\n") == 2

    restored = ExperimentStore(path)
    assert len(restored) == 2
    assert restored.get(first.run_id) == first
    assert restored.get(second.run_id) == second
    restored.close()

    # No duplicates on a repeated restore.
    assert len(ExperimentStore(path)._results) == 2  # noqa: SLF001


def test_p3_status_filter_and_time_sort() -> None:
    """P3: filter by status and sort by time."""
    store = ExperimentStore()
    early = _make_result(
        run_id=uuid.uuid4(),
        label="early",
        started_at=_ts(0.0),
        finished_at=_ts(10.0),
        duration_seconds=10.0,
    )
    late = _make_result(
        run_id=uuid.uuid4(),
        label="late",
        started_at=_ts(100.0),
        finished_at=_ts(150.0),
        duration_seconds=50.0,
    )
    failed = _make_result(
        run_id=uuid.uuid4(),
        label="broken",
        status=RunStatus.failed,
        winner=None,
        winner_score=None,
        error="all algorithms failed",
        started_at=_ts(50.0),
        finished_at=_ts(60.0),
        duration_seconds=10.0,
        runs=[],
    )
    for result in (early, failed, late):
        store.add(result)

    assert [s.run_id for s in store.list_runs()] == [
        early.run_id,
        failed.run_id,
        late.run_id,
    ]
    assert [s.run_id for s in store.list_runs(descending=True)] == [
        late.run_id,
        failed.run_id,
        early.run_id,
    ]
    assert [s.run_id for s in store.list_runs(status=RunStatus.completed)] == [
        early.run_id,
        late.run_id,
    ]
    # A plain string status is accepted too.
    assert [s.run_id for s in store.list_runs(status="failed")] == [failed.run_id]
    assert [s.run_id for s in store.list_runs(label="late")] == [late.run_id]

    # Sorting by duration: smallest first, largest last.
    by_duration = store.list_runs(order_by="duration_seconds")
    assert [s.run_id for s in by_duration] == [
        early.run_id,
        failed.run_id,
        late.run_id,
    ]
    by_duration_desc = store.list_runs(order_by="duration_seconds", descending=True)
    # The sort is stable: the tied durations (early, failed) keep the
    # registration order even in the descending run.
    assert [s.run_id for s in by_duration_desc] == [
        late.run_id,
        early.run_id,
        failed.run_id,
    ]

    # Sorting by finished_at.
    by_finished = store.list_runs(order_by="finished_at")
    assert [s.run_id for s in by_finished] == [
        early.run_id,
        failed.run_id,
        late.run_id,
    ]


def test_p3_find_by_label_returns_list_in_registration_order() -> None:
    """Labels are not unique: the label lookup returns every match."""
    store = ExperimentStore()
    shared_label = "baseline"
    first = _make_result(run_id=uuid.uuid4(), label=shared_label)
    other = _make_result(run_id=uuid.uuid4(), label="other")
    second = _make_result(run_id=uuid.uuid4(), label=shared_label)
    for result in (first, other, second):
        store.add(result)

    matches = store.find_by_label(shared_label)
    assert [r.run_id for r in matches] == [first.run_id, second.run_id]
    assert all(r == expected for r, expected in zip(matches, (first, second)))


# ────────────────────────── negative scenarios ────────────────────────── #


def test_n1_duplicate_run_id_raises() -> None:
    """N1: a duplicate run_id raises DuplicateRunError (error policy)."""
    store = ExperimentStore()
    result = _make_result()
    store.add(result)

    duplicate = _make_result(run_id=result.run_id, label="second registration")
    with pytest.raises(DuplicateRunError, match="already registered"):
        store.add(duplicate)

    # The original record survives the rejected registration.
    assert store.get(result.run_id) == result
    assert len(store) == 1


def test_n1_force_replaces_duplicate() -> None:
    """N1 (policy): force=True replaces the already registered run."""
    store = ExperimentStore()
    result = _make_result()
    store.add(result)

    replacement = _make_result(run_id=result.run_id, label="replacement")
    store.add(replacement, force=True)

    assert len(store) == 1
    assert store.get(result.run_id) == replacement
    assert store.find_by_label("replacement") == [replacement]


def test_n2_damaged_jsonl_line_skipped_with_warning(tmp_path) -> None:
    """N2: a damaged JSONL line is skipped with a warning, others survive."""
    path = tmp_path / "history.jsonl"
    good = _make_result(label="good")
    # Write a good line, a truncated (interrupted write) line, a garbage
    # line and a second good line.
    truncated = good.to_json()[: len(good.to_json()) // 2]
    path.write_text(
        good.to_json() + "\n" + truncated + "\n{not json at all}\n\n",
        encoding="utf-8",
    )
    another = _make_result(label="another")
    with ExperimentStore(path, restore=False) as store:
        store.add(another)

    with pytest.warns(RuntimeWarning, match="damaged history line"):
        store = ExperimentStore(path)

    assert len(store) == 2
    assert store.get(good.run_id) == good
    assert store.get(another.run_id) == another
    store.close()


def test_n2b_schema_invalid_record_skipped_same_policy(tmp_path) -> None:
    """N2b: a record failing the model validation is skipped, not a crash."""
    path = tmp_path / "history.jsonl"
    good = _make_result(label="good")
    # A parsed-JSON record that cannot validate: a failed run without an
    # error message (a schema/model-level violation, not a JSON error).
    broken_payload = {
        **__import__("json").loads(good.to_json()),
        "run_id": str(uuid.uuid4()),
        "status": "failed",
        "error": None,
        "winner": None,
        "winner_score": None,
        "runs": [],
    }
    path.write_text(
        good.to_json() + "\n" + __import__("json").dumps(broken_payload) + "\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="damaged history line"):
        store = ExperimentStore(path)

    assert len(store) == 1
    assert store.get(good.run_id) == good
    store.close()


def test_n3_missing_run_id_raises_key_error() -> None:
    """N3: a missing run_id raises KeyError (the documented contract)."""
    store = ExperimentStore()
    missing = uuid.uuid4()
    with pytest.raises(KeyError, match=str(missing)):
        store.get(missing)
    with pytest.raises(KeyError, match=str(missing)):
        store.get(str(missing))


def test_n3_invalid_run_id_string_raises_key_error() -> None:
    """A malformed UUID string is rejected as a missing (unparsable) run."""
    store = ExperimentStore()
    with pytest.raises((KeyError, ValueError)):
        store.get("not-a-uuid")


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_empty_store() -> None:
    """B1: an empty store lists nothing and lookups come back empty."""
    store = ExperimentStore()
    assert len(store) == 0
    assert store.list_runs() == []
    assert store.find_by_label("anything") == []
    assert "00000000-0000-0000-0000-000000000000" not in store

    with pytest.raises(KeyError):
        store.get(uuid.uuid4())


def test_b2_many_runs_list_does_not_degrade() -> None:
    """B2: thousands of runs — list_runs stays fast and correct."""
    store = ExperimentStore()
    n = 3000
    results = []
    for index in range(n):
        result = _make_result(
            run_id=uuid.uuid4(),
            label=f"run-{index}",
            started_at=_ts(float(index)),
            finished_at=_ts(float(index) + 1.0),
            duration_seconds=1.0,
        )
        results.append(result)
        store.add(result)

    assert len(store) == n
    summaries = store.list_runs()
    assert len(summaries) == n
    # Sorted ascending by started_at == the registration order.
    assert [s.run_id for s in summaries] == [r.run_id for r in results]
    assert summaries[0].label == "run-0"
    assert summaries[-1].label == f"run-{n - 1}"


def test_b2_many_runs_restore_from_file(tmp_path) -> None:
    """B2 (persist): a large history restores fully from JSONL."""
    path = tmp_path / "history.jsonl"
    n = 1500
    with ExperimentStore(path) as store:
        for index in range(n):
            store.add(
                _make_result(
                    run_id=uuid.uuid4(),
                    label=f"run-{index}",
                    started_at=_ts(float(index)),
                    finished_at=_ts(float(index) + 1.0),
                    duration_seconds=1.0,
                )
            )

    restored = ExperimentStore(path)
    try:
        assert len(restored) == n
        assert len(restored.list_runs(status="completed")) == n
    finally:
        restored.close()


def test_b3_concurrent_registration_from_threads(tmp_path) -> None:
    """B3: parallel registration from threads — no lost updates."""
    store = ExperimentStore()
    n_threads = 8
    per_thread = 25
    errors: list[Exception] = []

    def register(thread_index: int) -> None:
        try:
            for index in range(per_thread):
                store.add(
                    _make_result(
                        run_id=uuid.uuid4(),
                        label=f"thread-{thread_index}-run-{index}",
                        started_at=_ts(float(thread_index * per_thread + index)),
                        finished_at=_ts(float(thread_index * per_thread + index) + 1.0),
                        duration_seconds=1.0,
                    )
                )
        except Exception as err:  # noqa: BLE001 - collected and asserted below
            errors.append(err)

    threads = [threading.Thread(target=register, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(store) == n_threads * per_thread
    assert len(store.list_runs()) == n_threads * per_thread
    # Every registered run is individually retrievable.
    for thread_index in range(n_threads):
        assert len(store.find_by_label(f"thread-{thread_index}-run-0")) == 1

    # The JSONL mirror is serialized and complete: all lines restore.
    path = tmp_path / "history.jsonl"
    with ExperimentStore(path) as persisted:
        for index in range(per_thread):
            persisted.add(
                _make_result(
                    run_id=uuid.uuid4(),
                    label=f"run-{index}",
                    started_at=_ts(float(index)),
                    finished_at=_ts(float(index) + 1.0),
                    duration_seconds=1.0,
                )
            )
    restored = ExperimentStore(path)
    try:
        assert len(restored) == per_thread
    finally:
        restored.close()


def test_b3_concurrent_adds_to_jsonl_file(tmp_path) -> None:
    """B3 (persist): concurrent adds append every line exactly once."""
    path = tmp_path / "history.jsonl"
    with ExperimentStore(path) as store:
        threads = [
            threading.Thread(
                target=lambda i=i: [
                    store.add(
                        _make_result(
                            run_id=uuid.uuid4(),
                            label=f"t{i}-{j}",
                            started_at=_ts(float(i * 10 + j)),
                            finished_at=_ts(float(i * 10 + j) + 1.0),
                            duration_seconds=1.0,
                        )
                    )
                    for j in range(10)
                ]
            )
            for i in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 50
    restored = ExperimentStore(path)
    try:
        assert len(restored) == 50
    finally:
        restored.close()


def test_b4_label_with_special_characters_and_unicode(tmp_path) -> None:
    """B4: a label with special characters/unicode survives the round-trip."""
    tricky_label = 'run #1 — "quoting" \\ | 🚀 100% ✓\n'
    path = tmp_path / "history.jsonl"
    result = _make_result(label=tricky_label)

    with ExperimentStore(path) as store:
        store.add(result)

    restored = ExperimentStore(path)
    try:
        assert restored.get(result.run_id).label == tricky_label
        assert [r.run_id for r in restored.find_by_label(tricky_label)] == [
            result.run_id
        ]
    finally:
        restored.close()


def test_b4_none_label_allowed() -> None:
    """B4: a run without a label is stored and filtered correctly."""
    store = ExperimentStore()
    unlabeled = _make_result(label=None)
    store.add(unlabeled)

    assert unlabeled.label is None
    assert store.get(unlabeled.run_id).label is None
    assert store.list_runs(label=None)[0].run_id == unlabeled.run_id
    assert store.list_runs(label="") == []


def test_b5_failed_run_stored_and_restored(tmp_path) -> None:
    """B5: a fully failed run is stored, filtered and restored from JSONL."""
    path = tmp_path / "history.jsonl"
    failed = _make_result(
        status=RunStatus.failed,
        winner=None,
        winner_score=None,
        winner_params={},
        model_path=None,
        oof_score=None,
        n_trials_evaluated=None,
        error="all algorithms failed: both HPO runs crashed",
        runs=[],
    )

    with ExperimentStore(path) as store:
        store.add(failed)

    assert failed.status is RunStatus.failed
    assert failed.winner is None

    restored = ExperimentStore(path)
    try:
        fetched = restored.get(failed.run_id)
        assert fetched == failed
        assert fetched.status is RunStatus.failed
        assert fetched.error == "all algorithms failed: both HPO runs crashed"
        assert restored.list_runs(status=RunStatus.failed)[0].run_id == failed.run_id
        assert restored.list_runs(status=RunStatus.completed) == []
    finally:
        restored.close()


def test_b5_failed_run_with_winner_restored() -> None:
    """A run may fail after a winner was determined; the store keeps it."""
    store = ExperimentStore()
    failed_after_winner = _make_result(
        status=RunStatus.failed,
        error="final artifact save crashed",
    )
    store.add(failed_after_winner)

    fetched = store.get(failed_after_winner.run_id)
    assert fetched == failed_after_winner
    assert fetched.winner == "ridge"
    assert fetched.status is RunStatus.failed
