"""Tests of the ``AutoMLExperiment`` object facade.

Covered scenarios (positive / negative / boundary):

- P1: full lifecycle ``run → leaderboard → best_model → report`` on a
  small dataset;
- P2: the result is registered in the ``ExperimentStore`` with a correct
  ``run_id`` (and in a JSONL-backed store);
- P3: the facade accepts all four config forms (``Config``, dict, str
  path, Path);
- P4: a repeated ``run()`` produces a new ``run_id`` with a
  ``RuntimeWarning`` and keeps the previous run in the store history;
- P5: a custom reporter chain replaces the default one;
- P6: a degrading custom reporter's ``None`` artifact propagates through
  the facade;
- N1: calling the result accessors before ``run()`` → a clear state
  error (``ExperimentNotRunError``, a ``RuntimeError`` subclass);
- N2: invalid config type / empty data / missing target → the legacy
  validation errors (``TypeError`` / ``ValueError``) propagate from
  ``run()``;
- N3: a failed run (all algorithms failed, ``phases: []``) → the
  exception propagates, a failed result is registered in the store, and
  ``best_model()`` reports the consistent "no model" state;
- B1: a single algorithm, a single phase;
- B2: ``phases: []`` → the failed run;
- B3: a frame with the target column only (no features);
- B4: a ``pathlib.Path`` config;
- B5: a full HPO failure — the failed status, no model artifact;
- B6: two experiments in parallel threads share no state;
- R1 (review B-1): a fresh experiment never adopts a stranger's result
  left in the engine's result registry (sequential and concurrent
  threads);
- R2 (review B-2): a store registration failure never masks the run's
  own error;
- R3: ``run()`` returns a result carrying this experiment's owner stamp
  (the engine-side token contract);
- R4: a repeated run that fails before reaching the engine does not adopt
  the previous run's result (a fresh token per run).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pytest

from configurable_automl_engine import AutoMLExperiment, train_best_model
from configurable_automl_engine.common.definitions import RunStatus
from configurable_automl_engine.experiment.reporting import (
    JSONReporter,
    MarkdownReporter,
    Reporter,
    ReporterArtifact,
)
from configurable_automl_engine.experiment.result import ExperimentResult
from configurable_automl_engine.experiment.store import ExperimentStore
from configurable_automl_engine.training_engine.config_parser import Config
from configurable_automl_engine.training_engine.experiment_facade import (
    DEFAULT_REPORT_JSON_NAME,
    ExperimentNotRunError,
)
from configurable_automl_engine.training_engine.result_registry import (
    clear_experiment_result,
    get_last_experiment_result,
    publish_experiment_result,
)

# ────────────────────────── helpers ────────────────────────── #


def _config_dict(tmp_path: Path, **general_overrides: Any) -> dict[str, Any]:
    """A valid two-phase config dict (HPO phase + terminal cv_best select)."""
    general: dict[str, Any] = {
        "comparison_metric": "rmse",
        "random_state": 42,
        "validation_strategy": "train_test_split",
        "holdout_fraction": None,
        "path_to_model": str(tmp_path / "model.pkl"),
        "phases": [
            {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
            {
                "name": "final",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
    }
    general.update(general_overrides)
    return {
        "general": general,
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            },
            "elasticnet": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.8]},
            },
        },
        "oversampling": {"enable": False},
    }


def _small_df() -> pd.DataFrame:
    """A tiny deterministic regression dataset."""
    rows = 40
    return pd.DataFrame(
        {
            "f1": [float(i % 5) for i in range(rows)],
            "f2": [i % 7 * 0.5 for i in range(rows)],
            "target": [10.0 - 0.2 * i for i in range(rows)],
        }
    )


def _yaml_config(tmp_path: Path, **overrides: Any) -> Path:
    """Write the config dict to a YAML file and return its path."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config_dict(tmp_path, **overrides)))
    return path


@pytest.fixture(autouse=True)
def _reset_engine_state() -> None:
    """Isolate the engine's result registry between tests.

    The registry is cleared before AND after every test, so a stale result
    from one test can never leak into another.
    """
    clear_experiment_result()
    yield
    clear_experiment_result()


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_full_lifecycle_run_leaderboard_best_model_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P1: run → leaderboard → best_model → report on a small dataset."""
    experiment = AutoMLExperiment(
        _config_dict(tmp_path), _small_df(), target="target", label="lifecycle"
    )
    result = experiment.run()

    assert result.status is RunStatus.completed
    assert experiment.result is result
    assert experiment.run_id == result.run_id

    board = experiment.leaderboard()
    assert len(board) >= 1
    assert board.metric_user == "rmse"
    assert board.winner is not None
    assert board.winner.algorithm == result.winner
    rendered = capsys.readouterr().out
    assert rendered == ""

    printed = experiment.leaderboard(print_table=True)
    assert printed.to_table() in capsys.readouterr().out

    summarized = experiment.leaderboard(summary=True)
    summary_out = capsys.readouterr().out
    # The UX summary block: the status header,
    # the ranking and the Winner line reach the stdout through the
    # facade (one call, no manual to_summary() wiring).
    assert "Experiment completed" in summary_out
    assert result.winner in summary_out
    assert f"Winner: {result.winner}" in summary_out
    assert summarized.to_summary(status=True) in summary_out

    model = experiment.best_model()
    assert model.algorithm == result.winner

    report = experiment.report()
    assert isinstance(report, str)
    stdout = capsys.readouterr().out
    assert "# Experiment report: lifecycle" in stdout
    assert result.winner in stdout
    # The default chain also wrote the JSON snapshot next to the artifact.
    json_path = Path(result.model_path).parent / DEFAULT_REPORT_JSON_NAME
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == str(result.run_id)


def test_p2_result_registered_in_store(tmp_path: Path) -> None:
    """P2: the run result is registered in the store with a correct run_id."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    result = experiment.run()
    assert len(experiment.store) == 1
    stored = experiment.store.get(result.run_id)
    assert stored == result
    assert stored.winner == result.winner

    # The same through a JSONL-backed store: the registration lands in
    # the file and is restorable.
    history = tmp_path / "history.jsonl"
    backed = AutoMLExperiment(
        _config_dict(tmp_path), _small_df(), store_path=history, label="backed"
    )
    backed_result = backed.run()
    assert history.exists()
    restored_store = ExperimentStore(path=history)
    assert restored_store.get(backed_result.run_id).label == "backed"
    assert len(backed.store) == 1


def test_p3_all_config_forms(tmp_path: Path) -> None:
    """P3: Config object / dict / mapping / str path / Path all keep working."""
    df = _small_df()
    results = []
    for form in (
        Config.model_validate(_config_dict(tmp_path)),
        _config_dict(tmp_path),
        MappingProxyType(_config_dict(tmp_path)),
        str(_yaml_config(tmp_path)),
        _yaml_config(tmp_path),
    ):
        experiment = AutoMLExperiment(form, df, target="target")
        results.append(experiment.run())
    assert all(r.status is RunStatus.completed for r in results)
    assert all(r.winner for r in results)


def test_p3_train_best_model_wrapper_contract(tmp_path: Path) -> None:
    """P3: the functional ``train_best_model`` keeps its legacy dict
    contract and stays an alias of the engine entry point."""
    res = train_best_model(_config_dict(tmp_path), _small_df(), target="target")
    for key in ("algorithm", "score", "metric", "params", "model_path", "random_state"):
        assert key in res, key
    assert res["metric"] == "rmse"
    assert res["random_state"] == 42


def test_p4_repeated_run_new_run_id_with_warning(tmp_path: Path) -> None:
    """P4: a repeated run() trains again, gets a new run_id, keeps the
    previous run in the store history, and warns about the repeat."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    first = experiment.run()
    with pytest.warns(RuntimeWarning, match="run\\(\\) called again"):
        second = experiment.run()
    assert second.run_id != first.run_id
    assert experiment.result is second
    assert len(experiment.store) == 2
    assert experiment.store.get(first.run_id).run_id == first.run_id


def test_p5_custom_reporter_chain(tmp_path: Path) -> None:
    """P5: report(reporter=...) runs the custom chain instead of the
    default one (no stdout Markdown, no default JSON file)."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    result = experiment.run()
    json_path = tmp_path / "custom.json"
    chain = MarkdownReporter(include_config=False)
    artifact = experiment.report(reporter=chain)
    assert isinstance(artifact, str)
    assert "Experiment report" in artifact
    assert not (Path(result.model_path).parent / DEFAULT_REPORT_JSON_NAME).exists()

    composite_artifacts = experiment.report(reporter=JSONReporter(json_path))
    assert composite_artifacts == json_path
    assert json_path.exists()


def test_p6_custom_reporter_may_return_none(tmp_path: Path) -> None:
    """P6: a documented degrading reporter returns ``None`` through the
    facade instead of raising."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()

    class _DegradingReporter(Reporter[ReporterArtifact | None]):
        """The MLflow-style "reported-failed" policy: ``None`` artifact."""

        def report(self, result: ExperimentResult) -> ReporterArtifact | None:
            return None

    assert experiment.report(reporter=_DegradingReporter()) is None


# ────────────────────────── negative scenarios ────────────────────────── #


@pytest.mark.parametrize(
    "method",
    ["leaderboard", "best_model", "report", "export", "explain"],
)
def test_n1_methods_before_run_raise_state_error(tmp_path: Path, method: str) -> None:
    """N1: every result accessor before run() → a clear state error."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    with pytest.raises(ExperimentNotRunError, match="has not been run"):
        if method == "export":
            experiment.export(tmp_path)
        else:
            getattr(experiment, method)()
    # The state is observable through the public result property.
    assert experiment.result is None


def test_n1_leaderboard_summary_before_run_raises_state_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """N1: ``leaderboard(summary=True)`` before run() → the state error."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    with pytest.raises(ExperimentNotRunError, match="has not been run"):
        experiment.leaderboard(summary=True)
    assert capsys.readouterr().out == ""


def test_n1_not_run_error_is_runtime_error(tmp_path: Path) -> None:
    """N1: the state error stays catchable as RuntimeError (the documented
    'Experiment has not been run yet' contract)."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    with pytest.raises(RuntimeError, match="Experiment has not been run yet"):
        experiment.best_model()


def test_n2_invalid_config_type_propagates(tmp_path: Path) -> None:
    """N2: an unsupported config type → the legacy TypeError from run()."""
    experiment = AutoMLExperiment(123.45, _small_df())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported config type"):
        experiment.run()
    assert experiment.result is None


def test_n2_empty_df_propagates(tmp_path: Path) -> None:
    """N2: an empty DataFrame → the legacy ValueError from run()."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), pd.DataFrame())
    with pytest.raises(ValueError, match="empty"):
        experiment.run()
    assert experiment.result is None


def test_n2_missing_target_propagates(tmp_path: Path) -> None:
    """N2: a missing target column → the legacy ValueError from run()."""
    experiment = AutoMLExperiment(
        _config_dict(tmp_path), _small_df(), target="missing_col"
    )
    with pytest.raises(ValueError, match="Target column"):
        experiment.run()
    assert experiment.result is None


def test_n3_failed_run_registered_and_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N3: a run where every algorithm fails → the error propagates, a
    failed result is registered, and the state stays consistent (no
    model artifact, best_model() explains it)."""
    from configurable_automl_engine.tuner import InvalidAlgorithmError

    def failing_run(*args: Any, **kwargs: Any) -> Any:
        raise InvalidAlgorithmError("broken algorithm")

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.hpo_phases.run_optimization",
        failing_run,
    )
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
        experiment.run()
    result = experiment.result
    assert result is not None
    assert result.status is RunStatus.failed
    assert result.winner is None
    assert len(experiment.store) == 1
    assert experiment.store.get(result.run_id).status is RunStatus.failed
    with pytest.raises(RuntimeError, match="No model artifact available"):
        experiment.best_model()
    # The failed run still reports its history through the leaderboard
    # (an empty board is not an error).
    assert len(experiment.leaderboard()) == 0


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_single_algorithm_single_phase(tmp_path: Path) -> None:
    """B1: one algorithm and one HPO phase + terminal select."""
    cfg = _config_dict(tmp_path)
    cfg["algorithms"] = {
        "ridge": {
            "enable": True,
            "limit_hyperparameters": True,
            "hyperparameters": {"alpha": [0.1, 1.0]},
        }
    }
    experiment = AutoMLExperiment(cfg, _small_df())
    result = experiment.run()
    assert result.status is RunStatus.completed
    assert result.winner == "ridge"
    assert {r.algorithm for r in result.runs} == {"ridge"}
    assert experiment.best_model().algorithm == "ridge"


def test_b2_empty_phases_failed_run(tmp_path: Path) -> None:
    """B2: phases: [] → RuntimeError from run(), failed status in store."""
    experiment = AutoMLExperiment(_config_dict(tmp_path, phases=[]), _small_df())
    with pytest.raises(RuntimeError, match="cannot select a winner"):
        experiment.run()
    result = experiment.result
    assert result is not None
    assert result.status is RunStatus.failed
    assert result.error and "cannot select a winner" in result.error


def test_b3_target_only_frame(tmp_path: Path) -> None:
    """B3: a frame with the target column and no features → the run
    fails with a clear engine error (no features to train on), the failed
    state is recorded."""
    df = _small_df()[["target"]]
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    with pytest.raises((RuntimeError, ValueError)):
        experiment.run()
    assert experiment.result is None or experiment.result.status is RunStatus.failed


def test_b4_path_config(tmp_path: Path) -> None:
    """B4: a pathlib.Path config loads like the string form."""
    experiment = AutoMLExperiment(_yaml_config(tmp_path), _small_df())
    result = experiment.run()
    assert result.status is RunStatus.completed


def test_b5_full_hpo_failure_failed_status_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5: a full HPO failure → failed status, no model artifact, and the
    store keeps the failed run."""
    from configurable_automl_engine.tuner import InvalidAlgorithmError

    def failing_run(*args: Any, **kwargs: Any) -> Any:
        raise InvalidAlgorithmError("fatal")

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.hpo_phases.run_optimization",
        failing_run,
    )
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    with pytest.raises(RuntimeError):
        experiment.run()
    assert experiment.result is not None
    assert experiment.result.status is RunStatus.failed
    assert experiment.result.model_path is None
    assert not (tmp_path / "model.pkl").exists()


def test_b6_parallel_experiments_share_no_state(tmp_path: Path) -> None:
    """B6: two experiments in different threads keep separate results and
    stores (no global state leaks between them)."""
    results: dict[str, Any] = {}
    errors: list[Exception] = []

    def worker(name: str) -> None:
        try:
            experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
            results[name] = experiment.run()
        except Exception as err:  # noqa: BLE001 - collected and asserted below
            errors.append(err)

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert results["a"].run_id != results["b"].run_id
    assert results["a"].winner is not None
    assert results["b"].winner is not None


def test_data_snapshot_semantics(tmp_path: Path) -> None:
    """Documented semantics: mutating the caller's frame object after
    run() does not change the run result (the frame is snapshotted at
    run() time); in-place value writes still write through, as with any
    pandas copy."""
    df = _small_df()
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    result = experiment.run()
    assert result.status is RunStatus.completed
    # Structural mutation of the caller's frame does not affect the
    # already-finished run (the config snapshot of the result is built
    # at run() time from the same frame object).
    del df["f2"]
    assert result.status is RunStatus.completed
    assert result.winner is not None
    # The result model is an immutable snapshot of what was trained: it
    # never re-reads the caller's frame.
    assert experiment.result is result


def test_store_injection_shared_history(tmp_path: Path) -> None:
    """A caller-provided store receives every run of the experiment."""
    store = ExperimentStore()
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df(), store=store)
    result = experiment.run()
    assert len(store) == 1
    assert store.get(result.run_id) == result


# ────────────────────────── review findings ────────────────────────── #


def test_r1_stranger_result_never_adopted(tmp_path: Path) -> None:
    """Review B-1 (sequential): a fresh experiment after a stranger's
    completed engine run never adopts the stranger's result from the
    engine's result registry."""
    stranger = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    good = stranger.run()
    assert good.status is RunStatus.completed

    # The registry holds the stranger's result.
    assert get_last_experiment_result() is good

    bad = AutoMLExperiment(_config_dict(tmp_path), pd.DataFrame())
    with pytest.raises(ValueError, match="empty"):
        bad.run()
    # The failed-before-run experiment stays result-less: the registry
    # result was NOT adopted (its owner stamp belongs to the stranger).
    assert bad.result is None
    with pytest.raises(ExperimentNotRunError):
        bad.leaderboard()
    # The stranger's result is not in the failed experiment's store.
    assert len(bad.store) == 0


def test_r1_concurrent_stranger_result_never_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review B-1 (concurrent): when the registry is overwritten by a
    stranger's stamped result before this experiment adopts it, the
    experiment NEVER adopts the stranger's result — it reports the
    unknown-outcome state instead of serving foreign data."""
    import uuid as _uuid

    from configurable_automl_engine.training_engine.run_result import (
        result_owned_by,
    )

    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    import configurable_automl_engine.training_engine.component as component_mod

    real_train = component_mod.train_best_model

    def interleaved_train(*args: Any, **kwargs: Any) -> Any:
        value = real_train(*args, **kwargs)
        # A concurrent experiment finished while ours was running: the
        # registry now carries the stranger's (stamped) result — our own
        # result was already overwritten there (the worst-case race).
        own_result = get_last_experiment_result()
        assert own_result is not None
        stranger_result = ExperimentResult.model_validate(
            own_result.model_dump(mode="json")
        )
        publish_experiment_result(stranger_result, run_token=_uuid.uuid4())
        return value

    monkeypatch.setattr(component_mod, "train_best_model", interleaved_train)
    with pytest.raises(ExperimentNotRunError, match="No experiment result recorded"):
        experiment.run()
    # Nothing foreign was adopted into the experiment's state.
    assert experiment.result is None
    assert len(experiment.store) == 0
    # The registry itself carries the stranger's stamp, not ours.
    latest = get_last_experiment_result()
    assert latest is not None
    assert experiment._run_token is not None
    assert not result_owned_by(latest, experiment._run_token)


def test_r2_store_failure_never_masks_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review B-2: a failing store registration never replaces the run's
    own error — a failed run after a successful one propagates the
    engine's ValueError, the successful result stays adopted."""
    store = ExperimentStore()
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df(), store=store)
    first = experiment.run()
    assert first.status is RunStatus.completed

    def broken_add(_result: ExperimentResult, **_kwargs: Any) -> None:
        from configurable_automl_engine.experiment.store import DuplicateRunError

        raise DuplicateRunError("simulated duplicate registration")

    monkeypatch.setattr(experiment._store, "add", broken_add)
    # A failed run on a shared store: the engine's error must win.
    failed = AutoMLExperiment(_config_dict(tmp_path), pd.DataFrame(), store=store)
    with pytest.raises(ValueError, match="empty"):
        failed.run()
    # And the successful experiment's repeat-run error propagates too
    # (the registration failure is logged, not raised).
    with pytest.warns(RuntimeWarning, match="run\\(\\) called again"):
        second = experiment.run()
    assert second.status is RunStatus.completed
    assert second.run_id != first.run_id
    # The second run was adopted despite the broken store.
    assert experiment.result is second


def test_r3_run_result_carries_owner_stamp(tmp_path: Path) -> None:
    """R3: the returned result carries this experiment's owner stamp (the
    engine-side token contract, the B-1 fix mechanism)."""
    import uuid as _uuid

    from configurable_automl_engine.training_engine.run_result import (
        result_owned_by,
    )

    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    result = experiment.run()
    assert experiment._run_token is not None
    assert result_owned_by(result, experiment._run_token)
    # A stranger's token does not own it.
    assert not result_owned_by(result, _uuid.uuid4())


def test_r4_repeated_failed_run_keeps_previous_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: a repeated run that fails before reaching the engine does not
    adopt the previous run's result (each run has a fresh owner stamp)."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    first = experiment.run()
    assert experiment.result is first

    adopted: list[ExperimentResult] = []

    def recording_add(result: ExperimentResult, **_kwargs: Any) -> None:
        adopted.append(result)
        raise AssertionError("a failed-before-engine run must not adopt a result")

    monkeypatch.setattr(experiment._store, "add", recording_add)
    # The repeat fails during data validation, before the engine publishes
    # anything: the registry still holds ``first`` with the old token.
    experiment._data = pd.DataFrame()
    with pytest.warns(RuntimeWarning, match="run\\(\\) called again"):
        with pytest.raises(ValueError, match="empty"):
            experiment.run()

    assert adopted == []
    assert experiment.result is first
    assert len(experiment.store) == 1
