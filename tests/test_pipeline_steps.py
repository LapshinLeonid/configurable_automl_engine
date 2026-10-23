"""
Unit tests of the decomposed pipeline steps of ``train_best_model``.

They cover the functions extracted from the monolithic ``train_best_model``:
``load_config``, ``prepare_dataset``, ``execute_phases``, ``persist_artifact``,
as well as the orchestration order of the steps in ``train_best_model`` itself.

Scenarios:
    • load_config — positive (dict / Config / path), negative (config type,
      empty DataFrame, missing target column), edge (default target,
      logging setup);
    • prepare_dataset — X/y split, metric aliases, resolution of 'auto' →
      k_fold/train_test_split, column type detection;
    • execute_phases — a single HPO phase plus a terminal select,
      ``PhaseContext`` (working set, circuit breaker, terminal select winner),
      refine with warm-start initial_params for the whole input set,
      disqualification (circuit breaker), filtering of invalid scores, empty
      phases, missing valid results;
    • persist_artifact — result assembly, score semantics,
      model_path_override, additional_metrics, ``sanity_gate`` sections of
      the select steps, fit error propagation;
    • train_best_model — orchestration of the steps in the right order,
      protection against empty results and an invalid winner score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import configurable_automl_engine.training_engine.component as component
from configurable_automl_engine.training_engine.component import (
    PhaseContext,
    PhaseResult,
    PreparedData,
    execute_phases,
    load_config,
    persist_artifact,
    prepare_dataset,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    ValidationStrategy,
)
from configurable_automl_engine.tuner import HPO_WORST_SCORE, InvalidAlgorithmError

# Terminal select step: the last phase of any valid config.
TERMINAL_SELECT: dict[str, Any] = {
    "name": "final_select",
    "n_trials": 1,
    "action": "select",
    "strategy": "cv_best",
    "output": "winner",
}


# ──────────────────────────────────────────────────────────────────────────────
# Common helpers
# ──────────────────────────────────────────────────────────────────────────────
# Terminal select step: the last phase of any valid config.
TERMINAL_SELECT: dict[str, Any] = {
    "name": "final_select",
    "n_trials": 1,
    "action": "select",
    "strategy": "cv_best",
    "output": "winner",
}


def _hpo_only_cfg_dict(phases: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Dict of a valid config truncated to HPO phases.

    The config schema requires the terminal select as the last phase.
    Unit tests of ``execute_phases`` add the terminal select only to pass
    validation and then truncate it (mutating after validation is safe:
    ``execute_phases`` reads only ``general.phases``).
    """
    return _cfg_dict(phases=list(phases) + [TERMINAL_SELECT], **kwargs)


def _hpo_only_cfg(phases: list[dict[str, Any]], **kwargs: Any) -> Config:
    """Valid ``Config`` truncated to HPO phases (see ``_hpo_only_cfg_dict``)."""
    cfg = Config.model_validate(_hpo_only_cfg_dict(phases, **kwargs))
    cfg.general.phases = cfg.general.phases[:-1]
    return cfg


def _cfg_dict(
    *,
    phases: list[dict[str, Any]],
    comparison_metric: str = "mae",
    validation_strategy: str = "train_test_split",
    n_folds: int = 5,
    algorithms: list[str] | None = None,
    additional_metrics: list[str] | None = None,
    log_to_file: str | None = None,
    terminal_select: bool = True,
) -> dict[str, Any]:
    """Build a valid configuration dict for tests.

    By default the terminal select (``output: winner``) is appended as the
    last phase — a requirement of the schema; tests that check the schema
    itself can disable it with ``terminal_select=False``.
    """
    if (
        terminal_select
        and phases
        and not (
            phases[-1].get("action") == "select"
            and phases[-1].get("output") == "winner"
        )
    ):
        phases = [*phases, dict(TERMINAL_SELECT)]
    algos: dict[str, Any] = {}
    for name in algorithms or ["random_forest"]:
        algos[name] = {
            "enable": True,
            "tuner": "mock.mock_tuner",
            "trainer_module": "mock.mock_trainer",
        }
    return {
        "general": {
            "comparison_metric": comparison_metric,
            "validation_strategy": validation_strategy,
            "n_folds": n_folds,
            "phases": phases,
            "path_to_model": "model.pkl",
            "log_to_file": log_to_file,
            "additional_metrics": additional_metrics or [],
        },
        "algorithms": algos,
        "oversampling": {"data_oversampling": False},
    }


def _df(n_rows: int = 10, n_features: int = 2) -> pd.DataFrame:
    """Small numeric DataFrame with a 'target' column."""
    data: dict[str, Any] = {f"f{i}": list(range(n_rows)) for i in range(n_features)}
    data["target"] = [float(i % 3) for i in range(n_rows)]
    return pd.DataFrame(data)


def _prepared(**overrides: Any) -> PreparedData:
    """Build a minimal PreparedData with defaults filled in."""
    defaults: dict[str, Any] = {
        "X": pd.DataFrame({"a": [1.0, 2.0]}),
        "y": pd.Series([1.0, 2.0]),
        "metric_user": "mae",
        "metric_sklearn": "mae",
        "resolved_validation": ValidationStrategy.train_test_split,
        "resolved_n_folds": 5,
        "resolved_test_size": 0.2,
        "categorical_features": [],
        "numerical_features": ["a"],
    }
    defaults.update(overrides)
    return PreparedData(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# load_config
# ──────────────────────────────────────────────────────────────────────────────
def test_load_config_from_dict():
    cfg, target_col = load_config(
        _cfg_dict(phases=[{"name": "p", "n_trials": 1}]), _df(), target="target"
    )
    assert isinstance(cfg, Config)
    assert target_col == "target"


def test_load_config_default_target_column():
    _, target_col = load_config(
        _cfg_dict(phases=[{"name": "p", "n_trials": 1}, TERMINAL_SELECT]), _df()
    )
    assert target_col == "target"


def test_load_config_explicit_target_column():
    df = _df()
    df["my_target"] = df["target"]
    cfg, target_col = load_config(
        _cfg_dict(phases=[{"name": "p", "n_trials": 1}, TERMINAL_SELECT]),
        df,
        target="my_target",
    )
    assert isinstance(cfg, Config)
    assert target_col == "my_target"


def test_load_config_accepts_config_object():
    cfg_obj = Config.model_validate(
        _cfg_dict(phases=[{"name": "p", "n_trials": 1}, TERMINAL_SELECT])
    )
    cfg, target_col = load_config(cfg_obj, _df(), target="target")
    assert cfg is cfg_obj  # the Config object is not re-created
    assert target_col == "target"


def test_load_config_from_str_path(mocker_patch_read_config):
    cfg_mock, read_mock = mocker_patch_read_config
    cfg, _ = load_config("cfg.yaml", _df(), target="target")
    assert cfg is cfg_mock
    read_mock.assert_called_once_with("cfg.yaml")


def test_load_config_from_path_object(mocker_patch_read_config):
    cfg_mock, read_mock = mocker_patch_read_config
    cfg, _ = load_config(Path("cfg.yaml"), _df(), target="target")
    assert cfg is cfg_mock
    read_mock.assert_called_once_with(Path("cfg.yaml"))


def test_load_config_unsupported_type_raises_type_error():
    with pytest.raises(TypeError, match="Unsupported config type"):
        load_config(123.45, _df(), target="target")


def test_load_config_empty_df_raises_value_error():
    with pytest.raises(ValueError, match="Input dataframe is empty"):
        load_config(_cfg_dict(phases=[]), pd.DataFrame(), target="target")


def test_load_config_missing_target_raises_value_error():
    with pytest.raises(ValueError, match="Target column 'missing' not found"):
        load_config(_cfg_dict(phases=[]), _df(), target="missing")


def test_load_config_sets_up_logging_when_configured():
    with patch.object(component, "setup_logging") as mock_setup:
        load_config(
            _cfg_dict(phases=[], log_to_file="logs/test.log"),
            _df(),
            target="target",
        )
    mock_setup.assert_called_once_with(Path("logs/test.log"))


def test_load_config_does_not_setup_logging_by_default():
    with patch.object(component, "setup_logging") as mock_setup:
        load_config(_cfg_dict(phases=[]), _df(), target="target")
    mock_setup.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# prepare_dataset
# ──────────────────────────────────────────────────────────────────────────────
def test_prepare_dataset_splits_x_y_and_metric_names():
    df = _df(n_rows=10)
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = prepare_dataset(cfg, df, "target")

    assert isinstance(prepared, PreparedData)
    assert prepared.X.shape == (10, 2)
    assert len(prepared.y) == 10
    assert "target" not in prepared.X.columns
    assert prepared.metric_user == "mae"
    assert prepared.metric_sklearn == "mae"
    assert prepared.resolved_validation == ValidationStrategy.train_test_split
    assert prepared.resolved_test_size == 0.2


def test_prepare_dataset_resolves_metric_alias_rmse():
    cfg = Config.model_validate(_cfg_dict(phases=[], comparison_metric="rmse"))
    prepared = prepare_dataset(cfg, _df(), "target")
    assert prepared.metric_user == "rmse"
    assert prepared.metric_sklearn == "neg_root_mean_squared_error"


def test_prepare_dataset_resolves_kfold_fold_count():
    # 100 rows, 3 features, explicit k_fold with n_folds=3
    cfg = Config.model_validate(
        _cfg_dict(phases=[], validation_strategy="k_fold", n_folds=3)
    )
    prepared = prepare_dataset(cfg, _df(n_rows=100, n_features=3), "target")
    assert prepared.resolved_validation == ValidationStrategy.k_fold
    assert prepared.resolved_n_folds == 3


def test_prepare_dataset_resolves_auto_to_kfold():
    # auto on 100x3: choose_validation_method selects kfold with k=5
    cfg = Config.model_validate(
        _cfg_dict(phases=[], validation_strategy="auto", n_folds=5)
    )
    prepared = prepare_dataset(cfg, _df(n_rows=100, n_features=3), "target")
    assert prepared.resolved_validation == ValidationStrategy.k_fold
    assert prepared.resolved_n_folds == 5
    assert prepared.resolved_test_size == 0.2


def test_prepare_dataset_resolves_auto_mid_size_to_fixed_kfold():
    # auto on 300×3: previously a hold-out of 30 rows; now a
    # fixed 5-fold CV so that model comparison is not random.
    cfg = Config.model_validate(
        _cfg_dict(phases=[], validation_strategy="auto", n_folds=5)
    )
    prepared = prepare_dataset(cfg, _df(n_rows=300, n_features=3), "target")
    assert prepared.resolved_validation == ValidationStrategy.k_fold
    assert prepared.resolved_n_folds == 5


def test_prepare_dataset_resolves_auto_large_to_train_test_split():
    # auto on "large" data (N >= 50000) resolves to train_test_split with an
    # integer test_size from choose_validation_method. The heuristic itself is
    # covered in test_validation_auto.py on numbers — here make_cv is patched
    # to avoid creating a 50k-row DataFrame; we check that the resolved values
    # are forwarded to PreparedData.
    cfg = Config.model_validate(
        _cfg_dict(phases=[], validation_strategy="auto", n_folds=5)
    )
    fake_decision = {
        "method": "train_test_split",
        "test_size": 500,
        "train_size": 49500,
    }
    with patch.object(
        component,
        "make_cv",
        return_value=("train_test_split", None, fake_decision),
    ):
        prepared = prepare_dataset(cfg, _df(n_rows=300, n_features=3), "target")
    assert prepared.resolved_validation == ValidationStrategy.train_test_split
    assert prepared.resolved_test_size == 500


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_prepare_dataset_rejects_non_finite_target(bad_value):
    """prepare_dataset fails fast on a non-finite target (issue #145).

    A NaN/±inf target must be rejected at the pipeline entry point: otherwise
    the HPO budget is spent on randomly "clean" folds and the final training
    crashes with an unclear error.
    """
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    df = _df()
    df.loc[0, "target"] = bad_value
    with pytest.raises(ValueError, match="non-finite"):
        prepare_dataset(cfg, df, "target")


def test_prepare_dataset_rejects_non_numeric_target():
    """prepare_dataset fails fast on a non-numeric target column."""
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    df = _df()
    df["target"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    with pytest.raises(ValueError, match="numeric"):
        prepare_dataset(cfg, df, "target")


def test_prepare_dataset_detects_feature_types():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "flag": [True, False, True],
            "target": [0, 1, 0],
        }
    )
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = prepare_dataset(cfg, df, "target")
    assert sorted(prepared.categorical_features) == ["cat", "flag"]
    assert prepared.numerical_features == ["num"]


# ──────────────────────────────────────────────────────────────────────────────
# execute_phases
# ──────────────────────────────────────────────────────────────────────────────
def test_execute_phases_single_phase_all_algorithms_with_terminal_select():
    """HPO phase + terminal select: the working set is replaced by the winner."""
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[{"name": "coarse", "n_trials": 2, "action": "all_algorithms"}],
            algorithms=["random_forest", "extra_trees"],
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        scores = {"random_forest": 0.9, "extra_trees": 0.8}
        return PhaseResult(score=scores[kw["algo_name"]], params={"p": 1})

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo) as mock_hpo:
        ctx = execute_phases(cfg, prepared)

    assert isinstance(ctx, PhaseContext)
    # The terminal select (cv_best, winner) replaces the working set
    # with a single winner entry.
    assert set(ctx.working_set) == {"random_forest"}
    assert ctx.working_set["random_forest"] == PhaseResult(score=0.9, params={"p": 1})
    assert ctx.winner_algo == "random_forest"
    assert ctx.disqualified_algorithms == {}
    assert ctx.sanity_gate_sections == {}
    assert ctx.pre_trained_trainer is None
    assert mock_hpo.call_count == 2
    # The HPO phase name travels to _run_hpo → tuner.optimize(phase_name=...)
    # for the pooled-fallback WARNING deduplication (issue #136).
    assert all(call.kwargs["phase_name"] == "coarse" for call in mock_hpo.call_args_list)


def test_execute_phases_select_cv_best_top_k_replaces_working_set():
    """select(cv_best, top_k) keeps the k best by CV score,
    after which the terminal select picks the winner from them."""
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[
                {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "top1",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "top_k",
                    "k": 1,
                },
            ],
            algorithms=["random_forest", "extra_trees"],
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        scores = {"random_forest": 0.9, "extra_trees": 0.8}
        return PhaseResult(score=scores[kw["algo_name"]], params={"p": 1})

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        ctx = execute_phases(cfg, prepared)

    # top_k replaced the working set, and the terminal select narrowed it
    # to the winner (the CV leader of top_k).
    assert set(ctx.working_set) == {"random_forest"}
    assert ctx.winner_algo == "random_forest"


def test_execute_phases_refine_uses_initial_params_for_all_candidates():
    """refine after all_algorithms — a "second round for everyone".

    Each candidate of the input set receives initial_params from ITS OWN
    record of the previous phase (warm start, enqueue_trial); there is no
    narrowing: both algorithms take part in refine and the output replaces
    the working set.
    """
    cfg = _hpo_only_cfg(
        [
            {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
            {"name": "fine", "n_trials": 1, "action": "refine"},
        ],
        algorithms=["random_forest", "extra_trees"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    calls: list[tuple[str, Any]] = []

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        algo = kw["algo_name"]
        calls.append((algo, kw.get("initial_params")))
        scores = {"random_forest": 0.9, "extra_trees": 0.8}
        return PhaseResult(score=scores[algo], params={"p": 1})

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        ctx = execute_phases(cfg, prepared)

    # Phase 1: both algorithms without initial_params; phase 2: both algorithms
    # with phase 1 parameters (warm start for the whole input set).
    assert calls == [
        ("random_forest", None),
        ("extra_trees", None),
        ("random_forest", {"p": 1}),
        ("extra_trees", {"p": 1}),
    ]
    # Same names, updated records: the output replaces the working set.
    assert set(ctx.working_set) == {"random_forest", "extra_trees"}
    assert ctx.working_set["random_forest"] == PhaseResult(score=0.9, params={"p": 1})
    assert ctx.working_set["extra_trees"] == PhaseResult(score=0.8, params={"p": 1})
    assert ctx.disqualified_algorithms == {}


def test_execute_phases_refine_after_select_warm_starts_from_select_output():
    """refine after a select step: warm start from the select output.

    The refine input is the previous select's output (top_k): initial_params
    are taken from the working set records; there is no narrowing.
    """
    cfg = _hpo_only_cfg(
        [
            {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
            {
                "name": "top1",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "top_k",
                "k": 1,
            },
            {"name": "fine", "n_trials": 1, "action": "refine"},
        ],
        algorithms=["random_forest", "extra_trees"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    calls: list[tuple[str, Any]] = []

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        algo = kw["algo_name"]
        calls.append((algo, kw.get("initial_params")))
        scores = {"random_forest": 0.9, "extra_trees": 0.8}
        return PhaseResult(score=scores[algo], params={"p": 1})

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        ctx = execute_phases(cfg, prepared)

    # select(cv_best, top_k, k=1) left only the CV leader; refine
    # retrains it with a warm start from the select output parameters.
    assert calls == [
        ("random_forest", None),
        ("extra_trees", None),
        ("random_forest", {"p": 1}),
    ]
    assert set(ctx.working_set) == {"random_forest"}


def test_execute_phases_refine_empty_working_set_raises():
    """refine with an empty input working set → RuntimeError.

    As the first phase, refine receives the full set of enabled algorithms;
    if there are no enabled algorithms, the working set is empty and the
    phase ends with the canonical RuntimeError (protection against a silently
    "successful" empty result).
    """
    cfg = _hpo_only_cfg(
        [{"name": "fine", "n_trials": 1, "action": "refine"}],
        algorithms=["random_forest"],
    )
    # All algorithms are disabled -> the refine input set is empty.
    cfg.algorithms.random_forest.enable = False
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    with patch.object(component, "_run_hpo", return_value=PhaseResult(score=0.9, params={})):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            execute_phases(cfg, prepared)


def test_execute_phases_empty_phases_returns_empty_context():
    cfg = Config.model_validate(_cfg_dict(phases=[], terminal_select=False))
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")
    ctx = execute_phases(cfg, prepared)
    assert ctx.working_set == {}
    assert ctx.disqualified_algorithms == {}
    assert ctx.winner_algo is None


def test_execute_phases_no_valid_scores_raises_runtime_error():
    cfg = _hpo_only_cfg([{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}])
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    # All trials are unsuccessful (best_score=None -> _run_hpo returns None)
    with patch.object(component, "_run_hpo", return_value=None):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            execute_phases(cfg, prepared)


def test_execute_phases_filters_worst_score_sentinel():
    cfg = _hpo_only_cfg(
        [{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}],
        algorithms=["random_forest", "extra_trees"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        if kw["algo_name"] == "random_forest":
            return PhaseResult(score=HPO_WORST_SCORE, params={"p": 1})  # sentinel -> excluded
        return PhaseResult(score=0.7, params={"p": 2})

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        ctx = execute_phases(cfg, prepared)

    assert set(ctx.working_set) == {"extra_trees"}
    assert ctx.disqualified_algorithms == {}


def test_execute_phases_disqualifies_invalid_algorithm():
    cfg = _hpo_only_cfg(
        [{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}],
        algorithms=["random_forest", "extra_trees"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        if kw["algo_name"] == "random_forest":
            raise InvalidAlgorithmError("incompatible with data")
        return PhaseResult(score=0.7, params={"p": 2})

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        ctx = execute_phases(cfg, prepared)

    assert set(ctx.working_set) == {"extra_trees"}
    assert "random_forest" in ctx.disqualified_algorithms
    assert "incompatible with data" in ctx.disqualified_algorithms["random_forest"]


def test_execute_phases_all_invalid_algorithms_raise():
    cfg = _hpo_only_cfg(
        [{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}],
        algorithms=["random_forest", "extra_trees"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    with patch.object(component, "_run_hpo", side_effect=InvalidAlgorithmError("bad")):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            execute_phases(cfg, prepared)


# ──────────────────────────────────────────────────────────────────────────────
# persist_artifact
# ──────────────────────────────────────────────────────────────────────────────
def test_persist_artifact_basic_result():
    cfg = Config.model_validate(_cfg_dict(phases=[], algorithms=["random_forest"]))
    prepared = _prepared(metric_user="mae", metric_sklearn="mae")
    trainer_mock = MagicMock()
    trainer_mock.additional_scores = {"r2": 0.7}

    with patch.object(
        component, "_fit_and_save", return_value=trainer_mock
    ) as mock_fit:
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=-1.5,
            final_params={"alpha": 1.0},
            model_path_override=None,
            disqualified_algorithms={},
        )

    assert result["algorithm"] == "random_forest"
    # For mae, the "raw" -1.5 is inverted into the natural positive value
    assert result["score"] == 1.5
    assert result["metric"] == "mae"
    assert result["params"] == {"alpha": 1.0}
    assert result["model_path"] == "model.pkl"
    assert "disqualified_algorithms" not in result
    assert "additional_metrics" not in result
    assert "sanity_gate" not in result

    mock_fit.assert_called_once()
    kwargs = mock_fit.call_args
    assert kwargs.kwargs["metric_name_sklearn"] == "mae"
    assert kwargs.kwargs["validation_strategy"] == ValidationStrategy.train_test_split
    assert kwargs.kwargs["n_folds"] == 5
    assert kwargs.kwargs["test_size"] == 0.2


def test_persist_artifact_r2_score_not_inverted():
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared(metric_user="r2", metric_sklearn="r2")
    with patch.object(component, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.85,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
        )
    assert result["score"] == 0.85


def test_persist_artifact_model_path_override(tmp_path: Path):
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared()
    override = tmp_path / "custom" / "best.pkl"
    with patch.object(component, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=override,
            disqualified_algorithms={},
        )
    assert result["model_path"] == str(override)


def test_persist_artifact_adds_disqualified_and_additional_metrics():
    cfg = Config.model_validate(_cfg_dict(phases=[], additional_metrics=["r2"]))
    prepared = _prepared()
    trainer_mock = MagicMock()
    trainer_mock.additional_scores = {"r2": 0.75}
    with patch.object(component, "_fit_and_save", return_value=trainer_mock):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={"elasticnet": "bad tuner"},
        )
    assert result["disqualified_algorithms"] == {"elasticnet": "bad tuner"}
    assert result["additional_metrics"] == {"r2": 0.75}


def test_persist_artifact_disqualified_copy_is_independent():
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared()
    disqualified = {"elasticnet": "boom"}
    with patch.object(component, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms=disqualified,
        )
    disqualified["elasticnet"] = "changed"
    assert result["disqualified_algorithms"] == {"elasticnet": "boom"}


def test_persist_artifact_propagates_fit_failure():
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared()
    with patch.object(
        component, "_fit_and_save", side_effect=RuntimeError("Disk full")
    ):
        with pytest.raises(RuntimeError, match="Disk full"):
            persist_artifact(
                cfg=cfg,
                prepared=prepared,
                winner_algo="random_forest",
                final_score=0.9,
                final_params={},
                model_path_override=None,
                disqualified_algorithms={},
            )


def test_persist_artifact_pre_trained_trainer_skips_retraining(tmp_path: Path):
    """pre_trained_trainer: the artifact is saved without retraining."""
    cfg = Config.model_validate(_cfg_dict(phases=[], algorithms=["random_forest"]))
    prepared = _prepared()
    trainer_mock = MagicMock()
    trainer_mock.additional_scores = {}
    override = tmp_path / "best.pkl"

    with patch.object(component, "_fit_and_save") as mock_fit:
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=override,
            disqualified_algorithms={},
            pre_trained_trainer=trainer_mock,
        )

    mock_fit.assert_not_called()  # no retraining of the winner is needed
    trainer_mock.save.assert_called_once_with(override)
    assert result["model_path"] == str(override)


# ──────────────────────────────────────────────────────────────────────────────
# persist_artifact: sanity_gate sections of the select steps
# ──────────────────────────────────────────────────────────────────────────────
def test_persist_artifact_adds_sanity_gate_sections():
    """Sections of the select steps end up in result['sanity_gate']."""
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared()
    trainer_mock = MagicMock()
    trainer_mock.additional_scores = {}
    with patch.object(component, "_fit_and_save", return_value=trainer_mock):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
            sanity_gate_sections={
                "gate_select": {
                    "strategy": "sanity_gate",
                    "pool": ["ridge", "elasticnet"],
                    "audit": {},
                    "disqualified": {"elasticnet": ["Circuit D: gap"]},
                    "fallback_used": False,
                    "winner_rmse_oof": 0.25,
                    "overhead_seconds": 1.25,
                }
            },
        )
    assert result["sanity_gate"] == {
        "gate_select": {
            "strategy": "sanity_gate",
            "pool": ["ridge", "elasticnet"],
            "audit": {},
            "disqualified": {"elasticnet": ["Circuit D: gap"]},
            "fallback_used": False,
            "winner_rmse_oof": 0.25,
            "overhead_seconds": 1.25,
        }
    }


def test_persist_artifact_sanity_sections_absent_by_default():
    """Without sections the ``sanity_gate`` key is absent (backward compatibility)."""
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared()
    with patch.object(component, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
        )
    assert "sanity_gate" not in result
    # The removed warn_only keys never appear again.
    for key in (
        "disqualified_by_sanity_gate",
        "sanity_gate_warn_only",
        "sanity_gate_overhead_seconds",
    ):
        assert key not in result


def test_persist_artifact_empty_sanity_sections_absent():
    """An empty sections dict does not add the ``sanity_gate`` key."""
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared()
    with patch.object(component, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
            sanity_gate_sections={},
        )
    assert "sanity_gate" not in result


def test_persist_artifact_sanity_sections_copy_is_independent():
    """The ``sanity_gate`` value is copied (mutations are not visible)."""
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = _prepared()
    sections = {"gate": {"pool": ["ridge"], "disqualified": {}}}
    with patch.object(component, "_fit_and_save", return_value=MagicMock()):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="random_forest",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
            sanity_gate_sections=sections,
        )
    sections["gate"]["pool"].append("mutated")
    assert result["sanity_gate"] == {"gate": {"pool": ["ridge"], "disqualified": {}}}


# ──────────────────────────────────────────────────────────────────────────────
# train_best_model: step orchestration
# ──────────────────────────────────────────────────────────────────────────────
def test_train_best_model_orchestrates_steps_in_order():
    """The winner and the trainer come from the terminal select context."""
    order: list[str] = []
    cfg = MagicMock()
    prepared = MagicMock()
    ctx = PhaseContext(
        working_set={"rf": PhaseResult(score=0.9, params={})},
        winner_algo="rf",
        pre_trained_trainer=None,
    )
    expected_result = {"algorithm": "rf"}

    def record(name: str, value: Any) -> Any:
        order.append(name)
        return value

    with (
        patch.object(
            component,
            "load_config",
            side_effect=lambda c, d, t=None: record("load_config", (cfg, "target")),
        ),
        patch.object(
            component,
            "prepare_dataset",
            side_effect=lambda c, d, t: record("prepare_dataset", prepared),
        ),
        patch.object(
            component,
            "execute_phases",
            side_effect=lambda c, p: record("execute_phases", ctx),
        ),
        patch.object(
            component,
            "persist_artifact",
            side_effect=lambda **kw: record("persist_artifact", expected_result),
        ) as persist_mock,
    ):
        out = train_best_model(config="x", df=MagicMock(), target="target")

    assert out is expected_result
    assert order == [
        "load_config",
        "prepare_dataset",
        "execute_phases",
        "persist_artifact",
    ]
    persist_kwargs = persist_mock.call_args.kwargs
    assert persist_kwargs["winner_algo"] == "rf"
    assert persist_kwargs["disqualified_algorithms"] == {}
    assert persist_kwargs["pre_trained_trainer"] is None
    assert persist_kwargs["sanity_gate_sections"] is None


def test_train_best_model_no_winner_skips_persist():
    """Empty context (e.g. phases: []) → RuntimeError without persist."""
    with (
        patch.object(component, "load_config", return_value=(MagicMock(), "target")),
        patch.object(component, "prepare_dataset", return_value=MagicMock()),
        patch.object(component, "execute_phases", return_value=PhaseContext()),
        patch.object(component, "persist_artifact") as mock_persist,
    ):
        with pytest.raises(RuntimeError, match="No valid results after HPO phases"):
            train_best_model(config={}, df=MagicMock())

    mock_persist.assert_not_called()


def test_train_best_model_invalid_winner_score_raises():
    """Winner-score invariant: the sentinel never reaches the report."""
    sentinel = float(np.finfo(np.float32).min)
    ctx = PhaseContext(working_set={"rf": PhaseResult(score=sentinel, params={})}, winner_algo="rf")
    with (
        patch.object(component, "load_config", return_value=(MagicMock(), "target")),
        patch.object(component, "prepare_dataset", return_value=MagicMock()),
        patch.object(component, "execute_phases", return_value=ctx),
        patch.object(component, "persist_artifact") as mock_persist,
    ):
        with pytest.raises(RuntimeError, match="invalid final score"):
            train_best_model(config={}, df=MagicMock())

    mock_persist.assert_not_called()


def test_train_best_model_passes_context_to_persist():
    """Disqualifications/trainer/context sections are forwarded to persist_artifact."""
    trainer = object()
    ctx = PhaseContext(
        working_set={"ridge": PhaseResult(score=-0.25, params={"alpha": 1.0})},
        disqualified_algorithms={"elasticnet": "Circuit A"},
        sanity_gate_sections={"gate": {"pool": ["ridge"]}},
        winner_algo="ridge",
        pre_trained_trainer=trainer,
    )
    expected = {"algorithm": "ridge"}
    with (
        patch.object(component, "load_config", return_value=(MagicMock(), "target")),
        patch.object(component, "prepare_dataset", return_value=MagicMock()),
        patch.object(component, "execute_phases", return_value=ctx),
        patch.object(
            component, "persist_artifact", return_value=expected
        ) as mock_persist,
    ):
        out = train_best_model(config={}, df=MagicMock(), target="target")

    assert out is expected
    kwargs = mock_persist.call_args.kwargs
    assert kwargs["winner_algo"] == "ridge"
    assert kwargs["disqualified_algorithms"] == {"elasticnet": "Circuit A"}
    assert kwargs["pre_trained_trainer"] is trainer
    assert kwargs["sanity_gate_sections"] == {"gate": {"pool": ["ridge"]}}


# ──────────────────────────────────────────────────────────────────────────────
# Fixture for load_config from a file
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mocker_patch_read_config():
    cfg_mock = MagicMock()
    cfg_mock.general.log_to_file = None
    with patch.object(component, "read_config", return_value=cfg_mock) as read_mock:
        yield cfg_mock, read_mock


# ──────────────────────────────────────────────────────────────────────────────
#  Refine: TPE history transfer (previous_trials) + no-fallback contract
# ──────────────────────────────────────────────────────────────────────────────
def _completed_trial() -> Any:
    """A single COMPLETED FrozenTrial to simulate the previous phase's history."""
    from optuna.distributions import CategoricalDistribution
    from optuna.trial import TrialState, create_trial

    return create_trial(
        value=0.9,
        params={"p": 1},
        distributions={"p": CategoricalDistribution(choices=[1])},
        state=TrialState.COMPLETE,
    )


def test_execute_phases_refine_passes_previous_trials():
    """refine after all_algorithms forwards the previous phase's own trials
    as previous_trials (the TPE history warm-start), in addition to
    initial_params (the monotonicity anchor)."""
    cfg = _hpo_only_cfg(
        [
            {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
            {"name": "fine", "n_trials": 1, "action": "refine"},
        ],
        algorithms=["random_forest", "extra_trees"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    calls: list[tuple[str, Any, Any]] = []
    history = _completed_trial()

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        algo = kw["algo_name"]
        prev = kw.get("previous_trials")
        calls.append((algo, kw.get("initial_params"), None if prev is None else list(prev)))
        return PhaseResult(score=0.9, params={"p": 1}, trials=[history])

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        execute_phases(cfg, prepared)

    # Phase 1 (all_algorithms): no warm-start; phase 2 (refine): the previous
    # phase's trials are passed as previous_trials.
    assert calls == [
        ("random_forest", None, None),
        ("extra_trees", None, None),
        ("random_forest", {"p": 1}, [history]),
        ("extra_trees", {"p": 1}, [history]),
    ]


def test_execute_phases_refine_total_failure_raises_with_history():
    """N1 (no fallback): if ALL NEW refine trials fail (produce no valid
    score) while transferred COMPLETED history is present, the working set is
    wiped out and RuntimeError is raised — no silent fallback to the
    pre-refine parameters."""
    cfg = _hpo_only_cfg(
        [
            {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
            {"name": "fine", "n_trials": 1, "action": "refine"},
        ],
        algorithms=["random_forest"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    def fake_run_hpo(**kw: Any) -> Any:
        if kw.get("initial_params") is None:
            # phase 1 succeeds and produces a COMPLETED trial (transferred history).
            return PhaseResult(score=0.9, params={"p": 1}, trials=[_completed_trial()])
        # refine: all new trials fail (None) despite the transferred history.
        return None

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            execute_phases(cfg, prepared)


def test_hpo_phases_use_different_sampler_seeds_but_fixed_folds():
    """B6/per-phase seeds: the TPE sampler seed of the first HPO phase
    (all_algorithms) is RANDOM_STATE (42), and each following HPO phase
    (refine) gets the next sampler seed (43, 44, ...). Select steps are not
    counted. The data-split/preprocessor seed (random_state) stays the fixed
    RANDOM_STATE for every phase (issue #119)."""
    cfg = _hpo_only_cfg(
        [
            {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
            {"name": "fine1", "n_trials": 1, "action": "refine"},
            {"name": "gate", "n_trials": 1, "action": "select",
             "strategy": "cv_best", "output": "top_k", "k": 1},
            {"name": "fine2", "n_trials": 1, "action": "refine"},
        ],
        algorithms=["random_forest", "extra_trees"],
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")
    sampler_seeds: list[int] = []
    split_seeds: list[int] = []

    def fake_run_hpo(**kw: Any) -> PhaseResult:
        sampler_seeds.append(kw["sampler_seed"])
        split_seeds.append(kw["random_state"])
        return PhaseResult(score=0.9, params={"p": 1})

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        execute_phases(cfg, prepared)

    # coarse: 2 algos × sampler seed 42; fine1 (refine): 2 algos × 43;
    # the gate select (top_k, k=1) narrows to 1 algo; fine2 (refine): 1 × 44.
    assert sampler_seeds == [42, 42, 43, 43, 44]
    # The folds/preprocessor seed is the fixed RANDOM_STATE for every phase.
    assert split_seeds == [42, 42, 42, 42, 42]
