"""
Юнит-тесты декомпозированных шагов пайплайна ``train_best_model`` (issue #31).

Покрывают функции, вынесенные из монолитного ``train_best_model``:
``load_config``, ``prepare_dataset``, ``execute_phases``, ``persist_artifact``,
а также порядок оркестрации шагов в самом ``train_best_model``.

Сценарии:
    • load_config — позитив (dict / Config / путь), негатив (тип конфига,
      пустой DataFrame, отсутствующая целевая колонка), edge (дефолтный
      target, настройка логирования);
    • prepare_dataset — split X/y, алиасы метрик, резолюция 'auto' →
      k_fold/train_test_split, детекция типов колонок;
    • execute_phases — одна фаза, refine_winner с initial_params,
      дисквалификация (circuit breaker, issue #12), фильтрация невалидных
      скоров (issue #13/#32), пустые фазы, отсутствие валидных результатов;
    • persist_artifact — сборка результата, семантика score (issue #26),
      model_path_override, additional_metrics, проброс ошибки fit;
    • train_best_model — оркестрация шагов в правильном порядке, защита от
      пустых результатов и невалидного winner-скора.
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


# ──────────────────────────────────────────────────────────────────────────────
# Общие помощники
# ──────────────────────────────────────────────────────────────────────────────
def _cfg_dict(
    *,
    phases: list[dict[str, Any]],
    comparison_metric: str = "mae",
    validation_strategy: str = "train_test_split",
    n_folds: int = 5,
    algorithms: list[str] | None = None,
    additional_metrics: list[str] | None = None,
    log_to_file: str | None = None,
) -> dict[str, Any]:
    """Собрать валидный словарь конфигурации для тестов."""
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
    """Небольшой числовой DataFrame с колонкой 'target'."""
    data: dict[str, Any] = {
        f"f{i}": list(range(n_rows)) for i in range(n_features)
    }
    data["target"] = [float(i % 3) for i in range(n_rows)]
    return pd.DataFrame(data)


def _prepared(**overrides: Any) -> PreparedData:
    """Собрать минимальный PreparedData с заполненными дефолтами."""
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
    cfg, target_col = load_config(_cfg_dict(phases=[{"name": "p", "n_trials": 1}]),
                                  _df(), target="target")
    assert isinstance(cfg, Config)
    assert target_col == "target"


def test_load_config_default_target_column():
    _, target_col = load_config(
        _cfg_dict(phases=[{"name": "p", "n_trials": 1}]), _df()
    )
    assert target_col == "target"


def test_load_config_explicit_target_column():
    df = _df()
    df["my_target"] = df["target"]
    cfg, target_col = load_config(
        _cfg_dict(phases=[{"name": "p", "n_trials": 1}]), df, target="my_target"
    )
    assert isinstance(cfg, Config)
    assert target_col == "my_target"


def test_load_config_accepts_config_object():
    cfg_obj = Config.model_validate(
        _cfg_dict(phases=[{"name": "p", "n_trials": 1}])
    )
    cfg, target_col = load_config(cfg_obj, _df(), target="target")
    assert cfg is cfg_obj  # объект Config не пересоздаётся
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
    cfg = Config.model_validate(
        _cfg_dict(phases=[], comparison_metric="rmse")
    )
    prepared = prepare_dataset(cfg, _df(), "target")
    assert prepared.metric_user == "rmse"
    assert prepared.metric_sklearn == "neg_root_mean_squared_error"


def test_prepare_dataset_resolves_kfold_fold_count():
    # 100 строк, 3 признака, явная k_fold c n_folds=3
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[], validation_strategy="k_fold", n_folds=3
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=100, n_features=3), "target")
    assert prepared.resolved_validation == ValidationStrategy.k_fold
    assert prepared.resolved_n_folds == 3


def test_prepare_dataset_resolves_auto_to_kfold():
    # auto на 100×3: choose_validation_method выбирает kfold c k=5
    cfg = Config.model_validate(
        _cfg_dict(phases=[], validation_strategy="auto", n_folds=5)
    )
    prepared = prepare_dataset(cfg, _df(n_rows=100, n_features=3), "target")
    assert prepared.resolved_validation == ValidationStrategy.k_fold
    assert prepared.resolved_n_folds == 5
    assert prepared.resolved_test_size == 0.2


def test_prepare_dataset_resolves_auto_test_size_to_int_rows():
    # auto на 300×3: train_test_split c целочисленным test_size=30
    cfg = Config.model_validate(
        _cfg_dict(phases=[], validation_strategy="auto", n_folds=5)
    )
    prepared = prepare_dataset(cfg, _df(n_rows=300, n_features=3), "target")
    assert prepared.resolved_validation == ValidationStrategy.train_test_split
    assert prepared.resolved_test_size == 30


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
def test_execute_phases_single_phase_all_algorithms():
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[{"name": "coarse", "n_trials": 2, "action": "all_algorithms"}],
            algorithms=["random_forest", "extra_trees"],
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    with patch.object(
        component,
        "_run_hpo",
        side_effect=lambda **kw: (0.9, {"p": 1}),
    ) as mock_hpo:
        phase_results, disqualified = execute_phases(cfg, prepared)

    assert set(phase_results) == {"random_forest", "extra_trees"}
    assert phase_results["random_forest"] == (0.9, {"p": 1})
    assert disqualified == {}
    assert mock_hpo.call_count == 2


def test_execute_phases_refine_winner_uses_initial_params():
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[
                {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
                {"name": "fine", "n_trials": 1, "action": "refine_winner"},
            ],
            algorithms=["random_forest", "extra_trees"],
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    calls: list[tuple[str, Any]] = []

    def fake_run_hpo(**kw: Any) -> tuple[float, dict[str, Any]]:
        algo = kw["algo_name"]
        calls.append((algo, kw.get("initial_params")))
        scores = {"random_forest": 0.9, "extra_trees": 0.8}
        return scores[algo], {"p": 1}

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        phase_results, disqualified = execute_phases(cfg, prepared)

    # Фаза 1: оба алгоритма без initial_params; фаза 2: только победитель
    # с параметрами фазы 1.
    assert calls == [
        ("random_forest", None),
        ("extra_trees", None),
        ("random_forest", {"p": 1}),
    ]
    assert set(phase_results) == {"random_forest"}
    assert disqualified == {}


def test_execute_phases_refine_winner_without_previous_raises():
    cfg = Config.model_validate(
        _cfg_dict(phases=[{"name": "fine", "n_trials": 1, "action": "refine_winner"}])
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    with patch.object(component, "_run_hpo", return_value=(0.9, {})):
        with pytest.raises(RuntimeError, match="requires a winner"):
            execute_phases(cfg, prepared)


def test_execute_phases_empty_phases_returns_empty():
    cfg = Config.model_validate(_cfg_dict(phases=[]))
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")
    phase_results, disqualified = execute_phases(cfg, prepared)
    assert phase_results == {}
    assert disqualified == {}


def test_execute_phases_no_valid_scores_raises_runtime_error():
    cfg = Config.model_validate(
        _cfg_dict(phases=[{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}])
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    # Все триалы неуспешны (best_score=None → _run_hpo возвращает None)
    with patch.object(component, "_run_hpo", return_value=None):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            execute_phases(cfg, prepared)


def test_execute_phases_filters_worst_score_sentinel():
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}],
            algorithms=["random_forest", "extra_trees"],
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    def fake_run_hpo(**kw: Any) -> tuple[float, dict[str, Any]]:
        if kw["algo_name"] == "random_forest":
            return HPO_WORST_SCORE, {"p": 1}  # сентинел → исключается
        return 0.7, {"p": 2}

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        phase_results, disqualified = execute_phases(cfg, prepared)

    assert set(phase_results) == {"extra_trees"}
    assert disqualified == {}


def test_execute_phases_disqualifies_invalid_algorithm():
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}],
            algorithms=["random_forest", "extra_trees"],
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    def fake_run_hpo(**kw: Any) -> tuple[float, dict[str, Any]]:
        if kw["algo_name"] == "random_forest":
            raise InvalidAlgorithmError("incompatible with data")
        return 0.7, {"p": 2}

    with patch.object(component, "_run_hpo", side_effect=fake_run_hpo):
        phase_results, disqualified = execute_phases(cfg, prepared)

    assert set(phase_results) == {"extra_trees"}
    assert "random_forest" in disqualified
    assert "incompatible with data" in disqualified["random_forest"]


def test_execute_phases_all_invalid_algorithms_raise():
    cfg = Config.model_validate(
        _cfg_dict(
            phases=[{"name": "coarse", "n_trials": 1, "action": "all_algorithms"}],
            algorithms=["random_forest", "extra_trees"],
        )
    )
    prepared = prepare_dataset(cfg, _df(n_rows=20), "target")

    with patch.object(
        component, "_run_hpo", side_effect=InvalidAlgorithmError("bad")
    ):
        with pytest.raises(RuntimeError, match="No algorithms produced valid scores"):
            execute_phases(cfg, prepared)


# ──────────────────────────────────────────────────────────────────────────────
# persist_artifact
# ──────────────────────────────────────────────────────────────────────────────
def test_persist_artifact_basic_result():
    cfg = Config.model_validate(
        _cfg_dict(phases=[], algorithms=["random_forest"])
    )
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
    # Для mae «сырой» -1.5 инвертируется в естественное положительное значение
    assert result["score"] == 1.5
    assert result["metric"] == "mae"
    assert result["params"] == {"alpha": 1.0}
    assert result["model_path"] == "model.pkl"
    assert "disqualified_algorithms" not in result
    assert "additional_metrics" not in result

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
    cfg = Config.model_validate(
        _cfg_dict(phases=[], additional_metrics=["r2"])
    )
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


# ──────────────────────────────────────────────────────────────────────────────
# train_best_model: оркестрация шагов
# ──────────────────────────────────────────────────────────────────────────────
def test_train_best_model_orchestrates_steps_in_order():
    order: list[str] = []
    cfg = MagicMock()
    prepared = MagicMock()
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
            side_effect=lambda c, p: record(
                "execute_phases", ({"rf": (0.9, {})}, {})
            ),
        ),
        patch.object(
            component,
            "select_winner",
            side_effect=lambda r: record("select_winner", "rf"),
        ),
        patch.object(
            component,
            "persist_artifact",
            side_effect=lambda **kw: record("persist_artifact", expected_result),
        ),
    ):
        out = train_best_model(config="x", df=MagicMock(), target="target")

    assert out is expected_result
    assert order == [
        "load_config",
        "prepare_dataset",
        "execute_phases",
        "select_winner",
        "persist_artifact",
    ]


def test_train_best_model_no_phase_results_skips_winner_and_persist():
    with (
        patch.object(component, "load_config", return_value=(MagicMock(), "target")),
        patch.object(component, "prepare_dataset", return_value=MagicMock()),
        patch.object(component, "execute_phases", return_value=({}, {})),
        patch.object(component, "select_winner") as mock_select,
        patch.object(component, "persist_artifact") as mock_persist,
    ):
        with pytest.raises(RuntimeError, match="No valid results after HPO phases"):
            train_best_model(config={}, df=MagicMock())

    mock_select.assert_not_called()
    mock_persist.assert_not_called()


def test_train_best_model_invalid_winner_score_raises():
    sentinel = float(np.finfo(np.float32).min)
    with (
        patch.object(component, "load_config", return_value=(MagicMock(), "target")),
        patch.object(component, "prepare_dataset", return_value=MagicMock()),
        patch.object(
            component,
            "execute_phases",
            return_value=({"rf": (sentinel, {})}, {}),
        ),
        patch.object(component, "persist_artifact") as mock_persist,
    ):
        with pytest.raises(RuntimeError, match="invalid final score"):
            train_best_model(config={}, df=MagicMock())

    mock_persist.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Фикстура для load_config из файла
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mocker_patch_read_config():
    cfg_mock = MagicMock()
    cfg_mock.general.log_to_file = None
    with patch.object(component, "read_config", return_value=cfg_mock) as read_mock:
        yield cfg_mock, read_mock