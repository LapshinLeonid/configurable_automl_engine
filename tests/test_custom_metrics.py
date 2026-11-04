"""Custom metrics and scorers declared by a dotted import path.

Covers the declaration (``CustomMetricSpec`` / ``general.custom_metrics``),
the metric helpers of ``common.metrics`` for registered custom names, the
direct APIs (``optimize``, ``ModelTrainer``), the full pipeline in the
threads and processes modes, and the saved artifact (it must load without
the metric module).
"""

from __future__ import annotations

import logging
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from configurable_automl_engine.common import custom_metrics as cm
from configurable_automl_engine.common import metrics as metrics_mod
from configurable_automl_engine.common.custom_metrics import (
    CustomMetricSpec,
    resolve_dotted_path,
)
from configurable_automl_engine.common.metrics import (
    direction_label,
    get_metric,
    get_scorer_object,
    is_error_metric,
    is_greater_better,
    is_non_additive_metric,
    register_custom_metrics,
    reserved_metric_names,
    to_sklearn_name,
    to_user_value,
    user_direction,
    validate_custom_metrics,
)
from configurable_automl_engine.trainer import ModelTrainer, TrainingError
from configurable_automl_engine.training_engine import train_best_model
from configurable_automl_engine.training_engine.config_parser import Config
from configurable_automl_engine.tuner import optimize

from . import custom_metric_fixtures as fx

FX = "tests.custom_metric_fixtures"


def _spec(attr: str, **kw) -> CustomMetricSpec:
    return CustomMetricSpec(path=f"{FX}:{attr}", **kw)


def _register(name: str, spec: CustomMetricSpec) -> str:
    register_custom_metrics({name: spec})
    return name


@pytest.fixture
def regression_xy():
    X, y = make_regression(n_samples=80, n_features=4, noise=5.0, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), pd.Series(y)


@pytest.fixture
def regression_df(regression_xy):
    X, y = regression_xy
    df = X.copy()
    df["target"] = y.to_numpy()
    return df


# --------------------------------------------------------------------------- #
#  resolve_dotted_path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        f"{FX}:custom_mae",
        f"{FX}.custom_mae",
        f"  {FX}:custom_mae  ",
    ],
)
def test_resolve_dotted_path_forms(path):
    assert resolve_dotted_path(path) is fx.custom_mae


def test_resolve_dotted_path_nested_attribute():
    assert resolve_dotted_path(f"{FX}:Holder.metric") is fx.Holder.metric


@pytest.mark.parametrize(
    "path, fragment",
    [
        ("no_dots", "not a valid import path"),
        (":attr", "not a valid import path"),
        (f"{FX}:", "not a valid import path"),
        (f"{FX}:a:b", "not a valid import path"),
        ("no_such_pkg_xyz.mod:f", "cannot import module 'no_such_pkg_xyz.mod'"),
        (f"{FX}:missing", "has no attribute 'missing'"),
        (f"{FX}:Holder.missing", "has no attribute 'Holder.missing'"),
    ],
)
def test_resolve_dotted_path_errors(path, fragment):
    with pytest.raises(ImportError, match=fragment):
        resolve_dotted_path(path)


def test_resolve_dotted_path_wraps_any_import_exception(tmp_path, monkeypatch):
    """An exception raised by the module body is reported as ImportError."""
    (tmp_path / "exploding_metric_mod.py").write_text("raise ZeroDivisionError('x')")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match="ZeroDivisionError"):
        resolve_dotted_path("exploding_metric_mod:f")


# --------------------------------------------------------------------------- #
#  CustomMetricSpec validation
# --------------------------------------------------------------------------- #
def _error_type(exc: ValidationError) -> str:
    return exc.errors()[0]["type"]


def test_spec_metric_valid_with_kwargs():
    spec = _spec("scaled_r2", greater_is_better=True, kwargs={"scale": 2.0})
    assert spec.kind == "metric"
    assert spec.additive is False
    assert spec.kwargs == {"scale": 2.0}


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"path": f"{FX}:custom_mae"}, cm.ERR_DIRECTION_REQUIRED),
        (
            {"path": f"{FX}:one_arg", "greater_is_better": False},
            cm.ERR_BAD_SIGNATURE,
        ),
        (
            {
                "path": f"{FX}:custom_mae",
                "greater_is_better": False,
                "kwargs": {"unknown": 1},
            },
            cm.ERR_BAD_SIGNATURE,
        ),
        (
            {"path": f"{FX}:NOT_CALLABLE", "greater_is_better": False},
            cm.ERR_NOT_CALLABLE,
        ),
        (
            {"path": "no_such_pkg_xyz:f", "greater_is_better": False},
            cm.ERR_IMPORT,
        ),
        ({"path": f"{FX}:custom_mae", "kind": "scorer"}, cm.ERR_BAD_SIGNATURE),
        (
            {"path": f"{FX}:plain_scorer", "kind": "scorer", "kwargs": {"a": 1}},
            cm.ERR_KWARGS_UNSUPPORTED,
        ),
        (
            {
                "path": f"{FX}:neg_mae_scorer",
                "kind": "scorer",
                "greater_is_better": True,
            },
            cm.ERR_DIRECTION_CONFLICT,
        ),
    ],
)
def test_spec_validation_errors(kwargs, code):
    with pytest.raises(ValidationError) as exc_info:
        CustomMetricSpec(**kwargs)
    assert _error_type(exc_info.value) == code


def test_spec_rejects_unknown_fields_and_empty_path():
    with pytest.raises(ValidationError):
        CustomMetricSpec(path=f"{FX}:custom_mae", greater_is_better=False, foo=1)
    with pytest.raises(ValidationError):
        CustomMetricSpec(path="", greater_is_better=False)


def test_spec_make_scorer_direction_taken_from_scorer():
    spec = _spec("neg_mae_scorer", kind="scorer")
    assert spec.natural_greater_is_better is False
    agreeing = _spec("neg_mae_scorer", kind="scorer", greater_is_better=False)
    assert agreeing.natural_greater_is_better is False


def test_spec_plain_scorer_without_additive_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _spec("plain_scorer", kind="scorer")
    assert "pooled out-of-fold" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _spec("plain_scorer", kind="scorer", additive=True)
    assert "pooled out-of-fold" not in caplog.text


def test_spec_is_picklable_without_the_callable():
    """The pickled spec is data: it does not reference the metric function."""
    spec = _spec("custom_mae", greater_is_better=False)
    blob = pickle.dumps(spec)
    assert pickle.loads(blob) == spec
    # The function object is not pickled (only the path string is).
    assert b"custom_mae" in blob
    assert pickle.dumps(fx.custom_mae) not in blob


# --------------------------------------------------------------------------- #
#  Names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["r2", "rmse", "neg_root_mean_squared_error", "global_nrmse", "accuracy"]
)
def test_name_collision_rejected(name):
    assert name in reserved_metric_names()
    with pytest.raises(ValueError, match="already a built-in or sklearn") as exc_info:
        validate_custom_metrics({name: _spec("custom_mae", greater_is_better=False)})
    assert exc_info.value.type == metrics_mod.ERR_CUSTOM_NAME_COLLISION


@pytest.mark.parametrize("name", ["MyMetric", "1abc", "with-dash", "", "a b"])
def test_invalid_name_rejected(name):
    with pytest.raises(ValueError, match="must match") as exc_info:
        validate_custom_metrics({name: _spec("custom_mae", greater_is_better=False)})
    assert exc_info.value.type == metrics_mod.ERR_CUSTOM_NAME_INVALID


def test_validate_custom_metrics_accepts_dicts_and_specs():
    spec = _spec("custom_mae", greater_is_better=False)
    out = validate_custom_metrics(
        {
            "cm_a": spec,
            "cm_b": {"path": f"{FX}:custom_rmse", "greater_is_better": False},
        }
    )
    assert out["cm_a"] is spec
    assert isinstance(out["cm_b"], CustomMetricSpec)
    assert validate_custom_metrics(None) == {}


# --------------------------------------------------------------------------- #
#  Metric helpers for registered custom names
# --------------------------------------------------------------------------- #
def test_helpers_error_metric():
    name = _register("cm_err", _spec("custom_rmse", greater_is_better=False))
    assert is_greater_better(name) is False
    assert is_error_metric(name) is True
    assert user_direction(name) == "minimize"
    assert direction_label(name) == "min better"
    assert to_user_value(name, -1.5) == 1.5
    assert is_non_additive_metric(name) is True
    assert to_sklearn_name(name) == name
    assert get_metric(name) is fx.custom_rmse


def test_helpers_score_metric_with_kwargs():
    name = _register(
        "cm_score", _spec("scaled_r2", greater_is_better=True, kwargs={"scale": 2.0})
    )
    assert is_greater_better(name) is True
    assert is_error_metric(name) is False
    assert user_direction(name) == "maximize"
    assert to_user_value(name, 0.7) == 0.7
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert get_metric(name)(y, y) == pytest.approx(2.0)
    scorer = get_scorer_object(name)
    assert scorer._kwargs == {"scale": 2.0}


def test_helpers_case_insensitive_lookup():
    _register("cm_case", _spec("custom_mae", greater_is_better=False, additive=True))
    assert is_error_metric("CM_CASE") is True
    assert is_non_additive_metric(" cm_case ") is False


def test_helpers_make_scorer_scorer():
    name = _register(
        "cm_negmae", _spec("neg_mae_scorer", kind="scorer", additive=True)
    )
    assert get_scorer_object(name) is fx.neg_mae_scorer
    assert is_error_metric(name) is True
    assert to_user_value(name, -0.3) == 0.3
    assert is_non_additive_metric(name) is False
    with pytest.raises(KeyError):
        get_metric(name)


def test_helpers_make_scorer_scorer_kwargs_are_merged_into_a_copy():
    spec = _spec("neg_mae_scorer", kind="scorer", kwargs={"multioutput": "raw_values"})
    name = _register("cm_negmae_kw", spec)
    scorer = get_scorer_object(name)
    assert scorer is not fx.neg_mae_scorer
    assert scorer._kwargs == {"multioutput": "raw_values"}
    assert fx.neg_mae_scorer._kwargs == {}


def test_helpers_plain_scorers(regression_xy):
    X, y = regression_xy
    model = Ridge().fit(X, y)
    score_name = _register(
        "cm_plain", _spec("plain_scorer", kind="scorer", additive=True)
    )
    err_name = _register(
        "cm_plain_err",
        _spec(
            "plain_error_scorer", kind="scorer", greater_is_better=False, additive=True
        ),
    )
    score_scorer = get_scorer_object(score_name)
    assert score_scorer(model, X, y) == pytest.approx(model.score(X, y))
    assert is_error_metric(score_name) is False
    assert getattr(score_scorer, "_score_func", None) is None

    err_scorer = get_scorer_object(err_name)
    raw = err_scorer(model, X, y)
    assert raw < 0  # maximized by the optimizer
    assert is_error_metric(err_name) is True
    assert to_user_value(err_name, raw) == pytest.approx(
        fx.plain_error_scorer(model, X, y)
    )


def test_custom_name_never_reaches_the_sklearn_direction_cache():
    name = _register("cm_cache", _spec("custom_mae", greater_is_better=False))
    before = metrics_mod._sklearn_direction.cache_info()
    is_greater_better(name)
    after = metrics_mod._sklearn_direction.cache_info()
    assert (after.hits, after.misses) == (before.hits, before.misses)


def test_scorer_is_cached_per_spec():
    spec = _spec("custom_mae", greater_is_better=False, additive=True)
    assert cm.build_custom_scorer(spec) is cm.build_custom_scorer(spec)


def test_register_same_spec_is_noop_and_redefinition_warns(caplog):
    spec = _spec("custom_mae", greater_is_better=False)
    register_custom_metrics({"cm_redef": spec})
    with caplog.at_level(logging.WARNING):
        register_custom_metrics({"cm_redef": spec})
    assert "redefined" not in caplog.text
    other = _spec("custom_rmse", greater_is_better=False)
    with caplog.at_level(logging.WARNING):
        register_custom_metrics({"cm_redef": other})
    assert "redefined" in caplog.text
    assert cm.lookup_custom_metric("cm_redef") == other


def test_unregistered_name_is_unknown():
    with pytest.raises(ValueError, match="not implemented"):
        is_error_metric("cm_never_registered")
    assert cm.lookup_custom_metric("cm_never_registered") is None


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
_ALGOS = {"ridge": {"enable": True}}


def _cfg(general: dict) -> dict:
    return {"general": general, "algorithms": _ALGOS}


_MAPE = {"path": f"{FX}:custom_mae", "greater_is_better": False, "additive": True}


def test_config_custom_comparison_and_additional():
    cfg = Config.model_validate(
        _cfg(
            {
                "custom_metrics": {
                    "my_mae": _MAPE,
                    "my_rmse": {
                        "path": f"{FX}.custom_rmse",
                        "greater_is_better": False,
                    },
                },
                "comparison_metric": "my_rmse",
                "additional_metrics": ["my_mae", "r2", "my_mae"],
            }
        )
    )
    assert cfg.general.comparison_metric == "my_rmse"
    assert cfg.general.additional_metrics == ["my_mae", "r2"]
    assert isinstance(cfg.general.custom_metrics["my_mae"], CustomMetricSpec)


def test_config_default_has_no_custom_metrics():
    cfg = Config.model_validate(_cfg({}))
    assert cfg.general.custom_metrics == {}
    assert cfg.general.comparison_metric == "r2"


@pytest.mark.parametrize(
    "general, fragment",
    [
        ({"comparison_metric": "my_mae"}, "comparison_metric: unknown metric"),
        ({"comparison_metric": "R2"}, "comparison_metric: unknown metric 'R2'"),
        (
            {"custom_metrics": {"my_mae": _MAPE}, "comparison_metric": "MY_MAE"},
            "unknown metric 'MY_MAE'",
        ),
        ({"additional_metrics": ["r2", "nope"]}, "additional_metrics: unknown"),
    ],
)
def test_config_unknown_metric_reference(general, fragment):
    with pytest.raises(ValidationError, match=fragment):
        Config.model_validate(_cfg(general))


def test_config_unknown_metric_lists_custom_names():
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(
            _cfg({"custom_metrics": {"my_mae": _MAPE}, "comparison_metric": "x"})
        )
    msg = str(exc_info.value)
    assert "'my_mae'" in msg and "'rmse'" in msg


def test_config_spec_error_names_the_entry():
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(
            _cfg({"custom_metrics": {"my_mae": {"path": f"{FX}:custom_mae"}}})
        )
    err = exc_info.value.errors()[0]
    assert err["type"] == cm.ERR_DIRECTION_REQUIRED
    assert err["loc"] == ("general", "custom_metrics", "my_mae")


def test_config_reserved_name_reported_before_import():
    """A reserved name fails even when its path cannot be imported."""
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(
            _cfg({"custom_metrics": {"rmse": {"path": "no_such_pkg_xyz:f"}}})
        )
    assert exc_info.value.errors()[0]["type"] == metrics_mod.ERR_CUSTOM_NAME_COLLISION


def test_config_unused_custom_metric_warns(caplog):
    with caplog.at_level(logging.WARNING):
        Config.model_validate(_cfg({"custom_metrics": {"my_mae": _MAPE}}))
    assert "general.custom_metrics.my_mae is declared but not used" in caplog.text


def test_config_round_trips_through_model_dump():
    data = _cfg({"custom_metrics": {"my_mae": _MAPE}, "comparison_metric": "my_mae"})
    cfg = Config.model_validate(data)
    again = Config.model_validate(cfg.model_dump(mode="json"))
    assert again.general.custom_metrics == cfg.general.custom_metrics


def test_config_schema_lists_custom_metrics():
    schema = Config.model_json_schema()
    general = schema["$defs"]["GeneralCfg"]["properties"]
    assert "custom_metrics" in general
    assert general["comparison_metric"]["type"] == "string"


# --------------------------------------------------------------------------- #
#  ModelTrainer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["k_fold", "loo"])
def test_trainer_custom_rmse_matches_builtin(regression_xy, strategy):
    """A non-additive custom RMSE gives exactly the pooled built-in RMSE."""
    X, y = regression_xy
    X, y = X.iloc[:30], y.iloc[:30]
    common = dict(algorithm="ridge", validation_strategy=strategy, n_folds=3)
    builtin = ModelTrainer(metric="rmse", additional_metrics=["mae"], **common)
    custom = ModelTrainer(
        metric="cm_t_rmse",
        additional_metrics=["cm_t_mae"],
        custom_metrics={
            "cm_t_rmse": {"path": f"{FX}:custom_rmse", "greater_is_better": False},
            "cm_t_mae": {
                "path": f"{FX}:custom_mae",
                "greater_is_better": False,
                "additive": True,
            },
        },
        **common,
    )
    builtin.fit(X, y)
    custom.fit(X, y)
    assert custom.val_score == pytest.approx(builtin.val_score, rel=1e-12)
    assert custom.additional_scores["cm_t_mae"] == pytest.approx(
        builtin.additional_scores["mae"], rel=1e-12
    )
    assert custom.oof_score_ == pytest.approx(builtin.oof_score_, rel=1e-12)


def test_trainer_invalid_custom_metrics_raises_training_error():
    with pytest.raises(TrainingError, match="Invalid custom_metrics"):
        ModelTrainer(custom_metrics={"r2": {"path": f"{FX}:custom_mae"}})
    with pytest.raises(TrainingError, match="greater_is_better"):
        ModelTrainer(custom_metrics={"cm_x": {"path": f"{FX}:custom_mae"}})


def test_trainer_failing_additional_custom_metric_is_skipped(regression_xy, caplog):
    X, y = regression_xy
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["cm_boom", "mae"],
        validation_strategy="train_test_split",
        custom_metrics={
            "cm_boom": {"path": f"{FX}:raising_metric", "greater_is_better": False}
        },
    )
    with caplog.at_level(logging.WARNING):
        trainer.fit(X, y)
    assert "cm_boom" not in trainer.additional_scores
    assert "mae" in trainer.additional_scores
    assert trainer.val_score is not None


def test_trainer_custom_metrics_property_tolerates_settings_without_field():
    """An artifact of the same format may predate the field: it reads as empty."""
    trainer = ModelTrainer()

    class _OldSettings:
        pass

    trainer.settings = _OldSettings()  # type: ignore[assignment]
    assert trainer.custom_metrics == {}


def test_trainer_registers_custom_metrics_again_in_fit(regression_xy, monkeypatch):
    X, y = regression_xy
    spec = _spec("custom_mae", greater_is_better=False, additive=True)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="cm_refit",
        validation_strategy="train_test_split",
        custom_metrics={"cm_refit": spec},
    )
    # Simulate a fresh process: the registry no longer knows the name.
    monkeypatch.setattr(cm, "_registry", {})
    trainer.fit(X, y)
    assert trainer.val_score is not None
    assert cm.lookup_custom_metric("cm_refit") == spec


# --------------------------------------------------------------------------- #
#  optimize()
# --------------------------------------------------------------------------- #
def test_optimize_custom_metric_matches_builtin(regression_xy):
    X, y = regression_xy
    common = dict(n_trials=4, val_method="k_fold", n_folds=3, random_state=0)
    builtin = optimize("ridge", X, y, metric="rmse", fit_best_model=False, **common)
    custom = optimize(
        "ridge",
        X,
        y,
        metric="cm_o_rmse",
        custom_metrics={
            "cm_o_rmse": {"path": f"{FX}:custom_rmse", "greater_is_better": False}
        },
        fit_best_model=False,
        **common,
    )
    assert custom.score == pytest.approx(builtin.score, rel=1e-12)
    assert custom.params == builtin.params


def test_optimize_with_pruning_and_custom_metric(regression_xy):
    X, y = regression_xy
    fx.CALLS["metric"] = 0
    result = optimize(
        "ridge",
        X,
        y,
        metric="cm_o_count",
        custom_metrics={
            "cm_o_count": {
                "path": f"{FX}:counting_mae",
                "greater_is_better": False,
                "additive": True,
            }
        },
        n_trials=4,
        val_method="k_fold",
        n_folds=3,
        pruning={"enable": True, "strategy": "median", "min_steps": 1},
        fit_best_model=False,
    )
    assert result is not None and result.score < 0
    assert fx.CALLS["metric"] > 0


def test_optimize_invalid_custom_metric_raises(regression_xy):
    X, y = regression_xy
    with pytest.raises(ValueError, match="greater_is_better"):
        optimize(
            "ridge",
            X,
            y,
            metric="cm_bad",
            custom_metrics={"cm_bad": {"path": f"{FX}:custom_mae"}},
            n_trials=1,
        )


# --------------------------------------------------------------------------- #
#  Full pipeline
# --------------------------------------------------------------------------- #
def _pipeline_config(
    model_path, comparison, *, mode="threads", start_method=None, select="cv_best"
) -> dict:
    general = {
        "custom_metrics": {
            "p_rmse": {"path": f"{FX}:custom_rmse", "greater_is_better": False},
            "p_mae": {
                "path": f"{FX}:custom_mae",
                "greater_is_better": False,
                "additive": True,
            },
        },
        "comparison_metric": comparison,
        "additional_metrics": ["p_mae", "r2"],
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "random_state": 0,
        "parallel_strategy": "algorithms",
        "parallel_mode": mode,
        "max_workers": 2,
        "phases": [
            {"name": "p1", "n_trials": 3, "action": "all_algorithms"},
            {"name": "final", "action": "select", "strategy": select, "output": "winner"},
        ],
        "path_to_model": str(model_path),
    }
    if start_method is not None:
        general["start_method"] = start_method
    return {
        "general": general,
        "algorithms": {"ridge": {"enable": True}, "elasticnet": {"enable": True}},
    }


def _summary(result: dict) -> dict:
    """The numeric estimates of a run (``None`` → NaN), keyed with the winner."""
    keys = ("score", "oof_score", "holdout_score", "bbc_score")
    return {
        f"{result['algorithm']}:{k}": float("nan") if result.get(k) is None else result[k]
        for k in keys
    }


def test_pipeline_custom_metric_matches_builtin(regression_df, tmp_path):
    builtin = train_best_model(
        _pipeline_config(tmp_path / "a.pkl", "rmse"), regression_df, target="target"
    )
    custom = train_best_model(
        _pipeline_config(tmp_path / "b.pkl", "p_rmse"), regression_df, target="target"
    )
    assert custom["metric"] == "p_rmse"
    assert _summary(custom) == pytest.approx(_summary(builtin), rel=1e-12, nan_ok=True)
    assert custom["additional_metrics"]["p_mae"] > 0


def test_pipeline_sanity_gate_select_with_custom_metric(regression_df, tmp_path):
    result = train_best_model(
        _pipeline_config(tmp_path / "m.pkl", "p_rmse", select="sanity_gate"),
        regression_df,
        target="target",
    )
    assert result["metric"] == "p_rmse"
    assert np.isfinite(result["score"]) and result["score"] > 0


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_pipeline_processes_mode(regression_df, tmp_path, start_method):
    """Workers rebuild the registry from the pickled specs (no fork needed)."""
    import multiprocessing

    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is not available")
    threads = train_best_model(
        _pipeline_config(tmp_path / "t.pkl", "p_rmse"), regression_df, target="target"
    )
    processes = train_best_model(
        _pipeline_config(
            tmp_path / "p.pkl", "p_rmse", mode="processes", start_method=start_method
        ),
        regression_df,
        target="target",
    )
    assert _summary(processes) == pytest.approx(_summary(threads), rel=1e-12, nan_ok=True)


def test_pipeline_bbc_with_plain_scorer_falls_back(regression_df, tmp_path, caplog):
    """No hold-out on small data: BBC-CV needs the raw metric function."""
    config = _pipeline_config(tmp_path / "m.pkl", "p_plain")
    config["general"]["custom_metrics"]["p_plain"] = {
        "path": f"{FX}:plain_scorer",
        "kind": "scorer",
    }
    config["general"]["holdout_fraction"] = None
    with caplog.at_level(logging.WARNING):
        result = train_best_model(config, regression_df, target="target")
    assert result["metric"] == "p_plain"
    assert "bbc_score" not in result
    assert "BBC-CV estimate could not be computed" in caplog.text


def test_pipeline_bbc_with_custom_metric(regression_df, tmp_path):
    builtin_cfg = _pipeline_config(tmp_path / "a.pkl", "rmse")
    custom_cfg = _pipeline_config(tmp_path / "b.pkl", "p_rmse")
    for cfg in (builtin_cfg, custom_cfg):
        cfg["general"]["holdout_fraction"] = None
    builtin = train_best_model(builtin_cfg, regression_df, target="target")
    custom = train_best_model(custom_cfg, regression_df, target="target")
    assert "bbc_score" in custom
    assert custom["bbc_score"] == pytest.approx(builtin["bbc_score"], rel=1e-12)


# --------------------------------------------------------------------------- #
#  Artifact
# --------------------------------------------------------------------------- #
def test_artifact_metadata_records_custom_metrics(regression_xy, tmp_path):
    X, y = regression_xy
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="cm_art",
        validation_strategy="train_test_split",
        custom_metrics={"cm_art": _MAPE},
    )
    trainer.fit(X, y)
    trainer.save(tmp_path / "m.pkl")
    meta = trainer.artifact_metadata_
    assert meta["custom_metrics"] == {
        "cm_art": {
            "path": f"{FX}:custom_mae",
            "kind": "metric",
            "greater_is_better": False,
            "additive": True,
            "kwargs": {},
        }
    }

    plain = ModelTrainer(algorithm="ridge", validation_strategy="train_test_split")
    plain.fit(X, y)
    plain.save(tmp_path / "plain.pkl")
    assert plain.artifact_metadata_["custom_metrics"] == {}
    assert plain.artifact_metadata_["config_hash"] != meta["config_hash"]


def test_artifact_loads_without_the_metric_module(regression_xy, tmp_path):
    """Loading and predicting never import the custom metric module."""
    X, y = regression_xy
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="cm_portable",
        validation_strategy="train_test_split",
        custom_metrics={"cm_portable": _MAPE},
    )
    trainer.fit(X, y)
    path = tmp_path / "m.pkl"
    trainer.save(path)
    X.iloc[:3].to_csv(tmp_path / "x.csv", index=False)
    script = textwrap.dedent(
        f"""
        import sys
        import pandas as pd
        from configurable_automl_engine.trainer import ModelTrainer
        m = ModelTrainer.load({str(path)!r})
        pred = m.predict(pd.read_csv({str(tmp_path / "x.csv")!r}))
        assert len(pred) == 3
        assert not any(k == "tests" or k.startswith("tests.") for k in sys.modules)
        print(sorted(m.custom_metrics))
        """
    )
    # -I: isolated mode, the repository root (and so ``tests``) is not on
    # sys.path; cwd is a temp dir.
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "['cm_portable']" in proc.stdout
