"""Tests for the ``additional_metrics`` feature.

Coverage:
    1. Configuration: the new optional ``general.additional_metrics`` parameter,
       deduplication, exclusion of the main comparison metric, and a validation
       error for an unknown metric.
    2. ModelTrainer: computing additional metrics for the trained model the same
       way as the main metric; non-numeric values (inf/NaN); a single metric
       failure does not affect training.
    3. train_best_model: returning ``additional_metrics`` in the results,
       backward compatibility, no impact on winner selection.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pydantic import ValidationError

from configurable_automl_engine.trainer import ModelTrainer, TrainingError
from configurable_automl_engine.training_engine.component import (
    PhaseResult,
    _fit_and_save,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    GeneralCfg,
)

# ──────────────────────────────────────────────────────────────────────────────
#  1. Configuration validation
# ──────────────────────────────────────────────────────────────────────────────

BASE = {
    "general": {
        "comparison_metric": "rmse",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "phases": [
            {"name": "search", "n_trials": 1, "action": "all_algorithms"},
            {
                "name": "final_select",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
    },
    "algorithms": {"elasticnet": {"enable": True}},
}


def _with_additional(metrics: list[str]) -> dict:
    return {
        **BASE,
        "general": {**BASE["general"], "additional_metrics": metrics},
    }


def test_additional_metrics_default_empty():
    """Parameter not set — the list is empty (backward compatibility)."""
    cfg = Config.model_validate(BASE)
    assert cfg.general.additional_metrics == []


def test_additional_metrics_explicit_empty_list():
    """An empty list is allowed and equivalent to the parameter being absent."""
    cfg = Config.model_validate(_with_additional([]))
    assert cfg.general.additional_metrics == []


def test_additional_metrics_valid_list_accepted():
    """A valid list of metrics from the supported set is accepted."""
    cfg = Config.model_validate(_with_additional(["r2", "mae", "mse"]))
    assert cfg.general.additional_metrics == ["r2", "mae", "mse"]


def test_additional_metrics_unknown_metric_rejected():
    """Unknown metric -> validation error before training starts."""
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(_with_additional(["r2", "bogus_metric"]))
    msg = str(exc_info.value)
    assert "additional_metrics" in msg
    assert "bogus_metric" in msg
    # The message lists the allowed values
    for allowed in ("nrmse", "rmse", "mae", "mse", "r2"):
        assert allowed in msg


def test_additional_metrics_duplicates_deduplicated():
    """Duplicates in the list are counted once (first-occurrence order)."""
    cfg = Config.model_validate(_with_additional(["r2", "mae", "r2", "mae"]))
    assert cfg.general.additional_metrics == ["r2", "mae"]


def test_additional_metrics_comparison_metric_excluded():
    """A match with the main comparison metric is excluded from the results."""
    cfg = Config.model_validate(_with_additional(["r2", "rmse", "mae"]))
    assert cfg.general.additional_metrics == ["r2", "mae"]


def test_additional_metrics_alias_of_comparison_excluded(caplog):
    """An alias of the main metric (rmse ↔ neg_root_mean_squared_error) is excluded."""
    data = {
        **BASE,
        "general": {
            **BASE["general"],
            "additional_metrics": ["neg_root_mean_squared_error", "mae"],
        },
    }
    with caplog.at_level(logging.WARNING):
        cfg = Config.model_validate(data)
    assert cfg.general.additional_metrics == ["mae"]
    assert "comparison metric" in caplog.text


def test_additional_metrics_all_filtered_to_empty():
    """If all metrics match the main one — the result is empty (no duplicates)."""
    cfg = Config.model_validate(_with_additional(["rmse"]))
    assert cfg.general.additional_metrics == []


def test_additional_metrics_non_string_element_rejected():
    """A non-string element gives a ValidationError, not an AttributeError.

    The ``_deduplicate_additional_metrics`` validator is typed as
    ``list[ComparisonMetric]``: the list elements must be strings from the
    allowed set, and the type check runs before ``.lower()`` is called.
    """
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(_with_additional(["r2", 123]))
    msg = str(exc_info.value)
    assert "additional_metrics" in msg
    assert "AttributeError" not in msg


def test_additional_metrics_tuple_input_accepted():
    """A tuple of metrics (not only a list) is accepted and converted to a list."""
    data = {
        **BASE,
        "general": {**BASE["general"], "additional_metrics": ("r2", "mae")},
    }
    cfg = Config.model_validate(data)
    assert cfg.general.additional_metrics == ["r2", "mae"]


def test_additional_metrics_post_validation_list_may_differ(caplog):
    """The final list after validation may differ from the user-provided one.

    Documented behavior: deduplication and exclusion of the main comparison
    metric happen during configuration validation, so
    ``cfg.general.additional_metrics`` may not match the original list.
    """
    with caplog.at_level(logging.WARNING):
        cfg = Config.model_validate(
            _with_additional(["rmse", "mae", "mae", "neg_root_mean_squared_error"])
        )
    # The main metric ('rmse' and its alias) is excluded, the 'mae' duplicate is removed
    assert cfg.general.additional_metrics == ["mae"]
    assert len(caplog.text) > 0
    assert "comparison metric" in caplog.text


def test_additional_metrics_works_with_default_comparison():
    """The default comparison_metric='r2' is also excluded from additional_metrics."""
    data = {
        "general": {
            "phases": [
                {"name": "p", "n_trials": 1},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "additional_metrics": ["r2", "mae"],
        },
        "algorithms": {"elasticnet": {"enable": True}},
    }
    cfg = Config.model_validate(data)
    assert cfg.general.additional_metrics == ["mae"]


def test_general_cfg_additional_metrics_field():
    """The field exists in GeneralCfg and is empty by default."""
    cfg = GeneralCfg(phases=[])
    assert cfg.additional_metrics == []


# ──────────────────────────────────────────────────────────────────────────────
#  2. ModelTrainer: computing additional metrics
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {"a": np.arange(50, dtype=float), "b": np.arange(50, dtype=float) * 2}
    )
    y = pd.Series(X["a"] * 1.5 + X["b"] * -0.5 + 1.0)
    return X, y


def test_trainer_computes_additional_scores(simple_regression_data):
    """Additional metrics are computed for the final model."""
    X, y = simple_regression_data
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["rmse", "mae", "mse"],
    )
    trainer.fit(X, y)

    assert set(trainer.additional_scores.keys()) == {"rmse", "mae", "mse"}
    # Error metrics are returned as positive "honest" values
    assert trainer.additional_scores["rmse"] >= 0
    assert trainer.additional_scores["mae"] >= 0
    assert trainer.additional_scores["mse"] >= 0
    assert all(isinstance(v, float) for v in trainer.additional_scores.values())


def test_trainer_additional_scores_match_main_metric_when_same(
    simple_regression_data,
):
    """For the same metric the value matches the main one (r2)."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", metric="r2", additional_metrics=["r2"])
    trainer.fit(X, y)
    assert trainer.additional_scores["r2"] == trainer.val_score


def test_trainer_error_metric_additional_values(simple_regression_data):
    """Errors (rmse) in additional metrics are returned unsigned, like val_score."""
    X, y = simple_regression_data
    trainer = ModelTrainer(
        algorithm="ridge", metric="rmse", additional_metrics=["r2", "nrmse"]
    )
    trainer.fit(X, y)
    assert trainer.val_score >= 0
    assert trainer.additional_scores["r2"] <= 1.0
    assert trainer.additional_scores["nrmse"] >= 0


def test_trainer_non_finite_additional_value_kept():
    """inf (e.g. NRMSE with a constant target) is returned as is."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([5.0, 5.0, 5.0])
    trainer = ModelTrainer(
        algorithm="ridge", metric="mse", additional_metrics=["nrmse"]
    )
    trainer.fit(X, y)
    assert trainer.additional_scores["nrmse"] == float("inf")
    assert trainer.val_score is not None  # training finished normally


def test_trainer_additional_metric_failure_skipped_with_warning(caplog):
    """A failure of a single additional metric does not interrupt training."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    real_scorer_factory = __import__(
        "configurable_automl_engine.trainer", fromlist=["get_scorer_object"]
    ).get_scorer_object

    def flaky_scorer_factory(name: str, global_y=None):
        if name == "mae":
            raise RuntimeError("scorer exploded")
        return real_scorer_factory(name, global_y)

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            side_effect=flaky_scorer_factory,
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge", metric="r2", additional_metrics=["mae", "rmse"]
        )
        trainer.fit(X, y)

    assert trainer.val_score is not None
    assert "rmse" in trainer.additional_scores  # the successful metric was computed
    assert "mae" not in trainer.additional_scores  # the failed one was skipped
    assert "could not be computed" in caplog.text


def test_trainer_additional_metric_none_skipped_with_warning(caplog):
    """A scorer that returns None for an additional metric is skipped with a log."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    real_scorer_factory = __import__(
        "configurable_automl_engine.trainer", fromlist=["get_scorer_object"]
    ).get_scorer_object

    def none_scorer_factory(name: str, global_y=None):
        if name == "r2":
            return lambda est, Xv, yv: None
        return real_scorer_factory(name, global_y)

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            side_effect=none_scorer_factory,
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge", metric="mse", additional_metrics=["r2", "rmse"]
        )
        trainer.fit(X, y)

    assert trainer.val_score is not None
    assert "r2" not in trainer.additional_scores
    assert "rmse" in trainer.additional_scores
    assert "Scorer returned None for additional metric 'r2'" in caplog.text


def test_trainer_no_additional_metrics_by_default(simple_regression_data):
    """Without the parameter additional_scores is empty and behavior is unchanged."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", metric="r2")
    trainer.fit(X, y)
    assert trainer.additional_scores == {}
    assert trainer.val_score is not None


@pytest.mark.parametrize(
    "bad_value",
    ["r2", b"r2", {"r2": 1}, [1, 2], [None], ("r2", 1), {1, 2}, [1.5]],
)
def test_trainer_invalid_additional_metrics_type_rejected(bad_value):
    """An invalid additional_metrics type is rejected in the constructor.

    Scalar strings are rejected (character-wise parsing is not allowed), as are
    mappings (iteration over keys is implicit) and any iterables containing
    non-string elements.
    """
    with pytest.raises(
        TrainingError, match="additional_metrics must be an iterable of strings"
    ):
        ModelTrainer(additional_metrics=bad_value)  # type: ignore[arg-type]


def test_trainer_additional_metrics_accepts_tuple(simple_regression_data):
    """A tuple of strings is accepted and normalized like a list."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", additional_metrics=("R2", "mae"))
    assert trainer.additional_metrics == ["r2", "mae"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"r2", "mae"}


def test_trainer_additional_metrics_accepts_set(simple_regression_data):
    """A set of strings is accepted (order does not matter)."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", additional_metrics={"rmse", "mse"})
    assert sorted(trainer.additional_metrics) == ["mse", "rmse"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"rmse", "mse"}


def test_trainer_additional_metrics_accepts_generator(simple_regression_data):
    """A generator of strings is accepted and materialized into a list."""
    X, y = simple_regression_data
    trainer = ModelTrainer(
        algorithm="ridge",
        additional_metrics=(m for m in ["r2", "mae"]),
    )
    assert trainer.additional_metrics == ["r2", "mae"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"r2", "mae"}


def test_trainer_additional_metrics_accepts_empty_tuple():
    """An empty tuple is equivalent to no metrics."""
    trainer = ModelTrainer(additional_metrics=())
    assert trainer.additional_metrics == []
    assert trainer.additional_scores == {}


def test_trainer_additional_metrics_lowercased(simple_regression_data):
    """Additional metric names are normalized to lowercase."""
    X, y = simple_regression_data
    trainer = ModelTrainer(algorithm="ridge", additional_metrics=["R2", "MAE"])
    assert trainer.additional_metrics == ["r2", "mae"]
    trainer.fit(X, y)
    assert set(trainer.additional_scores.keys()) == {"r2", "mae"}


# ──────────────────────────────────────────────────────────────────────────────
#  3. train_best_model / _fit_and_save
# ──────────────────────────────────────────────────────────────────────────────


def _component_config(additional: list[str] | None = None) -> dict:
    general: dict = {
        "comparison_metric": "rmse",
        "validation_strategy": "train_test_split",
        "phases": [
            {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
            {
                "name": "final_select",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
        "path_to_model": "model.pkl",
    }
    if additional is not None:
        general["additional_metrics"] = additional
    return {
        "general": general,
        "algorithms": {
            "elasticnet": {
                "enable": True,
                "tuner": "unittest.mock",
                "trainer_module": "unittest.mock",
            }
        },
        "oversampling": {"enable": False},
    }


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0], "target": [0.0, 1.0, 0.0, 1.0]})


def test_train_best_model_returns_additional_metrics(tiny_df):
    """The training result is enriched with additional metric values."""
    fake_trainer = SimpleNamespace(additional_scores={"r2": 0.87, "mae": 0.12})
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            # comparison_metric='rmse': "raw" scorer value -RMSE,
            # result["score"] is the positive RMSE
            return_value=PhaseResult(score=-0.9, params={"alpha": 0.1}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=fake_trainer,
        ) as mock_save,
    ):
        result = train_best_model(
            config=_component_config(additional=["r2", "mae"]),
            df=tiny_df,
            target="target",
        )

    assert result["additional_metrics"] == {"r2": 0.87, "mae": 0.12}
    assert result["score"] == 0.9
    assert result["metric"] == "rmse"
    mock_save.assert_called_once()
    # Additional metrics are forwarded from the config to the final training
    cfg_passed = mock_save.call_args.args[6]
    assert cfg_passed.general.additional_metrics == ["r2", "mae"]


def test_train_best_model_no_additional_key_when_not_configured(tiny_df):
    """Without the parameter the results match the current behavior (no key)."""
    fake_trainer = SimpleNamespace(additional_scores={})
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=PhaseResult(score=-0.9, params={}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=fake_trainer,
        ),
    ):
        result = train_best_model(
            config=_component_config(), df=tiny_df, target="target"
        )

    assert "additional_metrics" not in result
    # The honest-score report keys (issue #161): oof_score is omitted only
    # when no OOF vector is available (a fake trainer here); n_trials_evaluated
    # is reported only when the count is known (a legacy tuner returning the
    # triple reports an UNKNOWN count → the key is omitted).
    assert set(result.keys()) == {
        "algorithm",
        "score",
        "metric",
        "params",
        "model_path",
    }
    assert "n_trials_evaluated" not in result


def test_train_best_model_no_key_when_additional_empty(tiny_df):
    """An empty additional metrics list — behavior stays as it is now."""
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=PhaseResult(score=-0.9, params={}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=SimpleNamespace(additional_scores={}),
        ),
    ):
        result = train_best_model(
            config=_component_config(additional=[]), df=tiny_df, target="target"
        )
    assert "additional_metrics" not in result


def test_train_best_model_comparison_only_not_duplicated(tiny_df):
    """A metric matching the main one is not duplicated in results."""
    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=PhaseResult(score=-0.9, params={}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=SimpleNamespace(additional_scores={}),
        ),
    ):
        result = train_best_model(
            config=_component_config(additional=["rmse"]), df=tiny_df, target="target"
        )
    # The rmse value is present exactly once — as score
    assert result["score"] == 0.9
    assert "additional_metrics" not in result


def test_additional_metrics_do_not_affect_winner_selection(tiny_df):
    """The winner is determined only by the main comparison metric."""
    cfg = _component_config(additional=["r2", "mae"])
    cfg["algorithms"] = {
        "elasticnet": {
            "enable": True,
            "tuner": "unittest.mock",
            "trainer_module": "unittest.mock",
        },
        "ridge": {
            "enable": True,
            "tuner": "unittest.mock",
            "trainer_module": "unittest.mock",
        },
    }

    def hpo_side_effect(**kwargs):
        # ridge gives the best MAE (0.60) — it must win.
        # "Raw" scorer values are negative (-MAE): maximization
        # picks the larger one (-0.60 > -0.95).
        if kwargs["algo_name"] == "ridge":
            return PhaseResult(score=-0.60, params={"alpha": 0.5})
        return PhaseResult(score=-0.95, params={"alpha": 0.1})

    with (
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            side_effect=hpo_side_effect,
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=SimpleNamespace(additional_scores={"r2": 0.9, "mae": 0.1}),
        ),
    ):
        result = train_best_model(config=cfg, df=tiny_df, target="target")

    assert result["algorithm"] == "ridge"
    assert result["score"] == 0.60
    assert result["additional_metrics"] == {"r2": 0.9, "mae": 0.1}


def test_fit_and_save_forwards_additional_metrics(tmp_path, tiny_df):
    """_fit_and_save forwards additional_metrics to ModelTrainer."""

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {"r2": 0.5}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    fake_module = SimpleNamespace(ModelTrainer=FakeTrainer)

    cfg = Config.model_validate(_component_config(additional=["r2", "mae"]))
    algo_cfg = cfg.algorithms.elasticnet

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=fake_module,
    ):
        trainer = _fit_and_save(
            "elasticnet",
            algo_cfg,
            tiny_df.drop(columns="target"),
            tiny_df["target"],
            {},
            tmp_path / "m.pkl",
            cfg,
        )

    assert isinstance(trainer, FakeTrainer)
    assert trainer.kwargs["additional_metrics"] == ["r2", "mae"]


def test_fit_and_save_forwards_empty_additional_metrics(tmp_path, tiny_df):
    """When no additional metrics are set, an empty list is passed."""

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    fake_module = SimpleNamespace(ModelTrainer=FakeTrainer)
    cfg = Config.model_validate(_component_config())
    algo_cfg = cfg.algorithms.elasticnet

    with patch(
        "configurable_automl_engine.training_engine.component._load_module",
        return_value=fake_module,
    ):
        trainer = _fit_and_save(
            "elasticnet",
            algo_cfg,
            tiny_df.drop(columns="target"),
            tiny_df["target"],
            {},
            tmp_path / "m.pkl",
            cfg,
        )

    assert trainer.kwargs["additional_metrics"] == []


# ──────────────────────────────────────────────────────────────────────────────
#  4. End-to-end test with real training
# ──────────────────────────────────────────────────────────────────────────────


def test_e2e_train_best_model_with_additional_metrics(tmp_path):
    """End-to-end scenario: additional metrics are returned for the final model."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "a": np.arange(80, dtype=float),
            "b": rng.normal(size=80),
            "target": np.arange(80, dtype=float) * 2.5 + rng.normal(size=80),
        }
    )
    model_path = tmp_path / "model.pkl"
    cfg = {
        "general": {
            "comparison_metric": "rmse",
            "additional_metrics": ["r2", "mae"],
            "path_to_model": str(model_path),
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            }
        },
    }

    result = train_best_model(config=cfg, df=df, target="target")

    assert result["algorithm"] == "ridge"
    assert isinstance(result["score"], float)
    assert model_path.exists()
    assert set(result["additional_metrics"].keys()) == {"r2", "mae"}
    assert "rmse" not in result["additional_metrics"]
    assert all(isinstance(v, float) for v in result["additional_metrics"].values())
