"""Tests for input-data diagnostics: ``validate_features`` and
``check_target_constant`` (issue #156).

The robustness matrix rows R03 (duplicate column names), R06 (inf in numeric
features, warning part), R08 (datetime columns) and R17 (ID-like columns), as
well as the "constant target" row (T04), are covered here both at the level of
the standalone helpers (:mod:`~configurable_automl_engine.common.validation_utils`)
and at the pipeline entry points (``prepare_dataset`` / ``ModelTrainer.fit``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from configurable_automl_engine.common.validation_utils import (
    check_target_constant,
    datetime_expansion_map,
    expand_datetime_columns,
    validate_features,
)
from configurable_automl_engine.trainer import ModelTrainer, TrainingError
from configurable_automl_engine.training_engine import train_best_model
from configurable_automl_engine.training_engine.component import (
    _run_hpo,
    prepare_dataset,
)
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    ValidationStrategy,
)
from configurable_automl_engine.tuner import _validate_data


# ──────────────────────────────────────────────────────────────────────────────
# check_target_constant
# ──────────────────────────────────────────────────────────────────────────────
def test_check_target_constant_accepts_varying_target():
    """A non-constant target passes the check."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="price")
    check_target_constant(y)  # must not raise


@pytest.mark.parametrize(
    "y",
    [
        pd.Series([5.0, 5.0, 5.0], name="y"),
        pd.Series([5, 5, 5], name="y"),
        np.array([1.0, 1.0, 1.0]),
        np.array([2, 2, 2]),
    ],
)
def test_check_target_constant_rejects_constant(y):
    """A constant target (std == 0 / single unique value) is rejected."""
    with pytest.raises(ValueError, match="constant"):
        check_target_constant(y)


def test_check_target_constant_message_names_target():
    """The error message names the target column and suggests the fix."""
    y = pd.Series([7.0, 7.0, 7.0], name="score")
    with pytest.raises(ValueError, match="'score'"):
        check_target_constant(y)


def test_check_target_constant_ignores_nan_target():
    """A target containing NaN is not diagnosed here (handled by
    ``check_target_numeric``) — the constant check must not crash on it."""
    y = pd.Series([1.0, np.nan, 1.0, 1.0])
    check_target_constant(y)  # must not raise


def test_check_target_constant_empty_no_error():
    """An empty target is not flagged as constant."""
    check_target_constant(pd.Series([], dtype=float))


# ──────────────────────────────────────────────────────────────────────────────
# validate_features — duplicate columns
# ──────────────────────────────────────────────────────────────────────────────
def test_validate_duplicate_columns_raises():
    """Duplicate column names → ValueError listing the duplicates."""
    X = pd.DataFrame(np.ones((4, 3)), columns=["a", "b", "a"])
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_features(X)
    # The message names the duplicated column
    with pytest.raises(ValueError, match=r"a"):
        validate_features(X)


def test_validate_duplicate_columns_message_suggests_fix():
    """The duplicate-columns message suggests renaming the columns."""
    X = pd.DataFrame(np.ones((3, 2)), columns=["x", "x"])
    with pytest.raises(ValueError, match="[Rr]ename"):
        validate_features(X)


def test_validate_no_duplicates_passes():
    """A frame without duplicates passes the duplicate check."""
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out, generated = validate_features(X)
    assert list(out.columns) == ["a", "b"]
    assert generated == []


# ──────────────────────────────────────────────────────────────────────────────
# validate_features — ±inf in numeric features
# ──────────────────────────────────────────────────────────────────────────────
def test_validate_inf_as_missing_replaces_with_nan(caplog):
    """Default ``as_missing`` policy: ±inf → NaN + WARNING with column/count."""
    X = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": [1.0, 2.0, 3.0]})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, generated = validate_features(X)
    assert np.isnan(out["a"].iloc[1])
    assert np.isnan(out["a"].iloc[2])
    assert out["b"].tolist() == [1.0, 2.0, 3.0]
    assert generated == []
    assert "Infinite values" in caplog.text
    assert "a" in caplog.text
    assert "2 value(s)" in caplog.text


def test_validate_inf_policy_error_raises():
    """``inf_policy='error'`` → ValueError naming the columns."""
    X = pd.DataFrame({"a": [1.0, np.inf], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="[Ii]nfinite"):
        validate_features(X, inf_policy="error")
    with pytest.raises(ValueError, match="a"):
        validate_features(X, inf_policy="error")


def test_validate_inf_policy_error_suggests_fix():
    """The inf error message suggests as_missing / removing rows."""
    X = pd.DataFrame({"a": [np.inf, 2.0], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="as_missing"):
        validate_features(X, inf_policy="error")


def test_validate_no_inf_no_warning(caplog):
    """A frame without inf produces no inf warning."""
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        validate_features(X)
    assert "Infinite values" not in caplog.text


def test_validate_inf_unknown_policy_rejected():
    """An unknown inf_policy value is rejected."""
    with pytest.raises(ValueError, match="inf_policy"):
        validate_features(pd.DataFrame({"a": [1.0]}), inf_policy="bogus")


# ──────────────────────────────────────────────────────────────────────────────
# validate_features — datetime/timedelta columns
# ──────────────────────────────────────────────────────────────────────────────
def test_validate_datetime_expand_generates_components(caplog):
    """``expand`` replaces a datetime column with year/month/day/dayofweek
    (hour is absent when all-zero) and warns listing the generated columns."""
    X = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "b": [1, 2, 3],
        }
    )
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, generated = validate_features(X)
    assert "ts" not in out.columns
    for part in ("year", "month", "day", "dayofweek"):
        assert f"ts_{part}" in out.columns
        assert f"ts_{part}" in generated
    assert "ts_hour" not in out.columns
    assert out["ts_year"].tolist() == [2024.0, 2024.0, 2024.0]
    assert out["ts_month"].tolist() == [1.0, 1.0, 1.0]
    assert out["ts_day"].tolist() == [1.0, 2.0, 3.0]
    assert out["ts_dayofweek"].tolist() == [0.0, 1.0, 2.0]
    assert "expanded" in caplog.text
    assert "ts_year" in caplog.text


def test_validate_datetime_expand_includes_hour_when_nonzero(caplog):
    """`hour` appears among the generated components only when it is non-zero."""
    X = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 23:30"])})
    out, generated = validate_features(X)
    assert "ts_hour" in out.columns
    assert "ts_hour" in generated
    assert out["ts_hour"].tolist() == [10.0, 23.0]


def test_validate_datetime_expand_nat_becomes_nan():
    """NaT values become NaN in the generated components."""
    X = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", pd.NaT, "2024-01-03"])})
    out, _ = validate_features(X)
    assert np.isnan(out["ts_year"].iloc[1])


def test_validate_datetime_drop_removes_column(caplog):
    """``drop`` removes the datetime column and warns."""
    X = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01", "2024-01-02"]), "b": [1.0, 2.0]}
    )
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, generated = validate_features(X, datetime_policy="drop")
    assert "ts" not in out.columns
    assert generated == []
    assert "dropped" in caplog.text
    assert "ts" in caplog.text


def test_validate_datetime_error_raises():
    """``error`` rejects datetime columns with a ValueError."""
    X = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    with pytest.raises(ValueError, match="datetime"):
        validate_features(X, datetime_policy="error")
    with pytest.raises(ValueError, match="ts"):
        validate_features(X, datetime_policy="error")


def test_validate_timedelta_expand():
    """Timedelta columns expand into days/seconds/microseconds."""
    X = pd.DataFrame({"td": pd.to_timedelta(["1 days 02:03:04", "2 days 00:00:05"])})
    out, generated = validate_features(X)
    assert "td" not in out.columns
    for part in ("days", "seconds", "microseconds"):
        assert f"td_{part}" in out.columns
        assert f"td_{part}" in generated
    assert out["td_days"].tolist() == [1.0, 2.0]


def test_validate_datetime_unknown_policy_rejected():
    """An unknown datetime_policy value is rejected."""
    with pytest.raises(ValueError, match="datetime_policy"):
        validate_features(pd.DataFrame({"a": [1.0]}), datetime_policy="bogus")


# ──────────────────────────────────────────────────────────────────────────────
# validate_features — ID-like columns
# ──────────────────────────────────────────────────────────────────────────────
def test_validate_id_like_object_column_warns(caplog):
    """Object/string column with nunique/n >= 0.95 → WARNING naming it."""
    X = pd.DataFrame({"id": [f"u{i}" for i in range(100)], "b": np.arange(100.0)})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, _ = validate_features(X)
    assert "ID-like" in caplog.text
    assert "id" in caplog.text
    assert "id" in out.columns  # kept by default


def test_validate_id_like_drop_when_configured():
    """``drop_id_like_columns=True`` removes the ID-like column."""
    X = pd.DataFrame({"id": [f"u{i}" for i in range(100)], "b": np.arange(100.0)})
    out, _ = validate_features(X, drop_id_like_columns=True)
    assert "id" not in out.columns
    assert "b" in out.columns


def test_validate_id_like_monotonic_integer_warns(caplog):
    """A unique monotonic integer column is ID-like."""
    X = pd.DataFrame({"idx": np.arange(50), "b": np.arange(50.0)})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, _ = validate_features(X)
    assert "ID-like" in caplog.text
    assert "idx" in caplog.text
    assert "idx" in out.columns


def test_validate_id_like_decreasing_integer_warns():
    """A unique monotonic DECREASING integer column is also ID-like."""
    X = pd.DataFrame({"idx": np.arange(50, 0, -1), "b": np.arange(50.0)})
    out, _ = validate_features(X, drop_id_like_columns=True)
    assert "idx" not in out.columns


def test_validate_id_like_non_monotonic_unique_integer_kept(caplog):
    """A unique but non-monotonic integer column is NOT flagged as ID-like."""
    X = pd.DataFrame({"idx": [3, 1, 2, 5, 4], "b": np.arange(5.0)})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, _ = validate_features(X)
    assert "ID-like" not in caplog.text
    assert "idx" in out.columns


def test_validate_id_like_low_cardinality_object_kept(caplog):
    """A low-cardinality object column is not ID-like."""
    X = pd.DataFrame({"cat": list("abcde") * 20, "b": np.arange(100.0)})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, _ = validate_features(X)
    assert "ID-like" not in caplog.text
    assert "cat" in out.columns


# ──────────────────────────────────────────────────────────────────────────────
# prepare_dataset integration (pipeline entry point)
# ──────────────────────────────────────────────────────────────────────────────
def _cfg_dict(
    *,
    inf_policy: str = "as_missing",
    datetime_policy: str = "expand",
    drop_id_like: bool = False,
) -> Config:
    return Config.model_validate(
        {
            "general": {
                "comparison_metric": "mae",
                "validation_strategy": "train_test_split",
                "n_folds": 5,
                "phases": [],
                "path_to_model": "model.pkl",
                "inf_policy": inf_policy,
                "datetime_policy": datetime_policy,
                "drop_id_like_columns": drop_id_like,
            },
            "algorithms": {
                "random_forest": {
                    "enable": True,
                    "tuner": "mock.mock_tuner",
                    "trainer_module": "mock.mock_trainer",
                }
            },
            "oversampling": {"data_oversampling": False},
        }
    )


def test_prepare_dataset_rejects_duplicate_columns():
    """prepare_dataset fails fast on duplicate column names."""
    cfg = _cfg_dict()
    df = pd.DataFrame(np.ones((4, 3)), columns=["a", "b", "a"])
    df["target"] = np.arange(4.0)
    with pytest.raises(ValueError, match="duplicate column names"):
        prepare_dataset(cfg, df, "target")


def test_prepare_dataset_inf_as_missing_replaces(caplog):
    """prepare_dataset repairs ±inf under the default policy."""
    cfg = _cfg_dict()
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0, 4.0], "target": [1, 2, 3, 4]})
    with caplog.at_level(logging.WARNING, logger="training_engine"):
        prepared = prepare_dataset(cfg, df, "target")
    assert prepared.X["a"].iloc[1] != np.inf
    assert np.isnan(prepared.X["a"].iloc[1])
    assert "Infinite values" in caplog.text


def test_prepare_dataset_inf_policy_error():
    """prepare_dataset rejects ±inf with inf_policy='error'."""
    cfg = _cfg_dict(inf_policy="error")
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0, 4.0], "target": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="[Ii]nfinite"):
        prepare_dataset(cfg, df, "target")


def test_prepare_dataset_datetime_expand(caplog):
    """prepare_dataset expands datetime columns into components.

    The two observations share the same year/month, so the constant
    components (ts_year/ts_month) are removed by the uninformative-feature
    cleaning that runs before type detection (S3) — the varying components
    (ts_day, ts_dayofweek) remain and are detected as numerical features.
    """
    cfg = _cfg_dict()
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01", "2024-01-02"]), "target": [1, 2]}
    )
    with caplog.at_level(logging.WARNING, logger="training_engine"):
        prepared = prepare_dataset(cfg, df, "target")
    assert "ts" not in prepared.X.columns
    assert "ts_day" in prepared.X.columns
    assert "ts_day" in prepared.numerical_features
    assert "expanded" in caplog.text


def test_prepare_dataset_datetime_drop():
    """prepare_dataset drops datetime columns with datetime_policy='drop'."""
    cfg = _cfg_dict(datetime_policy="drop")
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01", "2024-01-02"]), "target": [1, 2]}
    )
    prepared = prepare_dataset(cfg, df, "target")
    assert "ts" not in prepared.X.columns


def test_prepare_dataset_datetime_error():
    """prepare_dataset rejects datetime columns with datetime_policy='error'."""
    cfg = _cfg_dict(datetime_policy="error")
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01", "2024-01-02"]), "target": [1, 2]}
    )
    with pytest.raises(ValueError, match="datetime"):
        prepare_dataset(cfg, df, "target")


def test_prepare_dataset_id_like_warned_and_kept(caplog):
    """prepare_dataset warns about ID-like columns but keeps them by default."""
    cfg = _cfg_dict()
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(50)],
            "b": np.arange(50.0),
            "target": np.arange(50.0),
        }
    )
    with caplog.at_level(logging.WARNING, logger="training_engine"):
        prepared = prepare_dataset(cfg, df, "target")
    assert "ID-like" in caplog.text
    assert "id" in prepared.X.columns


def test_prepare_dataset_id_like_dropped_when_configured():
    """prepare_dataset drops ID-like columns with drop_id_like_columns=True."""
    cfg = _cfg_dict(drop_id_like=True)
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(50)],
            "b": np.arange(50.0),
            "target": np.arange(50.0),
        }
    )
    prepared = prepare_dataset(cfg, df, "target")
    assert "id" not in prepared.X.columns
    assert "b" in prepared.X.columns


def test_prepare_dataset_rejects_constant_target():
    """prepare_dataset fails fast on a constant target."""
    cfg = _cfg_dict()
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "target": [5, 5, 5, 5]})
    with pytest.raises(ValueError, match="constant"):
        prepare_dataset(cfg, df, "target")


# ──────────────────────────────────────────────────────────────────────────────
# ModelTrainer.fit integration
# ──────────────────────────────────────────────────────────────────────────────
def test_fit_rejects_duplicate_columns():
    """ModelTrainer.fit rejects duplicate column names with a clear error."""
    X = pd.DataFrame(np.ones((4, 3)), columns=["a", "b", "a"])
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(TrainingError, match="duplicate column names"):
        ModelTrainer(algorithm="ridge").fit(X, y)


def test_fit_inf_error_policy():
    """ModelTrainer.fit honors inf_policy='error'."""
    X = pd.DataFrame({"a": [1.0, np.inf, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    trainer = ModelTrainer(algorithm="ridge", inf_policy="error")
    with pytest.raises(TrainingError, match="[Ii]nfinite"):
        trainer.fit(X, y)


def test_fit_inf_as_missing_default(caplog):
    """ModelTrainer.fit replaces ±inf with NaN by default (as_missing)."""
    X = pd.DataFrame(
        {
            "a": np.concatenate([[1.0, np.inf, 3.0, 4.0], np.arange(16.0)]),
            "b": np.arange(20.0),
        }
    )
    y = pd.Series(np.arange(20, dtype=float))
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        trainer = ModelTrainer(algorithm="ridge").fit(X, y)
    assert trainer.val_score is not None
    assert "Infinite values" in caplog.text


def test_fit_datetime_expand(caplog):
    """ModelTrainer.fit expands datetime columns into components."""
    # Two different years so that no expanded component is constant and thus
    # dropped by the uninformative-feature cleaning.
    X = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2023-12-31", "2024-01-02", "2024-06-15", "2025-03-01"]
            ),
            "b": np.arange(4.0),
        }
    )
    y = pd.Series(np.arange(4.0) * 2.0 + 1.0)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        trainer = ModelTrainer(algorithm="ridge").fit(X, y)
    assert trainer.val_score is not None
    assert any(name.startswith("ts_") for name in trainer.feature_names or [])
    assert "expanded" in caplog.text


def test_fit_rejects_constant_target():
    """ModelTrainer.fit rejects a constant target."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([5.0, 5.0, 5.0, 5.0])
    with pytest.raises(TrainingError, match="constant"):
        ModelTrainer(algorithm="ridge").fit(X, y)


def test_fit_datetime_drop(caplog):
    """ModelTrainer.fit drops datetime columns with datetime_policy='drop'."""
    X = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2023-12-31", "2024-01-02", "2024-06-15", "2025-03-01"]
            ),
            "b": np.arange(4.0) * 2.0,
        }
    )
    y = pd.Series(np.arange(4.0) * 2.0 + 1.0)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        trainer = ModelTrainer(algorithm="ridge", datetime_policy="drop").fit(X, y)
    assert trainer.val_score is not None
    assert not any(name.startswith("ts_") for name in trainer.feature_names or [])
    assert "dropped" in caplog.text


def test_fit_datetime_error():
    """ModelTrainer.fit rejects datetime columns with datetime_policy='error'."""
    X = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])}
    )
    y = pd.Series(np.arange(4.0))
    with pytest.raises(TrainingError, match="datetime"):
        ModelTrainer(algorithm="ridge", datetime_policy="error").fit(X, y)


def test_fit_drop_id_like_columns():
    """ModelTrainer.fit drops ID-like columns with drop_id_like_columns=True."""
    X = pd.DataFrame({"id": [f"u{i}" for i in range(50)], "b": np.arange(50.0) * 2.0})
    y = pd.Series(np.arange(50.0) * 2.0 + 1.0)
    trainer = ModelTrainer(algorithm="ridge", drop_id_like_columns=True).fit(X, y)
    assert trainer.val_score is not None
    assert "id" not in (trainer.feature_names or [])


def test_fit_datetime_expand_name_collision_rejected():
    """B1: datetime expansion must not silently overwrite existing columns.

    If the data already contains a ``ts_year`` column next to the datetime
    ``ts`` column, the expansion would overwrite user data — reject with a
    clear error instead.
    """
    X = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2023-12-31", "2024-01-02", "2024-06-15", "2025-03-01"]
            ),
            "ts_year": [1.0, 2.0, 3.0, 4.0],
            "b": np.arange(4.0),
        }
    )
    y = pd.Series(np.arange(4.0) * 2.0 + 1.0)
    with pytest.raises(TrainingError, match="overwrite existing"):
        ModelTrainer(algorithm="ridge").fit(X, y)


def test_validate_datetime_expand_name_collision_rejected():
    """B1: the standalone validator rejects datetime-expansion collisions."""
    X = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2023-12-31", "2024-01-02", "2024-06-15", "2025-03-01"]
            ),
            "ts_year": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="overwrite existing"):
        validate_features(X)


def test_fit_id_like_warning_suppressed_by_default(caplog):
    """B1/B2: ModelTrainer.fit warns about ID-like columns once by default."""
    X = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(30)],
            "b": np.arange(30.0) * 2.0 + 1.0,
        }
    )
    y = pd.Series(np.arange(30.0) * 2.0 + 3.0)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        ModelTrainer(algorithm="ridge").fit(X, y)
    assert caplog.text.count("ID-like") == 1


def test_check_target_constant_tiny_values_pass():
    """S1: tiny non-constant values (e.g. ~1e-200) are not flagged as constant."""
    y = pd.Series([1e-200, 2e-200, 3e-200])
    check_target_constant(y)  # must not raise


def test_train_best_model_id_like_warning_emitted_once(tmp_path, caplog):
    """B2: the full engine path warns about ID-like columns exactly once.

    ``prepare_dataset`` emits the WARNING; the final fit via
    ``_build_trainer``/``ModelTrainer.fit`` must not duplicate it
    (warn_id_like=False on the engine path).
    """
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(n)],
            "b": rng.normal(size=n),
        }
    )
    df["target"] = 2.0 * df["b"] + rng.normal(0, 0.1, size=n)
    cfg = {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(tmp_path / "model.pkl"),
            "validation_strategy": "train_test_split",
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
            "elasticnet": {"enable": True},
        },
        "oversampling": {"enable": False},
    }
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        train_best_model(cfg, df, target="target")
    assert caplog.text.count("ID-like") == 1


def _engine_cfg(model_path, **general) -> dict:
    return {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "validation_strategy": "train_test_split",
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
            **general,
        },
        "algorithms": {"elasticnet": {"enable": True}},
        "oversampling": {"enable": False},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Review B4: drop_id_like_columns does not depend on warn_id_like
# ──────────────────────────────────────────────────────────────────────────────
def _id_frame(n: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"id": [f"u{i}" for i in range(n)], "b": np.arange(n) * 2.0})
    y = pd.Series(np.arange(n) * 2.0 + 1.0)
    return X, y


def test_validate_id_like_dropped_without_warning(caplog):
    """B4: drop_id_like_columns=True drops the column even with warn_id_like=False."""
    X, _ = _id_frame()
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, _ = validate_features(X, drop_id_like_columns=True, warn_id_like=False)
    assert "id" not in out.columns
    assert "b" in out.columns
    assert "ID-like" not in caplog.text


def test_validate_id_like_kept_without_warning_by_default(caplog):
    """warn_id_like=False and no drop: the column stays and nothing is logged."""
    X, _ = _id_frame()
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out, _ = validate_features(X, warn_id_like=False)
    assert "id" in out.columns
    assert "ID-like" not in caplog.text


def test_fit_drop_id_like_columns_without_warning():
    """B4: ModelTrainer.fit drops ID-like columns on the engine path (warn off)."""
    X, y = _id_frame()
    trainer = ModelTrainer(
        algorithm="ridge", drop_id_like_columns=True, warn_id_like=False
    ).fit(X, y)
    assert "id" not in (trainer.feature_names or [])


def test_tuner_validate_data_drops_id_like_columns():
    """B4: tuner._validate_data honors drop_id_like_columns=True."""
    X, y = _id_frame()
    out = _validate_data(X, y, drop_id_like_columns=True)
    assert isinstance(out, pd.DataFrame)
    assert "id" not in out.columns


# ──────────────────────────────────────────────────────────────────────────────
# Review S10: large int64 targets
# ──────────────────────────────────────────────────────────────────────────────
def test_check_target_constant_large_int64_not_constant():
    """S10: distinct int64 values above 2**53 are not collapsed by float64."""
    check_target_constant(pd.Series([2**53, 2**53 + 1], dtype="int64"))


def test_check_target_constant_large_int64_constant_rejected():
    """A truly constant large int64 target is rejected with the exact value."""
    with pytest.raises(ValueError, match=str(2**53 + 1)):
        check_target_constant(pd.Series([2**53 + 1] * 3, dtype="int64", name="y"))


def test_check_target_constant_bool_constant_rejected():
    """A constant boolean target is rejected via the native-dtype path."""
    with pytest.raises(ValueError, match="constant"):
        check_target_constant(np.array([True, True, True]))


# ──────────────────────────────────────────────────────────────────────────────
# Review S7: ModelTrainer.fit validates the target like prepare_dataset
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_non_finite_target(bad):
    """S7: NaN/±inf in y fails fast with a clear message naming the target."""
    X = pd.DataFrame({"a": np.arange(6.0)})
    y = pd.Series([1.0, 2.0, bad, 4.0, 5.0, 6.0], name="price")
    with pytest.raises(TrainingError, match="price.*NaN or infinite"):
        ModelTrainer(algorithm="ridge").fit(X, y)


def test_fit_rejects_non_numeric_target():
    """S7: a non-numeric target is rejected before the validation splits."""
    X = pd.DataFrame({"a": np.arange(4.0)})
    y = pd.Series(["a", "b", "c", "d"], name="label")
    with pytest.raises(TrainingError, match="must be numeric"):
        ModelTrainer(algorithm="ridge").fit(X, y)


# ──────────────────────────────────────────────────────────────────────────────
# Review B3 / S11: datetime expansion is re-applied at inference
# ──────────────────────────────────────────────────────────────────────────────
def _dt_frame(n: int = 40, *, with_hour: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    freq = "7h" if with_hour else "17D"
    ts = pd.date_range("2022-01-01", periods=n, freq=freq)
    df = pd.DataFrame({"ts": ts, "b": rng.normal(size=n)})
    df["target"] = 3.0 * df["b"] + 0.01 * np.arange(n) + rng.normal(0, 0.05, size=n)
    return df


def test_datetime_expansion_map_records_components():
    """The map lists exactly the generated components per source column."""
    X = _dt_frame(with_hour=True).drop(columns=["target"])
    X["dur"] = pd.to_timedelta(np.arange(len(X)), unit="h")
    _, generated = validate_features(X)
    assert datetime_expansion_map(X, generated) == {
        "ts": ["ts_year", "ts_month", "ts_day", "ts_dayofweek", "ts_hour"],
        "dur": ["dur_days", "dur_seconds", "dur_microseconds"],
    }


def test_datetime_expansion_map_empty_without_generated():
    """No generated columns → an empty map (e.g. datetime_policy='drop')."""
    X = _dt_frame().drop(columns=["target"])
    assert datetime_expansion_map(X, []) == {}


def test_expand_datetime_columns_matches_training_expansion():
    """Inference expansion reproduces validate_features' values exactly."""
    X = _dt_frame(with_hour=True).drop(columns=["target"])
    expanded, generated = validate_features(X)
    again = expand_datetime_columns(X, datetime_expansion_map(X, generated))
    pd.testing.assert_frame_equal(again[expanded.columns], expanded)


def test_expand_datetime_columns_uses_recorded_component_set():
    """'hour' recorded at training is produced even if all-zero in new data."""
    X = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    out = expand_datetime_columns(X, {"ts": ["ts_year", "ts_hour"]})
    assert list(out.columns) == ["ts_year", "ts_hour"]
    assert out["ts_hour"].tolist() == [0.0, 0.0]


def test_expand_datetime_columns_parses_object_dtype():
    """Raw string dates (object dtype) at inference are parsed first."""
    X = pd.DataFrame({"ts": ["2024-03-05", "2025-07-01"], "b": [1.0, 2.0]})
    out = expand_datetime_columns(X, {"ts": ["ts_year", "ts_month"]})
    assert out["ts_year"].tolist() == [2024.0, 2025.0]
    assert out["ts_month"].tolist() == [3.0, 7.0]
    assert "ts" not in out.columns


def test_expand_datetime_columns_unparsable_raises():
    """Values that cannot be parsed raise a ValueError naming the column."""
    X = pd.DataFrame({"ts": ["not a date", "nope"]})
    with pytest.raises(ValueError, match="'ts'"):
        expand_datetime_columns(X, {"ts": ["ts_year"]})


def test_expand_datetime_columns_noop_cases():
    """Absent source column or already-expanded data → the input is returned."""
    X = pd.DataFrame({"ts_year": [2024.0], "b": [1.0]})
    assert expand_datetime_columns(X, {"ts": ["ts_year"]}) is X
    assert expand_datetime_columns(X, {}) is X
    both = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"]), "ts_year": [1999.0]})
    assert expand_datetime_columns(both, {"ts": ["ts_year"]}) is both


def test_fit_predict_on_raw_datetime_frame(tmp_path):
    """B3: a trainer fit on raw datetime data predicts on the same raw schema."""
    df = _dt_frame()
    X, y = df.drop(columns=["target"]), df["target"]
    trainer = ModelTrainer(algorithm="ridge").fit(X, y)
    assert "ts" in trainer.datetime_expansion_
    preds = trainer.predict(X)
    assert preds.shape == (len(X),)
    # The already-expanded schema gives the same predictions.
    expanded = expand_datetime_columns(X, trainer.datetime_expansion_)
    np.testing.assert_allclose(trainer.predict(expanded), preds)
    # And so does a save/load round trip.
    path = tmp_path / "m.joblib"
    trainer.save(path)
    np.testing.assert_allclose(ModelTrainer.load(path).predict(X), preds)


def test_fit_resets_datetime_expansion_on_refit():
    """A refit on data without datetime columns clears the old expansion."""
    df = _dt_frame()
    trainer = ModelTrainer(algorithm="ridge").fit(
        df.drop(columns=["target"]), df["target"]
    )
    assert trainer.datetime_expansion_
    trainer.fit(df[["b"]], df["target"])
    assert trainer.datetime_expansion_ == {}


def test_register_datetime_expansion_merges_and_validates():
    """Upstream expansion is merged; entries from fit take precedence."""
    trainer = ModelTrainer(algorithm="ridge")
    trainer.datetime_expansion_ = {"ts": ["ts_year"]}
    trainer.register_datetime_expansion({"ts": ["ts_month"], "d": ["d_days"]})
    assert trainer.datetime_expansion_ == {"ts": ["ts_year"], "d": ["d_days"]}
    with pytest.raises(TrainingError, match="dict"):
        trainer.register_datetime_expansion({"ts": "ts_year"})


def test_predict_unparsable_datetime_raises_training_error():
    """Unparsable raw dates at inference surface as a TrainingError."""
    df = _dt_frame()
    X, y = df.drop(columns=["target"]), df["target"]
    trainer = ModelTrainer(algorithm="ridge").fit(X, y)
    bad = X.copy()
    bad["ts"] = "garbage"
    with pytest.raises(TrainingError, match="cannot be parsed"):
        trainer.predict(bad)


def test_prepare_dataset_records_datetime_expansion():
    """prepare_dataset exposes the expansion for the artifact."""
    df = _dt_frame()
    assert set(prepare_dataset(_cfg_dict(), df, "target").datetime_expansion) == {
        "ts"
    }
    prepared_drop = prepare_dataset(_cfg_dict(datetime_policy="drop"), df, "target")
    assert prepared_drop.datetime_expansion == {}


def test_train_best_model_artifact_predicts_on_raw_datetime(tmp_path):
    """B3/S11 e2e: train_best_model → load → predict on the raw input frame.

    All dates fall in one year, so ``ts_year`` is constant and removed by the
    uninformative-feature cleaning in prepare_dataset, and the ID-like column
    is dropped: the raw frame (with ``ts`` and ``id``) must still predict.
    """
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=n, freq="5D"),
            "b": rng.normal(size=n),
            "id": [f"u{i}" for i in range(n)],
        }
    )
    df["target"] = 2.0 * df["b"] + rng.normal(0, 0.1, size=n)
    model_path = tmp_path / "model.pkl"
    train_best_model(
        _engine_cfg(model_path, drop_id_like_columns=True), df, target="target"
    )
    loaded = ModelTrainer.load(model_path)
    assert "ts" in loaded.datetime_expansion_
    assert "ts_year" not in (loaded.feature_names or [])
    assert "id" not in (loaded.feature_names or [])
    raw = df.drop(columns=["target"])
    preds = loaded.predict(raw)
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))


def test_train_best_model_holdout_artifact_predicts_on_raw_datetime(tmp_path):
    """B3: the hold-out refit path (_fit_and_save) registers the expansion too."""
    rng = np.random.default_rng(4)
    n = 80
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2022-01-01", periods=n, freq="11D"),
            "b": rng.normal(size=n),
        }
    )
    df["target"] = 2.0 * df["b"] + rng.normal(0, 0.1, size=n)
    model_path = tmp_path / "model.pkl"
    train_best_model(
        _engine_cfg(model_path, holdout_fraction=0.2), df, target="target"
    )
    loaded = ModelTrainer.load(model_path)
    assert set(loaded.datetime_expansion_) == {"ts"}
    assert loaded.predict(df.drop(columns=["target"])).shape == (n,)


# ──────────────────────────────────────────────────────────────────────────────
# Review S8: the diagnostics policies are forwarded to the tuner
# ──────────────────────────────────────────────────────────────────────────────
def test_run_hpo_forwards_diagnostics_policies():
    """_run_hpo passes the configured policies to tuners that accept them."""
    received: dict[str, object] = {}

    class PolicyTuner:
        def optimize(
            self,
            algo_name,
            X,
            y,
            metric,
            n_trials,
            validation_strategy,
            inf_policy="error",
            datetime_policy="expand",
            drop_id_like_columns=False,
        ):
            received.update(
                inf_policy=inf_policy,
                datetime_policy=datetime_policy,
                drop_id_like_columns=drop_id_like_columns,
            )
            return ("model", {"p": 1}, 0.5)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.module"
    with patch("importlib.import_module", return_value=PolicyTuner()):
        _run_hpo(
            algo_name="ridge",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1.0, 2.0]}),
            y=pd.Series([1.0, 2.0]),
            metric_name_sklearn="r2",
            n_trials=1,
            validation_strategy=ValidationStrategy.train_test_split,
            inf_policy="as_missing",
            datetime_policy="drop",
            drop_id_like_columns=True,
        )
    assert received == {
        "inf_policy": "as_missing",
        "datetime_policy": "drop",
        "drop_id_like_columns": True,
    }


def test_run_hpo_omits_policies_for_tuner_without_them():
    """Tuners without the policy parameters do not receive the keys."""
    received: dict[str, object] = {}

    class PlainTuner:
        def optimize(self, algo_name, X, y, metric, n_trials, validation_strategy):
            received["called"] = True
            return ("model", {"p": 1}, 0.5)

    algo_cfg = MagicMock(spec=AlgoCfg)
    algo_cfg.tuner = "some.module"
    with patch("importlib.import_module", return_value=PlainTuner()):
        _run_hpo(
            algo_name="ridge",
            algo_cfg=algo_cfg,
            X=pd.DataFrame({"a": [1.0, 2.0]}),
            y=pd.Series([1.0, 2.0]),
            metric_name_sklearn="r2",
            n_trials=1,
            validation_strategy=ValidationStrategy.train_test_split,
            inf_policy="as_missing",
        )
    assert received == {"called": True}
