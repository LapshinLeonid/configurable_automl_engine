"""Tests of excluding datetime-derived columns from winsorization (8.7).

Cover the ``Winsorizer(exclude=...)`` parameter (name-based exclusion with
``(-inf, +inf)`` bounds, positions for nameless input, validation errors),
the ``build_preprocessor(winsorizer_exclude=...)`` wiring (other numeric
columns stay clipped, the output column order is unchanged), the ModelTrainer
path (datetime expansion registered before fit excludes the component
columns; the artifact reproduces the behaviour after save/load) and the HPO
path (``HpoRequest.winsorizer_exclude`` reaches the trial preprocessor).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.settings import EncodingSettings
from configurable_automl_engine.hpo.request import HpoRequest
from configurable_automl_engine.preprocessing.builder import build_preprocessor
from configurable_automl_engine.preprocessing.outliers import Winsorizer
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.trainer_data import build_trainer_preprocessor

# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────


def _datetime_frame(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    """A frame with a datetime column expanded into bounded components.

    ``a`` is heavy-tailed (winsorization matters), the ``ts_*`` components
    are bounded by construction and carry an out-of-range value that would
    be clipped without the exclusion."""
    rng = np.random.RandomState(0)
    a = rng.lognormal(size=n)
    a[5] = 1e9  # extreme value that must stay clipped
    ts = pd.date_range("2019-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "a": a,
            "ts_year": float(ts[0].year),
            "ts_month": np.tile(np.arange(1, 13), n // 12).astype(float),
            "ts_day": (np.arange(n) % 28 + 1).astype(float),
            "b": rng.normal(size=n),
        }
    )
    df.loc[0, "ts_year"] = 2099.0  # would be clipped without the exclusion
    df.loc[1, "ts_month"] = 99.0
    y = pd.Series(a * 0.5 + rng.rand(n) * 0.1)
    return df, y


def _num_pipeline_steps(pre: Any) -> list[str]:
    """Names of the steps of the numerical transformer of a preprocessor."""
    for _, transformer, columns in pre.transformers:
        if isinstance(columns, str) or not hasattr(transformer, "steps"):
            continue
        step_names = [name for name, _ in transformer.steps]
        if "imputer" in step_names:
            return step_names
    return []


# ──────────────────────────────────────────────────────────────────────────
#  Winsorizer(exclude=...)
# ──────────────────────────────────────────────────────────────────────────


def test_excluded_columns_get_infinite_bounds_and_pass_through():
    """Excluded columns get (-inf, +inf) bounds and pass transform unchanged."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "ts_month": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.exclude = ["ts_month"]
    wins.fit(df)

    assert wins.lower_bounds_[1] == -np.inf
    assert wins.upper_bounds_[1] == np.inf
    assert wins.lower_bounds_[0] > -np.inf
    assert wins.upper_bounds_[0] < np.inf

    out = wins.transform(pd.DataFrame({"a": [0.0, 9.0], "ts_month": [-5.0, 99.0]}))
    assert out[0, 0] == wins.lower_bounds_[0]
    assert out[1, 0] == wins.upper_bounds_[0]
    np.testing.assert_array_equal(out[:, 1], [-5.0, 99.0])


def test_excluded_columns_do_not_shift_the_bounds_of_others():
    """The bounds of the non-excluded columns are computed from their data."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "ts_day": [1.0] * 8,
        }
    )
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.exclude = ["ts_day"]
    wins.fit(df)
    ref = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    ref.fit(df[["a"]])
    assert wins.lower_bounds_[0] == ref.lower_bounds_[0]
    assert wins.upper_bounds_[0] == ref.upper_bounds_[0]


def test_exclude_accepts_positions_for_nameless_input():
    """An integer exclude entry is the column position for ndarray input."""
    X = np.arange(40, dtype=float).reshape(20, 2)
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.exclude = [1]
    wins.fit(X)
    assert wins.lower_bounds_[1] == -np.inf
    assert wins.upper_bounds_[1] == np.inf
    assert wins.lower_bounds_[0] > -np.inf


def test_exclude_unknown_name_raises():
    """A name that matches no input column raises ValueError."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    wins = Winsorizer()
    wins.exclude = ["ts_month"]
    with pytest.raises(ValueError, match="does not match any input column"):
        wins.fit(df)


def test_exclude_out_of_range_position_raises():
    """An integer exclude entry outside the column range raises ValueError."""
    X = np.arange(8, dtype=float).reshape(4, 2)
    wins = Winsorizer()
    wins.exclude = [2]
    with pytest.raises(ValueError, match="out of range"):
        wins.fit(X)


def test_exclude_invalid_entry_type_raises():
    """A non-name, non-position exclude entry raises ValueError."""
    X = np.arange(8, dtype=float).reshape(4, 2)
    wins = Winsorizer()
    wins.exclude = [1.5]
    with pytest.raises(ValueError, match="must be column names or positions"):
        wins.fit(X)


def test_excluded_columns_named_in_the_degenerate_warning(caplog):
    """Excluded columns appear in the WARNING about unclipped columns."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "ts_month": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.exclude = ["ts_month"]
    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.preprocessing.outliers"
    ):
        wins.fit(df)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "ts_month" in warnings[0].getMessage()


# ──────────────────────────────────────────────────────────────────────────
#  build_preprocessor(winsorizer_exclude=...)
# ──────────────────────────────────────────────────────────────────────────


def test_builder_excludes_datetime_columns_and_clips_the_rest():
    """With outliers enabled the datetime components pass through unchanged.

    Every other numeric column is clipped and the output column order stays
    unchanged (one-to-one winsorizer)."""
    df, _ = _datetime_frame()
    pre = build_preprocessor(
        list(df.columns),
        [],
        ["a", "ts_year", "ts_month", "ts_day", "b"],
        outliers="quantile",
        scaling="none",
        winsorizer_exclude=["ts_year", "ts_month", "ts_day"],
    )
    assert _num_pipeline_steps(pre) == ["winsorizer", "imputer"]
    out = pre.fit_transform(df)
    wins = pre.named_transformers_["num"].named_steps["winsorizer"]

    # The excluded columns keep their values (including the out-of-range ones).
    np.testing.assert_array_equal(out[:, 1], df["ts_year"].to_numpy())
    np.testing.assert_array_equal(out[:, 2], df["ts_month"].to_numpy())
    np.testing.assert_array_equal(out[:, 3], df["ts_day"].to_numpy())
    # The heavy-tailed column is still clipped at the learned bounds.
    assert out[5, 0] == wins.upper_bounds_[0]
    assert out[5, 0] < df["a"].iloc[5]


def test_builder_without_exclude_clips_datetime_columns():
    """Without ``winsorizer_exclude`` the bounded components are clipped."""
    df, _ = _datetime_frame()
    pre = build_preprocessor(
        list(df.columns),
        [],
        ["a", "ts_year", "ts_month", "ts_day", "b"],
        outliers="quantile",
        scaling="none",
    )
    out = pre.fit_transform(df)
    wins = pre.named_transformers_["num"].named_steps["winsorizer"]
    # The month column is not constant: without the exclusion it gets real
    # bounds and the out-of-range value 99 was clipped down.
    assert out[1, 2] == wins.upper_bounds_[2]
    assert out[1, 2] < 99.0


def test_builder_logs_the_excluded_columns(caplog):
    """One INFO line names the excluded columns."""
    df, _ = _datetime_frame()
    with caplog.at_level(
        logging.INFO, logger="configurable_automl_engine.preprocessing.builder"
    ):
        pre = build_preprocessor(
            list(df.columns),
            [],
            ["a", "ts_year", "ts_month", "ts_day", "b"],
            outliers="quantile",
            scaling="none",
            winsorizer_exclude=["ts_year", "ts_month", "ts_day"],
        )
        pre.fit(df)
    infos = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "ts_month" in r.getMessage()
    ]
    assert infos, "expected an INFO line naming the excluded columns"


def test_builder_exclude_unknown_name_raises():
    """A numerical name missing from ``feature_names`` raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown winsorizer_exclude"):
        build_preprocessor(
            ["a"],
            [],
            ["a", "ts_month"],
            outliers="quantile",
            winsorizer_exclude=["ts_month"],
        )


def test_builder_exclude_non_numerical_name_is_ignored():
    """A categorical name in ``winsorizer_exclude`` is ignored (DEBUG only)."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["x"] * 6,
        }
    )
    pre = build_preprocessor(
        ["a", "cat"],
        ["cat"],
        ["a"],
        outliers="quantile",
        scaling="none",
        winsorizer_exclude=["cat"],
    )
    out = pre.fit_transform(df)
    wins = pre.named_transformers_["num"].named_steps["winsorizer"]
    assert not wins.exclude
    assert out.shape[1] == df.shape[1]  # a + encoded cat


def test_builder_exclude_without_outliers_is_harmless():
    """``winsorizer_exclude`` without ``outliers`` keeps the step absent."""
    pre = build_preprocessor(
        ["a"],
        [],
        ["a"],
        winsorizer_exclude=["a"],
    )
    assert _num_pipeline_steps(pre) == ["imputer", "scaler"]


# ──────────────────────────────────────────────────────────────────────────
#  Trainer path: datetime expansion excluded from winsorization
# ──────────────────────────────────────────────────────────────────────────


def test_build_trainer_preprocessor_forwards_exclude():
    """``build_trainer_preprocessor`` forwards the exclusion to the builder."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "a": rng.lognormal(size=48),
            "ts_month": np.tile(np.arange(1, 13), 4).astype(float),
        }
    )
    setup = build_trainer_preprocessor(
        "ridge",
        [],
        ["a", "ts_month"],
        feature_names=["a", "ts_month"],
        encoding=EncodingSettings(encoding="one_hot"),
        override={"outliers": "quantile"},
        random_state=42,
        winsorizer_exclude=["ts_month"],
    )
    setup.preprocessor.fit(df)
    wins = setup.preprocessor.named_transformers_["num"].named_steps["winsorizer"]
    assert list(wins.exclude) == ["ts_month"]


def test_trainer_datetime_expansion_registered_before_fit():
    """An expansion registered before fit survives fit and drives the exclude."""
    rng = np.random.RandomState(0)
    n = 240
    a = rng.lognormal(size=n)
    a[5] = 1e9
    df = pd.DataFrame(
        {
            "a": a,
            "ts_year": (1800 + np.arange(n) % 200).astype(float),
            "ts_month": np.tile(np.arange(1, 13), n // 12).astype(float),
            "b": rng.normal(size=n),
        }
    )
    y = pd.Series(a * 0.5 + rng.rand(n) * 0.1)

    trainer = ModelTrainer(
        algorithm="ridge",
        preprocessing_override={"outliers": "quantile"},
    )
    trainer.register_datetime_expansion({"ts": ["ts_year", "ts_month"]})
    trainer.fit(df, y)
    assert trainer.datetime_expansion_ == {"ts": ["ts_year", "ts_month"]}
    wins = (
        trainer.pipeline.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["winsorizer"]
    )
    assert list(wins.exclude) == ["ts_year", "ts_month"]


def test_trainer_save_load_reproduces_exclude_behaviour(tmp_path):
    """A saved artifact reproduces the same behaviour after load."""
    rng = np.random.RandomState(0)
    n = 240
    a = rng.lognormal(size=n)
    a[5] = 1e9
    df = pd.DataFrame(
        {
            "a": a,
            "ts_month": np.tile(np.arange(1, 13), n // 12).astype(float),
            "b": rng.normal(size=n),
        }
    )
    y = pd.Series(a * 0.5 + rng.rand(n) * 0.1)

    trainer = ModelTrainer(
        algorithm="ridge",
        preprocessing_override={"outliers": "quantile"},
    )
    trainer.register_datetime_expansion({"ts": ["ts_month"]})
    trainer.fit(df, y)
    path = tmp_path / "model.pkl"
    trainer.save(path)

    loaded = ModelTrainer.load(path)
    wins_loaded = (
        loaded.pipeline.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["winsorizer"]
    )
    assert list(wins_loaded.exclude) == ["ts_month"]

    pre = trainer.pipeline.named_steps["preprocessor"]
    pre_loaded = loaded.pipeline.named_steps["preprocessor"]
    np.testing.assert_allclose(pre.transform(df), pre_loaded.transform(df))


# ──────────────────────────────────────────────────────────────────────────
#  HPO path: the exclusion reaches the trial preprocessor
# ──────────────────────────────────────────────────────────────────────────


def test_hpo_request_default_has_no_exclusion():
    """``HpoRequest.winsorizer_exclude`` defaults to ``None``."""
    assert HpoRequest(algo_name="ridge").winsorizer_exclude is None
