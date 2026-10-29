"""E2E tests: oversampling on continuous regression targets.

Before the fix, enabling ``oversampling.enable: true`` on a dataset with a
continuous target crashed every HPO trial with
``ValueError: Unknown label type: continuous`` (every unique ``y`` value was
treated as a separate class by imbalanced-learn), which made
``train_best_model`` fail with "No algorithms produced valid scores".

These tests verify that all three oversampling algorithms (random, smote,
adasyn) work end-to-end on continuous ``y`` through ``train_best_model``,
and that the oversampler-level behavior (original rows keep their real
target values, synthetic rows get plausible continuous values) is correct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.training_engine import train_best_model


def _make_regression_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic regression dataset with a truly continuous target."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "num": num,
            "cat": rng.choice(["red", "green", "blue"], size=n),
        }
    )
    df["target"] = (
        2.0 * num + 1.0 * (df["cat"] == "green") + rng.normal(0, 0.3, size=n)
    )
    return df


def _make_config(model_path: Path, *, os_algorithm: str) -> dict[str, Any]:
    """Build a train_best_model config with oversampling enabled."""
    return {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "serialization_format": "pickle",
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
        "oversampling": {
            "enable": True,
            "multiplier": 1.5,
            "algorithm": os_algorithm,
        },
    }


@pytest.mark.parametrize("os_algorithm", ["random", "smote", "adasyn"])
def test_train_best_model_continuous_target_with_oversampling(
    tmp_path: Path, os_algorithm: str
) -> None:
    """train_best_model trains successfully on a continuous target with
    each oversampling algorithm."""
    df = _make_regression_df()
    model_path = tmp_path / "models" / f"{os_algorithm}.pkl"
    config = _make_config(model_path, os_algorithm=os_algorithm)

    result = train_best_model(config=config, df=df, target="target")

    assert isinstance(result, dict)
    assert result["algorithm"] == "elasticnet"
    assert result["score"] is not None
    assert np.isfinite(result["score"])
    assert Path(result["model_path"]).exists()


def test_train_best_model_continuous_target_no_oversampling(tmp_path: Path) -> None:
    """The same config with oversampling disabled still works (regression
    sanity check for the test harness itself)."""
    df = _make_regression_df()
    model_path = tmp_path / "models" / "no_os.pkl"
    config = _make_config(model_path, os_algorithm="random")
    config["oversampling"] = {"enable": False}

    result = train_best_model(config=config, df=df, target="target")

    assert result["score"] is not None
    assert np.isfinite(result["score"])


@pytest.mark.parametrize("os_algorithm", ["random", "smote", "adasyn"])
def test_oversampler_continuous_target_values(
    os_algorithm: str,
) -> None:
    """The oversampler itself: original rows keep their exact target values,
    the result is bigger, and the synthetic values are plausible continuous
    numbers (within the observed target range)."""
    df = _make_regression_df()
    sampler = DataOversampler(
        algorithm=os_algorithm, multiplier=1.5, random_state=42
    )

    res = sampler.oversample(df, target="target")

    n_orig = len(df)
    assert len(res) > n_orig
    # Original rows keep their real target values (row order is preserved).
    assert np.array_equal(
        res["target"].iloc[:n_orig].to_numpy(), df["target"].to_numpy()
    )
    # Synthetic values are finite and stay within the observed range.
    synth = res["target"].iloc[n_orig:]
    assert np.isfinite(synth).all()
    assert synth.between(df["target"].min(), df["target"].max()).all()


def test_oversampler_continuous_target_with_nan() -> None:
    """NaN targets pass through: rows with NaN are never synthesized and the
    NaN values are preserved."""
    df = _make_regression_df(n=150, seed=7)
    df.loc[df.index[:15], "target"] = np.nan

    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")

    n_orig = len(df)
    assert len(res) > n_orig
    # All original rows (including the NaN ones) are present first.
    assert res["target"].iloc[:n_orig].isna().sum() == 15
    # No new NaN was generated.
    assert res["target"].iloc[n_orig:].notna().all()


def _make_count_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A count (integer-valued Poisson) regression dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(["red", "green", "blue"], size=n),
        }
    )
    df["target"] = (
        np.floor(2.5 + 0.8 * df["num"] + (df["cat"] == "green")).clip(lower=0).astype(int)
    )
    return df


@pytest.mark.parametrize("os_algorithm", ["random", "smote", "adasyn"])
def test_train_best_model_count_target_with_oversampling(
    tmp_path: Path, os_algorithm: str
) -> None:
    """train_best_model trains successfully on an INTEGER (count) target with
    oversampling: the integer target must take the regression path instead of
    being treated as a class label (which crashed SMOTE/ADASYN on singleton
    classes)."""
    df = _make_count_df()
    model_path = tmp_path / "models" / f"{os_algorithm}.pkl"
    config = _make_config(model_path, os_algorithm=os_algorithm)

    result = train_best_model(config=config, df=df, target="target")

    assert isinstance(result, dict)
    assert result["algorithm"] == "elasticnet"
    assert result["score"] is not None
    assert np.isfinite(result["score"])
    assert Path(result["model_path"]).exists()


@pytest.mark.parametrize("os_algorithm", ["random", "smote", "adasyn"])
def test_train_best_model_diabetes_target_with_oversampling(
    tmp_path: Path, os_algorithm: str
) -> None:
    """train_best_model succeeds on the sklearn diabetes target (214 unique
    float values) with all three oversampling algorithms."""
    from sklearn.datasets import load_diabetes

    df = load_diabetes(as_frame=True).frame
    model_path = tmp_path / "models" / f"{os_algorithm}.pkl"
    config = _make_config(model_path, os_algorithm=os_algorithm)

    result = train_best_model(config=config, df=df, target="target")

    assert result["algorithm"] == "elasticnet"
    assert result["score"] is not None
    assert np.isfinite(result["score"])


def test_oversampler_diabetes_honest_synthetic_targets() -> None:
    """Diabetes at the oversampler level: random restores real source values
    (not bin midpoints); smote interpolates between real rows."""
    from sklearn.datasets import load_diabetes

    df = load_diabetes(as_frame=True).frame

    # Random: every synthetic y is a real source value; far more distinct
    # values than the 8 quantile bins.
    s_random = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    res_random = s_random.oversample(df, target="target")
    n = len(df)
    synth_random = res_random["target"].iloc[n:]
    assert set(synth_random) <= set(df["target"])
    assert synth_random.nunique() > 50

    # SMOTE: synthetic y values interpolate between parents, so they stay
    # within the observed range and are nearly all distinct.
    s_smote = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    res_smote = s_smote.oversample(df, target="target")
    synth_smote = res_smote["target"].iloc[n:]
    assert synth_smote.between(df["target"].min(), df["target"].max()).all()
    assert synth_smote.nunique() >= 0.8 * len(synth_smote)

@pytest.mark.parametrize("experimental", [True, False])
def test_experimental_warning_logged_once_per_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch, experimental: bool
) -> None:
    """Robustness scenario "oversampling on continuous target": a run with
    3 trials x 5 folds logs the experimental WARNING exactly once from the
    sampler plus once from config validation (<= 2 total)."""
    import logging

    import configurable_automl_engine.oversampling.experimental as os_mod

    monkeypatch.setattr(os_mod, "OVERSAMPLING_EXPERIMENTAL", experimental)
    df = _make_regression_df()
    config = _make_config(tmp_path / "models" / "warn.pkl", os_algorithm="random")
    config["general"]["validation_strategy"] = "k_fold"
    config["general"]["n_folds"] = 5
    config["general"]["phases"][0]["n_trials"] = 3

    with caplog.at_level(logging.WARNING):
        train_best_model(config=config, df=df, target="target")

    experimental_records = [
        r for r in caplog.records if "Oversampling is EXPERIMENTAL" in r.getMessage()
    ]
    if not experimental:
        assert experimental_records == []
        return
    sampler_records = [
        r for r in experimental_records if r.name == "configurable_automl_engine.oversampling.experimental"
    ]
    config_records = [
        r
        for r in experimental_records
        if r.name == "configurable_automl_engine.training_engine.component"
    ]
    assert len(sampler_records) == 1
    assert len(config_records) == 1
    assert len(experimental_records) <= 2
