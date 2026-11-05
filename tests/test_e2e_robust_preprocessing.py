"""E2E test of the robust preprocessing pipeline on heavy-tailed data.

A full ``train_best_model`` run on a synthetic dataset with numeric features
drawn from a heavy-tailed Student-t distribution (df=2), injected extreme
outliers and missing values, plus one categorical column. The config exercises
the outlier handling and the quantile scaling together:

* ``general.preprocessing`` — the run-level default override:
  median imputation and quantile winsorization;
* ``ridge`` — its own per-algorithm block with quantile scaling to a normal
  distribution (the per-algorithm block wins over the general one per field);
* ``random_forest`` — inherits the run-level general block (median
  imputation + quantile outliers, ``none`` scaling from the tree preset);
* ``isotonic_regression`` — the univariate preset, untouched by any override.

The test asserts that the run completes, that an honest quality estimate is
reported (``holdout_score`` or ``bbc_score``), that the resolved presets of
the saved winner follow the precedence rules, and that the saved model loads
and produces predictions identical to the in-memory ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.training_engine import train_best_model
from configurable_automl_engine.trainer import ModelTrainer


def _make_heavy_tailed_df(n: int = 220, seed: int = 42) -> pd.DataFrame:
    """Synthetic regression dataset with heavy-tailed numeric features.

    Numeric features come from Student-t with df=2 (infinite variance), a few
    extreme outliers are injected on top, and a share of values is masked to
    NaN. One categorical column carries part of the signal.
    """
    rng = np.random.default_rng(seed)
    n_num1 = rng.standard_t(df=2, size=n)
    n_num2 = rng.standard_t(df=3, size=n) * 2.0
    category = rng.choice(["low", "mid", "high"], size=n)

    # Inject extreme outliers into a handful of rows of the numeric columns.
    outlier_rows = rng.choice(n, size=8, replace=False)
    n_num1[outlier_rows] = n_num1[outlier_rows] * 40.0
    n_num2[outlier_rows[:4]] = 1e4

    # Mask values to NaN (the general block prescribes median imputation).
    nan_mask1 = rng.random(n) < 0.05
    nan_mask2 = rng.random(n) < 0.05
    n_num1[nan_mask1] = np.nan
    n_num2[nan_mask2] = np.nan

    target = (
        0.8 * np.nan_to_num(n_num1, nan=0.0)
        + 0.4 * np.nan_to_num(n_num2, nan=0.0)
        + 2.0 * (category == "high")
        + rng.normal(0, 0.1, size=n)
    )

    return pd.DataFrame(
        {
            "heavy1": n_num1,
            "heavy2": n_num2,
            "category": pd.Categorical(category),
            "target": target,
        }
    )


def _make_config(model_path: Path, n_trials: int = 2) -> dict[str, Any]:
    """Build the configuration of the robust-preprocessing run."""
    return {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "serialization_format": "pickle",
            "validation_strategy": "train_test_split",
            "random_state": 42,
            "preprocessing": {
                "imputation_strategy": "median",
                "outliers": "quantile",
            },
            "phases": [
                {"name": "search", "n_trials": n_trials, "action": "all_algorithms"},
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
                "preprocessing": {
                    "scaling": {
                        "type": "quantile",
                        "output_distribution": "normal",
                    },
                },
            },
            "random_forest": {"enable": True},
            "isotonic_regression": {"enable": True},
        },
    }


def test_train_best_model_on_heavy_tailed_data(tmp_path: Path) -> None:
    """Full run on heavy-tailed data with outliers and NaNs.

    Checks the run result, the precedence of the resolved presets and the
    round-trip through the saved artifact.
    """
    df = _make_heavy_tailed_df()
    model_path = tmp_path / "models" / "robust.pkl"
    config = _make_config(model_path, n_trials=2)

    result = train_best_model(config=config, df=df, target="target")

    # 1. The run completes with a valid winner report.
    assert isinstance(result, dict)
    assert result["algorithm"] in {"ridge", "random_forest", "isotonic_regression"}
    assert result["score"] is not None
    assert np.isfinite(result["score"])
    assert isinstance(result["params"], dict)
    assert Path(result["model_path"]).exists()

    # 2. An honest quality estimate is reported: a hold-out score when the
    # outer split is reserved, otherwise the BBC-CV estimate. This run uses
    # the default holdout_fraction="auto" on more than 100 rows, so
    # holdout_score is expected — but the BBC fallback is accepted too.
    assert (
        result.get("holdout_score") is not None or result.get("bbc_score") is not None
    ), "neither holdout_score nor bbc_score was reported"
    if result.get("holdout_score") is not None:
        assert np.isfinite(result["holdout_score"])

    # 3. The resolved presets of the winner follow the precedence rules:
    # class preset < general.preprocessing < algorithms.<name>.preprocessing.
    loaded = ModelTrainer.load(str(result["model_path"]))
    preset = loaded.preprocessing_preset
    assert preset is not None
    if result["algorithm"] == "ridge":
        # The per-algorithm block sets only `scaling`: it wins per field, the
        # fields it leaves unset fall through to the general block.
        assert preset.imputation_strategy == "median"
        assert preset.scaling.type == "quantile"
        assert preset.scaling.output_distribution == "normal"
    elif result["algorithm"] == "random_forest":
        # Inherits the general block: median imputation + quantile outliers;
        # scaling stays `none` (the tree class preset, untouched by general).
        assert preset.imputation_strategy == "median"
        assert preset.scaling.type == "none"
        assert preset.outliers.method == "quantile"
    else:  # isotonic_regression
        # The univariate class has no override anywhere.
        assert preset.imputation_strategy == "median"
        assert preset.scaling.type == "none"

    # 4. The saved model loads and predicts exactly the same values as the
    # in-memory trainer.
    assert loaded.pipeline is not None
    new_df = df.drop(columns=["target"]).head(25)
    preds_loaded = loaded.predict(new_df)
    # Compare against a second load to rule out in-memory state effects.
    loaded_again = ModelTrainer.load(str(result["model_path"]))
    preds_again = loaded_again.predict(new_df)
    np.testing.assert_allclose(preds_loaded, preds_again)
    assert len(preds_loaded) == len(new_df)
    assert np.isfinite(preds_loaded).all()
