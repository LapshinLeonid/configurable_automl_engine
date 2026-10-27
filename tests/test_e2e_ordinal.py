"""E2E tests of training models with ordinal encoding of categories.

Verify that the engine trains correctly on data with categorical features
when ``general.categorical_encoding='ordinal'`` is used through
``train_best_model``, and that ``tuner.optimize(..., encoding='ordinal')``
produces no pruned trials. Additionally, the joint operation of ordinal
encoding with SMOTE oversampling and the serialization of ``ModelTrainer``
with the ordinal strategy are verified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.training_engine import train_best_model
from configurable_automl_engine.trainer import ModelTrainer


def _make_regression_df(n: int = 180, seed: int = 42) -> pd.DataFrame:
    """Synthetic regression dataset with categorical features."""
    rng = np.random.default_rng(seed)
    color = rng.choice(["red", "green", "blue"], size=n)
    size = rng.choice(["S", "M", "L"], size=n)
    id_code = rng.choice(["10", "20", "30"], size=n)  # numeric-string categories
    num = rng.normal(size=n)

    target = (
        2.0 * (color == "green")
        + 1.5 * (size == "L")
        + 0.8 * (id_code == "30")
        + num * 0.5
        + rng.normal(0, 0.05, size=n)
    )

    df = pd.DataFrame(
        {
            "color": color,
            "size": pd.Categorical(size),
            "id_code": id_code,
            "num": num,
        }
    )
    df["target"] = target
    return df


def _make_binary_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Synthetic dataset with a binary target (for SMOTE oversampling)."""
    rng = np.random.default_rng(seed)
    color = rng.choice(["red", "green", "blue"], size=n)
    size = rng.choice(["S", "M", "L"], size=n)
    num = rng.normal(size=n)

    logit = (
        2.0 * (color == "green")
        + 1.0 * (size == "L")
        + num * 1.5
        + rng.normal(0, 1.0, size=n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (prob > 0.5).astype(int)

    df = pd.DataFrame({"color": color, "size": size, "num": num})
    df["target"] = y
    return df


def _make_config(
    model_path: Path,
    *,
    oversampling: bool = False,
    os_algorithm: str = "random",
    n_trials: int = 2,
    encoding: str = "ordinal",
) -> dict[str, Any]:
    """Build a train_best_model configuration with ordinal encoding."""
    return {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "serialization_format": "pickle",
            "validation_strategy": "train_test_split",
            "n_folds": 3,
            "categorical_encoding": encoding,
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
            "elasticnet": {"enable": True},
            "random_forest": {"enable": True},
        },
        "oversampling": {
            "enable": oversampling,
            "multiplier": 1.5,
            "algorithm": os_algorithm,
        },
    }


def test_train_best_model_ordinal_encoding(tmp_path: Path) -> None:
    """Full training cycle with categorical_encoding='ordinal': the model is saved
    and predicts finite values on new data."""
    df = _make_regression_df()
    model_path = tmp_path / "models" / "ordinal.pkl"
    config = _make_config(model_path, n_trials=2)

    result = train_best_model(config=config, df=df, target="target")

    assert isinstance(result, dict)
    assert result["algorithm"] in {"elasticnet", "random_forest"}
    assert result["score"] is not None
    assert np.isfinite(result["score"])
    assert isinstance(result["params"], dict) and result["params"]
    assert Path(result["model_path"]).exists()

    loaded = ModelTrainer.load(str(result["model_path"]))
    assert loaded.pipeline is not None
    # The encoding strategy was preserved in the trainer
    assert loaded.encoding_strategy == "ordinal"

    new_df = df.head(10).drop(columns=["target"])
    preds = loaded.predict(new_df)
    assert len(preds) == len(new_df)
    assert np.isfinite(preds).all()


def test_tuner_optimize_ordinal_not_pruned() -> None:
    """tuner.optimize(..., encoding='ordinal') returns a trained model
    and a finite score (no pruned trials)."""
    from configurable_automl_engine.hpo.default_spaces import DEFAULT_SPACES
    from configurable_automl_engine.tuner import optimize

    df = _make_regression_df(n=120, seed=3)
    X = df.drop(columns=["target"])
    y = df["target"]

    opt_res = optimize(
        "elasticnet",
        X,
        y,
        n_trials=2,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides={"elasticnet": DEFAULT_SPACES["elasticnet"]},
        encoding="ordinal",
    )
    assert opt_res is not None
    model, params, score = opt_res.model, opt_res.params, opt_res.score


    assert model is not None
    assert isinstance(params, dict) and params
    assert np.isfinite(score)


def test_train_best_model_ordinal_with_smote(tmp_path: Path) -> None:
    """Ordinal encoding works together with SMOTE oversampling
    (SMOTE receives an already-encoded numeric matrix)."""
    df = _make_binary_df()
    model_path = tmp_path / "models" / "ordinal_smote.pkl"
    config = _make_config(
        model_path,
        oversampling=True,
        os_algorithm="smote",
        n_trials=2,
    )

    result = train_best_model(config=config, df=df, target="target")

    assert result["score"] is not None
    assert np.isfinite(result["score"])
    assert Path(result["model_path"]).exists()


def test_model_trainer_invalid_encoding_strategy() -> None:
    """An invalid ModelTrainer encoding_strategy raises TrainingError."""
    with pytest.raises(Exception, match="Unknown encoding_strategy"):
        ModelTrainer(algorithm="elasticnet", encoding_strategy="binary")


@pytest.mark.parametrize(
    "fmt", [SerializationFormat.pickle, SerializationFormat.joblib]
)
def test_model_trainer_ordinal_serialization_roundtrip(
    tmp_path: Path, fmt: SerializationFormat
) -> None:
    """ModelTrainer with the ordinal strategy saves/loads correctly."""
    df = _make_regression_df(n=60, seed=11)
    trainer = ModelTrainer(
        algorithm="elasticnet",
        hyperparams={"alpha": 0.01},
        metric="r2",
        serialization_format=fmt,
        encoding_strategy="ordinal",
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])

    path = (
        tmp_path / f"ordinal.{'joblib' if fmt == SerializationFormat.joblib else 'pkl'}"
    )
    trainer.save(path)

    loaded = ModelTrainer.load(path, fmt=fmt)
    assert loaded.encoding_strategy == "ordinal"

    preds = loaded.predict(df.drop(columns=["target"]).head(10))
    assert np.isfinite(preds).all()
