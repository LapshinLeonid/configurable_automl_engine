"""E2E-тесты обучения моделей на данных с категориальными признаками.

Проверяют, что движок корректно обучается на наборах данных с категориальными
признаками (one-hot encoding по умолчанию) через внешний метод
``train_best_model``: HPO не даёт pruned-испытаний, финальная модель
сохраняется и способна предсказывать на новых данных.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.training_engine import train_best_model
from configurable_automl_engine.trainer import ModelTrainer


def _make_regression_df(n: int = 180, seed: int = 42) -> pd.DataFrame:
    """Синтетический датасет регрессии с категориальными признаками.

    Включает object/category колонки (кардинальность 2-5), в том числе колонку
    'числовых строк' (``id_code``), и одну числовую колонку.
    """
    rng = np.random.default_rng(seed)
    color = rng.choice(["red", "green", "blue"], size=n)
    size = rng.choice(["S", "M", "L"], size=n)
    id_code = rng.choice(["10", "20", "30"], size=n)  # 'числовые строки'-категории
    city = rng.choice(["NY", "LA", "SF", "CH"], size=n)
    num = rng.normal(size=n)

    target = (
        2.0 * (color == "green")
        + 1.5 * (size == "L")
        + 0.8 * (id_code == "30")
        + 0.5 * (city == "SF")
        + num * 0.5
        + rng.normal(0, 0.05, size=n)
    )

    df = pd.DataFrame(
        {
            "color": color,
            "size": pd.Categorical(size),
            "id_code": id_code,
            "city": city,
            "num": num,
        }
    )
    df["target"] = target
    return df


def _make_binary_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Синтетический датасет с бинарным таргетом (для оверсэмплинга SMOTE)."""
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
) -> dict[str, Any]:
    """Собрать конфигурацию для train_best_model."""
    return {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "serialization_format": "pickle",
            "validation_strategy": "train_test_split",
            "n_folds": 3,
            "phases": [
                {"name": "search", "n_trials": n_trials, "action": "all_algorithms"}
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


def _assert_valid_result(result: dict[str, Any], expected_algos: set[str]) -> None:
    """Общие проверки структуры результата train_best_model."""
    assert isinstance(result, dict)
    assert result["algorithm"] in expected_algos
    assert result["score"] is not None
    assert np.isfinite(result["score"]), f"score не конечен: {result['score']}"
    assert result["score"] != float("-inf")
    assert isinstance(result["params"], dict) and result["params"]
    assert Path(result["model_path"]).exists()


def test_train_best_model_on_categorical_data(tmp_path: Path) -> None:
    """Полный цикл обучения на данных с категориальными признаками."""
    df = _make_regression_df()
    model_path = tmp_path / "models" / "best.pkl"
    config = _make_config(model_path, n_trials=2)

    result = train_best_model(config=config, df=df, target="target")

    _assert_valid_result(result, {"elasticnet", "random_forest"})

    # Загружаем модель и проверяем предсказание на новых данных
    loaded = ModelTrainer.load(str(result["model_path"]))
    assert loaded.pipeline is not None

    new_df = df.head(10).drop(columns=["target"])
    preds = loaded.predict(new_df)
    assert len(preds) == len(new_df)
    assert np.isfinite(preds).all()


def test_train_best_model_categorical_with_smote_oversampling(tmp_path: Path) -> None:
    """Совместная работа one-hot кодирования и SMOTE-оверсэмплинга."""
    df = _make_binary_df()
    model_path = tmp_path / "models" / "smote.pkl"
    config = _make_config(
        model_path,
        oversampling=True,
        os_algorithm="smote",
        n_trials=2,
    )

    result = train_best_model(config=config, df=df, target="target")

    _assert_valid_result(result, {"elasticnet", "random_forest"})
    assert Path(result["model_path"]).exists()


def test_tuner_optimize_categorical_not_pruned() -> None:
    """tuner.optimize на категориальном DataFrame не даёт pruned-испытаний
    и возвращает обученную модель (не None)."""
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
    from configurable_automl_engine.tuner import optimize

    df = _make_regression_df(n=120, seed=3)
    X = df.drop(columns=["target"])
    y = df["target"]

    model, params, score = optimize(
        "elasticnet",
        X,
        y,
        n_trials=2,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides={"elasticnet": DEFAULT_SPACES["elasticnet"]},
    )

    assert model is not None
    assert isinstance(params, dict) and params
    assert np.isfinite(score)