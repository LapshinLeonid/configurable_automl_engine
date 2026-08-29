"""E2E-тесты обучения моделей с ординальным (порядковым) кодированием категорий.

Проверяют, что движок корректно обучается на данных с категориальными признаками
при ``general.categorical_encoding='ordinal'`` через ``train_best_model`` и что
``tuner.optimize(..., encoding='ordinal')`` не даёт pruned-испытаний. Дополнительно
проверяется совместная работа ordinal-кодирования с SMOTE-оверсэмплингом и
сериализация ``ModelTrainer`` с ordinal-стратегией.
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
    """Синтетический датасет регрессии с категориальными признаками."""
    rng = np.random.default_rng(seed)
    color = rng.choice(["red", "green", "blue"], size=n)
    size = rng.choice(["S", "M", "L"], size=n)
    id_code = rng.choice(["10", "20", "30"], size=n)  # 'числовые строки'-категории
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
    encoding: str = "ordinal",
) -> dict[str, Any]:
    """Собрать конфигурацию для train_best_model с ordinal-кодированием."""
    return {
        "general": {
            "comparison_metric": "r2",
            "path_to_model": str(model_path),
            "serialization_format": "pickle",
            "validation_strategy": "train_test_split",
            "n_folds": 3,
            "categorical_encoding": encoding,
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


def test_train_best_model_ordinal_encoding(tmp_path: Path) -> None:
    """Полный цикл обучения с categorical_encoding='ordinal': модель сохраняется
    и предсказывает конечные значения на новых данных."""
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
    # Стратегия кодирования сохранилась в тренере
    assert loaded.encoding_strategy == "ordinal"

    new_df = df.head(10).drop(columns=["target"])
    preds = loaded.predict(new_df)
    assert len(preds) == len(new_df)
    assert np.isfinite(preds).all()


def test_tuner_optimize_ordinal_not_pruned() -> None:
    """tuner.optimize(..., encoding='ordinal') возвращает обученную модель
    и конечный score (нет pruned-испытаний)."""
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
        encoding="ordinal",
    )

    assert model is not None
    assert isinstance(params, dict) and params
    assert np.isfinite(score)


def test_train_best_model_ordinal_with_smote(tmp_path: Path) -> None:
    """Совместная работа ordinal-кодирования и SMOTE-оверсэмплинга
    (SMOTE получает уже закодированную числовую матрицу)."""
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
    """Невалидная encoding_strategy у ModelTrainer вызывает TrainingError."""
    with pytest.raises(Exception, match="Unknown encoding_strategy"):
        ModelTrainer(algorithm="elasticnet", encoding_strategy="target")


@pytest.mark.parametrize("fmt", [SerializationFormat.pickle, SerializationFormat.joblib])
def test_model_trainer_ordinal_serialization_roundtrip(
    tmp_path: Path, fmt: SerializationFormat
) -> None:
    """ModelTrainer с ordinal-стратегией корректно сохраняется/загружается."""
    df = _make_regression_df(n=60, seed=11)
    trainer = ModelTrainer(
        algorithm="elasticnet",
        hyperparams={"alpha": 0.01},
        metric="r2",
        serialization_format=fmt,
        encoding_strategy="ordinal",
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])

    path = tmp_path / f"ordinal.{'joblib' if fmt == SerializationFormat.joblib else 'pkl'}"
    trainer.save(path)

    loaded = ModelTrainer.load(path, fmt=fmt)
    assert loaded.encoding_strategy == "ordinal"

    preds = loaded.predict(df.drop(columns=["target"]).head(10))
    assert np.isfinite(preds).all()


def test_model_trainer_backward_compat_missing_encoding_strategy(
    tmp_path: Path,
) -> None:
    """Модель, сериализованная до появления атрибута encoding_strategy,
    загружается и продолжает работать (обратная совместимость)."""
    df = _make_regression_df(n=60, seed=12)
    trainer = ModelTrainer(
        algorithm="elasticnet",
        hyperparams={"alpha": 0.01},
        metric="r2",
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])

    # Имитируем модель, созданную до введения параметра encoding_strategy:
    # атрибут отсутствует у распикленного объекта.
    del trainer.encoding_strategy
    path = tmp_path / "legacy.pkl"
    trainer.save(path)

    loaded = ModelTrainer.load(path)
    assert not hasattr(loaded, "encoding_strategy")

    # predict работает (использует сохранённый pipeline)...
    preds = loaded.predict(df.drop(columns=["target"]).head(10))
    assert np.isfinite(preds).all()

    # ...и повторный fit не падает: _build_preprocessor() откатывается на one_hot.
    loaded.fit(df.drop(columns=["target"]), df["target"])
    assert loaded.pipeline is not None
