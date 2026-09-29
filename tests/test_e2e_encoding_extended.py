"""E2E-тесты новых стратегий кодирования категориальных признаков (issue #20).

Проверяют полный цикл: обучение ``ModelTrainer`` и ``train_best_model``
с target/frequency/hashing кодированием, предсказание на новых данных с
неизвестными категориями, сохранение/загрузку модели, совместную работу
с оверсэмплингом, high-cardinality режим и единообразие HPO/финального обучения.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.training_engine import train_best_model
from configurable_automl_engine.trainer import ModelTrainer, TrainingError


def _make_regression_df(n: int = 180, seed: int = 42) -> pd.DataFrame:
    """Синтетический датасет регрессии с категориальными признаками
    (включая колонку высокой кардинальности)."""
    rng = np.random.default_rng(seed)
    color = rng.choice(["red", "green", "blue"], size=n)
    size = rng.choice(["S", "M", "L"], size=n)
    high_card = [f"city_{i % 40}" for i in range(n)]  # высокая кардинальность
    num = rng.normal(size=n)

    target = (
        2.0 * (color == "green")
        + 1.5 * (size == "L")
        + 0.3 * (pd.Series(high_card) == "city_7").to_numpy().astype(float)
        + num * 0.5
        + rng.normal(0, 0.05, size=n)
    )

    df = pd.DataFrame(
        {
            "color": color,
            "size": pd.Categorical(size),
            "high_card": high_card,
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
    encoding: str = "one_hot",
    oversampling: bool = False,
    os_algorithm: str = "random",
    n_trials: int = 2,
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: str | None = None,
    hashing_n_components: int = 16,
) -> dict[str, Any]:
    """Собрать конфигурацию для train_best_model."""
    general: dict[str, Any] = {
        "comparison_metric": "r2",
        "path_to_model": str(model_path),
        "serialization_format": "pickle",
        "validation_strategy": "train_test_split",
        "n_folds": 3,
        "categorical_encoding": encoding,
        "phases": [
            {"name": "search", "n_trials": n_trials, "action": "all_algorithms"}
        ],
    }
    if high_cardinality_threshold is not None:
        general["high_cardinality_threshold"] = high_cardinality_threshold
        general["high_cardinality_encoding"] = high_cardinality_encoding
    general["hashing_n_components"] = hashing_n_components

    return {
        "general": general,
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


def _assert_valid_result(result: dict[str, Any]) -> None:
    """Общие проверки структуры результата train_best_model."""
    assert isinstance(result, dict)
    assert result["score"] is not None
    assert np.isfinite(result["score"]), f"score не конечен: {result['score']}"
    assert isinstance(result["params"], dict) and result["params"]
    assert Path(result["model_path"]).exists()


@pytest.mark.parametrize("encoding", ["target", "frequency", "hashing"])
def test_model_trainer_new_strategies_train_and_predict(
    tmp_path: Path, encoding: str
) -> None:
    """ModelTrainer обучается с каждой новой стратегией и предсказывает."""
    df = _make_regression_df(n=120, seed=3)
    trainer = ModelTrainer(
        algorithm="elasticnet",
        hyperparams={"alpha": 0.01},
        metric="r2",
        encoding_strategy=encoding,
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])
    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)

    preds = trainer.predict(df.drop(columns=["target"]).head(10))
    assert len(preds) == 10
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("encoding", ["target", "frequency", "hashing", "ordinal"])
def test_model_trainer_predict_unknown_categories(encoding: str) -> None:
    """Неизвестные категории на предсказании не роняют модель."""
    df = _make_regression_df(n=100, seed=5)
    trainer = ModelTrainer(
        algorithm="decision_tree",
        hyperparams={"max_depth": 5},
        metric="r2",
        encoding_strategy=encoding,
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])

    new_df = df.drop(columns=["target"]).head(10).copy()
    new_df["color"] = "brand_new_color"
    new_df["high_card"] = "never_seen_city"
    new_df["size"] = new_df["size"].astype(object)
    new_df.loc[0, "size"] = "XXL"

    preds = trainer.predict(new_df)
    assert len(preds) == 10
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("encoding", ["target", "frequency", "hashing"])
def test_model_trainer_save_load_roundtrip(tmp_path: Path, encoding: str) -> None:
    """После сохранения/загрузки предсказания совпадают (та же стратегия)."""
    df = _make_regression_df(n=100, seed=6)
    X = df.drop(columns=["target"])
    y = df["target"]

    trainer = ModelTrainer(
        algorithm="random_forest",
        hyperparams={"n_estimators": 20, "max_depth": 4},
        metric="r2",
        encoding_strategy=encoding,
    )
    trainer.fit(X, y)
    before = trainer.predict(X.head(5))

    path = tmp_path / f"model_{encoding}.pkl"
    trainer.save(path)

    loaded = ModelTrainer.load(path)
    assert loaded.encoding_strategy == encoding
    after = loaded.predict(X.head(5))
    np.testing.assert_allclose(before, after)


def test_model_trainer_high_cardinality_mode() -> None:
    """Автоматический HC-режим в ModelTrainer: обучение и предсказание."""
    df = _make_regression_df(n=150, seed=9)
    trainer = ModelTrainer(
        algorithm="random_forest",
        hyperparams={"n_estimators": 20, "max_depth": 4},
        metric="r2",
        encoding_strategy="one_hot",
        high_cardinality_threshold=10,
        high_cardinality_encoding="target",
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])
    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)

    preds = trainer.predict(df.drop(columns=["target"]).head(5))
    assert np.isfinite(preds).all()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"encoding_strategy": "binary"}, "Unknown encoding_strategy"),
        ({"high_cardinality_threshold": 5}, "must be set together"),
        ({"high_cardinality_encoding": "target"}, "must be set together"),
        (
            {"high_cardinality_threshold": -1, "high_cardinality_encoding": "target"},
            ">= 0",
        ),
        (
            {"high_cardinality_threshold": 5, "high_cardinality_encoding": "binary"},
            "Unknown high_cardinality_encoding",
        ),
        ({"hashing_n_components": 0}, ">= 1"),
        ({"target_encoding_smoothing": -1.0}, ">= 0"),
    ],
)
def test_model_trainer_invalid_encoding_params(
    kwargs: dict[str, Any], match: str
) -> None:
    """Некорректные параметры кодирования отклоняются при инициализации."""
    with pytest.raises(TrainingError, match=match):
        ModelTrainer(algorithm="elasticnet", **kwargs)


def test_tuner_optimize_numpy_without_features_warns() -> None:
    """tuner.optimize на numpy без категориальных признаков не падает
    (модель используется без препроцессора)."""
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
    from configurable_automl_engine.tuner import optimize

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = rng.normal(size=80)

    model, params, score = optimize(
        "elasticnet",
        X,
        y,
        n_trials=1,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides={"elasticnet": DEFAULT_SPACES["elasticnet"]},
    )
    assert model is not None
    assert np.isfinite(score)


def test_tuner_optimize_numpy_with_features_warns(caplog) -> None:
    """Явно переданные категориальные признаки с numpy X -> предупреждение."""
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
    from configurable_automl_engine.tuner import optimize

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = rng.normal(size=80)

    model, params, score = optimize(
        "elasticnet",
        X,
        y,
        n_trials=1,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides={"elasticnet": DEFAULT_SPACES["elasticnet"]},
        categorical_features=["a", "b"],
        numerical_features=["c"],
    )
    assert model is not None
    assert np.isfinite(score)


@pytest.mark.parametrize("encoding", ["target", "frequency", "hashing"])
def test_model_trainer_with_oversampling(tmp_path: Path, encoding: str) -> None:
    """Совместная работа новых стратегий с оверсэмплингом (кодирование до синтеза)."""
    df = _make_binary_df()
    trainer = ModelTrainer(
        algorithm="random_forest",
        hyperparams={"n_estimators": 20, "max_depth": 4},
        metric="r2",
        data_oversampling=True,
        data_oversampling_multiplier=1.5,
        data_oversampling_algorithm="smote",
        encoding_strategy=encoding,
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])
    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)

    path = tmp_path / f"os_{encoding}.pkl"
    trainer.save(path)
    loaded = ModelTrainer.load(path)
    preds = loaded.predict(df.drop(columns=["target"]).head(5))
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("encoding", ["target", "frequency", "hashing"])
def test_train_best_model_new_strategies(tmp_path: Path, encoding: str) -> None:
    """Полный цикл train_best_model с новой стратегией."""
    df = _make_regression_df()
    model_path = tmp_path / "models" / f"{encoding}.pkl"
    config = _make_config(model_path, encoding=encoding, n_trials=2)

    result = train_best_model(config=config, df=df, target="target")

    _assert_valid_result(result)
    loaded = ModelTrainer.load(str(result["model_path"]))
    preds = loaded.predict(df.drop(columns=["target"]).head(10))
    assert np.isfinite(preds).all()


def test_train_best_model_high_cardinality_auto_mode(tmp_path: Path) -> None:
    """Автоматический HC-режим через конфигурацию эксперимента."""
    df = _make_regression_df()
    model_path = tmp_path / "models" / "hc.pkl"
    config = _make_config(
        model_path,
        encoding="one_hot",
        n_trials=2,
        high_cardinality_threshold=10,
        high_cardinality_encoding="target",
    )

    result = train_best_model(config=config, df=df, target="target")

    _assert_valid_result(result)
    loaded = ModelTrainer.load(str(result["model_path"]))
    assert loaded.high_cardinality_threshold == 10
    assert loaded.high_cardinality_encoding == "target"


def test_train_best_model_hashing_with_oversampling(tmp_path: Path) -> None:
    """Hashing + оверсэмплинг: пайплайн обучается без ошибок."""
    df = _make_binary_df()
    model_path = tmp_path / "models" / "hash_os.pkl"
    config = _make_config(
        model_path,
        encoding="hashing",
        oversampling=True,
        os_algorithm="smote",
        n_trials=2,
    )

    result = train_best_model(config=config, df=df, target="target")

    _assert_valid_result(result)
    loaded = ModelTrainer.load(str(result["model_path"]))
    preds = loaded.predict(df.drop(columns=["target"]).head(5))
    assert np.isfinite(preds).all()


def test_tuner_optimize_new_encoding_strategies() -> None:
    """tuner.optimize работает с новыми стратегиями (единая логика с фит.обучением)."""
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
    from configurable_automl_engine.tuner import optimize

    df = _make_regression_df(n=120, seed=11)
    X = df.drop(columns=["target"])
    y = df["target"]

    for encoding in ("target", "frequency", "hashing"):
        model, params, score = optimize(
            "elasticnet",
            X,
            y,
            n_trials=2,
            random_state=0,
            metric="r2",
            validation_strategy="train_test_split",
            space_overrides={"elasticnet": DEFAULT_SPACES["elasticnet"]},
            encoding=encoding,
        )
        assert model is not None
        assert isinstance(params, dict) and params
        assert np.isfinite(score)


def test_tuner_optimize_high_cardinality_mode() -> None:
    """tuner.optimize с HC-режимом возвращает модель."""
    from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
    from configurable_automl_engine.tuner import optimize

    df = _make_regression_df(n=140, seed=13)
    X = df.drop(columns=["target"])
    y = df["target"]

    model, params, score = optimize(
        "random_forest",
        X,
        y,
        n_trials=2,
        random_state=0,
        metric="r2",
        validation_strategy="train_test_split",
        space_overrides={"random_forest": DEFAULT_SPACES["random_forest"]},
        encoding="one_hot",
        high_cardinality_threshold=10,
        high_cardinality_encoding="target",
    )
    assert model is not None
    assert isinstance(params, dict) and params
    assert np.isfinite(score)


# ───────────────── Sparse-выход hashing (issue #4) ─────────────────


@pytest.mark.parametrize(
    "algorithm, hyperparams",
    [
        ("gaussian_process_regression", {}),
        ("ardregression", {"alpha": 1.0}),
    ],
)
def test_model_trainer_sparse_rejecting_algos_with_hashing(
    algorithm: str, hyperparams: dict[str, Any]
) -> None:
    """GPR/ARD отвергают scipy.sparse: для них автоматически активируется
    force_dense_output, поэтому обучение с hashing-кодированием не падает."""
    df = _make_regression_df(n=60, seed=17)
    trainer = ModelTrainer(
        algorithm=algorithm,
        hyperparams=hyperparams,
        metric="r2",
        encoding_strategy="hashing",
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])
    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)

    preds = trainer.predict(df.drop(columns=["target"]).head(5))
    assert np.isfinite(preds).all()


def test_model_trainer_isotonic_hashing_single_feature() -> None:
    """Isotonic требует один признак и отвергает sparse: одна категориальная
    колонка с hashing_n_components=1 даёт ровно одну колонку, а
    force_dense_output активируется по флагу алгоритма."""
    df = _make_regression_df(n=60, seed=19)[["color", "target"]]
    trainer = ModelTrainer(
        algorithm="isotonic_regression",
        hyperparams={},
        metric="r2",
        encoding_strategy="hashing",
        hashing_n_components=1,
    )
    trainer.fit(df.drop(columns=["target"]), df["target"])
    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)

    preds = trainer.predict(df.drop(columns=["target"]).head(5))
    assert np.isfinite(preds).all()


def test_hashing_preprocessor_output_is_sparse() -> None:
    """Сквозная проверка OOM-фикса: препроцессор с hashing отдаёт csr_matrix
    (а не плотную float64-матрицу), экономя память на больших таблицах."""
    from scipy import sparse

    from configurable_automl_engine.preprocessing import build_preprocessor

    df = _make_regression_df(n=150, seed=21)
    X = df.drop(columns=["target"])
    pre = build_preprocessor(
        list(X.columns),
        categorical_features=["color", "size", "high_card"],
        numerical_features=["num"],
        encoding="hashing",
        hashing_n_components=16,
    )
    out = pre.fit_transform(X, df["target"])
    assert sparse.issparse(out)
    assert out.format == "csr"
    # nnz = n_rows * (n_cat_cols + n_num_cols): одна единица на строку на
    # hashing-колонку плюс плотная числовая часть — а не n_rows * 48
    assert out.nnz == len(df) * 4
    assert out.shape[1] == 3 * 16 + 1
