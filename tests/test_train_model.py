# research/tests/test_train_model.py
import os
import pytest
import numpy as np
import pandas as pd

from configurable_automl_engine.trainer import train_model, TrainingError

# Синтетические данные для тестирования
X = pd.DataFrame({
    "a": np.random.rand(50),
    "b": np.random.rand(50)
})
y = X["a"] * 2 + X["b"] * -3 + np.random.randn(50) * 0.1

@pytest.fixture
def base_params():
    return {"alpha": 0.1, "l1_ratio": 0.5}


def test_successful_training(tmp_path, monkeypatch, base_params):
    # Логи появляются, когда enable_logging=True
    monkeypatch.chdir(tmp_path)
    score = train_model(
        "ElasticNet",
        "r2",
        base_params,
        X, y,
        enable_logging=True
    )
    assert isinstance(score, float)
    assert 0.3 < score <= 1.0
    assert (tmp_path / "training.log").exists()


def test_no_logging(tmp_path, monkeypatch, base_params):
    # Без логирования файл не создаётся
    monkeypatch.chdir(tmp_path)
    score = train_model(
        "ElasticNet",
        "r2",
        base_params,
        X, y,
        enable_logging=False
    )
    assert isinstance(score, float)
    assert not (tmp_path / "training.log").exists()


def test_invalid_algorithm(base_params):
    with pytest.raises(TrainingError):
        train_model(None, "r2", base_params, X, y)


def test_invalid_metric(base_params):
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "mse", base_params, X, y)


def test_empty_params(base_params):
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", {}, X, y)


def test_empty_data(base_params=None):
    # Оба DataFrame пустые
    empty_df = pd.DataFrame([])
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", base_params or {}, empty_df, empty_df)


def test_mismatch_dimensions(base_params):
    # X и y разной длины
    X2 = X.iloc[:10]
    y2 = y.iloc[:9]
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", base_params, X2, y2)


def test_too_few_records(base_params):
    # Меньше двух записей
    X_small = X.iloc[:1]
    y_small = y.iloc[:1]
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", base_params, X_small, y_small)


def test_invalid_param_key(base_params):
    # Параметр неизвестный для ElasticNet
    bad_params = {"foobar": 1}
    with pytest.raises(TypeError):
        train_model("ElasticNet", "r2", bad_params, X, y)


def test_negative_alpha(base_params):
    # Негативный alpha приводит к ValueError из sklearn
    neg_params = {"alpha": -1.0, "l1_ratio": 0.5}
    with pytest.raises(ValueError):
        train_model("ElasticNet", "r2", neg_params, X, y)


def test_invalid_data_type(base_params):
    # Неподдерживаемый тип данных (list)
    with pytest.raises(Exception):
        train_model("ElasticNet", "r2", base_params, [1, 2, 3], [1, 2, 3])


