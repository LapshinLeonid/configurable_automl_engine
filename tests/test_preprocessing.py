"""Unit tests for the shared preprocessing module (categorical feature handling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from configurable_automl_engine.preprocessing import (
    build_preprocessor,
    detect_feature_types,
)


def test_detect_feature_types_mixed_dataframe():
    """detect_feature_types различает категориальные и числовые колонки."""
    df = pd.DataFrame(
        {
            "cat_color": ["red", "green", "blue"],
            "cat_size": pd.Categorical(["S", "M", "L"]),
            "flag": [True, False, True],
            "num_a": [1.0, 2.0, 3.0],
            "num_b": [10, 20, 30],
        }
    )
    cat, num = detect_feature_types(df)

    assert sorted(cat) == ["cat_color", "cat_size", "flag"]
    assert sorted(num) == ["num_a", "num_b"]


def test_detect_feature_types_all_numeric():
    """На чисто числовом DataFrame категориальных колонок нет."""
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.7]})
    cat, num = detect_feature_types(df)

    assert cat == []
    assert sorted(num) == ["a", "b"]


def test_build_preprocessor_onehot_for_categorical():
    """build_preprocessor возвращает ColumnTransformer с OneHotEncoder для категорий."""
    feature_names = ["cat", "num"]
    preprocessor = build_preprocessor(
        feature_names,
        categorical_features=["cat"],
        numerical_features=["num"],
    )

    assert isinstance(preprocessor, ColumnTransformer)

    names = [name for name, _, _ in preprocessor.transformers]
    assert "cat" in names
    assert "num" in names

    # Категориальный трансформер использует SplitCategoricalEncoder с
    # default-энкодером OneHotEncoder (high-cardinality режим отключён).
    # Внутренние энкодеры создаются в fit() (sklearn-конвенция: атрибуты
    # с завершающим подчёркиванием появляются только после обучения),
    # поэтому сначала обучаем препроцессор на данных.
    df = pd.DataFrame(
        {"cat": ["red", "green", "red", "blue"], "num": [1.0, 2.0, 3.0, 4.0]}
    )
    preprocessor.fit(df)

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    assert isinstance(encoder.default_encoder_, OneHotEncoder)
    assert encoder.hc_encoder_ is None


def test_build_preprocessor_end_to_end_encoding():
    """Проверка сквозного кодирования категорий через препроцессор."""
    df = pd.DataFrame(
        {
            "cat": ["red", "green", "red", "blue"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
    )
    out = preprocessor.fit_transform(df)

    assert out.shape == (4, 3 + 1)  # 3 one-hot колонки + 1 числовая


def test_build_preprocessor_no_features_passthrough():
    """При отсутствии совпавших колонок возвращается passthrough-трансформер."""
    preprocessor = build_preprocessor(
        ["some_random_column"],
        categorical_features=[],
        numerical_features=[],
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.transformers[0][0] == "bypass"
    assert preprocessor.transformers[0][1] == "passthrough"


def test_build_preprocessor_numeric_strings_treated_as_categorical():
    """Колонки 'числовых строк' (ID '10','20') детектируются как категории по dtype."""
    df = pd.DataFrame({"id_code": ["10", "20", "30", "20"]})
    cat, _ = detect_feature_types(df)

    assert cat == ["id_code"]

    preprocessor = build_preprocessor(
        list(df.columns), categorical_features=cat, numerical_features=[]
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (4, 3)

    # Все значения в матрице являются числами (one-hot)
    assert np.isfinite(out).all()


def test_build_preprocessor_bool_column_not_crash():
    """Чисто bool-колонка не роняет препроцессор при fit_transform.

    Регрессия на критический баг: SimpleImputer(strategy="most_frequent")
    падал на numpy-bool ("SimpleImputer does not support data with dtype bool").
    """
    df = pd.DataFrame(
        {
            "flag": [True, False, True, False],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cat, num = detect_feature_types(df)

    assert cat == ["flag"]
    assert num == ["num"]

    preprocessor = build_preprocessor(list(df.columns), cat, num)
    out = preprocessor.fit_transform(df)

    # bool -> one-hot (2 категории) + 1 числовая
    assert out.shape == (4, 3)
    assert np.isfinite(out).all()


def test_build_preprocessor_mixed_object_and_bool():
    """Совместная обработка object- и bool-колонок через один категориальный
    пайплайн без падения на dtype bool."""
    df = pd.DataFrame(
        {
            "color": ["red", "green", "red", "blue"],
            "flag": [True, False, True, False],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cat, num = detect_feature_types(df)

    assert sorted(cat) == ["color", "flag"]
    assert num == ["num"]

    preprocessor = build_preprocessor(list(df.columns), cat, num)
    out = preprocessor.fit_transform(df)

    # color (3 категории) + flag (2 категории) + 1 числовая
    assert out.shape == (4, 6)
    assert np.isfinite(out).all()


# ───────────────────────── Ordinal encoding ─────────────────────────


def test_build_preprocessor_ordinal_uses_ordinal_encoder():
    """build_preprocessor(encoding='ordinal') использует OrdinalEncoder (не OneHotEncoder)."""
    preprocessor = build_preprocessor(
        ["cat", "num"],
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="ordinal",
    )

    assert isinstance(preprocessor, ColumnTransformer)
    names = [name for name, _, _ in preprocessor.transformers]
    assert "cat" in names
    assert "num" in names

    # Внутренние энкодеры создаются в fit(), поэтому сначала обучаем.
    df = pd.DataFrame({"cat": ["red", "green", "red"], "num": [1.0, 2.0, 3.0]})
    preprocessor.fit(df)

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    assert isinstance(encoder.default_encoder_, OrdinalEncoder)
    assert not isinstance(encoder.default_encoder_, OneHotEncoder)
    assert encoder.hc_encoder_ is None


def test_build_preprocessor_default_is_onehot():
    """По умолчанию (encoding не задан / 'one_hot') сохраняется OneHotEncoder."""
    df = pd.DataFrame({"cat": ["red", "green", "red"], "num": [1.0, 2.0, 3.0]})
    for kwargs in ({}, {"encoding": "one_hot"}):
        preprocessor = build_preprocessor(
            ["cat", "num"],
            categorical_features=["cat"],
            numerical_features=["num"],
            **kwargs,
        )
        # Внутренние энкодеры создаются в fit() (sklearn-конвенция).
        preprocessor.fit(df)
        assert isinstance(
            preprocessor.named_transformers_["cat"].named_steps[
                "encoder"
            ].default_encoder_,
            OneHotEncoder,
        )


def test_build_preprocessor_ordinal_end_to_end_shape():
    """Сквозное ordinal-кодирование: категории НЕ расширяются (1 столбец на колонку)."""
    df = pd.DataFrame(
        {
            "cat": ["red", "green", "red", "blue"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="ordinal",
    )
    out = preprocessor.fit_transform(df)

    # 1 ordinal колонка (для 'cat') + 1 числовая = 2
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_build_preprocessor_ordinal_bool_column():
    """bool-колонка с ordinal не падает и даёт конечную числовую матрицу."""
    df = pd.DataFrame(
        {
            "flag": [True, False, True, False],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cat, num = detect_feature_types(df)
    assert cat == ["flag"]

    preprocessor = build_preprocessor(list(df.columns), cat, num, encoding="ordinal")
    out = preprocessor.fit_transform(df)

    # 1 ordinal (flag) + 1 числовая
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_build_preprocessor_ordinal_unknown_category_no_raise():
    """Неизвестная категория на transform в ordinal-режиме не бросает исключение
    (handle_unknown='use_encoded_value', unknown_value=-1)."""
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=[],
        encoding="ordinal",
    )
    preprocessor.fit(df)

    new_df = pd.DataFrame({"cat": ["unknown_category"]})
    out = preprocessor.transform(new_df)
    assert out.shape == (1, 1)
    assert np.isfinite(out).all()
    # неизвестная категория -> -1
    assert out[0, 0] == -1.0


def test_build_preprocessor_invalid_encoding_raises():
    """Невалидное значение encoding -> ValueError."""
    with pytest.raises(ValueError, match="Unknown encoding strategy"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            encoding="binary",
        )


def test_build_preprocessor_ordinal_single_unique_category():
    """Категориальная колонка с единственной уникальной категорией даёт 1 столбец."""
    df = pd.DataFrame({"cat": ["only", "only", "only"]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=[],
        encoding="ordinal",
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (3, 1)
    assert np.isfinite(out).all()


def test_build_preprocessor_ordinal_all_missing_after_impute():
    """Полностью пропущенная категориальная колонка после impute most_frequent
    корректно кодируется ordinal."""
    df = pd.DataFrame({"cat": [None, None, None], "num": [1.0, 2.0, 3.0]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="ordinal",
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (3, 2)
    assert np.isfinite(out).all()


def test_build_preprocessor_ordinal_empty_features_passthrough():
    """Пустые categorical/numerical -> passthrough ('bypass') независимо от encoding."""
    for enc in ("one_hot", "ordinal"):
        preprocessor = build_preprocessor(
            ["some_random_column"],
            categorical_features=[],
            numerical_features=[],
            encoding=enc,
        )
        assert isinstance(preprocessor, ColumnTransformer)
        assert preprocessor.transformers[0][0] == "bypass"
        assert preprocessor.transformers[0][1] == "passthrough"


# ───────────────────────── Adaptive presets (issue #18) ─────────────────────────


def _num_pipeline(preprocessor: ColumnTransformer):
    """Извлечь числовой пайплайн из ColumnTransformer."""
    transformers = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )
    assert "num" in transformers, "числовой трансформер отсутствует"
    return transformers["num"]


def test_build_preprocessor_default_imputation_is_mean():
    """По умолчанию (пресет scale_sensitive) — импутация mean."""
    preprocessor = build_preprocessor(
        ["a", "b"], categorical_features=[], numerical_features=["a", "b"]
    )
    num = _num_pipeline(preprocessor)
    assert isinstance(num.named_steps["imputer"], SimpleImputer)
    assert num.named_steps["imputer"].strategy == "mean"


def test_build_preprocessor_median_imputation():
    """imputation_strategy='median' → SimpleImputer(strategy='median')."""
    preprocessor = build_preprocessor(
        ["a", "b"],
        categorical_features=[],
        numerical_features=["a", "b"],
        imputation_strategy="median",
    )
    num = _num_pipeline(preprocessor)
    assert num.named_steps["imputer"].strategy == "median"


def test_build_preprocessor_scaling_none_omits_scaler():
    """scaling='none' → шаг масштабирования отсутствует (AC-3 для деревьев)."""
    preprocessor = build_preprocessor(
        ["a", "b"],
        categorical_features=[],
        numerical_features=["a", "b"],
        scaling="none",
    )
    num = _num_pipeline(preprocessor)
    step_names = [s[0] for s in num.steps]
    assert step_names == ["imputer"]
    assert "scaler" not in step_names


def test_build_preprocessor_scaling_standard_uses_standard_scaler():
    """scaling='standard' → StandardScaler (AC-4 для масштабо-чувствительных)."""
    preprocessor = build_preprocessor(
        ["a", "b"],
        categorical_features=[],
        numerical_features=["a", "b"],
        scaling="standard",
    )
    num = _num_pipeline(preprocessor)
    assert isinstance(num.named_steps["scaler"], StandardScaler)


def test_build_preprocessor_scaling_robust_uses_robust_scaler():
    """scaling='robust' → RobustScaler (AC-5 для GLM со скошенными распределениями)."""
    preprocessor = build_preprocessor(
        ["a", "b"],
        categorical_features=[],
        numerical_features=["a", "b"],
        scaling="robust",
    )
    num = _num_pipeline(preprocessor)
    assert isinstance(num.named_steps["scaler"], RobustScaler)


def test_build_preprocessor_invalid_imputation_strategy_raises():
    with pytest.raises(ValueError, match="Unknown imputation strategy"):
        build_preprocessor(
            ["a"],
            categorical_features=[],
            numerical_features=["a"],
            imputation_strategy="mode",
        )


def test_build_preprocessor_invalid_scaling_raises():
    with pytest.raises(ValueError, match="Unknown scaling type"):
        build_preprocessor(
            ["a"],
            categorical_features=[],
            numerical_features=["a"],
            scaling="quantile",
        )


def test_tree_preset_end_to_end_no_scaling():
    """Пресет деревьев (median + none): пропуски заполняются медианой, данные
    передаются в модель без масштабирования."""
    df = pd.DataFrame(
        {"num_a": [1.0, 2.0, np.nan, 4.0], "num_b": [10.0, 20.0, 30.0, np.nan]}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        imputation_strategy="median",
        scaling="none",
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()
    # Без масштабирования значения остаются в исходном диапазоне
    assert set(np.round(out[:, 0]).astype(int)) <= {1, 2, 4}
    assert set(np.round(out[:, 1]).astype(int)) <= {10, 20, 30}


def test_glm_preset_end_to_end_robust():
    """Пресет GLM (median + robust): корректная обработка выбросов и пропусков."""
    df = pd.DataFrame(
        {
            "num_a": [1.0, 2.0, np.nan, 4.0, 1000.0],
            "num_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        imputation_strategy="median",
        scaling="robust",
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (5, 2)
    assert np.isfinite(out).all()


# ────────────────── Sparse-выход hashing (issue #4) ──────────────────


def test_build_preprocessor_hashing_sparse_by_default():
    """hashing-кодирование возвращает csr_matrix через ColumnTransformer
    (стандартный sparse_threshold=0.3 отдаёт csr при любой sparse-части)."""
    df = pd.DataFrame(
        {"cat": ["a", "b", "c"] * 4, "num": np.arange(12.0)}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="hashing",
        hashing_n_components=8,
    )
    out = preprocessor.fit_transform(df)
    assert sparse.issparse(out)
    assert out.format == "csr"
    assert out.shape == (12, 9)
    assert out.nnz <= 12 * 9


def test_build_preprocessor_force_dense_output():
    """force_dense_output=True конвертирует sparse-части (hashing) в dense:
    требуется для GPR/Isotonic/ARD, отвергающих scipy.sparse."""
    df = pd.DataFrame(
        {"cat": ["a", "b", "c"] * 4, "num": np.arange(12.0)}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="hashing",
        hashing_n_components=8,
        force_dense_output=True,
    )
    out = preprocessor.fit_transform(df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (12, 9)
    assert np.isfinite(out).all()


def test_build_preprocessor_force_dense_passthrough():
    """force_dense_output не ломает passthrough-ветку (нет совпавших колонок)."""
    preprocessor = build_preprocessor(
        ["some_random_column"],
        categorical_features=[],
        numerical_features=[],
        encoding="hashing",
        force_dense_output=True,
    )
    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.transformers[0][0] == "bypass"
    out = preprocessor.fit_transform(pd.DataFrame({"some_random_column": [1, 2, 3]}))
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 1)
