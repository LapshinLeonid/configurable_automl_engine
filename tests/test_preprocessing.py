"""Unit tests for the shared preprocessing module (categorical feature handling)."""

from __future__ import annotations

import pickle

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
    ColumnNameSelector,
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
    """По умолчанию (пресет scale_sensitive) — импутация mean с add_indicator."""
    preprocessor = build_preprocessor(
        ["a", "b"], categorical_features=[], numerical_features=["a", "b"]
    )
    num = _num_pipeline(preprocessor)
    assert isinstance(num.named_steps["imputer"], SimpleImputer)
    assert num.named_steps["imputer"].strategy == "mean"
    # Индикаторы пропусков включены для числового пайплайна (issue #56)
    assert num.named_steps["imputer"].add_indicator is True


def test_build_preprocessor_median_imputation():
    """imputation_strategy='median' → SimpleImputer(strategy='median',
    add_indicator=True)."""
    preprocessor = build_preprocessor(
        ["a", "b"],
        categorical_features=[],
        numerical_features=["a", "b"],
        imputation_strategy="median",
    )
    num = _num_pipeline(preprocessor)
    assert num.named_steps["imputer"].strategy == "median"
    assert num.named_steps["imputer"].add_indicator is True


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
    передаются в модель без масштабирования; индикаторы пропусков добавляются."""
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
    # 2 импутированные колонки + 2 индикатора пропусков (по одному на признак)
    assert out.shape == (4, 4)
    assert np.isfinite(out).all()
    # Без масштабирования значения остаются в исходном диапазоне
    assert set(np.round(out[:, 0]).astype(int)) <= {1, 2, 4}
    assert set(np.round(out[:, 1]).astype(int)) <= {10, 20, 30}
    # Индикаторы: 1 там, где был NaN, 0 иначе
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 0])
    np.testing.assert_array_equal(out[:, 3], [0, 0, 0, 1])


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
    # 2 скалированные колонки + 1 индикатор пропуска (NaN только в num_a)
    assert out.shape == (5, 3)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 0, 0])


# ─────────────────── Индикаторы пропусков (issue #56) ───────────────────


def test_build_preprocessor_indicator_columns_on_fit_transform():
    """fit_transform с NaN в числовых признаках: индикаторы = 1 на пропусках,
    импутированные значения корректны."""
    df = pd.DataFrame(
        {"num_a": [1.0, 2.0, np.nan, 4.0], "num_b": [10.0, 20.0, 30.0, np.nan]}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        imputation_strategy="mean",
        scaling="none",
    )
    out = preprocessor.fit_transform(df)

    # 2 импутированные колонки + 2 индикатора (по одному на признак с NaN)
    assert out.shape == (4, 4)
    # Импутация средним: mean([1,2,4]) = 7/3, mean([10,20,30]) = 20
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 7 / 3, 4.0])
    np.testing.assert_allclose(out[:, 1], [10.0, 20.0, 30.0, 20.0])
    # Индикаторы: 1 там, где был NaN, 0 иначе (порядок = порядок признаков)
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 0])
    np.testing.assert_array_equal(out[:, 3], [0, 0, 0, 1])


def test_build_preprocessor_indicator_only_for_missing_features():
    """Полностью заполненный признак не получает индикатор (missing-only)."""
    df = pd.DataFrame(
        {"num_a": [1.0, 2.0, np.nan], "num_b": [10.0, 20.0, 30.0]}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        scaling="none",
    )
    out = preprocessor.fit_transform(df)

    # num_b заполнен полностью → индикатор только для num_a
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1])


def test_build_preprocessor_no_indicators_when_all_features_filled():
    """Все числовые признаки заполнены → выход без индикаторов
    (обратная совместимость с текущим поведением)."""
    df = pd.DataFrame({"num_a": [1.0, 2.0, 3.0], "num_b": [10.0, 20.0, 30.0]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        scaling="none",
    )
    out = preprocessor.fit_transform(df)

    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, df.to_numpy())


def test_build_preprocessor_all_nan_on_transform_indicator_ones():
    """Колонка со всеми NaN на transform: импутация константой из train +
    индикатор со всеми 1."""
    df_train = pd.DataFrame({"num_a": [1.0, np.nan, 3.0]})
    df_test = pd.DataFrame({"num_a": [np.nan, np.nan]})
    preprocessor = build_preprocessor(
        list(df_train.columns),
        categorical_features=[],
        numerical_features=list(df_train.columns),
        imputation_strategy="mean",
        scaling="none",
    )
    preprocessor.fit(df_train)
    out = preprocessor.transform(df_test)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 0], [2.0, 2.0])  # mean([1,3]) = 2
    np.testing.assert_array_equal(out[:, 1], [1, 1])


def test_build_preprocessor_missing_on_inference_without_indicator():
    """Пропуск на инференсе в признаке без пропусков на train: значение
    импутируется, индикатор не создаётся (features='missing-only')."""
    df_train = pd.DataFrame({"num_a": [1.0, 2.0, 3.0]})
    df_test = pd.DataFrame({"num_a": [1.0, np.nan, 3.0]})
    preprocessor = build_preprocessor(
        list(df_train.columns),
        categorical_features=[],
        numerical_features=list(df_train.columns),
        imputation_strategy="mean",
        scaling="none",
    )
    preprocessor.fit(df_train)
    out = preprocessor.transform(df_test)

    assert out.shape == (3, 1)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("scaling", ["standard", "robust", "none"])
def test_build_preprocessor_indicator_with_all_scaling_types(scaling):
    """Индикаторы корректно работают при scaling='standard'/'robust'/'none'."""
    df = pd.DataFrame({"num_a": [1.0, 2.0, np.nan, 4.0]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        scaling=scaling,
    )
    out = preprocessor.fit_transform(df)

    assert out.shape == (4, 2)
    assert np.isfinite(out).all()
    # Индикатор отделяет строку с пропуском при любом scaling:
    # значение в строке с NaN строго больше значений без пропуска
    indicator = out[:, 1]
    assert indicator[2] > indicator[0]
    assert indicator[2] > indicator[1]
    assert indicator[2] > indicator[3]


def test_build_preprocessor_indicator_end_to_end_with_model():
    """Сквозной сценарий: препроцессор + модель на данных с пропусками —
    обучение и предсказание работают; число колонок = базовое + число
    признаков с пропусками."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(size=n),
            "num_c": rng.normal(size=n),
        }
    )
    df.iloc[0:10, 0] = np.nan
    df.iloc[5:15, 2] = np.nan
    y = df["num_a"].fillna(0.0).to_numpy() + df["num_b"].to_numpy()

    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        scaling="none",
    )
    pipeline = make_pipeline(preprocessor, LinearRegression())
    pipeline.fit(df, y)
    preds = pipeline.predict(df)

    assert preds.shape == (n,)
    assert np.isfinite(preds).all()
    # 3 базовые колонки + 2 индикатора (пропуски были в num_a и num_c)
    assert preprocessor.transform(df).shape == (n, 5)


def test_build_preprocessor_categorical_imputer_without_indicator():
    """Категориальный пайплайн не меняется: SimpleImputer(strategy=
    'most_frequent') без add_indicator (индикаторы только для числовых)."""
    preprocessor = build_preprocessor(
        ["cat", "num"],
        categorical_features=["cat"],
        numerical_features=["num"],
    )
    cat_transformers = dict(
        (name, transformer)
        for name, transformer, _ in preprocessor.transformers
    )
    cat_imputer = cat_transformers["cat"].named_steps["imputer"]
    assert isinstance(cat_imputer, SimpleImputer)
    assert cat_imputer.strategy == "most_frequent"
    assert cat_imputer.add_indicator is False


def test_build_preprocessor_add_indicator_false_omits_indicators():
    """add_indicator=False отключает индикаторы пропусков (совместимость со
    строго одномерными алгоритмами вроде isotonic_regression)."""
    df = pd.DataFrame({"num_a": [1.0, 2.0, np.nan, 4.0]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        scaling="none",
        add_indicator=False,
    )
    out = preprocessor.fit_transform(df)

    # NaN импутирован, но индикаторная колонка не добавлена
    assert out.shape == (4, 1)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 7 / 3, 4.0])
    imputer = _num_pipeline(preprocessor).named_steps["imputer"]
    assert imputer.add_indicator is False


# ─────────────────── Выбор колонок по имени (issue #2) ───────────────────────


def _mixed_df() -> pd.DataFrame:
    """DataFrame с категориальной и числовой колонками для тестов issue #2."""
    return pd.DataFrame(
        {
            "cat": ["red", "green", "red", "blue"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_preprocessor_reordered_dataframe_transform_identical():
    """Переупорядоченный DataFrame на transform даёт тот же результат (issue #2).

    Регрессия: позиционные индексы в ColumnTransformer игнорируют имена колонок,
    и энкодер применялся к числовым колонкам, а скалер — к категориальным.
    """
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    train_out = preprocessor.fit_transform(df)

    reordered = df[["num", "cat"]]
    out = preprocessor.transform(reordered)

    assert out.shape == train_out.shape
    assert np.allclose(out, train_out)


def test_preprocessor_numpy_positional_fallback():
    """numpy-вход в обучающем порядке обрабатывается позиционно (issue #2)."""
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    train_out = preprocessor.fit_transform(df)

    out = preprocessor.transform(df.to_numpy())

    assert np.allclose(out, train_out)


def test_preprocessor_missing_column_raises():
    """Отсутствующая колонка на transform даёт явную ошибку (issue #2)."""
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    preprocessor.fit(df)

    with pytest.raises(ValueError):
        preprocessor.transform(df[["num"]])  # колонки 'cat' нет


def test_preprocessor_extra_columns_ignored():
    """Лишние колонки на transform не влияют на результат (issue #2)."""
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    train_out = preprocessor.fit_transform(df)

    extra = df.copy()
    extra["extra_col"] = 0.0
    out = preprocessor.transform(extra)

    assert np.allclose(out, train_out)


def test_build_preprocessor_unknown_feature_name_raises():
    """Имя колонки вне feature_names отклоняется на этапе сборки (issue #2).

    Раньше отсутствующие имена молча пропускались; теперь это явная ошибка.
    """
    with pytest.raises(ValueError, match="Unknown feature name"):
        build_preprocessor(
            ["cat", "num"],
            categorical_features=["cat", "missing_col"],
            numerical_features=["num"],
        )
    with pytest.raises(ValueError, match="Unknown feature name"):
        build_preprocessor(
            ["cat", "num"],
            categorical_features=["cat"],
            numerical_features=["num", "missing_col"],
        )


def test_column_name_selector_picklable():
    """ColumnNameSelector переживает pickle (важно для trainer.save())."""
    selector = ColumnNameSelector(["cat", "num"], ["cat"])
    restored = pickle.loads(pickle.dumps(selector))

    assert restored.feature_names == ["cat", "num"]
    assert restored.column_names == ["cat"]
    df = pd.DataFrame({"num": [1.0], "cat": ["x"]})
    assert restored(df) == [1]


def test_column_name_selector_dataframe_matching():
    """Для DataFrame селектор сопоставляет по имени, а не по позиции."""
    selector = ColumnNameSelector(["cat", "num"], ["cat", "num"])
    df = pd.DataFrame({"num": [1.0], "cat": ["x"]})  # порядок переставлен
    assert selector(df) == [1, 0]


def test_column_name_selector_numpy_fallback():
    """Для numpy-входа селектор использует обучающий порядок (feature_names)."""
    selector = ColumnNameSelector(["cat", "num"], ["num", "cat"])
    assert selector(np.zeros((1, 2))) == [1, 0]


def test_column_name_selector_missing_columns_raise():
    """Отсутствующие колонки дают явную ошибку для обоих типов входов."""
    selector = ColumnNameSelector(["a", "b"], ["a", "z"])
    with pytest.raises(ValueError, match="not found"):
        selector(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="training order"):
        selector(np.zeros((1, 2)))


def test_column_name_selector_duplicated_columns_raise():
    """Дублирующиеся колонки в DataFrame дают явную ошибку (неоднозначность)."""
    selector = ColumnNameSelector(["a", "b"], ["a"])
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="more than once"):
        selector(df)


def test_preprocessor_hc_and_target_reorder_identical():
    """High-cardinality + target кодирование устойчиво к переупорядочиванию."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "low": rng.choice(["a", "b", "c"], size=n),
            "high": [f"x{i % 40}" for i in range(n)],
            "num": rng.normal(size=n),
        }
    )
    y = pd.Series(
        df["low"].map({"a": 1.0, "b": 2.0, "c": 3.0}).to_numpy()
        + rng.normal(0, 0.1, n)
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=["num"],
        encoding="one_hot",
        high_cardinality_threshold=10,
        high_cardinality_encoding="target",
    )
    train_out = preprocessor.fit_transform(df, y)

    out = preprocessor.transform(df[["num", "high", "low"]])
    assert np.allclose(out, train_out)


@pytest.mark.parametrize(
    "enc", ["one_hot", "ordinal", "target", "frequency", "hashing"]
)
def test_preprocessor_reorder_all_encoding_strategies(enc):
    """Все стратегии кодирования корректны при переупорядочивании колонок."""
    df = pd.DataFrame(
        {"cat": ["a", "b", "a", "c"], "num": [1.0, 2.0, 3.0, 4.0]}
    )
    y = pd.Series([1.0, 2.0, 1.0, 3.0])
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding=enc,
    )
    train_out = preprocessor.fit_transform(df, y)

    out = preprocessor.transform(df[["num", "cat"]])
    # hashing-кодирование возвращает sparse (csr_matrix, issue #4):
    # np.allclose не поддерживает sparse-операнды, поэтому сравниваем плотно.
    expected = train_out.toarray() if sparse.issparse(train_out) else train_out
    actual = out.toarray() if sparse.issparse(out) else out
    assert np.allclose(actual, expected), f"encoding={enc}"


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

