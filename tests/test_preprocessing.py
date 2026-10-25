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
    SplitCategoricalEncoder,
    _clean_uninformative_features,
    _to_string_array,
    build_preprocessor,
    detect_feature_types,
)


def test_detect_feature_types_mixed_dataframe():
    """detect_feature_types distinguishes categorical and numeric columns."""
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
    """On a purely numeric DataFrame there are no categorical columns."""
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.7]})
    cat, num = detect_feature_types(df)

    assert cat == []
    assert sorted(num) == ["a", "b"]


def test_build_preprocessor_onehot_for_categorical():
    """build_preprocessor returns a ColumnTransformer with OneHotEncoder for categories."""
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

    # The categorical transformer uses SplitCategoricalEncoder with a
    # default OneHotEncoder (high-cardinality mode is disabled).
    # Internal encoders are created in fit() (sklearn convention: attributes
    # with a trailing underscore appear only after training),
    # so we first fit the preprocessor on the data.
    df = pd.DataFrame(
        {"cat": ["red", "green", "red", "blue"], "num": [1.0, 2.0, 3.0, 4.0]}
    )
    preprocessor.fit(df)

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    assert isinstance(encoder.default_encoder_, OneHotEncoder)
    assert encoder.hc_encoder_ is None


def test_build_preprocessor_end_to_end_encoding():
    """Check end-to-end category encoding through the preprocessor."""
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

    assert out.shape == (4, 3 + 1)  # 3 one-hot columns + 1 numeric column


def test_build_preprocessor_no_features_passthrough():
    """A passthrough transformer is returned when no columns match."""
    preprocessor = build_preprocessor(
        ["some_random_column"],
        categorical_features=[],
        numerical_features=[],
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.transformers[0][0] == "bypass"
    assert preprocessor.transformers[0][1] == "passthrough"


def test_build_preprocessor_numeric_strings_treated_as_categorical():
    """'Numeric string' columns (ID '10','20') are detected as categorical by dtype."""
    df = pd.DataFrame({"id_code": ["10", "20", "30", "20"]})
    cat, _ = detect_feature_types(df)

    assert cat == ["id_code"]

    preprocessor = build_preprocessor(
        list(df.columns), categorical_features=cat, numerical_features=[]
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (4, 3)

    # All values in the matrix are numeric (one-hot)
    assert np.isfinite(out).all()


def test_build_preprocessor_bool_column_not_crash():
    """A pure bool column does not crash the preprocessor during fit_transform.

    Regression for a critical bug: SimpleImputer(strategy="most_frequent")
    failed on numpy-bool ("SimpleImputer does not support data with dtype bool").
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

    # bool -> one-hot (2 categories) + 1 numeric column
    assert out.shape == (4, 3)
    assert np.isfinite(out).all()


def test_build_preprocessor_mixed_object_and_bool():
    """Object and bool columns are processed together through one categorical
    pipeline without failing on the bool dtype."""
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

    # color (3 categories) + flag (2 categories) + 1 numeric column
    assert out.shape == (4, 6)
    assert np.isfinite(out).all()


# ───────────────────────── Ordinal encoding ─────────────────────────


def test_build_preprocessor_ordinal_uses_ordinal_encoder():
    """build_preprocessor(encoding='ordinal') uses OrdinalEncoder (not OneHotEncoder)."""
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

    # Internal encoders are created in fit(), so we fit first.
    df = pd.DataFrame({"cat": ["red", "green", "red"], "num": [1.0, 2.0, 3.0]})
    preprocessor.fit(df)

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    assert isinstance(encoder.default_encoder_, OrdinalEncoder)
    assert not isinstance(encoder.default_encoder_, OneHotEncoder)
    assert encoder.hc_encoder_ is None


def test_build_preprocessor_default_is_onehot():
    """By default (encoding unset / 'one_hot') OneHotEncoder is kept."""
    df = pd.DataFrame({"cat": ["red", "green", "red"], "num": [1.0, 2.0, 3.0]})
    for kwargs in ({}, {"encoding": "one_hot"}):
        preprocessor = build_preprocessor(
            ["cat", "num"],
            categorical_features=["cat"],
            numerical_features=["num"],
            **kwargs,
        )
        # Internal encoders are created in fit() (sklearn convention).
        preprocessor.fit(df)
        assert isinstance(
            preprocessor.named_transformers_["cat"].named_steps[
                "encoder"
            ].default_encoder_,
            OneHotEncoder,
        )


def test_build_preprocessor_ordinal_end_to_end_shape():
    """End-to-end ordinal encoding: categories are NOT expanded (one column per input column)."""
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

    # 1 ordinal column (for 'cat') + 1 numeric = 2
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_build_preprocessor_ordinal_bool_column():
    """A bool column with ordinal does not crash and yields a finite numeric matrix."""
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

    # 1 ordinal (flag) + 1 numeric column
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_build_preprocessor_ordinal_unknown_category_no_raise():
    """An unknown category on transform in ordinal mode does not raise an exception
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
    # unknown category -> -1
    assert out[0, 0] == -1.0


def test_build_preprocessor_invalid_encoding_raises():
    """Invalid encoding value -> ValueError."""
    with pytest.raises(ValueError, match="Unknown encoding strategy"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            encoding="binary",
        )


def test_build_preprocessor_ordinal_single_unique_category():
    """A categorical column with a single unique category yields 1 column."""
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
    """A fully missing categorical column is correctly ordinal-encoded
    after most_frequent imputation."""
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


def test_to_string_array_preserves_missing_values():
    """_to_string_array keeps NaN/None as missing values (not as 'nan'/'None' strings).

    Regression: missing values were stringified before the
    SimpleImputer ran, so the imputer never saw a gap. NaN became the literal
    "nan" category and None (e.g. from a JSON payload) became "None" — the
    same "unknown city" row produced different predictions depending on how
    the missing value was represented.
    """
    cases = [
        np.array([["a"], ["b"], [np.nan]], dtype=object),
        np.array([["a"], ["b"], [None]], dtype=object),
        np.array([[np.nan], [1.0]], dtype=float),
        pd.DataFrame({"x": ["a", "b", None]}),
        pd.DataFrame({"x": [np.nan, np.nan]}),
    ]
    for X in cases:
        out = _to_string_array(X)
        assert out.dtype == object
        expected_missing = int(pd.isna(np.asarray(X)).sum())
        assert int(pd.isna(out).sum()) == expected_missing, f"missing lost for {X!r}"
    # Non-missing values are still stringified.
    out = _to_string_array(pd.DataFrame({"x": [True, False]}))
    assert set(out.ravel().tolist()) == {"True", "False"}


def test_build_preprocessor_missing_categorical_imputed_consistently():
    """NaN and None in categorical columns are imputed identically.

    Regression: the categorical pipeline converts columns to
    strings, but the imputer must see the gaps. After the fix, a missing value
    is filled with the most-frequent category during fitting, and the one-hot
    matrix contains no "nan"/"None" category — NaN/None rows map to the same
    columns.
    """
    df = pd.DataFrame(
        {
            "city": ["msk", "spb", np.nan, "msk", np.nan, "spb"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["city"],
        numerical_features=["num"],
        encoding="one_hot",
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (6, 3)  # msk, spb + num — no "nan" column
    assert np.isfinite(out).all()

    enc = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cats = enc.default_encoder_.categories_[0]
    assert "nan" not in cats
    assert "None" not in cats
    # "msk" appears twice, NaN rows get imputed with "msk" (most frequent).
    assert out[:3, 0].tolist() == [1.0, 0.0, 1.0]

    # Inference: None and NaN (float-typed column) both resolve identically.
    nan_row = preprocessor.transform(pd.DataFrame({"city": [np.nan], "num": [1.0]}))
    none_row = preprocessor.transform(pd.DataFrame({"city": [None], "num": [1.0]}))
    known_row = preprocessor.transform(pd.DataFrame({"city": ["msk"], "num": [1.0]}))
    np.testing.assert_array_equal(nan_row, known_row)
    np.testing.assert_array_equal(none_row, known_row)


def test_build_preprocessor_missing_categorical_trained_on_none():
    """A column trained on None (e.g. from JSON) imputes None with the most-frequent value."""
    df = pd.DataFrame(
        {
            "city": ["msk", "spb", None, "msk", None, "spb"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["city"],
        numerical_features=["num"],
        encoding="one_hot",
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (6, 3)
    enc = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    assert "nan" not in enc.default_encoder_.categories_[0]

    none_row = preprocessor.transform(pd.DataFrame({"city": [None], "num": [1.0]}))
    known_row = preprocessor.transform(pd.DataFrame({"city": ["msk"], "num": [1.0]}))
    np.testing.assert_array_equal(none_row, known_row)


def test_get_feature_names_out_with_category_missing_values():
    """get_feature_names_out works on a preprocessor with missing categorical values.

    Regression: the "to_string" step had no get_feature_names_out,
    so calling it on the preprocessor raised AttributeError.
    """
    df = pd.DataFrame(
        {
            "city": ["msk", "spb", np.nan, "msk"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    for enc in ("one_hot", "ordinal", "frequency", "hashing"):
        preprocessor = build_preprocessor(
            list(df.columns),
            categorical_features=["city"],
            numerical_features=["num"],
            encoding=enc,
        )
        preprocessor.fit(df)
        names = preprocessor.get_feature_names_out()
        assert len(names) == preprocessor.transform(df.head(1)).shape[1]
        assert "city" in " | ".join(names)
        assert all(isinstance(n, str) for n in names)


def test_build_preprocessor_all_missing_categorical_imputed():
    """A fully-missing categorical column is imputed (kept) instead of being dropped.

    Regression: SimpleImputer(keep_empty_features=True) preserves
    the column, so the preprocessor keeps a stable dimensionality and
    get_feature_names_out stays consistent.
    """
    df = pd.DataFrame({"cat": [None, None, None], "num": [1.0, 2.0, 3.0]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="one_hot",
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (3, 2)
    assert np.isfinite(out).all()
    names = preprocessor.get_feature_names_out()
    assert len(names) == out.shape[1]
    # Inference with a non-missing value hits the single learned category.
    inf = preprocessor.transform(pd.DataFrame({"cat": ["msk"], "num": [1.0]}))
    assert inf.shape == (1, 2)
    assert np.isfinite(inf).all()


def test_build_preprocessor_ordinal_empty_features_passthrough():
    """Empty categorical/numerical -> passthrough ('bypass') regardless of encoding."""
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


# ─────────────────────────────── Adaptive presets ───────────────────────────────


def _num_pipeline(preprocessor: ColumnTransformer):
    """Extract the numeric pipeline from a ColumnTransformer."""
    transformers = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )
    assert "num" in transformers, "numeric transformer is missing"
    return transformers["num"]


def test_build_preprocessor_default_imputation_is_mean():
    """By default (scale_sensitive preset) — mean imputation with add_indicator."""
    preprocessor = build_preprocessor(
        ["a", "b"], categorical_features=[], numerical_features=["a", "b"]
    )
    num = _num_pipeline(preprocessor)
    assert isinstance(num.named_steps["imputer"], SimpleImputer)
    assert num.named_steps["imputer"].strategy == "mean"
    # Missing-value indicators are enabled for the numeric pipeline
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
    """scaling='none' → the scaling step is absent (trees)."""
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
    """scaling='standard' → StandardScaler (scale-sensitive models)."""
    preprocessor = build_preprocessor(
        ["a", "b"],
        categorical_features=[],
        numerical_features=["a", "b"],
        scaling="standard",
    )
    num = _num_pipeline(preprocessor)
    assert isinstance(num.named_steps["scaler"], StandardScaler)


def test_build_preprocessor_scaling_robust_uses_robust_scaler():
    """scaling='robust' → RobustScaler (GLM with skewed distributions)."""
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
    """Tree preset (median + none): missing values are filled with the median, data
    is passed to the model without scaling; missing-value indicators are added."""
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
    # 2 imputed columns + 2 missing-value indicators (one per feature)
    assert out.shape == (4, 4)
    assert np.isfinite(out).all()
    # Without scaling, values stay in their original range
    assert set(np.round(out[:, 0]).astype(int)) <= {1, 2, 4}
    assert set(np.round(out[:, 1]).astype(int)) <= {10, 20, 30}
    # Indicators: 1 where there was a NaN, 0 otherwise
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 0])
    np.testing.assert_array_equal(out[:, 3], [0, 0, 0, 1])


def test_glm_preset_end_to_end_robust():
    """GLM preset (median + robust): correct handling of outliers and missing values."""
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
    # 2 scaled columns + 1 missing-value indicator (NaN only in num_a)
    assert out.shape == (5, 3)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 0, 0])


# ───────────────────────── Missing-value indicators ─────────────────────────


def test_build_preprocessor_indicator_columns_on_fit_transform():
    """fit_transform with NaN in numeric features: indicators = 1 on missing values,
    imputed values are correct."""
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

    # 2 imputed columns + 2 indicators (one per feature with NaN)
    assert out.shape == (4, 4)
    # Mean imputation: mean([1,2,4]) = 7/3, mean([10,20,30]) = 20
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 7 / 3, 4.0])
    np.testing.assert_allclose(out[:, 1], [10.0, 20.0, 30.0, 20.0])
    # Indicators: 1 where there was a NaN, 0 otherwise (order = feature order)
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1, 0])
    np.testing.assert_array_equal(out[:, 3], [0, 0, 0, 1])


def test_build_preprocessor_indicator_only_for_missing_features():
    """A fully populated feature does not get an indicator (missing-only)."""
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

    # num_b is fully populated → indicator only for num_a
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:, 2], [0, 0, 1])


def test_build_preprocessor_no_indicators_when_all_features_filled():
    """All numeric features are populated → output without indicators
    (backward compatibility with the current behavior)."""
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
    """A column with all NaN on transform: imputation with the train constant +
    an indicator with all 1s."""
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
    """A missing value at inference in a feature with no train-time missing values:
    the value is imputed, no indicator is created (features='missing-only')."""
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
    """Indicators work correctly with scaling='standard'/'robust'/'none'."""
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
    # The indicator separates the row with a missing value under any scaling:
    # the value in the NaN row is strictly greater than the values without missing values
    indicator = out[:, 1]
    assert indicator[2] > indicator[0]
    assert indicator[2] > indicator[1]
    assert indicator[2] > indicator[3]


def test_build_preprocessor_indicator_end_to_end_with_model():
    """End-to-end scenario: preprocessor + model on data with missing values —
    training and prediction work; the number of columns = base + number of
    features with missing values."""
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
    # 3 base columns + 2 indicators (missing values were in num_a and num_c)
    assert preprocessor.transform(df).shape == (n, 5)


def test_build_preprocessor_categorical_imputer_without_indicator():
    """The categorical pipeline is unchanged: SimpleImputer(strategy=
    'most_frequent') without add_indicator (indicators only for numeric features)."""
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
    """add_indicator=False disables missing-value indicators (compatibility with
    strictly univariate algorithms such as isotonic_regression)."""
    df = pd.DataFrame({"num_a": [1.0, 2.0, np.nan, 4.0]})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        scaling="none",
        add_indicator=False,
    )
    out = preprocessor.fit_transform(df)

    # NaN is imputed, but no indicator column is added
    assert out.shape == (4, 1)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0, 7 / 3, 4.0])
    imputer = _num_pipeline(preprocessor).named_steps["imputer"]
    assert imputer.add_indicator is False


# ─────────────────────────── Column selection by name ──────────────────────────


def _mixed_df() -> pd.DataFrame:
    """A DataFrame with categorical and numeric columns for these tests."""
    return pd.DataFrame(
        {
            "cat": ["red", "green", "red", "blue"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_preprocessor_reordered_dataframe_transform_identical():
    """A reordered DataFrame on transform yields the same result.

    Regression: positional indices in ColumnTransformer ignored column names,
    and the encoder was applied to numeric columns while the scaler to categorical ones.
    """
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    train_out = preprocessor.fit_transform(df)

    reordered = df[["num", "cat"]]
    out = preprocessor.transform(reordered)

    assert out.shape == train_out.shape
    assert np.allclose(out, train_out)


def test_preprocessor_numpy_positional_fallback():
    """A numpy input in training order is handled positionally."""
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    train_out = preprocessor.fit_transform(df)

    out = preprocessor.transform(df.to_numpy())

    assert np.allclose(out, train_out)


def test_preprocessor_missing_column_raises():
    """A missing column on transform raises an explicit error."""
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    preprocessor.fit(df)

    with pytest.raises(ValueError):
        preprocessor.transform(df[["num"]])  # the 'cat' column is absent


def test_preprocessor_extra_columns_ignored():
    """Extra columns on transform do not affect the result."""
    df = _mixed_df()
    preprocessor = build_preprocessor(list(df.columns), ["cat"], ["num"])
    train_out = preprocessor.fit_transform(df)

    extra = df.copy()
    extra["extra_col"] = 0.0
    out = preprocessor.transform(extra)

    assert np.allclose(out, train_out)


def test_build_preprocessor_unknown_feature_name_raises():
    """A column name outside feature_names is rejected at build time.

    Previously missing names were silently skipped; now it is an explicit error.
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
    """ColumnNameSelector survives pickle (important for trainer.save())."""
    selector = ColumnNameSelector(["cat", "num"], ["cat"])
    restored = pickle.loads(pickle.dumps(selector))

    assert restored.feature_names == ["cat", "num"]
    assert restored.column_names == ["cat"]
    df = pd.DataFrame({"num": [1.0], "cat": ["x"]})
    assert restored(df) == [1]


def test_column_name_selector_dataframe_matching():
    """For a DataFrame the selector matches by name, not by position."""
    selector = ColumnNameSelector(["cat", "num"], ["cat", "num"])
    df = pd.DataFrame({"num": [1.0], "cat": ["x"]})  # order reordered
    assert selector(df) == [1, 0]


def test_column_name_selector_numpy_fallback():
    """For a numpy input the selector uses the training order (feature_names)."""
    selector = ColumnNameSelector(["cat", "num"], ["num", "cat"])
    assert selector(np.zeros((1, 2))) == [1, 0]


def test_column_name_selector_missing_columns_raise():
    """Missing columns raise an explicit error for both input types."""
    selector = ColumnNameSelector(["a", "b"], ["a", "z"])
    with pytest.raises(ValueError, match="not found"):
        selector(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="training order"):
        selector(np.zeros((1, 2)))


def test_column_name_selector_duplicated_columns_raise():
    """Duplicate columns in a DataFrame raise an explicit error (ambiguity)."""
    selector = ColumnNameSelector(["a", "b"], ["a"])
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="more than once"):
        selector(df)


def test_preprocessor_hc_and_target_reorder_identical():
    """High-cardinality + target encoding is robust to reordering.

    Two transform calls are compared (full statistics): the fit_transform of
    target encoding returns cross-fitted values and is not required to match
    transform on the same rows.
    """
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
    preprocessor.fit(df, y)

    out = preprocessor.transform(df)
    reordered = preprocessor.transform(df[["num", "high", "low"]])
    assert np.allclose(reordered, out)


@pytest.mark.parametrize(
    "enc", ["one_hot", "ordinal", "target", "frequency", "hashing"]
)
def test_preprocessor_reorder_all_encoding_strategies(enc):
    """All encoding strategies are correct when columns are reordered.

    For target encoding, transform calls are compared (full statistics):
    fit_transform returns cross-fitted values and is not required to match
    transform on the same rows.
    """
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
    preprocessor.fit(df, y)

    out = preprocessor.transform(df)
    reordered = preprocessor.transform(df[["num", "cat"]])
    # hashing encoding returns sparse (csr_matrix):
    # np.allclose does not support sparse operands, so we compare densely.
    expected = reordered.toarray() if sparse.issparse(reordered) else reordered
    actual = out.toarray() if sparse.issparse(out) else out
    assert np.allclose(actual, expected), f"encoding={enc}"


# ──────────────────────── Sparse output of hashing ───────────────────────


def test_build_preprocessor_hashing_sparse_by_default():
    """hashing encoding returns a csr_matrix through ColumnTransformer
    (the default sparse_threshold=0.3 yields csr with any sparse part)."""
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
    """force_dense_output=True converts the sparse parts (hashing) to dense:
    required for GPR/Isotonic/ARD, which reject scipy.sparse."""
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
    """force_dense_output does not break the passthrough branch (no matching columns)."""
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


# ──────────────── Sparse one-hot output ────────────────


def test_split_encoder_one_hot_is_sparse_by_default():
    """OneHotEncoder is created with sparse_output=True (no dense n×n blow-up).

    Previously the default dense one-hot of an ID-like column allocated a
    dense n×n float matrix per fold (≈3 GB at 20k rows). The encoder now keeps
    the sparse representation, so a high-cardinality column does not explode.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"id": [f"id_{i}" for i in range(2000)]})
    enc = SplitCategoricalEncoder(encoding="one_hot")
    out = enc.fit_transform(X)
    assert enc.default_encoder_.sparse_output is True
    assert sparse.issparse(out)
    assert out.shape == (2000, 2000)
    # nnz = n_rows * n_cols (one active column per row per input column).
    assert out.nnz == 2000


def test_build_preprocessor_one_hot_sparse_for_high_cardinality():
    """The ColumnTransformer keeps a sparse one-hot output when the density is low.

    A single high-cardinality one-hot column + a numeric column stays
    sparse: the dense float64 matrix would be n×(n+1) — the exact blow-up
    the sparse output prevents.
    """
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"id": [f"id_{i}" for i in range(3000)], "num": rng.normal(size=3000)}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["id"],
        numerical_features=["num"],
        encoding="one_hot",
    )
    out = preprocessor.fit_transform(df)
    assert sparse.issparse(out)
    assert out.format == "csr"
    assert out.shape == (3000, 3001)
    # One non-zero per row in the one-hot part + the dense numeric column:
    # the stored elements must be far below the dense n×(n+1).
    assert out.nnz <= 3000 + 3000


def test_build_preprocessor_one_hot_dense_when_density_high():
    """Small-cardinality one-hot stays dense (no behavior regression)."""
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 4, "num": np.arange(12.0)})
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="one_hot",
    )
    out = preprocessor.fit_transform(df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (12, 4)


def test_build_preprocessor_one_hot_sparse_force_dense():
    """force_dense_output converts sparse one-hot to a dense matrix
    (for GPR/Isotonic/ARD, which reject scipy.sparse)."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"id": [f"id_{i}" for i in range(500)], "num": rng.normal(size=500)}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["id"],
        numerical_features=["num"],
        encoding="one_hot",
        force_dense_output=True,
    )
    out = preprocessor.fit_transform(df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (500, 501)
    assert np.isfinite(out).all()


def test_build_preprocessor_one_hot_sparse_transform_consistency():
    """transform() after fit() keeps the sparse representation that fit chose."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"id": [f"id_{i}" for i in range(600)], "num": rng.normal(size=600)}
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["id"],
        numerical_features=["num"],
        encoding="one_hot",
    )
    preprocessor.fit(df)
    out = preprocessor.transform(df)
    assert sparse.issparse(out)
    assert out.shape == (600, 601)



# ───────────────────── Cleaning of uninformative features ─────────────────────


def test_clean_drops_all_nan_column():
    """A column with 100% NaN is dropped; the other columns are kept."""
    df = pd.DataFrame(
        {
            "good_num": [1.0, 2.0, 3.0],
            "empty": [np.nan, np.nan, np.nan],
            "good_cat": ["a", "b", "a"],
        }
    )
    out = _clean_uninformative_features(df)

    assert list(out.columns) == ["good_num", "good_cat"]
    assert out.equals(df[["good_num", "good_cat"]])


def test_clean_drops_constant_numeric_column():
    """A strictly constant numeric column is dropped."""
    df = pd.DataFrame({"informative": [1.0, 2.0, 3.0], "const": [5.0, 5.0, 5.0]})
    out = _clean_uninformative_features(df)

    assert list(out.columns) == ["informative"]


def test_clean_drops_constant_categorical_column():
    """A constant categorical column is dropped (useless after one-hot)."""
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "const_cat": ["only", "only", "only"],
            "multi_cat": ["x", "y", "x"],
        }
    )
    out = _clean_uninformative_features(df)

    assert list(out.columns) == ["num", "multi_cat"]


def test_clean_nan_ratio_threshold_boundary():
    """nan_ratio exactly at the threshold (0.95) — the column is dropped; slightly below — it is kept."""
    # A column with a missing-value ratio of exactly 0.95: 19 NaN out of 20.
    df_border = pd.DataFrame({"good": np.arange(20.0)})
    df_border["border"] = [np.nan] * 19 + [1.0]
    out_at = _clean_uninformative_features(df_border, max_nan_ratio=0.95)
    assert "border" not in out_at.columns

    # Slightly below the threshold: 18/20 = 0.9 < 0.95 — the column is kept.
    df_below = df_border.copy()
    df_below["border"] = [np.nan] * 18 + [1.0, 2.0]
    out_below = _clean_uninformative_features(df_below, max_nan_ratio=0.95)
    assert "border" in out_below.columns


def test_clean_combination_single_unique_with_nans_dropped():
    """A column with a single unique value after dropna and 10% NaN is dropped."""
    df = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "almost_const": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, np.nan, 7.0],
        }
    )
    out = _clean_uninformative_features(df)

    assert list(out.columns) == ["good"]


def test_clean_min_unique_two_keeps_two_unique_values():
    """min_unique=2: a column with exactly 2 unique values is kept."""
    df = pd.DataFrame({"a": [1.0, 2.0, 1.0, 2.0], "b": [0.0, 0.0, 0.0, 0.0]})
    out = _clean_uninformative_features(df, min_unique=2)

    assert list(out.columns) == ["a"]


def test_clean_informative_two_unique_column_not_dropped():
    """An informative column with two unique values is not dropped."""
    df = pd.DataFrame({"flag": [0, 1, 0, 1], "num": [1.0, 2.0, 3.0, 4.0]})
    out = _clean_uninformative_features(df)

    assert list(out.columns) == ["flag", "num"]


def test_clean_all_columns_dropped_raises():
    """All columns are uninformative -> ValueError (guard against an empty matrix)."""
    df = pd.DataFrame({"c1": [1.0, 1.0, 1.0], "c2": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError, match="uninformative"):
        _clean_uninformative_features(df)


def test_clean_single_column_preserved_even_if_constant():
    """The only column is kept even if it is constant or empty."""
    const_df = pd.DataFrame({"only": [1.0, 1.0, 1.0]})
    out_const = _clean_uninformative_features(const_df)
    assert list(out_const.columns) == ["only"]

    empty_df = pd.DataFrame({"only": [np.nan, np.nan, np.nan]})
    out_empty = _clean_uninformative_features(empty_df)
    assert list(out_empty.columns) == ["only"]


def test_clean_is_idempotent():
    """Calling again on already cleaned data does not change the result."""
    df = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0, 4.0],
            "const": [0.5, 0.5, 0.5, 0.5],
            "all_nan": [np.nan] * 4,
        }
    )
    once = _clean_uninformative_features(df)
    twice = _clean_uninformative_features(once)

    assert list(twice.columns) == list(once.columns)
    assert twice.equals(once)


def test_clean_clean_data_unchanged():
    """Columns without missing values and with variance are kept (regression)."""
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
    )
    out = _clean_uninformative_features(df)

    assert list(out.columns) == ["a", "b"]
    assert out.equals(df)


def test_clean_empty_dataframe_no_crash():
    """An empty DataFrame (0 rows / 0 columns) is handled without crashes."""
    empty_cols = pd.DataFrame()
    out_cols = _clean_uninformative_features(empty_cols)
    assert out_cols.shape == (0, 0)

    empty_rows = pd.DataFrame({"a": [], "b": []})
    out_rows = _clean_uninformative_features(empty_rows)
    assert list(out_rows.columns) == ["a", "b"]
    assert out_rows.shape == (0, 2)


def test_clean_numpy_input_raises_type_error():
    """An np.ndarray without column names is rejected with an explicit error (documented
    limitation, same as detect_feature_types)."""
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        _clean_uninformative_features(np.zeros((5, 3)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_nan_ratio": 0.0}, "max_nan_ratio"),
        ({"max_nan_ratio": 1.5}, "max_nan_ratio"),
        ({"min_unique": 0}, "min_unique"),
    ],
)
def test_clean_invalid_thresholds_raise(kwargs, match):
    """Invalid thresholds are rejected with an explicit error."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match=match):
        _clean_uninformative_features(df, **kwargs)


def test_clean_does_not_mutate_input():
    """The function does not mutate the input DataFrame."""
    df = pd.DataFrame({"good": [1.0, 2.0], "const": [0.0, 0.0]})
    _clean_uninformative_features(df)

    assert list(df.columns) == ["good", "const"]


def test_clean_removes_zero_variance_constant_column_via_preprocessor():
    """Motivation: a constant column yields zero variance through
    the preprocessor (q75 == q25 for RobustScaler; in older sklearn versions —
    NaN/±inf due to division by zero IQR), and cleaning removes the garbage before build."""
    df = pd.DataFrame(
        {"informative": [1.0, 2.0, 3.0, 4.0, 5.0], "const": [7.0] * 5}
    )

    # Without cleaning, a constant column passes through the scaler with zero
    # variance — a "garbage" dimension in the distance matrix of metric
    # models (SVR/KNN/Ridge).
    dirty_preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=[],
        numerical_features=list(df.columns),
        imputation_strategy="median",
        scaling="robust",
    )
    dirty_out = dirty_preprocessor.fit_transform(df)
    assert np.isfinite(dirty_out).all()
    assert dirty_out[:, 1].std() == 0.0  # constant column -> zero variance

    # With cleaning, the constant column is dropped, leaving one informative
    # feature with non-zero variance.
    cleaned = _clean_uninformative_features(df)
    assert list(cleaned.columns) == ["informative"]
    clean_preprocessor = build_preprocessor(
        list(cleaned.columns),
        categorical_features=[],
        numerical_features=list(cleaned.columns),
        imputation_strategy="median",
        scaling="robust",
    )
    clean_out = clean_preprocessor.fit_transform(cleaned)
    assert np.isfinite(clean_out).all()
    assert clean_out.shape == (5, 1)
    assert clean_out[:, 0].std() > 0.0
