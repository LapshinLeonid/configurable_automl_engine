"""Unit tests for the FeatureSelector transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from configurable_automl_engine.feature_selection import FeatureSelector


def _noisy_regression_data(
    n_samples: int = 500,
    n_informative: int = 3,
    n_noise: int = 7,
    coefs: tuple[float, ...] = (3.0, -2.0, 1.5, 0.5, 0.25),
    noise_scale: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic regression data: informative + noisy features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_informative + n_noise))
    informative = X[:, :n_informative]
    y = informative @ np.asarray(coefs[:n_informative])
    y = y + rng.normal(scale=noise_scale, size=n_samples)
    return X, y


# --------------------------------------------------------------------------- #
#                            Backend tests                                     #
# --------------------------------------------------------------------------- #


def test_importance_drops_noise_and_keeps_informative():
    """importance: noise is dropped, informative features are kept."""
    X, y = _noisy_regression_data()
    selector = FeatureSelector(method="importance", min_features=1, random_state=42)
    selector.fit(X, y)

    support = selector.get_support()
    assert support.shape[0] == X.shape[1]
    # All informative features remain in the support mask.
    assert support[:3].all()
    # Some noise features are dropped (the space is reduced).
    assert support.sum() < X.shape[1]
    Xt = selector.transform(X)
    assert Xt.shape == (X.shape[0], support.sum())


def test_percentile_selects_exactly_half_of_ten_features():
    """percentile=50 on 10 features leaves exactly 5."""
    X, y = _noisy_regression_data(
        n_informative=5, n_noise=5, coefs=(3.0, -2.0, 1.5, 0.5, 0.25)
    )
    selector = FeatureSelector(method="percentile", percentile=50.0, min_features=1)
    selector.fit(X, y)

    assert selector.get_support().sum() == 5
    assert selector.transform(X).shape == (X.shape[0], 5)
    # Of the kept half, the most significant f_regression features are present.
    informative = selector.get_support()[:5]
    assert informative.any()


def test_variance_removes_constant_columns():
    """variance: constant columns are removed, varying ones remain."""
    rng = np.random.default_rng(0)
    X = np.hstack(
        [
            rng.normal(size=(100, 3)),  # varying columns
            np.full((100, 4), 7.0),  # constant columns
        ]
    )
    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(X)

    support = selector.get_support()
    assert list(support[:3]) == [True, True, True]
    assert list(support[3:]) == [False, False, False, False]
    assert selector.transform(X).shape == (100, 3)


def test_mutual_info_on_nonlinear_synthetic():
    """mutual_info: nonlinearly related features are kept."""
    rng = np.random.default_rng(7)
    n_samples = 500
    x1 = rng.uniform(-np.pi, np.pi, size=n_samples)
    x2 = rng.uniform(-np.pi, np.pi, size=n_samples)
    noise = rng.normal(size=(n_samples, 4))
    X = np.column_stack([x1, x2, noise])
    y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(scale=0.05, size=n_samples)

    selector = FeatureSelector(
        method="mutual_info", percentile=50.0, min_features=1, random_state=42
    )
    selector.fit(X, y)

    support = selector.get_support()
    # Both nonlinearly informative features make it into the top half.
    assert support[0] and support[1]
    # 6 features * 50% -> exactly 3 (ceil).
    assert support.sum() == 3


# --------------------------------------------------------------------------- #
#                        Sparse compatibility                                 #
# --------------------------------------------------------------------------- #


def test_sparse_csr_input_preserves_sparse_output_variance():
    """csr_matrix on input -> csr_matrix on output with fewer columns."""
    rng = np.random.default_rng(3)
    dense = np.hstack(
        [
            rng.normal(size=(80, 3)),
            np.full((80, 2), 1.0),  # constant columns
        ]
    )
    X_sparse = sparse.csr_matrix(dense)

    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(X_sparse)

    Xt = selector.transform(X_sparse)
    assert sparse.issparse(Xt)
    assert isinstance(Xt, sparse.csr_matrix)
    assert Xt.shape == (80, 3)
    assert selector.get_support().sum() == 3


def test_sparse_csr_input_preserves_sparse_output_mutual_info():
    """Target-based method on sparse input: the output stays csr_matrix (no .toarray()).

    sklearn treats the features of a sparse matrix as discrete (``discrete_features
    == 'auto'``), so integer-valued synthetic data is used — exactly the data for
    which ``mutual_info_regression`` is correct on sparse input.
    """
    rng = np.random.default_rng(3)
    X_int = rng.integers(0, 5, size=(200, 6))
    y = 2.0 * X_int[:, 0] + 1.5 * X_int[:, 1] + rng.normal(scale=0.5, size=200)
    X_sparse = sparse.csr_matrix(X_int)

    selector = FeatureSelector(
        method="mutual_info", percentile=50.0, min_features=1, random_state=42
    )
    selector.fit(X_sparse, y)

    support = selector.get_support()
    assert support[0] and support[1]  # informative features at the top
    assert support.sum() == 3  # 6 features * 50% -> exactly 3

    Xt = selector.transform(X_sparse)
    assert sparse.issparse(Xt)
    assert isinstance(Xt, sparse.csr_matrix)
    assert Xt.shape == (200, 3)


@pytest.mark.parametrize("const_value", [3.0, 5.0, 7.0])
def test_sparse_variance_all_constant_keeps_min_features(const_value):
    """Sparse input with all-constant columns: the guard holds min_features.

    Regression from review: the naive formula ``E[x^2] - E[x]^2`` produced
    noisy non-zero "variances" (~1e-14) for constant columns on sparse input,
    so the precomputation diverged from sklearn's computation and
    ``VarianceThreshold.fit`` raised ``ValueError`` instead of triggering
    ``min_features_guard``. The value 5.0 reproduces the bug (scipy.sparse
    squaring loses precision for some values only); 3.0 is the review repro
    (10x5, min_features=2).
    """
    n_rows, n_cols, min_features = (10, 5, 2) if const_value == 3.0 else (50, 6, 3)
    X_sparse = sparse.csr_matrix(np.full((n_rows, n_cols), const_value))
    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=min_features
    )
    selector.fit(X_sparse)

    support = selector.get_support()
    assert support.sum() == min_features
    assert selector.n_selected_ == min_features
    # With fully equal variances, the lowest indices survive (tie-break).
    np.testing.assert_array_equal(support, np.arange(n_cols) < min_features)
    Xt = selector.transform(X_sparse)
    assert isinstance(Xt, sparse.csr_matrix)
    assert Xt.shape == (n_rows, min_features)


@pytest.mark.parametrize(
    "fmt",
    [
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.coo_matrix,
    ],
)
def test_sparse_format_preserved_on_transform(fmt):
    """The sparse matrix format is preserved: CSR/CSC/COO -> the same format."""
    rng = np.random.default_rng(11)
    dense = np.hstack(
        [
            rng.normal(size=(60, 3)),
            np.full((60, 2), 1.0),  # constant columns
        ]
    )
    X_sparse = fmt(dense)

    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(X_sparse)

    Xt = selector.transform(X_sparse)
    assert isinstance(Xt, fmt)
    assert Xt.shape == (60, 3)
    assert selector.get_support().sum() == 3


def test_sparse_variance_large_magnitude_no_false_guard():
    """Sparse with a large shift: variance is computed like in sklearn (no guard).

    Regression: for values around ``1e12 + noise`` the formula ``E[x^2]-E[x]^2``
    loses the whole signal (catastrophic cancellation) and wrongly sent varying
    columns to ``min_features_guard``. Variances are now computed the
    numerically stable sklearn way (``mean_variance_axis``).
    """
    rng = np.random.default_rng(0)
    X = 1e12 + rng.normal(size=(200, 4))
    X_sparse = sparse.csr_matrix(X)

    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(X_sparse)

    # All four columns vary -> kept without triggering the guard.
    assert selector.get_support().all()
    assert selector.n_selected_ == 4
    assert selector.transform(X_sparse).shape == (200, 4)


# --------------------------------------------------------------------------- #
#                       min_features_guard                                    #
# --------------------------------------------------------------------------- #


def test_min_features_guard_zero_variance_keeps_exact_minimum():
    """All features are constant (K=0): the guard keeps exactly min_features.

    With fully equal scores the lowest indices survive — a stable index-based
    tie-break (M1 from review).
    """
    X = np.full((50, 6), 3.0)
    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=3
    )
    selector.fit(X)

    support = selector.get_support()
    assert support.sum() == 3
    np.testing.assert_array_equal(support, [True, True, True, False, False, False])
    assert selector.transform(X).shape == (50, 3)
    assert selector.n_selected_ == 3


def test_top_indices_breaks_ties_by_index():
    """_top_indices: with equal scores the lowest indices survive."""
    from configurable_automl_engine.feature_selection import _top_indices

    scores = np.array([0.0, 0.0, 0.0, 5.0])
    np.testing.assert_array_equal(_top_indices(scores, 2), np.array([3, 0]))
    # Fully equal scores -> the first k indices in order.
    ties = np.zeros(6)
    np.testing.assert_array_equal(_top_indices(ties, 3), np.array([0, 1, 2]))
    # NaNs are treated as -inf and do not make it into the top.
    nan_scores = np.array([np.nan, 1.0, 2.0, np.nan])
    np.testing.assert_array_equal(_top_indices(nan_scores, 2), np.array([2, 1]))


def test_min_features_guard_partial_selection_takes_top_scores():
    """K between 1 and min_features: the guard tops up the selection by score."""
    rng = np.random.default_rng(5)
    X = np.column_stack(
        [
            rng.normal(size=100),  # the only varying column
            np.ones(100),
            np.ones(100),
            np.ones(100),
            np.ones(100),
        ]
    )
    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=3
    )
    selector.fit(X)

    support = selector.get_support()
    assert support.sum() == 3
    # The column with the maximum variance must remain.
    assert support[0]


def test_min_features_guard_importance_forces_minimum():
    """Plain importance selects 1 feature: the guard brings it up to min_features."""
    X, y = _noisy_regression_data(n_informative=1, n_noise=9, coefs=(5.0,))
    selector = FeatureSelector(method="importance", min_features=3, random_state=42)
    selector.fit(X, y)

    support = selector.get_support()
    assert support.sum() == 3
    # The most important feature (first column) is in the forced top.
    assert support[0]


def test_percentile_guard_extends_selection_to_min_features():
    """The guard extends the percentile selection (K=5) to min_features by scores."""
    X, y = _noisy_regression_data(n_informative=5, n_noise=5)
    selector = FeatureSelector(method="percentile", percentile=50.0, min_features=7)
    selector.fit(X, y)

    support = selector.get_support()
    assert support.sum() == 7
    # The five informative features have the maximum f_regression scores
    # and are guaranteed to be in the forced top.
    assert support[:5].all()


# --------------------------------------------------------------------------- #
#                          Edge cases                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_features", [1, 2])
def test_passthrough_when_features_leq_min_features(n_features):
    """P <= min_features: passthrough without changes (all-ones mask)."""
    X = np.random.default_rng(0).normal(size=(20, n_features))
    y = np.arange(20, dtype=float)
    selector = FeatureSelector(method="importance", min_features=2)
    selector.fit(X, y)

    support = selector.get_support()
    assert support.all()
    assert selector.n_selected_ == n_features
    assert selector.base_selector_ is None
    np.testing.assert_array_equal(selector.transform(X), X)


def test_unknown_method_raises_value_error():
    """Unknown method -> ValueError."""
    selector = FeatureSelector(method="bogus")
    X = np.random.default_rng(0).normal(size=(10, 4))
    with pytest.raises(ValueError, match="Unknown method"):
        selector.fit(X, np.arange(10, dtype=float))


def test_transform_before_fit_raises_not_fitted():
    """transform() before fit() -> NotFittedError."""
    selector = FeatureSelector()
    with pytest.raises(NotFittedError):
        selector.transform(np.zeros((5, 3)))


def test_clone_compatibility():
    """FeatureSelector is compatible with sklearn.base.clone()."""
    selector = FeatureSelector(
        method="percentile",
        percentile=60.0,
        min_features=3,
        random_state=7,
    )
    cloned = clone(selector)
    assert cloned.get_params() == selector.get_params()
    assert cloned is not selector

    X, y = _noisy_regression_data()
    cloned.fit(X, y)
    assert cloned.transform(X).shape[1] == cloned.get_support().sum()


# --------------------------------------------------------------------------- #
#                 Parameter and metadata validation                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["importance", "percentile", "mutual_info"])
def test_target_methods_require_y(method):
    """Target-requiring methods raise ValueError without y."""
    selector = FeatureSelector(method=method)
    X = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValueError, match="requires y"):
        selector.fit(X)


def test_variance_does_not_require_y():
    """variance fits without a target."""
    X = np.random.default_rng(0).normal(size=(10, 3))
    selector = FeatureSelector(method="variance", min_features=1)
    selector.fit(X)
    assert selector.get_support().all()


@pytest.mark.parametrize("percentile", [0.0, -5.0, 150.0])
def test_invalid_percentile_raises_value_error(percentile):
    """Invalid percentile -> ValueError."""
    selector = FeatureSelector(method="percentile", percentile=percentile)
    X = np.random.default_rng(0).normal(size=(10, 4))
    with pytest.raises(ValueError, match="percentile"):
        selector.fit(X, np.arange(10, dtype=float))


def test_invalid_min_features_raises_value_error():
    """min_features < 1 -> ValueError."""
    selector = FeatureSelector(method="variance", min_features=0)
    X = np.random.default_rng(0).normal(size=(10, 4))
    with pytest.raises(ValueError, match="min_features"):
        selector.fit(X)


def test_dataframe_transform_preserves_type_and_names():
    """DataFrame on input -> DataFrame on output with column names."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        np.column_stack([rng.normal(size=60), rng.normal(size=60), np.ones(60)]),
        columns=["a", "b", "const"],
    )
    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(df)

    Xt = selector.transform(df)
    assert isinstance(Xt, pd.DataFrame)
    assert list(Xt.columns) == ["a", "b"]

    names = selector.get_feature_names_out()
    assert list(names) == ["a", "b"]
    # Explicitly passed names take priority over feature_names_in_.
    explicit = selector.get_feature_names_out(["a", "b", "const"])
    assert list(explicit) == ["a", "b"]


def test_get_support_indices_mode():
    """get_support(indices=True) returns the positions of the selected features."""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=30), np.ones(30)])
    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(X)

    indices = selector.get_support(indices=True)
    np.testing.assert_array_equal(indices, np.array([0], dtype=int))


def test_get_feature_names_out_positional_defaults():
    """Without names: get_feature_names_out generates positional x0...xN."""
    X, y = _noisy_regression_data(n_informative=5, n_noise=5)
    selector = FeatureSelector(method="percentile", percentile=50.0, min_features=1)
    selector.fit(X, y)

    names = selector.get_feature_names_out()
    assert len(names) == selector.get_support().sum()
    assert all(name.startswith("x") for name in names)


def test_get_feature_names_out_rejects_wrong_length():
    """input_features with a mismatched length -> ValueError."""
    X, y = _noisy_regression_data()
    selector = FeatureSelector(method="variance", min_features=1)
    selector.fit(X)

    with pytest.raises(ValueError, match="input_features"):
        selector.get_feature_names_out(["a", "b"])


def test_transform_rejects_wrong_number_of_features():
    """transform with a mismatched number of columns -> ValueError."""
    X, y = _noisy_regression_data()
    selector = FeatureSelector(method="variance", min_features=1)
    selector.fit(X)

    with pytest.raises(ValueError, match="features"):
        selector.transform(np.zeros((10, X.shape[1] + 1)))


def test_transform_rejects_dataframe_with_wrong_column_order():
    """DataFrame with the same count but a different column name order -> ValueError.

    A positional slice with reordered columns would silently take the wrong
    features — names are checked against feature_names_in_ (like in sklearn).
    """
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        np.column_stack([rng.normal(size=40), rng.normal(size=40), np.ones(40)]),
        columns=["a", "b", "const"],
    )
    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(df)

    shuffled = df[["const", "a", "b"]]
    with pytest.raises(ValueError, match="feature names"):
        selector.transform(shuffled)


def test_pipeline_integration_and_feature_names_out():
    """Integration with sklearn.pipeline.Pipeline and get_feature_names_out."""
    from sklearn.pipeline import Pipeline

    X, y = _noisy_regression_data(n_informative=5, n_noise=5)
    pipeline = Pipeline(
        [
            (
                "selector",
                FeatureSelector(method="percentile", percentile=50.0, min_features=1),
            ),
        ]
    )
    pipeline.fit(X, y)

    Xt = pipeline.transform(X)
    assert Xt.shape == (X.shape[0], 5)

    names = pipeline.get_feature_names_out()
    assert len(names) == 5
    assert all(name.startswith("x") for name in names)

    # DataFrame through the pipeline: column names reach the selector.
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    pipeline.fit(df, y)
    names_df = pipeline.get_feature_names_out()
    assert len(names_df) == 5
    assert all(name.startswith("col_") for name in names_df)


def test_min_features_guard_mutual_info_extends_selection():
    """Guard for mutual_info: extends the percentile selection to min_features."""
    X, y = _noisy_regression_data(n_informative=3, n_noise=3)
    selector = FeatureSelector(
        method="mutual_info", percentile=50.0, min_features=5, random_state=42
    )
    selector.fit(X, y)

    support = selector.get_support()
    # 6 features * 50% -> K=3 < min_features=5 -> the guard activates the top-5.
    assert support.sum() == 5
    # Informative features have the maximum MI scores and are in the top.
    assert support[:3].all()


def test_invalid_n_estimators_raises_value_error():
    """n_estimators < 1 -> ValueError (explicit validation instead of a sklearn error)."""
    selector = FeatureSelector(method="importance", n_estimators=0)
    X = np.random.default_rng(0).normal(size=(10, 4))
    with pytest.raises(ValueError, match="n_estimators"):
        selector.fit(X, np.arange(10, dtype=float))


@pytest.mark.parametrize("threshold", [-0.1, -1.0, -100.0])
def test_invalid_negative_variance_threshold_raises_value_error(threshold):
    """Negative variance_threshold -> ValueError.

    Regression: a negative threshold makes no sense for VarianceThreshold and
    used to silently remove all columns; it is now rejected at fit() with a
    clear message.
    """
    selector = FeatureSelector(method="variance", variance_threshold=threshold)
    X = np.random.default_rng(0).normal(size=(10, 4))
    with pytest.raises(ValueError, match="variance_threshold"):
        selector.fit(X)


def test_refit_on_columnless_input_drops_stale_feature_names_in():
    """Refitting on nameless input drops stale feature_names_in_.

    Regression: after a fit on a DataFrame, a subsequent fit on
    an ndarray kept the DataFrame's feature_names_in_, so transforming a
    positionally correct array wrongly checked "names" and either failed or
    took the wrong columns.
    """
    rng = np.random.default_rng(6)
    df = pd.DataFrame(
        np.column_stack([rng.normal(size=40), rng.normal(size=40), np.ones(40)]),
        columns=["a", "b", "const"],
    )
    arr = np.asarray(df)

    selector = FeatureSelector(
        method="variance", variance_threshold=0.0, min_features=1
    )
    selector.fit(df)
    assert hasattr(selector, "feature_names_in_")
    # The constant column is dropped, varying ones remain.
    assert list(selector.get_support()) == [True, True, False]

    # Refit on an ndarray (no names): stale names are removed,
    # transform on the array works without name checks.
    selector.fit(arr)
    assert not hasattr(selector, "feature_names_in_")
    Xt = selector.transform(arr)
    assert Xt.shape == (40, 2)

    # Reverse transition: fitting on a DataFrame records names again.
    selector.fit(df)
    assert hasattr(selector, "feature_names_in_")
    np.testing.assert_array_equal(
        selector.feature_names_in_, np.asarray(["a", "b", "const"], dtype=object)
    )


def test_sparse_mutual_info_rejects_continuous_features():
    """float-sparse + mutual_info -> a clear ValueError (not a cryptic sklearn error).

    sklearn treats the features of a sparse matrix as discrete, and
    mutual_info_regression on continuous values fails with the obscure error
    "Found array with 0 sample(s)".
    """
    rng = np.random.default_rng(9)
    X_sparse = sparse.csr_matrix(rng.normal(size=(50, 4)))
    selector = FeatureSelector(
        method="mutual_info", percentile=50.0, min_features=1, random_state=42
    )
    with pytest.raises(ValueError, match="integer-valued"):
        selector.fit(X_sparse, np.arange(50, dtype=float))

    # Integer sparse passes without errors.
    X_int = sparse.csr_matrix(rng.integers(0, 5, size=(50, 4)))
    selector.fit(X_int, np.arange(50, dtype=float))
    assert selector.get_support().sum() == 2  # 4 * 50% -> exactly 2

    # float-sparse with integer values is also correct (discrete).
    X_float_int = sparse.csr_matrix(rng.integers(0, 5, size=(50, 4)).astype(np.float64))
    selector.fit(X_float_int, np.arange(50, dtype=float))
    assert selector.get_support().sum() == 2
