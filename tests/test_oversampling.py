import pytest
import pandas as pd
import numpy as np
import threading
import pickle
import logging
from scipy import sparse
from unittest.mock import patch
from math import ceil

from pandas.api.types import is_numeric_dtype

# Imports according to your project structure
from configurable_automl_engine.oversampling import (
    CONTINUOUS_TARGET_MIN_BINS,
    CONTINUOUS_TARGET_QUANTILES,
    DataOversampler,
    _bin_continuous_target,
    _bin_regression_target,
    _binary_indicator_columns,
    _decode_continuous_y,
    _is_categorical_col,
    _is_continuous_target,
    _prepare_continuous_bins,
    oversample as functional_oversample,
    oversample,
)


class TestDataOversampler:
    """Full test suite for DataOversampler (20 scenarios)"""

    @pytest.fixture
    def sample_data(self):
        """Standard dataset: 3 samples of class '0', 2 samples of class '1'"""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "target": [0, 0, 0, 1, 1],
            }
        )

    # --- 1. ALGORITHM TESTS (Basic scenarios) ---

    def test_basic_oversampling_smote(self, sample_data):
        """Test SMOTE: multiplier=2 -> 10 rows"""
        sampler = DataOversampler(multiplier=2, algorithm="smote")
        result = sampler.oversample(sample_data, target="target")
        assert len(result) == 10
        assert "target" in result.columns
        assert result.notna().all().all()

    def test_basic_oversampling_adasyn(self):
        # Create more data so the minority class has
        # neighbors from the majority class
        data = pd.DataFrame({"f1": np.arange(20), "target": [0] * 15 + [1] * 5})
        sampler = DataOversampler(multiplier=1.2, algorithm="adasyn")
        result = sampler.oversample(data, target="target")
        assert len(result) > 20
        assert result.notna().all().all()

    def test_random_without_noise(self, sample_data):
        """Test Random without noise"""
        sampler = DataOversampler(multiplier=3, algorithm="random", add_noise=False)
        result = sampler.oversample(sample_data)
        assert len(result) == 15
        # Check that the result is just duplicates
        assert result.duplicated().any()

    def test_random_with_noise_flag(self, sample_data):
        """Test the add_noise flag in Random"""
        sampler = DataOversampler(multiplier=3, algorithm="random", add_noise=True)
        result = sampler.oversample(sample_data)
        # Values must be unique due to Gaussian noise
        assert not result.drop("target", axis=1).duplicated().any()

    # --- 2. DATA TYPE TESTS (sklearn validation) ---

    def test_dataframe_input_output(self, sample_data):
        """DataFrame input -> DataFrame output"""
        sampler = DataOversampler()
        result = sampler.oversample(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_array_input(self):
        """numpy array input via fit_resample (imblearn standard)"""
        X = np.random.rand(10, 2)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        sampler = DataOversampler(multiplier=2)
        X_res, y_res = sampler.fit_resample(X, y)
        assert isinstance(X_res, np.ndarray)
        assert len(X_res) == 20

    def test_list_input(self):
        """list input (must be handled by the internal _check_X_y)"""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 0, 1, 1]
        sampler = DataOversampler(multiplier=2)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) == 8

    # --- 3. PARAMETERS AND BOUNDARIES ---

    def test_multiplier_1(self, sample_data):
        """multiplier=1 must not change the data size"""
        sampler = DataOversampler(multiplier=1.0)
        result = sampler.oversample(sample_data)
        assert len(result) == len(sample_data)

    def test_fractional_multiplier(self, sample_data):
        """Test the fractional multiplier (ceil)"""
        sampler = DataOversampler(multiplier=1.5)  # 3*1.5=4.5->5, 2*1.5=3. Total 8.
        result = sampler.oversample(sample_data)
        assert len(result) == 8

    def test_invalid_multiplier(self, sample_data):
        sampler = DataOversampler(multiplier=-1.0)
        # The error will be raised inside fit_resample -> _strategy -> ValueError
        # when building the strategy dictionary
        with pytest.raises(Exception):
            sampler.oversample(sample_data)

    def test_unknown_algorithm(self, sample_data):
        """Unsupported algorithm"""
        sampler = DataOversampler(algorithm="invalid_method")
        with pytest.raises(ValueError, match="Unsupported algorithm:"):
            sampler.oversample(sample_data)

    # --- 4. SPECIFIC LOGIC (New code) ---

    def test_smote_single_minority_sample_error(self):
        """
        SMOTE is mathematically undefined with a single minority-class sample.
        We expect a ValueError per the imbalanced-learn contract.
        """
        data = pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 0, 0, 1]})

        sampler = DataOversampler(algorithm="smote", multiplier=2.0)

        with pytest.raises(ValueError):
            sampler.oversample(data)

    def test_find_target_in_middle(self):
        """Finds the target when it is not the last column"""
        df = pd.DataFrame(
            {"f1": [1, 2, 3, 4], "target": [0, 0, 1, 1], "f2": [5, 6, 7, 8]}
        )
        sampler = DataOversampler(multiplier=2)
        result = sampler.oversample(df, target="target")
        # Building in oversample() moves the target to the end
        assert list(result.columns) == ["f1", "f2", "target"]
        assert len(result) == 8

    # --- 5. SYSTEM TESTS ---

    def test_logging_error(self, caplog):
        sampler = DataOversampler()
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                sampler.oversample(None)
        assert "ERROR" in caplog.text
        assert "Oversampling error" in caplog.text

    def test_pickle_compatibility(self):
        """Pickle (important for multiprocessing in GridSearchCV)"""
        sampler = DataOversampler(multiplier=2, algorithm="smote")

        dump = pickle.dumps(sampler)
        load = pickle.loads(dump)

        assert load.algorithm == "smote"

        # lock was restored
        assert hasattr(load, "_lock")
        assert load._lock is not None

        # lock is reentrant (a key property)
        acquired = load._lock.acquire(blocking=False)
        try:
            assert acquired is True
        finally:
            if acquired:
                load._lock.release()

    def test_multithreading_safety(self, sample_data):
        """Check the lock when logging from different threads"""
        sampler = DataOversampler()

        def run():
            for _ in range(5):
                sampler.oversample(sample_data)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_functional_interface(self, sample_data):
        """Test the quick wrapper function"""
        result = functional_oversample(sample_data, multiplier=2, algorithm="random")
        assert len(result) == 10


@pytest.fixture
def sample_data():
    """Creates a minimal DataFrame for tests."""
    return pd.DataFrame(
        {"feature1": np.random.rand(10), "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]}
    )


def test_critical_error_logging():
    # 1. Prepare the data
    X = pd.DataFrame(np.random.rand(10, 2), columns=["f1", "f2"])
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Balanced data

    # 2. Create an instance
    sampler = DataOversampler(algorithm="random")

    # 3. Patch only the logger in the oversampling module
    with patch("configurable_automl_engine.oversampling.sampler.logger") as mock_logger:
        # 4. Simulate an "unexpected" error through the _strategy method
        # This method is called inside _fit_resample RIGHT AFTER _check_X_y
        error_text = "Unexpected Crash during Strategy"

        with patch.object(sampler, "_strategy", side_effect=AttributeError(error_text)):
            # Check that the error (AttributeError) is propagated outward
            with pytest.raises(AttributeError, match=error_text):
                sampler.fit_resample(X, y)

        # 5. LOG CHECKS (the very lines 136-139)

        # Check logger.debug(..., exc_info=True)
        mock_logger.debug.assert_called_once_with(
            "Unexpected error trace in _fit_resample:", exc_info=True
        )

        # Check logger.error(f"Critical error...")
        mock_logger.error.assert_called_once_with(
            f"Critical error during data oversampling: {error_text}"
        )


def test_catch_generic_exception_only():
    """
    Verifies that the block catches exactly Exception,
    which is not in the list (ValueError, TypeError, etc.)
    """
    with patch("configurable_automl_engine.oversampling.sampler.logger") as mock_log:
        sampler = DataOversampler()

        # Raise an error that this block must NOT log (ValueError)
        # because it is caught higher up and simply re-raised
        with patch.object(
            sampler, "_check_X_y", side_effect=ValueError("Standard Data Error")
        ):
            with pytest.raises(ValueError):
                sampler.fit_resample(None, None)

        # The logger must not have been called because ValueError was caught by the first except
        assert mock_log.error.call_count == 0


# 1. Coverage of lines 90-91: Noise in object columns (containing numbers)
def test_noise_on_object_numeric_cols():
    df = pd.DataFrame(
        {
            "A": ["1.0", "2.0", "3.0", "4.0"],  # Object, but numeric inside
            "target": [0, 0, 1, 1],
        }
    )
    # Use a multiplier of 1 and noise
    sampler = DataOversampler(multiplier=1.0, add_noise=True, noise_level=0.1)
    res = sampler.oversample(df, target="target")

    # Check that the values in 'A' changed (noise was added)
    assert not np.array_equal(
        res["A"].astype(float).values, df["A"].astype(float).values
    )


# 2. Coverage of lines 131-140: Serialization (pickle)
def test_serialization_pickle():
    sampler = DataOversampler(multiplier=2.0, algorithm="random")
    # Check the removal and restoration of _lock
    dumped = pickle.dumps(sampler)
    loaded = pickle.loads(dumped)

    assert loaded.multiplier == 2.0
    assert hasattr(loaded, "_lock")
    assert isinstance(loaded._lock, type(sampler._lock))


# 3. Coverage of lines 180, 184, 189, 193: Type restoration (int/float)
@pytest.mark.parametrize(
    "dtype, add_noise, expected_dtype",
    [
        (np.int64, True, np.float64),  # Line 180
        (np.int32, True, np.float32),  # Line 184 (via else)
        (np.float32, False, np.float32),  # Line 189
    ],
)
def test_dtype_restoration(dtype, add_noise, expected_dtype):
    df = pd.DataFrame({"A": np.array([1, 2], dtype=dtype), "target": [0, 1]})
    sampler = DataOversampler(multiplier=1.0, add_noise=add_noise)
    res = sampler.oversample(df, target="target")
    assert res["A"].dtype == expected_dtype


# 4. Coverage of line 211: Error when multiplier < 1
def test_invalid_multiplier():
    sampler = DataOversampler(multiplier=0.5)
    df = pd.DataFrame({"A": [1, 2], "target": [0, 1]})
    with pytest.raises(ValueError, match="multiplier must be >= 1"):
        sampler.fit_resample(df[["A"]], df["target"])


# 5. Noise + sparse matrix: only the stored non-zero elements are noised
def test_sparse_random_noise_keeps_sparsity_pattern():
    """Random + add_noise on a csr_matrix stays sparse, zeros stay zeros and
    every non-zero element is perturbed."""
    X_sparse = sparse.csr_matrix(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 6.0],
            [7.0, 0.0, 8.0],
            [0.0, 9.0, 0.0],
        ]
    )
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    sampler = DataOversampler(
        algorithm="random",
        multiplier=2.0,
        add_noise=True,
        noise_level=0.1,
        target_bins=2,
        random_state=0,
    )
    X_res, y_res = sampler.fit_resample(X_sparse, y)
    assert sparse.issparse(X_res)
    assert X_res.shape[1] == 3
    assert len(y_res) == X_res.shape[0] > X_sparse.shape[0]
    clean = DataOversampler(
        algorithm="random", multiplier=2.0, target_bins=2, random_state=0
    )
    X_clean, _ = clean.fit_resample(X_sparse, y)
    dense_noisy = X_res.toarray()
    dense_clean = X_clean.toarray()
    # The sparsity pattern is unchanged: zeros are not noised
    np.testing.assert_array_equal(dense_noisy == 0, dense_clean == 0)
    # Non-zero elements are perturbed
    nz = dense_clean != 0
    assert np.all(dense_noisy[nz] != dense_clean[nz])
    # The input matrix is not modified in place
    assert X_sparse[0, 0] == 1.0


def test_sparse_noise_scale_uses_full_column_std():
    """The noise scale is noise_level * column std over all rows (ddof=1)."""
    rng = np.random.default_rng(0)
    dense = rng.normal(5.0, 2.0, size=(2000, 2)) * (rng.random((2000, 2)) < 0.5)
    X_sparse = sparse.csr_matrix(dense)
    sampler = DataOversampler(add_noise=True, noise_level=0.1, random_state=1)
    X_noisy = sampler._add_sparse_gaussian_noise(X_sparse)
    diff = (X_noisy - X_sparse.astype(np.float64)).tocsc()
    for j in range(2):
        col_diff = diff[:, j].data
        expected = 0.1 * dense[:, j].std(ddof=1)
        assert col_diff.std() == pytest.approx(expected, rel=0.1)


def test_sparse_noise_ignores_explicit_zeros_and_ints():
    """Explicitly stored zeros stay zeros; integer input is cast to float."""
    X_sparse = sparse.csr_matrix(
        (np.array([0, 2, 3]), np.array([0, 1, 1]), np.array([0, 1, 2, 3])),
        shape=(3, 2),
    )
    assert X_sparse.nnz == 3  # one explicit zero
    sampler = DataOversampler(add_noise=True, noise_level=0.5, random_state=0)
    X_noisy = sampler._add_sparse_gaussian_noise(X_sparse)
    assert X_noisy.dtype == np.float64
    assert X_noisy[0, 0] == 0.0
    assert X_noisy.nnz == 2


def test_sparse_noise_empty_matrix():
    """A sparse matrix without non-zero elements is returned unchanged."""
    X_sparse = sparse.csr_matrix((4, 3))
    sampler = DataOversampler(add_noise=True)
    X_noisy = sampler._add_sparse_gaussian_noise(X_sparse)
    assert X_noisy.nnz == 0
    assert X_noisy.shape == (4, 3)


@pytest.mark.parametrize("algorithm", ["smote", "adasyn"])
def test_sparse_synthetic_noise_keeps_zeros(algorithm):
    """SMOTE/ADASYN densify a sparse input; with add_noise the zero cells
    stay exact zeros, as in the sparse path."""
    X_sparse = sparse.csr_matrix(
        [
            [1.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 3.5],
            [4.5, 1.5, 0.0],
            [0.0, 2.0, 1.2],
            [3.1, 0.0, 2.7],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    kwargs = dict(algorithm=algorithm, multiplier=1.5, random_state=42)
    X_clean, _ = DataOversampler(**kwargs).fit_resample(X_sparse, y)
    X_noisy, _ = DataOversampler(
        add_noise=True, noise_level=0.1, **kwargs
    ).fit_resample(X_sparse, y)
    assert isinstance(X_noisy, np.ndarray)
    X_clean = np.asarray(X_clean, dtype=float)
    X_noisy = np.asarray(X_noisy, dtype=float)
    np.testing.assert_array_equal(X_noisy[X_clean == 0], 0.0)
    assert np.any(X_noisy[X_clean != 0] != X_clean[X_clean != 0])


# ------------------------------------------------------------------ #
#  Sparse branch (output of hashing encoding)                          #
# ------------------------------------------------------------------ #


def test_sparse_random_oversampling_keeps_sparse():
    """Random oversampling on a csr_matrix stays sparse
    (pd.DataFrame(csr) is not built — it would be a silently garbage dataframe)."""
    X_sparse = sparse.csr_matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    sampler = DataOversampler(algorithm="random", multiplier=2.0, random_state=42)
    X_res, y_res = sampler.fit_resample(X_sparse, y)
    assert sparse.issparse(X_res)
    assert X_res.format == "csr"
    assert X_res.shape[0] == 12
    assert len(y_res) == 12
    # Values are binary, the dimensionality is preserved
    assert X_res.shape[1] == 3


def test_sparse_smote_densifies():
    """SMOTE on a sparse input (hashing) is converted to dense:
    synthetic algorithms require dense distance matrices."""
    X_sparse = sparse.csr_matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    X_res, y_res = sampler.fit_resample(X_sparse, y)
    assert isinstance(X_res, np.ndarray)
    # multiplier=1.5: ceil(3 * 1.5) = 5 per class -> 10 rows
    assert len(X_res) == 10
    assert len(y_res) == 10
    assert X_res.shape[1] == 3


def test_sparse_adasyn_densifies():
    """ADASYN on a sparse input is converted to dense and works."""
    X_sparse = sparse.csr_matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    sampler = DataOversampler(algorithm="adasyn", multiplier=1.5, random_state=42)
    X_res, y_res = sampler.fit_resample(X_sparse, y)
    assert isinstance(X_res, np.ndarray)
    assert len(X_res) > X_sparse.shape[0]
    assert X_res.shape[1] == 3


def test_sparse_unsupported_algorithm_raises():
    """Unknown algorithm on a sparse input -> ValueError."""
    X_sparse = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    sampler = DataOversampler(algorithm="magic")
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        sampler.fit_resample(X_sparse, y)


def test_smote_adasyn_require_non_missing_target():
    """SMOTE/ADASYN raise a clear error when there is no usable target."""
    X = np.arange(4.0).reshape(2, 2)
    y = np.array([])
    for algorithm in ("smote", "adasyn"):
        sampler = DataOversampler(algorithm=algorithm, multiplier=2.0)
        with pytest.raises(ValueError, match="require at least one non-missing"):
            sampler.fit_resample(X, y)


def test_decode_continuous_y_rejects_short_input():
    """A resampled vector shorter than the original target must raise."""
    y = pd.Series(np.linspace(0.0, 1.0, 64))
    binned, bin_mids = _prepare_continuous_bins(y)

    with pytest.raises(ValueError, match="shorter than the original"):
        _decode_continuous_y(binned.to_numpy()[:10], y, binned, bin_mids)


def test_sparse_random_continuous_target_keeps_sparse():
    """Random oversampling of a continuous target on a sparse input stays
    sparse and decodes the target back to real continuous values."""
    X_sparse = sparse.csr_matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    rng = np.random.RandomState(5)
    y = rng.randn(X_sparse.shape[0])
    sampler = DataOversampler(algorithm="random", multiplier=2.0, random_state=42)
    X_res, y_res = sampler.fit_resample(X_sparse, y)

    assert sparse.issparse(X_res)
    assert X_res.format == "csr"
    n_orig = len(y)
    assert X_res.shape[0] > n_orig
    # Original rows keep their exact values; the tail is bin midpoints.
    assert np.array_equal(y_res[:n_orig], y)
    assert np.isfinite(y_res[n_orig:]).all()


# 6. Coverage of line 259: Unsupported algorithm
def test_unsupported_algorithm():
    # Bypass the validator by calling directly or changing the attribute
    sampler = DataOversampler(algorithm="unknown_algo")
    df = pd.DataFrame({"A": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        sampler.fit_resample(df[["A"]], df["target"])


def test_restoration_of_boolean_and_unusual_types():
    """Check type restoration for data that is neither numeric nor categorical."""
    df = pd.DataFrame({"is_valid": [True, False, True, False], "target": [0, 0, 1, 1]})
    sampler = DataOversampler(algorithm="random", multiplier=1.2)
    result = sampler.oversample(df, target="target")
    # Check that the bool type survived the numpy conversions
    assert result["is_valid"].dtype == bool


def test_type_restoration_error_handling(caplog):
    """Check the system's resilience to errors during data type restoration."""
    sampler = DataOversampler()
    df = pd.DataFrame({"feature": [1, 2, 3]})
    # Pass an invalid series of types to provoke an exception in the handling block
    with caplog.at_level(logging.WARNING):
        sampler._restore_dtypes(df, pd.Series({"feature": "non_existent_type_object"}))
    assert "Failed to restore type for column" in caplog.text


def test_internal_resampling_mechanism_and_locks():
    """Test the protected resampling method and the locking mechanism."""
    sampler = DataOversampler(algorithm="random")
    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    labels = np.array([0, 0, 1, 1])
    # Check the internal API directly
    res_features, res_labels = sampler._fit_resample(features, labels)
    assert isinstance(res_features, np.ndarray)
    assert len(res_features) == len(res_labels)


def test_synthetic_algorithm_validation_without_numeric_data():
    """Check that SMOTE is protected from running on data without numeric features."""
    df = pd.DataFrame(
        {
            "category_a": ["high", "low", "high", "low"],
            "category_b": ["red", "blue", "red", "blue"],
            "target": [0, 0, 1, 1],
        }
    )
    sampler = DataOversampler(algorithm="smote")
    # We expect an error because SMOTE cannot work with only categorical features
    with pytest.raises(TypeError, match="requires at least one numeric feature"):
        sampler.oversample(df, target="target")


def test_exception_propagation_for_invalid_parameters():
    """Check correct exception propagation for invalid input parameters."""
    # A multiplier below 1 is invalid according to the library logic
    sampler = DataOversampler(multiplier=0.1)
    X = pd.DataFrame({"f1": [1, 2]})
    y = pd.Series([0, 1])
    with pytest.raises(ValueError, match="multiplier must be >= 1"):
        sampler._fit_resample(X, y)


def test_unsupported_algorithm_name_validation():
    """Check algorithm name validation in the public interface."""
    df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
    sampler = DataOversampler(algorithm="magic_boost")  # Non-existent algorithm
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        sampler.oversample(df, target="target")


def test_blind_oversampling_without_target_column():
    """Test 'blind' oversampling without specifying the target variable."""
    df = pd.DataFrame(
        {"feature_1": [0.1, 0.2, 0.3, 0.4], "feature_2": ["A", "B", "A", "B"]}
    )
    # Check operation in simple row duplication mode (Random)
    result = oversample(df, multiplier=2.0, algorithm="random", target=None)
    assert len(result) == 8
    # Make sure the temporary helper columns are removed from the result
    assert "temp_target" not in result.columns


# Test case for covering the balance=True branch
def test_strategy_balance_true_coverage(mocker):
    # Create a mock object of the class where _strategy is implemented
    # Simulate self.balance = True
    mock_self = mocker.Mock()
    mock_self.balance = True

    # Input data: class 0 (10 samples), class 1 (100 samples)
    y = pd.Series([0] * 10 + [1] * 100)
    multiplier = 1.5

    # Call the method (assume it is defined in the Sampler class)
    # If you test a specific class, replace Sampler with your class name

    # Call directly through the class, passing our mock_self
    result = DataOversampler._strategy(mock_self, y, multiplier)

    # Calculation: base_size = 100 (max of 10 and 100)
    # Result: ceil(100 * 1.5) = 150 for ALL classes
    expected = {0: 150, 1: 150}

    assert result == expected
    assert result[0] == result[1], "class sizes must be equal when balance=True"


def test_restore_dtypes_none_handling():
    """
    Test for covering the branch:
    if original_dtypes is None:
        return df
    """
    # 1. Create an instance of the class
    # Initialization parameters (add_noise, etc.) are not important for this test,
    # since we return from the method before they are used.
    sampler = DataOversampler()

    # 2. Prepare the test DataFrame
    test_df = pd.DataFrame(
        {"feature_1": [1.0, 2.5, 3.1], "feature_2": ["cat", "dog", "bird"]}
    )

    # 3. Call the target method with original_dtypes=None
    # This should trigger the early return from the function
    result = sampler._restore_dtypes(df=test_df, original_dtypes=None)

    # 4. Assertions
    # Verify that the DataFrame was returned unchanged
    pd.testing.assert_frame_equal(result, test_df)

    # Check object identity (in this implementation the same df should be returned)
    assert result is test_df, (
        "the method must return the original object when types are not provided"
    )


def test_restore_dtypes_missing_column_coverage():
    """
    Test for covering the lines:
    if col not in df.columns:
        continue
    """
    sampler = DataOversampler()

    # 1. Create a DataFrame that does NOT have the 'target' column
    df = pd.DataFrame({"feature_1": [1, 2, 3]})

    # 2. Create a Series with dtypes where 'target' IS PRESENT
    # (simulating a situation where the original dtypes were measured for more columns)
    original_dtypes = pd.Series(
        {
            "feature_1": np.dtype("int64"),
            "target": np.dtype("int64"),  # This column is not in df
        }
    )

    # 3. Call the method
    # The program should:
    # - process 'feature_1'
    # - notice that 'target' is not in df.columns
    # - execute 'continue' (move to the next iteration or exit the loop)
    result = sampler._restore_dtypes(df=df, original_dtypes=original_dtypes)

    # 4. Assertions
    assert "feature_1" in result.columns
    assert "target" not in result.columns
    assert len(result.columns) == 1
    # Make sure the method completed successfully and did not fail with KeyError
    assert isinstance(result, pd.DataFrame)


def test_restore_dtypes_categorical_coverage():
    """
    Test for covering the lines:
    if isinstance(dtype, pd.CategoricalDtype):
        df[col] = df[col].astype(dtype)
    """
    sampler = DataOversampler()

    # 1. Define a strict categorical dtype with a given order
    cat_type = pd.CategoricalDtype(
        categories=["small", "medium", "large"], ordered=True
    )

    # 2. Simulate a DataFrame after oversampling
    # (generation libraries often degrade the dtypes to object or int)
    df = pd.DataFrame(
        {
            "size": ["small", "large", "medium"]  # Currently of dtype object
        }
    )

    # 3. Set the original dtypes where the 'size' column was categorical
    original_dtypes = pd.Series({"size": cat_type})

    # Check the initial state (that it is not a category)
    assert not isinstance(df["size"].dtype, pd.CategoricalDtype)

    # 4. Call dtype restoration
    result = sampler._restore_dtypes(df=df, original_dtypes=original_dtypes)

    # 5. Assertions
    # Verify that the dtype became categorical
    assert isinstance(result["size"].dtype, pd.CategoricalDtype)

    # Verify that the categories were restored correctly
    assert list(result["size"].dtype.categories) == ["small", "medium", "large"]
    assert result["size"].dtype.ordered is True

    # Make sure the data was not corrupted
    assert result["size"].iloc[1] == "large"


def test_restore_dtypes_object_coverage():
    """
    Test for covering the lines:
    elif dtype == object:
        df[col] = df[col].astype(object)
    """
    sampler = DataOversampler()

    # 1. Create a DataFrame where the 'description' column has a dtype
    # different from plain object (e.g., simulating a generation result)
    # In Pandas 3.0 this might be StringDtype; here we simply create a Series
    df = pd.DataFrame({"description": ["item1", "item2", "item3"]})

    # 2. Specify that the original dtype was strictly object
    original_dtypes = pd.Series(
        {
            "description": np.dtype("O")  # This is the object dtype
        }
    )

    # 3. Call the method
    result = sampler._restore_dtypes(df=df, original_dtypes=original_dtypes)

    # 4. Assertions
    # Verify that the resulting dtype is object
    assert result["description"].dtype == object

    # Make sure the values were preserved
    assert result["description"].iloc[0] == "item1"


def test_fit_resample_smotenc_via_smote_coverage():
    """
    Covers the block:
    if algo_local == "smote":
        if cat_features_idx:
            sampler = SMOTENC(...)
    """
    # 1. Initialize with the SMOTE algorithm
    sampler = DataOversampler(algorithm="SMOTE", multiplier=1.5, random_state=42)

    # 2. Create data with a categorical column (object)
    X = pd.DataFrame(
        {
            "numeric_feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "categorical_feat": ["A", "A", "B", "B", "A", "B"],  # Non-numeric column
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1])  # Imbalance: class 1 has only 2 samples

    # 3. Run resampling
    # Internally cat_features_idx will become [1], and SMOTENC will be used
    X_res, y_res = sampler._fit_resample(X, y)

    # 4. Assertions
    assert X_res.shape[0] > X.shape[0]
    assert X_res.shape[1] == 2
    # Verify that the categorical column stayed correct (did not become NaN)
    assert "A" in X_res[:, 1] or "B" in X_res[:, 1]


def test_fit_resample_adasyn_to_smotenc_fallback_coverage(caplog):
    """
    Test for covering the branches:
    elif algo_local == "adasyn":
        if cat_features_idx:
            logger.warning(...)
            sampler = SMOTENC(...)
    """
    # 1. Initialize the oversampler with the ADASYN algorithm
    sampler = DataOversampler(algorithm="ADASYN", multiplier=2.0, random_state=42)

    # 2. Create data with at least one categorical column (object/string dtype)
    # Important: vanilla ADASYN would fail on such data,
    # but our code must fall back to SMOTENC.
    X = pd.DataFrame(
        {
            "feature_numeric": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            "feature_cat": ["type_A", "type_B", "type_A", "type_B", "type_A", "type_B"],
        }
    )

    # Create an imbalanced target (at least 2 samples for the minority class)
    y = np.array([0, 0, 0, 0, 1, 1])
    sampler = DataOversampler(
        algorithm="ADASYN",
        multiplier=2.0,
        task="classification",
        random_state=42,
    )

    # 3. Call the method. Use caplog to check the logs.
    with caplog.at_level("WARNING"):
        X_res, y_res = sampler._fit_resample(X, y)

    # 4. Assertions

    # Check log coverage: look for the message text in the logs
    assert any(
        "ADASYN does not support categorical columns" in record.message
        for record in caplog.records
    ), "a warning about the fallback to SMOTENC must be emitted"

    # Check the result: the data must be synthesized
    assert len(y_res) > len(y), (
        "the number of samples must increase after oversampling"
    )
    assert X_res.shape[1] == 2, "the number of features must be preserved"

    # Check data integrity: the categorical column must not be empty or corrupted
    # (we check for the presence of strings in the resulting numpy array)
    unique_cats = np.unique(X_res[:, 1])
    assert "type_A" in unique_cats or "type_B" in unique_cats


def test_restore_dtypes_noise_numeric_pass():
    """
    The test covers the 'pass' branch when:
    1. add_noise = True
    2. The original dtype was object
    3. The current dtype in df is numeric
    """
    # 1. Initialize the oversampler with noise enabled
    oversampler = DataOversampler(add_noise=True)
    # 2. Create the original dtypes (the column once was string/object)
    original_dtypes = pd.Series(
        {
            "feature_col": np.dtype("O")  # object dtype
        }
    )
    # 3. Create a DataFrame where this column BECAME numeric
    # (for example, during data generation or noise addition)
    df = pd.DataFrame(
        {
            "feature_col": [1.1, 2.5, 3.8]  # Numeric values (float64)
        }
    )
    # Confirm the initial state before calling the method
    assert is_numeric_dtype(df["feature_col"])

    # 4. Call the method
    result_df = oversampler._restore_dtypes(df, original_dtypes)
    # 5. Assertions:
    # The dtype must STAY numeric (float64), because the 'pass' branch ran
    # If the 'else' branch had run, the dtype would become 'object'
    assert is_numeric_dtype(result_df["feature_col"])
    assert result_df["feature_col"].dtype != object
    assert result_df["feature_col"].iloc[0] == 1.1


def test_fit_resample_smote_requires_numeric_feature():
    """
    Coverage of lines 332-334: check the TypeError in _fit_resample
    when calling SMOTE on data without numeric features.
    """
    sampler = DataOversampler(algorithm="smote", multiplier=1.5)
    X = pd.DataFrame(
        {
            "cat1": ["a", "b", "c", "d", "e", "f"],
            "cat2": ["x", "y", "x", "y", "x", "y"],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1])

    with pytest.raises(TypeError, match="SMOTE requires at least one numeric feature"):
        sampler._fit_resample(X, y)


def test_fit_resample_adasyn_requires_numeric_feature():
    """
    Coverage of lines 332-334: check the TypeError in _fit_resample
    when calling ADASYN on data without numeric features.
    """
    sampler = DataOversampler(algorithm="adasyn", multiplier=1.5)
    X = pd.DataFrame(
        {
            "cat1": ["a", "b", "c", "d", "e", "f"],
            "cat2": ["x", "y", "x", "y", "x", "y"],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1])

    with pytest.raises(TypeError, match="ADASYN requires at least one numeric feature"):
        sampler._fit_resample(X, y)
    """
    Verify that with Random + add_noise=True categorical columns are not
    corrupted by noise: noise is excluded from the ordinal-encoded columns,
    and they are then correctly decoded back into the original categories.
    """
    df = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "category": ["low", "medium", "high", "low", "high"],
            "target": [0, 0, 0, 1, 1],
        }
    )
    sampler = DataOversampler(
        multiplier=2,
        algorithm="random",
        add_noise=True,
        noise_level=0.5,
        random_state=42,
    )
    result = sampler.oversample(df, target="target")

    # Check 1: The size increased (5 * 2 = 10)
    assert len(result) == 10
    # Check 2: The categorical column contains ONLY the original labels
    valid_categories = {"low", "medium", "high"}
    assert set(result["category"].unique()) == valid_categories, (
        "noise must not create new values in the categorical column"
    )
    # Check 3: The numeric column changed (noise was applied)
    assert not np.array_equal(
        result["numeric"].values[:5], df["numeric"].values
    ), "noise must change the numeric values"
    # Check 4: All values are not NaN
    assert result.notna().all().all()


# ------------------------------------------------------------------ #
#  Categorical column detection (numeric-strings edge cases)      #
# ------------------------------------------------------------------ #


def test_numeric_strings_treated_as_numeric():
    """'Numeric strings' ('10','20') are treated as numeric, not categorical."""
    assert _is_categorical_col(pd.Series(["10", "20", "30"])) is False


def test_mixed_strings_treated_as_categorical():
    """Mixed values (numbers + letters) — a categorical column."""
    assert _is_categorical_col(pd.Series(["10", "a", "20"])) is True


def test_true_category_treated_as_categorical():
    """Non-numeric categories remain categorical."""
    assert _is_categorical_col(pd.Series(["red", "green", "blue"])) is True


def test_numeric_dtype_treated_as_numeric():
    """A numeric dtype is not categorical."""
    assert _is_categorical_col(pd.Series([1.0, 2.0, 3.0])) is False


def test_smote_guard_not_triggered_for_numeric_strings():
    """
    Oversampling with a column of 'numeric strings' must not
    mistakenly treat it as categorical and raise a TypeError
    "requires at least one numeric feature" (consistency between the guard check
    in oversample() and the detection in _fit_resample).
    """
    df = pd.DataFrame(
        {
            "f1": ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"],
            "target": [0, 0, 0, 0, 1, 1],
        }
    )
    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    result = sampler.oversample(df, target="target")
    assert len(result) > len(df)


# ------------------------------------------------------------------ #
#  Continuous regression targets (quantile-binned oversampling)       #
# ------------------------------------------------------------------ #


def test_is_continuous_target_detection():
    """Any numeric (binnable) target is regression data; non-numeric are not.

    The engine is regression-only: integer targets (counts, whole-unit
    prices) are regression targets just like truly continuous floats. Only
    non-numeric / bool / NaN-only targets are not regression data.
    """
    rng = np.random.RandomState(0)
    # True continuous (non-integer floats)
    assert _is_continuous_target(pd.Series(rng.randn(200))) is True
    assert _is_continuous_target(pd.Series([0.5, 1.5, 2.5, 0.5, 1.5])) is True
    # Two unique non-integer floats are also regression for the engine
    assert _is_continuous_target(pd.Series([2.5, 5.5, 2.5, 5.5])) is True
    # Integer dtypes and integer-valued floats are regression too (the
    # engine never treats a numeric target as a class label).
    assert _is_continuous_target(pd.Series(np.arange(100) % 40)) is True
    assert _is_continuous_target(pd.Series([0, 1, 0, 1])) is True
    assert _is_continuous_target(pd.Series([0.0, 1.0, 0.0, 1.0])) is True
    assert _is_continuous_target(pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])) is True
    # Bool targets are not regression data
    assert _is_continuous_target(pd.Series([True, False, True, False])) is False
    # Non-numeric targets are not regression data
    assert _is_continuous_target(pd.Series(["a", "b", "c", "a"])) is False
    # NaN-only / empty targets are not regression data
    assert _is_continuous_target(pd.Series([np.nan, np.nan])) is False
    assert _is_continuous_target(pd.Series([], dtype=float)) is False


def test_prepare_continuous_bins_raises_on_non_discretizable():
    """_prepare_continuous_bins raises a ValueError when the target cannot
    be split into the required number of quantile bins."""
    with pytest.raises(ValueError, match="quantile bins is required"):
        _prepare_continuous_bins(pd.Series([2.5] * 5))


def test_prepare_continuous_bins_structure():
    """Bins cover all finite rows; NaN rows are marked with code -1."""
    y = pd.Series([1.0, 2.5, 4.0, 8.0, 16.0, np.nan, 32.0, 64.0, 128.0])
    binned, bin_mids = _prepare_continuous_bins(y)

    assert len(binned) == len(y)
    assert int(binned.iloc[5]) == -1  # NaN row -> -1
    finite_codes = binned[binned != -1]
    assert set(finite_codes.unique()) == set(bin_mids.keys())
    assert set(finite_codes.unique()) == set(range(len(bin_mids)))
    # Midpoints are finite and (weakly) ordered like the bins
    mids = [bin_mids[k] for k in sorted(bin_mids)]
    assert all(np.isfinite(m) for m in mids)
    assert mids == sorted(mids)


def test_decode_continuous_y_keeps_originals_and_synthesizes():
    """Original rows keep their real values; synthetic rows get bin mids."""
    y = pd.Series(np.linspace(0.0, 1.0, 64))
    binned, bin_mids = _prepare_continuous_bins(y)
    n = len(y)

    # Simulate a resampled bin vector: originals + 3 synthetic for bin 0 and
    # 2 synthetic for bin 1 (synthetic rows always come last).
    y_binned = np.concatenate(
        [binned.to_numpy(), np.array([0, 0, 0, 1, 1], dtype=binned.dtype)]
    )
    decoded = _decode_continuous_y(y_binned, y, binned, bin_mids)

    assert len(decoded) == n + 5
    assert np.array_equal(decoded[:n], y.to_numpy())
    assert np.allclose(decoded[n:], [bin_mids[0]] * 3 + [bin_mids[1]] * 2)


@pytest.mark.parametrize("algorithm", ["random", "smote", "adasyn"])
def test_oversample_continuous_target_all_algorithms(algorithm):
    """The three algorithms oversample a continuous target without crashing
    and preserve the original target values."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {"f1": rng.randn(120), "f2": rng.randn(120), "f3": rng.randn(120)}
    )
    df["target"] = df["f1"] + 0.5 * df["f2"] + rng.randn(120) * 0.2

    sampler = DataOversampler(algorithm=algorithm, multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")

    n_orig = len(df)
    assert len(res) > n_orig
    assert np.array_equal(res["target"].iloc[:n_orig].to_numpy(), df["target"].to_numpy())
    synth = res["target"].iloc[n_orig:]
    assert np.isfinite(synth).all()
    assert synth.between(df["target"].min(), df["target"].max()).all()


def test_oversample_continuous_target_with_nan():
    """NaN targets pass through unchanged; no synthetic NaN is produced."""
    rng = np.random.RandomState(7)
    df = pd.DataFrame({"f1": rng.randn(100)})
    df["target"] = rng.randn(100)
    df.loc[df.index[:10], "target"] = np.nan

    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")

    n_orig = len(df)
    assert len(res) > n_orig
    assert res["target"].iloc[:n_orig].isna().sum() == 10
    assert res["target"].iloc[n_orig:].notna().all()


def test_binary_float_target_takes_regression_path():
    """Binary float 0.0/1.0 targets take the regression path (the engine is
    regression-only): the result contains the real class-like values, and
    synthetic rows may interpolate between them."""
    rng = np.random.RandomState(5)
    df = pd.DataFrame({"f1": rng.randn(80), "f2": rng.randn(80)})
    df["target"] = np.where(rng.rand(80) > 0.5, 1.0, 0.0)

    sampler = DataOversampler(algorithm="smote", multiplier=2.0, random_state=42)
    res = sampler.oversample(df, target="target")

    assert len(res) > len(df)
    # Original rows are preserved exactly at the front.
    assert np.array_equal(
        res["target"].iloc[: len(df)].to_numpy(), df["target"].to_numpy()
    )
    # The regression path keeps the values in the observed range.
    assert res["target"].between(0.0, 1.0).all()


@pytest.mark.parametrize("algorithm", ["smote", "adasyn"])
def test_continuous_target_singleton_bin_pass_through(algorithm):
    """A quantile bin with a single row cannot be synthesized (SMOTE/ADASYN
    need k_neighbors); such bins are passed through instead of crashing.

    Regression: the previous implementation computed k_neighbors from the
    global minimum class count, so a singleton bin crashed with
    ``n_neighbors <= n_samples_fit``. The buggy test did not actually
    produce a singleton bin; this dataset reproduces the reviewer's case
    (bin counts {0: 20, 1: 1, 2: 3, 3: 4}).
    """
    # Truly continuous target with a singleton quantile bin (99.0 outlier).
    target = [0.0] * 10 + [1.5] * 10 + [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 99.0]
    rng = np.random.RandomState(3)
    df = pd.DataFrame({"f1": rng.randn(len(target)), "f2": rng.randn(len(target))})
    df["target"] = target

    sampler = DataOversampler(algorithm=algorithm, multiplier=2.0, random_state=42)
    res = sampler.oversample(df, target="target")

    n_orig = len(df)
    assert len(res) >= n_orig
    # Original rows keep their exact values (including the singleton row).
    assert np.array_equal(res["target"].iloc[:n_orig].to_numpy(), np.asarray(target))
    # The synthetic tail is finite and within the observed range.
    assert res["target"].iloc[n_orig:].notna().all()
    assert res["target"].iloc[n_orig:].between(min(target), max(target)).all()


def test_continuous_target_all_singleton_bins_pass_through():
    """When every quantile bin is a singleton nothing can be synthesized:
    SMOTE/ADASYN must keep the data unchanged instead of crashing."""
    # 8 distinct continuous values -> qcut produces 8 singleton bins.
    target = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"f1": rng.randn(len(target)), "f2": rng.randn(len(target))})
    df["target"] = target

    for algorithm in ("smote", "adasyn"):
        sampler = DataOversampler(
            algorithm=algorithm, multiplier=2.0, random_state=42
        )
        res = sampler.oversample(df, target="target")
        assert len(res) == len(df)
        assert np.allclose(
            res["target"].to_numpy(), np.asarray(target, dtype=float), equal_nan=True
        )


def test_continuous_target_singleton_bin_with_nan_pass_through():
    """Singleton bins and NaN targets together: nothing crashes, NaN rows are
    passed through, the singleton row keeps its value."""
    target = [0.0] * 10 + [1.5] * 10 + [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 99.0]
    target[12] = np.nan  # NaN inside the 1.5 block; bins: {0:10, 1:9, 2:1, 3:3, 4:4}
    rng = np.random.RandomState(4)
    df = pd.DataFrame({"f1": rng.randn(len(target)), "f2": rng.randn(len(target))})
    df["target"] = target

    for algorithm in ("smote", "adasyn"):
        sampler = DataOversampler(
            algorithm=algorithm, multiplier=2.0, random_state=42
        )
        res = sampler.oversample(df, target="target")
        n_orig = len(df)
        assert len(res) >= n_orig
        assert res["target"].iloc[:n_orig].isna().sum() == 1
        # All original rows keep their exact values (NaN at the same place).
        assert np.allclose(
            res["target"].iloc[:n_orig].to_numpy(),
            np.asarray(target, dtype=float),
            equal_nan=True,
        )
        assert res["target"].iloc[n_orig:].notna().all()


def test_continuous_target_balance_true_with_nan():
    """balance=True with NaN targets does not crash and NaN rows are passed
    through unchanged."""
    target = [1.0, 2.5, 4.0, 8.0, 16.0, 32.0] + [np.nan] * 2
    target = target * 3
    rng = np.random.RandomState(9)
    df = pd.DataFrame({"f1": rng.randn(len(target)), "f2": rng.randn(len(target))})
    df["target"] = target

    for algorithm in ("smote", "adasyn", "random"):
        sampler = DataOversampler(
            algorithm=algorithm, multiplier=1.5, balance=True, random_state=42
        )
        res = sampler.oversample(df, target="target")
        n_orig = len(df)
        assert len(res) >= n_orig
        assert res["target"].iloc[:n_orig].isna().sum() == 6
        assert res["target"].iloc[n_orig:].notna().all()


def test_decode_continuous_y_rejects_unknown_synth_code():
    """A synthetic row with a bin code absent from bin_mids must raise a
    KeyError instead of silently substituting 0.0."""
    y = pd.Series(np.linspace(0.0, 1.0, 64))
    binned, bin_mids = _prepare_continuous_bins(y)
    n = len(y)

    # A valid int32 code that is NOT a bin of this target.
    valid_int32_code = 1_000_000
    y_binned = np.concatenate(
        [binned.to_numpy(), np.array([valid_int32_code], dtype=np.int32)]
    )

    with pytest.raises(KeyError):
        _decode_continuous_y(y_binned, y, binned, bin_mids)


def test_decode_continuous_y_rejects_unexpected_row_order():
    """If the resampled vector does not start with the original rows, the
    decoding must fail loudly instead of silently corrupting the targets."""
    y = pd.Series(np.linspace(0.0, 1.0, 64))
    binned, bin_mids = _prepare_continuous_bins(y)

    # Rotate the vector: the original codes are no longer the prefix.
    rotated = np.roll(binned.to_numpy(), 1)
    with pytest.raises(ValueError, match="does not start with the original rows"):
        _decode_continuous_y(rotated, y, binned, bin_mids)


def test_bin_continuous_target_single_pass_consistency():
    """The combined detection+binning helper agrees with the standalone
    helpers and produces identical codes/mids."""
    y = pd.Series(np.linspace(0.0, 1.0, 200) ** 2)
    is_cont, binned, bin_mids = _bin_continuous_target(y)

    assert is_cont == _is_continuous_target(y)
    assert is_cont is True
    binned2, bin_mids2 = _prepare_continuous_bins(y)
    assert bin_mids == bin_mids2
    assert np.array_equal(binned.to_numpy(), binned2.to_numpy())


# ------------------------------------------------------------------ #
#  Regression oversampling for ANY numeric target                    #
# ------------------------------------------------------------------ #


def _diabetes_df() -> pd.DataFrame:
    """The sklearn diabetes dataset (a float target with many unique values)."""
    from sklearn.datasets import load_diabetes

    return load_diabetes(as_frame=True).frame


def _count_target_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A count (integer-valued Poisson) regression target."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    df["target"] = rng.poisson(3, n).astype(int)
    return df


@pytest.mark.parametrize("algorithm", ["random", "smote", "adasyn"])
def test_integer_target_uses_regression_path(algorithm):
    """Integer-valued regression targets (counts) are NOT treated as classes:
    every unique value must not become a 'class' (which broke SMOTE/ADASYN
    with singleton classes and multiplied random by per-class ceil)."""
    df = _count_target_df()
    sampler = DataOversampler(algorithm=algorithm, multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")
    n = len(df)
    assert len(res) >= n
    # Row count: ceil(n * multiplier) ± number of bins.
    expected = ceil(n * 1.5)
    n_bins = 6  # qcut(duplicates='drop') of the Poisson target
    assert abs(len(res) - expected) <= n_bins, (len(res), expected)
    # Original rows keep their exact integer values.
    assert np.array_equal(res["target"].iloc[:n].to_numpy(), df["target"].to_numpy())


@pytest.mark.parametrize("algorithm", ["random", "smote", "adasyn"])
def test_diabetes_target_oversamples(algorithm):
    """The diabetes target (214 unique float values) must take the regression
    path: SMOTE/ADASYN succeed (previously: singleton-class crash) and the
    row count matches ceil(n * multiplier) ± number of bins."""
    df = _diabetes_df()
    sampler = DataOversampler(algorithm=algorithm, multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")
    n = len(df)
    expected = ceil(n * 1.5)
    assert abs(len(res) - expected) <= 8, (len(res), expected)
    assert np.array_equal(res["target"].iloc[:n].to_numpy(), df["target"].to_numpy())


def test_random_oversampling_restores_real_targets():
    """Random oversampling: EVERY synthetic row's y equals the y of its
    source row (RandomOverSampler.sample_indices_), not a bin midpoint."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"f1": rng.randn(200), "f2": rng.randn(200)})
    df["target"] = df["f1"] * 3.0 + 1.0 + rng.randn(200) * 0.1

    sampler = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    X = df.drop("target", axis=1)
    y = df["target"].to_numpy()
    X_res, y_res = sampler.fit_resample(X, y)
    n = len(y)
    assert len(y_res) > n
    # sample_indices_ must be used: every output row maps to a real source
    # row, so the synthetic tail must contain only values from the original
    # target (never bin midpoints / interpolated values).
    synth_vals = set(y_res[n:])
    orig_vals = set(y)
    assert synth_vals <= orig_vals
    assert len(synth_vals) > 1
    # With 200 rows and multiplier=1.5 the old midpoint decode produced only
    # 8 distinct synthetic values; the row-level restore must produce many
    # more (the number of distinct synthetic y equals the number of
    # *distinct source rows that were duplicated*).
    assert len(synth_vals) > 50


def test_smote_synthetic_y_between_parents():
    """SMOTE-style regression generation: synthetic y values lie between the
    two parents' values (interpolation), and the number of distinct synthetic
    y ≈ number of synthetic rows (honest targets, not bin midpoints)."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"f1": rng.randn(300), "f2": rng.randn(300)})
    df["target"] = df["f1"] + 0.5 * df["f2"] + rng.randn(300) * 0.1

    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    X = df.drop("target", axis=1)
    y = df["target"].to_numpy()
    X_res, y_res = sampler.fit_resample(X, y)
    n = len(y)
    synth = y_res[n:]
    assert len(synth) > 0
    # Every synthetic value is an interpolation, so it stays inside the
    # observed target range.
    assert synth.min() >= y.min() and synth.max() <= y.max()
    # Distinct synthetic values ≈ number of synthetic rows (unlike the 8
    # bin midpoints of the previous implementation).
    n_distinct = len(np.unique(np.round(synth, 8)))
    assert n_distinct >= 0.8 * len(synth)


def test_smote_synthetic_parents_real():
    """SMOTE-style generation interpolates between REAL rows: the synthetic
    row x must lie in the convex hull of the parents. With identical
    duplicated features (a single cluster) every synthetic target must be
    between the min and max of the real rows whose features match."""
    rng = np.random.RandomState(3)
    y_real = np.linspace(10.0, 20.0, 200)
    df = pd.DataFrame({"f1": np.zeros(200), "f2": np.zeros(200)})
    df["target"] = y_real
    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=1)
    X = df.drop("target", axis=1)
    X_res, y_res = sampler.fit_resample(X, y_real)
    n = len(y_real)
    synth = y_res[n:]
    assert len(synth) > 0
    # x is a convex combination of two real rows: with constant features the
    # synthetic x stays 0 (the fractional change is allowed).
    assert np.allclose(X_res[n:], 0.0)
    # y is the same convex combination of two real targets.
    assert synth.min() >= y_real.min() and synth.max() <= y_real.max()


def test_task_classification_keeps_classification_path():
    """An explicit task='classification' call keeps the class-based path: a
    binary 0.0/1.0 target yields only the original classes."""
    rng = np.random.RandomState(5)
    df = pd.DataFrame({"f1": rng.randn(80), "f2": rng.randn(80)})
    df["target"] = np.where(rng.rand(80) > 0.5, 1.0, 0.0)

    for algo in ("smote", "adasyn", "random"):
        sampler = DataOversampler(
            algorithm=algo, multiplier=2.0, task="classification", random_state=42
        )
        res = sampler.oversample(df, target="target")
        assert len(res) > len(df)
        assert set(res["target"].unique()) == {0.0, 1.0}


def test_default_task_is_regression():
    """The default task='regression' bins an integer target instead of
    treating every unique value as a class."""
    df = _count_target_df()
    sampler = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    X = df.drop("target", axis=1)
    X_res, y_res = sampler.fit_resample(X, df["target"].to_numpy())
    assert len(y_res) > len(df)


def test_target_bins_parameter_controls_bin_count():
    """target_bins controls the quantile discretization used for the
    sampling strategy."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"f1": rng.randn(400), "f2": rng.randn(400)})
    df["target"] = rng.randn(400)
    y = df["target"]

    is_reg4, binned4, mids4 = _bin_regression_target(pd.Series(y), 4)
    is_reg16, binned16, mids16 = _bin_regression_target(pd.Series(y), 16)
    assert is_reg4 and is_reg16
    assert len(mids4) == 4
    assert len(mids16) == 16


def test_target_bins_less_than_two_raises():
    """target_bins < 2 is a configuration error."""
    sampler = DataOversampler(target_bins=1)
    df = pd.DataFrame({"f1": [1.0, 2.0], "target": [0.5, 1.5]})
    with pytest.raises(ValueError, match="target_bins"):
        sampler.fit_resample(df[["f1"]], df["target"])


def test_experimental_oversampling_warns_when_enabled(caplog, monkeypatch):
    """When the evidence gate marks oversampling experimental, running the
    oversampler with a real sample increase logs a WARNING."""
    import configurable_automl_engine.oversampling.experimental as os_mod

    monkeypatch.setattr(os_mod, "OVERSAMPLING_EXPERIMENTAL", True)
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [0.0, 1.0, 2.0, 3.0]})
    sampler = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        sampler.oversample(df, target="target")
    assert "EXPERIMENTAL" in caplog.text
    assert "degradation" in caplog.text


def test_non_experimental_oversampling_does_not_warn(caplog, monkeypatch):
    """When the evidence gate passes (flag False), no experimental warning
    is emitted."""
    import configurable_automl_engine.oversampling.experimental as os_mod

    monkeypatch.setattr(os_mod, "OVERSAMPLING_EXPERIMENTAL", False)
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [0.0, 1.0, 2.0, 3.0]})
    sampler = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        sampler.oversample(df, target="target")
    assert "EXPERIMENTAL" not in caplog.text


def test_experimental_warns_on_balance_even_with_unit_multiplier(caplog, monkeypatch):
    """N3: balancing generates rows even at multiplier == 1.0, so the
    experimental WARNING must be emitted for balance=True as well."""
    import configurable_automl_engine.oversampling.experimental as os_mod

    monkeypatch.setattr(os_mod, "OVERSAMPLING_EXPERIMENTAL", True)
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"f1": rng.randn(40), "f2": rng.randn(40)})
    # An imbalanced binned target: balancing will generate rows.
    df["target"] = np.concatenate([np.full(10, 0.0), np.full(30, 1.0)])
    sampler = DataOversampler(algorithm="random", multiplier=1.0, balance=True, random_state=42)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        res = sampler.oversample(df, target="target")
    assert len(res) > len(df), "balancing must generate additional rows"
    assert "EXPERIMENTAL" in caplog.text


def test_infinite_target_values_stay_classification():
    """A numeric target with an infinite value cannot be binned: it falls
    back to the classification path instead of crashing."""
    y = pd.Series([0.0, 1.0, 2.0, np.inf])
    is_reg, binned, mids = _bin_regression_target(y, 8)
    assert is_reg is False
    assert mids == {}


def test_decode_regression_y_falls_back_to_legacy_decode(caplog):
    """RandomOverSampler without sample_indices_ falls back to the
    quantile-bin decode (defensive path for future imblearn versions)."""
    y = pd.Series([0.5, 1.5, 2.5, 3.5])
    sampler = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    y_res = np.array([0, 1, 2, 3, 0, 1])  # bin codes: originals + 2 synthetic
    fake_sampler = type("Fake", (), {"sample_indices_": None})()
    decoded = sampler._decode_regression_y(
        y_res, y, pd.Series([0, 1, 2, 3]),
        {0: 0.75, 1: 1.75, 2: 2.75, 3: 3.75}, fake_sampler,
    )
    assert len(decoded) == 6
    assert np.allclose(decoded[:4], y.to_numpy())
    assert np.allclose(decoded[4:], [0.75, 1.75])


def test_smote_requires_non_missing_target_all_nan():
    """SMOTE/ADASYN with an all-NaN target raise a clear error."""
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]})
    for algorithm in ("smote", "adasyn"):
        sampler = DataOversampler(algorithm=algorithm, multiplier=2.0, random_state=42)
        with pytest.raises(ValueError):
            sampler.fit_resample(df[["f1"]], pd.Series([np.nan] * 4))


# ------------------------------------------------------------------ #
#  Review follow-up (B1/B2): dtype honesty and ADASYN != SMOTE        #
# ------------------------------------------------------------------ #


def test_smote_integer_target_interpolation_not_rounded():
    """B1: the public oversample() must return the honest SMOTER-interpolated
    (float) values for an integer regression target instead of casting them
    back to int (which collapsed 152 synthetic rows into a handful of
    distinct values)."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"f1": rng.randn(300), "f2": rng.randn(300)})
    df["target"] = rng.poisson(3, 300).astype(int)

    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")
    n = len(df)
    synth = res["target"].iloc[n:]
    # Result is float: the interpolation is not rounded away.
    assert np.issubdtype(res["target"].dtype, np.floating)
    # The synthetic values are actual interpolations, not just the observed
    # integers: at least one non-integer-representable value must exist
    # (otherwise the interpolation was still destroyed).
    non_integer = (synth % 1 != 0).sum()
    assert non_integer > 0, "the SMOTER interpolation was rounded away"
    # Original rows keep their exact integer values.
    assert np.array_equal(res["target"].iloc[:n].to_numpy(), df["target"].to_numpy())


def test_smote_nullable_int64_target_with_nan():
    """B1: a nullable Int64 target with NaN must not crash the public
    oversample() (previously astype(target_dtype) raised
    'cannot safely cast non-equivalent float64 to int64')."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"f1": rng.randn(200), "f2": rng.randn(200)})
    df["target"] = pd.Series(rng.poisson(3, 200)).astype("Int64")
    df.loc[df.index[:10], "target"] = pd.NA

    sampler = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")
    n = len(df)
    assert np.issubdtype(res["target"].dtype, np.floating)
    # The original NaN rows are preserved (10), no new NaN appeared.
    assert res["target"].iloc[:n].isna().sum() == 10
    assert res["target"].iloc[n:].notna().all()


def test_random_integer_target_values_stay_exact():
    """B1: random oversampling of an integer target keeps the EXACT source
    values (the float64 result contains only the original integers)."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"f1": rng.randn(200), "f2": rng.randn(200)})
    df["target"] = rng.poisson(4, 200).astype(int)

    sampler = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")
    n = len(df)
    assert np.issubdtype(res["target"].dtype, np.floating)
    orig = set(df["target"].astype(float))
    assert set(res["target"].iloc[n:]) <= orig


def test_adasyn_differs_from_smote_regression_path():
    """B2: in the regression path ADASYN must NOT be a byte-identical alias
    of SMOTE: density-weighted parent selection produces a different
    synthetic set."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"f1": rng.randn(400), "f2": rng.randn(400)})
    df["target"] = df["f1"] + 0.5 * df["f2"] + rng.randn(400) * 0.1

    X = df.drop("target", axis=1)
    y = df["target"].to_numpy()
    smote = DataOversampler(algorithm="smote", multiplier=1.5, random_state=42)
    adasyn = DataOversampler(algorithm="adasyn", multiplier=1.5, random_state=42)
    Xr_s, yr_s = smote.fit_resample(X, y)
    Xr_a, yr_a = adasyn.fit_resample(X, y)

    assert len(yr_s) == len(yr_a)
    assert not np.array_equal(Xr_s, Xr_a) or not np.array_equal(yr_s, yr_a), (
        "ADASYN must differ from SMOTE in the regression path"
    )


def test_adasyn_zero_cross_bin_density_no_unbound_local():
    """B3: ADASYN with well-separated feature clusters (every k-NN neighbor
    belongs to the same bin -> zero cross-bin density for all parents) must
    not crash with UnboundLocalError: the density branch falls back to the
    uniform parent selection."""
    rng = np.random.RandomState(0)
    n_per_cluster = 30
    n_clusters = 8
    centers = np.arange(n_clusters)[:, None] * 100.0
    X = np.vstack(
        [centers[i] + rng.randn(n_per_cluster, 2) * 0.01 for i in range(n_clusters)]
    )
    y = np.repeat(np.linspace(0.0, 100.0, n_clusters), n_per_cluster)
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["target"] = y

    sampler = DataOversampler(algorithm="adasyn", multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")
    assert len(res) > len(df)


def test_string_target_with_default_task_preserved():
    """B4: a string (non-numeric) target on the default regression task goes
    through the classification fallback path and keeps its original dtype
    (previously the unconditional float64 cast crashed with
    'could not convert string to float')."""
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "f2": [2.0, 3.0, 1.0, 5.0, 4.0, 6.0, 8.0, 7.0],
            "target": ["cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog"],
        }
    )
    sampler = DataOversampler(algorithm="random", multiplier=1.5, random_state=42)
    res = sampler.oversample(df, target="target")
    assert len(res) > len(df)
    assert res["target"].dtype == df["target"].dtype
    assert set(res["target"]) <= {"cat", "dog", "bird"}


def _count_experimental(caplog) -> int:
    return sum("EXPERIMENTAL" in r.getMessage() for r in caplog.records)


def test_experimental_warning_emitted_once_per_configuration(caplog, monkeypatch):
    """Repeated fit_resample calls with the same (algorithm, multiplier)
    log the experimental WARNING only once."""
    import configurable_automl_engine.oversampling.experimental as os_mod

    monkeypatch.setattr(os_mod, "OVERSAMPLING_EXPERIMENTAL", True)
    df = pd.DataFrame({"f1": np.arange(20.0), "target": np.arange(20.0)})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        for seed in range(15):
            DataOversampler(algorithm="random", multiplier=1.5, random_state=seed).fit_resample(
                df[["f1"]], df["target"]
            )
    assert _count_experimental(caplog) == 1


def test_experimental_warning_once_per_distinct_configuration(caplog, monkeypatch):
    """A different algorithm or multiplier is a new configuration and warns
    again; algorithm name normalization maps to the same key."""
    import configurable_automl_engine.oversampling.experimental as os_mod

    monkeypatch.setattr(os_mod, "OVERSAMPLING_EXPERIMENTAL", True)
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"f1": rng.randn(60), "f2": rng.randn(60)})
    df["target"] = rng.randn(60)
    X, y = df[["f1", "f2"]], df["target"]
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        DataOversampler(algorithm="random", multiplier=1.5, random_state=0).fit_resample(X, y)
        DataOversampler(algorithm="RANDOM", multiplier=1.5, random_state=0).fit_resample(X, y)
        DataOversampler(algorithm="random", multiplier=2.0, random_state=0).fit_resample(X, y)
        DataOversampler(algorithm="smote", multiplier=1.5, random_state=0).fit_resample(X, y)
    assert _count_experimental(caplog) == 3


def test_experimental_warning_not_emitted_without_generation(caplog, monkeypatch):
    """multiplier == 1 without balancing generates no rows: no WARNING and
    the configuration is not recorded."""
    import configurable_automl_engine.oversampling.experimental as os_mod

    monkeypatch.setattr(os_mod, "OVERSAMPLING_EXPERIMENTAL", True)
    df = pd.DataFrame({"f1": np.arange(10.0), "target": np.arange(10.0)})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        DataOversampler(algorithm="random", multiplier=1.0, random_state=0).fit_resample(
            df[["f1"]], df["target"]
        )
    assert _count_experimental(caplog) == 0
    assert os_mod._experimental_warned == set()


def test_reset_experimental_warnings_allows_reemission(caplog, monkeypatch):
    """reset_experimental_warnings clears the dedup state."""
    import configurable_automl_engine.oversampling.experimental as os_mod

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        os_mod.warn_experimental_once(algorithm="random", multiplier=1.5)
        os_mod.warn_experimental_once(algorithm="random", multiplier=1.5)
        os_mod.reset_experimental_warnings()
        os_mod.warn_experimental_once(algorithm="random", multiplier=1.5)
    assert _count_experimental(caplog) == 2


def test_experimental_warning_thread_safe(caplog):
    """Concurrent emissions for the same configuration log exactly once."""
    import threading

    import configurable_automl_engine.oversampling.experimental as os_mod

    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        os_mod.warn_experimental_once(algorithm="smote", multiplier=2.0)

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.oversampling"):
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert _count_experimental(caplog) == 1


# ------------------------------------------------------------------ #
#  Binary indicator columns (one-hot / hashing) are not noised        #
# ------------------------------------------------------------------ #


def test_binary_indicator_columns_detection():
    """0/1 columns (incl. NaN and numeric strings) are indicators; others not."""
    df = pd.DataFrame(
        {
            "onehot": [0.0, 1.0, 0.0, 1.0],
            "with_nan": [1.0, np.nan, 0.0, 1.0],
            "str01": ["0", "1", "1", "0"],
            "numeric": [0.0, 1.0, 2.0, 0.5],
            "all_nan": [np.nan] * 4,
            "cat": ["a", "b", "a", "b"],
            "flag": [True, False, True, False],
            # A 0/1 flag after StandardScaler / RobustScaler
            "scaled": [-1.02, 0.98, 0.98, -1.02],
            "shifted": [-1.0, 0.0, 0.0, np.nan],
            "str_scaled": ["-0.5", "1.5", "1.5", "-0.5"],
        }
    )
    assert set(_binary_indicator_columns(df)) == {
        "onehot",
        "with_nan",
        "str01",
        "scaled",
        "shifted",
        "str_scaled",
    }


def test_binary_indicator_columns_empty_frame():
    """A frame without numeric-like columns has no indicators."""
    assert _binary_indicator_columns(pd.DataFrame({"c": ["x", "y"]})) == []


@pytest.mark.parametrize("algorithm", ["random", "smote", "adasyn"])
def test_dense_onehot_columns_not_noised(algorithm):
    """After a one-hot preprocessor the 0/1 columns stay 0/1 for random and
    are not perturbed beyond the interpolation for SMOTE/ADASYN; the
    continuous column is noised."""
    rng = np.random.default_rng(0)
    n = 40
    onehot = np.eye(3)[rng.integers(0, 3, size=n)]
    cont = rng.normal(size=(n, 1))
    X = np.hstack([cont, onehot])
    y = np.linspace(0.0, 1.0, n)
    kwargs = dict(algorithm=algorithm, multiplier=1.5, target_bins=2, random_state=1)
    X_clean, _ = DataOversampler(**kwargs).fit_resample(X, y)
    X_noisy, _ = DataOversampler(
        add_noise=True, noise_level=0.2, **kwargs
    ).fit_resample(X, y)
    X_clean = np.asarray(X_clean, dtype=float)
    X_noisy = np.asarray(X_noisy, dtype=float)
    np.testing.assert_array_equal(X_noisy[:, 1:], X_clean[:, 1:])
    assert not np.allclose(X_noisy[:, 0], X_clean[:, 0])
    if algorithm == "random":
        assert set(np.unique(X_noisy[:, 1:])) <= {0.0, 1.0}


def test_dataframe_binary_flag_not_noised():
    """A raw 0/1 flag in a DataFrame is excluded from the noise."""
    df = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "flag": [0, 1, 0, 1, 1, 0],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    sampler = DataOversampler(
        multiplier=2, add_noise=True, noise_level=0.5, target_bins=2, random_state=0
    )
    result = sampler.oversample(df, target="target")
    assert set(result["flag"].unique()) <= {0, 1}
    assert not np.array_equal(result["numeric"].values[:6], df["numeric"].values)


def test_sparse_binary_columns_not_noised():
    """Sparse input: columns whose stored values are all 1 (one-hot /
    hashing) are left untouched; columns with real magnitudes are noised."""
    X_sparse = sparse.csr_matrix(
        [
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 6.0],
        ]
    )
    sampler = DataOversampler(add_noise=True, noise_level=0.5, random_state=0)
    X_noisy = sampler._add_sparse_gaussian_noise(X_sparse).toarray()
    dense = X_sparse.toarray()
    np.testing.assert_array_equal(X_noisy[:, :2], dense[:, :2])
    nz = dense[:, 2] != 0
    assert np.all(X_noisy[nz, 2] != dense[nz, 2])


def test_sparse_scaled_binary_columns_not_noised():
    """Sparse input: two-valued columns that are not 0/1 (a scaled flag with
    both levels stored, a shifted flag with an implicit zero) are left
    untouched; a column with three distinct values is noised."""
    X_sparse = sparse.csr_matrix(
        [
            [-0.5, -1.0, 3.0],
            [1.9, 0.0, 0.0],
            [-0.5, -1.0, 2.0],
            [1.9, 0.0, 3.0],
        ]
    )
    sampler = DataOversampler(add_noise=True, noise_level=0.5, random_state=0)
    X_noisy = sampler._add_sparse_gaussian_noise(X_sparse).toarray()
    dense = X_sparse.toarray()
    np.testing.assert_array_equal(X_noisy[:, :2], dense[:, :2])
    nz = dense[:, 2] != 0
    assert np.all(X_noisy[nz, 2] != dense[nz, 2])


@pytest.mark.parametrize("scaling", ["standard", "robust", "none"])
@pytest.mark.parametrize("algorithm", ["random", "smote"])
def test_preprocessed_flags_and_missing_indicators_not_noised(scaling, algorithm):
    """Raw 0/1 flags and missing-value indicators keep their two levels
    after the real preprocessor, whatever the scaler, so the noise skips them
    while the continuous column is still noised."""
    from configurable_automl_engine.preprocessing import build_preprocessor

    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "flag": rng.integers(0, 2, n).astype(float),
            "rare": (rng.random(n) < 0.1).astype(float),
        }
    )
    X.loc[rng.random(n) < 0.2, "x"] = np.nan
    y = rng.normal(size=n)
    pre = build_preprocessor(list(X.columns), [], ["x", "flag", "rare"], scaling=scaling)
    Xt = np.asarray(pre.fit_transform(X, y), dtype=float)
    names = list(pre.get_feature_names_out())
    two_valued = [i for i, nm in enumerate(names) if nm != "num__x"]
    assert any("missingindicator" in names[i] for i in two_valued)

    kwargs = dict(algorithm=algorithm, multiplier=2.0, target_bins=2, random_state=1)
    X_clean, _ = DataOversampler(**kwargs).fit_resample(Xt, y)
    X_noisy, _ = DataOversampler(
        add_noise=True, noise_level=0.2, **kwargs
    ).fit_resample(Xt, y)
    X_clean = np.asarray(X_clean, dtype=float)
    X_noisy = np.asarray(X_noisy, dtype=float)
    np.testing.assert_array_equal(X_noisy[:, two_valued], X_clean[:, two_valued])
    x_idx = names.index("num__x")
    assert not np.allclose(X_noisy[:, x_idx], X_clean[:, x_idx])
    if algorithm == "random":
        for i in two_valued:
            assert set(np.unique(X_noisy[:, i])) == set(np.unique(Xt[:, i]))
