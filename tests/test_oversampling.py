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
    with patch("configurable_automl_engine.oversampling.logger") as mock_logger:
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
    with patch("configurable_automl_engine.oversampling.logger") as mock_log:
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


# 5. Coverage of lines 220-223: Noise + sparse matrix
def test_sparse_matrix_noise_error():
    X_sparse = sparse.csr_matrix([[1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 0])
    sampler = DataOversampler(add_noise=True)
    with pytest.raises(TypeError, match="does not support sparse matrices"):
        sampler.fit_resample(X_sparse, y)


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
    """Continuous (regression) targets are detected; discrete targets are not."""
    rng = np.random.RandomState(0)
    # True continuous (non-integer floats)
    assert _is_continuous_target(pd.Series(rng.randn(200))) is True
    assert _is_continuous_target(pd.Series([0.5, 1.5, 2.5, 0.5, 1.5])) is True
    # Two unique non-integer floats are also continuous for imblearn
    assert _is_continuous_target(pd.Series([2.5, 5.5, 2.5, 5.5])) is True
    # Integer dtypes and integer-valued floats stay on the classification path
    assert _is_continuous_target(pd.Series(np.arange(100) % 40)) is False
    assert _is_continuous_target(pd.Series([0, 1, 0, 1])) is False
    assert _is_continuous_target(pd.Series([0.0, 1.0, 0.0, 1.0])) is False
    assert _is_continuous_target(pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])) is False
    # Bool targets stay on the classification path
    assert _is_continuous_target(pd.Series([True, False, True, False])) is False
    # Non-numeric targets are never continuous
    assert _is_continuous_target(pd.Series(["a", "b", "c", "a"])) is False
    # NaN-only / empty targets are not continuous
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


def test_binary_float_target_stays_classification():
    """Binary float 0.0/1.0 targets must NOT take the continuous path:
    the result contains only the original class values."""
    rng = np.random.RandomState(5)
    df = pd.DataFrame({"f1": rng.randn(80), "f2": rng.randn(80)})
    df["target"] = np.where(rng.rand(80) > 0.5, 1.0, 0.0)

    sampler = DataOversampler(algorithm="smote", multiplier=2.0, random_state=42)
    res = sampler.oversample(df, target="target")

    assert len(res) > len(df)
    assert set(res["target"].unique()) == {0.0, 1.0}


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

    # A valid int8 code that is NOT a bin of this target.
    valid_int8_max = 127
    y_binned = np.concatenate(
        [binned.to_numpy(), np.array([valid_int8_max], dtype=np.int8)]
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
