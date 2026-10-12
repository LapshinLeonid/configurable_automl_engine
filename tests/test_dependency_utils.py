import pytest
from unittest.mock import patch

# Assume the function is in utils.py
from configurable_automl_engine.common.dependency_utils import is_installed


def test_is_installed_empty_string():
    """Check the empty-string case (your target string)."""
    assert is_installed("") is False


def test_is_installed_none():
    """Check the case of passing None (if types are not enforced)."""
    # This test is useful if strict typing is not used
    assert is_installed(None) is False


def test_is_installed_standard_library():
    """Check a standard-library package (always available)."""
    assert is_installed("os") is True
    assert is_installed("sys") is True


def test_is_installed_non_existent():
    """Check a definitely non-existent package."""
    assert is_installed("this_package_definitely_does_not_exist_123") is False


def test_is_installed_mocked_success():
    """Test with a mocked successful find_spec lookup."""
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = True  # find_spec returns a non-None object
        assert is_installed("any_package") is True
        mock_find.assert_called_once_with("any_package")


def test_is_installed_mocked_failure():
    """Test with a mocked missing package."""
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = None
        assert is_installed("missing_package") is False
