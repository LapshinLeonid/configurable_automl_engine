import pytest
import pickle
import os
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch
from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.serialization_utils import (
    save_artifact,
    load_artifact,
)

# Test data
TEST_DATA = {"key": "value", "number": 42}


@pytest.fixture
def temp_path(tmp_path):
    """Fixture that creates a temporary file path."""
    return tmp_path / "test_artifact.pkl"


class TestSerializationUtils:
    # --- Tests for save_artifact ---

    def test_save_artifact_pickle(self, temp_path):
        """Check saving via pickle (else branch)."""
        save_artifact(TEST_DATA, temp_path, SerializationFormat.pickle)

        assert temp_path.exists()
        with open(temp_path, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == TEST_DATA

    def test_save_artifact_joblib(self, temp_path):
        """Check saving via joblib (lines 13-14 coverage)."""
        # Mock joblib so tests do not depend on it being installed
        with patch("joblib.dump") as mock_dump:
            save_artifact(TEST_DATA, temp_path, SerializationFormat.joblib)
            mock_dump.assert_called_once_with(TEST_DATA, Path(temp_path))

    # --- Tests for load_artifact ---

    def test_load_artifact_file_not_found(self):
        """Check that FileNotFoundError is raised (line 25 coverage)."""
        non_existent_path = "non_existent_file.art"
        with pytest.raises(FileNotFoundError) as excinfo:
            load_artifact(non_existent_path, SerializationFormat.pickle)
        assert "Artifact not found" in str(excinfo.value)

    def test_load_artifact_pickle(self, temp_path):
        """Check loading via pickle (else branch)."""
        # First save manually
        with open(temp_path, "wb") as f:
            pickle.dump(TEST_DATA, f)

        loaded_data = load_artifact(temp_path, SerializationFormat.pickle)
        assert loaded_data == TEST_DATA

    def test_load_artifact_joblib(self, temp_path):
        """Check loading via joblib (lines 28-29 coverage)."""
        # Create an empty file so the path.exists() check passes
        temp_path.touch()

        with patch("joblib.load") as mock_load:
            mock_load.return_value = TEST_DATA
            result = load_artifact(temp_path, SerializationFormat.joblib)

            mock_load.assert_called_once_with(Path(temp_path))
            assert result == TEST_DATA

    def test_save_load_integration_pickle(self, temp_path):
        """Integration test: save and load via pickle."""
        save_artifact(TEST_DATA, temp_path, SerializationFormat.pickle)
        result = load_artifact(temp_path, SerializationFormat.pickle)
        assert result == TEST_DATA

    def test_path_as_string(self, tmp_path):
        """Check that the functions accept a path as a string (lines 11 and 23 coverage)."""
        str_path = str(tmp_path / "str_path.pkl")
        save_artifact(TEST_DATA, str_path, SerializationFormat.pickle)
        assert os.path.exists(str_path)

        result = load_artifact(str_path, SerializationFormat.pickle)
        assert result == TEST_DATA

    # --- Tests for the extension/format mismatch warning ---

    _LOGGER_NAME = "configurable_automl_engine.common.serialization_utils"

    def _records_from_module(self, caplog):
        """Log records only from the serialization_utils module."""
        return [r for r in caplog.records if r.name == self._LOGGER_NAME]

    def test_save_artifact_warns_on_extension_mismatch(self, tmp_path, caplog):
        """Warning when saving pickle to a file with a .joblib extension."""
        path = tmp_path / "model.joblib"
        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            save_artifact(TEST_DATA, path, SerializationFormat.pickle)

        assert path.exists()
        records = self._records_from_module(caplog)
        assert any(
            ".joblib" in record.message and "pickle" in record.message
            for record in records
        )

    def test_load_artifact_warns_on_extension_mismatch(self, tmp_path, caplog):
        """Warning when loading pickle from a file with a .joblib extension."""
        path = tmp_path / "model.joblib"
        with open(path, "wb") as f:
            pickle.dump(TEST_DATA, f)

        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            result = load_artifact(path, SerializationFormat.pickle)

        assert result == TEST_DATA
        records = self._records_from_module(caplog)
        assert any(
            ".joblib" in record.message and "pickle" in record.message
            for record in records
        )

    def test_save_artifact_no_warning_on_matching_extension(self, tmp_path, caplog):
        """No warning when the extension matches the format (.pkl + pickle)."""
        path = tmp_path / "model.pkl"
        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            save_artifact(TEST_DATA, path, SerializationFormat.pickle)

        records = self._records_from_module(caplog)
        assert not any(record.levelno >= 30 for record in records)

    def test_save_artifact_joblib_warns_on_pkl_extension(self, tmp_path, caplog):
        """Warning when saving joblib to a file with a .pkl extension."""
        path = tmp_path / "model.pkl"
        with (
            caplog.at_level("WARNING", logger=self._LOGGER_NAME),
            patch("joblib.dump") as mock_dump,
        ):
            save_artifact(TEST_DATA, path, SerializationFormat.joblib)
            mock_dump.assert_called_once_with(TEST_DATA, Path(path))

        records = self._records_from_module(caplog)
        assert any(
            ".pkl" in record.message and "joblib" in record.message
            for record in records
        )

    def test_load_artifact_joblib_warns_on_pkl_extension(self, tmp_path, caplog):
        """Warning when loading joblib from a file with a .pkl extension."""
        path = tmp_path / "model.pkl"
        path.touch()

        with (
            caplog.at_level("WARNING", logger=self._LOGGER_NAME),
            patch("joblib.load") as mock_load,
        ):
            mock_load.return_value = TEST_DATA
            result = load_artifact(path, SerializationFormat.joblib)

        assert result == TEST_DATA
        records = self._records_from_module(caplog)
        assert any(
            ".pkl" in record.message and "joblib" in record.message
            for record in records
        )

    def test_save_artifact_case_insensitive_extension_no_warning(
        self, tmp_path, caplog
    ):
        """No warning for an extension in a different case (.PKL + pickle)."""
        path = tmp_path / "model.PKL"
        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            save_artifact(TEST_DATA, path, SerializationFormat.pickle)

        records = self._records_from_module(caplog)
        assert not any(record.levelno >= 30 for record in records)

    def test_save_artifact_no_extension_warns(self, tmp_path, caplog):
        """Warning for a path without an extension (the extension convention is broken)."""
        path = tmp_path / "model"
        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            save_artifact(TEST_DATA, path, SerializationFormat.pickle)

        records = self._records_from_module(caplog)
        assert any(
            "does not match serialization format" in record.message
            and "pickle" in record.message
            for record in records
        )

    def test_save_artifact_no_warning_on_unknown_format(self, tmp_path, caplog):
        """No warning for a format missing from the extension reference.

        Format correctness is ensured by typing and the checks inside
        save_artifact/load_artifact, so an unknown format must not produce a
        warning. The unknown value is cast to the type to stay within the
        function's domain.
        """
        path = tmp_path / "model.pkl"
        unknown_fmt = cast(SerializationFormat, "unknown_format")
        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            save_artifact(TEST_DATA, path, unknown_fmt)

        assert path.exists()
        with open(path, "rb") as f:
            assert pickle.load(f) == TEST_DATA

        records = self._records_from_module(caplog)
        assert not any(record.levelno >= 30 for record in records)
