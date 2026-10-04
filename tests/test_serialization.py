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

# Тестовые данные
TEST_DATA = {"key": "value", "number": 42}


@pytest.fixture
def temp_path(tmp_path):
    """Фикстура для создания временного пути к файлу."""
    return tmp_path / "test_artifact.pkl"


class TestSerializationUtils:
    # --- Тесты для save_artifact ---

    def test_save_artifact_pickle(self, temp_path):
        """Проверка сохранения через pickle (ветка else)."""
        save_artifact(TEST_DATA, temp_path, SerializationFormat.pickle)

        assert temp_path.exists()
        with open(temp_path, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == TEST_DATA

    def test_save_artifact_joblib(self, temp_path):
        """Проверка сохранения через joblib (покрытие строк 13-14)."""
        # Мокаем joblib, чтобы не зависеть от его наличия в окружении при тестах
        with patch("joblib.dump") as mock_dump:
            save_artifact(TEST_DATA, temp_path, SerializationFormat.joblib)
            mock_dump.assert_called_once_with(TEST_DATA, Path(temp_path))

    # --- Тесты для load_artifact ---

    def test_load_artifact_file_not_found(self):
        """Проверка вызова исключения FileNotFoundError (покрытие строки 25)."""
        non_existent_path = "non_existent_file.art"
        with pytest.raises(FileNotFoundError) as excinfo:
            load_artifact(non_existent_path, SerializationFormat.pickle)
        assert "Artifact not found" in str(excinfo.value)

    def test_load_artifact_pickle(self, temp_path):
        """Проверка загрузки через pickle (ветка else)."""
        # Сначала сохраним вручную
        with open(temp_path, "wb") as f:
            pickle.dump(TEST_DATA, f)

        loaded_data = load_artifact(temp_path, SerializationFormat.pickle)
        assert loaded_data == TEST_DATA

    def test_load_artifact_joblib(self, temp_path):
        """Проверка загрузки через joblib (покрытие строк 28-29)."""
        # Создаем пустой файл, чтобы проверка path.exists() прошла
        temp_path.touch()

        with patch("joblib.load") as mock_load:
            mock_load.return_value = TEST_DATA
            result = load_artifact(temp_path, SerializationFormat.joblib)

            mock_load.assert_called_once_with(Path(temp_path))
            assert result == TEST_DATA

    def test_save_load_integration_pickle(self, temp_path):
        """Интеграционный тест: сохранение и загрузка через pickle."""
        save_artifact(TEST_DATA, temp_path, SerializationFormat.pickle)
        result = load_artifact(temp_path, SerializationFormat.pickle)
        assert result == TEST_DATA

    def test_path_as_string(self, tmp_path):
        """Проверка того, что функции принимают путь в виде строки (покрытие строк 11 и 23)."""
        str_path = str(tmp_path / "str_path.pkl")
        save_artifact(TEST_DATA, str_path, SerializationFormat.pickle)
        assert os.path.exists(str_path)

        result = load_artifact(str_path, SerializationFormat.pickle)
        assert result == TEST_DATA

    # --- Тесты предупреждения о несоответствии расширения формату ---

    _LOGGER_NAME = "configurable_automl_engine.common.serialization_utils"

    def _records_from_module(self, caplog):
        """Записи логов только от модуля serialization_utils."""
        return [r for r in caplog.records if r.name == self._LOGGER_NAME]

    def test_save_artifact_warns_on_extension_mismatch(self, tmp_path, caplog):
        """Предупреждение при сохранении pickle в файл с расширением .joblib."""
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
        """Предупреждение при загрузке pickle из файла с расширением .joblib."""
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
        """Предупреждения нет, если расширение соответствует формату (.pkl + pickle)."""
        path = tmp_path / "model.pkl"
        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            save_artifact(TEST_DATA, path, SerializationFormat.pickle)

        records = self._records_from_module(caplog)
        assert not any(record.levelno >= 30 for record in records)

    def test_save_artifact_joblib_warns_on_pkl_extension(self, tmp_path, caplog):
        """Предупреждение при сохранении joblib в файл с расширением .pkl."""
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
        """Предупреждение при загрузке joblib из файла с расширением .pkl."""
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
        """Предупреждения нет для расширения в другом регистре (.PKL + pickle)."""
        path = tmp_path / "model.PKL"
        with caplog.at_level("WARNING", logger=self._LOGGER_NAME):
            save_artifact(TEST_DATA, path, SerializationFormat.pickle)

        records = self._records_from_module(caplog)
        assert not any(record.levelno >= 30 for record in records)

    def test_save_artifact_no_extension_warns(self, tmp_path, caplog):
        """Предупреждение для пути без расширения (соглашение о расширении нарушено)."""
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
        """Предупреждения нет для формата, отсутствующего в справочнике расширений.

        Корректность формата обеспечивается типизацией и проверками
        в save_artifact/load_artifact, поэтому при неизвестном формате
        предупреждение выдаваться не должно. Неизвестное значение
        приводится к типу через cast, чтобы остаться в домене функции.
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
