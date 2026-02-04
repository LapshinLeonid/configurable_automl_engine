import pytest
import pickle
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.serialization_utils import save_artifact, load_artifact

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