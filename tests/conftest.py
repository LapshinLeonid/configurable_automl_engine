import pytest
from .data_factory import create_mock_df

@pytest.fixture
def small_dataset():
    """
    Фикстура для быстрого тестирования базовой логики.
    Создает минимальный набор данных: 10 строк, 3 признака.
    """
    return create_mock_df(rows=10, cols=3, target="target")

@pytest.fixture
def regression_dataset():
    """
    Фикстура для тестирования алгоритмов регрессии или обучения.
    Создает расширенный набор данных: 200 строк, 10 признаков.
    """
    return create_mock_df(rows=200, cols=10, target="yield_score")