import pandas as pd
import numpy as np

def create_mock_df(rows=100, cols=5, target="target"):
    """
    Генерирует синтетический DataFrame для тестирования.
    
    Параметры:
    ----------
    rows : int
        Количество строк в генерируемом наборе данных.
    cols : int
        Количество признаков (колонок с префиксом 'feature_').
    target : str
        Название целевой переменной.
        
    Возвращает:
    -----------
    pd.DataFrame
        Объект DataFrame, содержащий случайные числа с плавающей точкой.
    """
    
    # Фиксируем seed для воспроизводимости тестов
    np.random.seed(42)
    
    # Генерация названий колонок: feature_0, feature_1, ...
    column_names = [f"feature_{i}" for i in range(cols)]
    
    # Генерация матрицы признаков (нормальное распределение)
    data = np.random.randn(rows, cols)
    
    # Создание DataFrame
    df = pd.DataFrame(data, columns=column_names)
    
    # Генерация целевой переменной (бинарная классификация или регрессия)
    # В данном случае создаем случайную непрерывную величину
    df[target] = np.random.rand(rows)
    
    return df

if __name__ == "__main__":
    # Пример использования для проверки
    test_df = create_mock_df(rows=10, cols=3)
    print("Сгенерированный DataFrame:")
    print(test_df.head())
    print(f"\nФормат данных: {test_df.shape}")