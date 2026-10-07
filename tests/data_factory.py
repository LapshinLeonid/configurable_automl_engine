from __future__ import annotations

import numpy as np
import pandas as pd


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


# ──────────────────────────────────────────────────────────────────────────────
#  T9 (issue #70): эталонные и вырожденные датасеты регрессионного бенчмарка
#  «до/после» Sanity Gate. Все генераторы используют фиксированные seed'ы —
#  воспроизводимость прогона гарантирована (чувствительные реальные данные
#  в git не добавляются, как в issue #60).
#
#  Возвращаемый формат: (X: np.ndarray (n, p), y: np.ndarray (n,)).
# ──────────────────────────────────────────────────────────────────────────────


def make_reference_noisy_r2_03(n: int = 300, p: int = 8, seed: int = 42):
    """Эталонный зашумлённый датасет с теоретическим R²≈0.3.

    Линейный сигнал (сумма первых 4 признаков) + шум, подобранный так, чтобы
    R² = Var(signal)/(Var(signal)+Var(noise)) = 0.3 ровно. Честные линейные
    модели (ridge/lasso/elasticnet) обязаны проходить все контуры гейта:
    diversity ≈ R² ≈ 0.3 > 0.15, gap ≤ 1.5.

    Args:
        n: Число строк.
        p: Число признаков.
        seed: Зерно генератора.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    signal = X[:, :4].sum(axis=1)
    noise_var = 4.0 * (1.0 / 0.3 - 1.0)
    y = signal + rng.randn(n) * np.sqrt(noise_var)
    return X, y


def make_reference_wide(
    n: int = 300, p: int = 200, k: int = 10, w: float = 0.6, seed: int = 42
):
    """Эталонный широкий датасет: сотни признаков, мало информативных.

    Слабая линейная связь через ``k`` информативных признаков на фоне p−k
    шумовых колонок. Sparse-линейные модели (lasso/elasticnet) честно
    выигрывают; ridge с дефолтным пространством поиска заметно хуже и не
    попадает в пул финалистов. Гейт не должен срабатывать.

    Args:
        n: Число строк.
        p: Число признаков (сотни).
        k: Число информативных признаков.
        w: Вес информативных признаков.
        seed: Зерно генератора.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = w * X[:, :k].sum(axis=1) + rng.randn(n)
    return X, y


def make_reference_small_n(n: int = 80, p: int = 6, seed: int = 42):
    """Эталонный датасет малого объёма N (малые выборки, T9).

    Чистый линейный сигнал с умеренным шумом: на N=80 честные линейные
    модели стабильно проходят гейт (diversity ≫ 0.15, gap ≤ 1.5), разброс
    результатов подавлен фиксированным seed.

    Args:
        n: Число строк (малое N).
        p: Число признаков.
        seed: Зерно генератора.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.5
    return X, y


def make_reference_boundary_diversity(
    n: int = 400, p: int = 4, signal: float = 0.46, seed: int = 42
):
    """Эталонный граничный датасет: diversity честной модели у порога 0.15.

    Слабый сигнал: OOF-diversity обученного Ridge ≈ 0.155 при пороге
    ``min_prediction_diversity=0.15`` (запас < 5%). Документированный порог
    допуска «ухудшение ≤ ε»: модель, находящаяся на границе порога, НЕ должна
    дисквалифицироваться (равенство/строгий недобор — единственная точка
    провала), и победитель не должен меняться.

    Args:
        n: Число строк.
        p: Число признаков.
        signal: Амплитуда сигнала (определяет diversity ≈ signal²/(signal²+1)).
        seed: Зерно генератора.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = signal * X[:, 0] + rng.randn(n)
    return X, y


def make_degenerate_flat_shelf(n: int = 150, p: int = 10, seed: int = 42):
    """Вырожденный датасет «полка»: сильная регуляризация зануляет веса.

    Слабая линейная связь: ElasticNet с alpha=500 обнуляет ВСЕ коэффициенты
    (``coef_ == 0``) и выдаёт плоские (константные) OOF-предсказания — гейт
    обязан сработать (контуры А/Б/В). Честный Ridge на тех же данных проходит.

    Args:
        n: Число строк.
        p: Число признаков.
        seed: Зерно генератора.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 0.5 * X[:, 0] + rng.randn(n)
    return X, y


def make_degenerate_tanh_plateau(n: int = 200, p: int = 6, seed: int = 42):
    """Вырожденный датасет «плато tanh»: SVR-sigmoid упирается в плато.

    Линейный сигнал с шумом: SVR(kernel='sigmoid', gamma=0.001, coef0=0.0)
    выдаёт почти константные предсказания (разброс ≪ разброса y) — гейт
    обязан сработать (контур А). Честный SVR-RBF (C=1.0) проходит.

    Args:
        n: Число строк.
        p: Число признаков.
        seed: Зерно генератора.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 1.2
    return X, y


def make_degenerate_overfit(n: int = 120, p: int = 6, seed: int = 42):
    """Вырожденный датасет «переобучение»: дерево с RMSE_train≈0.

    Линейный сигнал с шумом: DecisionTreeRegressor(max_depth=20) идеально
    обучается на train (RMSE_full≈0) — гейт обязан сработать (контур Г,
    деление на ноль). Честный Ridge проходит.

    Args:
        n: Число строк.
        p: Число признаков.
        seed: Зерно генератора.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.5
    return X, y
