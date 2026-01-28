import pytest
import pandas as pd
import numpy as np
from configurable_automl_engine.trainer import ModelTrainer

# Список всех поддерживаемых сокращений из вашего models.py
ALGORITHMS = [
    "elasticnet", "dt", "sgd", "knn", "gpr", "svr",
    "isotonic", "ard", "rf", "glm", "xgboost", "ridge_regression"
]

# Список всех поддерживаемых алгоритмов
ALGORITHMS = [
    "elasticnet", "dt", "sgd", "knn", "gpr", "svr",
    "isotonic", "ard", "rf", "glm", "xgboost", "ridge_regression"
]
@pytest.mark.parametrize("algo", ALGORITHMS)
def test_every_algorithm_can_fit_and_predict(algo, small_dataset):
    """
    Тестирование всех алгоритмов на возможность обучения и предсказания.
    Использует централизованную фикстуру small_dataset.
    """
    # 1. Подготовка данных из фикстуры
    target_column = "target"
    X = small_dataset.drop(columns=[target_column])
    y = small_dataset[target_column]
    # 2. Специфика для Isotonic (требует ровно 1 признак)
    if algo in ["isotonic", "isotonic_regression"]:
        X = X[["feature_0"]]
    # 3. Инициализация тренера
    # ВАЖНО: Убедитесь, что ModelTrainer принимает аргумент 'algorithm'
    trainer = ModelTrainer(algorithm=algo)
    # 4. Обучение
    trainer.fit(X, y)
    # 5. Предсказание
    preds = trainer.predict(X)
    # 6. Проверки (Assertions)
    assert preds.shape[0] == y.shape[0], f"Алгоритм {algo}: несоответствие размера предсказаний"
    assert not np.isnan(preds).any(), f"Алгоритм {algo} вернул NaN в предсказаниях"
    
    # Проверка наличия рассчитанной метрики R2 после фита
    assert hasattr(trainer, 'val_r2_'), f"Алгоритм {algo} не установил атрибут val_r2_"
    assert trainer.val_r2_ is not None

