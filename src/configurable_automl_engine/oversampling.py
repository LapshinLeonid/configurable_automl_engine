import sys
import os
import numpy as np
import pandas as pd
import threading
from collections import Counter
from typing import Any, Optional, Dict

from math import ceil
import logging

from imblearn.base import BaseSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


logger = logging.getLogger(__name__)

class DataOversampler(BaseSampler):
    """
    Потокобезопасный класс для увеличения объёма и / или балансировки данных.
    Совместим с scikit-learn Pipeline и imbalanced-learn API.
    """
    _sampling_type = "over-sampling"

    _parameter_constraints: dict = {
        "multiplier": [float, int],
        "algorithm": [str],
        "add_noise": ["boolean"],
        "n_jobs": [int, None],
        "log_dir": [str, None],
    }

    def __init__(
        self,
        *,
        multiplier: float | int = 1.0,
        algorithm: str = "random",
        add_noise: bool = False,
        n_jobs: int = 1,
    ):
        # Сохраняем ровно то, что пришло, чтобы потом работал clone
        self.multiplier = multiplier
        self.algorithm = algorithm
        self.add_noise = add_noise
        self.n_jobs = n_jobs

        super().__init__()
        self._lock = threading.RLock()

    def _strategy(self, y: pd.Series) -> Dict[Any, int]:
        counts = Counter(y)
        return {cls: ceil(cnt * self.multiplier) for cls, cnt in counts.items()}

    def _add_gaussian_noise(self, X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        return X + np.random.normal(scale=noise_level, size=X.shape)

    def __getstate__(self):
            # Копируем состояние объекта
            state = self.__dict__.copy()
            # Удаляем несериализуемый объект lock по правильному имени
            if '_lock' in state:
                del state['_lock']
            return state
    def __setstate__(self, state):
        # Восстанавливаем состояние
        self.__dict__.update(state)
        # Заново инициализируем lock после десериализации
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Пункт 2: Реализация защищенного метода _fit_resample              #
    # ------------------------------------------------------------------ #
    def _fit_resample(self, X, y):
        """
        Внутренняя логика ресемплирования с использованием валидации sklearn/imblearn.
        """
        self.multiplier = float(self.multiplier)
        self.algorithm = self.algorithm.lower().replace(" ", "_")

        if self.multiplier <= 0:
            raise ValueError("multiplier must be > 0")
        try:
            # 1. Стандартная валидация входных данных
            # accept_sparse=False, так как SMOTE/ADASYN требуют плотных матриц в текущей реализации
            X_validated, y_validated, *rest = self._check_X_y(X, y)
            
            # Сохраняем названия колонок, если X был DataFrame
            feature_names = X.columns if hasattr(X, "columns") else None
            target_name = y.name if hasattr(y, "name") else "target"

            # Работаем с DataFrame внутри для удобства алгоритмов
            X_df = pd.DataFrame(X_validated, columns=feature_names)
            y_s = pd.Series(y_validated, name=target_name)

            # 2. Определение алгоритма и выполнение ресемплирования
            
            # Случай RANDOM
            if self.algorithm in ("random", "random_with_noise"):
                # Если multiplier > 1, сначала используем стратегию увеличения
                # Если multiplier = 1, RandomOverSampler просто сбалансирует классы
                sampling_strategy = self._strategy(y_s) if self.multiplier >= 1.0 else "auto"
                
                ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=None)
                X_res, y_res = ros.fit_resample(X_df, y_s)
                
                if self.add_noise or self.algorithm == "random_with_noise":
                    arr = self._add_gaussian_noise(X_res.to_numpy())
                    X_res = pd.DataFrame(arr, columns=X_df.columns)
                
                logger.info(f"Random resample: {len(X_validated)} -> {len(X_res)}")
                return X_res.to_numpy(), y_res.to_numpy()

            # Случай SMOTE / ADASYN
            counts = Counter(y_s)
            strategy = self._strategy(y_s)
            
            # Динамическая настройка k_neighbors (не может быть больше размера миноритарного класса - 1)
            min_samples = min(counts.values())
            k_neighbors = max(1, min(5, min_samples - 1))
            
            if self.algorithm == "smote":
                sampler = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors)
            elif self.algorithm == "adasyn":
                sampler = ADASYN(sampling_strategy=strategy, n_neighbors=k_neighbors)
            else:
                raise ValueError(f"Неподдерживаемый алгоритм: {self.algorithm}")

            X_res, y_res = sampler.fit_resample(X_df, y_s)
            logger.info(f"{self.algorithm.upper()} resample: {len(X_validated)} -> {len(X_res)}")
            
            # Возвращаем numpy массивы (стандарт для внутренних методов sklearn/imblearn)
            return X_res.to_numpy(), y_res.to_numpy()

        except Exception as e:
            logger.error(f"Ошибка в _fit_resample: {e}", exc_info=True)
            raise

    def oversample(self, data: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """
        Высокоуровневая обертка для работы с DataFrame.
        Использует публичный fit_resample(), который сам вызовет _fit_resample().
        """
        try:
            if target:
                y = data[target]
                X = data.drop(target, axis=1)
            else:
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
                target = data.columns[-1]

            # Вызываем публичный метод (он сделает дополнительные проверки)
            X_res, y_res = self.fit_resample(X, y)
            
            # Сборка итогового DataFrame
            res_df = pd.DataFrame(X_res, columns=X.columns)
            res_df[target] = y_res
            return res_df.reset_index(drop=True)
        except Exception as e:
            logger.error(f"Ошибка в oversample: {e}", exc_info=True)
            raise

# ------------------------------------------------------------------ #
#  Функциональные интерфейсы                                         #
# ------------------------------------------------------------------ #

def oversample(
    data: pd.DataFrame,
    multiplier: float = 1.0,
    algorithm: str = "random",
    add_noise: bool = False,
    target: Optional[str] = None,
) -> pd.DataFrame:
    sampler = DataOversampler(
        multiplier=multiplier,
        algorithm=algorithm,
        add_noise=add_noise
    )
    return sampler.oversample(data, target=target)