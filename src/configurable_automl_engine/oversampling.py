"""
    Data Oversampling Engine: Интеллектуальный расширитель выборок.
    Класс обеспечивает балансировку классов и синтетическое увеличение объема данных 
    с сохранением типизации и поддержкой многопоточности.
    
    Ключевые возможности:
        1. Multi-Algorithm: Поддержка Random, SMOTE и ADASYN.
        2. Noise Injection: Опциональное наложение Гауссовского шума для регуляризации.
        3. Type Integrity: Механизм `_restore_dtypes` предотвращает деградацию типов 
           (например, каст Categorical -> Object) после преобразований в Numpy.
        4. Thread-Safety: Использование RLock для 
        безопасного доступа к параметрам в Pipeline.
"""

import numpy as np
import pandas as pd
import threading
from collections import Counter
from typing import Any, Optional, Dict

from math import ceil
import logging

from imblearn.base import BaseSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC
from pandas.api.types import is_numeric_dtype


logger = logging.getLogger(__name__)

class DataOversampler(BaseSampler): # type: ignore[misc]
    """Увеличить объем данных и сбалансировать классы.
    
    Интеллектуальный оберточный класс над imbalanced-learn, адаптированный для Pandas.
    Поддерживает автоматическую обработку категориальных признаков (SMOTENC) и 
    сохранение целостности типов после синтетической генерации.

    Attributes:
        multiplier (float | int): Коэффициент увеличения выборки.
        algorithm (str): Название алгоритма ('random', 'smote', 'adasyn').
        add_noise (bool): Флаг добавления Гауссовского шума к числовым признакам.
        balance (bool): Флаг приведения всех классов к размеру мажоритарного.
        random_state (int | None): Зерно генератора случайных чисел.
        noise_level (float): Интенсивность шума относительно стандартного отклонения.
    """
    _sampling_type = "over-sampling"

    _parameter_constraints: dict [str, list[Any]] = {
        "multiplier": [float, int],
        "algorithm": [str],
        "add_noise": ["boolean"],
        "balance": ["boolean"],
        "random_state": [int, np.random.RandomState, None],
        "noise_level": [float]
    }

    def __init__(
        self,
        *,
        multiplier: float | int = 1.0,
        algorithm: str = "random",
        add_noise: bool = False,
        balance: bool = False,
        random_state: Optional[int] = 42,
        noise_level: float = 0.01
    ):
        # Сохраняем ровно то, что пришло, чтобы потом работал clone
        self.multiplier = multiplier
        self.algorithm = algorithm
        self.add_noise = add_noise
        self.balance = balance
        self.random_state = random_state
        self.noise_level = noise_level

        super().__init__()
        self._lock = threading.RLock()

    def _strategy(self, y: pd.Series, multiplier: float) -> Dict[Any, int]:
        """Рассчитать целевое количество экземпляров для каждого класса.
        Args:
            y (pd.Series): Вектор целевой переменной.
            multiplier (float): Коэффициент масштабирования.
        Returns:
            Dict[Any, int]: Словарь, где ключи — метки классов, 
                а значения — итоговое количество строк.
        """
        counts = Counter(y)
         # Если включена балансировка, подтягиваем все классы к мажоритарному,
        # умноженному на коэффициент. Если выключена — пропорционально множим каждый.
        if self.balance:
            # База — самый крупный класс
            base_size = max(counts.values())
            return {
                cls: ceil(base_size * multiplier) 
                for cls in counts.keys()
            }
        
        # Важно: imbalanced-learn ожидает итоговое количество экземпляров 
        # (Total), а не дельту
        return {cls: ceil(cnt * multiplier) for cls, cnt in counts.items()}

    def _add_gaussian_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить Гауссов шум в числовые столбцы и числовые элементы столбцов object.
        Применяет векторизированное вычисление шума для повышения производительности.
        Для нечисловых колонок пытается выполнить безопасное приведение типов.
        Args:
            df (pd.DataFrame): Набор данных для модификации.
        Returns:
            pd.DataFrame: Набор данных с добавленным шумом.
        """
        rng = np.random.default_rng(self.random_state)
        
        # 1. Сначала принудительно конвертируем "числовые строки" в числа
        # Это критично для pandas 3.0 и StringDtype
        potential_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in potential_cols:
            # Если в колонке '1.0', она станет float. 
            # Если 'abc', останется object/string.
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isna().any(): 
                df[col] = converted
        # 2. Теперь выбираем ВСЕ числовые колонки 
        # (включая те, что только что конвертировали)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if not numeric_cols.empty:
            # Векторизованное добавление шума для всех числовых колонок сразу
            stds = df[numeric_cols].std().fillna(1.0).replace(0, 1.0).values
            scales = stds * self.noise_level
            noise = rng.normal(0, 1.0, size=(len(df), len(numeric_cols))) * scales
            
            # Используем .values для скорости
            df[numeric_cols] = df[numeric_cols].values + noise
        return df

    def __getstate__(self) -> dict[str, Any]:
            # Копируем состояние объекта
            state = self.__dict__.copy()
            # Удаляем несериализуемый объект lock по правильному имени
            if '_lock' in state:
                del state['_lock']
            return state
    def __setstate__(self,
                     state: dict[str, Any]
                     )-> None:
        # Восстанавливаем состояние
        self.__dict__.update(state)
        # Заново инициализируем lock после десериализации
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Пункт 2: Реализация защищенного метода _fit_resample              #
    # ------------------------------------------------------------------ #
    
    def _restore_dtypes(self, 
                        df: pd.DataFrame, 
                        original_dtypes: pd.Series
                        ) -> pd.DataFrame:
        """Восстановить типы данных с адаптацией под внесенные изменения.
        
        Логика восстановления:
        1. Categorical: Восстанавливаются строго (включая категории и порядок).
        2. Integer: Если add_noise=True, кастуются в Float (32/64) для сохранения 
           дробной части шума. Если шума нет — возвращается исходный Integer.
        3. Object: Принудительно возвращаются к object для исключения каста в float/int
           библиотеками генерации.
        4. Floating: Сохраняют исходную точность (32/64 бита).
        Args:
            df (pd.DataFrame): Преобразованный набор данных.
            original_dtypes (pd.Series): Серия с исходными типами (`X.dtypes`).
        Returns:
            pd.DataFrame: Набор данных с восстановленной типизацией.
        """
        if original_dtypes is None:
            return df
            
        for col, dtype in original_dtypes.items():
            if col not in df.columns:
                continue
            try:
                # 1. Если это категория — восстанавливаем строго
                #  (важно для памяти и логики)
                if isinstance(dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype(dtype)
                
                # 2. Если это объект (строки) — возвращаем тип object
                elif (dtype is object or isinstance(dtype, np.dtype) 
                      and dtype.kind == 'O'):
                    # Если добавили шум и колонка стала числовой,
                    # то НЕ откатываем к object
                    if self.add_noise and is_numeric_dtype(df[col]):
                        pass 
                    else:
                        df[col] = df[col].astype(object)
                
                # 3. Если это целое число:
                elif np.issubdtype(dtype, np.integer):
                    # Если добавляли шум, НЕЛЬЗЯ возвращать int, иначе потеряем точность
                    if not self.add_noise:
                        # Используем pd.to_numeric для безопасности 
                        # и приводим к исходному int-типу
                        converted = pd.to_numeric(df[col], errors='coerce')
                        df[col] = (
                            converted.astype(dtype) if not converted.isna().any() 
                            else converted
                        )
                    else:
                        # ВЫБОР ТОЧНОСТИ: если был int64 -> float64, иначе float32
                        if dtype == np.int64:
                            df[col] = df[col].astype(np.float64)
                        else:
                            df[col] = df[col].astype(np.float32)
                # 4. Если исходный тип — число с плавающей точкой (float32 или float64):
                elif np.issubdtype(dtype, np.floating):
                    # Сохраняем исходный тип 
                    # (если был float64 -> будет float64, если float32 -> float32)
                    df[col] = df[col].astype(dtype)

                # 5. В остальных случаях (float и т.д.) пробуем вернуть исходный тип
                else:
                    if not pd.Series(df[col]).isna().any():
                        df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Failed to restore type for column {col}: {e}")
        return df
    
    def _fit_resample(
            self, 
            X: Any,
            y:Any
            ) -> tuple[np.ndarray, np.ndarray]:
        """Выполнить ресемплирование данных согласно выбранному алгоритму.
        Автоматически переключается на SMOTENC, если обнаружены нечисловые колонки.
        Для ADASYN реализован прозрачный fallback к SMOTENC при наличии категорий,
        так как оригинальный ADASYN работает только с числовыми данными.
        Для SMOTE/ADASYN: Динамический расчет k_neighbors.
        Предотвращает ValueError, если количество объектов в миноритарном классе 
        меньше стандартного значения (5). Устанавливается как min(5, n_samples-1).
        Args:
            X (Any): Признаки (массив, DataFrame или разреженная матрица).
            y (Any): Целевая переменная.
        Returns:
            tuple: Кортеж из двух numpy-массивов (X_resampled, y_resampled).
        Raises:
        
            ValueError: Если `multiplier` < 1 или алгоритм не поддерживается.
            TypeError: Если в данных отсутствуют числовые признаки (необходимы для 
                       вычисления расстояний в SMOTE/ADASYN) или при попытке 
                       добавить шум в разреженную матрицу.
        """
        with self._lock:
            # Сохраняем результат в атрибут с подчеркиванием 
            # (стандарт sklearn)
            self.multiplier_ = float(self.multiplier)
            # Используем локальную переменную вместо изменения self.algorithm
            algo_local = self.algorithm.lower().replace(" ", "_")

        if self.multiplier_ < 1:
            raise ValueError("multiplier must be >= 1")
        
        # Проверка на разреженные матрицы (scipy.sparse)
        is_sparse = hasattr(X, "tocsr") or hasattr(X, "issparse")
        
        # Разреженные матрицы несовместимы с добавлением шума.
        # Наложение Гауссовского шума делает все нулевые элементы ненулевыми,
        # что приводит к взрывному росту потребления памяти (денерилизации).
        if self.add_noise and is_sparse:
            raise TypeError(
                f"The '{self.algorithm}' algorithm with " 
                "noise injection does not support "
                "sparse matrices. Please convert your data to a dense format "
                "using X.toarray() or set add_noise=False."
            )

        try:
            # Стандартная валидация входных данных
            y_validated = np.asarray(y)
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            y_s = pd.Series(y_validated)
            
            # Определяем индексы нечисловых колонок для SMOTENC
            cat_features_idx = [
                i for i, col in enumerate(X_df.columns) 
                if not is_numeric_dtype(X_df[col])
            ]
            # Проверка на отсутствие числовых колонок для SMOTE/ADASYN
            if (algo_local in ["smote", "adasyn"] 
                and len(cat_features_idx) == len(X_df.columns)):
                raise TypeError(f"{self.algorithm.upper()} requires at"
                                " least one numeric feature.")
            
            # Для всех алгоритмов в текущей логике передаем полный набор данных
            X_to_resample = X_df

            # Определение алгоритма и выполнение ресемплирования
            strategy = self._strategy(y_s, self.multiplier_)


            if algo_local == "random":
                sampler = RandomOverSampler(sampling_strategy=strategy, 
                                      random_state=self.random_state)
            else:
                # Случай SMOTE / ADASYN
                counts = Counter(y_s)
                
                # Для SMOTE/ADASYN: Динамический расчет k_neighbors.
                # Предотвращает ValueError, если объектов в классе меньше
                # чем дефолтные 5 соседей.
                min_samples = min(counts.values())
                k_neighbors = max(1, min(5, min_samples - 1))

                if algo_local == "smote":
                    if cat_features_idx:
                        sampler = SMOTENC(categorical_features=cat_features_idx,
                                        sampling_strategy=strategy, 
                                        k_neighbors=k_neighbors,
                                        random_state=self.random_state)
                    else:
                        sampler = SMOTE(sampling_strategy=strategy, 
                                        k_neighbors=k_neighbors,
                                        random_state=self.random_state)
                elif algo_local == "adasyn":
                    if cat_features_idx:
                        # ВАЖНО: Логируем замену, чтобы не вводить в заблуждение
                        logger.warning(
                            "ADASYN does not support categorical columns. "
                            "Automatically falling back to SMOTENC for data integrity."
                        )
                        sampler = SMOTENC(categorical_features=cat_features_idx,
                                        sampling_strategy=strategy, 
                                        k_neighbors=k_neighbors,
                                        random_state=self.random_state)
                    else:
                        sampler = ADASYN(sampling_strategy=strategy,
                                        n_neighbors=k_neighbors,
                                        random_state=self.random_state)
                else:
                    raise ValueError(f"Unsupported algorithm: {algo_local}")

            X_res_raw, y_res = sampler.fit_resample(X_to_resample, y_s)
            # Приводим к DF, так как SMOTE может вернуть numpy
            X_res_df = pd.DataFrame(X_res_raw, columns=X_df.columns)[X_df.columns]
            
            # Единая точка наложения шума для всех алгоритмов
            if self.add_noise:
                X_res_df = self._add_gaussian_noise(X_res_df)
                
            logger.info(
                "%s resample: %d -> %d", 
                algo_local.upper(), 
                len(X_df), 
                len(X_res_df)
            )
            
            return X_res_df.to_numpy(), y_res.to_numpy()
        except (ValueError, TypeError, ImportError, RuntimeError):
            # Re-raise known data-related or configuration errors directly
            raise
        except Exception as e:
            logger.debug("Unexpected error trace in _fit_resample:", exc_info=True)
            logger.error(f"Critical error during data oversampling: {e}")
            raise

    def oversample(self, 
                   data: pd.DataFrame, 
                   target: Optional [str] = None
                   ) -> pd.DataFrame:
        """Увеличить выборку в формате DataFrame с сохранением метаданных.
        Args:
            data (pd.DataFrame): Исходный DataFrame.
            target (str, optional): Имя целевой колонки. Если None, возможен только 
                алгоритм 'random' (дублирование всей выборки). Для SMOTE/ADASYN 
                и балансировки параметр обязателен.
        Returns:
            pd.DataFrame: Результирующий DataFrame с новыми строками.
        Raises:
            ValueError: Если `target` не указан при включенной балансировке 
                или использовании алгоритмов SMOTE/ADASYN.
        """

        try:
            algo_local = self.algorithm.lower().replace(" ", "_")

            valid_algorithms = ['smote', 'adasyn', 'random']
            if algo_local not in valid_algorithms:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            # ПРОВЕРКА: Если таргет не указан, нельзя использовать балансировку 
            # или синтетические алгоритмы
            if target is None:
                if self.balance or algo_local in ("smote", "adasyn"):
                    raise ValueError(
                        f"The 'target' parameter must be specified for "
                        f"the '{self.algorithm}' algorithm "
                        f"or when balance=True. Without a target, only 'random' " 
                        f"oversampling of the entire dataset is possible." 
                    )
            
            # Если таргет не указан, создаем фиктивный вектор для 
            # совместимости с API imblearn,
            # что позволяет использовать 'random' оверсемплинг 
            # для всей таблицы без разметки.
            if target is None and algo_local == 'random':
                X = data
                # Создаем фиктивный y
                y_array = np.zeros(len(data), dtype=int)
                if len(y_array) > 1:
                    y_array[0] = 1 
                y = pd.Series(y_array, name='temp_target')
            else:
                y = data[target]
                X = data.drop(target, axis=1)

            original_dtypes = X.dtypes
            target_dtype = y.dtype
            # 3. Вызываем публичный метод ресемплирования
            X_res_np, y_res_np = self.fit_resample(X, y)
            # 4. Сборка итогового DataFrame
            res_df = pd.DataFrame(X_res_np, columns=X.columns)
            
            # 5. Восстановление типов данных (Фикс потери типов)
            res_df = self._restore_dtypes(res_df, original_dtypes)
            
            if target is not None:
                res_df[target] = pd.Series(y_res_np).astype(target_dtype)

            return res_df.reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"Oversampling error: {e}", exc_info=True)
            raise

# ------------------------------------------------------------------ #
#  Функциональные интерфейсы                                         #
# ------------------------------------------------------------------ #

def oversample(
    data: pd.DataFrame,
    multiplier: float = 1.0,
    algorithm: str = "random",
    add_noise: bool = False,
    balance: Optional[bool] = False,
    target: Optional[str] = None,
    random_state: Optional[int] = 42,
    noise_level: float = 0.01,
) -> pd.DataFrame:
    """Увеличить объем DataFrame через интерфейс функционального вызова.
    Создает временный экземпляр `DataOversampler` и применяет его к данным.
    Args:
        data (pd.DataFrame): Входной набор данных.
        multiplier (float): Множитель строк. По умолчанию 1.0.
        algorithm (str): Алгоритм ('random', 'smote', 'adasyn'). По умолчанию 'random'.
        add_noise (bool): Флаг добавления шума. По умолчанию False.
        balance (bool, optional): Флаг балансировки классов. По умолчанию False.
        target (str, optional): Имя целевого столбца. По умолчанию None.
        random_state (int, optional): Зерно случайности. По умолчанию 42.
        noise_level (float): Уровень шума. По умолчанию 0.01.
    Returns:
        pd.DataFrame: Расширенный набор данных.
    """
    sampler = DataOversampler(
        multiplier=multiplier,
        algorithm=algorithm,
        add_noise=add_noise,
        balance=balance if balance is not None else False,
        random_state=random_state,
        noise_level=noise_level
    )
    return sampler.oversample(data, target=target)