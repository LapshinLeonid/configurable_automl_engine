""" Hyperparameter Optimisation Module: 
Двигатель автоматизированного поиска параметров. 

Модуль обеспечивает высокоуровневую обертку над Optuna для 
автоматического подбора конфигураций моделей с поддержкой 
динамических пространств поиска, кросс-валидации и интегрированного оверсэмплинга.

Ключевые возможности:
    1. Hybrid Model Zoo: Полная поддержка базового пула моделей AutoML-движка 
       с расширением за счет SGD, GaussianProcess, Isotonic и GLM-семейства.
    2. Neural Filter: Автоматическая детекция и исключение нейросетевых архитектур 
       (alias "nn") для оптимизации ресурсов в классическом ML-пайплайне.
    3. Adaptive Validation: Интеллектуальное переключение между k-fold, 
       Leave-One-Out и Train-Test Split (80/20) в зависимости от объема выборки 
       (fallback при n_samples < 2k).
    4. Integrated Oversampling: Бесшовная интеграция балансировки классов через 
       Imbalance-Pipeline прямо внутри процесса оптимизации.
    5. Dynamic Search Spaces: Поддержка как жестко заданных пространств (например, 
       адаптивный KNN-space), так и внешних конфигураций через YAML/SearchSpaceEntry.
    6. Metric Agnostic: Возможность оптимизации по любой стандартной или 
       кастомной метрике Sklearn (по умолчанию R²).
"""


from __future__ import annotations
from functools import partial

# ─────────────────────────────── stdlib
import logging as _logging
from typing import Any, Callable, cast

# ──────────────────────────── third-party
import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial

from sklearn import model_selection

from sklearn.model_selection import (
    train_test_split,
)

# ──────────────────────────── project
from configurable_automl_engine.common.definitions import ValidationStrategy

from configurable_automl_engine.validation import make_cv

from imblearn.pipeline import Pipeline as ImbPipeline
from configurable_automl_engine.oversampling import DataOversampler

from configurable_automl_engine.training_engine.metrics import get_scorer_object

from configurable_automl_engine.models import create_model

from configurable_automl_engine.validation import iter_splits

logging = _logging  # alias

# ═════════════════════════════════════ exceptions ════════════════════════════
class HyperoptError(Exception):
    """Базовая ошибка модуля гиперпараметрической оптимизации."""

class InvalidAlgorithmError(HyperoptError):
    """Ошибка, возникающая, если алгоритм не найден или помечен как неиспользуемый."""

class InvalidDataError(HyperoptError):
    """Ошибка, возникающая при передаче некорректных структур данных X или y."""


# ═══════════════════════════════════ logging setup ═══════════════════════════
log = logging.getLogger(__name__)

# ═══════════════════════════════════ search spaces ═══════════════════════════

# ══════════ KNN-space зависит от размера выборки ══════════
def _make_knn_space(n_samples: int) -> Callable[[Trial], dict[str, Any]]:
    """Создать генератор пространства поиска для алгоритма KNN.
    Args:
        n_samples (int): Количество образцов в выборке 
            для расчета верхнего порога n_neighbors.
    Returns:
        Callable[[Trial], dict[str, Any]]: Функция-обертка, которая принимает объект 
            optuna.Trial и возвращает словарь с предложенными гиперпараметрами.
    """
    def _space(t: Trial) -> dict[str, Any]:
        max_k = int(max(1, min(30, int(n_samples * 0.8))))  # ≥1 и ≤30
        return {
            "n_neighbors": t.suggest_int("n_neighbors", 1, max_k),
            "weights": t.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": t.suggest_int("p", 1, 2),
        }

    return _space

# ═════════════════════════════ helper-utilities ═════════════════════════════

def _apply_dynamic_space(trial: Trial, space_dict: dict[str, Any]) -> dict[str, Any]:
    """Преобразовать конфигурационный словарь в параметры модели через методы Optuna.
    Args:
        trial (Trial): Объект текущей итерации Optuna.
        space_dict (dict[str, Any]): Словарь, содержащий объекты SearchSpaceEntry 
            (с границами и типами распределений) или константные значения.
    Returns:
        dict[str, Any]: Словарь конкретных значений гиперпараметров для данной итерации.
    """
    params: dict[str, Any] = {}
    for key, value in space_dict.items():
        # Если это SearchSpaceEntry 
         # (используем свойства low, high, dist_type, step)
        if hasattr(value, "dist_type"):
            low, high = value.low, value.high
            dist_type = value.dist_type
            step = value.step
            if dist_type == "int":
                low_val, high_val = int(cast(float, low)), int(cast(float, high))
                params[key] = trial.suggest_int(
                    key, 
                    low_val, 
                    high_val,
                    step=int(step) if step is not None else 1
                    )
            elif dist_type == "float":
                params[key] = trial.suggest_float(
                    key, 
                    float(low), 
                    float(high),
                    step=float(step) if step is not None else None
                    )
            elif dist_type == "float_log":
                params[key] = trial.suggest_float(
                    key, 
                    float(low), 
                    float(high), 
                    log=True
                    )
            elif dist_type == "categorical":
                options = low
                params[key] = trial.suggest_categorical(key,options)
        else:
            # Если это просто значение (константа), используем как есть
            params[key] = value
    return params

def _validate_data(X: Any, y: Any) -> None:
    """Проверить типы и размеры входных данных X и y.
    Args:
        X (Any): Признаковое описание (ожидается np.ndarray или pd.DataFrame).
        y (Any): Вектор целевой переменной 
            (ожидается np.ndarray, pd.Series или pd.DataFrame).
    Raises:
        InvalidDataError: Если типы данных не поддерживаются 
            или размеры X и y не совпадают.
    """
    ok_types = (np.ndarray, pd.DataFrame)
    if not isinstance(X, ok_types):
        raise InvalidDataError("X must be a numpy.ndarray or pandas.DataFrame")
    if not isinstance(y, ok_types + (pd.Series,)):
        raise InvalidDataError(
            "y must be a numpy.ndarray, pandas.Series, or pandas.DataFrame"
        )
    if len(X) != len(y):
        raise InvalidDataError(f"Size mismatch: X={len(X)} and y={len(y)}")


def _get_estimator(algo: str) -> Any:
    """Проверить доступность и валидность указанного алгоритма.
    Args:
        algo (str): Название (alias) алгоритма.
    Returns:
        Any: Возвращает True, если модель успешно создается базовой фабрикой.
    Raises:
        InvalidAlgorithmError: Если алгоритм не поддерживается или отсутствует 
            необходимая зависимость (библиотека).
    """
    try:
        # Пробуем создать модель с минимальными параметрами для проверки существования
        create_model(algo)
        return True
    except (ValueError, ImportError) as err:
        # Если в models.py алгоритм не найден или не установлен пакет
        #  (например, XGBoost)
        raise InvalidAlgorithmError(f"Algorithm '{algo}' is not supported: {err}")


def _build_scorer(name: str)-> Any:
    """Создать объект метрики (scorer) по названию.
    Args:
        name (str): Строковое название метрики 
            (например, 'r2' или 'neg_mean_squared_error').
    Returns:
        Any: Объект метрики, совместимый с API sklearn.
    Raises:
        HyperoptError: Если указано неизвестное название метрики.
    """
    try:
        # Используем новый API, который возвращает либо объект make_scorer, либо строку
        return get_scorer_object(name)
    except Exception as err:
        raise HyperoptError(f"Unknown metric name: '{name}'") from err


def _can_stratify(y: Any) -> bool:
    """Определить возможность применения стратификации для вектора y.
    Args:
        y (Any): Вектор целевой переменной.
    Returns:
        bool: True, если данные дискретны (целые числа/bool) и количество уникальных 
            классов не превышает 15. В противном случае — False.
    """
    if isinstance(y, (pd.Series, pd.DataFrame)):
        arr = y.values
    else:
        arr = np.asarray(y)

    if arr.ndim != 1:
        return False
    uniq = np.unique(arr)
    return (
        uniq.size <= 15
        and np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.bool_)
    )


def _split_train_test(
        X: Any, 
        y: Any,
        *,
        test_size: float =0.2,
        random_state: int | None =42
        ) -> tuple[Any, Any, Any, Any]:
    """Разбить данные на обучающую и тестовую выборки с автоматической стратификацией.
    Args:
        X (Any): Матрица признаков.
        y (Any): Вектор ответов.
        test_size (float): Доля тестовой выборки. По умолчанию 0.2.
        random_state (int | None): Зерно генератора случайных чисел. По умолчанию 42.
    Returns:
        tuple[Any, Any, Any, Any]: Кортеж из (X_train, X_test, y_train, y_test).
    """
    strat = y if _can_stratify(y) else None
    try:
        return cast(tuple[Any, Any, Any, Any], train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
            stratify=strat,
        ))
    except ValueError:
        return cast(tuple[Any, Any, Any, Any], train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=random_state
        ))



# ═════════════════════ PUBLIC: optimize() ═══════════════════════════════════
def optimize(
    algo_name: str,
    X:Any,
    y:Any,
    *,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
    metric: str = "r2",
    # старый аргумент оставляем-для-совместимости
    val_method: ValidationStrategy | str = "k_fold",
    # alias, который шлёт training_engine.component
    validation_strategy: ValidationStrategy |str | None = None,
    n_folds: int = 5,
    n_trials: int = 50,
    random_state: int | None = 42,
    train_test_split_test_size: float = 0.2,
    space_overrides: dict[str, Callable[[Trial], dict[str, Any]]] | None = None,
) -> tuple[Any, dict[str, Any], float]:
    """Запустить процесс оптимизации гиперпараметров модели с использованием Optuna.
    Функция автоматически выбирает стратегию валидации, настраивает пространство поиска 
    параметров и обучает финальную модель на всех предоставленных данных.
    Args:
        algo_name (str): Название алгоритма для оптимизации.
        X (Any): Входные признаки.
        y (Any): Целевая переменная.
        data_oversampling (bool): Флаг включения балансировки классов. 
            По умолчанию False.
        data_oversampling_multiplier (float): Коэффициент масштабирования 
            (для оверсэмплинга).
        data_oversampling_algorithm (str): Название алгоритма балансировки 
            (например, 'random', 'smote').
        metric (str): Название метрики для максимизации. По умолчанию 'r2'.
        val_method (ValidationStrategy | str): Метод валидации 
            ('k_fold', 'leave_one_out', 'train_test_split').
        validation_strategy (ValidationStrategy | str | None): Алиас для val_method 
            (имеет приоритет).
        n_folds (int): Количество фолдов для кросс-валидации. По умолчанию 5.
        n_trials (int): Количество итераций поиска (испытаний). По умолчанию 50.
        random_state (int | None): Состояние случайности для воспроизводимости. 
            По умолчанию 42.
        train_test_split_test_size (float): Размер теста для валидации через split. 
            По умолчанию 0.2.
        space_overrides (dict | None): Словарь для переопределения пространств поиска.
    Returns:
        tuple[Any, dict[str, Any], float]: Кортеж, содержащий:
            - best_model: Обученная модель с лучшими параметрами.
            - best_params: Словарь найденных оптимальных гиперпараметров.
            - best_score: Лучшее значение метрики на валидации.
    Raises:
        ValueError: Если n_trials не является положительным целым числом.
        HyperoptError: Если для выбранного алгоритма не определено пространство поиска.
    """
    # --- Формируем конфиг для использования внутри _objective ---
    oversampling_config: dict[str, Any] = {
        "active": data_oversampling,
        "params": {
            "multiplier": data_oversampling_multiplier,
            "algorithm": data_oversampling_algorithm,
        }
    }
    
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")
    
    # -------------------- 0. нормализация входа -------------------- #
    if validation_strategy is not None:       # alias имеет приоритет
        val_method = validation_strategy

    algo = algo_name.lower()
    _validate_data(X, y)

    _get_estimator(algo)

    # -------------------- 1. стратегия CV -------------------------- #
    n_samples = len(y)
    val_method_eff, cv_obj = make_cv(
        n_samples,
        val_method=val_method,
        n_folds=n_folds,
        random_state=random_state,
        test_size= train_test_split_test_size
    )

    # -------------------- 2. estimator + поисковое пространство ---- #
    base_space_fn: Callable[[Trial], dict[str, Any]] | None = None

    if algo == "knn":
        base_space_fn = _make_knn_space(n_samples)

    # Приоритет 1: Прямые переопределения (функции)
    # Приоритет 2: Динамический конфиг из YAML (dict с SearchSpaceEntry)
    external_config = (space_overrides or {}).get(algo)
    
    # Если пришла функция (старый механизм) — используем её
    if callable(external_config):
        space_fn : Callable[[Trial], dict[str, Any]] | None = external_config
    # Если пришел словарь (новый механизм из YAML) — создаем обертку
    elif isinstance(external_config, dict):
        space_fn  = partial(_apply_dynamic_space, space_dict=external_config)
    else:
        space_fn = base_space_fn
    if space_fn is None:
        raise HyperoptError(f"Для «{algo}» нет search-space")

    scorer = _build_scorer(metric)

    # -------------------- 3. objective для Optuna ------------------ #
    def _objective(trial: Trial) -> float:
        """Целевая функция для минимизации/максимизации в Optuna.
    Args:
        trial (Trial): Объект текущего испытания Optuna.
    Returns:
        float: Значение целевой метрики на текущем наборе параметров.
    Raises:
        optuna.TrialPruned: Если в процессе обучения возникла ошибка (ValueError).
    """

        # 1. гиперпараметры и модель
        params = space_fn(trial)
        model = create_model(algo, **params)

        # --- ШАГ 2: ПОДГОТОВКА ОБЕРТКИ (WRAPPER) ---
        if oversampling_config["active"]:
            # Важно: используем DataOversampler
            sampler = DataOversampler(**oversampling_config["params"])
            current_estimator = ImbPipeline([
                ('sampler', sampler),
                ('model', model)
            ])
        else:
            current_estimator = model

        # -------------------------------------------

        if val_method_eff == "train_test_split":
            # Используем iter_splits для унификации
            # Так как это генератор, берем next()
            X_tr, X_te, y_tr, y_te = next(iter_splits(
                X, y, method="train_test_split", 
                test_size=train_test_split_test_size, random_state=random_state
            ))
            current_estimator.fit(X_tr, y_tr)
            return float(scorer(current_estimator, X_te, y_te))
            
        # 3. k-fold или Leave-One-Out
        try:
            # cv_obj здесь гарантированно не None, 
            # так как мы проверили final_method выше
            scores = model_selection.cross_val_score(
                current_estimator,
                X, y,
                cv=cv_obj,
                scoring=scorer,
                n_jobs=1
            )
            avg_score = float(np.mean(scores))
            # Если получили +inf (ошибка в nrmse) или NaN, возвращаем худший float
            if not np.isfinite(avg_score):
                return -3.4028235e+38 # аналог минимального float32 или float('-inf')
            return avg_score
        except ValueError as err:
            trial.set_user_attr("fail_reason", str(err))
            raise optuna.TrialPruned()

    # -------------------- 4. запуск Optuna ------------------------- #
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(_objective, n_trials=n_trials)


    best_params = study.best_params
    best_score = study.best_value

    # --- ФИНАЛЬНЫЙ ЭТАП: Обучение лучшей модели ---
    # Важно: если оверсэмплинг был включен, финальная модель тоже должна его пройти!
    if oversampling_config["active"]:
        best_sampler = DataOversampler(**oversampling_config["params"])
        best_model = ImbPipeline([
            ('sampler', best_sampler),
            ('model', create_model(algo, **study.best_params))
        ])
    else:
        best_model = create_model(algo, **study.best_params)

    best_model.fit(X, y)

    log.info(
        "Hyperopt: algo=%s | val=%s | score=%.5f | params=%s",
        algo,
        val_method_eff,
        best_score,
        best_params,
    )
    
    return best_model, best_params, best_score
