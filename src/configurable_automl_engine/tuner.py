"""
Hyperparameter optimisation module

• Поддерживает все «классические» алгоритмы из configurable_automl_engine.models,
  где MODELS[key]["use"] is True,
  + SGDRegressor, GaussianProcessRegressor, IsotonicRegression,
    ARDRegression и GLM-семейство (Poisson/Gamma/Tweedie Regressor).
• Нейросети (alias содержит "nn") исключаются автоматически.
• Валидация — train_test_split (80 / 20), k-fold (по умолчанию 5) или
  leave-one-out. Если наблюдений мало для k-fold (< 2 × k), автоматически
  переключаемся на train_test_split.
• По умолчанию метрика R², но можно указать любую
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
    """Базовая ошибка модуля."""


class InvalidAlgorithmError(HyperoptError):
    """Алгоритм не найден или помечен use=False."""

class InvalidDataError(HyperoptError):
    """Некорректный X / y."""


# ═══════════════════════════════════ logging setup ═══════════════════════════
log = logging.getLogger(__name__)

# ═══════════════════════════════════ search spaces ═══════════════════════════

# ══════════ KNN-space зависит от размера выборки ══════════
def _make_knn_space(n_samples: int) -> Callable[[Trial], dict[str, Any]]:
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


ALGO_SPACES: dict[str, Callable[[Trial], dict[str, Any]]] = {}

# ═════════════════════════════ helper-utilities ═════════════════════════════

def _apply_dynamic_space(trial: Trial, space_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Преобразует словарь из конфига в параметры для модели через trial.suggest_*.
    Поддерживает как SearchSpaceEntry (объекты с полем bounds), так и константы.
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
    ok_types = (np.ndarray, pd.DataFrame)
    if not isinstance(X, ok_types):
        raise InvalidDataError("X должен быть numpy.ndarray или pandas.DataFrame")
    if not isinstance(y, ok_types + (pd.Series,)):
        raise InvalidDataError(
            "y должен быть numpy.ndarray, pandas.Series или DataFrame"
        )
    if len(X) != len(y):
        raise InvalidDataError(f"Размеры не совпадают: X={len(X)} y={len(y)}")


def _get_estimator(algo: str) -> Any:
    """
    Проверяет валидность алгоритма через models.py.
    Возвращает True, если алгоритм поддерживается, иначе кидает исключение.
    """
    try:
        # Пробуем создать модель с минимальными параметрами для проверки существования
        create_model(algo)
        return True
    except (ValueError, ImportError) as err:
        # Если в models.py алгоритм не найден или не установлен пакет
        #  (например, XGBoost)
        raise InvalidAlgorithmError(f"Алгоритм «{algo}» не поддерживается: {err}")


def _build_scorer(name: str)-> Any:
    try:
        # Используем новый API, который возвращает либо объект make_scorer, либо строку
        return get_scorer_object(name)
    except Exception as err:
        raise HyperoptError(f"Неизвестная метрика «{name}»") from err


def _can_stratify(y: Any) -> bool:
    """Можно ли безопасно stratify=y в train_test_split?"""
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
    """train_test_split с попыткой стратификации."""
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
    """
    Подбор гиперпараметров через Optuna.

    Возвращает:
        best_model  – обученный estimator с лучшими параметрами
        best_params – словарь лучших гиперпараметров
        best_score  – метрика на валидации
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

    if algo == "knn":
        base_space_fn: (
            Callable[[Trial], dict[str, Any]] | None
            ) = _make_knn_space(n_samples)
    else:
        base_space_fn = ALGO_SPACES.get(algo)

    # Приоритет 1: Прямые переопределения (функции)
    # Приоритет 2: Динамический конфиг из YAML (dict с SearchSpaceEntry)
    # Приоритет 3: Хардкод из ALGO_SPACES
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
        """Возвращает средний nrmse (меньше — лучше, Optuna → maximize)."""

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
                test_size=0.2, random_state=random_state
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
