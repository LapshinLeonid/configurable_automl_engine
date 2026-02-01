"""
Hyperparameter optimisation module (CF-17 + CF-18 + CF-19).

• Поддерживает все «классические» алгоритмы из configurable_automl_engine.models,
  где MODELS[key]["use"] is True,
  + SGDRegressor, GaussianProcessRegressor, IsotonicRegression,
    ARDRegression и GLM-семейство (Poisson/Gamma/Tweedie Regressor).
• Нейросети (alias содержит "nn") исключаются автоматически.
• Валидация — train_test_split (80 / 20), k-fold (по умолчанию 5) или
  leave-one-out. Если наблюдений мало для k-fold (< 2 × k), автоматически
  переключаемся на train_test_split.
• По умолчанию метрика R², но можно указать любую
  (sklearn.metrics.get_scorer).
• Логи: logs/hyperopt.log (ro-rotation 10 × 1 MB).
"""
from __future__ import annotations


# ─────────────────────────────── stdlib
import importlib
import logging as _logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

# ──────────────────────────── third-party
import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn import ensemble, metrics, model_selection, neighbors, svm, tree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    ARDRegression,
    ElasticNet,
    GammaRegressor,
    Lasso,
    PoissonRegressor,
    Ridge,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    train_test_split,
)

# ──────────────────────────── project
from configurable_automl_engine.common.definitions import ValidationStrategy

from configurable_automl_engine.validation import norm_val_method,make_cv

from imblearn.pipeline import Pipeline as ImbPipeline
from configurable_automl_engine.oversampling import DataOversampler

# ═════════════════════════════ pseudo-safe logging init ══════════════════════
if not hasattr(_logging, "handlers"):  # edge-case в Jupyter
    sys.modules.pop("logging", None)
    _logging = importlib.import_module("logging")
logging = _logging  # alias

# ═════════════════════════════════════ exceptions ════════════════════════════
class HyperoptError(Exception):
    """Базовая ошибка модуля."""


class InvalidAlgorithmError(HyperoptError):
    """Алгоритм не найден или помечен use=False."""

class InvalidDataError(HyperoptError):
    """Некорректный X / y."""


# ═══════════════════════════════════ logging setup ═══════════════════════════
_ROOT = Path(__file__).resolve().parents[2]
_LOG_FILE = _ROOT / "logs" / "hyperopt.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_handler = RotatingFileHandler(
    _LOG_FILE, maxBytes=1_000_000, backupCount=10, encoding="utf-8", delay=True
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

log = logging.getLogger("hyperopt")
log.setLevel(logging.INFO)
log.propagate = False
if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
    log.addHandler(_handler)

# ═══════════════════════════════════ search spaces ═══════════════════════════

# ══════════ KNN-space зависит от размера выборки ══════════
def _make_knn_space(n_samples: int) -> Callable[[Trial], dict[str, Any]]:
    def _space(t: Trial) -> dict[str, Any]:
        max_k = max(1, min(30, int(n_samples * 0.8)))  # ≥1 и ≤30
        return {
            "n_neighbors": t.suggest_int("n_neighbors", 1, max_k),
            "weights": t.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
            "p": t.suggest_int("p", 1, 2),
        }

    return _space


ALGO_SPACES: dict[str, Callable[[Trial], dict[str, Any]]] = {}

# ═══════════════════════ fallback-фабрика моделей ════════════════════════════
_FALLBACK_FACTORY: dict[str, Any] = {
    "elasticnet": ElasticNet,
    "lasso": Lasso,
    "ridge": Ridge,
    "knn": neighbors.KNeighborsRegressor,
    "decision_tree": tree.DecisionTreeRegressor,
    "random_forest": ensemble.RandomForestRegressor,
    "extra_trees": ensemble.ExtraTreesRegressor,
    "gradient_boosting": ensemble.GradientBoostingRegressor,
    "adaboost": ensemble.AdaBoostRegressor,
    "svr": svm.SVR,
    "sgdregressor": SGDRegressor,
    "gaussianprocessregressor": GaussianProcessRegressor,
    "isotonicregression": IsotonicRegression,
    "ardregression": ARDRegression,
    "poissonregressor": PoissonRegressor,
    "gammaregressor": GammaRegressor,
    "tweedieregressor": TweedieRegressor,
}

try:  # опциональный XGBoost
    import xgboost as _xgb  # type: ignore

    _FALLBACK_FACTORY["xgb"] = _xgb.XGBRegressor
except ModuleNotFoundError:  # pragma: no cover
    pass

# ═════════════════════════════ helper-utilities ═════════════════════════════

def _apply_dynamic_space(trial: Trial, space_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Преобразует словарь из конфига в параметры для модели через trial.suggest_*.
    Поддерживает как SearchSpaceEntry (объекты с полем bounds), так и константы.
    """
    params = {}
    for key, value in space_dict.items():
        # Если это SearchSpaceEntry (у него есть атрибут bounds после валидации Pydantic)
        if hasattr(value, "bounds"):
            b = value.bounds
            low, high = b[0], b[1]
            dist_type = b[2] if len(b) > 2 else "float"
            if dist_type == "int":
                params[key] = trial.suggest_int(key, int(low), int(high))
            elif dist_type == "float":
                params[key] = trial.suggest_float(key, float(low), float(high))
            elif dist_type == "float_log":
                params[key] = trial.suggest_float(key, float(low), float(high), log=True)
            elif dist_type == "categorical":
                # В случае категориального, bounds[0] должен быть списком опций
                params[key] = trial.suggest_categorical(key, low if isinstance(low, list) else b[:-1])
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
    """Сначала ищем в configurable_automl_engine.models.MODELS, иначе fallback."""
    try:
        models_mod = importlib.import_module("configurable_automl_engine.models")
        factory = getattr(models_mod, "MODELS", {})
        if algo in factory and factory[algo].get("use", True):
            return factory[algo]["class"]
    except ModuleNotFoundError:
        pass

    if algo in _FALLBACK_FACTORY:
        return _FALLBACK_FACTORY[algo]

    raise InvalidAlgorithmError(f"Алгоритм «{algo}» не поддерживается")


def _build_scorer(name: str):
    try:
        return metrics.get_scorer(name)
    except ValueError as err:
        raise HyperoptError(f"Неизвестная метрика «{name}»") from err


def _can_stratify(y) -> bool:
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


def _split_train_test(X, y, *, test_size=0.2, random_state=42):
    """train_test_split с попыткой стратификации."""
    strat = y if _can_stratify(y) else None
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=random_state
        )



# ═════════════════════ PUBLIC: optimize() ═══════════════════════════════════
def optimize(
    algo_name: str,
    X,
    y,
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
    oversampling_config = {
        "active": data_oversampling,
        "params": {
            "sampling_strategy": data_oversampling_multiplier,
            "algorithm": data_oversampling_algorithm,
            "random_state": random_state
        }
    }
    
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")
    
    # -------------------- 0. нормализация входа -------------------- #
    if validation_strategy is not None:       # alias имеет приоритет
        val_method = validation_strategy

    algo = algo_name.lower()
    _validate_data(X, y)

    Estimator = _get_estimator(algo)

    # -------------------- 1. стратегия CV -------------------------- #
    n_samples = len(y)
    val_method_eff, cv_obj = make_cv(
        n_samples,
        val_method=val_method,
        n_folds=n_folds,
        random_state=random_state,
    )

    # -------------------- 2. estimator + поисковое пространство ---- #

    if algo == "knn":
        base_space_fn = _make_knn_space(n_samples)
    else:
        base_space_fn = ALGO_SPACES.get(algo)

    # Приоритет 1: Прямые переопределения (функции)
    # Приоритет 2: Динамический конфиг из YAML (dict с SearchSpaceEntry)
    # Приоритет 3: Хардкод из ALGO_SPACES
    external_config = (space_overrides or {}).get(algo)
    
    # Если пришла функция (старый механизм) — используем её
    if callable(external_config):
        space_fn = external_config
    # Если пришел словарь (новый механизм из YAML) — создаем обертку
    elif isinstance(external_config, dict):
        space_fn = lambda t: _apply_dynamic_space(t, external_config)
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
        model = Estimator(**params)

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

        # 2. train / test split (если CV заменён на hold-out)
        if val_method_eff == "train_test_split":
            X_tr, X_te, y_tr, y_te = _split_train_test(
                X, y, test_size=0.2, random_state=random_state
            )
            current_estimator.fit(X_tr, y_tr)
            return float(scorer(current_estimator, X_te, y_te))

        # 3. k-fold или Leave-One-Out
        try:
            scores = model_selection.cross_val_score(
                current_estimator, # Передаем подготовленную обертку
                X,
                y,
                cv=cv_obj,
                scoring=scorer,
                error_score="raise",
                n_jobs=1
            )
            return float(np.mean(scores))
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
            ('model', Estimator(**study.best_params))
        ])
    else:
        best_model = Estimator(**study.best_params)

    best_model.fit(X, y)

    log.info(
        "Hyperopt: algo=%s | val=%s | score=%.5f | params=%s",
        algo,
        val_method_eff,
        best_score,
        best_params,
    )
    
    return best_model, best_params, best_score
