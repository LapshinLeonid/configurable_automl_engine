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
def _space_elasticnet(t: Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _space_lasso(t: Trial) -> dict[str, Any]:
    return {"alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True)}


def _space_ridge(t: Trial) -> dict[str, Any]:
    return {"alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True)}


def _space_decision_tree(t: Trial) -> dict[str, Any]:
    return {
        "max_depth": t.suggest_int("max_depth", 2, 32, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
    }


def _space_random_forest(t: Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": t.suggest_int("max_depth", 2, 32, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "bootstrap": t.suggest_categorical("bootstrap", [True, False]),
    }


def _space_extra_trees(t: Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": t.suggest_int("max_depth", 2, 32, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
    }


def _space_gradient_boosting(t: Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 5),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
    }


def _space_adaboost(t: Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "loss": t.suggest_categorical(
            "loss", ["linear", "square", "exponential"]
        ),
    }


def _space_svr(t: Trial) -> dict[str, Any]:
    return {
        "C": t.suggest_float("C", 1e-2, 100.0, log=True),
        "epsilon": t.suggest_float("epsilon", 1e-3, 1.0, log=True),
        "kernel": t.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        "gamma": t.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _space_xgb(t: Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": t.suggest_float("gamma", 0.0, 5.0),
    }


def _space_sgd(t: Trial) -> dict[str, Any]:
    return {
        "loss": t.suggest_categorical(
            "loss",
            [
                "squared_error",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
        ),
        "penalty": t.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
        "alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate": t.suggest_categorical(
            "learning_rate",
            ["constant", "optimal", "invscaling", "adaptive"],
        ),
        "eta0": t.suggest_float("eta0", 1e-4, 1e-1, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
        "max_iter": t.suggest_int("max_iter", 500, 5000, step=500),
    }


def _space_gpr(t: Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-12, 1e-3, log=True),
        "n_restarts_optimizer": t.suggest_int("n_restarts_optimizer", 0, 10),
        "normalize_y": t.suggest_categorical("normalize_y", [True, False]),
    }


def _space_isotonic(t: Trial) -> dict[str, Any]:
    return {
        "increasing": t.suggest_categorical("increasing", [True, False]),
        "out_of_bounds": t.suggest_categorical(
            "out_of_bounds", ["nan", "clip", "raise"]
        ),
    }


def _space_ard(t: Trial) -> dict[str, Any]:
    return {
        "max_iter": t.suggest_int("max_iter", 300, 1500, step=100),
        "alpha_1": t.suggest_float("alpha_1", 1e-6, 1e-1, log=True),
        "alpha_2": t.suggest_float("alpha_2", 1e-6, 1e-1, log=True),
        "lambda_1": t.suggest_float("lambda_1", 1e-6, 1e-1, log=True),
        "lambda_2": t.suggest_float("lambda_2", 1e-6, 1e-1, log=True),
    }


def _space_glm_common(t: Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        "max_iter": t.suggest_int("max_iter", 50, 1000, step=50),
    }


def _space_poisson(t: Trial) -> dict[str, Any]:
    return _space_glm_common(t)


def _space_gamma(t: Trial) -> dict[str, Any]:
    return _space_glm_common(t)


def _space_tweedie(t: Trial) -> dict[str, Any]:
    d = _space_glm_common(t)
    d["power"] = t.suggest_float("power", 1.0, 1.9)
    d["link"] = t.suggest_categorical("link", ["auto", "identity", "log"])
    return d


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


ALGO_SPACES: dict[str, Callable[[Trial], dict[str, Any]]] = {
    "elasticnet": _space_elasticnet,
    "lasso": _space_lasso,
    "ridge": _space_ridge,
    # knn — динамический, см. optimize()
    "decision_tree": _space_decision_tree,
    "random_forest": _space_random_forest,
    "extra_trees": _space_extra_trees,
    "gradient_boosting": _space_gradient_boosting,
    "adaboost": _space_adaboost,
    "svr": _space_svr,
    "xgb": _space_xgb,
    "sgdregressor": _space_sgd,
    "gaussianprocessregressor": _space_gpr,
    "isotonicregression": _space_isotonic,
    "ardregression": _space_ard,
    "poissonregressor": _space_poisson,
    "gammaregressor": _space_gamma,
    "tweedieregressor": _space_tweedie,
}

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

    space_fn = (space_overrides or {}).get(algo) or base_space_fn
    if space_fn is None:
        raise HyperoptError(f"Для «{algo}» нет search-space")

    scorer = _build_scorer(metric)

    # -------------------- 3. objective для Optuna ------------------ #
    def _objective(trial: Trial) -> float:
        """Возвращает средний nrmse (меньше — лучше, Optuna → maximize)."""

        # 1. гиперпараметры и модель
        params = space_fn(trial)
        model = Estimator(**params)

        # 2. train / test split (если CV заменён на hold-out)
        if val_method_eff == "train_test_split":
            X_tr, X_te, y_tr, y_te = _split_train_test(
                X, y, test_size=0.2, random_state=random_state
            )
            model.fit(X_tr, y_tr)
            return float(scorer(model, X_te, y_te))

        # 3. k-fold или Leave-One-Out
        try:
            scores = model_selection.cross_val_score(
                model,
                X,
                y,
                cv=cv_obj,
                scoring=scorer,
                n_jobs=-1,
                error_score="raise",       # ← не затираем NaN/Inf
            )
            return float(np.mean(scores))

        except ValueError as err:
            # Деление на 0, non-finite predictions, «all fits failed» и т.д.
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
    best_model = Estimator(**best_params).fit(X, y)

    log.info(
        "Hyperopt: algo=%s | val=%s | score=%.5f | params=%s",
        algo,
        val_method_eff,
        best_score,
        best_params,
    )
    
    return best_model, best_params, best_score
