from __future__ import annotations

"""
Training-engine: coarse HPO → accurate HPO → финальный fit & save.

Обновлено:
* поддержан новый параметр `general.n_folds` (кол-во сплитов для k-fold CV).
  Если стратегия валидации -- `k_fold`, значение прокидывается в hyperopt-модули.
* ничего не меняется для `loo` и `train_test_split`.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
from types import ModuleType

import pandas as pd

from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    ValidationStrategy,
    read_config,
)
from .metrics import (
    get_metric,           # noqa: F401  — зарезервировано «на будущее»
    is_greater_better,
    to_sklearn_name,
)
from .thread_pool import run_parallel

from configurable_automl_engine.trainer import ModelTrainer


# ───────────────────────── canonical IAE ─────────────────────── #
from ..hyperopt_module import InvalidAlgorithmError as _CanonicalIAE

_LOG = logging.getLogger("training_engine")


# --------------------------------------------------------------------------- #
#  Dyn-import helper                                                          #
# --------------------------------------------------------------------------- #
def _load_module(path: str) -> ModuleType:  # noqa: D401
    """Импортирует модуль по dotted-path; пробрасывает осмысленный ImportError."""
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(f"Can't import module '{path}'") from exc


# --------------------------------------------------------------------------- #
#  HPO wrapper                                                                #
# --------------------------------------------------------------------------- #
def _run_hpo(
    *,
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    metric_name_sklearn: str,
    n_trials: int,
    validation_strategy: ValidationStrategy,
    n_folds: int | None = None,
    search_space_override: Dict[str, Any] | None = None,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
) -> Tuple[float, Dict[str, Any]]:
    """
    Запускает hyperopt-модуль и возвращает (score, best_params).

    Если стратегия валидации -- k-fold, дополнительно передаём `n_folds`
    в функцию `optimize`, **только** если параметр поддерживается
    сигнатурой целевой функции.
    """
    hyperopt_module = _load_module(algo_cfg.hyperopt_module)
    if not hasattr(hyperopt_module, "optimize"):
        raise AttributeError(f"Module {algo_cfg.hyperopt_module} lacks `optimize`")

    sig = inspect.signature(hyperopt_module.optimize)
    kwargs: Dict[str, Any] = {
        "algo_name": algo_name,
        "X": X,
        "y": y,
        "metric": metric_name_sklearn,
        "n_trials": n_trials,
        "validation_strategy": validation_strategy,
    }

    # прокидываем n_folds, если модуль это умеет
    if (
        validation_strategy == ValidationStrategy.k_fold
        and n_folds is not None
        and "n_folds" in sig.parameters
    ):
        kwargs["n_folds"] = n_folds

    # прокидываем кастомный search space, если предусмотрен
    if search_space_override is not None:
        # Мы передаем словарь {algo_name: hyperparameters_dict}
        overrides = {algo_name: search_space_override}
        if "space_overrides" in sig.parameters:
            kwargs["space_overrides"] = overrides
        elif "search_space_override" in sig.parameters: # для обратной совместимости
            kwargs["search_space_override"] = overrides

    try:
        _model, best_params, best_score = hyperopt_module.optimize(**kwargs)
        return best_score, best_params

    except Exception as err:  # noqa: BLE001
        if err.__class__.__name__ == "InvalidAlgorithmError":
            raise _CanonicalIAE(str(err)) from err
        _LOG.warning("Skip %s: %s", algo_name, err, exc_info=True)
        return float("-inf"), {}


# --------------------------------------------------------------------------- #
#  Final fit & save                                                           #
# --------------------------------------------------------------------------- #
def _fit_and_save(
    algo_name: str,
    algo_cfg: AlgoCfg,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, Any],
    model_path: Path,
    cfg: Config,
):
    trainer_module = _load_module(algo_cfg.trainer_module)
    if not hasattr(trainer_module, "ModelTrainer"):
        raise AttributeError(
            f"Module {algo_cfg.trainer_module} lacks `ModelTrainer` class"
        )

    trainer = getattr(trainer_module, "ModelTrainer")(
        algorithm=algo_name, 
        model_params=best_params,
        # Пробрасываем настройки оверсэмплинга из конфига в тренер
        data_oversampling=cfg.oversampling.enable,
        data_oversampling_multiplier=cfg.oversampling.multiplier,
        data_oversampling_algorithm=cfg.oversampling.algorithm,
        serialization_format=cfg.general.serialization_format
    )
    trainer.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(model_path)

# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
def train_best_model(
    config_path: str | Path,
    df: pd.DataFrame,
    target: str | None = None,
):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise TypeError("Input data must be non-empty pandas.DataFrame")

    cfg: Config = read_config(config_path)

    metric_user = cfg.general.comparison_metric
    metric_sklearn = to_sklearn_name(metric_user)
    greater_is_better = is_greater_better(metric_sklearn)

    target_col = target or "target"

    def _prepare_data(df, target):
        """
        Валидация наличия целевой переменной и разделение на признаки и таргет.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
        
        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    X, y = _prepare_data(df, target_col)

    def _execute_hpo_phase(phase_name, algo, a_cfg, n_trials, search_space=None):
        """
        Универсальная обертка для выполнения фазы HPO с логированием и обработкой метрик.
        """
        _LOG.info(f"=== {phase_name} phase: {algo} ({n_trials} tries) ===")
        
        # Обращаемся к полям согласно определению в config_parser.py
        ovr = cfg.oversampling 
        
        try:
            score, params = _run_hpo(
                algo_name=algo,
                algo_cfg=a_cfg,
                X=X,
                y=y,
                metric_name_sklearn=metric_sklearn,
                n_trials=n_trials,
                validation_strategy=cfg.general.validation_strategy,
                n_folds=cfg.general.n_folds,
                search_space_override=search_space,
                data_oversampling=ovr.enable,
                data_oversampling_multiplier=ovr.multiplier,
                data_oversampling_algorithm=ovr.algorithm.value, # .value т.к. это Enum
            )
            
            disp = -score if metric_sklearn == "neg_root_mean_squared_error" else score
            _LOG.info(f"{phase_name} {algo:15} | score {disp:.5f} | params {params}")
            return score, params
        except Exception as err:
            _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
            raise
    

    # ------------------ COARSE HPO ------------------------------- #
    _LOG.info("=== Coarse HPO phase (%d tries) ===", cfg.general.n_rude_tries)
    coarse_results: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _coarse(algo: str, a_cfg: AlgoCfg):
        if not a_cfg.enable:
            return
        # Используем новую обертку
        override = a_cfg.hyperparameters if a_cfg.hyperparameters else None
        try:
            score, params = _execute_hpo_phase(
                "Coarse HPO", algo, a_cfg, cfg.general.n_rude_tries, override
            )
            coarse_results[algo] = (score, params)
        except _CanonicalIAE:
            raise  # Пробрасываем критическую ошибку для тестов
        except Exception:
            pass # Игнорируем только прочие ошибки (например, сбои обучения конкретной модели)


    if cfg.general.parallel_strategy == "algorithms":
        run_parallel(
            _coarse,
            args_seq=[(n, a) for n, a in cfg.algorithms.items() if a.enable],
            max_workers=cfg.general.max_workers,
            mode=cfg.general.parallel_mode
        )
    else:
        for n, a in cfg.algorithms.items():
            if a.enable:
                _coarse(n, a)

    if not coarse_results:
        raise RuntimeError("No algorithms finished HPO (coarse phase)")

    select = max if greater_is_better else min
    winner_algo = select(coarse_results.items(), key=lambda kv: kv[1][0])[0]
    _LOG.info("Winner after coarse: %s", winner_algo)

    winner_cfg = cfg.algorithms[winner_algo]

    # ------------------ ACCURATE HPO ------------------------------ #
    _LOG.info("=== Accurate HPO phase (%d tries) ===", cfg.general.n_accurate_tries)
    override = (
        winner_cfg.hyperparameters if winner_cfg.limit_hyperparameters else None
    )
    final_score, final_params = _execute_hpo_phase(
        "Accurate HPO", winner_algo, winner_cfg, cfg.general.n_accurate_tries, override
    )
    
    # ------------------ FINAL FIT & SAVE -------------------------- #
    model_path = Path(cfg.general.path_to_model)

    try:
        _fit_and_save(winner_algo,winner_cfg, X,y, final_params, model_path, cfg)
        _LOG.info("Model saved to %s", model_path.resolve())
    except Exception as e:
        _LOG.error(f"Failed to save final model: {e}")
        raise
    return {
        "algorithm": winner_algo,
        "score": final_score,
        "params": final_params,
        "model_path": str(model_path),
    }


