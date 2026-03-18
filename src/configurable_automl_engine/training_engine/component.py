"""Модуль управления жизненным циклом обучения моделей (Training Engine).
Обеспечивает автоматизированный процесс от валидации входных данных до
сохранения финальной модели. Поддерживает многофазовый поиск гиперпараметров
(HPO), динамическую загрузку алгоритмов и параллельное выполнение вычислений.
Основные компоненты:
    - train_best_model: Публичный интерфейс для запуска полного цикла обучения.
    - _run_hpo: Обертка для поиска гиперпараметров через внешние тюнеры.
    - _fit_and_save: Финальное обучение модели на полном наборе данных.
    - _load_module: Помощник для динамического импорта модулей по пути.
Пример использования:
    results = train_best_model(
        config="path/to/config.yaml",
        df=my_dataframe,
        target="target_col",
        model_path_override="models/best.pkl"
    )
"""

from __future__ import annotations
import importlib
import inspect
import logging
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional
from types import ModuleType

import pandas as pd

from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    ValidationStrategy,
    read_config,
)

from configurable_automl_engine.common.validation_utils import (
    validate_df_not_empty,
    check_target_exists,
    prepare_X_y
)

from .logger import setup_logging

from .metrics import (
    is_greater_better,
    to_sklearn_name,
)
from .thread_pool import run_parallel

from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES

# ───────────────────────── canonical IAE ─────────────────────── #

from ..tuner import InvalidAlgorithmError as _CanonicalIAE

_LOG = logging.getLogger("training_engine")


# --------------------------------------------------------------------------- #
#  Dyn-import helper                                                          #
# --------------------------------------------------------------------------- #
def _load_module(path: str) -> ModuleType:  # noqa: D401
    """Импортировать модуль по указанному пути.
    Динамически загружает Python-модуль, используя dotted-path нотацию. 
    В случае неудачи генерирует информативное исключение.
    Args:
        path (str): Путь к модулю в формате 'package.module'.
    Returns:
        ModuleType: Объект импортированного модуля.
    Raises:
        ImportError: Если модуль не найден по указанному пути.
    """
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
) -> Optional[Tuple[float, Dict[str, Any]]]:
    """Запустить поиск оптимальных гиперпараметров для алгоритма.
    Логика работы:
    1. Загрузка тюнера: Импортируется модуль, указанный в `algo_cfg.tuner`.
    2. Проверка сигнатуры: Метод `optimize` тюнера проверяется на поддержку 
       специфичных параметров (n_folds, oversampling, space_overrides).
    3. Выполнение: Запускается оптимизация. Если алгоритм признан невалидным 
       через `InvalidAlgorithmError`, выбрасывается каноничное исключение.
    Args:
        algo_name (str): Название алгоритма.
        algo_cfg (AlgoCfg): Конфигурация алгоритма с путями к тюнеру.
        X (pd.DataFrame): Матрица признаков.
        y (pd.Series): Вектор целевой переменной.
        metric_name_sklearn (str): Название метрики в формате sklearn.
        n_trials (int): Количество итераций поиска.
        validation_strategy (ValidationStrategy): Стратегия валидации 
            (k_fold, loo и т.д.).
        n_folds (int | None): Количество фолдов для кросс-валидации.
        search_space_override (Dict[str, Any] | None): 
            Переопределенное пространство поиска.
        data_oversampling (bool): Флаг включения оверсэмплинга.
        data_oversampling_multiplier (float): Коэффициент увеличения выборки.
        data_oversampling_algorithm (str): Название алгоритма оверсэмплинга.
    Returns:
        Optional[Tuple[float, Dict[str, Any]]]: 
            Кортеж (лучшая метрика, лучшие параметры) 
            или None, если произошла ошибка при выполнении HPO.
    Raises:
        _CanonicalIAE: Если тюнер сообщает о несовместимости алгоритма с данными.
        AttributeError: Если в модуле тюнера отсутствует функция `optimize`.
    """
    tuner = _load_module(algo_cfg.tuner)
    if not hasattr(tuner, "optimize"):
        raise AttributeError(f"Module {algo_cfg.tuner} lacks `optimize`")

    sig = inspect.signature(tuner.optimize)
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

    # прокидываем oversampling параметры, если tuner их поддерживает
    if "data_oversampling" in sig.parameters:
        kwargs["data_oversampling"] = data_oversampling

    if "data_oversampling_multiplier" in sig.parameters:
        kwargs["data_oversampling_multiplier"] = data_oversampling_multiplier

    if "data_oversampling_algorithm" in sig.parameters:
        kwargs["data_oversampling_algorithm"] = data_oversampling_algorithm

    # прокидываем кастомный search space, если предусмотрен
    if search_space_override is not None:
        # Мы передаем словарь {algo_name: hyperparameters_dict}
        overrides = {algo_name: search_space_override}
        if "space_overrides" in sig.parameters:
            kwargs["space_overrides"] = overrides

    try:
        _model, best_params, best_score = tuner.optimize(**kwargs)
        return best_score, best_params
    except Exception as err:  # noqa: BLE001
        if err.__class__.__name__ == "InvalidAlgorithmError":
            raise _CanonicalIAE(str(err)) from err
        _LOG.error("Algorithm %s failed during HPO: %s", algo_name, err, exc_info=True)
        # возвращаем None, чтобы главный цикл знал, что алгоритм не прошёл
        return None


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
) -> None:
    """Выполнить финальное обучение модели и сохранить результат на диск.
    Args:
        algo_name (str): Название выбранного алгоритма.
        algo_cfg (AlgoCfg): Конфигурация алгоритма с путем к тренеру.
        X (pd.DataFrame): Полная матрица признаков для обучения.
        y (pd.Series): Полный вектор целевой переменной.
        best_params (Dict[str, Any]): Найденные оптимальные гиперпараметры.
        model_path (Path): Путь для сохранения файла модели.
        cfg (Config): Общий объект конфигурации для получения настроек оверсэмплинга.
    Returns:
        None
    Raises:
        AttributeError: Если в модуле тренера отсутствует класс `ModelTrainer`.
    """
    trainer_module = _load_module(algo_cfg.trainer_module)
    if not hasattr(trainer_module, "ModelTrainer"):
        raise AttributeError(
            f"Module {algo_cfg.trainer_module} lacks `ModelTrainer` class"
        )

    trainer = getattr(trainer_module, "ModelTrainer")(
        algorithm=algo_name, 
        hyperparams=best_params,
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
    config: Union[str, Path, Config, Dict [str, Any]],
    df: pd.DataFrame,
    target: str | None = None,
    model_path_override: str | Path | None = None,
)-> Dict[str, Any]:
    """Основной интерфейс обучения лучшей модели.
    Выполняет полный цикл: валидация данных -> многофазовый поиск гиперпараметров (HPO) 
    -> выбор победителя -> финальное обучение -> сохранение.
    Args:
        config (Union[str, Path, Config, Dict[str, Any]]): Конфигурация обучения. 
            Может быть путем к файлу, словарем или объектом Config.
        df (pd.DataFrame): Исходные данные.
        target (str | None): Имя целевого столбца. По умолчанию 'target'.
        model_path_override (str | Path | None): Альтернативный путь сохранения модели.
    Returns:
        Dict[str, Any]: Словарь с результатами: название алгоритма, score, 
            параметры и путь к файлу.
    Raises:
        TypeError: При передаче конфига неподдерживаемого типа.
        RuntimeError: Если ни один алгоритм не смог успешно завершить фазу HPO.
    """
    # Centralized validation
    validate_df_not_empty(df)
    # Определяем имя таргета (приоритет: аргумент функции -> дефолт 'target')
    target_col = target or "target"

    #Проверка наличия таргета до инициализации тяжелых ресурсов
    check_target_exists(df, target_col)

    # Если передана строка или Path, читаем файл. 
    # Если объект Config или dict, обрабатываем их.
    if isinstance(config, Config):
        cfg = config
    elif isinstance(config, dict):
        cfg = Config.model_validate(config)
    elif isinstance(config, (str, Path)):
        cfg = read_config(config)
    else:
        raise TypeError(f"Unsupported config type: {type(config)}. "
                        f"Expected Config, dict, str, or Path.")

    # Если в конфиге указан путь к лог-файлу, настраиваем логирование
    if cfg.general.log_to_file:
        setup_logging(cfg.general.log_to_file)

    metric_user = cfg.general.comparison_metric
    metric_sklearn = to_sklearn_name(metric_user)
    greater_is_better = is_greater_better(metric_sklearn)

    # Centralized splitting
    X, y = prepare_X_y(df, target_col)

    def prepare_search_space(algo_name: str, user_overrides: Dict[str, Any] | None
                             ) -> Dict[str, Any]:
        """Подготовить пространство поиска гиперпараметров.
        Объединяет системные значения по умолчанию с пользовательскими 
        переопределениями из конфигурации.
        Args:
            algo_name (str): Имя алгоритма для поиска дефолтов.
            user_overrides (Dict[str, Any] | None): Словарь с параметрами для замены.
        Returns:
            Dict[str, Any]: Итоговое пространство поиска.
        """
        # Получаем базовый спейс для алгоритма 
        # (копируем, чтобы не менять глобальный объект)
        space = DEFAULT_SPACES.get(algo_name, {}).copy()
        
        if user_overrides:
            # Перезаписываем или добавляем параметры из конфига пользователя
            space.update(user_overrides)
            
        return space

    def _execute_hpo_phase(
            phase_name: str, 
            algo: str, 
            a_cfg: AlgoCfg, 
            n_trials: int, 
            search_space: Dict[str, Any] | None = None
            ) -> Tuple[float, Dict[str, Any]] :
        """Выполнить конкретную фазу HPO для алгоритма.
        Обеспечивает логирование этапа и обработку результатов оверсэмплинга.
        Args:
            phase_name (str): Название текущей фазы (напр. 'coarse').
            algo (str): Имя алгоритма.
            a_cfg (AlgoCfg): Конфигурация алгоритма.
            n_trials (int): Количество итераций в этой фазе.
            search_space (Dict[str, Any] | None): Пространство поиска для фазы.
        Returns:
            Tuple[float, Dict[str, Any]]: Кортеж (метрика, параметры).
        Raises:
            ValueError: Если HPO вернул пустой результат.
        """
        _LOG.info(f"=== {phase_name} phase: {algo} ({n_trials} tries) ===")
        
        # Обращаемся к полям согласно определению в config_parser.py
        ovr = cfg.oversampling 
        
        try:
            result = _run_hpo(
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

            #Проверяем на None перед распаковкой
            if result is None:
                _LOG.warning(f"HPO returned None for {algo} in {phase_name}")
                raise ValueError(f"HPO returned None for {algo} in {phase_name}")

            score, params = result
            
            disp = -score if metric_sklearn == "neg_root_mean_squared_error" else score
            _LOG.info(f"{phase_name} {algo:15} | score {disp:.5f} | params {params}")
            return score, params
        except Exception as err:
            _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
            raise

    # Начальный список кандидатов (все включенные алгоритмы)
    current_candidates = {n: a for n, a in cfg.algorithms.items() if a.enable}
    phase_results: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for phase in cfg.general.phases:
        _LOG.info(f"--- Starting Phase: {phase.name} ({phase.n_trials}"
                  f" trials, action: {phase.action}) ---")
        
        # Если фаза требует только победителя, фильтруем кандидатов
        if phase.action == "refine_winner":
            if not phase_results:
                raise RuntimeError(f"Phase '{phase.name}' requires a winner,"
                                   f" but no previous results exist.")
            
            select = max if greater_is_better else min
            winner_algo = select(phase_results.items(), key=lambda kv: kv[1][0])[0]
            _LOG.info(f"Phase '{phase.name}' filtering for winner: {winner_algo}")
            current_candidates = {winner_algo: cfg.algorithms[winner_algo]}
        # Очищаем результаты для текущей фазы
        phase_results = {}
        
        def _worker(
                algo_name: str, 
                algo_cfg: AlgoCfg
                ) -> Tuple[str, float, Dict[str, Any]] | None:
            """Воркер для параллельного или последовательного запуска задачи HPO.
            Args:
                algo_name (str): Имя алгоритма.
                algo_cfg (AlgoCfg): Конфигурация алгоритма.
            Returns:
                Optional[Tuple[str, float, Dict[str, Any]]]: Название, скор и параметры 
                    или None в случае ошибки.
            """
            # 1. Берем системные дефолты + накладываем то, что в AlgoCfg (из YAML/JSON)
            full_search_space = prepare_search_space(
                algo_name, 
                algo_cfg.hyperparameters # это dict из вашего config_parser
            )
            try:
                score, params = _execute_hpo_phase(
                    phase.name, algo_name, algo_cfg, phase.n_trials, full_search_space
                )
                return algo_name, score, params
            except _CanonicalIAE:
                raise
            except Exception as e:
                _LOG.warning(f"Algorithm {algo_name} failed in phase {phase.name}: {e}")
                return None
        # Выполнение (параллельное или последовательное)
        if (cfg.general.parallel_strategy == "algorithms" 
            and len(current_candidates) > 1):
            results = run_parallel(
                _worker,
                args_seq=[(n, a) for n, a in current_candidates.items()],
                max_workers=cfg.general.max_workers,
                mode=cfg.general.parallel_mode
            )

            for res in results:
                if res:
                    name, sc, pr = res
                    phase_results[name] = (sc, pr)
        else:
            for n, a in current_candidates.items():
                res = _worker(n, a)
                if res:
                    name, sc, pr = res
                    phase_results[name] = (sc, pr)
        valid_results = {
            name: (score, params)
            for name, (score, params) in phase_results.items()
            if score is not None and not math.isnan(score) and score != float("-inf")
        }

        if not valid_results:
            failed_algos = list(current_candidates.keys())
            raise RuntimeError(
                f"No algorithms produced valid scores in phase '{phase.name}'. "
                f"Failed algorithms: {failed_algos}"
            )
    # После завершения всех фаз определяем финального победителя
    select = max if greater_is_better else min
    winner_algo = select(phase_results.items(), key=lambda kv: kv[1][0])[0]
    final_score, final_params = phase_results[winner_algo]
    winner_cfg = cfg.algorithms[winner_algo]

    
    
    
    # ------------------ FINAL FIT & SAVE -------------------------- #
    model_path = Path(model_path_override or cfg.general.path_to_model)

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