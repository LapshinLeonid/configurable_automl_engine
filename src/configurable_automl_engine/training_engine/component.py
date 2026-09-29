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
from types import ModuleType
from typing import Any

import pandas as pd

from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
from configurable_automl_engine.common.validation_utils import (
    check_target_exists,
    prepare_X_y,
    validate_df_not_empty,
)
from configurable_automl_engine.preprocessing import detect_feature_types
from configurable_automl_engine.training_engine.config_parser import (
    AlgoCfg,
    Config,
    ValidationStrategy,
    read_config,
)

# ───────────────────────── canonical IAE ─────────────────────── #
from ..tuner import InvalidAlgorithmError as _CanonicalIAE
from .logger import setup_logging
from .metrics import (
    to_sklearn_name,
)
from .thread_pool import run_parallel

_LOG = logging.getLogger("training_engine")


def _algorithms_as_dict(algorithms_cfg: Any) -> dict[str, AlgoCfg]:
    """Преобразует AlgorithmsConfig в обычный словарь {name: AlgoCfg}."""
    # model_fields через экземпляр тоже работает, но deprecated
    # (PydanticDeprecatedSince211, удаление в V3.0) — используем класс
    return {
        name: algo_cfg
        for name in type(algorithms_cfg).model_fields
        if (algo_cfg := getattr(algorithms_cfg, name)) is not None
    }


# --------------------------------------------------------------------------- #
#  Dyn-import helper                                                          #
# --------------------------------------------------------------------------- #
def _load_module(path: str) -> ModuleType:
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
    search_space_override: dict[str, Any] | None = None,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
    initial_params: dict[str, Any] | None = None,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    encoding: str = "one_hot",
    pruning: dict[str, Any] | None = None,
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: str | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
) -> tuple[float, dict[str, Any]] | None:
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
        initial_params (dict[str, Any] | None): Гиперпараметры из предыдущей фазы HPO
            для enqueue_trial (монотонность улучшения между фазами).
        categorical_features (list[str] | None): Имена категориальных колонок,
            определённые один раз в ``train_best_model`` и передаваемые тюнеру.
        numerical_features (list[str] | None): Имена числовых колонок.
        encoding (str): Стратегия кодирования категорий ('one_hot' или 'ordinal'),
            прокидываемая тюнеру для согласованной предобработки с финальным fit.
        pruning (dict[str, Any] | None): Настройки ранней остановки (pruning)
            из секции ``general.pruning``. Передаются только тюнерам, которые
            поддерживают аргумент ``pruning``; кастомные тюнеры не затрагиваются.
    Returns:
        Optional[Tuple[float, Dict[str, Any]]]:
            Кортеж (лучшая метрика, лучшие параметры)
            или None, если произошла ошибка при выполнении HPO.
    Raises:
        _CanonicalIAE: Если тюнер сообщает о несовместимости алгоритма с данными.
        AttributeError: Если в модуле тюнера отсутствует функция `optimize`.
    """
    if algo_cfg.tuner is None:
        raise ValueError("Tuner path is not configured")
    tuner = _load_module(algo_cfg.tuner)
    if not hasattr(tuner, "optimize"):
        raise AttributeError(f"Module {algo_cfg.tuner} lacks `optimize`")

    sig = inspect.signature(tuner.optimize)
    kwargs: dict[str, Any] = {
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

    # прокидываем initial_params, если есть (для refine_winner phase)
    if initial_params is not None and "initial_params" in sig.parameters:
        kwargs["initial_params"] = initial_params

    # прокидываем детектированные категориальные/числовые колонки,
    # чтобы фаза HPO строила препроцессор согласованно с финальным обучением
    if "categorical_features" in sig.parameters:
        kwargs["categorical_features"] = categorical_features

    if "numerical_features" in sig.parameters:
        kwargs["numerical_features"] = numerical_features

    # прокидываем стратегию кодирования, если тюнер её поддерживает
    if "encoding" in sig.parameters:
        kwargs["encoding"] = encoding
    else:
        # Кастомный тюнер не принимает encoding: стратегия молча игнорируется
        # в фазе HPO, но применяется в финальном обучении -> рассинхрон
        # представления признаков. Предупреждаем, чтобы пользователь мог
        # обновить тюнер до поддерживающего аргумент `encoding`.
        _LOG.warning(
            "Tuner %s does not accept `encoding`; categorical_encoding=%r "
            "will NOT be applied during HPO (final fit may diverge).",
            algo_cfg.tuner,
            encoding,
        )

    # прокидываем переопределение пресета предобработки (FR-5): явное указание
    # пользователя из конфигурации применяется и в HPO, и в финальном обучении
    preprocessing_override = getattr(algo_cfg, "preprocessing", None)
    if "preprocessing_override" in sig.parameters:
        kwargs["preprocessing_override"] = preprocessing_override

    # прокидываем настройки ранней остановки (pruning), если тюнер их
    # поддерживает; кастомные тюнеры без аргумента `pruning` не затрагиваются
    if pruning is not None and "pruning" in sig.parameters:
        kwargs["pruning"] = pruning

    # прокидываем параметры high-cardinality кодирования и новых стратегий
    # (issue #20), если тюнер их поддерживает; согласованно с финальным обучением
    if "high_cardinality_threshold" in sig.parameters:
        kwargs["high_cardinality_threshold"] = high_cardinality_threshold

    if "high_cardinality_encoding" in sig.parameters:
        kwargs["high_cardinality_encoding"] = high_cardinality_encoding

    if "hashing_n_components" in sig.parameters:
        kwargs["hashing_n_components"] = hashing_n_components

    if "target_encoding_smoothing" in sig.parameters:
        kwargs["target_encoding_smoothing"] = target_encoding_smoothing

    if "target_encoding_fallback" in sig.parameters:
        kwargs["target_encoding_fallback"] = target_encoding_fallback

    try:
        _, best_params, best_score = tuner.optimize(**kwargs)
        return best_score, best_params
    except Exception as err:
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
    best_params: dict[str, Any],
    model_path: Path,
    cfg: Config,
    metric_name_sklearn: str = "r2",
) -> Any:
    """Выполнить финальное обучение модели и сохранить результат на диск.
    Args:
        algo_name (str): Название выбранного алгоритма.
        algo_cfg (AlgoCfg): Конфигурация алгоритма с путем к тренеру.
        X (pd.DataFrame): Полная матрица признаков для обучения.
        y (pd.Series): Полный вектор целевой переменной.
        best_params (Dict[str, Any]): Найденные оптимальные гиперпараметры.
        model_path (Path): Путь для сохранения файла модели.
        cfg (Config): Общий объект конфигурации для получения настроек оверсэмплинга.
        metric_name_sklearn (str): Имя основной метрики в формате sklearn.
    Returns:
        Any: Экземпляр ``ModelTrainer`` после обучения и сохранения
            (используется для доступа к значениям дополнительных метрик).
    Raises:
        AttributeError: Если в модуле тренера отсутствует класс `ModelTrainer`.
    """
    if algo_cfg.trainer_module is None:
        raise ValueError("Trainer module path is not configured")
    trainer_module = _load_module(algo_cfg.trainer_module)
    if not hasattr(trainer_module, "ModelTrainer"):
        raise AttributeError(
            f"Module {algo_cfg.trainer_module} lacks `ModelTrainer` class"
        )

    trainer = trainer_module.ModelTrainer(
        algorithm=algo_name,
        hyperparams=best_params,
        metric=metric_name_sklearn,
        # Пробрасываем настройки оверсэмплинга из конфига в тренер
        data_oversampling=cfg.oversampling.enable,
        data_oversampling_multiplier=cfg.oversampling.multiplier,
        data_oversampling_algorithm=cfg.oversampling.algorithm,
        serialization_format=cfg.general.serialization_format,
        encoding_strategy=cfg.general.categorical_encoding,
        additional_metrics=cfg.general.additional_metrics,
        # Явное переопределение пресета предобработки из конфига (FR-5):
        # применяется согласованно с фазой HPO (AC-7).
        preprocessing_override=getattr(algo_cfg, "preprocessing", None),
        # Параметры high-cardinality кодирования и новых стратегий (issue #20)
        high_cardinality_threshold=cfg.general.high_cardinality_threshold,
        high_cardinality_encoding=cfg.general.high_cardinality_encoding,
        hashing_n_components=cfg.general.hashing_n_components,
        target_encoding_smoothing=cfg.general.target_encoding_smoothing,
        target_encoding_fallback=cfg.general.target_encoding_fallback,
    )
    trainer.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(model_path)
    return trainer


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #


def train_best_model(
    config: str | Path | Config | dict[str, Any],
    df: pd.DataFrame,
    target: str | None = None,
    model_path_override: str | Path | None = None,
) -> dict[str, Any]:
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
            параметры и путь к файлу. Если в конфигурации заданы дополнительные
            метрики (``general.additional_metrics``), в результат добавляется
            ключ ``additional_metrics`` — словарь {метрика: значение},
            рассчитанных для финальной модели на том же наборе данных, что и
            основная метрика.
    Raises:
        TypeError: При передаче конфига неподдерживаемого типа.
        RuntimeError: Если ни один алгоритм не смог успешно завершить фазу HPO.
    """
    # Centralized validation
    validate_df_not_empty(df)
    # Определяем имя таргета (приоритет: аргумент функции -> дефолт 'target')
    target_col = target or "target"

    # Проверка наличия таргета до инициализации тяжелых ресурсов
    check_target_exists(df, target_col)

    # Если передана строка или Path, читаем файл.
    # Если объект Config или dict, обрабатываем их.
    if isinstance(config, Config):
        cfg = config
    elif isinstance(config, dict):
        _LOG.debug("CONFIG TYPE:", type(config))
        _LOG.debug(
            "ALGORITHMS:",
            (config.get("algorithms") if isinstance(config, dict) else "N/A"),
        )
        cfg = Config.model_validate(config)
    elif isinstance(config, (str, Path)):
        cfg = read_config(config)
    else:
        raise TypeError(
            f"Unsupported config type: {type(config)}. "
            f"Expected Config, dict, str, or Path."
        )

    # Если в конфиге указан путь к лог-файлу, настраиваем логирование
    if cfg.general.log_to_file:
        setup_logging(cfg.general.log_to_file)

    metric_user = cfg.general.comparison_metric
    metric_sklearn = to_sklearn_name(metric_user)

    # Centralized splitting
    X, y = prepare_X_y(df, target_col)

    # Определяем типы колонок ровно один раз на входном DataFrame и прокидываем
    # в фазу HPO. Это гарантирует одинаковую предобработку (one-hot) между HPO
    # и финальным обучением ModelTrainer.
    categorical_features, numerical_features = detect_feature_types(X)

    def prepare_search_space(
        algo_name: str, user_overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
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
        search_space: dict[str, Any] | None = None,
        initial_params: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]] | None:
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
        pruning_cfg: dict[str, Any] | None = None
        if cfg.general.pruning.enable:
            pruning_cfg = {
                "enable": cfg.general.pruning.enable,
                "strategy": cfg.general.pruning.strategy.value,
                "min_steps": cfg.general.pruning.min_steps,
                "n_startup_trials": cfg.general.pruning.n_startup_trials,
                "reduction_factor": cfg.general.pruning.reduction_factor,
            }

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
                data_oversampling_algorithm=ovr.algorithm.value,  # .value т.к. это Enum
                initial_params=initial_params,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                encoding=cfg.general.categorical_encoding,
                pruning=pruning_cfg,
                high_cardinality_threshold=cfg.general.high_cardinality_threshold,
                high_cardinality_encoding=cfg.general.high_cardinality_encoding,
                hashing_n_components=cfg.general.hashing_n_components,
                target_encoding_smoothing=cfg.general.target_encoding_smoothing,
                target_encoding_fallback=cfg.general.target_encoding_fallback,
            )

            if result is None:
                return None

            score, params = result

            disp = -score if metric_sklearn == "neg_root_mean_squared_error" else score
            _LOG.info(f"{phase_name} {algo:15} | score {disp:.5f} | params {params}")
            return score, params
        except Exception as err:
            _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
            raise

    # Начальный список кандидатов (все включенные алгоритмы)
    all_algorithms = _algorithms_as_dict(cfg.algorithms)
    current_candidates = {n: a for n, a in all_algorithms.items() if a.enable}
    phase_results: dict[str, tuple[float, dict[str, Any]]] = {}
    for phase in cfg.general.phases:
        _LOG.info(
            f"--- Starting Phase: {phase.name} ({phase.n_trials}"
            f" trials, action: {phase.action}) ---"
        )

        # Если фаза требует только победителя, фильтруем кандидатов
        if phase.action == "refine_winner":
            if not phase_results:
                raise RuntimeError(
                    f"Phase '{phase.name}' requires a winner,"
                    f" but no previous results exist."
                )

            select = max
            winner_algo = select(phase_results.items(), key=lambda kv: kv[1][0])[0]
            _LOG.info(f"Phase '{phase.name}' filtering for winner: {winner_algo}")
            current_candidates = {winner_algo: all_algorithms[winner_algo]}

        def _worker(
            algo_name: str, algo_cfg: AlgoCfg, p=phase
        ) -> tuple[str, float, dict[str, Any]] | None:
            """Воркер для параллельного или последовательного запуска задачи HPO.
            Args:
                algo_name (str): Имя алгоритма.
                algo_cfg (AlgoCfg): Конфигурация алгоритма.
            Returns:
                Optional[Tuple[str, float, Dict[str, Any]]]: Название, скор и параметры
                    или None в случае ошибки.
            """
            # Определяем initial_params для refine_winner: сохраняем лучшие
            # параметры предыдущей фазы для enqueue_trial
            init_params = None
            if p.action == "refine_winner" and algo_name in phase_results:
                _, prev_params = phase_results[algo_name]
                init_params = prev_params

            # 1. Берем системные дефолты + накладываем то, что в AlgoCfg (из YAML/JSON)
            full_search_space = prepare_search_space(
                algo_name,
                algo_cfg.hyperparameters,  # это dict из вашего config_parser
            )
            try:
                result = _execute_hpo_phase(
                    p.name,
                    algo_name,
                    algo_cfg,
                    p.n_trials,
                    full_search_space,
                    initial_params=init_params,
                )

                if result is None:
                    return None

                score, params = result

                return algo_name, score, params
            except _CanonicalIAE:
                raise
            except Exception as e:  # noqa: BLE001
                _LOG.warning(f"Algorithm {algo_name} failed in phase {p.name}: {e}")
                return None

        # Выполнение (параллельное или последовательное)
        if (
            cfg.general.parallel_strategy == "algorithms"
            and len(current_candidates) > 1
        ):
            results = run_parallel(
                _worker,
                args_seq=[(n, a) for n, a in current_candidates.items()],
                max_workers=cfg.general.max_workers,
                mode=cfg.general.parallel_mode,
                timeout=cfg.general.phase_timeout or 3600,
                task_timeout=cfg.general.task_timeout,
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
    select = max
    winner_algo = select(phase_results.items(), key=lambda kv: kv[1][0])[0]
    final_score, final_params = phase_results[winner_algo]
    winner_cfg = all_algorithms[winner_algo]

    # ------------------ FINAL FIT & SAVE -------------------------- #
    model_path = Path(model_path_override or cfg.general.path_to_model)

    try:
        trainer = _fit_and_save(
            winner_algo,
            winner_cfg,
            X,
            y,
            final_params,
            model_path,
            cfg,
            metric_name_sklearn=metric_sklearn,
        )
        _LOG.info("Model saved to %s", model_path.resolve())
    except Exception as e:
        _LOG.error(f"Failed to save final model: {e}")
        raise
    result: dict[str, Any] = {
        "algorithm": winner_algo,
        "score": final_score,
        "params": final_params,
        "model_path": str(model_path),
    }
    # Дополнительные информационные метрики финальной модели. Ключ добавляется
    # только при наличии метрик (после дедупликации и исключения основной
    # метрики сравнения), иначе результаты идентичны прежнему поведению.
    if cfg.general.additional_metrics:
        result["additional_metrics"] = dict(trainer.additional_scores)
    return result
