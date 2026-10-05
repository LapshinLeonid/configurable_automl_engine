"""Модуль управления жизненным циклом обучения моделей (Training Engine).
Обеспечивает автоматизированный процесс от валидации входных данных до
сохранения финальной модели. Поддерживает многофазовый поиск гиперпараметров
(HPO), динамическую загрузку алгоритмов и параллельное выполнение вычислений.
Публичный цикл обучения декомпозирован на шаги (issue #31):
    - load_config: Загрузка и валидация конфигурации и входных данных.
    - prepare_dataset: Подготовка X/y, резолюция валидации и типов колонок.
    - execute_phases: Многофазовый поиск гиперпараметров (HPO).
    - select_winner: Детерминированный выбор алгоритма-победителя.
    - persist_artifact: Финальное обучение, сохранение модели и сборка отчёта.
Основные компоненты:
    - train_best_model: Публичный интерфейс — оркестрация перечисленных шагов.
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
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
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
    FeatureSelectionCfg,
    FeatureSelectionMode,
    HPOPhaseCfg,
    ValidationStrategy,
    read_config,
)
from configurable_automl_engine.validation import RANDOM_STATE, make_cv

# ───────────────────────── canonical IAE ─────────────────────── #
from ..tuner import WORST_SCORE_THRESHOLD
from ..tuner import InvalidAlgorithmError as _CanonicalIAE
from .logger import setup_logging
from .metrics import (
    to_sklearn_name,
    to_user_value,
)
from .thread_pool import run_parallel

_LOG = logging.getLogger("training_engine")


def is_valid_winner_score(score: Any) -> bool:
    """Проверить, что «сырое» значение скора может быть скором победителя.

    Валидный скор победителя обязан быть:
    - числом (int/float/np.floating): None, строки (в т.ч. числовые, например
      ``"0.5"``) и прочие «мусорные» типы от кастомных тюнеров не проходят
      (issue #32);
    - конечным (NaN и ±inf — сигналы отсутствия валидной метрики);
    - строго выше класса worst-score сентинела: любой скор <= точного
      float32-минимума (``WORST_SCORE_THRESHOLD``) трактуется как сентинел.
      Порог покрывает и константу ``HPO_WORST_SCORE`` (возвращается тюнером
      для нефинитных метрик), и «сырое» значение ``float(np.finfo(np.float32).min)``
      (возможно от numpy-cast метрик или кастомных тюнеров), которое старый
      фильтр через ``math.isclose(rel_tol=1e-9)`` пропускал (относительная
      разница ≈ 9.88e-9 > 1e-9). Реальные метрики такой величины практически
      невозможны, поэтому ложный отсев исключён.

    Args:
        score (Any): «Сырое» значение скора из фазы HPO.

    Returns:
        bool: True, если скор может попасть в ``result["score"]``.
    """
    if not isinstance(score, (int, float, np.floating)):
        return False
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > WORST_SCORE_THRESHOLD


def select_winner(results: dict[str, tuple[float, dict[str, Any]]]) -> str:
    """Детерминированно выбрать алгоритм-победителя по «сырому» скору.

    Побеждает алгоритм с максимальным значением скора (семантика
    оптимизатора — «больше лучше»). Ничьи разрешаются порядком итерации по
    ``results``: ``max`` стабилен, а словарь сохраняет порядок вставки —
    для последовательного пути это порядок конфигурации, поэтому при равных
    скорах побеждает первый настроенный алгоритм (задокументированный
    tie-break, issue #32).

    Args:
        results (dict[str, tuple[float, dict[str, Any]]]): Словарь
            «алгоритм → (скор, параметры)» текущей фазы. Не должен быть пустым.

    Returns:
        str: Имя алгоритма-победителя.

    Raises:
        ValueError: Если ``results`` пуст — победитель не существует.
    """
    if not results:
        raise ValueError("Cannot select a winner from empty results")
    return max(results.items(), key=lambda kv: kv[1][0])[0]


def _algorithms_as_dict(algorithms_cfg: Any) -> dict[str, AlgoCfg]:
    """Преобразует AlgorithmsConfig в обычный словарь {name: AlgoCfg}."""
    # model_fields через экземпляр тоже работает, но deprecated
    # (PydanticDeprecatedSince211, удаление в V3.0) — используем класс
    return {
        name: algo_cfg
        for name in type(algorithms_cfg).model_fields
        if (algo_cfg := getattr(algorithms_cfg, name)) is not None
    }


def _feature_selection_mode(
    cfg: FeatureSelectionCfg | dict[str, Any] | None,
) -> str | None:
    """Извлечь строковое значение ``mode`` из конфигурации отбора признаков.

    Args:
        cfg: Объект ``FeatureSelectionCfg``, словарь конфигурации или ``None``.

    Returns:
        str | None: Режим ('disabled'/'always'/'auto') либо ``None``, если
            конфигурация не задана или не содержит режима.
    """
    if isinstance(cfg, FeatureSelectionCfg):
        return cfg.mode.value
    if isinstance(cfg, dict):
        mode = cfg.get("mode")
        if isinstance(mode, FeatureSelectionMode):
            return mode.value
        return mode if isinstance(mode, str) else None
    return None


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
    train_test_split_test_size: float = 0.2,
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
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
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
        train_test_split_test_size (float): Размер hold-out части для стратегии
            'train_test_split' (доля в (0, 1) либо целое число строк после
            резолюции 'auto'). Прокидывается тюнеру, поддерживающему параметр,
            чтобы оценка HPO не расходилась с финальным val_score (D3 issue #24).
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
        feature_selection_cfg (FeatureSelectionCfg | dict | None): Конфигурация
            отбора признаков из корневого ``Config.general.feature_selection``.
            Передаётся только тюнерам, которые поддерживают аргумент
            ``feature_selection_cfg`` или принимают ``**kwargs`` (сигнатура
            проверяется через ``inspect.signature``); кастомные тюнеры не
            затрагиваются.
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
    # Сигнатура с **kwargs считается совместимой (аналогично _fit_and_save):
    # тюнер сам решает, что делать с лишними ключами, поэтому служебные
    # аргументы отбора признаков можно безопасно прокидывать.
    tuner_accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
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

    # прокидываем разрешённый размер hold-out (issue #24, D3): при 'auto' →
    # 'train_test_split' choose_validation_method возвращает целое число строк,
    # которое обязано совпадать и в HPO, и в финальном fit.
    if "train_test_split_test_size" in sig.parameters:
        kwargs["train_test_split_test_size"] = train_test_split_test_size

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

    # прокидываем конфигурацию отбора признаков (issue #11), если тюнер
    # поддерживает аргумент `feature_selection_cfg` или принимает **kwargs;
    # кастомные тюнеры без поддержки не затрагиваются. При активном режиме
    # отбора предупреждаем о рассинхроне HPO ↔ финальный fit (аналогично
    # encoding): конфигурация не будет применена в фазе поиска.
    if "feature_selection_cfg" in sig.parameters or tuner_accepts_var_kwargs:
        kwargs["feature_selection_cfg"] = feature_selection_cfg
    elif _feature_selection_mode(feature_selection_cfg) not in (None, "disabled"):
        _LOG.warning(
            "Tuner %s does not accept `feature_selection_cfg`; feature "
            "selection mode=%r will NOT be applied during HPO "
            "(final fit may diverge).",
            algo_cfg.tuner,
            _feature_selection_mode(feature_selection_cfg),
        )

    try:
        _, best_params, best_score = tuner.optimize(**kwargs)
        if best_score is None or best_params is None:
            # Tuner couldn't get a valid result: either all trials failed
            # (optuna.TrialPruned) or it failed otherwise. Return None so the
            # main loop knows the algorithm didn't pass (issue #13): the dummy
            # candidate with score -3.4e38 and params=None should no longer
            # end up in phase_results.
            _LOG.error(
                "Algorithm %s failed during HPO: no valid result "
                "(best_score=%r, best_params=%r)",
                algo_name,
                best_score,
                best_params,
            )
            return None
        return best_score, best_params
    except Exception as err:
        if err.__class__.__name__ == "InvalidAlgorithmError":
            raise _CanonicalIAE(str(err)) from err
        _LOG.error("Algorithm %s failed during HPO: %s", algo_name, err, exc_info=True)
        # Return None so the main loop knows the algorithm didn't pass.
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
    *,
    validation_strategy: ValidationStrategy | str | None = None,
    n_folds: int | None = None,
    test_size: float | None = None,
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
        validation_strategy (ValidationStrategy | str | None): Разрешённая
            стратегия валидации финальной модели (issue #24, D3). ``None`` —
            дефолт ``ModelTrainer`` ('train_test_split').
        n_folds (int | None): Число фолдов (для 'k_fold').
        test_size (float | int | None): Размер hold-out части (доля либо число
            строк после резолюции 'auto').
    Returns:
        Any: Экземпляр ``ModelTrainer`` после обучения и сохранения
            (используется для доступа к значениям дополнительных метрик).
    Raises:
        AttributeError: Если в модуле тренера отсутствует класс `ModelTrainer`.
        ValueError: If ``best_params`` is ``None`` — a sign that HPO returned
            no valid hyperparameters (protection against the TypeError
            'NoneType' object is not iterable, issue #13).
    """
    if algo_cfg.trainer_module is None:
        raise ValueError("Trainer module path is not configured")
    if best_params is None:
        raise ValueError(
            f"Algorithm '{algo_name}' produced no hyperparameters "
            "(best_params is None): HPO phase failed completely. The algorithm "
            "should have been excluded before the final fit."
        )
    trainer_module = _load_module(algo_cfg.trainer_module)
    if not hasattr(trainer_module, "ModelTrainer"):
        raise AttributeError(
            f"Module {algo_cfg.trainer_module} lacks `ModelTrainer` class"
        )

    # Решение победителя HPO по отбору признаков (issue #11): в режиме
    # 'auto' ключ "use_feature_selection" зафиксирован в best_params Optuna;
    # в режимах 'always'/'disabled' ключа нет -> None, и ModelTrainer сам
    # разрешает активность по конфигурации. Служебный ключ извлекается из
    # копии: исходный best_params (он же возвращается в result["params"]) не
    # мутируется, а "use_feature_selection" гарантированно не попадает в
    # гиперпараметры ModelTrainer (раньше выпадал только неявно через
    # clean_hyperparameters).
    trainer_params = dict(best_params)
    fs_active = trainer_params.pop("use_feature_selection", None)

    # Проверяем сигнатуру ModelTrainer (аналогично тюнерам в _run_hpo):
    # кастомные тренеры без поддержки отбора признаков не должны получать
    # новые аргументы (иначе TypeError). Сигнатура с **kwargs считается
    # совместимой — тренер сам решает, что делать с лишними ключами.
    trainer_sig = inspect.signature(trainer_module.ModelTrainer)
    trainer_accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_sig.parameters.values()
    )
    trainer_has_fs_cfg = "feature_selection_cfg" in trainer_sig.parameters
    trainer_has_fs_active = "feature_selection_active" in trainer_sig.parameters

    trainer_kwargs: dict[str, Any] = {
        "algorithm": algo_name,
        "hyperparams": trainer_params,
        "metric": metric_name_sklearn,
        # Пробрасываем настройки оверсэмплинга из конфига в тренер
        "data_oversampling": cfg.oversampling.enable,
        "data_oversampling_multiplier": cfg.oversampling.multiplier,
        "data_oversampling_algorithm": cfg.oversampling.algorithm,
        "serialization_format": cfg.general.serialization_format,
        "encoding_strategy": cfg.general.categorical_encoding,
        "additional_metrics": cfg.general.additional_metrics,
        # Явное переопределение пресета предобработки из конфига (FR-5):
        # применяется согласованно с фазой HPO (AC-7).
        "preprocessing_override": getattr(algo_cfg, "preprocessing", None),
        # Параметры high-cardinality кодирования и новых стратегий (issue #20)
        "high_cardinality_threshold": cfg.general.high_cardinality_threshold,
        "high_cardinality_encoding": cfg.general.high_cardinality_encoding,
        "hashing_n_components": cfg.general.hashing_n_components,
        "target_encoding_smoothing": cfg.general.target_encoding_smoothing,
        "target_encoding_fallback": cfg.general.target_encoding_fallback,
    }
    # Отбор признаков: конфигурация из корневого Config + решение
    # победителя HPO (issue #11). Аргументы передаются только тренерам,
    # которые их поддерживают; при активном режиме отбора предупреждаем
    # о молчаливом игнорировании.
    if trainer_has_fs_cfg or trainer_accepts_var_kwargs:
        trainer_kwargs["feature_selection_cfg"] = cfg.general.feature_selection
    elif _feature_selection_mode(cfg.general.feature_selection) not in (
        None,
        "disabled",
    ):
        _LOG.warning(
            "ModelTrainer %s does not accept `feature_selection_cfg`; "
            "feature selection mode=%r will NOT be applied in the final fit.",
            algo_cfg.trainer_module,
            _feature_selection_mode(cfg.general.feature_selection),
        )

    if trainer_has_fs_active or trainer_accepts_var_kwargs:
        trainer_kwargs["feature_selection_active"] = fs_active
    elif fs_active is not None:
        _LOG.warning(
            "ModelTrainer %s does not accept `feature_selection_active`; "
            "the HPO decision use_feature_selection=%r will NOT be applied "
            "in the final fit.",
            algo_cfg.trainer_module,
            fs_active,
        )

    # Параметры валидации (issue #24, D3): в финальный fit передаются
    # разрешённые значения (метод + число фолдов/размер hold-out), а не строка
    # 'auto', чтобы финальный val_score считался тем же методом, что и оценка
    # сравнения моделей в HPO. Кастомным тренерам без поддержки аргументов
    # новые ключи не передаются (сигнатура проверяется, как для отбора
    # признаков); поведение по умолчанию ModelTrainer при этом совпадает.
    if validation_strategy is not None:
        if (
            "validation_strategy" in trainer_sig.parameters
            or trainer_accepts_var_kwargs
        ):
            trainer_kwargs["validation_strategy"] = validation_strategy
        else:
            _LOG.warning(
                "ModelTrainer %s does not accept `validation_strategy`; "
                "the resolved strategy %r will NOT be applied in the final fit.",
                algo_cfg.trainer_module,
                validation_strategy,
            )
    if n_folds is not None and (
        "n_folds" in trainer_sig.parameters or trainer_accepts_var_kwargs
    ):
        trainer_kwargs["n_folds"] = n_folds
    if test_size is not None and (
        "test_size" in trainer_sig.parameters or trainer_accepts_var_kwargs
    ):
        trainer_kwargs["test_size"] = test_size

    trainer = trainer_module.ModelTrainer(**trainer_kwargs)
    trainer.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(model_path)
    return trainer


# --------------------------------------------------------------------------- #
#  Orchestration helpers (issue #31)                                          #
#  Шаги пайплайна, вынесенные из монолитного ``train_best_model``:           #
#  load_config → prepare_dataset → execute_phases → select_winner →          #
#  persist_artifact.                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreparedData:
    """Подготовленные данные и разрешённые настройки для фаз HPO.

    Единый контейнер, который ``prepare_dataset`` создаёт ровно один раз,
    а ``execute_phases`` / ``persist_artifact`` потребляют. Гарантирует, что
    все этапы пайплайна используют одинаковое представление признаков,
    метрик и стратегии валидации (issue #24, D3) без дублирования
    вычислений.

    Attributes:
        X (pd.DataFrame): Матрица признаков без целевой колонки.
        y (pd.Series): Вектор целевой переменной.
        metric_user (str): Пользовательское имя метрики сравнения
            (``general.comparison_metric``).
        metric_sklearn (str): Имя метрики в sklearn-семантике
            (например, ``neg_root_mean_squared_error``).
        resolved_validation (ValidationStrategy): Разрешённая стратегия
            валидации: 'auto' уже раскрыта в конкретный метод
            ('k_fold'/'train_test_split'/'loo').
        resolved_n_folds (int): Число фолдов (актуально для 'k_fold').
        resolved_test_size (float | int): Размер hold-out части — доля либо
            целое число строк после резолюции 'auto'.
        categorical_features (list[str]): Детектированные категориальные
            колонки.
        numerical_features (list[str]): Детектированные числовые колонки.
    """

    X: pd.DataFrame
    y: pd.Series
    metric_user: str
    metric_sklearn: str
    resolved_validation: ValidationStrategy
    resolved_n_folds: int
    resolved_test_size: float | int
    categorical_features: list[str]
    numerical_features: list[str]


def load_config(
    config: str | Path | Config | dict[str, Any],
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[Config, str]:
    """Загрузить и провалидировать конфигурацию запуска.

    Шаг 1 пайплайна (issue #31). Проверяет входные данные и целевую колонку,
    приводит конфигурацию к объекту ``Config`` (из файла, словаря или уже
    готового объекта) и при необходимости настраивает файловое логирование.

    Args:
        config (Union[str, Path, Config, Dict[str, Any]]): Конфигурация
            обучения.
        df (pd.DataFrame): Исходные данные.
        target (str | None): Имя целевого столбца. По умолчанию 'target'.

    Returns:
        Tuple[Config, str]: Кортеж (объект ``Config``, имя целевой колонки).

    Raises:
        TypeError: При передаче конфига неподдерживаемого типа.
        ValueError: Если DataFrame пуст или целевая колонка отсутствует.
    """
    # Centralized validation входных данных до инициализации тяжёлых ресурсов
    validate_df_not_empty(df)
    target_col = target or "target"
    check_target_exists(df, target_col)

    # Если передана строка или Path, читаем файл.
    # Если объект Config или dict, обрабатываем их.
    if isinstance(config, Config):
        cfg = config
    elif isinstance(config, dict):
        _LOG.debug("CONFIG TYPE: %s", type(config))
        _LOG.debug(
            "ALGORITHMS: %s",
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
    return cfg, target_col


def prepare_dataset(cfg: Config, df: pd.DataFrame, target_col: str) -> PreparedData:
    """Подготовить данные и разрешить настройки для фаз HPO.

    Шаг 2 пайплайна (issue #31). Разделяет DataFrame на X/y, резолвит
    стратегию валидации ('auto' → конкретный метод/фолды/test_size, issue
    #24, D3) и детектирует типы колонок ровно один раз — гарантия
    согласованной предобработки между HPO и финальным обучением.

    Args:
        cfg (Config): Объект конфигурации.
        df (pd.DataFrame): Исходные данные.
        target_col (str): Имя целевой колонки.

    Returns:
        PreparedData: Подготовленный контейнер данных и настроек.
    """
    metric_user = cfg.general.comparison_metric
    metric_sklearn = to_sklearn_name(metric_user)

    # Centralized splitting
    X, y = prepare_X_y(df, target_col)

    # Единая резолюция стратегии валидации (issue #24, D3): 'auto' разрешается
    # ровно один раз по сырым X/y — до фаз HPO. Разрешённые значения (метод +
    # число фолдов / размер hold-out) передаются и в HPO, и в финальный fit,
    # чтобы оценка сравнения моделей и финальный val_score не расходились
    # (например, auto → train_test_split с целочисленным test_size из
    # choose_validation_method, а не фиксированным 0.2).
    val_method_eff, cv_obj, auto_decision = make_cv(
        n_samples=len(X),
        val_method=cfg.general.validation_strategy,
        n_folds=cfg.general.n_folds,
        random_state=RANDOM_STATE,
        test_size=0.2,
        n_features=X.shape[1],
    )
    resolved_validation = ValidationStrategy(val_method_eff)
    if val_method_eff == "k_fold":
        # Число фолдов берём из готового cv_obj: для 'auto'→kfold он содержит
        # k из решения choose_validation_method, отличное от cfg.general.n_folds.
        assert cv_obj is not None
        resolved_n_folds = int(cv_obj.get_n_splits())
    else:
        resolved_n_folds = cfg.general.n_folds
    # Для auto → train_test_split choose_validation_method возвращает целое
    # число строк; для явной стратегии — фиксированная доля 0.2.
    if val_method_eff == "train_test_split" and auto_decision is not None:
        resolved_test_size: float | int = int(auto_decision["test_size"])
    else:
        resolved_test_size = 0.2

    # Определяем типы колонок ровно один раз на входном DataFrame и прокидываем
    # в фазу HPO. Это гарантирует одинаковую предобработку (one-hot) между HPO
    # и финальным обучением ModelTrainer.
    categorical_features, numerical_features = detect_feature_types(X)

    return PreparedData(
        X=X,
        y=y,
        metric_user=metric_user,
        metric_sklearn=metric_sklearn,
        resolved_validation=resolved_validation,
        resolved_n_folds=resolved_n_folds,
        resolved_test_size=resolved_test_size,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )


def execute_phases(
    cfg: Config,
    prepared: PreparedData,
) -> tuple[dict[str, tuple[float, dict[str, Any]]], dict[str, str]]:
    """Выполнить все фазы HPO и вернуть результаты последней фазы.

    Шаг 3 пайплайна (issue #31). Содержит вложенные помощники
    ``prepare_search_space`` и ``_execute_hpo_phase`` (ранее объявленные
    внутри ``train_best_model``), цикл по фазам, параллельное выполнение,
    circuit breaker дисквалификации и фильтрацию невалидных результатов.

    Multi-phase semantics (issue #13): phase results are not accumulated —
    each phase rebuilds its results from scratch, so an algorithm that fails
    in the current phase (total HPO failure: returned ``None`` or an invalid
    score) is fully excluded and does not keep a stale record from a previous
    phase. The final winner is chosen by the results of the LAST phase: for
    the typical ``all_algorithms → refine_winner`` pipeline this is the
    refined winner; a winner failure during ``refine_winner`` raises
    ``RuntimeError`` instead of silently rolling back to the previous phase
    results.

    Args:
        cfg (Config): Объект конфигурации.
        prepared (PreparedData): Подготовленные данные и разрешённые настройки.

    Returns:
        Tuple[Dict[str, Tuple[float, Dict[str, Any]]], Dict[str, str]]:
            Кортеж (результаты последней фазы «алгоритм → (скор, параметры)»,
            дисквалифицированные алгоритмы «имя → причина»). Результаты пусты,
            если фазы не выполнялись (``general.phases`` пуст) или ни один
            алгоритм не дал валидного результата — тогда ``select_winner``
            не вызывается (защита в ``train_best_model``).

    Raises:
        RuntimeError: Если фаза 'refine_winner' не имеет предыдущих
            результатов, либо ни один алгоритм не дал валидного скора
            в фазе.
    """
    X = prepared.X
    y = prepared.y
    metric_sklearn = prepared.metric_sklearn
    resolved_validation = prepared.resolved_validation
    resolved_n_folds = prepared.resolved_n_folds
    resolved_test_size = prepared.resolved_test_size
    categorical_features = prepared.categorical_features
    numerical_features = prepared.numerical_features

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

        # Адаптивный epsilon для SVR (issue #55): границы поиска epsilon
        # масштабируются разбросом y_train в рантайме (tuner.resolve_search_space),
        # поэтому дефолтный слепой диапазон [1e-3, 1.0] из DEFAULT_SPACES
        # опускается. Если пользователь задал epsilon явно — он уже в space
        # (space.update(user_overrides) выше), и приоритет пользователя
        # сохраняется: адаптивная логика не вмешивается.
        if algo_name == "svr" and not (user_overrides and "epsilon" in user_overrides):
            space.pop("epsilon", None)

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
            Tuple[float, Dict[str, Any]] | None: tuple (score, params),
                or ``None`` when HPO did not return a valid result (the
                algorithm is excluded from candidates, issue #13).
        Raises:
            Exception: Exceptions from ``_run_hpo`` are logged and re-raised
                (including ``InvalidAlgorithmError``).
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
                validation_strategy=resolved_validation,
                n_folds=resolved_n_folds,
                train_test_split_test_size=resolved_test_size,
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
                feature_selection_cfg=cfg.general.feature_selection,
            )

            if result is None:
                return None

            score, params = result

            # Логируем значение в пользовательской семантике: для neg_*-метрик
            # (например, neg_root_mean_squared_error) «сырое» значение скорера
            # инвертировано, пользователю показываем естественное (положительное).
            disp = to_user_value(metric_sklearn, score)
            _LOG.info(f"{phase_name} {algo:15} | score {disp:.5f} | params {params}")
            return score, params
        except Exception as err:
            _LOG.warning(f"Skip {algo} in {phase_name}: {err}")
            raise

    # Начальный список кандидатов (все включенные алгоритмы)
    all_algorithms = _algorithms_as_dict(cfg.algorithms)
    phase_results: dict[str, tuple[float, dict[str, Any]]] = {}
    # Дисквалифицированные алгоритмы (circuit breaker): имя -> причина.
    # Дисквалификация не прерывает запуск: алгоритм исключается из
    # кандидатов и результатов, остальные продолжают обучение (issue #12).
    disqualified_algorithms: dict[str, str] = {}
    for phase in cfg.general.phases:
        _LOG.info(
            f"--- Starting Phase: {phase.name} ({phase.n_trials}"
            f" trials, action: {phase.action}) ---"
        )

        # Snapshot of the previous phases' results. Used ONLY to select the
        # refine_winner and to pass initial_params to the worker. Records from
        # previous phases are NOT carried into the current one: phase_results
        # is rebuilt from the current phase only, so an algorithm that failed
        # in it (returned None or an invalid score) does not keep a stale
        # record from a previous phase and does not compete with real results
        # (issue #13). Consequence: the final winner is chosen by the results
        # of the LAST phase.
        prev_results = phase_results

        if phase.action == "refine_winner":
            if not prev_results:
                raise RuntimeError(
                    f"Phase '{phase.name}' requires a winner,"
                    f" but no previous results exist."
                )

            # Детерминированный выбор победителя (issue #32): максимальный
            # «сырой» скор; ничья разрешается порядком конфигурации.
            winner_algo = select_winner(prev_results)
            _LOG.info(f"Phase '{phase.name}' filtering for winner: {winner_algo}")
            current_candidates = {winner_algo: all_algorithms[winner_algo]}
            # Only the previous phase's winner takes part in refine_winner:
            # the losers' records must not "resurrect" if the winner fails
            # during refinement. The refine result replaces the winner's
            # record; if the winner fails (None/invalid score) the phase ends
            # with RuntimeError — no silent rollback to phase 1.
        else:
            # all_algorithms phase: every enabled algorithm, results from
            # previous phases are not accumulated. Algorithms disqualified by
            # the circuit breaker do not return to the phase (issue #12).
            current_candidates = {
                n: a
                for n, a in all_algorithms.items()
                if a.enable and n not in disqualified_algorithms
            }

        # Each phase starts with an empty phase_results: it is rebuilt from the
        # current phase only, so an algorithm that fails in this phase does not
        # keep a stale record from a previous phase (issue #13).
        phase_results = {}

        # Кандидаты, участвующие в текущей фазе (нужны для диагностики
        # в сообщении об отсутствии валидных результатов)
        phase_candidates = list(current_candidates.keys())

        def _worker(
            algo_name: str,
            algo_cfg: AlgoCfg,
            p: HPOPhaseCfg = phase,
            pr: dict[str, tuple[float, dict[str, Any]]] = prev_results,
        ) -> tuple[str, float, dict[str, Any]] | None:
            """Воркер для параллельного или последовательного запуска задачи HPO.
            Args:
                algo_name (str): Имя алгоритма.
                algo_cfg (AlgoCfg): Конфигурация алгоритма.
            Returns:
                Optional[Tuple[str, float, Dict[str, Any]]]: Название, скор и параметры
                    или None в случае ошибки.
            """
            # Determine initial_params for refine_winner: keep the best parameters of
            # the previous phase for enqueue_trial. The prev_results snapshot is
            # fixed via a default argument (like `p`), so the worker uses the
            # state at definition time (B023). Workers only READ this dict;
            # writes to phase_results happen after all phase workers finish, and
            # at the end of the phase phase_results is re-bound to the filtered
            # valid_results (objects are never mutated) — therefore the snapshot
            # is stable in both parallel and sequential modes.
            init_params = None
            if p.action == "refine_winner" and algo_name in pr:
                _, prev_params = pr[algo_name]
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
            except _CanonicalIAE as err:
                # Circuit breaker (issue #12): алгоритм дисквалифицирован после
                # MAX_FATAL_FAILURES подряд фатальных ошибок. Это НЕ аварийный
                # стоп всего запуска — алгоритм исключается из кандидатов,
                # остальные продолжают обучение (аналогично None-результату).
                _LOG.warning(
                    "Algorithm %s disqualified in phase %s: %s",
                    algo_name,
                    p.name,
                    err,
                )
                disqualified_algorithms[algo_name] = str(err)
                return None
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

        # Disqualified algorithms (circuit breaker) are removed from the results and
        # from the candidates of the following phases: they must not win and must
        # not waste compute again.
        # phase_results.pop is a safety net for disqualification in the same
        # phase (a no-op for the already-rebuilt dict).
        for name in list(disqualified_algorithms):
            phase_results.pop(name, None)
            current_candidates.pop(name, None)

        # Filter out invalid phase results: None/NaN/±inf and the worst-score
        # sentinel class. The sentinel class is defined as any score at or below
        # the exact float32 minimum (WORST_SCORE_THRESHOLD): it covers both the
        # HPO_WORST_SCORE constant (returned by the tuner when a trial metric is
        # non-finite) and the raw float(np.finfo(np.float32).min) value (possible
        # from numpy-cast metrics or custom tuners), which the old
        # math.isclose(rel_tol=1e-9) check missed (relative difference ≈ 9.88e-9,
        # issue #32). Real metrics of such magnitude are practically impossible,
        # so the threshold cannot drop a legitimate winner. params must be a
        # dict (the tuner failure signal is params=None, issue #13); garbage
        # score types from custom tuners (None, strings, etc.) are rejected by
        # is_valid_winner_score.
        valid_results: dict[str, tuple[float, dict[str, Any]]] = {}
        for name, (score, params) in phase_results.items():
            if not isinstance(params, dict):
                _LOG.warning(
                    "Algorithm %s produced invalid params %r (expected dict); "
                    "excluding from phase candidates",
                    name,
                    params,
                )
                continue
            if not is_valid_winner_score(score):
                _LOG.warning(
                    "Algorithm %s produced invalid score %r (non-finite or "
                    "worst-score sentinel); excluding from phase candidates",
                    name,
                    score,
                )
                continue
            valid_results[name] = (score, params)

        # Keep only the valid results of the current phase in phase_results: entries
        # with an invalid score (HPO_WORST_SCORE etc.) are removed. Since
        # phase_results started the phase empty, here it is guaranteed to
        # contain only this phase's results (issue #13).
        phase_results = valid_results

        if not valid_results:
            failed_algos = [n for n in phase_candidates if n not in valid_results]
            raise RuntimeError(
                f"No algorithms produced valid scores in phase '{phase.name}'. "
                f"Failed algorithms: {failed_algos}"
            )
    return phase_results, disqualified_algorithms


def persist_artifact(
    *,
    cfg: Config,
    prepared: PreparedData,
    winner_algo: str,
    final_score: float,
    final_params: dict[str, Any],
    model_path_override: str | Path | None,
    disqualified_algorithms: dict[str, str],
) -> dict[str, Any]:
    """Финальное обучение победителя, сохранение артефакта и сборка отчёта.

    Шаг 5 пайплайна (issue #31). Обучает модель алгоритма-победителя на
    полном наборе данных через ``_fit_and_save``, сохраняет артефакт на диск
    и формирует словарь результата: ``algorithm``, ``score`` (в
    пользовательской семантике, issue #26), ``metric``, ``params``,
    ``model_path``; при наличии добавляются ``disqualified_algorithms``
    (issue #12) и ``additional_metrics``.

    Args:
        cfg (Config): Объект конфигурации.
        prepared (PreparedData): Подготовленные данные и настройки.
        winner_algo (str): Имя алгоритма-победителя.
        final_score (float): «Сырой» скор победителя (уже прошёл инвариант
            ``is_valid_winner_score``).
        final_params (Dict[str, Any]): Лучшие гиперпараметры победителя.
        model_path_override (str | Path | None): Альтернативный путь
            сохранения модели.
        disqualified_algorithms (Dict[str, str]): Дисквалифицированные
            алгоритмы (имя → причина); добавляются в результат при наличии.

    Returns:
        Dict[str, Any]: Словарь с результатами обучения.

    Raises:
        Exception: Пробрасывается ошибка финального обучения/сохранения
            (логируется перед повторным поднятием).
    """
    model_path = Path(model_path_override or cfg.general.path_to_model)
    winner_cfg = _algorithms_as_dict(cfg.algorithms)[winner_algo]

    try:
        trainer = _fit_and_save(
            winner_algo,
            winner_cfg,
            prepared.X,
            prepared.y,
            final_params,
            model_path,
            cfg,
            metric_name_sklearn=prepared.metric_sklearn,
            validation_strategy=prepared.resolved_validation,
            n_folds=prepared.resolved_n_folds,
            test_size=prepared.resolved_test_size,
        )
        _LOG.info("Model saved to %s", model_path.resolve())
    except Exception as e:
        _LOG.error(f"Failed to save final model: {e}")
        raise

    result: dict[str, Any] = {
        "algorithm": winner_algo,
        # Единая пользовательская семантика (issue #26): score всегда отражает
        # естественное значение метрики (положительный RMSE/MAE, обычный R²),
        # metric — пользовательское имя метрики сравнения из конфигурации.
        # «Сырое» (инвертированное для neg_*-скореров) значение остаётся
        # внутренней деталью оптимизатора и до пользователя не доходит.
        "score": to_user_value(prepared.metric_user, final_score),
        "metric": prepared.metric_user,
        "params": final_params,
        "model_path": str(model_path),
    }
    # Диагностика circuit breaker (issue #12): какие алгоритмы и почему были
    # дисквалифицированы. Ключ добавляется только при наличии дисквалификаций,
    # иначе результаты идентичны прежнему поведению.
    if disqualified_algorithms:
        result["disqualified_algorithms"] = dict(disqualified_algorithms)
    # Дополнительные информационные метрики финальной модели. Ключ добавляется
    # только при наличии метрик (после дедупликации и исключения основной
    # метрики сравнения), иначе результаты идентичны прежнему поведению.
    if cfg.general.additional_metrics:
        result["additional_metrics"] = dict(trainer.additional_scores)
    return result


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

    Оркестрирует декомпозированный пайплайн (issue #31):
    ``load_config → prepare_dataset → execute_phases → select_winner →
    persist_artifact``. Логика каждого шага вынесена в отдельные функции
    модуля; здесь остаются только выбор победителя и инвариант winner-скора.

    Multi-phase semantics (issue #13): phase results are not accumulated — each
    phase rebuilds its results from scratch, so an algorithm that fails in the
    current phase (total HPO failure: returned ``None`` or an invalid score) is
    fully excluded and does not keep a stale record from a previous phase.
    The final winner is chosen by the results of the LAST phase: for the typical
    ``all_algorithms → refine_winner`` pipeline this is the refined winner;
    a winner failure during ``refine_winner`` raises ``RuntimeError`` instead of
    silently rolling back to the previous phase results.
    Args:
        config (Union[str, Path, Config, Dict[str, Any]]): Конфигурация обучения.
            Может быть путем к файлу, словарем или объектом Config.
        df (pd.DataFrame): Исходные данные.
        target (str | None): Имя целевого столбца. По умолчанию 'target'.
        model_path_override (str | Path | None): Альтернативный путь сохранения модели.
    Returns:
        Dict[str, Any]: Словарь с результатами: название алгоритма,
            ``score`` — значение метрики в пользовательской семантике
            (положительный RMSE/MAE, обычный R²; для neg_-скореров значение
            инвертировано обратно), ``metric`` — пользовательское имя метрики
            сравнения, параметры и путь к файлу. Если в конфигурации заданы
            дополнительные метрики (``general.additional_metrics``),
            в результат добавляется ключ ``additional_metrics`` — словарь
            {метрика: значение}, рассчитанных для финальной модели на том же
            наборе данных, что и основная метрика. Если в ходе запуска
            какие-либо алгоритмы были дисквалифицированы circuit breaker'ом
            (5 подряд фатальных ошибок), в результат добавляется ключ
            ``disqualified_algorithms`` — словарь {имя алгоритма: причина
            дисквалификации}.
    Raises:
        TypeError: При передаче конфига неподдерживаемого типа.
        RuntimeError: Если ни один алгоритм не смог успешно завершить фазу HPO.
            Дисквалификация отдельного алгоритма (``InvalidAlgorithmError``)
            запуск НЕ прерывает: алгоритм исключается из кандидатов, остальные
            продолжают обучение (issue #12).
    """
    cfg, target_col = load_config(config, df, target)
    prepared = prepare_dataset(cfg, df, target_col)
    phase_results, disqualified_algorithms = execute_phases(cfg, prepared)

    # After all phases, pick the final winner. Since phase_results is rebuilt per
    # phase, the winner is chosen by the LAST phase's results (for the typical
    # all_algorithms → refine_winner pipeline this is the refined winner).
    # Детерминированный tie-break при равных скорах: побеждает первый
    # алгоритм в порядке конфигурации (select_winner, issue #32).
    if not phase_results:
        # Защита от пустого списка фаз (конфиг допускает phases: []): цикл
        # в execute_phases не выполнялся, и select_winner({}) бросил бы
        # ValueError — поднимаем понятную RuntimeError (ревью PR #22).
        raise RuntimeError("No valid results after HPO phases; cannot select a winner.")
    winner_algo = select_winner(phase_results)
    final_score, final_params = phase_results[winner_algo]
    # Финальный инвариант (issue #32): score победителя обязан быть конечным и
    # строго выше класса worst-score сентинела, прежде чем попасть в
    # result["score"]. Иначе сентинел мог бы протечь в отчёт — для neg_-метрик
    # to_user_value инвертировал бы его в абсурдные +3.4e38.
    if not is_valid_winner_score(final_score):
        raise RuntimeError(
            f"Winner '{winner_algo}' produced an invalid final score "
            f"{final_score!r} (non-finite or worst-score sentinel); "
            f"refusing to report it."
        )

    return persist_artifact(
        cfg=cfg,
        prepared=prepared,
        winner_algo=winner_algo,
        final_score=final_score,
        final_params=final_params,
        model_path_override=model_path_override,
        disqualified_algorithms=disqualified_algorithms,
    )
