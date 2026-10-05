"""Hyperparameter Optimisation Module:
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

# ─────────────────────────────── stdlib
import logging as _logging
from collections.abc import Callable
from functools import partial
from typing import Any, NoReturn, cast

# ──────────────────────────── third-party
import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from optuna.trial import Trial
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
)

# ──────────────────────────── project
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.common.hyperopt_defaults import (
    FloatSpace,
    clip_search_space,
)
from configurable_automl_engine.common.validation_utils import get_effective_train_size
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.models import (
    create_model,
    requires_dense_input,
    resolve_algorithm_name,
)
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
    build_preprocessor,
    detect_feature_types,
)
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from configurable_automl_engine.training_engine.metrics import get_scorer_object
from configurable_automl_engine.validation import iter_splits, make_cv, norm_val_method

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


# ═══════════════════════════════════ constants ═══════════════════════════════
# The worst possible trial score — an analog of the float32 minimum or
# float('-inf'). Returned as the objective value when the real metric is not
# finite (NaN/+inf, e.g. an nrmse error). The value is practically unreachable
# for real metrics, hence used as a sentinel.
#
# Used in two places:
# 1. Inside `_objective`: a non-finite trial avg_score is replaced with this
#    constant (the trial completes, but with a deliberately worst score).
# 2. On the orchestrator side: the `valid_results` filter in
#    `training_engine/component.py` drops results whose score belongs to the
#    worst-score sentinel class — any score at or below WORST_SCORE_THRESHOLD —
#    treating the algorithm as failed (issues #13, #32).
HPO_WORST_SCORE = -3.4028235e38

# The lower bound of the "worst-score sentinel" class: the exact float32
# minimum. Any score <= this bound is treated as a sentinel — real metrics of
# such magnitude are practically impossible (documented threshold decision,
# issue #32). Comparing against this bound (instead of the exact
# HPO_WORST_SCORE via math.isclose) also catches the raw
# float(np.finfo(np.float32).min) value that numpy-cast metrics or custom
# tuners may return: its relative difference from HPO_WORST_SCORE is
# ≈ 9.88e-9, which the old isclose(rel_tol=1e-9) check silently missed.
WORST_SCORE_THRESHOLD = float(np.finfo(np.float32).min)

# ═══════════════════════════════════ search spaces ═══════════════════════════


# ══════════ KNN-space зависит от размера выборки ══════════
def _make_knn_space(n_samples: int) -> Callable[[Trial], dict[str, Any]]:
    """Создать генератор пространства поиска для алгоритма KNN."""

    def _space(t: Trial) -> dict[str, Any]:
        # Ограничение n_neighbors физическим пределом обучающей выборки (N_eff - 1).
        # Используем n_samples_eff вместо общего количества строк в датасете.
        physical_limit = max(1, n_samples - 1)
        max_k = int(min(30, physical_limit))

        return {
            "n_neighbors": t.suggest_int("n_neighbors", 1, max_k),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
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
                    key, low_val, high_val, step=int(step) if step is not None else 1
                )
            elif dist_type == "float":
                params[key] = trial.suggest_float(
                    key,
                    float(low),
                    float(high),
                    step=float(step) if step is not None else None,
                )
            elif dist_type == "float_log":
                FloatSpace.validate_log_low(float(low))
                params[key] = trial.suggest_float(
                    key, float(low), float(high), log=True
                )
            elif dist_type == "categorical":
                # Извлекаем список опций из атрибута options или из вложенного config
                options = getattr(value, "options", None)
                if options is None and hasattr(value, "config"):
                    options = getattr(value.config, "options", None)

                # Если всё еще None, откатываемся к low (для совместимости)
                if options is None:
                    options = low

                params[key] = trial.suggest_categorical(key, options)
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


def _build_scorer(name: str) -> Any:
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
    # np.asarray гарантирует чистый np.ndarray и np.dtype для mypy (без ExtensionArray/Dtype)
    arr = np.asarray(y)

    if arr.ndim != 1:
        return False

    uniq = np.unique(arr)
    return uniq.size <= 15 and (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)
    )


def _split_train_test(
    X: Any, y: Any, *, test_size: float = 0.2, random_state: int | None = 42
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
        return cast(
            tuple[Any, Any, Any, Any],
            train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=True,
                random_state=random_state,
                stratify=strat,
            ),
        )
    except ValueError:
        return cast(
            tuple[Any, Any, Any, Any],
            train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_state
            ),
        )


# ═══════════════ early stopping (pruning) helpers ═══════════════════════════
_DEFAULT_PRUNING_CONFIG: dict[str, Any] = {
    "enable": False,
    "strategy": "median",
    "min_steps": 1,
    "n_startup_trials": 5,
    "reduction_factor": 3,
}


def _normalize_pruning_config(
    pruning: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Нормализовать конфигурацию ранней остановки в словарь с валидными полями.

    Принимает dict из настроек (например, прокинутый из ``training_engine``)
    или None. Возвращает ``None``, если ранняя остановка выключена или
    конфиг не передан — в этом случае поведение тюнера идентично текущему.

    Args:
        pruning (dict[str, Any] | None): Словарь с ключами enable, strategy,
            min_steps, n_startup_trials, reduction_factor (или их подмножеством).
    Returns:
        dict[str, Any] | None: Нормализованная конфигурация прайнера или None.
    """
    if not pruning:
        return None
    enabled = bool(pruning.get("enable", False))
    if not enabled:
        return None
    return {
        "strategy": str(pruning.get("strategy", _DEFAULT_PRUNING_CONFIG["strategy"])),
        "min_steps": int(
            pruning.get("min_steps", _DEFAULT_PRUNING_CONFIG["min_steps"])
        ),
        "n_startup_trials": int(
            pruning.get("n_startup_trials", _DEFAULT_PRUNING_CONFIG["n_startup_trials"])
        ),
        "reduction_factor": int(
            pruning.get("reduction_factor", _DEFAULT_PRUNING_CONFIG["reduction_factor"])
        ),
    }


def _build_pruner(cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    """Создать объект прайнера Optuna по нормализованной конфигурации.

    Args:
        cfg (dict[str, Any]): Нормализованная конфигурация ранней остановки
            (результат ``_normalize_pruning_config``).
    Returns:
        optuna.pruners.BasePruner: Настроенный прайнер Optuna.
    Raises:
        HyperoptError: Если указана неизвестная стратегия отсечения или
            недопустимые значения параметров.
    """
    strategy = cfg["strategy"]
    min_steps = cfg["min_steps"]
    if min_steps < 1:
        raise HyperoptError(
            f"pruning.min_steps must be a positive integer, got {min_steps}"
        )
    if strategy == "median":
        n_startup_trials = cfg["n_startup_trials"]
        if n_startup_trials < 1:
            raise HyperoptError(
                "pruning.n_startup_trials must be a positive integer, "
                f"got {n_startup_trials}"
            )
        # n_warmup_steps=min_steps: шаги нумеруются с 1, поэтому первое
        # решение об отсечении возможно ровно после min_steps отчётов.
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=min_steps,
        )
    if strategy == "hyperband":
        reduction_factor = cfg["reduction_factor"]
        if reduction_factor < 2:
            raise HyperoptError(
                "pruning.reduction_factor must be at least 2 (Hyperband), "
                f"got {reduction_factor}"
            )
        return optuna.pruners.HyperbandPruner(
            min_resource=min_steps,
            reduction_factor=reduction_factor,
        )
    raise HyperoptError(
        f"Unknown pruning strategy: '{strategy}'. Must be 'median' or 'hyperband'."
    )


def _evaluate_with_intermediate_reports(
    trial: Trial,
    estimator: Any,
    X: Any,
    y: Any,
    *,
    method: str,
    n_folds: int,
    test_size: float,
    random_state: int | None,
    scorer: Any,
) -> float:
    """Оценить estimator пофолдово с публикацией промежуточных результатов.

    Является «энейблером» ранней остановки: после каждого фолда
    публикуется кумулятивное среднее значение метрики через
    ``trial.report``, после чего вызывается ``trial.should_prune()``.
    Безнадёжный триал прерывается исключением ``optuna.TrialPruned`` до
    завершения полного объёма оценки.

    Args:
        trial (Trial): Текущий триал Optuna.
        estimator (Any): Оценщик (возможно, пайплайн) для оценки.
        X (Any): Матрица признаков.
        y (Any): Вектор целевой переменной.
        method (str): Стратегия разбиения ('k_fold' или 'loo').
        n_folds (int): Количество фолдов для k_fold.
        test_size (float): Доля теста (не используется для k_fold/loo,
            передаётся для унификации вызова iter_splits).
        random_state (int | None): Зерно случайности.
        scorer (Any): Объект метрики sklearn.
    Returns:
        float: Среднее значение метрики по всем фолдам.
    Raises:
        optuna.TrialPruned: Если прайнер решил отсечь триал.
    """
    fold_scores: list[float] = []
    for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(
        iter_splits(
            X,
            y,
            method=method,
            n_folds=n_folds,
            test_size=test_size,
            random_state=random_state,
        ),
        start=1,
    ):
        # Клонируем оценщик на каждый фолд (как это делает cross_val_score),
        # чтобы избежать накопления состояния между фолдами.
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_tr, y_tr)
        fold_scores.append(float(scorer(fold_estimator, X_te, y_te)))
        running_mean = float(np.mean(fold_scores))
        # Публикуем промежуточный результат: кумулятивное среднее точнее
        # оценивает финальную метрику (среднее по фолдам), чем одиночный фолд.
        trial.report(running_mean, step=fold_idx)
        if trial.should_prune():
            log.debug(
                "Trial %d pruned by early stopping at step %d",
                trial.number,
                fold_idx,
            )
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores))


# ═════════════════════ PUBLIC: optimize() ═══════════════════════════════════
class _OptimizeRunner:
    """Один прогон гиперпараметрической оптимизации, разбитый на шаги.

    ``optimize()`` остаётся публичной точкой входа с прежней сигнатурой и
    делегирует всю работу этому классу (issue #30). Монолитная логика
    исходной функции разнесена по изолированным шагам:

        run()
         ├─ resolve_validation()
         ├─ resolve_search_space()
         ├─ build_trial_pipeline()
         ├─ handle_pruning()
         ├─ evaluate_trial() / handle_failure()  (в цикле Optuna)
         ├─ select_best_trial()
         └─ fit_best_model()

    Состояние прогона хранится в атрибутах экземпляра: каждый шаг заполняет
    свои поля, а последующие шаги (и целевая функция Optuna) читают их без
    длинных цепочек аргументов.
    """

    MAX_FATAL_FAILURES = 5
    """Лимит подряд идущих фатальных сбоев до дисквалификации алгоритма."""

    def __init__(
        self,
        *,
        algo_name: str,
        X: Any,
        y: Any,
        data_oversampling: bool,
        data_oversampling_multiplier: float,
        data_oversampling_algorithm: str,
        metric: str,
        val_method: ValidationStrategy | str,
        validation_strategy: ValidationStrategy | str | None,
        n_folds: int,
        n_trials: int,
        random_state: int | None,
        train_test_split_test_size: float,
        space_overrides: dict[str, Callable[[Trial], dict[str, Any]]] | None,
        initial_params: dict[str, Any] | None,
        preprocessor: Any | None,
        categorical_features: list[str] | None,
        numerical_features: list[str] | None,
        encoding: EncodingStrategy | None,
        preprocessing_override: PreprocessingOverride | dict[str, Any] | None,
        pruning: dict[str, Any] | None,
        high_cardinality_threshold: int | None,
        high_cardinality_encoding: EncodingStrategy | None,
        hashing_n_components: int,
        target_encoding_smoothing: float,
        target_encoding_fallback: float | None,
        feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None,
    ) -> None:
        """Сохранить параметры одного прогона оптимизации.

        Args:
            Все аргументы повторяют одноимённые параметры ``optimize()`` —
            см. её docstring за описанием семантики.
        """
        self.algo_name = algo_name
        self.X = X
        self.y = y
        self.metric = metric
        self.val_method = val_method
        self.validation_strategy = validation_strategy
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.train_test_split_test_size = train_test_split_test_size
        self.space_overrides = space_overrides
        self.initial_params = initial_params
        self.preprocessor = preprocessor
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoding = encoding
        self.preprocessing_override = preprocessing_override
        self.pruning = pruning
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.hashing_n_components = hashing_n_components
        self.target_encoding_smoothing = target_encoding_smoothing
        self.target_encoding_fallback = target_encoding_fallback
        self.feature_selection_cfg = feature_selection_cfg

        # Конфиг оверсэмплинга для сборки пайплайна (единая точка).
        self.oversampling_config: dict[str, Any] = {
            "active": data_oversampling,
            "params": {
                "multiplier": data_oversampling_multiplier,
                "algorithm": data_oversampling_algorithm,
            },
        }

        # ── Поля, заполняемые шагами run() ────────────────────────────────
        self.algo: str = ""
        self.fs_cfg = FeatureSelectionCfg()
        self.fs_mode = self.fs_cfg.mode
        self.is_isotonic = False
        self.n_samples_eff: int = 0
        self.val_method_eff: str = ""
        self.cv_obj: Any = None
        self.auto_decision: dict[str, Any] | None = None
        self.resolved_via_auto = False
        self.effective_n_folds = self.n_folds
        self.space_fn: Callable[[Trial], dict[str, Any]]
        self.scorer: Any = None
        self.pruner: optuna.pruners.BasePruner | None = None
        self.pruning_active = False
        self.study: optuna.Study
        self.consecutive_fatal_failures = 0
        self.best_params: dict[str, Any]
        self.best_score: float
        self.clean_model_params: dict[str, Any]
        self.best_apply_fs: bool

    def run(self) -> tuple[Any | None, dict[str, Any] | None, float | None]:
        """Выполнить полный цикл оптимизации, разбитый на шаги (issue #30)."""
        if not self._validate_n_trials():
            return None, None, None

        self.resolve_validation()
        self.resolve_search_space()
        self.build_trial_pipeline()
        self.handle_pruning()

        # Если ранняя остановка выключена — явно отключаем любой прайнер
        # (NopPruner), чтобы триалы гарантированно выполнялись полностью.
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=self.pruner if self.pruning_active else optuna.pruners.NopPruner(),
        )
        # Если есть параметры из предыдущей фазы — enqueue как первый trial
        if self.initial_params is not None:
            self.study.enqueue_trial(self.initial_params)
            log.info(
                "Enqueued initial params from previous phase: %s", self.initial_params
            )
        try:
            self.study.optimize(self._objective, n_trials=self.n_trials)
        except InvalidAlgorithmError:
            log.warning(
                "Algorithm %s disqualified after %d consecutive fatal failures",
                self.algo,
                self.MAX_FATAL_FAILURES,
            )
            raise

        if not self.select_best_trial():
            return None, None, None
        return self.fit_best_model()

    # ─────────────────── шаги прогона (issue #30) ──────────────────────────

    def resolve_validation(self) -> None:
        """Разрешить стратегию валидации и связанные с ней величины (шаг 1).

        Выполняет нормализацию входа (alias ``validation_strategy``), проверку
        данных и доступности алгоритма, нормализацию конфигурации отбора
        признаков (issue #11) и выбор CV-стратегии через ``make_cv``. Результаты
        сохраняются в атрибутах экземпляра для последующих шагов.
        """
        # alias имеет приоритет
        if self.validation_strategy is not None:
            self.val_method = self.validation_strategy

        self.algo = self.algo_name.lower()
        _validate_data(self.X, self.y)

        _get_estimator(self.algo)

        # Нормализуем конфигурацию отбора признаков: None -> дефолт
        # (mode='disabled'), dict -> FeatureSelectionCfg (валидация на этапе
        # запуска), объект FeatureSelectionCfg -> как есть. Изотоническая
        # регрессия строго одномерная и использует собственный
        # IsotonicDataTransformer, поэтому для неё отбор принудительно
        # отключается независимо от режима.
        fs_cfg = self.feature_selection_cfg
        if fs_cfg is None:
            self.fs_cfg = FeatureSelectionCfg()
        elif isinstance(fs_cfg, FeatureSelectionCfg):
            self.fs_cfg = fs_cfg
        elif isinstance(fs_cfg, dict):
            try:
                self.fs_cfg = FeatureSelectionCfg.model_validate(fs_cfg)
            except Exception as err:
                raise HyperoptError(f"Invalid feature_selection_cfg: {err}") from err
        else:
            raise TypeError(
                "feature_selection_cfg must be a FeatureSelectionCfg, dict or None, "
                f"got {type(fs_cfg).__name__}"
            )
        self.fs_mode = self.fs_cfg.mode
        self.is_isotonic = resolve_algorithm_name(self.algo) == "isotonic_regression"
        log.info(
            "Feature selection mode for algorithm '%s': %s",
            self.algo,
            self.fs_mode.value,
        )

        # Определяем, сколько строк реально "увидит" модель при обучении
        # внутри CV/Split, и разрешаем стратегию валидации (в т.ч. 'auto').
        n_samples = len(self.y)
        n_features = self.X.shape[1]
        self.n_samples_eff = get_effective_train_size(
            n_total=n_samples,
            strategy=self.validation_strategy or self.val_method,
            n_folds=self.n_folds,
            test_size=self.train_test_split_test_size,
            n_features=n_features,
        )
        self.val_method_eff, self.cv_obj, self.auto_decision = make_cv(
            n_samples,
            val_method=self.val_method,
            n_folds=self.n_folds,
            random_state=self.random_state,
            test_size=self.train_test_split_test_size,
            n_features=n_features,
        )
        # 'auto' разрешается ровно один раз (здесь, в make_cv). Полученное
        # решение передаётся в iter_splits, чтобы test-size не пересчитывался
        # повторно и не мог разойтись между n_samples_eff и фактическим
        # разбиением.
        self.resolved_via_auto = norm_val_method(self.val_method) == "auto"

        # Единое число фолдов для ветки ранней остановки (k_fold). Стратегия
        # 'auto' могла разрешиться в kfold с числом фолдов k, отличным от
        # исходного n_folds: в этом случае pruning-ветка обязана использовать
        # вычисленное k, иначе iter_splits внутри
        # _evaluate_with_intermediate_reports откатится на дефолтный n_folds и
        # оценка разойдётся с cross_val_score(cv=cv_obj).
        if self.val_method_eff == "k_fold":
            # Число фолдов берём из готового cv_obj (для k_fold он
            # гарантированно не None — make_cv всегда создаёт KFold в этой
            # ветке). Одна формула покрывает auto→kfold (k из решения),
            # fallback 'auto' без P (клампинг до max(2, k)) и явный k_fold.
            assert self.cv_obj is not None
            self.effective_n_folds = int(self.cv_obj.get_n_splits())
        else:
            # Явный loo / train_test_split: число фолдов не используется.
            self.effective_n_folds = self.n_folds

    def resolve_search_space(self) -> None:
        """Разрешить пространство поиска гиперпараметров и скорер (шаг 2).

        Поддерживает динамическое пространство KNN (зависит от эффективного
        размера выборки), прямые переопределения-функции и словарные
        конфигурации YAML (SearchSpaceEntry). Сохраняет ``space_fn`` и
        ``scorer`` в атрибутах экземпляра.

        Raises:
            HyperoptError: Если для выбранного алгоритма не определено
                пространство поиска.
        """
        base_space_fn: Callable[[Trial], dict[str, Any]] | None = None
        if self.algo == "knn":
            base_space_fn = _make_knn_space(self.n_samples_eff)

        # Приоритет 1: Прямые переопределения (функции)
        # Приоритет 2: Динамический конфиг из YAML (dict с SearchSpaceEntry)
        external_config = (self.space_overrides or {}).get(self.algo)

        # Если пришла функция (старый механизм) — используем её
        if callable(external_config):
            space_fn: Callable[[Trial], dict[str, Any]] | None = external_config
        # Если пришел словарь (новый механизм из YAML) — создаем обертку
        elif isinstance(external_config, dict):
            clipped_config = clip_search_space(external_config, self.n_samples_eff)
            space_fn = partial(_apply_dynamic_space, space_dict=clipped_config)
        else:
            space_fn = base_space_fn
        if space_fn is None:
            raise HyperoptError(f"Для «{self.algo}» нет search-space")

        self.space_fn = space_fn
        self.scorer = _build_scorer(self.metric)

    def build_trial_pipeline(self) -> None:
        """Построить препроцессор для фазы HPO (шаг 3).

        Единая точка построения препроцессора: категории -> one-hot (или
        ordinal, если задан encoding), числа -> пресет предобработки, выбранный
        автоматически по алгоритму (FR-1). Используется та же логика, что и в
        финальном обучении (trainer.ModelTrainer), поэтому HPO и финальный fit
        согласованы (AC-6). Готовая фабрика пайплайна — ``assemble_estimator``.
        """
        if self.preprocessor is None:
            encoding_strategy: EncodingStrategy = self.encoding or "one_hot"
            # Автоматический выбор пресета + применение явного override (FR-5).
            # Пресет разрешается один раз до обучения (производительность)
            # и логируется для наблюдаемости (AC-9).
            preset = resolve_preprocessing_preset(
                self.algo, self.preprocessing_override
            )
            log.info(
                "Resolved preprocessing preset for algorithm '%s': %s",
                self.algo,
                preset,
            )
            if (
                self.categorical_features is not None
                or self.numerical_features is not None
            ):
                # Явно переданные списки колонок (основной путь из
                # training_engine)
                if isinstance(self.X, pd.DataFrame):
                    self.preprocessor = build_preprocessor(
                        list(self.X.columns),
                        self.categorical_features or [],
                        self.numerical_features or [],
                        encoding=encoding_strategy,
                        imputation_strategy=preset.imputation_strategy,
                        scaling=preset.scaling,
                        high_cardinality_threshold=self.high_cardinality_threshold,
                        high_cardinality_encoding=self.high_cardinality_encoding,
                        hashing_n_components=self.hashing_n_components,
                        target_encoding_smoothing=self.target_encoding_smoothing,
                        target_encoding_fallback=self.target_encoding_fallback,
                        random_state=self.random_state,
                        force_dense_output=requires_dense_input(self.algo),
                    )
                else:
                    log.warning(
                        "categorical/numerical_features provided, but X is not a "
                        "pandas.DataFrame — skipping automatic preprocessor."
                    )
            elif isinstance(self.X, pd.DataFrame):
                # Автодетекция категорий (default-поведение при вызове вне
                # движка). Препроцессор строится при наличии ЛЮБЫХ признаков
                # (и категориальных, и числовых): это гарантирует, что на чисто
                # числовом DataFrame в фазе HPO применяется скалирование так же,
                # как в финальном ModelTrainer.
                cats, nums = detect_feature_types(self.X)
                if cats or nums:
                    self.preprocessor = build_preprocessor(
                        list(self.X.columns),
                        cats,
                        nums,
                        encoding=encoding_strategy,
                        imputation_strategy=preset.imputation_strategy,
                        scaling=preset.scaling,
                        high_cardinality_threshold=self.high_cardinality_threshold,
                        high_cardinality_encoding=self.high_cardinality_encoding,
                        hashing_n_components=self.hashing_n_components,
                        target_encoding_smoothing=self.target_encoding_smoothing,
                        target_encoding_fallback=self.target_encoding_fallback,
                        random_state=self.random_state,
                        force_dense_output=requires_dense_input(self.algo),
                    )
            else:
                log.warning(
                    "X is not a pandas.DataFrame and categorical_features/"
                    "numerical_features were not provided — categorical columns "
                    "(if any) will be treated as numeric."
                )

    def assemble_estimator(self, model_impl: Any, apply_fs: bool = False) -> Any:
        """Собрать пайплайн в порядке preprocessor -> [selector] -> [sampler] -> model.

        Оверсэмплинг получает уже закодированные числовые признаки (one-hot),
        поэтому SMOTE/ADASYN всегда работают с числовыми данными. Отбор
        признаков активируется строго между препроцессором и оверсэмплером:
        синтетические строки генерируются только для информативных признаков
        (порядок согласован с ModelTrainer, issue #11).

        Args:
            model_impl: Базовый регрессор.
            apply_fs: Флаг включения шага ``feature_selector``.
        """
        steps: list[tuple[str, Any]] = []
        if self.preprocessor is not None:
            steps.append(("preprocessor", self.preprocessor))
        # Новый шаг: отбор признаков активируется строго между preprocessor
        # и oversampler (issue #11).
        if apply_fs:
            steps.append(("feature_selector", self._fs_transformer_factory()))
        if self.oversampling_config["active"]:
            steps.append(
                ("sampler", DataOversampler(**self.oversampling_config["params"]))
            )
        steps.append(("model", model_impl))
        if len(steps) == 1:
            return model_impl
        return ImbPipeline(steps)

    def evaluate_trial(self, trial: Trial) -> float:
        """Оценить один триал Optuna и вернуть целевую метрику (шаг 4).

        Генерирует гиперпараметры через ``space_fn``, собирает пайплайн
        (``assemble_estimator``) и оценивает его по выбранной стратегии
        валидации. Нефинитный результат (+inf при ошибке nrmse или NaN)
        заменяется на ``HPO_WORST_SCORE`` — сентинел «худшего» исхода.

        Args:
            trial (Trial): Объект текущего испытания Optuna.
        Returns:
            float: Значение целевой метрики на текущем наборе параметров.
        Raises:
            ValueError: Нефатальная ошибка оценки (триал отсекается).
            MemoryError, RuntimeError, InvalidDataError: Фатальные сбои —
                учитываются в счётчике ``handle_failure``.
            optuna.TrialPruned: Отсечение прунером ранней остановки.
        """
        # 1. гиперпараметры и модель
        params = self.space_fn(trial)
        model = create_model(self.algo, **params)

        # --- ШАГ 2: ПОДГОТОВКА ОБЕРТКИ (WRAPPER) ---
        # Определение необходимости отбора признаков для текущего испытания:
        # в режиме 'auto' Optuna сама исследует пространство решений
        # (выгодно ли сокращать признаки), в 'always'/'disabled' решение
        # жёстко следует конфигурации. Для isotonic_regression категория
        # use_feature_selection не предлагается даже в режиме 'auto': отбор
        # принудительно выключен (одномерный алгоритм со своим
        # IsotonicDataTransformer), поэтому предложение было бы бессмысленным,
        # а служебный ключ не должен попадать в best_params.
        if self.fs_mode == FeatureSelectionMode.auto and not self.is_isotonic:
            apply_fs = trial.suggest_categorical("use_feature_selection", [True, False])
        elif self.fs_mode == FeatureSelectionMode.always:
            apply_fs = True
        else:
            apply_fs = False
        # Изотоническая регрессия требует ровно один признак и использует
        # собственный IsotonicDataTransformer — отбор принудительно выключен
        # (в т.ч. если 'always' или enqueued initial_params пронесли флаг).
        if self.is_isotonic:
            apply_fs = False

        current_estimator = self.assemble_estimator(model, apply_fs=apply_fs)

        # 3. оценка по стратегии валидации
        if self.val_method_eff == "train_test_split":
            # Если 'auto' было разрешено в make_cv — используем уже вычисленный
            # (dataset-зависимый) целочисленный test_size, чтобы iter_splits
            # не пересчитывал решение повторно. Иначе — фиксированная доля.
            if self.resolved_via_auto and self.auto_decision is not None:
                split_test_size: float | int = int(self.auto_decision["test_size"])
            else:
                split_test_size = self.train_test_split_test_size
            # Используем iter_splits для унификации; так как это генератор,
            # берём next()
            X_tr, X_te, y_tr, y_te = next(
                iter_splits(
                    self.X,
                    self.y,
                    method="train_test_split",
                    test_size=split_test_size,
                    random_state=self.random_state,
                )
            )
            current_estimator.fit(X_tr, y_tr)
            return float(self.scorer(current_estimator, X_te, y_te))
        if self.pruning_active:
            # Ранняя остановка: оцениваем пофолдово и публикуем промежуточные
            # результаты для прайнера Optuna.
            avg_score = _evaluate_with_intermediate_reports(
                trial,
                current_estimator,
                self.X,
                self.y,
                method=self.val_method_eff,
                n_folds=self.effective_n_folds,
                test_size=self.train_test_split_test_size,
                random_state=self.random_state,
                scorer=self.scorer,
            )
            # Если получили +inf (ошибка в nrmse) или NaN, возвращаем худший
            # float (аналог минимального float32 или float('-inf')).
            if not np.isfinite(avg_score):
                return HPO_WORST_SCORE
            return avg_score
        # cv_obj здесь гарантированно не None: ветка не-train_test_split и
        # не-pruning означает k_fold/loo, где make_cv всегда создаёт cv-объект.
        scores = model_selection.cross_val_score(
            current_estimator,
            self.X,
            self.y,
            cv=self.cv_obj,
            scoring=self.scorer,
            n_jobs=1,
        )
        avg_score = float(np.mean(scores))
        # Если получили +inf (ошибка в nrmse) или NaN, возвращаем худший float
        # (аналог минимального float32 или float('-inf')).
        if not np.isfinite(avg_score):
            return HPO_WORST_SCORE
        return avg_score

    def handle_pruning(self) -> None:
        """Настроить раннюю остановку (pruning) под выбранную валидацию (шаг 5).

        Нормализует конфигурацию ранней остановки и строит прайнер Optuna.
        Прайнинг применяется только когда есть естественные шаги оценки
        (k_fold / loo); для hold-out (train_test_split) промежуточных шагов
        нет, поэтому ранняя остановка не применяется (поведение
        задокументировано).
        """
        pruning_cfg = _normalize_pruning_config(self.pruning)
        self.pruner = None
        if pruning_cfg is not None:
            self.pruner = _build_pruner(pruning_cfg)
            log.info(
                "Early stopping enabled: strategy=%s, min_steps=%d, "
                "n_startup_trials=%d, reduction_factor=%d",
                pruning_cfg["strategy"],
                pruning_cfg["min_steps"],
                pruning_cfg["n_startup_trials"],
                pruning_cfg["reduction_factor"],
            )
        self.pruning_active = (
            self.pruner is not None and self.val_method_eff != "train_test_split"
        )
        if self.pruner is not None and not self.pruning_active:
            log.warning(
                "Early stopping is enabled, but validation strategy '%s' has no "
                "intermediate steps — pruning will not be applied.",
                self.val_method_eff,
            )

    def _objective(self, trial: Trial) -> float:
        """Целевая функция Optuna: evaluate_trial + handle_failure (issue #30).

        Разделяет оценку триала (``evaluate_trial``) и обработку его сбоя
        (``handle_failure``) на отдельные шаги. Счётчик подряд идущих
        фатальных сбоев сбрасывается при любом нефатальном исходе: успехе,
        отсечении прунером (optuna.TrialPruned из
        ``_evaluate_with_intermediate_reports``) или нефатальном ValueError.

        Args:
            trial (Trial): Объект текущего испытания Optuna.
        Returns:
            float: Значение целевой метрики на текущем наборе параметров.
        Raises:
            optuna.TrialPruned: Если триал отсечён (ошибкой или прайнером).
            InvalidAlgorithmError: При превышении лимита фатальных ошибок.
        """
        fatal_failure = False
        try:
            return self.evaluate_trial(trial)
        except (ValueError, MemoryError, RuntimeError, InvalidDataError) as err:
            fatal_failure = not isinstance(err, ValueError)
            self.handle_failure(trial, err)
            raise AssertionError(
                "unreachable"
            )  # pragma: no cover — handle_failure всегда бросает
        finally:
            # Любой нефатальный исход триала прерывает последовательность
            # фатальных сбоев: успешный возврат _score, отсечение прунером
            # (optuna.TrialPruned) или нефатальный ValueError. Без этого
            # сброса счётчик накапливал бы ошибки, разделённые штатными
            # триалами, и ложно дисквалифицировал алгоритм.
            if not fatal_failure:
                self.consecutive_fatal_failures = 0

    def handle_failure(self, trial: Trial, err: Exception) -> NoReturn:
        """Обработать сбой оценки триала по контракту исходов (шаг 6).

        Нефатальный ``ValueError`` просто отсекает триал (``optuna.TrialPruned``)
        без учёта в счётчике. ``MemoryError``/``RuntimeError``/``InvalidDataError``
        считаются фатальными: инкрементируют счётчик подряд идущих сбоев и при
        достижении лимита дисквалифицируют алгоритм (``InvalidAlgorithmError``).

        Args:
            trial (Trial): Текущий триал Optuna (для user_attr "fail_reason").
            err (Exception): Пойманное исключение оценки.
        Raises:
            optuna.TrialPruned: Всегда (нефатальный или фатальный сбой).
            InvalidAlgorithmError: Если достигнут лимит фатальных сбоев.
        """
        trial.set_user_attr("fail_reason", str(err))
        if isinstance(err, ValueError):
            raise optuna.TrialPruned() from err
        self.consecutive_fatal_failures += 1
        if self.consecutive_fatal_failures >= self.MAX_FATAL_FAILURES:
            raise InvalidAlgorithmError(
                f"Algorithm '{self.algo}' disqualified after "
                f"{self.consecutive_fatal_failures} consecutive fatal failures"
            ) from err
        raise optuna.TrialPruned() from err

    def select_best_trial(self) -> bool:
        """Выбрать лучший завершённый триал (шаг 7).

        Явный подсчёт состояний триалов (issue #32): наличие валидного
        результата определяется по числу завершённых (COMPLETED) триалов, а не
        по внутреннему поведению ``study.best_params`` (ValueError при
        отсутствии завершённых триалов — деталь реализации Optuna, на которую
        нельзя опираться). Счётчики логируются для наблюдаемости («нет
        валидного результата» должно быть объяснимо по логам).

        Returns:
            bool: True при наличии завершённого триала; False — контракт
                «нет валидного результата» (все триалы PRUNED/FAILED или
                список пуст).
        """
        trials = self.study.get_trials(deepcopy=False)
        n_completed = sum(
            1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        n_pruned = sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)
        n_failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)
        if self.pruning_active:
            log.info(
                "Early stopping: pruned %d of %d trials (algo=%s)",
                n_pruned,
                len(trials),
                self.algo,
            )
        log.info(
            "Trial states for algorithm '%s': completed=%d, pruned=%d, failed=%d",
            self.algo,
            n_completed,
            n_pruned,
            n_failed,
        )
        if n_completed == 0:
            # Контракт «нет валидного результата» (issues #13, #32): все
            # триалы отсечены (PRUNED), упали (FAILED) или список триалов
            # пуст. Возвращаем сплошной None вместо «магической» константы
            # HPO_WORST_SCORE, чтобы вызывающий код (_run_hpo) мог отличить
            # провал от валидного результата и исключить алгоритм из
            # кандидатов.
            log.warning(
                "Algorithm '%s' produced no completed trials "
                "(completed=%d, pruned=%d, failed=%d); returning no result",
                self.algo,
                n_completed,
                n_pruned,
                n_failed,
            )
            return False

        best_params = self.study.best_params
        self.best_score = self.study.best_value

        # Решение по отбору признаков берётся из study.best_params (mode='auto':
        # ключ "use_feature_selection" уже зафиксирован Optuna) либо из режима
        # конфигурации ('always' -> True, 'disabled' -> False). Для
        # isotonic_regression отбор принудительно игнорируется.
        best_apply_fs = bool(
            best_params.get(
                "use_feature_selection", self.fs_mode == FeatureSelectionMode.always
            )
        )
        if self.is_isotonic:
            best_apply_fs = False
            # Для isotonic_regression служебный ключ исключается и из
            # возвращаемого best_params (защита от initial_params, пронесённых
            # через enqueue_trial из предыдущей фазы с ключом
            # use_feature_selection).
            best_params = {
                k: v for k, v in best_params.items() if k != "use_feature_selection"
            }
        self.best_params = best_params
        # Параметры для конструктора базовой модели очищаются от служебного
        # ключа тюнера ("use_feature_selection" не является гиперпараметром
        # модели). Явное копирование: исходный study.best_params не мутируется,
        # а ключ гарантированно не попадает в конструктор модели (issue #11).
        clean_model_params = dict(self.study.best_params)
        clean_model_params.pop("use_feature_selection", None)
        self.clean_model_params = clean_model_params
        self.best_apply_fs = best_apply_fs
        return True

    def fit_best_model(self) -> tuple[Any | None, dict[str, Any] | None, float | None]:
        """Обучить финальную модель на всех предоставленных данных (шаг 8).

        Собирает пайплайн с лучшими параметрами (с учётом решения по отбору
        признаков из ``select_best_trial``) и обучает его на полном датасете.
        Если оверсэмплинг был включён, финальная модель тоже проходит его.

        Returns:
            tuple[Any | None, dict[str, Any] | None, float | None]: Кортеж
                (best_model, best_params, best_score).
        """
        best_model = self.assemble_estimator(
            create_model(self.algo, **self.clean_model_params),
            apply_fs=self.best_apply_fs,
        )
        best_model.fit(self.X, self.y)

        log.info(
            "Hyperopt: algo=%s | val=%s | score=%.5f | params=%s",
            self.algo,
            self.val_method_eff,
            self.best_score,
            self.best_params,
        )
        return best_model, self.best_params, self.best_score

    # ─────────────────── приватные помощники ────────────────────────────────

    def _validate_n_trials(self) -> bool:
        """Проверить корректность ``n_trials`` и контракт «нет валидного результата».

        Returns:
            bool: True, если нужно запускать Optuna; False — вернуть
                ``(None, None, None)`` без запуска.
        Raises:
            ValueError: Если ``n_trials`` отрицателен или не является целым.
        """
        if not isinstance(self.n_trials, int) or self.n_trials < 0:
            raise ValueError(
                f"n_trials must be a non-negative integer, got {self.n_trials}"
            )
        if self.n_trials == 0:
            # Пустой список триалов — контракт «нет валидного результата»
            # (issue #32): возвращаем сплошной None без запуска Optuna и без
            # исключений (в т.ч. для алгоритмов без search-space).
            log.warning(
                "Algorithm '%s' requested with n_trials=0; no trials to run — "
                "returning no result",
                self.algo_name,
            )
            return False
        return True

    def _fs_transformer_factory(self) -> FeatureSelector:
        """Создать свежий экземпляр селектора признаков для шага пайплайна.

        Каждый вызов возвращает новый ``FeatureSelector`` с детерминированным
        ``random_state`` (согласован с финальным обучением ModelTrainer),
        что гарантирует воспроизводимость отбора между триалами и потоками.
        При ``random_state=None`` (полностью недетерминированный запуск Optuna)
        отбор всё равно использует фиксированный seed 42: иначе каждый вызов
        фабрики создавал бы селектор с новым случайным зерном, и отбор был бы
        невоспроизводим даже при детерминированных остальных этапах пайплайна.
        """
        return FeatureSelector(
            method=self.fs_cfg.method.value,
            percentile=self.fs_cfg.percentile,
            min_features=self.fs_cfg.min_features,
            variance_threshold=self.fs_cfg.variance_threshold,
            n_estimators=self.fs_cfg.n_estimators,
            random_state=self.random_state if self.random_state is not None else 42,
        )


def optimize(
    algo_name: str,
    X: Any,
    y: Any,
    *,
    data_oversampling: bool = False,
    data_oversampling_multiplier: float = 1.0,
    data_oversampling_algorithm: str = "random",
    metric: str = "r2",
    # старый аргумент оставляем-для-совместимости
    val_method: ValidationStrategy | str = "k_fold",
    # alias, который шлёт training_engine.component
    validation_strategy: ValidationStrategy | str | None = None,
    n_folds: int = 5,
    n_trials: int = 50,
    random_state: int | None = 42,
    train_test_split_test_size: float = 0.2,
    space_overrides: dict[str, Callable[[Trial], dict[str, Any]]] | None = None,
    initial_params: dict[str, Any] | None = None,
    preprocessor: Any | None = None,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
    encoding: EncodingStrategy | None = None,
    preprocessing_override: PreprocessingOverride | dict[str, Any] | None = None,
    pruning: dict[str, Any] | None = None,
    high_cardinality_threshold: int | None = None,
    high_cardinality_encoding: EncodingStrategy | None = None,
    hashing_n_components: int = 16,
    target_encoding_smoothing: float = 20.0,
    target_encoding_fallback: float | None = None,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None, float | None]:
    """Запустить процесс оптимизации гиперпараметров модели с использованием Optuna.

    Функция автоматически выбирает стратегию валидации, настраивает пространство
    поиска параметров и обучает финальную модель на всех предоставленных данных.

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
        validation_strategy (ValidationStrategy | str | None): Алиас для
            ``val_method`` (имеет приоритет).
        n_folds (int): Количество фолдов для кросс-валидации. По умолчанию 5.
        n_trials (int): Количество итераций поиска (испытаний). По умолчанию 50.
        random_state (int | None): Состояние случайности для воспроизводимости.
            По умолчанию 42. При ``None`` случайность Optuna (TPE-сэмплер,
            разбиения) не фиксируется, однако шаг отбора признаков
            (``FeatureSelector``) всегда использует фиксированный seed 42,
            чтобы отбор оставался воспроизводимым между запусками.
        train_test_split_test_size (float): Размер теста для валидации через split.
            По умолчанию 0.2.
        space_overrides (dict | None): Словарь для переопределения пространств поиска.
        initial_params (dict[str, Any] | None): Гиперпараметры из предыдущей фазы
            для enqueue_trial. Позволяет сохранить монотонность улучшения между фазами HPO.
        preprocessor (Any | None): Готовый ``ColumnTransformer`` для предобработки
            признаков (категории -> one-hot, числа -> StandardScaler). Если передан,
            имеет приоритет над ``categorical_features``/``numerical_features``.
        categorical_features (list[str] | None): Имена категориальных колонок.
            Если ``preprocessor`` не передан, по ним строится препроцессор.
        numerical_features (list[str] | None): Имена числовых колонок.
            Используется вместе с ``categorical_features``.
        encoding (EncodingStrategy | None): Стратегия кодирования категорий
            ('one_hot', 'ordinal', 'target', 'frequency' или 'hashing').
            По умолчанию ``None`` — используется 'one_hot'. Применяется при
            построении препроцессора, когда ``preprocessor`` не передан.
        preprocessing_override (PreprocessingOverride | dict | None): Явное
            переопределение пресета предобработки признаков (FR-5). Задаётся
            частично или полностью; имеет приоритет над автоматическим выбором
            и применяется согласованно с финальным обучением (AC-6, AC-7).
        pruning (dict[str, Any] | None): Настройки ранней остановки (pruning).
            Словарь с полями: ``enable`` (bool, False по умолчанию),
            ``strategy`` ('median' | 'hyperband', 'median' по умолчанию),
            ``min_steps`` (int >= 1, 1 по умолчанию), ``n_startup_trials``
            (int >= 1, только для 'median', 5 по умолчанию),
            ``reduction_factor`` (int >= 2, только для 'hyperband', 3 по
            умолчанию). При ``enable=False`` или ``None`` поведение идентично
            текущему: каждый триал выполняется полностью. Для стратегий
            валидации без естественных шагов (train_test_split) прайнер не
            применяется.
        high_cardinality_threshold (int | None): Порог кардинальности для
            автоматического режима (>= 0). Колонки с числом уникальных значений
            строго больше порога кодируются ``high_cardinality_encoding``,
            остальные — ``encoding``. ``None`` — режим отключён.
        high_cardinality_encoding (EncodingStrategy | None): Стратегия
            кодирования high-cardinality колонок. Задаётся вместе с
            ``high_cardinality_threshold``.
        hashing_n_components (int): Число бинарных колонок на категориальную
            колонку при кодировании 'hashing' (>= 1).
        target_encoding_smoothing (float): Параметр сглаживания target encoding
            (>= 0).
        target_encoding_fallback (float | None): Fallback-значение target
            encoding для неизвестных категорий (None — глобальное среднее).
        feature_selection_cfg (FeatureSelectionCfg | dict | None): Настройки
            отбора признаков (issue #11). ``None`` — режим ``'disabled'``
            (отбор не применяется, поведение по умолчанию). В режиме
            ``'always'`` шаг ``feature_selector`` присутствует в каждом
            пайплайне триала; в режиме ``'disabled'`` — отсутствует; в
            режиме ``'auto'`` Optuna сама выбирает ``use_feature_selection``
            (True/False) на валидационных сплитах в конкуренции с
            гиперпараметрами моделей. Для ``isotonic_regression`` отбор
            принудительно отключается (одномерный алгоритм).

    Returns:
        tuple[Any, dict[str, Any], float]: Кортеж, содержащий:
            - best_model: Обученная модель с лучшими параметрами.
            - best_params: Словарь найденных оптимальных гиперпараметров
              (в режиме ``'auto'`` содержит служебный ключ
              ``use_feature_selection`` с решением по отбору).
            - best_score: Лучшее значение метрики на валидации.
            If no trial finished with a valid result (e.g., every trial was
            turned into ``optuna.TrialPruned`` by a non-fatal error, every
            trial failed, or ``n_trials`` is 0 — an empty trial list),
            ``(None, None, None)`` is returned — the failure signal for the
            caller (the algorithm is excluded from candidates).

    Raises:
        ValueError: Если ``n_trials`` является отрицательным целым числом
            (не целым числом). Значение ``n_trials=0`` допустимо и приводит
            к пустому списку триалов → ``(None, None, None)`` (issue #32).
        HyperoptError: Если для выбранного алгоритма не определено пространство
            поиска или передан невалидный ``feature_selection_cfg``.
    """
    # Тонкая обёртка: вся логика прогона вынесена в шаги _OptimizeRunner
    # (issue #30), чтобы optimize() оставалась читаемой точкой входа.
    return _OptimizeRunner(
        algo_name=algo_name,
        X=X,
        y=y,
        data_oversampling=data_oversampling,
        data_oversampling_multiplier=data_oversampling_multiplier,
        data_oversampling_algorithm=data_oversampling_algorithm,
        metric=metric,
        val_method=val_method,
        validation_strategy=validation_strategy,
        n_folds=n_folds,
        n_trials=n_trials,
        random_state=random_state,
        train_test_split_test_size=train_test_split_test_size,
        space_overrides=space_overrides,
        initial_params=initial_params,
        preprocessor=preprocessor,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        encoding=encoding,
        preprocessing_override=preprocessing_override,
        pruning=pruning,
        high_cardinality_threshold=high_cardinality_threshold,
        high_cardinality_encoding=high_cardinality_encoding,
        hashing_n_components=hashing_n_components,
        target_encoding_smoothing=target_encoding_smoothing,
        target_encoding_fallback=target_encoding_fallback,
        feature_selection_cfg=feature_selection_cfg,
    ).run()
