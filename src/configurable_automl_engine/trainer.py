"""Model Trainer: Универсальный оркестратор жизненного цикла регрессионных моделей.

Данный модуль инкапсулирует логику построения обучающих пайплайнов,
объединяя предобработку данных, балансировку выборок и обучение алгоритмов Scikit-learn.

Ключевой особенностью является полная автоматизация подготовки признаков:
система самостоятельно
классифицирует типы данных,
обрабатывает пропуски и масштабирует значения,
гарантируя корректную передачу данных в финальный оценщик.

Основные компоненты:

ModelTrainer: Главный класс-контейнер для обучения, валидации и деплоя моделей.
train_model: Функциональный интерфейс (Facade) для обеспечения
    обратной совместимости со старым API.
TrainingError: Специализированное исключение
    для унифицированной обработки сбоев обучения.

Особенности реализации:

Pipeline Integration: Использование imblearn.pipeline позволяет бесшовно интегрировать
    механизмы оверсэмплинга (SMOTE, ADASYN) непосредственно в процесс обучения.
Feature Selection: Опциональный шаг отбора признаков (FeatureSelector) встраивается
    строго между preprocessor и oversampler; активность управляется явным флагом
    (результат HPO) либо режимом конфигурации (always/disabled/auto). Режим ``auto``
    выполняет автономную проверку целесообразности на каждом fit() (два
    дополнительных обучения пайплайна, ~2–3x времени), поэтому в цикле HPO
    рекомендуется явный флаг ``feature_selection_active``.
Automated Feature Engineering: Встроенный ColumnTransformer автоматически применяет
    One-Hot кодирование для категорий и StandardScaler для числовых признаков.
Thread Safety: Использование рекурсивных блокировок (threading.RLock) гарантирует
    безопасность при обращении к состоянию модели из нескольких потоков.
Isotonic Support: Специальный трансформер IsotonicDataTransformer адаптирует
    многомерные данные под строгие требования алгоритмов изотонической регрессии.
Persistence: Встроенные методы save и load с поддержкой различных форматов сериализации
    (Pickle, Joblib) через систему артефактов."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.serialization_utils import (
    load_artifact,
    save_artifact,
)
from configurable_automl_engine.feature_selection import FeatureSelector
from configurable_automl_engine.oversampling import DataOversampler
from configurable_automl_engine.preprocessing import (
    EncodingStrategy,
    build_preprocessor,
)
from configurable_automl_engine.preprocessing_presets import (
    PreprocessingOverride,
    PreprocessingPreset,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
    FeatureSelectionMode,
)
from configurable_automl_engine.training_engine.metrics import (
    get_scorer_object,
    is_greater_better,
)
from configurable_automl_engine.training_engine.thread_pool import SharedDataFrame

from .models import (
    _ALIASES,
    create_model,
    requires_dense_input,
    resolve_algorithm_name,
)

__all__ = ["ModelTrainer", "TrainingError", "train_model"]

#: Минимальное число строк для автономной проверки отбора признаков (режим
#: ``auto``). Порог выбран так, чтобы обе части hold-out сплита (80/20)
#: содержали не менее 2 примеров: 10 * 0.8 = 8 и 10 * 0.2 = 2.
_MIN_AUTO_CHECK_SAMPLES = 10


def _sign_corrected_value(metric_name: str, raw_value: Any) -> float:
    """Привести «сырое» значение скорера к пользовательскому представлению.

    Логика работы:
    1. Для метрик-ошибок (RMSE, MAE, MSE и т.д.) sklearn возвращает
       отрицательное значение (так как оптимизирует максимизацию).
       Пользователю возвращается «честное» положительное значение.
    2. Для score-метрик (R² и т.д.) значение возвращается как есть.

    Args:
        metric_name (str): Имя метрики.
        raw_value (Any): «Сырое» значение, возвращённое объектом-скорером.

    Returns:
        float: Значение метрики в пользовательском представлении.
    """
    if not is_greater_better(metric_name):
        return float(abs(raw_value))
    return float(raw_value)


class TrainingError(RuntimeError):
    """Исключение для ошибок, связанных с данными или параметрами обучения."""


class IsotonicDataTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Трансформер для подготовки данных под IsotonicRegression.
    Выбирает указанный столбец и обеспечивает корректную обработку пропусков
    для соответствия требованиям алгоритма изотонической регрессии.
    Attributes:
        feature_index (int): Порядковый номер признака для извлечения.
        median_ (float): Вычисленное медианное значение для заполнения пропусков.
    """

    def __init__(self, feature_index: int = 0):
        """Инициализировать трансформер с указанием индекса целевого признака."""
        self.feature_index = feature_index

    def _get_dimensions(self, X: Any) -> tuple[int, int]:
        """Получить количество строк и столбцов входных данных
        в универсальном формате."""
        if hasattr(X, "shape"):
            return X.shape[0], (X.shape[1] if len(X.shape) > 1 else 1)
        n_rows = len(X)
        n_cols = len(X[0]) if n_rows > 0 and isinstance(X[0], (list, np.ndarray)) else 1
        return n_rows, n_cols

    def fit(self, X: Any, y: Any = None) -> IsotonicDataTransformer:
        """Вычислить медиану выбранного признака для последующей импутации пропусков."""
        _, n_cols = self._get_dimensions(X)
        idx = self.feature_index if self.feature_index < n_cols else 0

        X_col: pd.Series | np.ndarray

        if isinstance(X, pd.DataFrame):
            X_col = X.iloc[:, idx]
        elif isinstance(X, np.ndarray):
            X_col = X[:, idx] if len(X.shape) > 1 else X
        else:
            X_col = np.asarray(X)[:, idx]

        s_col = pd.Series(X_col)
        self.median_ = float(s_col.median()) if not s_col.isna().all() else 0.0
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Извлечь признак, заполнить пропуски
        и преобразовать в 2D-массив (n_samples, 1)."""
        try:
            # 1. Получаем метаданные без принудительного копирования в DataFrame
            # Используем логику из _extract_metadata, которую мы обсуждали ранее
            _n_rows, n_cols = self._get_dimensions(X)

            # 2. Валидация индекса признака
            if self.feature_index >= n_cols:
                raise TrainingError(
                    f"Data transformation error: feature_index {self.feature_index} "
                    f"out of bounds. Dataset has only {n_cols} columns."
                )
            # 3. Извлечение колонки (Zero-copy для numpy и pandas)
            # Если X - DataFrame, используем iloc. Если numpy - обычный слайсинг.
            X_col: pd.Series | np.ndarray
            if isinstance(X, pd.DataFrame):
                X_col = X.iloc[:, self.feature_index]
            elif isinstance(X, np.ndarray):
                X_col = X[:, self.feature_index]
            else:
                # Для списков или других типов приводим к numpy (минимальная копия)
                X_col = np.asarray(X)[:, self.feature_index]
            # 4. Логика обработки NaN
            # Проверяем, являются ли все значения в колонке NaN
            # Используем pd.Series для удобства вычисления медианы,
            # если это еще не Series
            s_col = X_col if isinstance(X_col, pd.Series) else pd.Series(X_col)

            if s_col.isna().all():
                raise TrainingError(
                    f"Column with index {self.feature_index} contains only NaN values."
                    " Training IsotonicRegression is impossible."
                )
            # Вычисляем медиану и заполняем пропуски
            fill_value = getattr(self, "median_", 0.0)
            X_imputed = s_col.fillna(fill_value)
            # Возвращаем результат в виде 2D массива, как ожидает scikit-learn
            result = X_imputed.to_numpy().reshape(-1, 1)
            return cast(np.ndarray, result)
        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            # Унификация сообщения об ошибке согласно тестам (#A8)
            raise TrainingError(f"Data transformation error: {e}")


class ModelTrainer:
    """Оркестратор процесса обучения и валидации регрессионных моделей.

    Автоматизирует построение пайплайна, включающего предобработку признаков
    (импутация, скалирование, кодирование), синтетическое увеличение выборки
    (oversampling) и обучение выбранного алгоритма с обязательной валидацией.
    Attributes:
        algorithm (str): Название алгоритма или алиас
            (например, 'elasticnet', 'xgboost').
        hyperparams (dict): Конфигурация гиперпараметров для инициализации модели..
        metric (str): Ключ метрики (r2, mse, mae) для оценки качества на валидации.
        random_state (int | None): Зерно для воспроизводимости разбиения и обучения.
        serialization_format (SerializationFormat): Формат сохранения (pickle/joblib).
        categorical_features (list[str] | None): Список колонок для One-Hot кодирования.
        numerical_features (list[str] | None): Список колонок для скалирования.
        id_column (str | None): Идентификатор, исключаемый из процесса обучения.
        encoding_strategy (str): Стратегия кодирования категорий
            ('one_hot', 'ordinal', 'target', 'frequency' или 'hashing').
            По умолчанию 'one_hot'.
        additional_metrics (Iterable[str] | None): Дополнительные метрики качества,
            рассчитываемые для обученной модели на том же наборе данных и тем же
            способом, что и основная метрика. Принимается любая итерируемая
            последовательность строк (list, tuple, set, генератор); имя метрики
            нормализуется к нижнему регистру. Носят информационный характер и
            не влияют на процесс обучения. Значения сохраняются в
            ``additional_scores`` после вызова fit().
        preprocessing_override (PreprocessingOverride | dict | None): Явное
            переопределение пресета предобработки признаков (FR-5). Задаётся
            частично или полностью и имеет приоритет над автоматическим выбором.
        os_enable (bool): Флаг активации балансировки/увеличения выборки.
        os_multiplier (float): Коэффициент генерации синтетических данных.
        os_algorithm (str): Алгоритм оверсэмплинга ('random', 'smote', 'adasyn').
        pipeline (Pipeline | None): Итоговый объект пайплайна после вызова fit().
        val_score (float | None): Значение метрики, полученное на hold-out выборке.
        additional_scores (dict[str, float]): Значения дополнительных метрик,
            рассчитанных для обученной модели (пустой словарь, если метрики
            не заданы).
        feature_names (list[str] | None): Список имен признаков,
            определенных при обучении.
        preprocessing_preset (PreprocessingPreset | None): Разрешённый пресет
            предобработки (заполняется в процессе fit()).
        feature_selection_cfg (FeatureSelectionCfg): Валидированная
            конфигурация отбора признаков (метод, процентиль, min_features
            и т.д.). По умолчанию ``FeatureSelectionCfg()`` (mode='disabled').
        feature_selection_active (bool | None): Явный флаг активности отбора
            признаков, переданный извне (результат триала HPO). Приоритетнее
            режима из ``feature_selection_cfg``; ``None`` — решение принимается
            по конфигурации.
        feature_selection_active_ (bool): Фактический статус отбора признаков
            в итоговом обучении (заполняется в fit(); False до первого fit).
        selected_features_mask_ (np.ndarray | None): Булева маска отобранных
            колонок после fit() (None, если отбор не применялся). Важно: маска
            имеет размерность пост-препроцессорной матрицы (после one-hot-
            раскрытия/скалирования), а не исходных колонок; сопоставление с
            исходными именами не выполняется — фильтрация прозрачна через
            pipeline.
        lock (threading.RLock): Рекурсивная блокировка для потокобезопасного обучения.
    """

    def __init__(
        self,
        algorithm: str = "elasticnet",
        hyperparams: dict[str, Any] | None = None,
        metric: str = "r2",
        random_state: int | None = 42,
        data_oversampling: bool = False,
        data_oversampling_multiplier: float = 1.0,
        data_oversampling_algorithm: str = "random",
        serialization_format: SerializationFormat = SerializationFormat.pickle,
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
        id_column: str | None = None,
        encoding_strategy: str = "one_hot",
        additional_metrics: Iterable[str] | None = None,
        preprocessing_override: PreprocessingOverride | dict[str, Any] | None = None,
        high_cardinality_threshold: int | None = None,
        high_cardinality_encoding: str | None = None,
        hashing_n_components: int = 16,
        target_encoding_smoothing: float = 20.0,
        target_encoding_fallback: float | None = None,
        feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
        feature_selection_active: bool | None = None,
    ):
        """Инициализировать тренер с параметрами модели и настройками предобработки.

        Args:
            feature_selection_cfg: Объект или словарь настроек отбора признаков
                (метод, процентиль, min_features и т.д.). ``None`` —
                ``FeatureSelectionCfg()`` (mode='disabled', отбор выключен).
            feature_selection_active: Явный булев флаг активности отбора
                (приоритет над режимом из ``feature_selection_cfg``). Передаётся
                внешним циклом HPO, который уже решил, нужен ли отбор
                (интеграция с component.py/tuner.py отслеживается в follow-up
                issue). ``None`` — решение по конфигурации.
        """

        self.logger = logging.getLogger(__name__)

        self.lock = threading.RLock()

        # Проверка алгоритма
        if not isinstance(algorithm, str):
            raise TrainingError(f"Invalid algorithm: {algorithm!r}")
        self.algorithm = algorithm.lower()

        # hyperparams
        if hyperparams is not None:
            if not isinstance(hyperparams, dict):
                raise TrainingError("hyperparams must be a dictionary")
            self.hyperparams = hyperparams
        else:
            self.hyperparams = {}

        # Остальные параметры
        self.metric = metric.lower()
        self.random_state = random_state
        self.serialization_format = serialization_format
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.id_column = id_column
        # Стратегия кодирования категорий: 'one_hot' (по умолчанию), 'ordinal',
        # 'target', 'frequency' или 'hashing'. Хранится как строковый примитив,
        # чтобы сохранять picklable-совместимость.
        if encoding_strategy not in (
            "one_hot",
            "ordinal",
            "target",
            "frequency",
            "hashing",
        ):
            raise TrainingError(
                f"Unknown encoding_strategy: {encoding_strategy!r}. "
                "Expected one of ('one_hot', 'ordinal', 'target', 'frequency', "
                "'hashing')."
            )
        self.encoding_strategy: EncodingStrategy = cast(
            EncodingStrategy, encoding_strategy
        )

        # Параметры high-cardinality кодирования и новых стратегий (issue #20).
        # Валидируются сразу, чтобы некорректные значения отклонялись на этапе
        # инициализации, до запуска обучения.
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.hashing_n_components = hashing_n_components
        self.target_encoding_smoothing = target_encoding_smoothing
        self.target_encoding_fallback = target_encoding_fallback
        if (high_cardinality_threshold is None) != (high_cardinality_encoding is None):
            raise TrainingError(
                "high_cardinality_threshold and high_cardinality_encoding must "
                "be set together (both provided or both None)."
            )
        if high_cardinality_threshold is not None and high_cardinality_threshold < 0:
            raise TrainingError(
                "high_cardinality_threshold must be >= 0, got "
                f"{high_cardinality_threshold}."
            )
        if high_cardinality_encoding is not None and high_cardinality_encoding not in (
            "one_hot",
            "ordinal",
            "target",
            "frequency",
            "hashing",
        ):
            raise TrainingError(
                f"Unknown high_cardinality_encoding: {high_cardinality_encoding!r}."
            )
        if hashing_n_components < 1:
            raise TrainingError(
                f"hashing_n_components must be >= 1, got {hashing_n_components}."
            )
        if target_encoding_smoothing < 0:
            raise TrainingError(
                f"target_encoding_smoothing must be >= 0, got "
                f"{target_encoding_smoothing}."
            )

        # ---------- additional (informational) metrics ----------
        # Допускается любая итерируемая последовательность строк (list, tuple,
        # set, генератор и т.п.). Строки-скаляры и отображения (dict) отклоняются:
        # строку нельзя разбирать посимвольно в имена метрик, а итерирование
        # dict по ключам было бы неявным и неожиданным поведением.
        # Итерируемый объект материализуется ровно один раз, чтобы генераторы
        # и другие одноразовые итераторы валидировались корректно.
        if additional_metrics is not None and isinstance(
            additional_metrics, (str, bytes, Mapping)
        ):
            raise TrainingError("additional_metrics must be an iterable of strings")
        normalized_metrics = list(additional_metrics or [])
        if not all(isinstance(m, str) for m in normalized_metrics):
            raise TrainingError("additional_metrics must be an iterable of strings")
        self.additional_metrics: list[str] = [m.lower() for m in normalized_metrics]
        self.additional_scores: dict[str, float] = {}

        # Явное переопределение пресета предобработки (FR-5): валидируем сразу,
        # чтобы некорректные значения отклонялись на этапе инициализации.
        self.preprocessing_override: PreprocessingOverride | None = (
            self._validate_preprocessing_override(preprocessing_override)
        )
        # Разрешённый пресет — заполняется в _build_preprocessor во время fit().
        self.preprocessing_preset: PreprocessingPreset | None = None

        # ---------- feature selection (issue #31) ----------
        # Валидируем конфигурацию отбора признаков сразу, чтобы некорректные
        # значения (percentile=150, неизвестный method/mode, лишние ключи)
        # отклонялись на этапе инициализации, до тяжёлого обучения.
        self.feature_selection_cfg: FeatureSelectionCfg = (
            self._validate_feature_selection_cfg(feature_selection_cfg)
        )
        # Явный флаг из HPO/component.py (приоритет над конфигом); None —
        # решение принимается в fit() по режиму из конфигурации. Тип строго
        # bool | None: строка "false" из конфигурации/YAML не должна молча
        # приводиться к True через bool("false") в _resolve_feature_selection_active
        # (ревью PR #8), поэтому небулевые значения отклоняются на этапе
        # инициализации, как и остальные параметры конструктора.
        if feature_selection_active is not None and not isinstance(
            feature_selection_active, bool
        ):
            raise TrainingError(
                "feature_selection_active must be a bool or None, got "
                f"{type(feature_selection_active).__name__}"
            )
        self.feature_selection_active: bool | None = feature_selection_active
        # Фактический статус отбора в итоговом обучении (заполняется в fit()).
        self.feature_selection_active_: bool = False
        # Булева маска отобранных колонок пост-препроцессорной размерности.
        self.selected_features_mask_: np.ndarray | None = None

        # ---------- oversampling ----------
        self.os_enable = data_oversampling
        self.os_multiplier = data_oversampling_multiplier
        self.os_algorithm = data_oversampling_algorithm
        if self.os_multiplier < 1:
            raise TrainingError("data_oversampling_multiplier must be ≥ 1")
        if self.os_algorithm not in {"random", "random_with_noise", "smote", "adasyn"}:
            raise TrainingError("Unknown data_oversampling_algorithm")

        # Поля, которые заполняются после fit(...)
        self.pipeline: Pipeline | None = None
        self.base_model: Any = None
        self.val_score: float | None = None
        self.feature_names: list[str] | None = None
        self._last_train_y: pd.Series | None = None
        self._last_val_y: pd.Series | None = None

        for attr_name, features in [
            ("categorical_features", categorical_features),
            ("numerical_features", numerical_features),
        ]:
            if features is not None and (
                not isinstance(features, list)
                or not all(isinstance(f, str) for f in features)
            ):
                raise TrainingError(
                    f"Parameter {attr_name} must be a list of strings (column names)"
                )

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Проверить наличие всех заданных имен признаков в переданном DataFrame."""

        specified_features = (self.categorical_features or []) + (
            self.numerical_features or []
        )
        missing = [col for col in specified_features if col not in X.columns]
        if missing:
            raise TrainingError(f"Specified columns not found in data: {missing}")

    def _detect_feature_types(self, X: pd.DataFrame, target_column: str) -> None:
        """Автоматически классифицировать колонки на числовые и категориальные."""
        with self.lock:
            # Если оба списка уже заполнены пользователем — просто валидируем
            if (
                self.categorical_features is not None
                and self.numerical_features is not None
            ):
                self._validate_features(X)
                return
            # Определяем список колонок для анализа (исключаем таргет и ID)
            exclude = {target_column}
            if self.id_column:
                exclude.add(self.id_column)

            df_to_analyze = X.drop(columns=list(exclude & set(X.columns)))
            # Авто-определение категориальных (object, category, bool)
            if self.categorical_features is None:
                self.categorical_features = df_to_analyze.select_dtypes(
                    include=["object", "str", "category", "bool"]
                ).columns.tolist()
            # Авто-определение числовых (все оставшиеся типы 'number')
            if self.numerical_features is None:
                self.numerical_features = df_to_analyze.select_dtypes(
                    include=["number"]
                ).columns.tolist()
            self.logger.info(
                f"Auto-detected features: {len(self.categorical_features)} cat, "
                f"{len(self.numerical_features)} num."
            )

    def _validate_preprocessing_override(
        self,
        override: PreprocessingOverride | dict[str, Any] | None,
    ) -> PreprocessingOverride | None:
        """Валидировать и нормализовать пользовательское переопределение пресета.

        Args:
            override: Словарь или модель :class:`PreprocessingOverride`.

        Returns:
            Нормализованная модель переопределения либо ``None``.

        Raises:
            TrainingError: Если передан объект неподдерживаемого типа или
                словарь с некорректными значениями.
        """
        if override is None:
            return None
        if isinstance(override, PreprocessingOverride):
            return override
        try:
            if isinstance(override, dict):
                return PreprocessingOverride.model_validate(override)
        except Exception as e:
            raise TrainingError(f"Invalid preprocessing_override: {e}") from e
        raise TrainingError(
            "preprocessing_override must be a dict or PreprocessingOverride, "
            f"got {type(override).__name__}"
        )

    def _validate_feature_selection_cfg(
        self,
        cfg: FeatureSelectionCfg | dict[str, Any] | None,
    ) -> FeatureSelectionCfg:
        """Валидировать и нормализовать конфигурацию отбора признаков.

        Логика по образцу :meth:`_validate_preprocessing_override`:
        ``None`` → дефолт ``FeatureSelectionCfg()`` (mode='disabled');
        ``dict`` → ``FeatureSelectionCfg.model_validate(...)``, ошибка → TrainingError;
        ``FeatureSelectionCfg`` → как есть;
        иной тип → TrainingError.

        Args:
            cfg: Словарь или модель :class:`FeatureSelectionCfg`.

        Returns:
            Нормализованная модель конфигурации отбора признаков.

        Raises:
            TrainingError: Если передан объект неподдерживаемого типа или
                словарь с некорректными значениями.
        """
        if cfg is None:
            return FeatureSelectionCfg()
        if isinstance(cfg, FeatureSelectionCfg):
            return cfg
        try:
            if isinstance(cfg, dict):
                return FeatureSelectionCfg.model_validate(cfg)
        except Exception as e:
            raise TrainingError(f"Invalid feature_selection_cfg: {e}") from e
        raise TrainingError(
            "feature_selection_cfg must be a dict or FeatureSelectionCfg, "
            f"got {type(cfg).__name__}"
        )

    def _resolve_preprocessing_preset(self) -> PreprocessingPreset:
        """Разрешить пресет предобработки для текущего алгоритма.

        Автоматический выбор по таблице соответствия
        (:func:`resolve_preprocessing_preset`) + применение явного
        пользовательского переопределения (FR-5).

        Returns:
            Итоговый пресет предобработки.
        """
        override = getattr(self, "preprocessing_override", None)
        return resolve_preprocessing_preset(self.algorithm, override)

    def _extract_metadata(self, X: Any) -> list[str] | None:
        """Получить имена колонок из различных структур данных
        без копирования содержимого."""
        if hasattr(X, "get_data_info"):
            return cast(list[str], X.get_data_info()["columns"])
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        if hasattr(X, "shape") and len(X.shape) > 1:
            return [str(f"col_{i}") for i in range(X.shape[1])]
        return None

    def __getstate__(self) -> dict[str, Any]:
        """Управление состоянием объекта для корректной сериализации
        (исключение lock и logger)."""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        if "lock" in state:
            del state["lock"]
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Восстановить состояние объекта после десериализации,
        инициализируя потокобезопасный lock и логгер"""
        self.__dict__.update(state)
        # Re-initialize the lock after unpickling
        self.lock = threading.RLock()
        # Re-initialize logger if necessary
        self.logger = logging.getLogger(__name__)

    def _build_preprocessor(self, feature_names: list[str]) -> ColumnTransformer:
        """Сконструировать ColumnTransformer для раздельной обработки типов данных.

        Делегирует сборку в общий модуль :mod:`preprocessing`, который
        используется также фазой HPO в ``tuner.optimize`` (DRY, единая логика).
        Стратегия обработки числовых признаков (импутация и масштабирование)
        выбирается автоматически по алгоритму через адаптивный пресет
        предобработки и логируется (AC-9).
        """
        # Логируем предупреждение, если ни одна колонка не совпала
        cat_indices = [
            feature_names.index(col)
            for col in (self.categorical_features or [])
            if col in feature_names
        ]
        num_indices = [
            feature_names.index(col)
            for col in (self.numerical_features or [])
            if col in feature_names
        ]
        if not cat_indices and not num_indices:
            self.logger.warning(
                "No features matched for preprocessing. Defaulting to passthrough."
            )
        # Обратная совместимость: модели, сериализованные до появления
        # параметра encoding_strategy (атрибут отсутствует у распикленных
        # объектов), по умолчанию обрабатываются как one_hot. Новые параметры
        # кодирования (issue #20) также получают значения по умолчанию через
        # getattr, чтобы старые сохранённые модели продолжали работать без
        # изменений поведения.
        encoding: EncodingStrategy = cast(
            EncodingStrategy, getattr(self, "encoding_strategy", "one_hot")
        )
        hc_threshold: int | None = getattr(self, "high_cardinality_threshold", None)
        hc_encoding_raw: str | None = getattr(self, "high_cardinality_encoding", None)
        hc_encoding: EncodingStrategy | None = (
            cast(EncodingStrategy, hc_encoding_raw)
            if hc_encoding_raw is not None
            else None
        )
        hashing_n_components: int = getattr(self, "hashing_n_components", 16)
        target_smoothing: float = getattr(self, "target_encoding_smoothing", 20.0)
        target_fallback: float | None = getattr(self, "target_encoding_fallback", None)
        # Автоматический выбор пресета предобработки по алгоритму (FR-1)
        # с учётом явного переопределения (FR-5). Пресет разрешается один раз
        # до обучения (производительность) и фиксируется для наблюдаемости (AC-9).
        preset = self._resolve_preprocessing_preset()
        self.preprocessing_preset = preset
        self.logger.info(
            "Resolved preprocessing preset for algorithm '%s': %s",
            self.algorithm,
            preset,
        )
        # Алгоритмы GPR/Isotonic/ARD отвергают разреженные матрицы: при
        # hashing-кодировании (sparse-выход) препроцессор обязан вернуть
        # плотную матрицу (issue: OOM fix, force_dense_output).
        force_dense = requires_dense_input(self.algorithm)
        return build_preprocessor(
            feature_names,
            self.categorical_features or [],
            self.numerical_features or [],
            encoding=encoding,
            imputation_strategy=preset.imputation_strategy,
            scaling=preset.scaling,
            high_cardinality_threshold=hc_threshold,
            high_cardinality_encoding=hc_encoding,
            hashing_n_components=hashing_n_components,
            target_encoding_smoothing=target_smoothing,
            target_encoding_fallback=target_fallback,
            random_state=self.random_state,
            force_dense_output=force_dense,
        )

    def _prepare_data(self, X: Any, y: Any) -> tuple[Any, Any]:
        """Валидировать входные типы данных
        и разделить признаки от целевой переменной."""
        try:
            self.feature_names = self._extract_metadata(X)

            self._detected_dtypes = X.dtypes if isinstance(X, pd.DataFrame) else None
            # 1. Проверка типов
            valid_types = (pd.DataFrame, pd.Series, np.ndarray, SharedDataFrame)
            if not isinstance(X, valid_types):
                raise TrainingError(f"Unsupported data type: {type(X)}")

            X_obj: pd.DataFrame | pd.Series | np.ndarray
            y_obj: pd.Series | np.ndarray

            if isinstance(y, str):
                if not isinstance(X, pd.DataFrame):
                    raise TrainingError(
                        f"Target column '{y}' specified, but X is not a DataFrame"
                    )
                self.feature_names = [col for col in X.columns if col != y]
                X_obj = X[self.feature_names]
                y_obj = X[y]
            else:
                X_obj = X.get_view() if isinstance(X, SharedDataFrame) else X
                if isinstance(y, (pd.Series, np.ndarray)):
                    y_obj = y
                elif isinstance(y, pd.DataFrame):
                    y_obj = y.iloc[:, 0]
                elif hasattr(y, "get_view"):  # Поддержка SharedDataFrame для y
                    y_obj = y.get_view().iloc[:, 0]
                else:
                    y_obj = np.asarray(y)
            # Консолидированная валидация
            n_samples = X_obj.shape[0] if hasattr(X_obj, "shape") else len(X_obj)
            if n_samples == 0:
                raise TrainingError("Data is empty")
            nx = X_obj.shape[0] if hasattr(X_obj, "shape") else len(X_obj)
            ny = y_obj.shape[0] if hasattr(y_obj, "shape") else len(y_obj)

            if nx != ny:
                raise TrainingError(f"Mismatched samples: X has {nx}, y has {ny}")

            return X_obj, y_obj
        except (ValueError, TypeError, IndexError) as e:
            if str(e) == "Data is empty":
                raise
            raise TrainingError(f"Data transformation error: {e}")

    def _resolve_feature_selection_active(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
    ) -> bool:
        """Разрешить фактическую активность отбора признаков для fit().

        Приоритет (строго в этом порядке):
        1. IsotonicRegression — отбор принудительно отключается (алгоритм
           строго одномерный, использует собственный ``IsotonicDataTransformer``);
        2. явный флаг ``feature_selection_active`` (из HPO/component.py) —
           имеет приоритет над конфигурацией;
        3. режим из ``feature_selection_cfg.mode``: ``always`` → True,
           ``disabled`` → False, ``auto`` → Standalone Auto-Check.

        Обратная совместимость со старыми pickle: атрибуты читаются через
        ``getattr`` (аналогично ``encoding_strategy`` в ``_build_preprocessor``),
        отсутствующие атрибуты дают поведение disabled.

        Args:
            X_train: Обучающая матрица признаков (для режима ``auto``).
            y_train: Целевая переменная (для режима ``auto``).
            preprocessor: ``ColumnTransformer`` предобработки (для режима ``auto``).
            base_model: Финальный регрессор (для режима ``auto``).

        Returns:
            ``True``, если шаг отбора должен присутствовать в пайплайне.
        """
        # 1. Isotonic-байпас: одномерный алгоритм несовместим с отбором признаков.
        algo_key = resolve_algorithm_name(self.algorithm)
        if algo_key == "isotonic_regression":
            self.logger.debug(
                "Feature selection is forcibly disabled for IsotonicRegression "
                "(univariate algorithm with its own data transformer)."
            )
            return False

        # 2. Явный флаг из HPO/component.py имеет приоритет над конфигом.
        explicit_flag = getattr(self, "feature_selection_active", None)
        if explicit_flag is not None:
            return bool(explicit_flag)

        # 3. Решение по режиму конфигурации.
        cfg = getattr(self, "feature_selection_cfg", None)
        if not isinstance(cfg, FeatureSelectionCfg):
            cfg = FeatureSelectionCfg()
        mode = cfg.mode
        if mode == FeatureSelectionMode.always:
            return True
        if mode == FeatureSelectionMode.disabled:
            return False
        # mode == auto: автономная проверка целесообразности отбора.
        return self._run_feature_selection_auto_check(
            X_train, y_train, preprocessor, base_model
        )

    def _run_feature_selection_auto_check(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
    ) -> bool:
        """Выполнить Standalone Auto-Check целесообразности отбора признаков.

        Режим ``auto`` используется при прямом вызове ``ModelTrainer.fit()``
        без HPO. Логика:
        1. Hold-out сплит 80/20 с фиксированным ``random_state``.
        2. Два контрольных пайплайна (без селектора и с селектором) обучаются
           на train-части; скоринг на hold-out через
           ``get_scorer_object(self.metric)`` + ``_sign_corrected_value``.
        3. Направленное сравнение с учётом ``is_greater_better(self.metric)``:
           отбор включается только при строгом улучшении качества.
        4. Fail-safe: малые данные, ошибки сплита/скоринга или None/nan-скоры
           дают WARNING и возвращают ``False`` — проверка не роняет fit().

        Стоимость: каждый ``fit()`` в режиме ``auto`` дополнительно обучает
        два полных пайплайна на 80% данных — для тяжёлых алгоритмов это
        ~2–3x времени обычного обучения. Режим рассчитан на прямые вызовы
        без HPO; в цикле HPO активность отбора должна передаваться явным
        флагом ``feature_selection_active``, чтобы проверка не выполнялась
        для каждого триала.

        Побочных эффектов нет: метод не трогает ``self.pipeline``,
        ``self.val_score`` и ``self.feature_names``.

        Args:
            X_train: Обучающая матрица признаков.
            y_train: Целевая переменная.
            preprocessor: ``ColumnTransformer`` предобработки.
            base_model: Финальный регрессор.

        Returns:
            ``True``, если качество с отбором строго выше, иначе ``False``.
        """
        n_samples = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
        if n_samples < _MIN_AUTO_CHECK_SAMPLES:
            self.logger.warning(
                "Standalone feature selection auto-check skipped: only %d "
                "samples available, need at least %d. Feature selection "
                "disabled.",
                n_samples,
                _MIN_AUTO_CHECK_SAMPLES,
            )
            return False
        try:
            # Детерминированность hold-out сплита: при random_state=None
            # тренера (нефиксированное основное обучение) сплит всё равно
            # обязан быть воспроизводимым, иначе решение об отборе будет
            # меняться между вызовами fit() (ревью PR #8). Используем фикс.
            # зерно-фолбэк, не трогая конфигурацию основного обучения.
            check_random_state = (
                self.random_state if self.random_state is not None else 42
            )
            X_sub, X_hold, y_sub, y_hold = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=check_random_state,
            )
            # Контрольные пайплайны изолированы друг от друга и от финального
            # обучения: sklearn Pipeline не клонирует переданные шаги, поэтому
            # общие экземпляры preprocessor/base_model мутировали бы состояние
            # соседнего пайплайна при повторном fit.
            pipe_full = Pipeline(
                self._assemble_steps(
                    clone(preprocessor),
                    clone(base_model),
                    feature_selection_active=False,
                )
            )
            pipe_reduced = Pipeline(
                self._assemble_steps(
                    clone(preprocessor),
                    clone(base_model),
                    feature_selection_active=True,
                )
            )
            pipe_full.fit(X_sub, y_sub)
            pipe_reduced.fit(X_sub, y_sub)

            scorer = cast(Callable[..., Any], get_scorer_object(self.metric))
            raw_full = scorer(pipe_full, X_hold, y_hold)
            raw_reduced = scorer(pipe_reduced, X_hold, y_hold)
            if (
                raw_full is None
                or raw_reduced is None
                or not np.isfinite(float(raw_full))
                or not np.isfinite(float(raw_reduced))
            ):
                self.logger.warning(
                    "Standalone feature selection auto-check produced "
                    "non-finite scores (full=%r, reduced=%r). Feature "
                    "selection disabled.",
                    raw_full,
                    raw_reduced,
                )
                return False

            score_full = _sign_corrected_value(self.metric, raw_full)
            score_reduced = _sign_corrected_value(self.metric, raw_reduced)
            greater_better = is_greater_better(self.metric)
            if greater_better:
                active = score_reduced > score_full
            else:
                active = score_reduced < score_full
            self.logger.info(
                "Standalone feature selection auto-check: score_full=%.4f, "
                "score_reduced=%.4f -> active=%s",
                score_full,
                score_reduced,
                active,
            )
            return active
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                "Standalone feature selection auto-check failed: %s. "
                "Feature selection disabled.",
                e,
            )
            return False

    def _assemble_steps(
        self,
        preprocessor: ColumnTransformer,
        base_model: Any,
        *,
        feature_selection_active: bool,
    ) -> list[tuple[str, Any]]:
        """Собрать список шагов обучающего пайплайна.

        Строгий порядок шагов (issue #31):
        ``preprocessor → [feature_selector] → [oversampler] → model``.
        Селектор признаков встраивается строго между препроцессором и
        оверсэмплером, чтобы синтетические строки генерировались только для
        информативных признаков.

        Args:
            preprocessor: ``ColumnTransformer`` предобработки признаков.
            base_model: Финальный регрессор.
            feature_selection_active: Флаг включения шага отбора признаков.

        Returns:
            Список кортежей ``(name, transformer)`` для построения Pipeline.
        """
        steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
        if feature_selection_active:
            # Обратная совместимость со старыми pickle: конфигурация читается
            # через getattr (аналогично encoding_strategy в _build_preprocessor),
            # отсутствие атрибута даёт дефолт FeatureSelectionCfg().
            cfg = getattr(self, "feature_selection_cfg", None)
            if not isinstance(cfg, FeatureSelectionCfg):
                cfg = FeatureSelectionCfg()
            selector = FeatureSelector(
                method=cfg.method.value,
                percentile=cfg.percentile,
                min_features=cfg.min_features,
                variance_threshold=cfg.variance_threshold,
                n_estimators=cfg.n_estimators,
                # Фикс-фолбэк зерна отбора (согласован с тюнером и
                # _check_auto_feature_selection): при random_state=None
                # основного обучения селектор всё равно обязан использовать
                # фиксированный seed 42, иначе отбор не воспроизводим между
                # вызовами fit() и расходится с фазой HPO.
                random_state=self.random_state if self.random_state is not None else 42,
            )
            steps.append(("feature_selector", selector))
        if self.os_enable:
            oversampler = DataOversampler(
                algorithm=self.os_algorithm,
                multiplier=self.os_multiplier,
            )
            steps.append(("oversampler", oversampler))
        steps.append(("model", base_model))
        return steps

    def _fit_internal(
        self,
        X_train: Any,
        y_train: Any,
        preprocessor: ColumnTransformer,
        base_model: Any,
        *,
        feature_selection_active: bool = False,
    ) -> Pipeline:
        """Собрать финальный пайплайн и запустить
        процесс обучения на подготовленных данных.

        Args:
            X_train: Обучающая матрица признаков.
            y_train: Целевая переменная.
            preprocessor: ``ColumnTransformer`` предобработки признаков.
            base_model: Финальный регрессор.
            feature_selection_active: Флаг включения шага отбора признаков
                (keyword-only; по умолчанию ``False`` — поведение без отбора).
        """
        # 0) Сбрасываем фактический статус отбора и маску отобранных колонок.
        # Если pipeline.fit упадёт, у тренера не должны остаться значения от
        # предыдущего обучения (рекомендация ревью PR #8). После успешного
        # фита значения перезаписываются в шаге 4.
        self.feature_selection_active_ = False
        self.selected_features_mask_ = None
        # 1) Формируем шаги пайплайна
        steps = self._assemble_steps(
            preprocessor,
            base_model,
            feature_selection_active=feature_selection_active,
        )
        # 2) Инициализируем пайплайн
        # Используем Pipeline из imblearn, чтобы шаги ресэмплинга работали корректно
        self.pipeline = Pipeline(steps=steps)
        # 3) Выполняем обучение
        try:
            self.pipeline.fit(X_train, y_train)
        except (ValueError, TypeError):
            # Пробрасываем валидационные ошибки напрямую, чтобы тесты могли их поймать
            # Это критично для тестов, проверяющих некорректные гиперпараметры
            raise
        except TrainingError:
            raise
        except Exception as e:
            # Все остальные системные ошибки оборачиваем в TrainingError
            self.logger.debug("Unexpected pipeline fit failure trace:", exc_info=True)
            self.logger.error(f"Unexpected error in pipeline fit: {e}")
            raise TrainingError(f"Internal training failure: {e}")
        # 4) Фиксируем фактический статус отбора и маску отобранных колонок.
        # Маска берётся с шага селектора внутри пайплайна и имеет размерность
        # пост-препроцессорной матрицы (после one-hot-раскрытия/скалирования).
        self.feature_selection_active_ = feature_selection_active
        selector_step = self.pipeline.named_steps.get("feature_selector")
        if selector_step is not None and hasattr(selector_step, "support_"):
            self.selected_features_mask_ = np.asarray(
                selector_step.support_, dtype=bool
            )
        else:
            self.selected_features_mask_ = None
        return self.pipeline

    def fit(self, X: Any, y: Any) -> ModelTrainer:
        """Запустить полный цикл подготовки данных, обучения модели и оценки метрик.
        Включает обязательное разбиение выборки для оценки качества."""

        with self.lock:
            # Этап 0: Сбрасываем фактический статус отбора и маску отобранных
            # колонок в начале fit(), до какой-либо обработки данных. Если
            # обучение упадёт на любом этапе (подготовка данных, разрешение
            # активности, pipeline.fit), у тренера не останутся значения от
            # предыдущего обучения (ревью PR #8). Дублирующий сброс в
            # _fit_internal сохраняется для прямых вызовов этого метода.
            self.feature_selection_active_ = False
            self.selected_features_mask_ = None

            # Этап 1: Валидация и подготовка данных
            X_prepared, y_s = self._prepare_data(X, y)

            n_samples = (
                X_prepared.shape[0] if hasattr(X_prepared, "shape") else len(X_prepared)
            )
            if n_samples < 2:
                raise TrainingError("Insufficient records for training")

            if isinstance(X_prepared, pd.DataFrame):
                self._detect_feature_types(X_prepared, target_column="")
            else:
                if self.categorical_features is None:
                    self.categorical_features = []
                if self.numerical_features is None:
                    n_cols = X_prepared.shape[1] if len(X_prepared.shape) > 1 else 1
                    self.numerical_features = [f"col_{i}" for i in range(n_cols)]
                if self.feature_names is None:
                    self.feature_names = self.numerical_features

            # Этап 2: Определение ключа алгоритма
            # (перенесено выше для использования в препроцессоре)
            _ALIASES.get(self.algorithm, self.algorithm)

            # Этап 3: Создаём модель через фабрику.
            # Выполняется ДО сборки препроцессора, чтобы неизвестный алгоритм
            # (опечатка в имени) отклонялся здесь с понятным сообщением,
            # а не маскировался default-пресетом предобработки.
            try:
                model_kwargs = self.hyperparams.copy()
                model_kwargs.pop("feature_index", None)
                base_model = create_model(self.algorithm, **model_kwargs)
            except (ValueError, ImportError) as e:
                raise TrainingError(f"Error creating model: {e}")

            # Этап 4: Создаём препроцессор через делегирование
            if self.feature_names is None:
                # Если имен нет, пытаемся извлечь их снова или создаем пустой список
                self.feature_names = self._extract_metadata(X_prepared) or []
            preprocessor = self._build_preprocessor(self.feature_names)

            # Этап 4.5: Разрешаем активность отбора признаков (issue #31).
            # Приоритет: isotonic-байпас → явный флаг → режим конфигурации
            # (always/disabled) → standalone auto-check для mode='auto'.
            feature_selection_active = self._resolve_feature_selection_active(
                X_prepared, y_s, preprocessor, base_model
            )

            # Этап 5: Сборка и обучение пайплайна
            # Здесь автоматически применится оверсэмплинг, если он включен,
            # и отбор признаков, если он разрешён на этапе 4.5.
            self.pipeline = self._fit_internal(
                X_prepared,
                y_s,
                preprocessor,
                base_model,
                feature_selection_active=feature_selection_active,
            )

            # Этап 6: Валидация и расчет метрик (финальный шаг)
            try:
                # 1. Получаем объект-скорер
                scorer = cast(Callable[..., Any], get_scorer_object(self.metric))

                # 2. Вычисляем raw_score (для ошибок sklearn вернет отрицательное число)
                raw_score = scorer(self.pipeline, X_prepared, y_s)
                if raw_score is None:
                    raise TrainingError("Scorer returned None")

                # 3. Инвертируем знак обратно, если это метрика-ошибка
                # (RMSE, MAE и т.д.)
                # Чтобы в val_score всегда лежало "честное"
                # положительное значение ошибки
                self.val_score = _sign_corrected_value(self.metric, raw_score)
                self.logger.debug(
                    f"Metric calculation: raw={raw_score:.4f},"
                    f" final val_score={self.val_score:.4f} "
                    f"(greater_is_better={is_greater_better(self.metric)})"
                )

                # 4. Дополнительные (информационные) метрики: рассчитываются для
                # финальной модели на том же наборе данных и тем же способом, что
                # и основная метрика. Они не влияют на ход обучения, выбор модели
                # и сохранение артефакта: при сбое отдельной метрики она
                # пропускается с предупреждением, обучение продолжается.
                self.additional_scores = {}
                for mname in self.additional_metrics:
                    try:
                        m_scorer = cast(Callable[..., Any], get_scorer_object(mname))
                        m_raw = m_scorer(self.pipeline, X_prepared, y_s)
                        if m_raw is None:
                            raise TrainingError(
                                f"Scorer returned None for additional metric '{mname}'"
                            )
                        self.additional_scores[mname] = _sign_corrected_value(
                            mname, m_raw
                        )
                    except Exception as err:  # noqa: BLE001
                        self.logger.warning(
                            "Additional metric '%s' could not be computed: %s. "
                            "Training result and artifact are unaffected.",
                            mname,
                            err,
                        )
            except Exception as e:  # noqa: BLE001
                raise TrainingError(f"Error calculating metrics on validation: {e}")

            return self

    def _align_predict_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Выровнять колонки DataFrame к обучающему порядку (issue #2).

        Защитный слой для ``predict``: валидирует наличие всех
        ``self.feature_names``, отбрасывает лишние колонки и возвращает
        DataFrame с колонками в обучающем порядке. Это чинит и устаревшие
        сериализованные модели, у которых препроцессор использует позиционные
        индексы: после выравнивания позиционный срез снова указывает на те же
        колонки, что и при обучении.

        Args:
            X: Входной DataFrame с предсказываемыми признаками.

        Returns:
            DataFrame с колонками в обучающем порядке (``self.feature_names``).

        Raises:
            TrainingError: Если часть ``self.feature_names`` отсутствует в ``X``
                либо встречается в ``X`` более одного раза.
        """
        feature_names = self.feature_names
        if not feature_names:
            return X
        missing = [col for col in feature_names if col not in X.columns]
        if missing:
            raise TrainingError(
                f"Missing columns in prediction data: {missing}. "
                f"Expected columns: {feature_names}."
            )
        duplicated = [col for col in feature_names if list(X.columns).count(col) > 1]
        if duplicated:
            raise TrainingError(
                f"Duplicated columns in prediction data: {duplicated}. "
                "Column selection by name is ambiguous."
            )
        return X[feature_names]

    def _align_predict_shared_dataframe(self, X: SharedDataFrame) -> Any:
        """Подготовить SharedDataFrame к предсказанию (issue #2).

        Если контейнер хранит имена колонок, данные восстанавливаются в
        DataFrame и выравниваются к обучающему порядку (те же гарантии, что
        для обычного ``pd.DataFrame``); иначе используется позиционный
        numpy-fallback по обучающему порядку.

        Args:
            X: Контейнер данных в разделяемой памяти.

        Returns:
            ``pd.DataFrame`` с колонками в обучающем порядке либо
            ``np.ndarray`` из разделяемой памяти, если имена колонок
            недоступны.

        Raises:
            TrainingError: Если часть ``self.feature_names`` отсутствует в
                колонках контейнера.
        """
        if X.columns is None or not self.feature_names:
            return X.shared_array
        missing = [col for col in self.feature_names if col not in X.columns]
        if missing:
            raise TrainingError(
                f"Missing columns in prediction data: {missing}. "
                f"Expected columns: {self.feature_names}."
            )
        return X.get_view(self.feature_names)

    def predict(self, X: Any) -> np.ndarray:
        """Получить предсказания модели для новых данных,
        используя обученный пайплайн.

        Для ``pd.DataFrame`` выполняется защитная валидация и выравнивание
        порядка колонок к обучающему (``self.feature_names``): недостающие
        колонки дают явную ошибку, лишние отбрасываются, дублирующиеся имена
        отклоняются (issue #2). ``SharedDataFrame`` восстанавливается в
        DataFrame и выравнивается так же, когда доступны имена колонок.
        Для ``np.ndarray`` и прочих входов без имён используется позиционный
        порядок колонок, зафиксированный при обучении.
        """

        with self.lock:
            # 1) Проверка: обучена ли модель
            if self.pipeline is None:
                raise TrainingError(
                    "The predict method called for an untrained model."
                    " Perform fit() first."
                )
            try:
                # Выравниваем колонки для входов с именами; для остальных
                # избегаем конвертации, если X уже массив или DataFrame
                X_input: Any
                if isinstance(X, pd.DataFrame):
                    X_input = self._align_predict_columns(X)
                elif isinstance(X, SharedDataFrame):
                    X_input = self._align_predict_shared_dataframe(X)
                elif isinstance(X, (pd.Series, np.ndarray)):
                    X_input = X
                else:
                    X_input = np.asarray(X)

                preds = self.pipeline.predict(X_input)
                return np.asarray(preds)

            except Exception as e:  # noqa: BLE001
                raise TrainingError(f"Error during prediction: {e}")

    def save(self, path: str | Path) -> None:
        """Сериализовать и сохранить текущий экземпляр тренера на диск."""
        with self.lock:
            if self.pipeline is None and self.base_model is None:
                raise TrainingError("Nothing to save: model is not trained")

            path_obj = Path(path)
            # Создаем директории, если они не существуют
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Вызываем новую утилиту вместо pickle.dump
            save_artifact(obj=self, path=path_obj, fmt=self.serialization_format)

    @classmethod
    def load(
        cls, path: str | Path, fmt: SerializationFormat = SerializationFormat.pickle
    ) -> ModelTrainer:
        """Загрузить ранее сохраненный объект ModelTrainer из файла."""
        path_obj = Path(path)

        # Попытка загрузки через утилиту
        try:
            obj = load_artifact(path=path_obj, fmt=fmt)
        except FileNotFoundError:
            raise TrainingError(f"File not found: {path}")
        except Exception as e:  # noqa: BLE001
            raise TrainingError(f"Error loading artifact: {e}")
        # Проверка типа загруженного объекта
        if not isinstance(obj, cls):
            raise TrainingError(f"Loaded object is not a ModelTrainer: {path}")

        return obj


def train_model(
    cfg_or_algo: dict[str, Any] | str,
    metric_or_testsize: str | float,
    params_or_metric: dict[str, Any] | str,
    X: Any = None,
    y: Any = None,
    enable_logging: bool = False,
    random_state: int | None = 42,
    log_path: str | Path | None = None,
    *,
    feature_selection_cfg: FeatureSelectionCfg | dict[str, Any] | None = None,
    feature_selection_active: bool | None = None,
) -> float:
    """Обеспечить совместимость со старым API для обучения моделей.
    Функция-фасад, которая принимает конфигурацию или набор позиционных аргументов,
    инициирует ModelTrainer и возвращает результат валидации.

    Параметры отбора признаков (``feature_selection_cfg`` /
    ``feature_selection_active``) — keyword-only: в ветке «config dict»
    одноимённые ключи словаря имеют приоритет над явными аргументами
    (аналогично ``data_oversampling``). Ключ, присутствующий в конфиге со
    значением ``None``, трактуется как «значение не задано»: в этом случае,
    как и при отсутствии ключа, используется явный аргумент функции
    (фолбэк). В ветке простого API используются непосредственно аргументы
    функции.

    Attributes:
        cfg_or_algo (dict | str): Словарь конфигурации или название алгоритма.
        metric_or_testsize (str | float): Метрика или размер тестовой выборки.
        params_or_metric (dict | str): Гиперпараметры или название метрики.
        X (Any): Матрица признаков.
        y (Any): Вектор целевой переменной.
        enable_logging (bool): Флаг активации ведения журналов.
        random_state (int | None): Зерно случайности.
        log_path (str | Path | None): Путь к файлу логов.
        feature_selection_cfg (FeatureSelectionCfg | dict | None):
            Конфигурация отбора признаков (метод, процентиль, min_features
            и т.д.). В ветке «config dict» значение ключа
            ``feature_selection_cfg`` имеет приоритет над этим аргументом,
            кроме случая, когда ключ присутствует со значением ``None`` —
            тогда используется этот аргумент (фолбэк).
            ``None`` — ``FeatureSelectionCfg()`` (mode='disabled').
        feature_selection_active (bool | None): Явный булев флаг активности
            отбора признаков (приоритет над режимом из
            ``feature_selection_cfg``). В ветке «config dict» значение ключа
            ``feature_selection_active`` имеет приоритет над этим аргументом,
            кроме случая, когда ключ присутствует со значением ``None`` —
            тогда используется этот аргумент (фолбэк).
            Небулевые значения отклоняются конструктором ``ModelTrainer``
            (строгая типизация, ревью PR #8). ``None`` — решение по
            конфигурации.
    """
    algo: str = ""
    metric: str = ""
    hyperparams: dict[str, Any] = {}
    rs: int | None = random_state

    fs_cfg: FeatureSelectionCfg | dict[str, Any] | None
    fs_active: bool | None

    # Случай «config dict»
    if isinstance(cfg_or_algo, dict):
        cfg: dict[str, Any] = cfg_or_algo
        algo = str(cfg.get("algorithm", ""))
        metric = str(cfg.get("metric", ""))
        hyperparams = cast(dict[str, Any], cfg.get("hyperparams", {}))
        rs = cast(int | None, cfg.get("random_state", 42))
        enable_logging = bool(cfg.get("enable_logging", False))
        data_os = bool(cfg.get("data_oversampling", False))
        data_os_mult = float(cfg.get("data_oversampling_multiplier", 1.0))
        data_os_alg = str(cfg.get("data_oversampling_algorithm", "random"))
        # Ключи конфига имеют приоритет над явными аргументами (аналогично
        # ``data_oversampling``); ключ со значением None трактуется как
        # «значение не задано» и даёт фолбэк на аргумент функции (см. docstring).
        if "feature_selection_cfg" in cfg and cfg["feature_selection_cfg"] is not None:
            fs_cfg = cast(
                FeatureSelectionCfg | dict[str, Any] | None,
                cfg["feature_selection_cfg"],
            )
        else:
            fs_cfg = feature_selection_cfg
        if (
            "feature_selection_active" in cfg
            and cfg["feature_selection_active"] is not None
        ):
            fs_active = cast(bool | None, cfg["feature_selection_active"])
        else:
            fs_active = feature_selection_active
    else:
        # Простой API
        # Если пришел None, мы НЕ превращаем его в строку "None" сразу,
        # чтобы сохранить логику оригинальной валидации для тестов.
        algo = cfg_or_algo if cfg_or_algo is not None else ""
        metric = str(metric_or_testsize)
        hyperparams = cast(dict[str, Any], params_or_metric)
        rs = random_state
        data_os = False
        data_os_mult = 1.0
        data_os_alg = "random"
        fs_cfg = feature_selection_cfg
        fs_active = feature_selection_active

    # Проверка алгоритма
    if not isinstance(cfg_or_algo, (str, dict)):
        raise TrainingError("Invalid algorithm")
    algo_key = algo.lower()

    if not hyperparams and algo_key not in ["isotonic", "isotonic_regression"]:
        raise TrainingError("Model parameters are not specified")

    # Создаём и обучаем ModelTrainer
    trainer = ModelTrainer(
        algorithm=algo,
        hyperparams=hyperparams,
        metric=metric,
        random_state=rs,
        data_oversampling=data_os,
        data_oversampling_multiplier=data_os_mult,
        data_oversampling_algorithm=data_os_alg,
        feature_selection_cfg=fs_cfg,
        feature_selection_active=fs_active,
    )
    trainer.fit(X, y)
    val_score = trainer.val_score
    if val_score is None:
        raise TrainingError("Model did not return a metric value")

    # Логирование (если enable_logging=True)
    if enable_logging:
        # Получаем логгер и пишем сообщение
        logger = logging.getLogger(__name__)
        # Определяем тип метрики для понятного лога
        metric_type = "Score" if is_greater_better(metric) else "Error (Natural)"

        logger.info(
            f"Training finished: Algorithm={algo_key}, "
            f"Metric={metric.upper()} ({metric_type}), "
            f"Value={val_score:.4f}"
        )

    return float(val_score)
