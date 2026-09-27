"""Адаптивные пресеты предобработки признаков.

Модуль реализует автоматический выбор стратегии предобработки признаков
в зависимости от выбранного регрессионного алгоритма (issue #18).

Ключевые понятия:

    * **Пресет предобработки** (:class:`PreprocessingPreset`) — декларативно
      описанный набор правил обработки числовых признаков: стратегия
      заполнения пропусков (:data:`ImputationStrategy`) и необходимость
      масштабирования с его типом (:data:`ScalingType`).
    * **Класс алгоритмов** (:class:`AlgorithmClass`) — группа регрессионных
      алгоритмов с общими требованиями к предобработке.
    * **Таблица соответствия** :data:`ALGORITHM_CLASS_MAPPING` — фиксирует
      принадлежность каждого поддерживаемого алгоритма к классу. Добавление
      нового алгоритма не требует правки ядра логики предобработки —
      достаточно зарегистрировать его в таблице (FR-6).

Правила выбора (FR-1):

    1. Имя алгоритма нормализуется (lowercase + раскрытие алиасов) и
       проверяется по реестру поддерживаемых алгоритмов: неизвестное имя
       отклоняется :class:`ValueError` (защита от опечаток в конфигурации).
    2. Класс алгоритма ищется в :data:`ALGORITHM_CLASS_MAPPING`; для
       поддерживаемого алгоритма без явной регистрации используется класс
       ``default`` (граничный случай — поведение по умолчанию).
    3. Пользовательское переопределение (:class:`PreprocessingOverride`)
       заменяет поля пресета полностью или частично (FR-5): явное указание
       пользователя имеет приоритет над автоматическим выбором.

Единая функция :func:`resolve_preprocessing_preset` используется на всех
этапах жизненного цикла — фазе HPO, финальном обучении и предсказании, —
что гарантирует единообразие применения пресета (FR-4).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from configurable_automl_engine.models import (
    AVAILABLE_ALGORITHMS,
    resolve_algorithm_name,
)

#: Стратегии заполнения пропусков для числовых признаков.
ImputationStrategy = Literal["mean", "median"]

#: Типы масштабирования числовых признаков.
ScalingType = Literal["standard", "robust", "none"]


class AlgorithmClass(str, Enum):
    """Классы регрессионных алгоритмов с общими требованиями к предобработке.

    Attributes:
        SCALE_SENSITIVE: Модели, чувствительные к масштабу (линейные, ядерные,
            основанные на расстояниях) — масштабирование обязательно,
            заполнение пропусков стандартной стратегией.
        TREES: Деревья и ансамбли деревьев — масштабирование не требуется и
            не должно применяться; заполнение пропусков устойчиво к выбросам.
        GLM_SKEWED: Обобщённые линейные модели со скошенными распределениями —
            заполнение пропусков, учитывающее скошенность данных; особые
            требования к масштабированию.
        UNIVARIATE: Специализированные одномерные алгоритмы — нестандартная
            предобработка (без масштабирования).
        DEFAULT: Поведение по умолчанию для алгоритмов без явной регистрации.
    """

    SCALE_SENSITIVE = "scale_sensitive"
    TREES = "trees"
    GLM_SKEWED = "glm_skewed"
    UNIVARIATE = "univariate"
    DEFAULT = "default"


class PreprocessingPreset(BaseModel):
    """Декларативно описанный пресет предобработки признаков (FR-3).

    Attributes:
        algorithm_class: Класс алгоритмов, для которого определён пресет.
        imputation_strategy: Стратегия заполнения пропусков для числовых
            признаков: ``'mean'`` (стандартная) или ``'median'``
            (устойчива к выбросам, учитывает скошенность).
        scaling: Необходимость масштабирования и его тип: ``'standard'``
            (StandardScaler), ``'robust'`` (RobustScaler, устойчив к выбросам)
            или ``'none'`` (масштабирование не применяется).
        description: Человекочитаемое описание пресета.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm_class: AlgorithmClass = Field(
        default=AlgorithmClass.DEFAULT,
        description="Класс алгоритмов, для которого определён пресет.",
    )
    imputation_strategy: ImputationStrategy = Field(
        default="mean",
        description=(
            "Стратегия заполнения пропусков для числовых признаков: "
            "'mean' или 'median'."
        ),
    )
    scaling: ScalingType = Field(
        default="standard",
        description=(
            "Масштабирование числовых признаков: 'standard', 'robust' или 'none'."
        ),
    )
    description: str = Field(
        default="",
        description="Человекочитаемое описание пресета.",
    )

    def __str__(self) -> str:
        """Компактная строка для логирования выбранного пресета (AC-9)."""
        return (
            f"PreprocessingPreset(class={self.algorithm_class.value!r}, "
            f"imputation_strategy={self.imputation_strategy!r}, "
            f"scaling={self.scaling!r})"
        )

    __repr__ = __str__


class PreprocessingOverride(BaseModel):
    """Явное переопределение пресета предобработки пользователем (FR-5).

    Задаётся частично или полностью: ``None`` в поле означает «оставить
    значение, выбранное автоматически». Некорректные значения отклоняются
    на этапе валидации конфигурации (Pydantic) с понятным сообщением.
    """

    model_config = ConfigDict(extra="forbid")

    imputation_strategy: ImputationStrategy | None = Field(
        default=None,
        description=(
            "Переопределение стратегии заполнения пропусков "
            "('mean' или 'median'). None — автовыбор."
        ),
    )
    scaling: ScalingType | None = Field(
        default=None,
        description=(
            "Переопределение масштабирования ('standard', 'robust' или 'none'). "
            "None — автовыбор."
        ),
    )


#: Таблица соответствия «алгоритм → класс алгоритмов» (FR-2, FR-6).
#: Ключи — канонические имена алгоритмов из ``AVAILABLE_ALGORITHMS``.
ALGORITHM_CLASS_MAPPING: dict[str, AlgorithmClass] = {
    # ── Масштабо-чувствительные модели (линейные, ядерные, по расстояниям) ──
    "elasticnet": AlgorithmClass.SCALE_SENSITIVE,
    "sgdregressor": AlgorithmClass.SCALE_SENSITIVE,
    "ridge": AlgorithmClass.SCALE_SENSITIVE,
    "lasso": AlgorithmClass.SCALE_SENSITIVE,
    "ardregression": AlgorithmClass.SCALE_SENSITIVE,
    "svr": AlgorithmClass.SCALE_SENSITIVE,
    "nearest_neighbors_regression": AlgorithmClass.SCALE_SENSITIVE,
    "gaussian_process_regression": AlgorithmClass.SCALE_SENSITIVE,
    # ── Деревья и ансамбли деревьев ──
    "decision_tree": AlgorithmClass.TREES,
    "random_forest": AlgorithmClass.TREES,
    "extra_trees": AlgorithmClass.TREES,
    "gradient_boosting": AlgorithmClass.TREES,
    "adaboost": AlgorithmClass.TREES,
    "xgboosting": AlgorithmClass.TREES,
    # ── GLM со скошенными распределениями ──
    "poissonregressor": AlgorithmClass.GLM_SKEWED,
    "gammaregressor": AlgorithmClass.GLM_SKEWED,
    "tweedieregressor": AlgorithmClass.GLM_SKEWED,
    "glm": AlgorithmClass.GLM_SKEWED,
    # ── Специализированные одномерные алгоритмы ──
    "isotonic_regression": AlgorithmClass.UNIVARIATE,
}

#: Пресет предобработки для каждого класса алгоритмов.
PRESET_BY_CLASS: dict[AlgorithmClass, PreprocessingPreset] = {
    AlgorithmClass.SCALE_SENSITIVE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.SCALE_SENSITIVE,
        imputation_strategy="mean",
        scaling="standard",
        description=(
            "Модели, чувствительные к масштабу: масштабирование обязательно "
            "(StandardScaler), заполнение пропусков стандартной стратегией (mean)."
        ),
    ),
    AlgorithmClass.TREES: PreprocessingPreset(
        algorithm_class=AlgorithmClass.TREES,
        imputation_strategy="median",
        scaling="none",
        description=(
            "Деревья и ансамбли деревьев: масштабирование не требуется и не "
            "применяется; заполнение пропусков медианой (устойчиво к выбросам)."
        ),
    ),
    AlgorithmClass.GLM_SKEWED: PreprocessingPreset(
        algorithm_class=AlgorithmClass.GLM_SKEWED,
        imputation_strategy="median",
        scaling="robust",
        description=(
            "GLM со скошенными распределениями: заполнение пропусков медианой "
            "(устойчиво к выбросам, учитывает скошенность); масштабирование "
            "RobustScaler (устойчиво к выбросам)."
        ),
    ),
    AlgorithmClass.UNIVARIATE: PreprocessingPreset(
        algorithm_class=AlgorithmClass.UNIVARIATE,
        imputation_strategy="median",
        scaling="none",
        description=(
            "Специализированные одномерные алгоритмы: масштабирование не "
            "применяется, заполнение пропусков медианой."
        ),
    ),
    AlgorithmClass.DEFAULT: PreprocessingPreset(
        algorithm_class=AlgorithmClass.DEFAULT,
        imputation_strategy="mean",
        scaling="standard",
        description=(
            "Поведение по умолчанию для алгоритмов без явной регистрации "
            "в таблице соответствия: mean + StandardScaler."
        ),
    ),
}


def normalize_algorithm(algorithm: str) -> str:
    """Нормализовать имя алгоритма: lowercase + раскрытие алиасов.

    Args:
        algorithm: Имя алгоритма или алиас (например, ``'RF'``, ``'xgb'``).

    Returns:
        Каноническое имя алгоритма для поиска в таблице соответствия.
    """
    return resolve_algorithm_name(algorithm)


def _coerce_override(
    override: PreprocessingOverride | dict[str, Any],
) -> PreprocessingOverride:
    """Привести пользовательское переопределение к модели :class:`PreprocessingOverride`.

    Raises:
        TypeError: Если передан объект неподдерживаемого типа.
    """
    if isinstance(override, PreprocessingOverride):
        return override
    if isinstance(override, dict):
        return PreprocessingOverride.model_validate(override)
    raise TypeError(
        f"preprocessing_override must be a dict or PreprocessingOverride, "
        f"got {type(override).__name__}"
    )


def resolve_preprocessing_preset(
    algorithm: str,
    override: PreprocessingOverride | dict[str, Any] | None = None,
) -> PreprocessingPreset:
    """Определить пресет предобработки для заданного алгоритма (FR-1, FR-5).

    Логика выбора:

    1. Имя алгоритма нормализуется (lowercase + алиасы) и проверяется
       по реестру поддерживаемых алгоритмов: неизвестное имя (опечатка)
       отклоняется явной ошибкой, чтобы не скрывать ошибки конфигурации.
    2. Класс алгоритма ищется в :data:`ALGORITHM_CLASS_MAPPING`; для
       поддерживаемого алгоритма без явной регистрации используется класс
       ``default`` (граничный случай — поведение по умолчанию).
    3. Если передан ``override``, его непустые поля заменяют соответствующие
       поля выбранного пресета (частичное или полное переопределение).

    Args:
        algorithm: Имя регрессионного алгоритма (или алиас).
        override: Явное переопределение пресета (FR-5). ``None`` —
            только автоматический выбор.

    Returns:
        Итоговый пресет предобработки для алгоритма.

    Raises:
        ValueError: Если ``algorithm`` не является поддерживаемым алгоритмом.
        TypeError: Если ``override`` имеет неподдерживаемый тип.
    """
    algo_key = normalize_algorithm(algorithm)
    if algo_key not in AVAILABLE_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            f"Available algorithms: {sorted(AVAILABLE_ALGORITHMS)}"
        )
    algo_class = ALGORITHM_CLASS_MAPPING.get(algo_key, AlgorithmClass.DEFAULT)
    preset = PRESET_BY_CLASS[algo_class]

    if override is not None:
        override_model = _coerce_override(override)
        update: dict[str, Any] = {}
        if override_model.imputation_strategy is not None:
            update["imputation_strategy"] = override_model.imputation_strategy
        if override_model.scaling is not None:
            update["scaling"] = override_model.scaling
        if update:
            preset = preset.model_copy(update=update)
    return preset
