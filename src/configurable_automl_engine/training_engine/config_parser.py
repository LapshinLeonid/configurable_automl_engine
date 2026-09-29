"""
AutoML Config Engine: Валидация и парсинг иерархических YAML-конфигураций.
Модуль реализует строго типизированную объектную модель конфигурации
на базе Pydantic v2, обеспечивая проверку целостности параметров
эксперимента, стратегий валидации и пространств поиска гиперпараметров
перед запуском пайплайна AutoML.
Ключевые возможности:
    1. Multi-Stage HPO Pipeline: Конфигурирование последовательных фаз
       оптимизации (Hyperparameter Optimization) с поддержкой различных
       действий: от глобального поиска до уточнения параметров победителя.
    2. Intelligent Validation Logic: Автоматическая проверка согласованности
       настроек, включая контроль количества фолдов для k-fold стратегии
       и валидацию границ числовых диапазонов.
    3. Flexible Search Space DSL: Поддержка компактного YAML-синтаксиса
       для определения пространств поиска (Categorical, Float, Int, Log)
       с автоматическим приведением типов из списочных структур.
    4. Dependency Guard: Встроенный механизм проверки наличия необходимых
       сторонних пакетов (XGBoost и др.) для всех включенных
       в конфиг алгоритмов еще на этапе инициализации.
    5. Data Balancing Schema: Управление стратегиями оверсэмплинга
       (SMOTE, ADASYN) с поддержкой псевдонимов полей (aliasing)
       для чистоты структуры YAML-файла.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)

from configurable_automl_engine.common.definitions import (
    ALGO_PACKAGE_MAPPING,
    ParallelStrategy,
    SerializationFormat,
    ValidationStrategy,
)
from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.common.hyperopt_defaults import (
    ALGO_HYPERPARAMETER_REGISTRY,
    SearchSpaceEntry,
)
from configurable_automl_engine.models import AVAILABLE_ALGORITHMS
from configurable_automl_engine.preprocessing import EncodingStrategy
from configurable_automl_engine.preprocessing_presets import PreprocessingOverride
from configurable_automl_engine.training_engine.metrics import (
    AVAILABLE_METRICS,
    to_sklearn_name,
)

# Создаем тип на лету. *AVAILABLE_METRICS распакует список в аргументы Literal
ComparisonMetric = Literal[*AVAILABLE_METRICS]  # type: ignore


_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")

__all__ = [
    "AlgoCfg",
    "Config",
    "ValidationStrategy",
    "read_config",
]


# ─────────────────── pruning (early stopping) ──────────────────── #
class PruningStrategy(str, Enum):
    """Поддерживаемые стратегии ранней остановки (pruning) триалов Optuna."""

    median = "median"
    hyperband = "hyperband"


class PruningCfg(BaseModel):
    """Настройки ранней остановки (pruning) триалов в фазе HPO.

    Механизм промежуточной оценки: при включении тюнер публикует оценку
    качества после каждого естественного шага валидации (например, после
    каждого фолда кросс-валидации), на основе которой прайнер Optuna может
    отсечь заведомо безнадёжный триал до завершения полной оценки.
    Для стратегий валидации без естественных шагов (hold-out /
    train_test_split) прайнер не применяется — поведение безопасно и
    задокументировано (см. API_REFERENCE.md).

    Attributes:
        enable (bool): Глобальный флаг включения ранней остановки.
            По умолчанию False — фича выключена, триалы выполняются полностью.
        strategy (PruningStrategy): Стратегия отсечения: 'median'
            (optuna.pruners.MedianPruner) или 'hyperband'
            (optuna.pruners.HyperbandPruner).
        min_steps (int): Минимальное число промежуточных шагов (например,
            фолдов CV) до первого решения об отсечении. Для 'median' это
            n_warmup_steps, для 'hyperband' — min_resource.
        n_startup_trials (int): Число завершённых триалов, после которого
            включается отсечение (только для 'median'; аналог
            n_startup_trials в Optuna).
        reduction_factor (int): Коэффициент сокращения бюджета между раундами
            Hyperband (только для 'hyperband'; контролирует агрессивность
            отсечения).
    """

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=False,
        description=(
            "Флаг включения ранней остановки (pruning) триалов Optuna. "
            "По умолчанию False — каждый триал выполняется полностью."
        ),
    )
    strategy: PruningStrategy = Field(
        default=PruningStrategy.median,
        description=(
            "Стратегия отсечения триалов: 'median' (MedianPruner) "
            "или 'hyperband' (HyperbandPruner)"
        ),
    )
    min_steps: int = Field(
        default=1,
        ge=1,
        description=(
            "Минимальное число промежуточных шагов (например, фолдов "
            "кросс-валидации) до первого решения об отсечении. Для 'median' "
            "соответствует n_warmup_steps, для 'hyperband' — min_resource"
        ),
    )
    n_startup_trials: int = Field(
        default=5,
        ge=1,
        description=(
            "Число завершённых триалов, после которого включается отсечение "
            "(только для 'median'; аналог n_startup_trials в Optuna)"
        ),
    )
    reduction_factor: int = Field(
        default=3,
        ge=2,
        description=(
            "Коэффициент сокращения бюджета между раундами Hyperband "
            "(только для 'hyperband'; агрессивность отсечения)"
        ),
    )


# ─────────────────── phases ──────────────────── #
class HPOPhaseCfg(BaseModel):
    """Конфигурация отдельной фазы поиска гиперпараметров (Hyperparameter Optimization).

    Класс описывает параметры конкретного этапа оптимизации, позволяя
    выстраивать многоступенчатые стратегии поиска (например, сначала
    грубый поиск по всем моделям, затем тонкая настройка лучшей).
    Attributes:
        name (str): Уникальный идентификатор фазы.
        n_trials (int): Лимит итераций (испытаний) для данной фазы.
        action (str): Тип действия ('all_algorithms' или 'refine_winner'),
            определяющий область поиска.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="phase",
        description=(
            "Определенное пользователем название фазы оптимизации"
            " (например, 'coarse_search' или 'fine_tuning')"
        ),
    )
    n_trials: int = Field(
        ge=1, description="Количество итераций (испытаний) в рамках данной фазы"
    )
    action: Literal["all_algorithms", "refine_winner"] = Field(
        default="all_algorithms",
        description=(
            "Действие фазы: поиск по всем алгоритмам или уточнение "
            "гиперпараметров для победителя предыдущих этапов"
        ),
    )


# ─────────────────── general ─────────────────── #
class GeneralCfg(BaseModel):
    """Общие настройки процесса AutoML, валидации и параллелизма.

    Центральный узел управления экспериментом, отвечающий за выбор
    метрик, стратегий оценки качества (Cross-Validation / Hold-out)
    и распределение вычислительных ресурсов.
    Attributes:
        comparison_metric (ComparisonMetric): Основная метрика для ранжирования моделей.
        additional_metrics (List[ComparisonMetric]): Дополнительные информационные
            метрики, рассчитываемые для финальной модели.
        path_to_model (Path): Путь в файловой системе для экспорта артефакта модели.
        serialization_format (SerializationFormat): Формат сохранения (pickle/joblib).
        log_to_file (Path | None): Файл для записи логов работы движка.
        phases (List[HPOPhaseCfg]): Последовательность этапов оптимизации.
        validation_strategy (ValidationStrategy): Метод оценки (k-fold или hold-out).
        n_folds (int): Количество разбиений для кросс-валидации.
        categorical_encoding (Literal['one_hot', 'ordinal']): Стратегия кодирования
            категориальных признаков. По умолчанию 'one_hot'.
        parallel_strategy (str): Уровень распараллеливания (по алгоритмам/фолдам).
        max_workers (int | None): Лимит потоков или процессов.
        parallel_mode (str): Технический режим исполнения ('threads' или 'processes').
    """

    model_config = ConfigDict(extra="forbid")

    comparison_metric: ComparisonMetric = Field(
        default="r2",
        description=(
            "Метрика для сравнения моделей"
            " и выбора лучшей (например, r2, rmse, accuracy)"
        ),
    )
    additional_metrics: list[ComparisonMetric] = Field(
        default_factory=list,
        description=(
            "Список дополнительных метрик качества, значения которых рассчитываются "
            "для финальной обученной модели и возвращаются вместе с результатами "
            "обучения (ключ 'additional_metrics'). Метрики носят исключительно "
            "информационный характер: они не участвуют в оптимизации гиперпараметров, "
            "сравнении моделей и выборе победителя. Допустимые значения совпадают "
            "с набором значений comparison_metric. Дубликаты в списке игнорируются "
            "(значение считается один раз), а совпадение с основной метрикой сравнения "
            "(включая алиасы, например rmse ↔ neg_root_mean_squared_error) исключается "
            "из списка — её значение и так возвращается как 'score'. Оба правила "
            "применяются на этапе валидации конфигурации, поэтому итоговый список "
            "в general.additional_metrics может отличаться от исходного, переданного "
            "пользователем."
        ),
    )
    path_to_model: Path = Field(
        default=Path("model.pkl"),
        description="Путь для сохранения/загрузки результирующей обученной модели",
    )
    serialization_format: SerializationFormat = Field(
        default=SerializationFormat.pickle,
        description="Формат сериализации модели (pickle, joblib и т.д.)",
    )
    log_to_file: Path | None = Field(
        default=None,
        description=(
            "Путь к файлу логов. Если не указан,логи выводятся только в консоль"
        ),
    )
    phases: list[HPOPhaseCfg] = Field(
        ..., description="Список последовательных фаз оптимизации гиперпараметров"
    )
    validation_strategy: ValidationStrategy = Field(
        default=ValidationStrategy.auto,
        description=(
            "Стратегия оценки качества:"
            "k-fold кросс-валидация, фиксированный hold-out,"
            "LOO или автоматический выбор"
        ),
    )
    n_folds: int = Field(
        default=5,
        description=(
            "Количество блоков (фолдов) для кросс-валидации."
            "Используется только если validation_strategy = 'k_fold'"
        ),
    )
    categorical_encoding: EncodingStrategy = Field(
        default="one_hot",
        description=(
            "Стратегия кодирования категориальных признаков: 'one_hot' (каждая "
            "категория -> отдельный бинарный столбец), 'ordinal' (каждая "
            "категориальная колонка -> один числовой столбец с наложенным "
            "порядком), 'target' (target encoding со сглаживанием), 'frequency' "
            "(частота категории) или 'hashing' (детерминированное хеширование "
            "в фиксированное число бинарных колонок). Ordinal кодирование не "
            "расширяет число колонок и полезно для линейных моделей, однако "
            "навязывает искусственный порядок категориям. Target/frequency/"
            "hashing эффективны для колонок высокой кардинальности."
        ),
    )
    high_cardinality_threshold: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Порог кардинальности для автоматического режима (>= 0). Колонки "
            "с числом уникальных значений строго больше порога кодируются "
            "стратегией high_cardinality_encoding, остальные — стратегией "
            "categorical_encoding. Должен задаваться вместе с "
            "high_cardinality_encoding. None — автоматический режим отключён."
        ),
    )
    high_cardinality_encoding: EncodingStrategy | None = Field(
        default=None,
        description=(
            "Стратегия кодирования для колонок высокой кардинальности "
            "(выше high_cardinality_threshold). Должна задаваться вместе с "
            "high_cardinality_threshold. None — режим отключён."
        ),
    )
    hashing_n_components: int = Field(
        default=16,
        ge=1,
        description=(
            "Число бинарных колонок на одну категориальную колонку при "
            "кодировании 'hashing' (>= 1). Управляет размерностью выходных "
            "признаков: рост числа признаков не зависит от кардинальности колонки."
        ),
    )
    target_encoding_smoothing: float = Field(
        default=20.0,
        ge=0,
        description=(
            "Параметр сглаживания (m) target encoding (>= 0). При m=0 "
            "используется чистое среднее целевой переменной по категории; "
            "при больших m редкие категории приближаются к глобальному среднему."
        ),
    )
    target_encoding_fallback: float | None = Field(
        default=None,
        description=(
            "Fallback-значение target encoding для категорий, отсутствующих "
            "в обучающей выборке. None — глобальное среднее целевой переменной."
        ),
    )
    parallel_strategy: ParallelStrategy = Field(
        default=next(iter(ParallelStrategy)),
        description=(
            "Стратегия распараллеливания."
            "Сейчас поддерживается только 'algorithms'"
            " (каждый алгоритм в своем потоке/процессе)."
        ),
    )
    max_workers: int | None = Field(
        default=None,
        description=(
            "Максимальное количество потоков/процессов."
            "Если null, используется количество ядер CPU"
        ),
    )
    parallel_mode: Literal["threads", "processes"] = Field(
        default="threads",
        description=(
            "Режим многозадачности: потоки (для I/O задач)"
            " или процессы (для CPU-интенсивных вычислений)"
        ),
    )
    phase_timeout: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Глобальный таймаут на всю фазу HPO (в секундах). "
            "Если None — используется значение по умолчанию 3600 секунд (1 час)."
        ),
    )
    task_timeout: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Таймаут на одну задачу (алгоритм) внутри фазы (в секундах). "
            "Если None — используется phase_timeout или глобальный timeout."
        ),
    )
    pruning: PruningCfg = Field(
        default_factory=PruningCfg,
        description=(
            "Настройки ранней остановки (pruning) триалов Optuna. "
            "По умолчанию выключены: каждый триал выполняется полностью."
        ),
    )

    @field_validator("additional_metrics")
    @classmethod
    def _deduplicate_additional_metrics(
        cls, v: list[ComparisonMetric]
    ) -> list[ComparisonMetric]:
        """Устранить дубликаты в списке дополнительных метрик.

        Задокументированное поведение: одна и та же метрика, указанная
        несколько раз, вычисляется и возвращается ровно один раз
        (порядок первого вхождения сохраняется). Дедупликация выполняется
        по строковому представлению метрики в нижнем регистре: элементы
        ``ComparisonMetric`` являются строками, поэтому вызов ``.lower()``
        типобезопасен и не зависит от произвольного типа элемента.
        """
        seen: set[str] = set()
        result: list[ComparisonMetric] = []
        for name in v:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                result.append(name)
        return result

    @model_validator(mode="after")
    def _exclude_comparison_from_additional(self) -> GeneralCfg:
        """Исключить основную метрику сравнения из списка дополнительных.

        Значение основной метрики сравнения уже возвращается в результатах
        обучения как ``score``, поэтому дублировать его в
        ``additional_metrics`` не нужно. Сравнение выполняется по
        sklearn-имени метрики, поэтому алиасы одной и той же метрики
        (например, 'rmse' и 'neg_root_mean_squared_error') считаются
        совпадением.
        """
        comparison_sklearn = to_sklearn_name(self.comparison_metric)
        filtered = [
            m
            for m in self.additional_metrics
            if to_sklearn_name(m) != comparison_sklearn
        ]
        if len(filtered) != len(self.additional_metrics):
            logging.getLogger(__name__).warning(
                "additional_metrics contains the comparison metric '%s'; "
                "its value is already returned as 'score' and will not be "
                "duplicated in 'additional_metrics'.",
                self.comparison_metric,
            )
        self.additional_metrics = filtered
        return self

    @model_validator(mode="after")
    def _check_n_folds(self) -> GeneralCfg:
        """Проверить логическую целостность настроек валидации и сериализации.

        Логика проверки:
        1. Проверка доступности пакета `joblib`, если выбран соответствующий формат.
        2. Гарантия, что `n_folds` является положительным числом для любых стратегий.
        3. Строгая проверка для `k_fold`: количество блоков должно быть не менее 2.

        Returns:
            GeneralCfg: Валидированный объект настроек.

        Raises:
            ValueError: Если пакет 'joblib' не установлен или значение `n_folds`
                недопустимо для выбранной стратегии.
        """
        if self.serialization_format == SerializationFormat.joblib and not is_installed(
            "joblib"
        ):
            raise ValueError(
                "serialization_format='joblib' requires"
                " the 'joblib' package to be installed"
            )
        if self.n_folds < 1:
            raise ValueError("`n_folds` must be at least 1")
        # Строгая проверка только для k-fold
        if self.validation_strategy == ValidationStrategy.k_fold and self.n_folds < 2:
            raise ValueError("n_folds must be ≥ 2 for k-fold validation")
        return self

    @model_validator(mode="after")
    def _check_pruning_consistency(self) -> GeneralCfg:
        """Проверить согласованность настроек ранней остановки с валидацией.

        Логика проверки:
        1. Для 'k_fold' прайнер должен иметь возможность сработать раньше
           окончания оценки: min_steps не может превышать n_folds.
        2. Для 'train_test_split' естественных шагов нет — ранняя остановка
           не применяется; выдаём предупреждение, а не ошибку (поведение
           задокументировано и безопасно).

        Returns:
            GeneralCfg: Валидированный объект настроек.

        Raises:
            ValueError: Если настройки pruning конфликтуют с валидацией.
        """
        if not self.pruning.enable:
            return self

        if (
            self.validation_strategy == ValidationStrategy.k_fold
            and self.pruning.min_steps > self.n_folds
        ):
            raise ValueError(
                f"pruning.min_steps ({self.pruning.min_steps}) cannot be greater "
                f"than general.n_folds ({self.n_folds}) for k-fold validation: "
                "the pruner would never get enough intermediate steps. "
                "Decrease pruning.min_steps or increase general.n_folds."
            )

        if self.validation_strategy == ValidationStrategy.train_test_split:
            logging.getLogger(__name__).warning(
                "pruning.enable=true with validation_strategy='train_test_split': "
                "early stopping will NOT be applied because hold-out evaluation "
                "has no intermediate steps."
            )
        return self

    @model_validator(mode="after")
    def _check_high_cardinality_encoding(self) -> GeneralCfg:
        """Проверить согласованность настроек кодирования high-cardinality колонок.

        Логика проверки:
        1. ``high_cardinality_threshold`` и ``high_cardinality_encoding``
           задаются только вместе (иначе поведение неоднозначно — ошибка
           отклоняется до запуска обучения).
        2. Значения ``categorical_encoding`` / ``high_cardinality_encoding``
           валидируются типом ``EncodingStrategy`` (неподдерживаемая стратегия
           отклоняется на этапе парсинга конфигурации).

        Returns:
            GeneralCfg: Валидированный объект настроек.

        Raises:
            ValueError: Если задан только один из пары параметров
                high-cardinality кодирования.
        """
        threshold = self.high_cardinality_threshold
        hc_encoding = self.high_cardinality_encoding
        if (threshold is None) != (hc_encoding is None):
            raise ValueError(
                "general.high_cardinality_threshold and "
                "general.high_cardinality_encoding must be set together: "
                f"got threshold={threshold!r}, encoding={hc_encoding!r}. "
                "Provide both parameters to enable automatic high-cardinality "
                "encoding, or omit both to disable it."
            )
        return self


# ──────────────── oversampling ──────────────── #
class OversamplingAlgorithm(str, Enum):
    random = "random"
    random_with_noise = "random_with_noise"
    smote = "smote"
    adasyn = "adasyn"


class OversamplingCfg(BaseModel):
    """Параметры балансировки классов и синтеза данных.

    Обеспечивает конфигурацию методов устранения дисбаланса целевой переменной.
    Использует механизм псевдонимов (aliases) для маппинга плоских ключей
    YAML в структурированный объект.
    Attributes:
        enable (bool): Глобальный флаг включения оверсэмплинга.
        multiplier (float): Целевой коэффициент увеличения выборки.
        algorithm (OversamplingAlgorithm): Выбранный метод генерации
            (SMOTE, ADASYN и др.).
    """

    # принимать alias‑имена
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    enable: bool = Field(
        default=False,
        alias="data_oversampling",
        description=(
            "Флаг включения балансировки данных.Применяется только к обучающей выборке"
        ),
    )
    multiplier: float = Field(
        default=1.0,
        alias="data_oversampling_multiplier",
        ge=1.0,
        description="Во сколько раз увеличить количество примеров миноритарных классов",
    )
    algorithm: OversamplingAlgorithm = Field(
        default=OversamplingAlgorithm.random,
        alias="data_oversampling_algorithm",
        description="Алгоритм синтеза новых данных (Random, SMOTE, ADASYN)",
    )

    @model_validator(mode="after")
    def _validate_oversampling_logic(self) -> OversamplingCfg:
        if self.enable and self.multiplier == 1.0:
            logging.getLogger(__name__).warning(
                "Oversampling multiplier = 1 ➜ class balance will not change."
            )
        return self


# ───────────────── algorithms ───────────────── #


class AlgoCfg(BaseModel):
    """Техническая конфигурация конкретного ML-алгоритма в пайплайне.

    Хранит настройки включения алгоритма, переопределенные пространства
    поиска и пути к программным модулям тренера и тюнера.
    Attributes:
        enable (bool): Флаг участия алгоритма в текущем эксперименте.
        limit_hyperparameters (bool): Режим сокращенного пространства поиска.
        hyperparameters (Dict | None): Кастомные границы параметров,
            перекрывающие значения по умолчанию.
        tuner (str): Dotted-path к модулю оптимизатора.
        trainer_module (str): Dotted-path к реализации обучения модели.
    """

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=True, description="Использовать ли данный алгоритм в пайплайне AutoML"
    )
    limit_hyperparameters: bool = Field(
        default=False, description=("Ограничить гиперпараметры пространства поиска")
    )
    hyperparameters: dict[str, SearchSpaceEntry] | None = Field(
        default=None,
        description=(
            "Ключи должны соответствовать допустимым гиперпараметрам алгоритма. "
            "См. ALGO_HYPERPARAMETER_REGISTRY."
        ),
    )
    preprocessing: PreprocessingOverride | None = Field(
        default=None,
        description=(
            "Явное переопределение пресета предобработки признаков для данного "
            "алгоритма (FR-5). Задаётся частично или полностью: 'imputation_strategy' "
            "('mean'|'median') и/или 'scaling' ('standard'|'robust'|'none'). "
            "Имеет приоритет над автоматическим выбором по классу алгоритма. "
            "None — автоматический выбор."
        ),
    )
    tuner: str | None = Field(
        default="configurable_automl_engine.tuner",
        description="Путь к модулю тюнера для оптимизации гиперпараметров",
    )
    trainer_module: str | None = Field(
        default="configurable_automl_engine.trainer",
        description=(
            "Dotted-path к модулю, содержащему класс `ModelTrainer`"
            "(например, 'configurable_automl_engine.trainer')."
        ),
    )

    def get_required_package(self, algo_name: str) -> str | None:
        """Определить имя внешнего Python-пакета, необходимого для работы алгоритма.

        Логика поиска:
        1. Обращается к глобальному маппингу `ALGO_PACKAGE_MAPPING`.
        2. Сопоставляет внутреннее имя алгоритма (например, 'xgboost')
            с названием в PyPI.

        Args:
            algo_name (str): Уникальный идентификатор алгоритма.
        Returns:
            Optional[str]: Название пакета для установки через pip или None,
                если зависимость не определена.
        """
        return ALGO_PACKAGE_MAPPING.get(algo_name)

    def get_unknown_hyperparameters(self, algo_name: str) -> list[str]:
        """Вернуть список гиперпараметров, несовместимых с данным алгоритмом.

        Сверяет ключи `self.hyperparameters` с допустимым множеством из
        `ALGO_HYPERPARAMETER_REGISTRY`. Если алгоритм отсутствует в реестре —
        проверка пропускается (мягкий fallback).

        Args:
            algo_name (str): Уникальный идентификатор алгоритма.
        Returns:
            List[str]: Список недопустимых ключей. Пустой список — всё корректно.
        """
        if self.hyperparameters is None:
            return []
        allowed = ALGO_HYPERPARAMETER_REGISTRY.get(algo_name)

        if not allowed:
            return []

        return [k for k in self.hyperparameters if k not in allowed]

    @field_validator("tuner", "trainer_module")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if v is None:
            return v
        if not _DOTTED_PATH_RE.fullmatch(v):
            raise ValueError(
                f"'{v}' is not a valid dotted path "
                "(expecting 'package.module' or 'a.b.c.Class' format)"
            )
        return v


class _AlgorithmsConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


AlgorithmsConfig = create_model(
    "AlgorithmsConfig",
    __base__=_AlgorithmsConfigBase,
    **{name: (AlgoCfg | None, Field(default=None)) for name in AVAILABLE_ALGORITHMS},
)  # type: ignore[call-overload]

if TYPE_CHECKING:
    # для mypy используем статический alias
    AlgorithmsConfigType = _AlgorithmsConfigBase
else:
    # runtime — динамический create_model
    AlgorithmsConfigType = AlgorithmsConfig


# ─────────────────── root ──────────────────── #
class Config(BaseModel):
    """Корневой объект всей системы конфигурации AutoML.

    Агрегирует все секции настроек и выполняет финальную валидацию
    целостности графа параметров, включая проверку системных зависимостей.
    Attributes:
        general (GeneralCfg): Общие параметры эксперимента.
        oversampling (OversamplingCfg): Настройки предобработки данных.
        algorithms (AlgorithmsConfig): Реестр доступных и активных алгоритмов.
    """

    model_config = ConfigDict(extra="forbid")

    general: GeneralCfg = Field(
        ..., description="Общие настройки эксперимента и валидации"
    )
    oversampling: OversamplingCfg = Field(
        default=OversamplingCfg(), description="Настройки балансировки данных"
    )
    algorithms: AlgorithmsConfigType = Field(
        ...,
        description=(
            "Словарь алгоритмов, где ключ — имя алгоритма "
            "(например, 'xgboost', 'random_forest')"
        ),
    )  # type: ignore[valid-type]

    @field_validator("algorithms")
    @classmethod
    def _must_have_enabled(cls, v: Any) -> Any:
        # Поскольку 'v' теперь — это AlgorithmsConfig (объект),
        # мы получаем все его поля через .model_dump()
        enabled_algorithms = [
            algo
            for algo in v.model_dump().values()
            if algo is not None and algo.get("enable") is True
        ]

        if not enabled_algorithms:
            raise ValueError("At least one algorithm must be enabled (enable: true)")
        return v

    @model_validator(mode="after")
    def _check_algorithm_dependencies(self) -> Config:
        # model_fields через экземпляр тоже работает, но deprecated
        # (PydanticDeprecatedSince211, удаление в V3.0) — используем класс
        for name in type(self.algorithms).model_fields:
            # Get the attribute value (could be None if not provided in data)
            algo_cfg = getattr(self.algorithms, name)

            # Skip if the algorithm entry is missing (None)
            if algo_cfg is None:
                continue

            # Ensure it has an 'enable' attribute before checking it
            # If it's a Pydantic model, check attribute, otherwise get() as dict
            enabled = getattr(algo_cfg, "enable", False)

            if enabled:
                # Safely get the required package
                required_pkg = getattr(
                    algo_cfg, "get_required_package", lambda n: None
                )(name)

                if required_pkg and not is_installed(required_pkg):
                    raise ValueError(
                        f"Algorithm '{name}' is enabled, but the package "
                        f"'{required_pkg}' is not installed. "
                        f"Please run: pip install {required_pkg}"
                    )
        return self

    @model_validator(mode="after")
    def _check_hyperparameter_compatibility(self) -> Config:
        errors = []
        for name in type(self.algorithms).model_fields:
            algo_cfg = getattr(self.algorithms, name)
            if algo_cfg is None or not algo_cfg.enable:
                continue
            unknown = algo_cfg.get_unknown_hyperparameters(name)
            if unknown:
                allowed = sorted(ALGO_HYPERPARAMETER_REGISTRY.get(name, set()))
                errors.append(
                    f"Algorithm '{name}': unknown hyperparameters {unknown}. "
                    f"Allowed parameters: {allowed}"
                )
        if errors:
            raise ValueError("\n".join(errors))
        return self


# ────────────────── API ────────────────────── #
def read_config(path: str | Path) -> Config:
    """Загрузить, распарсить и валидировать конфигурационный файл эксперимента.

    Логика работы:
    1. Открывает файл по указанному пути в кодировке UTF-8.
    2. Выполняет безопасную загрузку YAML-структуры в Python-словарь.
    3. Инициирует каскадную валидацию Pydantic
        для создания типизированного объекта `Config`.
    Args:
        path (str | Path): Путь к файлу конфигурации в формате .yaml или .yml.
    Returns:
        Config: Корневой объект конфигурации, готовый к использованию в движке AutoML.

    Raises:
        FileNotFoundError: Если файл по указанному пути не найден.
        ValidationError: Если структура файла не соответствует схеме или
            нарушена логика параметров.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
