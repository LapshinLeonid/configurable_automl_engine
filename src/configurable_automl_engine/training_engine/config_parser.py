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
       сторонних пакетов (XGBoost, CatBoost и др.) для всех включенных 
       в конфиг алгоритмов еще на этапе инициализации.
    5. Data Balancing Schema: Управление стратегиями оверсэмплинга 
       (SMOTE, ADASYN) с поддержкой псевдонимов полей (aliasing) 
       для чистоты структуры YAML-файла.
"""
from __future__ import annotations
import yaml
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Literal, List
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from configurable_automl_engine.common.definitions import (ValidationStrategy, 
                                                           SerializationFormat,
                                                           ALGO_PACKAGE_MAPPING)
from configurable_automl_engine.common.dependency_utils import is_installed

from configurable_automl_engine.common.hyperopt_defaults import SearchSpaceEntry

__all__ = [
    "AlgoCfg",
    "Config",
    "ValidationStrategy",
    "read_config",
] 
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
    name: str = Field(
        ..., 
        description=("Уникальное название фазы оптимизации"
                     " (например, 'coarse_search' или 'fine_tuning')")
    )
    n_trials: int = Field(
        ge=1, 
        description="Количество итераций (испытаний) в рамках данной фазы"
    )
    action: Literal["all_algorithms", "refine_winner"] = Field(
        default="all_algorithms",
        description=("Действие фазы: поиск по всем алгоритмам или уточнение "
                     "гиперпараметров для победителя предыдущих этапов")
    )
# ─────────────────── general ─────────────────── #
class GeneralCfg(BaseModel):
    """Общие настройки процесса AutoML, валидации и параллелизма.
    
    Центральный узел управления экспериментом, отвечающий за выбор 
    метрик, стратегий оценки качества (Cross-Validation / Hold-out) 
    и распределение вычислительных ресурсов.
    Attributes:
        comparison_metric (str): Основная метрика для ранжирования моделей.
        path_to_model (Path): Путь в файловой системе для экспорта артефакта модели.
        serialization_format (SerializationFormat): Формат сохранения (pickle/joblib).
        log_to_file (Path | None): Файл для записи логов работы движка.
        phases (List[HPOPhaseCfg]): Последовательность этапов оптимизации.
        validation_strategy (ValidationStrategy): Метод оценки (k-fold или hold-out).
        n_folds (int): Количество разбиений для кросс-валидации.
        parallel_strategy (str): Уровень распараллеливания (по алгоритмам/фолдам).
        max_workers (int | None): Лимит потоков или процессов.
        parallel_mode (str): Технический режим исполнения ('threads' или 'processes').
    """
    comparison_metric: str = Field(
        default="r2",
        description=("Метрика для сравнения моделей"
                     " и выбора лучшей (например, r2, rmse, accuracy)")
    )
    path_to_model: Path = Field(
        default=Path("model.pkl"),
        description="Путь для сохранения/загрузки результирующей обученной модели"
    )
    serialization_format: SerializationFormat = Field(
        default=SerializationFormat.pickle,
        description="Формат сериализации модели (pickle, joblib и т.д.)"
    )
    log_to_file: Optional[Path] = Field(
        default=None,
        description=("Путь к файлу логов. Если не указан,"
                     "логи выводятся только в консоль")
    )
    phases: List[HPOPhaseCfg] = Field(
        ..., 
        description="Список последовательных фаз оптимизации гиперпараметров"
    )
    validation_strategy: ValidationStrategy = Field(
        default=ValidationStrategy.k_fold,
        description=("Стратегия оценки качества:"
                     "k-fold кросс-валидация или фиксированный hold-out"),
    )
    n_folds: int = Field(
        default=5,
        description=("Количество блоков (фолдов) для кросс-валидации." 
                     "Используется только если validation_strategy = 'k_fold'")
    )
    parallel_strategy: str = Field(
        default="algorithms",
        description=("Стратегия распараллеливания." 
                     "Сейчас поддерживается только 'algorithms'"
                     " (каждый алгоритм в своем потоке/процессе).")
    )
    max_workers: Optional[int] = Field(
        default=None,
        description=("Максимальное количество потоков/процессов."
                     "Если null, используется количество ядер CPU")
    )
    parallel_mode: Literal["threads", "processes"] = Field(
        default="threads",
        description=("Режим многозадачности: потоки (для I/O задач)"
                     " или процессы (для CPU-интенсивных вычислений)")
    )
    @model_validator(mode="after")
    def _check_n_folds(self) -> "GeneralCfg":
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
        if (
            self.serialization_format == SerializationFormat.joblib
            and not is_installed("joblib")
        ):
            raise ValueError(
                "serialization_format='joblib' требует установленный пакет 'joblib'"
            )
        if self.n_folds < 1:
            raise ValueError("`n_folds` must be at least 1")
        # Строгая проверка только для k-fold
        if (
            self.validation_strategy == ValidationStrategy.k_fold
            and self.n_folds < 2
        ):
            raise ValueError("`n_folds` must be ≥ 2 при k-fold валидации")
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
    model_config = ConfigDict(populate_by_name=True)  # принимать alias‑имена
    enable: bool = Field(
        default=False,
        alias="data_oversampling",
        description=("Флаг включения балансировки данных." 
                     "Применяется только к обучающей выборке"),
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
    def _validate_oversampling_logic(self) -> "OversamplingCfg":
        if self.enable and self.multiplier == 1.0:
            logging.getLogger(__name__).warning(
                "Oversampling multiplier = 1 ➜ баланс классов не изменится."
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
    enable: bool = Field(
        default=True, 
        description="Использовать ли данный алгоритм в пайплайне AutoML"
    )
    limit_hyperparameters: bool = Field(
        default=False,
        description=("Если True, ограничивает пространство поиска"
                     "только базовыми параметрами (ускоряет работу)"
        )
    )
    hyperparameters: Dict[str, SearchSpaceEntry | Any] | None = Field(
        default=None,
        description=("Переопределение пространства поиска гиперпараметров."
                     "Ключ — имя параметра"
        )
    )
    tuner: str = Field(
        default="configurable_automl_engine.tuner",
        description="Путь к модулю тюнера для оптимизации гиперпараметров"
    )
    trainer_module: str = Field(
        default="configurable_automl_engine.trainer",
        description=(
            "Dotted-path к модулю, содержащему класс `ModelTrainer`"
            "(например, 'configurable_automl_engine.trainer')."
            )
    )
    def get_required_package(self, algo_name: str) -> Optional[str]:
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
    @field_validator("tuner", "trainer_module")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("module path must be non-empty")
        return v
# ─────────────────── root ──────────────────── #
class Config(BaseModel):
    """Корневой объект всей системы конфигурации AutoML.
    
    Агрегирует все секции настроек и выполняет финальную валидацию 
    целостности графа параметров, включая проверку системных зависимостей.
    Attributes:
        general (GeneralCfg): Общие параметры эксперимента.
        oversampling (OversamplingCfg): Настройки предобработки данных.
        algorithms (Dict[str, AlgoCfg]): Реестр доступных и активных алгоритмов.
    """
    general: GeneralCfg = Field(
        ..., 
        description="Общие настройки эксперимента и валидации"
    )
    oversampling: OversamplingCfg = Field(
        default = OversamplingCfg(),
        description="Настройки балансировки данных"
    )
    algorithms: Dict[str, AlgoCfg] = Field(
        ..., 
        description=(
            "Словарь алгоритмов, где ключ — имя алгоритма "
            "(например, 'xgboost', 'random_forest')"
        )
    )
    @field_validator("algorithms")
    @classmethod
    def _must_have_enabled(cls, v: Dict[str, AlgoCfg]) -> Dict[str, AlgoCfg]:
        if not any(a.enable for a in v.values()):
            raise ValueError("no algorithms enabled in config")
        return v
    @model_validator(mode="after")
    def _check_algorithm_dependencies(self) -> "Config":
        """Проверить наличие установленных зависимостей для всех активных алгоритмов.
        
        Логика проверки:
        1. Итерируется по словарю алгоритмов, пропуская отключенные (`enable=False`).
        2. Для каждого активного алгоритма запрашивает имя требуемого пакета.
        3. Использует `is_installed` для проверки окружения.
        
        Returns:
            Config: Полностью провалидированный корневой объект конфигурации.
        Raises:
            ValueError: Если для включенного алгоритма 
            отсутствует необходимая библиотека.
        """
        for name, algo_cfg in self.algorithms.items():
            if not algo_cfg.enable:
                continue
            required_pkg = algo_cfg.get_required_package(name)
            if required_pkg and not is_installed(required_pkg):
                raise ValueError(
                    f"Алгоритм '{name}' включён, "
                    f"но пакет '{required_pkg}' не установлен. "
                    f"Пожалуйста, выполните: pip install {required_pkg}"
                )
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