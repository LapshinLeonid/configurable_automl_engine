"""
Парсинг YAML-конфига (pydantic v2).
Добавлено:
* Подробные описания (description) для всех полей для генерации документации.
* Поле `n_folds` в секции `general` — задаёт количество сплитов при стратегии k-fold.
* Валидация логики n_folds.
"""
from __future__ import annotations
import yaml
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Literal, Annotated, Union, List
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from configurable_automl_engine.common.definitions import ValidationStrategy, SerializationFormat

# ─────────────────── phases ──────────────────── #

class HPOPhaseCfg(BaseModel):
    """Конфигурация отдельной фазы поиска гиперпараметров (Hyperparameter Optimization)."""
    name: str = Field(
        ..., 
        description="Уникальное название фазы оптимизации (например, 'coarse_search' или 'fine_tuning')"
    )
    n_trials: int = Field(
        ge=1, 
        description="Количество итераций (испытаний) в рамках данной фазы"
    )
    action: Literal["all_algorithms", "refine_winner"] = Field(
        default="all_algorithms",
        description="Действие фазы: поиск по всем алгоритмам или уточнение гиперпараметров для победителя предыдущих этапов"
    )

# ─────────────────── general ─────────────────── #

class GeneralCfg(BaseModel):
    """Общие настройки процесса AutoML и валидации."""
    comparison_metric: str = Field(
        default="r2",
        description="Метрика для сравнения моделей и выбора лучшей (например, r2, rmse, accuracy)"
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
        description="Путь к файлу логов. Если не указан, логи выводятся только в консоль"
    )
    phases: List[HPOPhaseCfg] = Field(
        ..., 
        description="Список последовательных фаз оптимизации гиперпараметров"
    )
    validation_strategy: ValidationStrategy = Field(
        default=ValidationStrategy.k_fold,
        description="Стратегия оценки качества: k-fold кросс-валидация или фиксированный hold-out",
    )
    n_folds: int = Field(
        default=5,
        description="Количество блоков (фолдов) для кросс-валидации. Используется только если validation_strategy = 'k_fold'"
    )
    parallel_strategy: str = Field(
        default="algorithms",
        description="Стратегия распараллеливания. Сейчас поддерживается только 'algorithms' (каждый алгоритм в своем потоке/процессе)."
    )
    max_workers: Optional[int] = Field(
        default=None,
        description="Максимальное количество потоков/процессов. Если null, используется количество ядер CPU"
    )
    parallel_mode: Literal["threads", "processes"] = Field(
        default="threads",
        description="Режим многозадачности: потоки (для I/O задач) или процессы (для CPU-интенсивных вычислений)"
    )

    @model_validator(mode="after")
    def _check_n_folds(self):
        # Базовая проверка на здравый смысл (для всех стратегий)
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
    """
    Параметры балансировки классов (oversampling).
    Настройки маппятся на ключи YAML с префиксом data_*.
    """

    model_config = ConfigDict(populate_by_name=True)  # принимать alias‑имена

    enable: bool = Field(
        False,
        alias="data_oversampling",
        description="Флаг включения балансировки данных. Применяется только к обучающей выборке",
    )
    multiplier: float = Field(
        1.0,
        alias="data_oversampling_multiplier",
        ge=1.0,
        description="Во сколько раз увеличить количество примеров миноритарных классов",
    )
    algorithm: OversamplingAlgorithm = Field(
        OversamplingAlgorithm.random,
        alias="data_oversampling_algorithm",
        description="Алгоритм синтеза новых данных (Random, SMOTE, ADASYN)",
    )

    @field_validator("multiplier")
    @classmethod
    def _warn_useless_multiplier(cls, v, info):
        if v == 1 and info.data.get("enable"):
            logging.getLogger(__name__).warning(
                "Oversampling multiplier = 1 ➜ баланс классов не изменится."
            )
        return v

# ───────────────── hyperopt ───────────────── #

class SearchSpaceEntry(BaseModel):
    """
    Описание пространства поиска для одного гиперпараметра.
    Формат в YAML: [min, max, type] или [options, 'categorical'].
    """

    bounds: Annotated[
        List[Union[float, int, str, bool]], 
        Field(
            min_length=2, 
            max_length=3,
            description=(
                "Параметры распределения гиперпараметра. "
                "Для чисел (int, float): [min, max, type]. "
                "Для логарифмических шкал: [min, max, 'float_log']. "
                "Для категорий: [[val1, val2], 'categorical'] или [val1, val2, 'categorical']."
            )
        )
    ]

    @model_validator(mode="after")
    def _validate_structure(self) -> SearchSpaceEntry:
        dist_type = self.bounds[-1]
        valid_types = ["int", "float", "float_log", "categorical"]
        if dist_type not in valid_types:
            return self
        
        if dist_type == "categorical":
            if not isinstance(self.bounds[0], list):
                raise ValueError(
                    f"For 'categorical' type, the first element must be a list of options. "
                    f"Got {type(self.bounds[0])} instead."
                )
        else:
            if isinstance(self.bounds[0], list) or isinstance(self.bounds[1], list):
                raise ValueError(f"Numerical distribution '{dist_type}' cannot have a list as bounds.")
        return self

# ───────────────── algorithms ───────────────── #

class AlgoCfg(BaseModel):
    """Конфигурация конкретного алгоритма машинного обучения."""
    enable: bool = Field(
        default=True, 
        description="Использовать ли данный алгоритм в пайплайне AutoML"
    )
    limit_hyperparameters: bool = Field(
        default=False,
        description="Если True, ограничивает пространство поиска только базовыми параметрами (ускоряет работу)"
    )
    hyperparameters: Dict[str, Union[SearchSpaceEntry, Any]] | None = Field(
        default=None,
        description="Переопределение пространства поиска гиперпараметров. Ключ — имя параметра"
    )
    tuner: str = Field(
        default="configurable_automl_engine.tuner",
        description="Путь к модулю тюнера для оптимизации гиперпараметров"
    )
    trainer_module: str = Field(
        default="configurable_automl_engine.trainer",
        description="Dotted-path к модулю, содержащему класс `ModelTrainer` (например, 'configurable_automl_engine.trainer')."
    )

    @field_validator("tuner", "trainer_module")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("module path must be non-empty")
        return v

# ─────────────────── root ──────────────────── #

class Config(BaseModel):
    """Корневой объект конфигурации всей системы AutoML."""
    general: GeneralCfg = Field(
        ..., 
        description="Общие настройки эксперимента и валидации"
    )
    oversampling: OversamplingCfg = Field(
        default_factory=OversamplingCfg,
        description="Настройки балансировки данных"
    )
    algorithms: Dict[str, AlgoCfg] = Field(
        ..., 
        description="Словарь алгоритмов, где ключ — имя алгоритма (например, 'xgboost', 'random_forest')"
    )

    @field_validator("algorithms")
    @classmethod
    def _must_have_enabled(cls, v: Dict[str, AlgoCfg]) -> Dict[str, AlgoCfg]:
        if not any(a.enable for a in v.values()):
            raise ValueError("no algorithms enabled in config")
        return v

# ────────────────── API ────────────────────── #

def read_config(path: str | Path) -> Config:
    """Загружает и валидирует YAML-конфигурацию."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)