"""
Парсинг YAML-конфига (pydantic v2).

Добавлено:
* поле `n_folds` в секции `general` — задаёт количество сплитов
  при стратегии k-fold (минимум 2, по умолчанию 5).
* соответствующая валидация.
"""
from __future__ import annotations

import yaml
import logging

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Literal, Annotated, Union, List
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from configurable_automl_engine.common.definitions import ValidationStrategy, SerializationFormat


# ─────────────────── general ─────────────────── #
class GeneralCfg(BaseModel):
    # ── базовые параметры ── #
    comparison_metric: str = "r2"
    path_to_model: Path = Path("model.pkl")

    serialization_format: SerializationFormat = SerializationFormat.pickle

    n_rude_tries: int = 20
    n_accurate_tries: int = 200

    validation_strategy: ValidationStrategy = Field(
        default=ValidationStrategy.k_fold,
        description="Стратегия валидации для оптимизации гиперпараметров",
    )
    n_folds: int = 5  # ≥ 2, если k-fold

    parallel_strategy: str = "algorithms"  # "algorithms" | "trials"
    max_workers: Optional[int] = None

    parallel_mode: Literal["threads", "processes"] = "threads" 

    # ──────────────── валидатор ──────────────── #
    @model_validator(mode="after")
    def _check_n_folds(self):
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
    """Параметры оверсэмплинга.

    Все поля читаются из YAML‑ключей:
      - data_oversampling
      - data_oversampling_multiplier
      - data_oversampling_algorithm
    """

    model_config = ConfigDict(populate_by_name=True)  # принимать alias‑имена

    enable: bool = Field(
        False,
        alias="data_oversampling",
        description="Включить oversampling только для train‑части",
    )
    multiplier: float = Field(
        1.0,
        alias="data_oversampling_multiplier",
        ge=1.0,
        description="Коэффициент умножения каждой миноритарной выборки (≥ 1)",
    )
    algorithm: OversamplingAlgorithm = Field(
        OversamplingAlgorithm.random,
        alias="data_oversampling_algorithm",
        description="Алгоритм генерации синтетических сэмплов",
    )

    @field_validator("multiplier")
    @classmethod
    def _warn_useless_multiplier(cls, v, info):
        # `info.data` доступен только при pydantic>=2.6; иначе — скопировать self.enable
        if v == 1 and info.data.get("enable"):
            logging.getLogger(__name__).warning(
                "Oversampling multiplier = 1 ➜ баланс классов не изменится."
            )
        return v

# ───────────────── hyperopt ───────────────── #
class SearchSpaceEntry(BaseModel):
    """
    Валидация записи пространства поиска гиперпараметра.
    Пример в YAML: [50, 500, "int"] или [0.01, 0.1, "float_log"]
    """
    bounds: Annotated[
        List[Union[float, int, str, bool]], 
        Field(min_length=2, max_length=3)
    ]
    @model_validator(mode="after")
    def _validate_structure(self) -> SearchSpaceEntry:
        # Если 3 элемента, последний должен быть типом распределения
        if len(self.bounds) == 3:
            dist_type = self.bounds[2]
            valid_types = ["int", "float", "float_log", "categorical"]
            if dist_type not in valid_types:
                raise ValueError(f"Type must be one of {valid_types}, got {dist_type}")
        return self

# ───────────────── algorithms ───────────────── #
class AlgoCfg(BaseModel):
    enable: bool = True
    limit_hyperparameters: bool = False
    hyperparameters: Dict[str, Union[SearchSpaceEntry, Any]] | None = None
    hyperopt_module: str = "configurable_automl_engine.hyperopt_module"
    trainer_module: str = "configurable_automl_engine.trainer"

    @field_validator("hyperopt_module", "trainer_module")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("module path must be non-empty")
        return v


# ─────────────────── root ──────────────────── #
class Config(BaseModel):
    general: GeneralCfg
    oversampling: OversamplingCfg = Field(default_factory=OversamplingCfg)
    algorithms: Dict[str, AlgoCfg]

    @field_validator("algorithms")
    @classmethod
    def _must_have_enabled(cls, v: Dict[str, AlgoCfg]) -> Dict[str, AlgoCfg]:
        if not any(a.enable for a in v.values()):
            raise ValueError("no algorithms enabled in config")
        return v


# ────────────────── API ────────────────────── #
def read_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
