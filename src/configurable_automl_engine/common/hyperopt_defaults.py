from typing import Dict, Any, List

from typing import Any, Dict, Optional, Literal, Annotated, Union, List
from pydantic import BaseModel, Field, model_validator

# ───────────────── hyperopt ───────────────── #
class BaseDistribution(BaseModel):
    """Абстрактный базовый класс для распределений поиска."""
    type: str
class CategoricalSpace(BaseDistribution):
    """Распределение для категориальных признаков: [options, 'categorical']."""
    type: Literal["categorical"]
    options: List[Any]
class NumericSpace(BaseDistribution):
    """Базовый класс для числовых диапазонов."""
    low: float
    high: float
    @model_validator(mode="after")
    def _validate_range(self) -> "NumericSpace":
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high})")
        return self
class FloatSpace(NumericSpace):
    """Распределение для чисел с плавающей точкой: [min, max, 'float', step?]."""
    type: Literal["float", "float_log"]
    step: Optional[float] = None
    @model_validator(mode="after")
    def _validate_float_constraints(self) -> "FloatSpace":
        if self.type == "float_log" and self.step is not None:
            raise ValueError("The 'step' parameter is not supported for 'float_log'")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Step must be positive. Got {self.step}")
        return self
class IntSpace(NumericSpace):
    """Распределение для целых чисел: [min, max, 'int', step?]."""
    type: Literal["int"]
    low: int
    high: int
    step: Optional[int] = None
    @model_validator(mode="after")
    def _validate_int_constraints(self) -> "IntSpace":
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Step must be positive. Got {self.step}")
        return self


class SearchSpaceEntry(BaseModel):
    """Описание пространства поиска для отдельного гиперпараметра.
    
    Универсальный контейнер, поддерживающий как категориальные списки, 
    так и числовые диапазоны. Реализует логику прозрачной конвертации 
    из сокращенной YAML-записи в типизированные объекты распределений.
    Attributes:
        config (Union[CategoricalSpace, FloatSpace, IntSpace]): Валидированный 
            объект конкретного типа распределения.
    """
    config: Annotated[
        Union[CategoricalSpace, FloatSpace, IntSpace],
        Field(discriminator="type")
    ]
    @model_validator(mode="before")
    @classmethod
    def _parse_list_to_dict(cls, data: Any) -> Any:
        """Преобразовать краткую списочную запись YAML 
        в структурированный словарь Pydantic.
        
        Логика парсинга:
        1. Если входные данные не список, возвращает их "как есть" 
            для стандартной обработки.
        2. Определяет тип распределения по индексу [1] (categorical) или [2] (numeric).
        3. Для числовых типов автоматически выводит подтип (int/float), 
            если он не указан явно.
        4. Формирует внутренний словарь с ключами `type`, `low`, `high`, `step` для 
           последующей дискриминации моделей.
        Args:
            data (Any): Исходные данные из YAML (список или словарь).
        Returns:
            Any: Словарь, готовый для инициализации Pydantic-модели.
        Raises:
            ValueError: Если список содержит менее 2 элементов.
        """

        if not isinstance(data, list):
            return data
        if len(data) < 2:
            raise ValueError("Search space list must have at least 2 elements")
        if data[1] == "categorical":
            return {"config": {"type": "categorical", "options": data[0]}}
        if len(data) >= 3:
            dist_type = str(data[2])
            payload = {"type": dist_type, "low": data[0], "high": data[1]}
            if len(data) == 4:
                payload["step"] = data[3]
            return {"config": payload}
        # Infer type for 2-element numeric lists
        if isinstance(data[0], int) and isinstance(data[1], int):
            return {"config": {"type": "int", "low": data[0], "high": data[1]}}
        return {"config": {"type": "float", "low": data[0], "high": data[1]}}
    @property
    def low(self) -> Any:
        return getattr(self.config, "low", None)
    @property
    def high(self) -> Any:
        return getattr(self.config, "high", None)
    @property
    def dist_type(self) -> str:
        return self.config.type
    @property
    def step(self) -> Optional[Union[int, float]]:
        return getattr(self.config, "step", None)
    @property
    def bounds(self) -> List[Any]:
        """Backward compatibility alias for the raw list structure."""
        if isinstance(self.config, CategoricalSpace):
            return [self.config.options, "categorical"]
        res = [self.low, self.high, self.dist_type]
        if self.step is not None:
            res.append(self.step)
        return res

# Общие параметры для Poisson, Gamma и Tweedie регрессоров
GLM_COMMON = {
    "alpha": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
    "fit_intercept": SearchSpaceEntry.model_validate([[True, False], "categorical"]),
    "max_iter": SearchSpaceEntry.model_validate([50, 1000, "int", 50]),
}
DEFAULT_SPACES: Dict[str, Dict[str, SearchSpaceEntry]] = {
    "elasticnet": {
        "alpha": SearchSpaceEntry.model_validate([1e-4, 10.0, "float_log"]),
        "l1_ratio": SearchSpaceEntry.model_validate([0.0, 1.0, "float"]),
    },
    "lasso": {
        "alpha": SearchSpaceEntry.model_validate([1e-4, 10.0, "float_log"]),
    },
    "ridge": {
        "alpha": SearchSpaceEntry.model_validate([1e-4, 10.0, "float_log"]),
    },
    "decision_tree": {
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
    },
    "random_forest": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
        "bootstrap": SearchSpaceEntry.model_validate([[True, False], "categorical"]),
    },
    "extra_trees": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([1, 10, "int"]),
    },
    "gradient_boosting": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.3, "float_log"]),
        "max_depth": SearchSpaceEntry.model_validate([2, 5, "int"]),
        "subsample": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
    },
    "svr": {
        "C": SearchSpaceEntry.model_validate([1e-2, 100.0, "float_log"]),
        "epsilon": SearchSpaceEntry.model_validate([1e-3, 1.0, "float_log"]),
        "kernel": SearchSpaceEntry.model_validate([["rbf", "poly", "sigmoid"], "categorical"]),
        "gamma": SearchSpaceEntry.model_validate([["scale", "auto"], "categorical"]),
    },
    "xgb": {
        "n_estimators": SearchSpaceEntry.model_validate([100, 800, "int", 100]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.3, "float_log"]),
        "max_depth": SearchSpaceEntry.model_validate([3, 10, "int"]),
        "subsample": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "colsample_bytree": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "gamma": SearchSpaceEntry.model_validate([0.0, 5.0, "float"]),
    },
    "sgdregressor": {
        "loss": SearchSpaceEntry.model_validate([
            ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], 
            "categorical"
        ]),
        "penalty": SearchSpaceEntry.model_validate([["l2", "l1", "elasticnet"], "categorical"]),
        "alpha": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "learning_rate": SearchSpaceEntry.model_validate([
            ["constant", "optimal", "invscaling", "adaptive"], 
            "categorical"
        ]),
        "eta0": SearchSpaceEntry.model_validate([1e-4, 1e-1, "float_log"]),
        "l1_ratio": SearchSpaceEntry.model_validate([0.0, 1.0, "float"]),
        "max_iter": SearchSpaceEntry.model_validate([500, 5000, "int", 500]),
    }
}

'''
# Общие параметры для Poisson, Gamma и Tweedie регрессоров
GLM_COMMON = {
    "alpha": SearchSpaceEntry(low=1e-6, high=1e-1, type="float", log=True),
    "fit_intercept": SearchSpaceEntry(choices=[True, False], type="categorical"),
    "max_iter": SearchSpaceEntry(low=50, high=1000, type="int", step=50),
}

DEFAULT_SPACES: Dict[str, Dict[str, SearchSpaceEntry]] = {
    "elasticnet": {
        "alpha": SearchSpaceEntry(low=1e-4, high=10.0, type="float", log=True),
        "l1_ratio": SearchSpaceEntry(low=0.0, high=1.0, type="float"),
    },
    "lasso": {
        "alpha": SearchSpaceEntry(low=1e-4, high=10.0, type="float", log=True),
    },
    "ridge": {
        "alpha": SearchSpaceEntry(low=1e-4, high=10.0, type="float", log=True),
    },
    "decision_tree": {
        "max_depth": SearchSpaceEntry(low=2, high=32, type="int", log=True),
        "min_samples_leaf": SearchSpaceEntry(low=1, high=10, type="int"),
    },
    "random_forest": {
        "n_estimators": SearchSpaceEntry(low=50, high=500, type="int", step=50),
        "max_depth": SearchSpaceEntry(low=2, high=32, type="int", log=True),
        "min_samples_leaf": SearchSpaceEntry(low=1, high=10, type="int"),
        "bootstrap": SearchSpaceEntry(choices=[True, False], type="categorical"),
    },
    "extra_trees": {
        "n_estimators": SearchSpaceEntry(low=50, high=500, type="int", step=50),
        "max_depth": SearchSpaceEntry(low=2, high=32, type="int", log=True),
        "min_samples_leaf": SearchSpaceEntry(low=1, high=10, type="int"),
    },
    "gradient_boosting": {
        "n_estimators": SearchSpaceEntry(low=50, high=500, type="int", step=50),
        "learning_rate": SearchSpaceEntry(low=0.01, high=0.3, type="float", log=True),
        "max_depth": SearchSpaceEntry(low=2, high=5, type="int"),
        "subsample": SearchSpaceEntry(low=0.5, high=1.0, type="float"),
    },
    "svr": {
        "C": SearchSpaceEntry(low=1e-2, high=100.0, type="float", log=True),
        "epsilon": SearchSpaceEntry(low=1e-3, high=1.0, type="float", log=True),
        "kernel": SearchSpaceEntry(choices=["rbf", "poly", "sigmoid"], type="categorical"),
        "gamma": SearchSpaceEntry(choices=["scale", "auto"], type="categorical"),
    },
    "xgb": {
        "n_estimators": SearchSpaceEntry(low=100, high=800, type="int", step=100),
        "learning_rate": SearchSpaceEntry(low=0.01, high=0.3, type="float", log=True),
        "max_depth": SearchSpaceEntry(low=3, high=10, type="int"),
        "subsample": SearchSpaceEntry(low=0.5, high=1.0, type="float"),
        "colsample_bytree": SearchSpaceEntry(low=0.5, high=1.0, type="float"),
        "gamma": SearchSpaceEntry(low=0.0, high=5.0, type="float"),
    },
    "sgdregressor": {
        "loss": SearchSpaceEntry(
            choices=["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], 
            type="categorical"
        ),
        "penalty": SearchSpaceEntry(choices=["l2", "l1", "elasticnet"], type="categorical"),
        "alpha": SearchSpaceEntry(low=1e-6, high=1e-1, type="float", log=True),
        "learning_rate": SearchSpaceEntry(
            choices=["constant", "optimal", "invscaling", "adaptive"], 
            type="categorical"
        ),
        "eta0": SearchSpaceEntry(low=1e-4, high=1e-1, type="float", log=True),
        "l1_ratio": SearchSpaceEntry(low=0.0, high=1.0, type="float"),
        "max_iter": SearchSpaceEntry(low=500, high=5000, type="int", step=500),
    }
}
'''