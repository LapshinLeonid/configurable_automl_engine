from typing import Any, Dict, Optional, Literal, Annotated, Union, List
from pydantic import BaseModel, Field, model_validator

from typing import TypeVar, Generic

T = TypeVar("T", str, int, float, bool)

# ───────────────── hyperopt ───────────────── #
class BaseDistribution(BaseModel):
    """Абстрактный базовый класс для распределений поиска."""
    type: str
class CategoricalSpace(BaseDistribution, Generic[T]):
    """Распределение для категориальных признаков: [options, 'categorical']."""
    type: Literal["categorical"]
    options: List[T]
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
        # Если это одиночное значение (строка, число) — обернуть как категориальное
        if len(data) == 1:
            # Одиночное значение → категориальное распределение с одним вариантом
            return {"config": {"type": "categorical", "options": data}}
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
        "kernel": SearchSpaceEntry.model_validate(
            [["rbf", "poly", "sigmoid"], "categorical"]),
        "gamma": SearchSpaceEntry.model_validate([["scale", "auto"], "categorical"]),
    },
    "xgboosting": {
        "n_estimators": SearchSpaceEntry.model_validate([100, 800, "int", 100]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.3, "float_log"]),
        "max_depth": SearchSpaceEntry.model_validate([3, 10, "int"]),
        "subsample": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "colsample_bytree": SearchSpaceEntry.model_validate([0.5, 1.0, "float"]),
        "gamma": SearchSpaceEntry.model_validate([0.0, 5.0, "float"]),
    },
    "sgdregressor": {
        "loss": SearchSpaceEntry.model_validate([
            ["squared_error", 
             "huber", 
             "epsilon_insensitive", 
             "squared_epsilon_insensitive"], "categorical"
        ]),
        "penalty": SearchSpaceEntry.model_validate(
            [["l2", "l1", "elasticnet"], "categorical"]),
        "alpha": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "learning_rate": SearchSpaceEntry.model_validate([
            ["constant", "optimal", "invscaling", "adaptive"], 
            "categorical"
        ]),
        "eta0": SearchSpaceEntry.model_validate([1e-4, 1e-1, "float_log"]),
        "l1_ratio": SearchSpaceEntry.model_validate([0.0, 1.0, "float"]),
        "max_iter": SearchSpaceEntry.model_validate([500, 5000, "int", 500]),
    },
    "adaboost": {
        "n_estimators": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 1.0, "float_log"]),
        "loss": SearchSpaceEntry.model_validate(
            [["linear", "square", "exponential"], "categorical"]),
    },
    "poissonregressor": {**GLM_COMMON},
    "gammaregressor": {**GLM_COMMON},
    "tweedieregressor": {**GLM_COMMON},
    "glm": {**GLM_COMMON},
    "ardregression": {
        "n_iter": SearchSpaceEntry.model_validate([100, 1000, "int", 100]),
        "alpha_1": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "alpha_2": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "lambda_1": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
        "lambda_2": SearchSpaceEntry.model_validate([1e-6, 1e-1, "float_log"]),
    },
    "nearest_neighbors_regression": {
        "n_neighbors": SearchSpaceEntry.model_validate([1, 50, "int"]),
        "weights": SearchSpaceEntry.model_validate(
            [["uniform", "distance"], "categorical"]),
        "p": SearchSpaceEntry.model_validate([1, 2, "int"]),
    },
    "isotonic_regression": {
        "increasing": SearchSpaceEntry.model_validate(
            [[True, False], "categorical"]),
    },
    "gaussian_process_regression": {},
}

ALGO_HYPERPARAMETER_REGISTRY: Dict[str, set[str]] = {
    algo: set(params.keys())
    for algo, params in DEFAULT_SPACES.items()
}