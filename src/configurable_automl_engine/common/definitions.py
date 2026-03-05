from enum import Enum

# ──────────────── validation enum ──────────────── #
class ValidationStrategy(str, Enum):
    """Разрешённые способы валидации при hyperopt."""
    train_test_split = "train_test_split"
    k_fold = "k_fold"
    loo = "loo"

# ──────────────── serialization enum ─────────────── #
class SerializationFormat(str, Enum):
    """
    Поддерживаемые форматы сериализации моделей.
    Rationale: Обеспечивает типизацию и единый источник истины для выбора
    между стандартным pickle и оптимизированным для тяжелых весов joblib.
    """
    pickle = "pickle"
    joblib = "joblib"

ALGO_PACKAGE_MAPPING: dict[str, str] = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "sklearn_rf": "scikit-learn",
}