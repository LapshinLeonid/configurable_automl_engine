from enum import Enum

# ──────────────── validation enum ──────────────── #
class ValidationStrategy(str, Enum):
    """Разрешённые способы валидации при hyperopt."""
    train_test_split = "train_test_split"
    k_fold = "k_fold"
    loo = "loo"
