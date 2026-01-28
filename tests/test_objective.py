import numpy as np
import optuna
from sklearn.base import BaseEstimator
from configurable_automl_engine.validation import RANDOM_STATE


from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut
# Исправленные импорты
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.validation import norm_val_method
from optuna.trial import Trial
from typing import Any, Callable

# ═════════════════════ EXPORT: _objective (для тестов) ══════════════════════
def _objective(
    trial: Trial,
    X,
    y,
    *,
    val_method: str | ValidationStrategy,
    n_folds: int,
    model_factory: Callable[[Trial], Any],
    random_state: int | None = 42,
) -> float:
    """Целевая функция «accuracy» для внешнего юнит-теста."""
    method = norm_val_method(val_method)

    if method == "train_test_split":
        clf = model_factory(trial)
        clf.fit(X, y)
        return accuracy_score(y, clf.predict(X))

    if method == "k_fold":
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif method == "loo":
        cv = LeaveOneOut()
    else:
        raise ValueError(
            "val_method должен быть 'train_test_split', 'k_fold' или 'loo'"
        )

    scores: list[float] = []
    for tr_idx, te_idx in cv.split(X):
        clf = model_factory(trial)
        clf.fit(X[tr_idx], y[tr_idx])
        scores.append(accuracy_score(y[te_idx], clf.predict(X[te_idx])))
    return float(np.mean(scores))

class DummyClf(BaseEstimator):
    """Модель, которая предсказывает самый частый класс."""

    def fit(self, X, y):
        counts = np.bincount(y)
        self.major = counts.argmax()
        return self

    def predict(self, X):
        return np.full(len(X), self.major)


def test_objective_mean_equals_accuracy():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(40, 5))
    y = rng.integers(0, 2, 40)

    def factory(_trial):
        return DummyClf()

    score = _objective(
        trial=optuna.trial.FixedTrial({}),
        X=X,
        y=y,
        val_method="train_test_split",
        n_folds=1,
        model_factory=factory,
    )
    # Accuracy dummy-классификатора = доля мажоритарного класса
    expected = max(np.mean(y == 0), np.mean(y == 1))
    assert abs(score - expected) < 1e-9
