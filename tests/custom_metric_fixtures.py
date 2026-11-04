"""Importable custom metrics and scorers for the custom-metric tests.

The tests refer to them by dotted path (``tests.custom_metric_fixtures:...``),
so they must live in an importable module (the worker processes import them
again from the path).
"""

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error

CALLS = {"metric": 0}


def custom_rmse(y_true, y_pred):
    """The plain RMSE: non-additive, lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def custom_mae(y_true, y_pred):
    """The plain MAE: additive, lower is better."""
    return float(mean_absolute_error(y_true, y_pred))


def scaled_r2(y_true, y_pred, scale=1.0):
    """R² multiplied by ``scale`` (exercises static kwargs)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(scale * (1.0 - ss_res / ss_tot))


def counting_mae(y_true, y_pred):
    """MAE that counts its calls (proves the function is really used)."""
    CALLS["metric"] += 1
    return float(mean_absolute_error(y_true, y_pred))


def raising_metric(y_true, y_pred):
    """Always fails at evaluation time."""
    raise RuntimeError("boom from a custom metric")


def one_arg(y_true):
    """A function with a wrong signature."""
    return 0.0


#: A ``make_scorer`` scorer: negated MAE (sklearn convention).
neg_mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


def plain_scorer(estimator, X, y):
    """A plain callable scorer returning the R² of the estimator."""
    return float(estimator.score(X, y))


def plain_error_scorer(estimator, X, y):
    """A plain callable scorer returning a natural error (lower is better)."""
    pred = estimator.predict(X)
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - pred)))


class Holder:
    """Nested attribute access (``module:Holder.metric``)."""

    @staticmethod
    def metric(y_true, y_pred):
        return custom_mae(y_true, y_pred)


NOT_CALLABLE = 42
