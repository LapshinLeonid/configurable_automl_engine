"""Tests of the stable scikit-learn adapter (common/sklearn_compat.py).

The adapter localizes the only two private scikit-learn details the engine
relies on:

- the raw metric parts of a ``make_scorer`` object (``_score_func`` /
  ``_sign`` / ``_kwargs``) used by the pooled out-of-fold and BBC-CV
  evaluation;
- the feature-name recording of the ``fit`` convention
  (sklearn's ``_check_feature_names`` with ``reset=True``).

The tests cover both parts of the contract against real sklearn scorers and
the defensive fallbacks for scorers that do not expose the private
attributes (plain callable adapters), plus the own feature-name recording
used when sklearn's private helper is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    get_scorer,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from configurable_automl_engine.common.sklearn_compat import (
    check_feature_names,
    scorer_raw_parts,
    scorer_sign,
)


# --------------------------------------------------------------------------- #
#  scorer_raw_parts / scorer_sign — the stable scorer contract
# --------------------------------------------------------------------------- #
def test_scorer_raw_parts_of_make_scorer() -> None:
    """A make_scorer scorer exposes its raw function, sign and kwargs."""
    scorer = make_scorer(
        mean_absolute_error, greater_is_better=False, foo="bar"
    )
    score_func, sign, kwargs = scorer_raw_parts(scorer)
    assert score_func is mean_absolute_error
    assert sign == -1
    assert kwargs == {"foo": "bar"}


def test_scorer_raw_parts_of_builtin_scorer() -> None:
    """A registry scorer (sklearn.metrics.get_scorer) exposes r2_score."""
    score_func, sign, kwargs = scorer_raw_parts(get_scorer("r2"))
    assert score_func is r2_score
    assert sign == 1
    assert kwargs == {}


def test_scorer_raw_parts_of_plain_callable_scorer() -> None:
    """A scorer without the private attributes: no raw function, defaults."""

    class PlainScorer:
        def __call__(self, estimator: Any, X: Any, y: Any) -> float:
            return 0.5

    score_func, sign, kwargs = scorer_raw_parts(PlainScorer())
    assert score_func is None
    assert sign == 1
    assert kwargs == {}


def test_scorer_raw_parts_copies_kwargs() -> None:
    """The returned kwargs dict is a copy: adding a key does not touch the scorer."""
    scorer = make_scorer(mean_absolute_error, foo="bar")
    _, _, kwargs = scorer_raw_parts(scorer)
    kwargs["extra"] = 1
    assert "extra" not in scorer._kwargs


@pytest.mark.parametrize(
    ("greater_is_better", "expected"),
    [(True, 1), (False, -1)],
)
def test_scorer_sign_matches_make_scorer_flag(
    greater_is_better: bool, expected: int
) -> None:
    """The sign matches the greater_is_better flag of make_scorer."""
    scorer = make_scorer(mean_squared_error, greater_is_better=greater_is_better)
    assert scorer_sign(scorer) == expected


def test_scorer_sign_of_plain_callable_is_none() -> None:
    """A scorer without the direction flag returns None (conservative fallback)."""

    class PlainScorer:
        def __call__(self, estimator: Any, X: Any, y: Any) -> float:
            return 0.5

    assert scorer_sign(PlainScorer()) is None


def test_scorer_raw_parts_agree_with_sklearn_pooling() -> None:
    """Re-applying the raw parts equals the scorer output on the same data."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=20)
    y_true = y[5:]
    y_pred = Ridge().fit(X[:15], y[:15]).predict(X[5:])
    scorer = get_scorer("neg_mean_absolute_error")
    score_func, sign, kwargs = scorer_raw_parts(scorer)
    manual = sign * score_func(y_true, y_pred, **kwargs)
    assert manual == pytest.approx(scorer(Ridge().fit(X[:15], y[:15]), X[5:], y_true))


# --------------------------------------------------------------------------- #
#  check_feature_names — the fit-time feature-name recording
# --------------------------------------------------------------------------- #
class _NameRecorder:
    """A minimal estimator standing in for the engine transformers."""

    def __init__(self) -> None:
        self.feature_names_in_: Any = None


def test_check_feature_names_records_dataframe_columns() -> None:
    """A DataFrame input records its column names as feature_names_in_."""
    est = _NameRecorder()
    X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    check_feature_names(est, X)
    assert list(est.feature_names_in_) == ["a", "b", "c"]


def test_check_feature_names_nameless_input_records_nothing() -> None:
    """An ndarray input records no attribute."""
    est = _NameRecorder()
    check_feature_names(est, np.zeros((2, 3)))
    assert not hasattr(est, "feature_names_in_")


def test_check_feature_names_refit_on_nameless_input_drops_stale_names() -> None:
    """Refitting on an ndarray drops feature_names_in_ recorded earlier."""
    est = _NameRecorder()
    check_feature_names(est, pd.DataFrame({"a": [1.0], "b": [2.0]}))
    assert hasattr(est, "feature_names_in_")
    check_feature_names(est, np.zeros((2, 2)))
    assert not hasattr(est, "feature_names_in_")


def test_check_feature_names_matches_sklearn_convention() -> None:
    """The recording equals the sklearn BaseEstimator convention."""
    from sklearn.base import BaseEstimator

    class SklearnRef(BaseEstimator):
        def fit(self, X: Any, y: Any = None) -> "SklearnRef":
            check_feature_names(self, X)
            return self

    X = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
    ref = SklearnRef().fit(X)
    own = _NameRecorder()
    check_feature_names(own, X)
    np.testing.assert_array_equal(own.feature_names_in_, ref.feature_names_in_)


def test_check_feature_names_uses_own_recording_when_private_helper_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without sklearn's private helper the own recording keeps the behavior."""
    import configurable_automl_engine.common.sklearn_compat as compat

    monkeypatch.setattr(compat, "_sk_check_feature_names", None)
    est = _NameRecorder()
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    check_feature_names(est, X)
    assert list(est.feature_names_in_) == ["a", "b"]
    # And the stale-name drop still works on the fallback path.
    check_feature_names(est, np.zeros((1, 2)))
    assert not hasattr(est, "feature_names_in_")


def test_check_feature_names_own_recording_matches_private_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The own recording produces the same attribute as the private helper."""
    import configurable_automl_engine.common.sklearn_compat as compat

    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    via_private = _NameRecorder()
    check_feature_names(via_private, X)
    monkeypatch.setattr(compat, "_sk_check_feature_names", None)
    via_own = _NameRecorder()
    check_feature_names(via_own, X)
    np.testing.assert_array_equal(via_own.feature_names_in_, via_private.feature_names_in_)


# --------------------------------------------------------------------------- #
#  Engine consumers keep working through the adapter
# --------------------------------------------------------------------------- #
def test_engine_pooled_value_works_for_all_builtin_registry_scorers() -> None:
    """The pooled evaluation stays intact for the engine's own scorers."""
    from configurable_automl_engine.common.metrics import pooled_scorer_value

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.8, 3.1, 3.9])
    for name in ("rmse", "mae", "mse", "r2"):
        scorer = make_scorer(
            {
                "rmse": lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
                "mae": mean_absolute_error,
                "mse": mean_squared_error,
                "r2": r2_score,
            }[name],
            greater_is_better=name == "r2",
        )
        value = pooled_scorer_value(scorer, y_true, y_pred)
        assert np.isfinite(value)
