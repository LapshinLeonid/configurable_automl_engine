"""Tests for applying ``inf_policy`` at prediction time.

``ModelTrainer.fit`` replaces ±inf with NaN under the default
``inf_policy="as_missing"``; ``ModelTrainer.predict`` must apply the same
policy so that predicting on the training frame works end-to-end
(train + predict).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.validation_utils import apply_inf_policy
from configurable_automl_engine.trainer import ModelTrainer, TrainingError


def _frame_with_inf() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            "b": rng.normal(size=40),
            "c": np.arange(40.0),
        }
    )
    y = pd.Series(2.0 * X["b"] + X["c"] + rng.normal(scale=0.1, size=40))
    X.loc[[1, 5], "a"] = np.inf
    X.loc[7, "b"] = -np.inf
    return X, y


# ---------------------------------------------------------------------------
# apply_inf_policy helper
# ---------------------------------------------------------------------------


def test_apply_inf_policy_dataframe_as_missing(caplog):
    """±inf → NaN in numeric columns; WARNING lists columns and counts."""
    X = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": [1.0, 2.0, 3.0], "s": list("xyz")})
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out = apply_inf_policy(X, "as_missing")
    assert out["a"].isna().sum() == 2
    assert out["b"].tolist() == [1.0, 2.0, 3.0]
    assert out["s"].tolist() == list("xyz")
    assert "a (2 value(s))" in caplog.text
    # The input is not modified in place.
    assert np.isinf(X["a"]).sum() == 2


def test_apply_inf_policy_dataframe_error_names_columns():
    X = pd.DataFrame({"a": [1.0, np.inf], "b": [1.0, 2.0], "c": [-np.inf, 0.0]})
    with pytest.raises(ValueError, match="a \\(1 value") as exc:
        apply_inf_policy(X, "error")
    assert "c (1 value" in str(exc.value)
    assert "b (" not in str(exc.value)


def test_apply_inf_policy_ndarray_as_missing(caplog):
    X = np.array([[1.0, np.inf], [2.0, 3.0]])
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        out = apply_inf_policy(X, "as_missing")
    assert np.isnan(out[0, 1])
    assert np.isinf(X[0, 1])  # not modified in place
    assert "column 1" in caplog.text


def test_apply_inf_policy_ndarray_error():
    with pytest.raises(ValueError, match="column 0"):
        apply_inf_policy(np.array([[np.inf, 1.0]]), "error")


def test_apply_inf_policy_no_inf_returns_input_unchanged(caplog):
    X = pd.DataFrame({"a": [1.0, np.nan]})
    arr = np.array([[1, 2], [3, 4]])
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        assert apply_inf_policy(X, "error") is X
        assert apply_inf_policy(arr, "error") is arr
    assert "Infinite" not in caplog.text


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame({"s": ["a", "b"]}),
        pd.DataFrame(),
        np.empty((0, 3)),
        np.array([["a", "b"]], dtype=object),
    ],
)
def test_apply_inf_policy_edge_inputs_pass_through(X):
    """Non-numeric / empty inputs are returned as is."""
    assert apply_inf_policy(X, "error") is X


def test_apply_inf_policy_unknown_policy():
    with pytest.raises(ValueError, match="inf_policy"):
        apply_inf_policy(np.array([[1.0]]), "bogus")


# ---------------------------------------------------------------------------
# ModelTrainer.fit + predict
# ---------------------------------------------------------------------------


def test_predict_on_training_frame_with_inf(caplog):
    """Fit on a frame with inf, then predict on the same frame."""
    X, y = _frame_with_inf()
    trainer = ModelTrainer(algorithm="ridge").fit(X, y)

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        preds = trainer.predict(X)
    assert "Infinite values" in caplog.text
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()

    expected = trainer.predict(X.replace([np.inf, -np.inf], np.nan))
    np.testing.assert_allclose(preds, expected)


def test_predict_ndarray_with_inf():
    X, y = _frame_with_inf()
    trainer = ModelTrainer(algorithm="ridge").fit(X, y)
    X_arr = X.to_numpy()
    preds = trainer.predict(X_arr)
    assert np.isfinite(preds).all()
    expected = trainer.predict(X.replace([np.inf, -np.inf], np.nan))
    np.testing.assert_allclose(preds, expected)
    assert np.isinf(X_arr).sum() == 3  # caller's array untouched


def test_inf_policy_error_raises_in_fit_and_predict():
    X, y = _frame_with_inf()
    with pytest.raises(TrainingError, match="[Ii]nfinite"):
        ModelTrainer(algorithm="ridge", inf_policy="error").fit(X, y)

    trainer = ModelTrainer(algorithm="ridge", inf_policy="error").fit(
        X.replace([np.inf, -np.inf], np.nan), y
    )
    with pytest.raises(TrainingError, match="[Ii]nfinite") as exc:
        trainer.predict(X)
    msg = str(exc.value)
    assert "a (2 value(s))" in msg and "b (1 value(s))" in msg
    assert not msg.startswith("Error during prediction")
    with pytest.raises(TrainingError, match="column 0"):
        trainer.predict(X.to_numpy())


def test_predict_inf_in_dropped_extra_column_is_ignored(caplog):
    """Extra columns are dropped before the check — no warning for them."""
    X, y = _frame_with_inf()
    clean = X.replace([np.inf, -np.inf], np.nan)
    trainer = ModelTrainer(algorithm="ridge", inf_policy="error").fit(clean, y)
    X_extra = clean.assign(extra=np.inf)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine"):
        preds = trainer.predict(X_extra)
    assert np.isfinite(preds).all()
    assert "Infinite" not in caplog.text
