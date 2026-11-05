"""Small-data behaviour of the quantile scaler and the quantile winsorizer.

* ``_FoldSafeQuantileTransformer(output_distribution="normal")`` must not
  map the training extremes to sklearn's ±5.2 clipping bound (that would
  manufacture two high-leverage points per column on small folds): every
  value is bounded by ``ppf(1 - 0.5/n_quantiles_)``.
* ``Winsorizer(method="quantile")`` warns (once per configuration) when
  ``lower_quantile × rows < 1``: the bounds then always clip the extremes.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer

from configurable_automl_engine.preprocessing.outliers import (
    Winsorizer,
    reset_small_sample_warnings,
)
from configurable_automl_engine.preprocessing.quantile import (
    _FoldSafeQuantileTransformer,
)

# ──────────────────────────────────────────────────────────────────────────
#  Quantile scaler, output_distribution="normal"
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_rows", [14, 50, 220, 2000])
def test_normal_output_is_bounded_by_the_half_bin_ppf(n_rows):
    """max|z| <= ppf(1 - 0.5/n_quantiles_) for every column."""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=n_rows), rng.standard_t(df=2, size=n_rows)])
    qt = _FoldSafeQuantileTransformer(output_distribution="normal")
    z = qt.fit_transform(X)
    bound = norm.ppf(1.0 - 0.5 / qt.n_quantiles_)
    assert np.all(np.abs(z) <= bound + 1e-12)
    # The extremes reach the bound (the mapping is not over-shrunk).
    np.testing.assert_allclose(np.abs(z).max(axis=0), bound)


def test_normal_output_on_14_rows_matches_the_expected_values():
    """On 14 rows the minimum is -1.80 and the second point -1.24."""
    X = np.arange(14, dtype=float).reshape(-1, 1)
    z = np.sort(
        _FoldSafeQuantileTransformer(output_distribution="normal")
        .fit_transform(X)
        .ravel()
    )
    np.testing.assert_allclose(z[:2], [-1.80, -1.24], atol=0.01)
    np.testing.assert_allclose(z, -z[::-1])  # symmetric


def test_normal_output_is_monotone_and_keeps_nan():
    """The mapping preserves the order of the uniform output and NaN."""
    rng = np.random.default_rng(1)
    X = rng.lognormal(size=(40, 1))
    X[3, 0] = np.nan
    qt = _FoldSafeQuantileTransformer(output_distribution="normal").fit(X)
    z = qt.transform(X).ravel()
    assert np.isnan(z[3])
    mask = ~np.isnan(X.ravel())
    order = np.argsort(X.ravel()[mask])
    assert np.all(np.diff(z[mask][order]) >= 0)


def test_uniform_output_is_unchanged():
    """The default uniform output still equals sklearn's own output."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    ours = _FoldSafeQuantileTransformer().fit_transform(X)
    theirs = QuantileTransformer(n_quantiles=30).fit_transform(X)
    np.testing.assert_allclose(ours, theirs)


def test_legacy_pickled_normal_transformer_keeps_its_output():
    """A model pickled with an inner 'normal' transformer is not remapped."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 1))
    qt = _FoldSafeQuantileTransformer(output_distribution="normal").fit(X)
    # Simulate the pre-fix artifact layout.
    qt._transformer_ = QuantileTransformer(
        n_quantiles=30, output_distribution="normal"
    ).fit(X)
    legacy = pickle.loads(pickle.dumps(qt))
    np.testing.assert_allclose(legacy.transform(X), qt._transformer_.transform(X))


# ──────────────────────────────────────────────────────────────────────────
#  Winsorizer: small-sample WARNING for method="quantile"
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_warn_once():
    reset_small_sample_warnings()
    yield
    reset_small_sample_warnings()


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "quantile bounds" in r.getMessage()
    ]


def test_quantile_winsorizer_warns_on_small_data(caplog):
    """lower_quantile × rows < 1 → one WARNING recommending iqr."""
    X = np.arange(14, dtype=float).reshape(-1, 1)
    with caplog.at_level(logging.WARNING):
        Winsorizer(method="quantile", algorithm_name="ridge").fit(X)
        Winsorizer(method="quantile", algorithm_name="ridge").fit(X)
    messages = _warnings(caplog)
    assert len(messages) == 1  # once per configuration
    assert "ridge" in messages[0]
    assert "iqr" in messages[0]


def test_quantile_winsorizer_does_not_warn_on_enough_rows(caplog):
    """With lower_quantile × rows >= 1 no WARNING is logged."""
    X = np.arange(200, dtype=float).reshape(-1, 1)
    with caplog.at_level(logging.WARNING):
        Winsorizer(method="quantile").fit(X)
    assert _warnings(caplog) == []


def test_upper_tail_alone_triggers_the_warning(caplog):
    """A coarse upper tail warns even when the lower one is fine."""
    X = np.arange(50, dtype=float).reshape(-1, 1)
    with caplog.at_level(logging.WARNING):
        Winsorizer(method="quantile", lower_quantile=0.1, upper_quantile=0.99).fit(X)
    assert len(_warnings(caplog)) == 1


def test_iqr_winsorizer_never_warns(caplog):
    """method='iqr' adapts to small data: no small-sample WARNING."""
    X = np.arange(14, dtype=float).reshape(-1, 1)
    with caplog.at_level(logging.WARNING):
        Winsorizer(method="iqr").fit(X)
    assert _warnings(caplog) == []


# ──────────────────────────────────────────────────────────────────────────
#  Quality vs the mean + StandardScaler baseline
# ──────────────────────────────────────────────────────────────────────────


def _spiky_x_dataset(n: int, seed: int):
    """Linear target on clean X; X then gets measurement spikes (x50)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = a + 0.5 * b + rng.normal(0, 0.3, n)
    rows = rng.choice(n, max(2, n // 25), replace=False)
    a = a.copy()
    a[rows] *= 50.0
    return pd.DataFrame({"a": a, "b": b}), pd.Series(y)


@pytest.mark.parametrize("n_rows", [30, 220])
def test_robust_presets_beat_the_mean_standard_baseline(n_rows):
    """With spikes in X the robust presets give a lower CV RMSE for ridge.

    The baseline (mean + StandardScaler) extrapolates on the spikes; iqr
    winsorization and the bounded normal quantile scaling do not. The quantile
    winsorizer at 1%/99% is also checked to lose to iqr on small data.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    from configurable_automl_engine.preprocessing.builder import build_preprocessor

    configs = {
        "baseline": {"imputation_strategy": "mean", "scaling": "standard"},
        "iqr": {
            "imputation_strategy": "median",
            "scaling": "standard",
            "outliers": "iqr",
        },
        "quantile_clip": {
            "imputation_strategy": "median",
            "scaling": "standard",
            "outliers": "quantile",
        },
        "iqr_qnormal": {
            "imputation_strategy": "median",
            "scaling": {"type": "quantile", "output_distribution": "normal"},
            "outliers": "iqr",
        },
    }
    rmse: dict[str, float] = {}
    for name, cfg in configs.items():
        scores = []
        for seed in range(10):
            X, y = _spiky_x_dataset(n_rows, seed)
            pre = build_preprocessor(["a", "b"], [], ["a", "b"], **cfg)
            scores.append(
                -cross_val_score(
                    make_pipeline(pre, Ridge()),
                    X,
                    y,
                    cv=KFold(5, shuffle=True, random_state=0),
                    scoring="neg_root_mean_squared_error",
                ).mean()
            )
        rmse[name] = float(np.mean(scores))
    assert rmse["iqr"] < 0.5 * rmse["baseline"], rmse
    assert rmse["iqr_qnormal"] < 0.5 * rmse["baseline"], rmse
    if n_rows < 100:
        assert rmse["iqr"] < rmse["quantile_clip"], rmse
