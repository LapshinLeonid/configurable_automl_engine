"""BBC-CV: bootstrap bias-corrected estimate of the selection procedure.

Bootstrap Bias Corrected CV (Tsamardinos, Greasidou, Borboudakis, 2018)
estimates the out-of-sample score of a selection procedure from the
out-of-fold predictions of ALL evaluated configurations, without retraining.
Split out of ``common.metrics`` (the module-size limit) as a cohesive unit;
``common.metrics`` re-exports the public names so the documented import path
(``configurable_automl_engine.common.metrics``) keeps working.

All configurations are compared on the SAME set of rows — the rows every
configuration predicted on top of the finite-target rows (a configuration
with non-finite predictions on a row other configurations predicted is
penalized by its shrinking usable row subset, instead of being scored on a
private row subset).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .metrics import _SCORER_OBJECTS, _nrmse, _rmse, pooled_scorer_value
from .sklearn_compat import scorer_raw_parts

# --------------------------------------------------------------------------- #
#  BBC-CV: bootstrap bias-corrected estimate of the selection procedure
# --------------------------------------------------------------------------- #
#: Number of bootstrap resamples of the BBC-CV estimate.
BBC_N_BOOTSTRAP = 500
#: Upper bound on the kept OOF values (trials x rows) for BBC-CV: about
#: 32 MB of float64; the bootstrap cost scales with it as well.
BBC_MAX_OOF_CELLS = 4_000_000


def _columnwise_raw_values(
    scorer: Any, y_true: np.ndarray, preds: np.ndarray
) -> np.ndarray:
    """Compute the raw (sign-applied) scorer value for every prediction column.

    The comparison metrics of the engine (RMSE, MSE, MAE, R², NRMSE) are
    computed vectorized over all columns at once with the same non-finite
    masking as ``pooled_scorer_value`` (per column); any other scorer falls
    back to one ``pooled_scorer_value`` call per column. A column without a
    valid pair yields NaN.

    The caller guarantees the BBC-CV common-row contract (``bbc_cv_score``
    keeps only rows every configuration predicted), so a column without a
    valid pair can only mean a non-finite target row — reported as NaN, as
    before.

    Args:
        scorer: A ``make_scorer`` object (its raw metric parts are read
            through ``common.sklearn_compat.scorer_raw_parts``).
        y_true: The true values, shape ``(n_rows,)``.
        preds: The predictions, shape ``(n_rows, n_columns)``.


    Returns:
        np.ndarray: The raw value of every column, shape ``(n_columns,)``.
    """
    score_func, raw_sign, kwargs = scorer_raw_parts(scorer)
    sign = float(raw_sign)
    vectorized = {_rmse, _nrmse, mean_squared_error, mean_absolute_error, r2_score}
    if score_func in vectorized and not kwargs:
        valid = np.isfinite(preds) & np.isfinite(y_true)[:, None]
        n_valid = valid.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.where(valid, preds - y_true[:, None], 0.0)
            if score_func is mean_absolute_error:
                values = np.abs(err).sum(axis=0) / n_valid
            else:
                ss_res = (err**2).sum(axis=0)
                if score_func is mean_squared_error:
                    values = ss_res / n_valid
                elif score_func is _rmse:
                    values = np.sqrt(ss_res / n_valid)
                elif score_func is _nrmse:
                    y_max = np.where(valid, y_true[:, None], -np.inf).max(axis=0)
                    y_min = np.where(valid, y_true[:, None], np.inf).min(axis=0)
                    span = y_max - y_min
                    values = np.where(
                        span < 1e-6, np.inf, np.sqrt(ss_res / n_valid) / span
                    )
                else:  # r2_score, with sklearn's force_finite semantics
                    y_col = np.where(valid, y_true[:, None], 0.0)
                    y_mean = y_col.sum(axis=0) / n_valid
                    ss_tot = np.where(valid, (y_true[:, None] - y_mean) ** 2, 0.0).sum(
                        axis=0
                    )
                    values = np.where(
                        ss_tot > 0.0,
                        1.0 - ss_res / ss_tot,
                        np.where(ss_res == 0.0, 1.0, 0.0),
                    )
        return np.where(n_valid > 0, sign * values, np.nan)
    out = np.full(preds.shape[1], np.nan, dtype=float)
    for col in range(preds.shape[1]):
        try:
            out[col] = pooled_scorer_value(scorer, y_true, preds[:, col])
        except (ValueError, TypeError):
            continue
    return out


def bbc_cv_score(
    scorer: Any,
    y_true: Any,
    oof_matrix: Any,
    *,
    n_bootstrap: int = BBC_N_BOOTSTRAP,
    random_state: int | None = None,
) -> float:
    """Estimate the out-of-sample score of the selection procedure (BBC-CV).

    Bootstrap Bias Corrected CV (Tsamardinos, Greasidou, Borboudakis, 2018):
    the best-trial CV score is optimistic because the winner is the maximum
    over many configurations evaluated on the same folds. BBC-CV removes this
    bias without any retraining, from the out-of-fold predictions of ALL
    evaluated configurations: each bootstrap resample of the rows selects the
    best configuration on the in-bag rows (the selection step) and scores it
    on the out-of-bag rows (rows the selection did not see); the estimate is
    the mean of the out-of-bag scores.

    For R² and NRMSE the bootstrap runs on MSE / RMSE and the result is
    normalized by the variance / range of the target over all usable rows:
    the selection is identical, and the small out-of-bag subsets do not bias
    the normalizer.

    Rows with a non-finite target and rows no configuration predicted (e.g.
    the train part of ``train_test_split``) are dropped. Non-finite
    predictions of a configuration on a row that SOME other configuration
    predicted are NOT masked per configuration: all configurations are
    compared on the SAME set of rows — the rows every configuration predicted
    on top of the finite-target rows (the previous per-configuration masking
    let models with different coverage be scored on different row subsets,
    which especially distorts a leaderboard comparison). Identical
    configurations (duplicate OOF vectors) are collapsed. Ties in the
    selection step go to the first configuration.

    Args:
        scorer: A ``make_scorer`` object of the comparison metric.
        y_true: The true target aligned with the OOF columns.
        oof_matrix: The OOF predictions, shape
            ``(n_configurations, n_rows)``.
        n_bootstrap: The number of bootstrap resamples (>= 1).
        random_state: The bootstrap seed; ``None`` — not
            reproducible.

    Returns:
        float: The raw (sign-applied, higher is better) BBC-CV estimate.

    Raises:
        ValueError: If the shapes do not match, ``n_bootstrap < 1``, fewer
            than 2 usable rows remain, or no resample produced a finite
            out-of-bag score.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    configs = np.asarray(oof_matrix, dtype=float)
    if configs.ndim != 2 or configs.shape[1] != y.shape[0] or configs.shape[0] == 0:
        raise ValueError(
            f"BBC-CV needs an OOF matrix of shape (n_configurations, {y.shape[0]}); "
            f"got {configs.shape}."
        )
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}.")
    preds = configs.T
    # The comparison must run on a COMMON set of rows for all configurations:
    # the rows with a finite target that EVERY configuration predicted. A
    # configuration with non-finite predictions on a row other
    # configurations predicted is thereby penalized — its usable row subset
    # shrinks (down to the "fewer than 2 usable rows" failure) — instead of
    # being scored on a private row subset (per-configuration masking would
    # let models with different coverage be compared on different rows).
    rows = np.isfinite(y) & np.isfinite(preds).all(axis=1)
    y = y[rows]
    preds = preds[rows]
    n_rows = y.shape[0]
    if n_rows < 2:
        raise ValueError(f"BBC-CV needs at least 2 usable rows; got {n_rows}.")
    preds = np.unique(preds, axis=1)

    # R² and NRMSE normalize by a statistic of y (variance / range). On a
    # small out-of-bag subset that statistic is itself biased (R² is pulled
    # down, NRMSE pushed up), so the bootstrap runs on the underlying error
    # (MSE / RMSE) and the normalization uses all usable rows. The selection
    # is unchanged: within one resample the normalizer is the same for every
    # configuration, so ranking by R² equals ranking by MSE (NRMSE by RMSE).
    score_func, scorer_sign_value, kwargs = scorer_raw_parts(scorer)
    if not kwargs and score_func is r2_score:
        y_var = float(np.var(y))
        if y_var > 0.0:
            mse = -_bbc_mean_oob(
                _SCORER_OBJECTS["mse"][0], y, preds, n_bootstrap, random_state
            )
            return float(scorer_sign_value) * (1.0 - mse / y_var)
    if not kwargs and score_func is _nrmse:
        y_span = float(np.max(y) - np.min(y))
        if y_span >= 1e-6:
            rmse = -_bbc_mean_oob(
                _SCORER_OBJECTS["rmse"][0], y, preds, n_bootstrap, random_state
            )
            return float(scorer_sign_value) * (rmse / y_span)
    return _bbc_mean_oob(scorer, y, preds, n_bootstrap, random_state)


def _bbc_mean_oob(
    scorer: Any,
    y: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int,
    random_state: int | None,
) -> float:
    """Run the BBC-CV bootstrap loop and return the mean out-of-bag score.

    Args:
        scorer: The scorer used for both selection and evaluation.
        y: The usable target rows, shape ``(n_rows,)``.
        preds: The OOF predictions, shape
            ``(n_rows, n_configurations)``.
        n_bootstrap: The number of bootstrap resamples.
        random_state: The bootstrap seed.


    Returns:
        float: The mean raw (sign-applied) out-of-bag score.
    """
    n_rows = y.shape[0]
    rng = np.random.default_rng(random_state)
    oob_scores: list[float] = []
    for _ in range(n_bootstrap):
        in_bag = rng.integers(0, n_rows, size=n_rows)
        oob_mask = np.ones(n_rows, dtype=bool)
        oob_mask[in_bag] = False
        if not oob_mask.any():
            continue
        in_bag_values = _columnwise_raw_values(scorer, y[in_bag], preds[in_bag])
        if not np.isfinite(in_bag_values).any():
            continue
        best = int(
            np.nanargmax(np.where(np.isfinite(in_bag_values), in_bag_values, np.nan))
        )
        oob_value = _columnwise_raw_values(
            scorer, y[oob_mask], preds[oob_mask][:, [best]]
        )[0]
        if np.isfinite(oob_value):
            oob_scores.append(float(oob_value))
    if not oob_scores:
        raise ValueError("BBC-CV: no bootstrap resample produced a finite score.")
    return float(np.mean(oob_scores))
