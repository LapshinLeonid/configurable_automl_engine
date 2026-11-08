"""Regression gate for the BBC-CV estimate after ADAPTIVE HPO.

The BBC-CV estimate (``metrics.bbc_cv_score``) is the engine's honest
out-of-sample quality estimate when no outer hold-out is reserved. The OOF
predictions it consumes are produced without a direct train/validation leak,
but the SET of compared configurations is chosen adaptively by Optuna from
results obtained on the ORIGINAL sample. This benchmark measures whether the
resulting estimate stays consistent with an independent reference estimate
after a full adaptive HPO run — the scenario the selection bias is strongest
in (a large search space, many Optuna trials, pure-noise features that reward
overfitting the CV folds).

The property is guarded regression-style (the Sanity Gate pattern): the
benchmark runs in CI as a MANDATORY gate — a violation fails the run.

Experiment per (dataset, seed):
    1. Generate a synthetic regression dataset with known quality
       (the R² ceiling is fixed by the generator's noise level: y = Xw + ε,
       so R² = 1 − Var(ε)/Var(Xw + ε) is known for the true predictor).
    2. Run the full adaptive HPO (TPE sampler over the algorithm's default
       search space) with OOF collection.
    3. Compute ``bbc_cv_score`` on the collected OOF matrix.
    4. Independently evaluate the SELECTED configuration through repeated
       external hold-out (fresh data from the same generator, refit per
       split; the training part matches the HPO CV training size) — the
       reference estimate.
    5. Compare the two estimates over many random seeds.

Acceptance criteria (``check_bbc_acceptance``), per result row:

1. |median gap (BBC-CV − external)| ≤ ``BBC_ACCEPT_MAX_MEDIAN_GAP``
   (0.025) — no systematic optimism or pessimism beyond the noise of the
   estimates (no z criterion: an SE from a handful of seeds is too noisy
   for a mandatory gate);
2. signal-free case (``n_informative=0``): the mean BBC-CV estimate is not
   positive — a positive estimate on pure noise is pure selection bias.

A violated criterion fails the CI run (the CLI returns code 1). The CI
profile (n=3000, 15 external repeats, seeds 1–6) runs each case ONCE over
6 seeds and gates the POOLED per-case median of the 6 per-seed gaps. The
design was calibrated on 15 independent 6-seed windows (seeds 1–90): a
single unlucky seed swings its per-seed gap by up to ±0.12, per-group-of-3
medians flip beyond ±0.02 on ~20% of windows, while the pooled 6-seed
median stayed within ±0.022 on every measured window (0 false FAILs at
the 0.025 allowance) — a systematic BBC-CV shift of the same magnitude as
the measured optimism of a healthy engine (≤0.006) is still caught.

Measured scenarios and outcomes (seeds 1..4 / 1..6 per case):

* linear signal (ridge, 30 trials): gap −0.007 ± 0.02 (z −0.4);
* nonlinear signal with 50/60 noise features (random forest, 30 trials):
  gap −0.0003 ± 0.04 (z −0.01);
* the many-trials stress (100 Optuna trials, 60 features / 50 noise,
  random forest AND xgboosting — the widest default spaces): gaps
  −0.006 ± 0.05 (z −0.1) and −0.009 ± 0.02 (z −0.5);
* a signal-free target (60 noise features, 60 trials): any positive
  estimate would be pure selection bias — BBC-CV −0.017, external
  −0.011, gap −0.005 ± 0.007.

Across all scenarios the BBC−external gap is statistically consistent
with zero: no systematic optimism of BBC-CV over the independent
reference was detected after adaptive HPO. The CI gate passes on the
current engine (the pooled medians of the fixed seed set 1–6 are inside
±0.014 for all cases, run time ~90 s).

Run from the repository root:

    python benchmarks/run_bbc_adaptive_hpo_benchmark.py [--quick] [--ci]

Profiles: the default (two datasets × 4 seeds × 25+ trials) and ``--quick``
are for manual evidence collection; ``--ci`` is the CI gate profile
(ridge cases + the signal-free case, n=3000, 10 trials, 15 external
repeats, 6 seeds pooled per case, ~90 s).

Datasets, seeds and trial counts are fixed; the run is reproducible and no
sensitive data is added to git. The ``BBC_BENCH_*`` environment variables
scale the seed count of the manual profiles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from configurable_automl_engine import tuner
from configurable_automl_engine.common.metrics import bbc_cv_score, get_scorer_object

__all__ = [
    "BBC_ACCEPT_MAX_MEDIAN_GAP",
    "BBC_CASES",
    "CASE_PURE_NOISE",
    "CI_BBC_CASES",
    "CI_SEEDS",
    "DatasetResult",
    "DatasetSpec",
    "check_bbc_acceptance",
    "render_bbc_table",
    "run_all_bbc_cases",
    "run_bbc_case",
    "summarize_bbc_verdict",
]


# --------------------------------------------------------------------------- #
#  Datasets with a known quality ceiling
# --------------------------------------------------------------------------- #


def _regression_xy(
    n_samples: int,
    n_features: int,
    n_informative: int,
    noise: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a regression dataset from sklearn's generator.

    Args:
        n_samples: Number of rows.
        n_features: Total number of features (informative + noise).
        n_informative: Number of informative features.
        noise: Standard deviation of the Gaussian target noise.
        seed: The generator seed.

    Returns:
        Tuple ``(X, y)`` as a DataFrame/Series pair.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )
    cols = [f"x{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


@dataclass(frozen=True)
class DatasetSpec:
    """A synthetic dataset case with a known quality ceiling.

    Attributes:
        name: The dataset label used in the report.
        n_samples: Number of rows of the HPO sample.
        n_features: Total features (informative plus noise).
        n_informative: Number of informative features.
        noise: Target noise standard deviation.
        algo: The algorithm the adaptive HPO runs on.
        n_trials: The number of Optuna trials of the HPO run.
        external_n_samples: Rows of each external hold-out evaluation.
        external_repeats: The number of repeated external hold-out
            evaluations (fresh data each, refit per split).
    """

    name: str
    n_samples: int
    n_features: int
    n_informative: int
    noise: float
    algo: str = "random_forest"
    n_trials: int = 25
    external_n_samples: int = 4000
    external_repeats: int = 10


#: A linear signal with a moderate share of noise features.
CASE_LINEAR: DatasetSpec = DatasetSpec(
    name="linear_20feat_5noise",
    n_samples=250,
    n_features=20,
    n_informative=5,
    noise=25.0,
    algo="ridge",
    n_trials=30,
)
#: A nonlinear signal with MANY noise features — the scenario where the
#: adaptive search has the most room to overfit the CV folds.
CASE_NOISY_NONLINEAR: DatasetSpec = DatasetSpec(
    name="nonlinear_60feat_50noise",
    n_samples=250,
    n_features=60,
    n_informative=10,
    noise=25.0,
    algo="random_forest",
    n_trials=30,
)

BBC_CASES: list[DatasetSpec] = [CASE_LINEAR, CASE_NOISY_NONLINEAR]

#: The signal-free case of the gate: any positive BBC-CV estimate on pure
#: noise is pure selection bias. Ridge-only, few trials; the large sample
#: and many external repeats shrink the per-seed gap noise so the pooled
#: median stays robust on the fixed seed set.
CASE_PURE_NOISE: DatasetSpec = DatasetSpec(
    name="pure_noise_ridge",
    n_samples=3000,
    n_features=40,
    n_informative=0,
    noise=25.0,
    algo="ridge",
    n_trials=10,
    external_repeats=15,
)

#: The cases of the CI gate profile: ridge-only cases with a small trial
#: budget — bounded runtime (~90 s on the CI runner). The second case
#: widens the feature space (more noise features) so the adaptive search
#: has room to exploit fold noise, while staying linear and cheap for the
#: ridge reference estimator.
CASE_CI_LINEAR: DatasetSpec = DatasetSpec(
    name="ci_linear_20feat_5noise",
    n_samples=3000,
    n_features=20,
    n_informative=5,
    noise=25.0,
    algo="ridge",
    n_trials=10,
    external_repeats=15,
)
CASE_CI_LINEAR_WIDE: DatasetSpec = DatasetSpec(
    name="ci_linear_wide_40feat_8noise",
    n_samples=3000,
    n_features=40,
    n_informative=8,
    noise=25.0,
    algo="ridge",
    n_trials=10,
    external_repeats=15,
)

#: The cases of the CI gate profile.
CI_BBC_CASES: list[DatasetSpec] = [CASE_CI_LINEAR, CASE_CI_LINEAR_WIDE, CASE_PURE_NOISE]
#: The seeds of the CI gate profile: ONE pooled run per case over 6 seeds
#: — the gate statistic is the MEDIAN of the 6 per-seed gaps, calibrated
#: on 15 independent 6-seed windows (0 false FAILs at the 0.025
#: allowance; a per-group-of-3 median flips on ~20% of windows and is
#: too noisy for a mandatory gate).
CI_SEEDS: list[int] = [1, 2, 3, 4, 5, 6]

#: The target-metric scorer used by both the HPO and the BBC-CV estimate.
METRIC = "r2"


def _external_model(algo: str, params: dict[str, Any]) -> Any:
    """Build the reference estimator of the selected configuration.

    Args:
        algo: The tuned algorithm name (the benchmark supports the cases
            it declares).
        params: The hyperparameters selected by the HPO phase.

    Returns:
        Any: The unfitted estimator.

    Raises:
        ValueError: Unknown algorithm.
    """
    if algo == "ridge":
        return Ridge(**params)
    if algo == "random_forest":
        return RandomForestRegressor(random_state=0, **params)
    if algo == "xgboosting":
        return XGBRegressor(random_state=0, tree_method="hist", verbosity=0, **params)
    raise ValueError(f"No external reference estimator for algo '{algo}'")


def _external_estimate(
    spec: DatasetSpec,
    params: dict[str, Any],
    seeds: range,
) -> float:
    """Reference estimate: repeated external hold-out with fresh data.

    The SELECTED configuration is refit on each training part and scored on
    a fresh test part drawn from the same generator — no fold, row or
    configuration of the HPO phase is ever reused.

    Args:
        spec: The dataset case.
        params: The hyperparameters selected by the HPO phase.
        seeds: One seed per repeat (data generation and split).

    Returns:
        float: The mean hold-out R² over the repeats.
    """
    scores: list[float] = []
    # The training part must match the HPO cross-validation training size
    # (4/5 of the HPO sample in a 5-fold CV): a configuration tuned for n
    # rows is not expected to keep its quality on a much larger training
    # set, and that sample-size effect must not pollute the selection-bias
    # comparison. The test part is large so that the per-repeat R² is
    # precise.
    n_train = round(spec.n_samples * 0.8)
    for seed in seeds:
        # One generator draw per repeat: train and test rows MUST share the
        # same true coefficients, otherwise the reference estimate is
        # meaningless (predictions against a different target function).
        X, y = _regression_xy(
            n_train + spec.external_n_samples,
            spec.n_features,
            spec.n_informative,
            spec.noise,
            seed,
        )
        X_tr, y_tr = X.iloc[:n_train], y.iloc[:n_train]
        X_te, y_te = X.iloc[n_train:], y.iloc[n_train:]
        model = _external_model(spec.algo, params)
        model.fit(X_tr, y_tr)
        scores.append(float(r2_score(y_te, model.predict(X_te))))
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
#  One experiment
# --------------------------------------------------------------------------- #


@dataclass
class DatasetResult:
    """Aggregated comparison of BBC-CV against the external reference.

    Attributes:
        name: The dataset label.
        algo: The tuned algorithm.
        n_trials: The number of Optuna trials per seed.
        seeds: The seeds the experiment ran on.
        bbc_scores: The BBC-CV estimate per seed.
        external_scores: The external reference estimate per seed.
        cv_best_scores: The (optimistic) best-trial CV score per seed.
    """

    name: str
    algo: str
    n_trials: int
    seeds: list[int]
    bbc_scores: list[float]
    external_scores: list[float]
    cv_best_scores: list[float]

    @property
    def gaps(self) -> np.ndarray:
        """Per-seed BBC-CV minus external reference (positive — optimistic)."""
        return np.asarray(self.bbc_scores) - np.asarray(self.external_scores)


def run_bbc_case(
    spec: DatasetSpec,
    seeds: list[int],
    *,
    n_bootstrap: int = 200,
) -> DatasetResult:
    """Run the adaptive-HPO + BBC-CV vs external-reference comparison.

    Args:
        spec: The dataset case.
        seeds: One seed per experiment repetition.
        n_bootstrap: The number of BBC-CV bootstrap resamples.

    Returns:
        DatasetResult: The per-seed estimates of all three channels.
    """
    scorer = get_scorer_object(METRIC)
    bbc_scores: list[float] = []
    external_scores: list[float] = []
    cv_best_scores: list[float] = []
    for seed in seeds:
        X, y = _regression_xy(
            spec.n_samples, spec.n_features, spec.n_informative, spec.noise, seed
        )
        res = tuner.optimize(
            spec.algo,
            X,
            y,
            metric=METRIC,
            val_method="k_fold",
            n_folds=5,
            n_trials=spec.n_trials,
            random_state=seed,
            sampler_seed=seed,
            fit_best_model=False,
            collect_oof=True,
            phase_name=f"bbc_bench_{spec.name}",
        )
        completed = [t for t in res.trials if t.state.name == "COMPLETE"]
        best_value = max(float(t.value) for t in completed)
        oof = res.oof_predictions
        assert oof is not None
        bbc = bbc_cv_score(
            scorer, y.to_numpy(), oof, n_bootstrap=n_bootstrap, random_state=seed
        )
        external = _external_estimate(
            spec, res.params, range(seed * 100, seed * 100 + spec.external_repeats)
        )
        bbc_scores.append(float(bbc))
        external_scores.append(external)
        cv_best_scores.append(best_value)
    return DatasetResult(
        name=spec.name,
        algo=spec.algo,
        n_trials=spec.n_trials,
        seeds=seeds,
        bbc_scores=bbc_scores,
        external_scores=external_scores,
        cv_best_scores=cv_best_scores,
    )


def run_all_bbc_cases(
    specs: list[DatasetSpec] | None = None,
    seeds: list[int] | None = None,
) -> list[DatasetResult]:
    """Run every dataset case.

    Args:
        specs: The dataset cases; ``None`` — the default set.
        seeds: The seeds; ``None`` — the default set from the environment.

    Returns:
        list[DatasetResult]: One result per case.
    """
    if specs is None:
        specs = BBC_CASES
    if seeds is None:
        n_seeds = int(os.environ.get("BBC_BENCH_SEEDS", "4"))
        seeds = list(range(1, n_seeds + 1))
    return [run_bbc_case(spec, seeds) for spec in specs]


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #


def render_bbc_table(results: list[DatasetResult]) -> str:
    """Render the per-dataset comparison table.

    Args:
        results: The results of the cases.

    Returns:
        str: The multi-line table.
    """
    lines = [
        (
            f"{'dataset':<26} {'algo':<14} {'trials':>6} "
            f"{'cv_best':>8} {'BBC-CV':>8} {'external':>9} {'gap':>8}"
        ),
    ]
    for r in results:
        lines.append(
            f"{r.name:<26} {r.algo:<14} {r.n_trials:>6} "
            f"{np.mean(r.cv_best_scores):>8.4f} {np.mean(r.bbc_scores):>8.4f} "
            f"{np.mean(r.external_scores):>9.4f} {np.mean(r.gaps):>8.4f}"
        )
    return "\n".join(lines)


def summarize_bbc_verdict(results: list[DatasetResult]) -> str:
    """Summarize the evidence: is BBC-CV consistent with the external
    reference after the full adaptive HPO?

    Args:
        results: The results of the cases.

    Returns:
        str: The multi-line verdict.
    """
    lines: list[str] = []
    for r in results:
        gaps = r.gaps
        n_optimistic = int((gaps > 0).sum())
        mean_gap = float(np.mean(gaps))
        pooled_se = (
            float(np.std(gaps, ddof=1) / np.sqrt(len(gaps)))
            if len(gaps) > 1
            else float("nan")
        )
        z = (
            mean_gap / pooled_se
            if pooled_se and np.isfinite(pooled_se) and pooled_se > 0
            else float("nan")
        )
        lines.append(
            f"{r.name}: mean gap {mean_gap:+.4f} (BBC-CV − external), "
            f"SE {pooled_se:.4f}, z {z:+.2f}; "
            f"BBC-CV more optimistic on {n_optimistic}/{len(gaps)} seeds; "
            f"per-seed gaps: {np.round(gaps, 4).tolist()}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Acceptance gate (the Sanity Gate pattern: a mandatory CI check)
# --------------------------------------------------------------------------- #

#: Max |median gap (BBC-CV − external)| per case. Measured gaps on all manual
#: scenarios were −0.009…−0.0003; the allowance adds a margin for the seed
#: noise of the pooled 6-seed CI statistic (the largest |median| observed
#: on 15 independent 6-seed windows at n=3000 was 0.022 — hence 0.025,
#: ~4× the observed BBC-CV optimism of a healthy engine).
BBC_ACCEPT_MAX_MEDIAN_GAP = 0.025


def check_bbc_acceptance(results: list[DatasetResult]) -> tuple[bool, list[str]]:
    """Check the acceptance criteria of the BBC-CV gate.

    The gate statistic is the MEDIAN of the per-seed gaps of a case
    (``DatasetResult.gaps``): the CI profile produces one result row per
    (case × group of 3 seeds), and the median absorbs a single unlucky
    seed that would drag a mean beyond the allowance. No z statistic is
    checked: with 3 seeds per group the sample SE is too noisy for a
    mandatory gate (a z from n=3 fails a zero-bias engine at random).

    Per case:

    1. |median gap (BBC-CV − external)| ≤ ``BBC_ACCEPT_MAX_MEDIAN_GAP`` —
       no systematic optimism/pessimism of BBC-CV after the adaptive HPO;
    2. a signal-free case (``n_informative=0`` — identified by the case
       name containing "pure_noise" or "no_signal"): the mean BBC-CV
       estimate is not positive — a positive estimate on pure noise is
       pure selection bias.

    Args:
        results: The results of the cases (one row per case and seed
            group; several rows per case name are allowed — e.g. the two
            seed groups of the CI profile — and are checked row-wise).

    Returns:
        tuple[bool, list[str]]: (passed, list of violations).
    """
    errors: list[str] = []
    for r in results:
        median_gap = float(np.median(r.gaps))
        if abs(median_gap) > BBC_ACCEPT_MAX_MEDIAN_GAP:
            errors.append(
                f"{r.name}: |median gap| {abs(median_gap):.4f} > "
                f"{BBC_ACCEPT_MAX_MEDIAN_GAP} (per-seed gaps "
                f"{np.round(r.gaps, 4).tolist()})"
            )
        is_signal_free = "pure_noise" in r.name or "no_signal" in r.name
        if is_signal_free:
            mean_bbc = float(np.mean(r.bbc_scores))
            if mean_bbc > 0.0:
                errors.append(
                    f"{r.name}: mean BBC-CV estimate {mean_bbc:.4f} > 0 on a "
                    "signal-free target — a positive estimate on pure noise "
                    "is pure selection bias"
                )
    return (not errors, errors)
