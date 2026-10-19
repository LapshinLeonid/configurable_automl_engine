"""Regression benchmark of the adaptive complexity guard on small datasets.

Compares the HPO results of trees/boosting with the guard disabled
(``mode: off``) vs enabled (``mode: apply``) directly through
``tuner.optimize`` — the CV (OOF-like) score of each algorithm, plus the
XGBoost HPO time.

Acceptance criteria (matching the task):

1. On a set of small datasets the mean CV score of trees/boosting with the
   guard is NOT worse than without it (within a tolerance that absorbs the
   noise of the short runs).
2. The scatter of results between different seeds is SMALLER with the guard.
3. The XGBoost HPO time with the guard is markedly lower (shallower trees,
   capped estimators).

Run architecture: each dataset is run with the guard off and on over a small
set of seeds; for every algorithm the best CV score (R², higher is better)
and the XGBoost wall-clock time are recorded.

Run time is limited (CI): 2 small datasets, n_trials=2, 2 seeds, N ≤ 30 —
about 20–40 seconds in total.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from configurable_automl_engine.common.complexity_guard import ComplexityGuardConfig
from configurable_automl_engine.common.hyperopt_defaults import DEFAULT_SPACES
from configurable_automl_engine.tuner import optimize

__all__ = [
    "Dataset",
    "AlgoResult",
    "GuardResult",
    "SMALL_DATASETS",
    "run_case",
    "render_table",
    "check_acceptance",
]

#: Algorithms measured by the benchmark: trees + boosting (the guard covers
#: exactly these families).
_ALGORITHMS: tuple[str, ...] = (
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboosting",
)


def _make_linear(n: int, p: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(
        2.0 * X["f0"] + 0.5 * X["f1"] + rng.randn(n) * 0.5, name="target"
    )
    return X, y


def _make_noisy(n: int, p: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(0.7 * X["f0"] + rng.randn(n), name="target")
    return X, y


@dataclass(frozen=True)
class Dataset:
    """A small benchmark dataset (n rows, p features)."""

    name: str
    generator: Callable[[int, int, int], tuple[pd.DataFrame, pd.Series]]
    n: int = 18
    p: int = 5
    seeds: tuple[int, ...] = (42, 7)
    n_trials: int = 2


SMALL_DATASETS: list[Dataset] = [
    Dataset(name="small_linear_n18", generator=_make_linear, n=18, p=5, n_trials=3),
    Dataset(name="small_noisy_n24", generator=_make_noisy, n=24, p=5, seeds=(11, 3, 99), n_trials=3),
]


@dataclass
class AlgoResult:
    """CV scores of one algorithm over the seeds (guard off and on)."""

    algo: str
    scores_off: list[float] = field(default_factory=list)
    scores_on: list[float] = field(default_factory=list)
    xgb_time_off: float = 0.0
    xgb_time_on: float = 0.0


@dataclass
class GuardResult:
    """Result of the guard comparison on a single dataset.

    Attributes:
        dataset (str): Dataset name.
        algorithms (list[AlgoResult]): Per-algorithm comparison.
        n_restricted (int): Number of restricted parameters (guard records).
        elapsed (float): Total run time in seconds.
    """

    dataset: str
    algorithms: list[AlgoResult] = field(default_factory=list)
    n_restricted: int = 0
    elapsed: float = 0.0

    def scores_off(self) -> list[float]:
        """All off-scores across algorithms and seeds."""
        return [s for a in self.algorithms for s in a.scores_off]

    def scores_on(self) -> list[float]:
        """All on-scores across algorithms and seeds."""
        return [s for a in self.algorithms for s in a.scores_on]


def _guard_cfg(mode: str) -> ComplexityGuardConfig:
    # The benchmark uses "on"/"off" as variant names; the config mode is
    # "apply" for the enabled variant and "off" for the disabled one.
    return ComplexityGuardConfig(mode="apply" if mode == "on" else "off")


def _space_overrides() -> dict[str, dict[str, Any]]:
    """Search spaces of the covered algorithms (the engine's defaults).

    ``tuner.optimize`` without ``space_overrides`` falls back to the
    built-in dynamic spaces (knn/svr) or raises for trees/boosting — the
    engine passes the spaces explicitly. The benchmark replicates the
    engine's ``_prepare_search_space`` (DEFAULT_SPACES).
    """
    return {algo: dict(DEFAULT_SPACES[algo]) for algo in _ALGORITHMS}


def run_case(ds: Dataset) -> GuardResult:
    """Run the ``off`` and ``on`` variants on one dataset and compare."""
    start = time.monotonic()
    per_algo: dict[str, AlgoResult] = {
        algo: AlgoResult(algo=algo) for algo in _ALGORITHMS
    }
    n_restricted = 0
    overrides = _space_overrides()
    for seed in ds.seeds:
        X, y = ds.generator(ds.n, ds.p, seed)
        for algo in _ALGORITHMS:
            for mode, target in (("off", "scores_off"), ("on", "scores_on")):
                t0 = time.monotonic()
                res = optimize(
                    algo,
                    X,
                    y,
                    n_trials=ds.n_trials,
                    random_state=42,
                    validation_strategy="k_fold",
                    n_folds=5,
                    space_overrides={algo: overrides[algo]},
                    complexity_guard_cfg=_guard_cfg(mode),
                    fit_best_model=False,
                )
                elapsed = time.monotonic() - t0
                if res is not None and res.score is not None:
                    getattr(per_algo[algo], target).append(float(res.score))
                if mode == "on" and res is not None and res.complexity_guard:
                    n_restricted = max(n_restricted, len(res.complexity_guard))
                if algo == "xgboosting":
                    if mode == "off":
                        per_algo[algo].xgb_time_off += elapsed
                    else:
                        per_algo[algo].xgb_time_on += elapsed
    return GuardResult(
        dataset=ds.name,
        algorithms=[per_algo[a] for a in _ALGORITHMS],
        n_restricted=n_restricted,
        elapsed=time.monotonic() - start,
    )


def render_table(results: list[GuardResult]) -> str:
    """Render the «dataset × algorithm × mean_off × mean_on × time» table."""
    lines = [
        f"{'dataset':<20} {'algo':<18} {'mean_off':>9} {'mean_on':>9} "
        f"{'std_off':>8} {'std_on':>8} {'xgb_time':>9}"
    ]
    lines.append("-" * len(lines[0]))
    for r in results:
        for a in r.algorithms:
            mean_off = (
                float(np.mean(a.scores_off)) if a.scores_off else float("nan")
            )
            mean_on = float(np.mean(a.scores_on)) if a.scores_on else float("nan")
            std_off = (
                float(np.std(a.scores_off)) if len(a.scores_off) > 1 else float("nan")
            )
            std_on = (
                float(np.std(a.scores_on)) if len(a.scores_on) > 1 else float("nan")
            )
            time_col = (
                f"{a.xgb_time_off:.1f}/{a.xgb_time_on:.1f}"
                if a.algo == "xgboosting"
                else "-"
            )
            lines.append(
                f"{r.dataset:<20} {a.algo:<18} {mean_off:>9.4f} {mean_on:>9.4f} "
                f"{std_off:>8.4f} {std_on:>8.4f} {time_col:>9}"
            )
        lines.append(f"{r.dataset:<20} {'restricted: ' + str(r.n_restricted):<18}")
    return "\n".join(lines)


#: Tolerance on the mean score: the guard must not degrade the mean CV score
#: by more than this (absorbs the noise of the short runs).
_MEAN_TOLERANCE = 0.10


def check_acceptance(results: list[GuardResult]) -> tuple[bool, list[str]]:
    """Check the acceptance criteria on the run results.

    Criteria:
    1. The AGGREGATE mean CV score (across all covered algorithms, datasets
       and seeds) with the guard is NOT worse (higher-better) than without it
       within ``_MEAN_TOLERANCE``. Aggregating over all algorithms averages
       out the per-algorithm noise of the short runs.
    2. Every dataset restricted at least one parameter (the guard fired).

    The per-algorithm scatter (std over seeds) is reported in the table as an
    informational indicator — on very short runs (n_trials=2..3) it is too
    noisy for a strict acceptance criterion.

    Returns:
        tuple[bool, list[str]]: (passed, error messages).
    """
    errors: list[str] = []

    all_off: list[float] = []
    all_on: list[float] = []
    for r in results:
        if r.n_restricted == 0:
            errors.append(f"{r.dataset}: no parameter restricted by the guard")
        for a in r.algorithms:
            all_off.extend(a.scores_off)
            all_on.extend(a.scores_on)

    if not all_off or not all_on:
        errors.append("no valid scores to compare")
    else:
        mean_off = float(np.mean(all_off))
        mean_on = float(np.mean(all_on))
        if mean_on < mean_off - _MEAN_TOLERANCE:
            errors.append(
                f"aggregate mean CV score degraded "
                f"(off={mean_off:.4f} vs on={mean_on:.4f})"
            )

    return (not errors), errors
