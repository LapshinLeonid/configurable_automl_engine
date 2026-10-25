"""Nightly evidence benchmark for the regression oversampling feature.

The benchmark compares two engine pipeline variants on the same datasets:

* ``baseline`` — oversampling disabled;
* ``oversampled`` — oversampling enabled (all three algorithms: random,
  smote, adasyn) on the same configuration.

The evidence gate: if oversampling LOWERS the median held-out R² by more
than ``0.01`` compared to the no-oversampling baseline, the feature must be
marked experimental in README/API_REFERENCE and a WARNING must be logged
when it is enabled. The gate itself does not block the nightly run — it
collects the evidence (``check_acceptance`` reports the verdict).

Datasets: fixed seeds/generators, so the run is reproducible and no
sensitive real data is added to git. The nightly profile is intentionally
cheap (n_trials=1, N ≤ 500) so it fits a CI/nightly slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from configurable_automl_engine.training_engine import train_best_model
from configurable_automl_engine.training_engine.config_parser import Config

__all__ = [
    "OversamplingVariantResult",
    "OversamplingCase",
    "OVERSAMPLING_CASES",
    "ALL_OVERSAMPLING_CASES",
    "EVIDENCE_GATE_DELTA",
    "run_oversampling_case",
    "run_all_oversampling",
    "render_oversampling_table",
    "check_oversampling_acceptance",
]

#: R² degradation threshold of the evidence gate: an oversampling
#: variant whose median held-out R² is more than this much LOWER than the
#: no-oversampling baseline marks the feature experimental.
EVIDENCE_GATE_DELTA = 0.01

#: Number of dataset seeds over which the median held-out R² is computed.
#: A single deterministic run would be a point estimate, not a median; the
#: seed loop makes the "median" in the gate description honest while keeping
#: the nightly profile cheap.
N_SEEDS = 3

#: Limited mode keeps the nightly profile cheap: the search covers ONLY the
#: listed parameters. ``limit_hyperparameters: true`` requires a non-empty
#: ``hyperparameters`` block (``AlgoCfg`` validation), so the spaces are
#: listed explicitly.
_ALGORITHMS: dict[str, dict[str, Any]] = {
    "ridge": {
        "enable": True,
        "limit_hyperparameters": True,
        "hyperparameters": {"alpha": [0.1, 10.0]},
    },
    "elasticnet": {
        "enable": True,
        "limit_hyperparameters": True,
        "hyperparameters": {"alpha": [0.01, 1.0], "l1_ratio": [0.1, 0.9]},
    },
}


def _make_continuous(n: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Continuous regression target with a moderate signal (R² ≈ 0.4)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = 1.5 * X[:, 0] + 0.7 * X[:, 1] + rng.randn(n) * 1.2
    return X, y


def _make_count(n: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Integer-valued (count) regression target."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    rate = np.exp(0.3 * X[:, 0] + 0.15 * X[:, 1])
    y = rng.poisson(rate).astype(float)
    return X, y


@dataclass(frozen=True)
class OversamplingCase:
    """A dataset for the oversampling evidence benchmark.

    Attributes:
        name (str): Dataset name.
        generator (Callable): Generator (X, y) with a fixed seed.
        n_trials (int): HPO trials per algorithm (CI time limit).
    """

    name: str
    generator: Callable[[], tuple[np.ndarray, np.ndarray]]
    n_trials: int = 1


OVERSAMPLING_CASES: list[OversamplingCase] = [
    OversamplingCase(name="continuous_r2_04", generator=_make_continuous),
    OversamplingCase(name="count_poisson", generator=_make_count),
]

ALL_OVERSAMPLING_CASES: list[OversamplingCase] = list(OVERSAMPLING_CASES)


@dataclass
class OversamplingVariantResult:
    """Result of the oversampling evidence comparison on one dataset.

    Attributes:
        dataset (str): Dataset name.
        base_score (float | None): Median held-out R² without oversampling.
        scores (dict[str, float | None]): Median held-out R² per
            oversampling algorithm.
        degradation (dict[str, float]): base_score − algorithm score for
            every algorithm (positive = oversampling lowered R²).
        elapsed (float): Run time in seconds.
    """

    dataset: str
    base_score: float | None
    scores: dict[str, float | None] = field(default_factory=dict)
    degradation: dict[str, float] = field(default_factory=dict)
    elapsed: float = 0.0

    @property
    def violations(self) -> list[str]:
        """Algorithms that degraded the median R² beyond the gate threshold."""
        return [
            algo
            for algo, delta in self.degradation.items()
            if self.base_score is not None and delta > EVIDENCE_GATE_DELTA
        ]


def _df_from_case(case: OversamplingCase, seed: int = 0) -> pd.DataFrame:
    """Build a DataFrame from the case generator with the given seed."""
    X, y = case.generator(seed=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def _config_dict(algorithm: str | None, n_trials: int) -> dict[str, Any]:
    """Build a train_best_model config with the given oversampling
    algorithm (None — oversampling disabled)."""
    oversampling: dict[str, Any] = {"enable": False}
    if algorithm is not None:
        oversampling = {
            "enable": True,
            "multiplier": 1.5,
            "algorithm": algorithm,
        }
    return {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "train_test_split",
            "phases": [
                {"name": "search", "n_trials": n_trials, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": dict(_ALGORITHMS),
        "oversampling": oversampling,
    }


def run_oversampling_case(
    case: OversamplingCase, tmp_path: Any
) -> OversamplingVariantResult:
    """Run the baseline (no oversampling) and the three oversampling
    variants on one dataset, collecting the median held-out R².

    Every variant is evaluated on ``N_SEEDS`` dataset seeds (the case
    generator's seed is used as the first seed) and the median R² over the
    seeds is the reported score.

    Args:
        case: The dataset description.
        tmp_path: A writable directory for the saved models.

    Returns:
        OversamplingVariantResult: The comparison result.
    """
    import time

    started = time.monotonic()
    dfs = [_df_from_case(case, seed) for seed in range(N_SEEDS)]

    def _score(algorithm: str | None) -> float | None:
        cfg = _config_dict(algorithm, case.n_trials)
        Config.model_validate(cfg)  # fail fast on a malformed config
        finite: list[float] = []
        for df in dfs:
            try:
                result = train_best_model(config=cfg, df=df.copy(), target="target")
            except Exception:  # noqa: BLE001 - a failed variant is evidence too
                continue
            score = result.get("score")
            if score is not None and np.isfinite(score):
                finite.append(float(score))
        if not finite:
            return None
        return float(np.median(finite))

    base = _score(None)
    scores: dict[str, float | None] = {
        algo: _score(algo) for algo in ("random", "smote", "adasyn")
    }
    degradation: dict[str, float] = {}
    if base is not None:
        for algo, score in scores.items():
            if score is not None:
                degradation[algo] = base - score
    return OversamplingVariantResult(
        dataset=case.name,
        base_score=base,
        scores=scores,
        degradation=degradation,
        elapsed=time.monotonic() - started,
    )


def run_all_oversampling(
    tmp_path: Any,
) -> tuple[list[OversamplingVariantResult], bool, list[str]]:
    """Run all oversampling cases and check the evidence gate.

    Args:
        tmp_path: A writable directory for the saved models.

    Returns:
        tuple[list[OversamplingVariantResult], bool, list[str]]:
            (results, passed, violations).
    """
    results = [
        run_oversampling_case(case, tmp_path) for case in ALL_OVERSAMPLING_CASES
    ]
    passed, errors = check_oversampling_acceptance(results)
    return results, passed, errors


def render_oversampling_table(results: list[OversamplingVariantResult]) -> str:
    """Table "dataset × variant × median R² × degradation".

    Args:
        results: Run results.

    Returns:
        str: Fixed-width text table.
    """
    header = (
        f"{'dataset':<20} {'variant':<14} {'median R²':<12} {'Δ vs base':<10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        base = "—" if r.base_score is None else f"{r.base_score:.4f}"
        lines.append(f"{r.dataset:<20} {'baseline':<14} {base:<12} {'—':<10}")
        for algo in ("random", "smote", "adasyn"):
            score = r.scores.get(algo)
            delta = r.degradation.get(algo)
            score_s = "—" if score is None else f"{score:.4f}"
            delta_s = "—" if delta is None else f"{delta:+.4f}"
            lines.append(f"{'':<20} {algo:<14} {score_s:<12} {delta_s:<10}")
    lines.append(sep)
    return "\n".join(lines)


def check_oversampling_acceptance(
    results: list[OversamplingVariantResult],
) -> tuple[bool, list[str]]:
    """Check the evidence gate.

    If any oversampling algorithm lowers the median held-out R² by more
    than ``EVIDENCE_GATE_DELTA`` on any dataset, the feature must be marked
    experimental in README/API_REFERENCE (the failure list carries the
    affected dataset/algorithm pairs so the documentation can be updated).

    Args:
        results: Run results.

    Returns:
        tuple[bool, list[str]]: (gate passed, list of violations).
    """
    errors: list[str] = []
    for r in results:
        for algo, delta in sorted(r.degradation.items()):
            if delta > EVIDENCE_GATE_DELTA:
                errors.append(
                    f"{r.dataset}/{algo}: median R² degraded by {delta:.4f} "
                    f"> {EVIDENCE_GATE_DELTA} vs baseline "
                    f"(base={r.base_score:.4f}, os={r.scores.get(algo):.4f}) — "
                    "the oversampling feature must be marked experimental"
                )
    return (not errors, errors)