"""Core of the Sanity Gate regression benchmark on the target pipeline.

The benchmark compares two engine pipeline variants on the same dataset.
The "before/after" comparison from earlier times is reinterpreted: a config
with the old schema (``refine_winner`` / ``sanity_gate.mode``) is unavailable
and rejected by validation, so the baseline is a modern analogue of the old
winner selection behavior:

* ``baseline``: ``all_algorithms → select(strategy: cv_best, output: winner)``
  — winner by the "raw" CV score without the Sanity Gate audit;
* ``target``: ``all_algorithms → select(sanity_gate, top_k) → refine →
  select(sanity_gate, winner)`` — the target DAG: a pool of finalists with
  audit, retraining (warm-start), repeated audit of the updated models
  (fair cost — finalists are retrained from scratch).

Acceptance criteria:

1. Reference datasets (normal data with an honest winner: noisy R²≈0.3, wide
   with hundreds of features, small N): the ``target`` winner matches the
   ``baseline`` winner, the gate does NOT trigger (0 disqualifications in
   any select step). A finite gap > 1.5 (e.g. an overfit lasso on wide data)
   in the warn mode of circuit D is a soft_warning, not a disqualification:
   the gate is considered not triggered and the winner is chosen by the
   honest ``RMSE_oof``.
2. Degenerate datasets (flat shelf / tanh plateau / overfitting): the CV
   leader is a degenerate model, the gate disqualifies it — the ``target``
   winner MUST differ from the ``baseline`` winner (100% of changes).
   Degenerate circuit D cases (RMSE_full≈0 — deep tree, 1-NN) run in the
   strict ``generalization_gap_mode: reject``: in the warn mode an infinite
   gap is a soft_warning, not a disqualification.

Run architecture:

* Reference datasets run through the engine's end-to-end pipeline
  (``execute_phases``) with real HPO (``n_trials=1``, N ≤ 300) — this checks
  that the audit does not degrade selection on normal data.
* Degenerate datasets use a deterministic replacement of ``_run_hpo``:
  the degenerate candidate receives parameters under which real training on
  100% of the data yields a degenerate model (``alpha=500`` for ElasticNet,
  ``max_depth=20`` for the tree, sigmoid kernel for SVR), and the "raw" CV
  score is better than the honest candidate's. This is the only deterministic
  way to guarantee a "degenerate CV leader": by construction such models do
  NOT win an honest cross-validation, so the end-to-end HPO cannot reproduce
  them reliably (the approach inherits the negative scenarios). The finalists
  are still trained and audited for real.

Run time is limited (CI): 6 datasets, n_trials=1, N ≤ 300 — about 10–20
seconds in total.

Output format — a table "dataset × variant × winner × whether the gate
triggered" (see ``render_table``); the acceptance criteria — ``check_acceptance``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import pandas as pd

import configurable_automl_engine.training_engine.hpo_phases as _hpo_phases
from configurable_automl_engine.training_engine.component import (
    PhaseContext,
    PhaseResult,
    execute_phases,
    prepare_dataset,
)
from configurable_automl_engine.training_engine.config_parser import Config

from tests.data_factory import (
    make_degenerate_flat_shelf,
    make_degenerate_overfit,
    make_degenerate_tanh_plateau,
    make_reference_noisy_r2_03,
    make_reference_small_n,
    make_reference_wide,
)

__all__ = [
    "VariantResult",
    "ReferenceCase",
    "DegenerateCase",
    "REFERENCE_CASES",
    "DEGENERATE_CASES",
    "ALL_CASES",
    "run_case",
    "render_table",
    "check_acceptance",
]

# ──────────────────────────────────────────────────────────────────────────────
#  Result of running pipeline variants
# ──────────────────────────────────────────────────────────────────────────────

#: Name of the first sanity-select phase of the target DAG (finalist pool, audit).
GATE_SELECT_PHASE = "GateSelect"


@dataclass(frozen=True)
class VariantResult:
    """Result of comparing ``baseline`` vs ``target`` on a single dataset.

    Attributes:
        dataset (str): Dataset name.
        kind (str): 'reference' | 'degenerate'.
winner_baseline (str): Winner of the ``cv_best`` pipeline (CV leader).
        winner_target (str): Winner of the target DAG with audit and refine.
        pool (list[str]): Finalist pool of the first sanity-select of the
            target DAG.
        disqualified (dict[str, list[str]]): Sanity Gate disqualifications
            (select phase name → disqualified candidates).
        audit_valid (dict[str, bool]): is_valid of each candidate in the pool
            (audit of the first sanity-select).
        gate_triggered (bool): True — at least one sanity-select of the target
            DAG has DISQUALIFICATIONS. Soft_warnings (including a finite gap
            above the threshold in the warn mode of circuit D) do NOT
            "trigger" the gate.
        changed (bool): True — the variant winners differ.
        elapsed (float): Run time in seconds.
    """

    dataset: str
    kind: str
    winner_baseline: str
    winner_target: str
    pool: list[str]
    disqualified: dict[str, list[str]] = field(default_factory=dict)
    audit_valid: dict[str, bool] = field(default_factory=dict)
    gate_triggered: bool = False
    changed: bool = False
    elapsed: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Case descriptions
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReferenceCase:
    """Reference case: end-to-end run with real HPO.

    Attributes:
        name (str): Dataset name.
        generator (Callable): Generator (X, y) with a fixed seed.
        algorithms (dict[str, dict]): Algorithm configuration for HPO
            (name → extra AlgoCfg fields, e.g. hyperparameters).
        n_trials (int): Number of HPO trials per algorithm (CI time limit).
        top_k_candidates (int): Cap on the finalist pool size.
        corridor_delta (float): Width of the pool corridor.
    """

    name: str
    kind: str = "reference"
    generator: Callable[[], tuple[np.ndarray, np.ndarray]] = field(
        default=lambda: (np.zeros((1, 1)), np.zeros(1))
    )
    algorithms: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_trials: int = 1
    top_k_candidates: int = 3
    corridor_delta: float = 0.3


@dataclass(frozen=True)
class DegenerateCase:
    """Degenerate case: end-to-end run with a deterministic HPO replacement.

    ``stub_hpo`` returns a ``PhaseResult`` for each algorithm name:
    the degenerate candidate receives parameters under which real training of
    the finalist yields a degenerate model, and the "raw" CV score is better
    than the honest candidate's (which would be the "winner" of the old
    behavior by CV score).

    Attributes:
        name (str): Dataset name.
        generator (Callable): Generator (X, y) with a fixed seed.
        algorithms (dict[str, dict]): Algorithm configuration.
        stub_hpo (Callable[[str], PhaseResult]): Deterministic
            "tuner": algorithm name → raw CV score + parameters.
        top_k_candidates (int): Cap on the finalist pool size.
        corridor_delta (float): Width of the pool corridor.
    """

    name: str
    generator: Callable[[], tuple[np.ndarray, np.ndarray]]
    algorithms: dict[str, dict[str, Any]]
    stub_hpo: Callable[[str], PhaseResult]
    kind: str = "degenerate"
    n_trials: int = 1
    top_k_candidates: int = 2
    corridor_delta: float = 0.3


# ──────────────────────────────────────────────────────────────────────────────
#  Benchmark case registry
# ──────────────────────────────────────────────────────────────────────────────

# Reference datasets (end-to-end run): the gate must neither trigger nor
# change the winner. The alpha-narrowing workaround for lasso is NOT needed
# anymore: with the default alpha space on wide data lasso may overfit
# (a finite gap > 1.5) — in the warn mode of circuit D this is only a
# soft_warning, not a disqualification, so the genuinely best model by
# RMSE_oof wins rather than "the one that passed the gate".
_REFERENCE_ALGORITHMS: dict[str, dict[str, Any]] = {
    "ridge": {"enable": True},
    "lasso": {"enable": True},
    "elasticnet": {"enable": True},
}

REFERENCE_CASES: list[ReferenceCase] = [
    ReferenceCase(
        name="reference_noisy_r2_03",
        generator=make_reference_noisy_r2_03,
        algorithms=dict(_REFERENCE_ALGORITHMS),
    ),
    ReferenceCase(
        name="reference_wide_p200",
        generator=make_reference_wide,
        algorithms=dict(_REFERENCE_ALGORITHMS),
    ),
    ReferenceCase(
        name="reference_small_n80",
        generator=make_reference_small_n,
        algorithms=dict(_REFERENCE_ALGORITHMS),
    ),
]

# Degenerate datasets (deterministic HPO replacement): the degenerate
# candidate is the CV leader (as if the old behavior had selected it), the
# honest candidate is second. During the Sanity Gate audit the degenerate
# candidate is disqualified — the target DAG winner must change (100% of
# changes).
#
# The "raw" scores (rmse metric, optimizer semantics — higher is better) are
# set so that both candidates enter the finalist pool (corridor δ=0.3 in
# user semantics: RMSE 0.08 vs 0.09) while the degenerate candidate remains
# the CV leader.
DEGENERATE_CASES: list[DegenerateCase] = [
    DegenerateCase(
        name="degenerate_flat_shelf",
        generator=make_degenerate_flat_shelf,
        algorithms={
            "elasticnet": {"enable": True},
            "ridge": {"enable": True},
        },
        stub_hpo=lambda name: {
            "elasticnet": PhaseResult(score=-0.08, params={"alpha": 500.0}),
            "ridge": PhaseResult(score=-0.09, params={"alpha": 1.0}),
        }[name],
    ),
    DegenerateCase(
        name="degenerate_tanh_plateau",
        generator=make_degenerate_tanh_plateau,
        algorithms={
            "svr": {"enable": True},
            "ridge": {"enable": True},
        },
        stub_hpo=lambda name: {
            # gamma=0.0 — the true tanh plateau: almost constant predictions
            # (R²_oof < 0, worse than the constant model). The historically
            # used gamma=0.001 is now honestly better than the constant
            # baseline (R²_oof ≈ +0.04) and is NOT a degenerate case under
            # the adaptive diversity circuit.
            "svr": PhaseResult(
                score=-0.08,
                params={"kernel": "sigmoid", "gamma": 0.0, "coef0": 0.0, "C": 1.0},
            ),
            "ridge": PhaseResult(score=-0.09, params={"alpha": 1.0}),
        }[name],
    ),
    DegenerateCase(
        name="degenerate_overfit",
        generator=make_degenerate_overfit,
        algorithms={
            "decision_tree": {"enable": True},
            "ridge": {"enable": True},
        },
        stub_hpo=lambda name: {
            "decision_tree": PhaseResult(score=-0.08, params={"max_depth": 20}),
            "ridge": PhaseResult(score=-0.09, params={"alpha": 1.0}),
        }[name],
    ),
]

ALL_CASES: list[ReferenceCase | DegenerateCase] = [
    *REFERENCE_CASES,
    *DEGENERATE_CASES,
]


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline variant configurations
# ──────────────────────────────────────────────────────────────────────────────


def _algo_cfg(extra: dict[str, Any]) -> dict[str, Any]:
    """Normalize an algorithm configuration from a case into an AlgoCfg dict."""
    cfg: dict[str, Any] = {"enable": True}
    cfg.update(extra)
    return cfg


def _phase(name: str, n_trials: int, action: str, **extra: Any) -> dict[str, Any]:
    """Build a pipeline phase."""
    phase: dict[str, Any] = {"name": name, "n_trials": n_trials, "action": action}
    phase.update(extra)
    return phase


def _cfg_dict(
    case: ReferenceCase | DegenerateCase,
    *,
    target_dag: bool,
) -> dict[str, Any]:
    """Build a pipeline variant configuration (baseline or target DAG).

    Args:
        case: Benchmark case.
        target_dag (bool): True — target DAG with audit and refine; False —
            the ``cv_best`` baseline without audit.

    Returns:
        dict[str, Any]: Configuration for ``Config.model_validate``.
    """
    n_trials = case.n_trials
    if target_dag:
        phases = [
            _phase("Search", n_trials, "all_algorithms"),
            _phase(
                GATE_SELECT_PHASE,
                1,
                "select",
                strategy="sanity_gate",
                output="top_k",
                k=case.top_k_candidates,
            ),
            _phase("Refine", n_trials, "refine"),
            _phase("FinalSelect", 1, "select", strategy="sanity_gate", output="winner"),
        ]
    else:
        phases = [
            _phase("Search", n_trials, "all_algorithms"),
            _phase(
                "FinalSelect", 1, "select", strategy="cv_best", output="winner"
            ),
        ]
    general: dict[str, Any] = {
        "comparison_metric": "rmse",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "phases": phases,
        "sanity_gate": {
            "top_k_candidates": case.top_k_candidates,
            "corridor_delta": case.corridor_delta,
            # Degenerate cases (a deep tree with RMSE_full≈0) require the strict
            # reject mode of circuit D: in the warn mode an infinite
            # gap is a soft_warning and does not disqualify, so the degenerate
            # CV leader would not be excluded from the first sanity-select.
            "generalization_gap_mode": (
                "reject" if isinstance(case, DegenerateCase) else "warn"
            ),
        },
    }
    return {
        "general": general,
        "algorithms": {
            name: _algo_cfg(extra) for name, extra in case.algorithms.items()
        },
        "oversampling": {"enable": False},
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Running the variants
# ──────────────────────────────────────────────────────────────────────────────


def _df_from_case(case: ReferenceCase | DegenerateCase) -> pd.DataFrame:
    """Build a DataFrame from the case generator."""
    X, y = case.generator()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def _execute(
    cfg_dict: dict[str, Any],
    df: pd.DataFrame,
    *,
    stub_hpo: Callable[[str], PhaseResult] | None = None,
) -> PhaseContext:
    """Run ``execute_phases`` on a configuration, optionally replacing HPO.

    Args:
        cfg_dict: Pipeline variant configuration.
        df: Dataset with the target column.
        stub_hpo: Deterministic replacement of ``_run_hpo`` (degenerate
            cases); ``None`` — real HPO.

    Returns:
        PhaseContext: Final phase context (winner in ``winner_algo``).
    """
    cfg = Config.model_validate(cfg_dict)
    prepared = prepare_dataset(cfg, df, "target")
    if stub_hpo is None:
        return execute_phases(cfg, prepared)
    with patch.object(
        _hpo_phases,
        "_run_hpo",
        side_effect=lambda **kwargs: stub_hpo(kwargs["algo_name"]),
    ):
        return execute_phases(cfg, prepared)


def _audit_valid_from_section(section: dict[str, Any]) -> dict[str, bool]:
    """is_valid of each candidate from the gate section."""
    return {name: bool(audit["is_valid"]) for name, audit in section["audit"].items()}


def run_case(case: ReferenceCase | DegenerateCase) -> VariantResult:
    """Run a "baseline vs target DAG" case.

    Args:
        case: Case description (``ReferenceCase`` or ``DegenerateCase``).

    Returns:
        VariantResult: Result of comparing the pipeline variants.
    """
    started = time.monotonic()
    df = _df_from_case(case)
    stub: Callable[[str], PhaseResult] | None = (
        case.stub_hpo if isinstance(case, DegenerateCase) else None
    )

    ctx_baseline = _execute(_cfg_dict(case, target_dag=False), df, stub_hpo=stub)
    ctx_target = _execute(_cfg_dict(case, target_dag=True), df, stub_hpo=stub)

    sections = ctx_target.sanity_gate_sections
    gate_select = sections.get(GATE_SELECT_PHASE, {})
    disqualified = {
        name: list(section["disqualified"]) for name, section in sections.items()
    }
    gate_triggered = any(section["disqualified"] for section in sections.values())
    winner_baseline = ctx_baseline.winner_algo
    winner_target = ctx_target.winner_algo
    return VariantResult(
        dataset=case.name,
        kind=case.kind,
        winner_baseline=winner_baseline or "",
        winner_target=winner_target or "",
        pool=list(gate_select.get("pool", [])),
        disqualified=disqualified,
        audit_valid=(
            _audit_valid_from_section(gate_select) if gate_select else {}
        ),
        gate_triggered=gate_triggered,
        changed=winner_baseline != winner_target,
        elapsed=time.monotonic() - started,
    )


def run_all() -> tuple[list[VariantResult], bool, list[str]]:
    """Run all cases and check the acceptance criteria.

    Returns:
        tuple[list[VariantResult], bool, list[str]]: (results, passed, errors).
    """
    results = [run_case(case) for case in ALL_CASES]
    passed, errors = check_acceptance(results)
    return results, passed, errors


# ──────────────────────────────────────────────────────────────────────────────
#  Table and acceptance criteria
# ──────────────────────────────────────────────────────────────────────────────


def render_table(results: list[VariantResult]) -> str:
    """Table "dataset × variant × winner × whether the gate triggered".

    For each dataset there are two rows: ``baseline`` (CV leader without
    audit) and ``target`` (winner of the target DAG with audit and refine,
    gate disqualifications for both select steps).

    Args:
        results: Run results.

    Returns:
        str: Fixed-width text table.
    """
    header = (
        f"{'dataset':<28} {'variant':<9} {'winner':<18} "
        f"{'gate':<9} {'disqualifications':<16}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        gate_b = "no"
        changed_mark = " (chg.)" if r.changed else ""
        gate_t = "yes" if r.gate_triggered else "no"
        disq = "; ".join(
            f"{phase}: {','.join(names)}"
            for phase, names in r.disqualified.items()
            if names
        )
        lines.append(
            f"{r.dataset:<28} {'baseline':<9} {r.winner_baseline:<18} "
            f"{gate_b:<9} {'—':<16}"
        )
        lines.append(
            f"{'':<28} {'target':<9} {r.winner_target + changed_mark:<18} "
            f"{gate_t:<9} {disq:<16}"
        )
    lines.append(sep)
    return "\n".join(lines)


def check_acceptance(results: list[VariantResult]) -> tuple[bool, list[str]]:
    """Check the acceptance criteria.

    Reference datasets: 0 winner changes and the gate does not trigger.
    Degenerate: 100% of changes, the degenerate CV leader is disqualified,
    and the honest winner is valid in the first sanity-select.

    Args:
        results: Run results.

    Returns:
        tuple[bool, list[str]]: (passed, list of violations).
    """
    errors: list[str] = []
    for r in results:
        if r.kind == "reference":
            if r.changed:
                errors.append(
                    f"{r.dataset}: winner changed on reference data "
                    f"({r.winner_baseline} -> {r.winner_target})"
                )
            if r.gate_triggered:
                errors.append(
                    f"{r.dataset}: gate triggered on reference data: "
                    f"{r.disqualified}"
                )
        else:  # degenerate
            if not r.changed:
                errors.append(
                    f"{r.dataset}: winner did NOT change on degenerate "
                    f"data ({r.winner_baseline} == {r.winner_target})"
                )
            if not r.gate_triggered:
                errors.append(
                    f"{r.dataset}: gate did not trigger on degenerate data"
                )
            if r.winner_baseline not in r.disqualified.get(GATE_SELECT_PHASE, []):
                errors.append(
                    f"{r.dataset}: degenerate CV leader {r.winner_baseline} "
                    f"was not disqualified in the first sanity-select"
                )
            if not r.audit_valid.get(r.winner_target, False):
                errors.append(
                    f"{r.dataset}: target winner {r.winner_target} did not "
                    f"pass the audit in the first sanity-select (an honest "
                    f"candidate was expected)"
                )
    return (not errors, errors)
