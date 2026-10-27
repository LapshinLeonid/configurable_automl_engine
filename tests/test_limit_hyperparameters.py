"""Tests of the ``limit_hyperparameters`` flag semantics.

Coverage:
- Unit tests of :func:`build_search_space`: with ``limit=True`` the space
  contains ONLY the user-listed parameters (no ``DEFAULT_SPACES`` entries);
  with ``limit=False`` (default) the user ranges override and are merged into
  ``DEFAULT_SPACES``.
- The flag is threaded from ``AlgoCfg`` through the HPO phase worker to the
  HPO request: the run receives ``limit_hyperparameters=True`` and the
  restricted search space.
- The SVR adaptive epsilon is NOT injected in the limited mode (the space is
  exactly what the user listed).
- Config validation: ``limit_hyperparameters: true`` without a non-empty
  ``hyperparameters`` block is rejected at load (no silent no-op).
- E2e: trial params differ between ``true`` and ``false``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.common.hyperopt_defaults import (
    DEFAULT_SPACES,
    SearchSpaceEntry,
)
from configurable_automl_engine.training_engine.component import (
    PhaseResult,
    train_best_model,
)
from configurable_automl_engine.hpo.search_space import build_search_space
from configurable_automl_engine.training_engine.hpo_phases import _hpo_phase_worker
from configurable_automl_engine.training_engine.config_parser import AlgoCfg
from tests.data_factory import create_mock_df
from tests.hpo_spec_factory import make_phase_spec


# ──────────────────────────────────────────────────────────────────────────────
# Unit: build_search_space(limit=...)
# ──────────────────────────────────────────────────────────────────────────────
def test_prepare_search_space_limit_true_only_user_params():
    """limit=True keeps ONLY the user-listed parameters (no DEFAULT_SPACES)."""
    user = {
        "alpha": SearchSpaceEntry.model_validate([0.1, 1.0]),
    }
    space = build_search_space("ridge", user, limit=True)
    assert set(space) == {"alpha"}
    assert space["alpha"].low == 0.1
    assert space["alpha"].high == 1.0


def test_prepare_search_space_limit_true_remaps_legacy_names():
    """limit=True still remaps legacy names (n_iter → max_iter for ARD)."""
    user = {
        "n_iter": SearchSpaceEntry.model_validate([50, 500, "int", 50]),
    }
    space = build_search_space("ardregression", user, limit=True)
    assert set(space) == {"max_iter"}
    assert "n_iter" not in space
    assert space["max_iter"].low == 50


def test_prepare_search_space_limit_false_merges_into_defaults():
    """limit=False (default): the user ranges override and merge into DEFAULT_SPACES.

    The default ridge space is just ``alpha``, so use elasticnet to prove that
    the DEFAULT_SPACES entries are kept alongside the user ones.
    """
    user = {
        "alpha": SearchSpaceEntry.model_validate([0.1, 1.0]),
    }
    space = build_search_space("elasticnet", user, limit=False)
    assert set(space) == set(DEFAULT_SPACES["elasticnet"]) | {"alpha"}
    assert space["alpha"].low == 0.1  # the user value wins
    assert space["l1_ratio"].low == DEFAULT_SPACES["elasticnet"]["l1_ratio"].low


def test_prepare_search_space_limit_true_without_user_overrides_empty():
    """limit=True without user overrides produces an empty space."""
    assert build_search_space("ridge", None, limit=True) == {}


def test_prepare_search_space_limit_true_svr_no_epsilon_adaptive_hook():
    """limit=True for SVR: only the user-listed params, no epsilon surprise.

    In the limited mode nothing is added beyond the user list — in particular
    no default-space epsilon and no assumption that the tuner will inject the
    adaptive one.
    """
    user = {"C": SearchSpaceEntry.model_validate([0.1, 1.0, "float_log"])}
    space = build_search_space("svr", user, limit=True)
    assert set(space) == {"C"}


# ──────────────────────────────────────────────────────────────────────────────
# Config validation
# ──────────────────────────────────────────────────────────────────────────────
def test_algo_cfg_limit_true_requires_hyperparameters():
    """limit_hyperparameters=true without a non-empty hyperparameters block is rejected."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        AlgoCfg(enable=True, limit_hyperparameters=True)
    assert "limit_hyperparameters=true requires a non-empty 'hyperparameters'" in str(
        exc_info.value
    )


def test_algo_cfg_limit_true_with_hyperparameters_ok():
    """limit_hyperparameters=true with a non-empty hyperparameters block is valid."""
    cfg = AlgoCfg(
        enable=True,
        limit_hyperparameters=True,
        hyperparameters={"alpha": [0.1, 1.0]},
    )
    assert cfg.limit_hyperparameters is True


def test_algo_cfg_limit_false_without_hyperparameters_ok():
    """limit_hyperparameters defaults to false; no hyperparameters is still valid."""
    cfg = AlgoCfg(enable=True)
    assert cfg.limit_hyperparameters is False
    assert cfg.hyperparameters is None


# ──────────────────────────────────────────────────────────────────────────────
# Threading: AlgoCfg → worker → HPO request
# ──────────────────────────────────────────────────────────────────────────────
def test_hpo_phase_worker_passes_limit_and_restricted_space(monkeypatch):
    """The worker passes the limited space and the limit flag to the HPO run."""
    algo_cfg = AlgoCfg(
        enable=True,
        limit_hyperparameters=True,
        hyperparameters={"alpha": [0.1, 1.0]},
    )
    spec = make_phase_spec(metric_sklearn="neg_root_mean_squared_error")

    captured: dict[str, object] = {}

    def fake_run_hpo(*, X, y, request):
        captured["request"] = request
        return PhaseResult(score=-0.9, params={"alpha": 0.5})

    monkeypatch.setattr(
        "configurable_automl_engine.training_engine.hpo_phases._run_hpo",
        fake_run_hpo,
    )

    result = _hpo_phase_worker("ridge", algo_cfg, spec, object())
    assert result.score == -0.9
    request = captured["request"]
    assert request.limit_hyperparameters is True
    assert set(request.space_overrides["ridge"]) == {"alpha"}
    # The limited run builds exactly the listed space.
    assert set(
        build_search_space("ridge", request.space_overrides["ridge"], limit=True)
    ) == {"alpha"}


# ──────────────────────────────────────────────────────────────────────────────
# Tuner: the flag suppresses the SVR adaptive epsilon injection
# ──────────────────────────────────────────────────────────────────────────────
def test_tuner_limit_suppresses_svr_adaptive_epsilon():
    """limit_hyperparameters=True: SVR epsilon is NOT injected beyond the user list.

    Without the flag the tuner injects the adaptive ``epsilon`` (bounds from
    the y spread) when the user did not list it; with the flag the space is
    exactly what the user listed (only ``C`` here).
    """
    from configurable_automl_engine.hpo import runner as hpo_runner

    X = pd.DataFrame(np.random.default_rng(0).standard_normal((40, 3)))
    y = pd.Series(np.arange(40, dtype=float))

    def run(limit: bool) -> list[dict[str, object]]:
        seen: list[dict[str, object]] = []

        def capturing_create_model(algo: str, **params):
            # The tuner always injects the engine seed; SVR does
            # not accept it, and it is not a searched parameter.
            params.pop("random_state", None)
            if params:
                seen.append(dict(params))
            from sklearn.svm import SVR

            return SVR(**params)

        with (
            pytest.MonkeyPatch.context() as mp,
        ):
            mp.setattr(hpo_runner, "create_model", capturing_create_model)
            mp.setattr(
                hpo_runner._OptimizeRunner,
                "assemble_estimator",
                lambda self, model, apply_fs=False, feature_index=None: model,
            )
            from configurable_automl_engine.tuner import optimize

            opt_res = optimize(
                "svr",
                X,
                y,
                n_trials=2,
                metric="rmse",
                validation_strategy="k_fold",
                n_folds=2,
                random_state=42,
                fit_best_model=False,
                space_overrides={"svr": {"C": [0.1, 1.0, "float_log"]}},
                limit_hyperparameters=limit,
            )

        assert opt_res is not None
        return seen

    limited = run(limit=True)
    merged = run(limit=False)
    assert limited, "no SVR trial captured in the limited mode"
    assert merged, "no SVR trial captured in the merged mode"
    # Limited mode: ONLY the user-listed C is proposed.
    assert all(set(p) == {"C"} for p in limited)
    # Merged mode: the adaptive epsilon is injected by the tuner.
    assert all("epsilon" in p for p in merged)
    assert all("C" in p for p in merged)


# ──────────────────────────────────────────────────────────────────────────────
# E2e: trial params differ between true and false
# ──────────────────────────────────────────────────────────────────────────────
def test_e2e_decision_tree_limit_false_has_more_params_than_true(tmp_path: Path):
    """Decision_tree trial params differ between true/false.

    DECISION_TREE's DEFAULT_SPACES has max_depth and min_samples_leaf. With
    ``limit_hyperparameters: true`` and user ``min_samples_leaf`` listed, the
    trials propose ONLY ``min_samples_leaf``; with ``false`` the defaults are
    merged in and trials propose ``min_samples_leaf`` AND ``max_depth``.
    """
    from unittest.mock import patch

    from configurable_automl_engine.hpo import runner as hpo_runner

    df = create_mock_df(rows=60, cols=4, target="target")

    real_create_model = hpo_runner.create_model
    seen: list[dict[str, object]] = []

    def capturing(algo: str, **params):
        if algo == "decision_tree" and params:
            # Skip the engine seed injected by the tuner: it is
            # not a searched parameter.
            seen.append({k: v for k, v in params.items() if k != "random_state"})
        return real_create_model(algo, **params)

    model_path = tmp_path / "model.pkl"
    base_cfg: dict[str, object] = {
        "general": {
            "comparison_metric": "rmse",
            "path_to_model": str(model_path),
            "phases": [
                {"name": "Coarse Search", "n_trials": 4, "action": "all_algorithms"},
                {
                    "name": "Final Select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
    }

    def tree_algos(limit: bool) -> dict[str, object]:
        tree: dict[str, object] = {"enable": True}
        if limit:
            tree["limit_hyperparameters"] = True
        tree["hyperparameters"] = {"min_samples_leaf": [1, 5]}
        return {"decision_tree": tree}

    with patch.object(hpo_runner, "create_model", side_effect=capturing):
        train_best_model(
            config={**base_cfg, "algorithms": tree_algos(limit=True)},
            df=df,
            target="target",
            model_path_override=model_path,
        )
    limited_seen = list(seen)
    seen.clear()

    with patch.object(hpo_runner, "create_model", side_effect=capturing):
        train_best_model(
            config={**base_cfg, "algorithms": tree_algos(limit=False)},
            df=df,
            target="target",
            model_path_override=model_path,
        )
    merged_seen = list(seen)

    assert limited_seen, "no decision_tree trial captured in the limited mode"
    assert merged_seen, "no decision_tree trial captured in the merged mode"
    # Limited mode: ONLY the user-listed parameter is proposed.
    assert all(set(p) == {"min_samples_leaf"} for p in limited_seen)
    # Merged mode: the DEFAULT_SPACES entry (max_depth) is searched too.
    assert all("max_depth" in p for p in merged_seen)
    assert all("min_samples_leaf" in p for p in merged_seen)
    # The key sets differ between the two modes — the no-op defect is gone.
    assert {tuple(sorted(p)) for p in limited_seen} != {
        tuple(sorted(p)) for p in merged_seen
    }
