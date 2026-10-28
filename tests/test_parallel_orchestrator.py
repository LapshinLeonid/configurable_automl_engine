"""The HPO phase in ``processes`` mode keeps going when one algorithm is lost.

One algorithm's worker crashes or times out: the other algorithms keep their
results, and the lost one is listed in ``disqualified_algorithms`` with the
reason. The HPO run is patched in the parent process, so the tests pin the
``fork`` start method (the patch then reaches the workers).
"""

from __future__ import annotations

import multiprocessing
import os
import time
from unittest.mock import patch

import pandas as pd
import pytest

from configurable_automl_engine.training_engine import thread_pool
from configurable_automl_engine.training_engine.component import train_best_model
from configurable_automl_engine.training_engine.state import PhaseResult

pytestmark = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="parent-process patches reach the workers only with fork",
)

_HPO = "configurable_automl_engine.training_engine.hpo_phases._run_hpo"
_SAVE = "configurable_automl_engine.training_engine.artifact._fit_and_save"


@pytest.fixture(autouse=True)
def _fork_workers(monkeypatch):
    monkeypatch.setattr(thread_pool, "DEFAULT_START_METHOD", "fork")


def _config(model_path: str, **general) -> dict:
    return {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "train_test_split",
            "parallel_strategy": "algorithms",
            "parallel_mode": "processes",
            "max_workers": 2,
            "phases": [
                {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "path_to_model": model_path,
            **general,
        },
        "algorithms": {
            "random_forest": {"enable": True},
            "elasticnet": {"enable": True},
            "ridge": {"enable": True},
        },
        "oversampling": {"enable": False},
    }


_SCORES = {"elasticnet": -0.8, "ridge": -0.6}


def _crash_random_forest(**kwargs):
    if kwargs["request"].algo_name == "random_forest":
        os._exit(1)
    return PhaseResult(score=_SCORES[kwargs["request"].algo_name], params={"alpha": 1})


def _hang_random_forest(**kwargs):
    if kwargs["request"].algo_name == "random_forest":
        time.sleep(30)
    return PhaseResult(score=_SCORES[kwargs["request"].algo_name], params={"alpha": 1})


_DF = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "target": [0, 1, 0, 1, 0, 1]})


def test_crashed_algorithm_is_disqualified_with_the_reason(tmp_path):
    with patch(_HPO, side_effect=_crash_random_forest), patch(_SAVE):
        result = train_best_model(
            config=_config(str(tmp_path / "m.pkl")), df=_DF, target="target"
        )
    assert result["algorithm"] == "ridge"
    assert result["score"] == 0.6
    assert result["disqualified_algorithms"] == {
        "random_forest": "worker process crashed (exit code 1)"
    }


def test_timed_out_algorithm_is_disqualified_with_the_reason(tmp_path):
    started = time.time()
    with patch(_HPO, side_effect=_hang_random_forest), patch(_SAVE):
        result = train_best_model(
            config=_config(str(tmp_path / "m.pkl"), task_timeout=1),
            df=_DF,
            target="target",
        )
    assert time.time() - started < 20
    assert result["algorithm"] == "ridge"
    assert result["disqualified_algorithms"] == {
        "random_forest": "timed out after 1 s"
    }
