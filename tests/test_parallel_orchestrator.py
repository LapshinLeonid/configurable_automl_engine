"""The HPO phase in ``processes`` mode keeps going when one algorithm is lost.

One algorithm's worker crashes or times out: the other algorithms keep their
results, and the lost one is listed in ``disqualified_algorithms`` with the
reason. The HPO run is patched in the parent process, so the tests set
``general.start_method: fork`` (the patch then reaches the workers).
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

import pandas as pd
import pytest

from configurable_automl_engine.training_engine.component import train_best_model
from configurable_automl_engine.training_engine.config_parser import Config
from configurable_automl_engine.training_engine.state import PhaseResult

pytestmark = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="parent-process patches reach the workers only with fork",
)

_HPO = "configurable_automl_engine.training_engine.hpo_phases._run_hpo"
_SAVE = "configurable_automl_engine.training_engine.artifact._fit_and_save"


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
            "start_method": "fork",
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


# ── general.start_method ──────────────────────────────────────────────────


def test_start_method_is_validated():
    base = _config("m.pkl")
    assert Config.model_validate(base).general.start_method == "fork"
    base["general"].pop("start_method")
    assert Config.model_validate(base).general.start_method is None

    threads = _config("m.pkl")
    threads["general"]["parallel_mode"] = "threads"
    with pytest.raises(ValueError, match="applies only to parallel_mode='processes'"):
        Config.model_validate(threads)

    bad = _config("m.pkl")
    bad["general"]["start_method"] = "vfork"
    with pytest.raises(ValueError):
        Config.model_validate(bad)


def test_start_method_unavailable_on_the_platform(monkeypatch):
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    with pytest.raises(ValueError, match="not available on this platform"):
        Config.model_validate(_config("m.pkl"))


_UNGUARDED_SCRIPT = textwrap.dedent(
    """
    import os, sys
    import numpy as np, pandas as pd
    from configurable_automl_engine import train_best_model

    # No `if __name__ == "__main__":` guard: everything runs at import time.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    df["target"] = df["a"] + rng.normal(size=40)
    general = {
        "comparison_metric": "rmse",
        "validation_strategy": "train_test_split",
        "parallel_strategy": "algorithms",
        "parallel_mode": "processes",
        "max_workers": 2,
        "path_to_model": sys.argv[1],
        "phases": [
            {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
            {"name": "final", "n_trials": 1, "action": "select",
             "strategy": "cv_best", "output": "winner"},
        ],
    }
    if sys.argv[2]:
        general["start_method"] = sys.argv[2]
    result = train_best_model(
        {"general": general,
         "algorithms": {"ridge": {"enable": True}, "lasso": {"enable": True}},
         "oversampling": {"enable": False}},
        df, target="target",
    )
    print("WINNER", result["algorithm"])
    """
)


def _run_unguarded(tmp_path, start_method: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "unguarded.py"
    script.write_text(_UNGUARDED_SCRIPT)
    return subprocess.run(
        [sys.executable, str(script), str(tmp_path / "m.pkl"), start_method],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )


def test_unguarded_script_works_with_fork_from_the_config(tmp_path):
    proc = _run_unguarded(tmp_path, "fork")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "WINNER" in proc.stdout


@pytest.mark.skipif(
    "forkserver" not in multiprocessing.get_all_start_methods(),
    reason="forkserver is not available",
)
def test_unguarded_script_under_forkserver_names_the_cause(tmp_path):
    proc = _run_unguarded(tmp_path, "")
    assert proc.returncode != 0
    assert 'guard its top-level code with `if __name__ == "__main__":`' in proc.stderr
