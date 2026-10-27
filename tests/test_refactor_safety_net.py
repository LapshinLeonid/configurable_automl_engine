"""Safety net for the structural refactor of the engine modules.

The tests pin the observable behavior of the package BEFORE the large modules
(``training_engine/component.py``, ``tuner.py``, ``trainer.py``) are split:

1. Golden run — ``train_best_model`` on two small fixed datasets (the BBC-CV
   path and the hold-out path) reproduces the committed snapshot: winner,
   best params, every reported score, ``n_trials_evaluated`` and the saved
   artifact's predictions. The same run in ``processes`` mode yields the
   same winner and params.
2. Pickle format — the ``configurable_automl_engine.*`` globals referenced
   by the pickle of a fitted ``ModelTrainer`` and its instance attribute
   names match the committed snapshot: the saved state is an artifact format
   (``trainer_state.ARTIFACT_FORMAT_VERSION``), so any change of it is
   deliberate. The snapshot was regenerated once, for artifact format 2
   (the grouped trainer state of 0.5.0); ``ModelTrainer.load`` refuses the
   artifacts of other formats.
3. Public API — the documented import paths keep working.
4. Imports — ``import configurable_automl_engine`` stays lightweight, and
   every module of the package imports on its own (no import cycles).

Snapshots live in ``tests/snapshots/``. They were generated on the code
BEFORE the refactor; regenerate them only for an intended behavior change
with ``UPDATE_SNAPSHOTS=1 pytest tests/test_refactor_safety_net.py``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import pickle
import pickletools
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import configurable_automl_engine
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.component import train_best_model

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_SNAPSHOTS = os.environ.get("UPDATE_SNAPSHOTS") == "1"
REL_TOL = 1e-9
PACKAGE = "configurable_automl_engine"


# ──────────────────────────────────────────────────────────────────────────────
#  Snapshot helpers
# ──────────────────────────────────────────────────────────────────────────────


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and containers into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _assert_matches(actual: Any, expected: Any, path: str = "") -> None:
    """Compare JSON-like values; floats with ``rel=REL_TOL``."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a dict, got {actual!r}"
        assert sorted(actual) == sorted(expected), f"{path}: keys differ"
        for key in expected:
            _assert_matches(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {actual!r}"
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_matches(a, e, f"{path}[{i}]")
        return
    if isinstance(expected, float) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (
            f"{path}: expected a number, got {actual!r}"
        )
        if math.isnan(expected):
            assert math.isnan(actual), f"{path}: expected NaN, got {actual!r}"
        else:
            assert actual == pytest.approx(expected, rel=REL_TOL), (
                f"{path}: {actual!r} != {expected!r}"
            )
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _check_snapshot(name: str, actual: Any) -> None:
    """Compare ``actual`` with the committed snapshot (or write it)."""
    path = SNAPSHOT_DIR / f"{name}.json"
    data = _to_jsonable(actual)
    if UPDATE_SNAPSHOTS:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return
    assert path.exists(), f"Missing snapshot {path}; run with UPDATE_SNAPSHOTS=1"
    expected = json.loads(path.read_text())
    _assert_matches(data, expected, name)


# ──────────────────────────────────────────────────────────────────────────────
#  1. Golden run
# ──────────────────────────────────────────────────────────────────────────────


def _numeric_dataset() -> pd.DataFrame:
    """~80 numeric rows: no outer hold-out is reserved → the BBC-CV path."""
    rng = np.random.default_rng(20240501)
    n = 80
    X = rng.normal(size=(n, 4))
    y = 10.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] ** 2
    y = y + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["target"] = np.abs(y) + 0.1
    return df


def _mixed_dataset() -> pd.DataFrame:
    """~300 rows with a categorical column, missing values and a datetime column.

    ``holdout_fraction='auto'`` reserves an outer hold-out on this size → the
    hold-out path.
    """
    rng = np.random.default_rng(20240502)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0.0, 5.0, size=n)
    city = rng.choice(["north", "south", "east", "west"], size=n)
    effect = pd.Series(city).map({"north": 3.0, "south": -1.0, "east": 0.5, "west": 1.5})
    ts = pd.Timestamp("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 720, size=n), unit="D"
    )
    y = 20.0 + 2.5 * x1 + 1.2 * x2 + effect.to_numpy() + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "city": city,
            "when": ts,
            "target": y,
        }
    )
    # Missing values in a numeric and in the categorical column.
    df.loc[rng.choice(n, size=15, replace=False), "x2"] = np.nan
    df.loc[rng.choice(n, size=12, replace=False), "city"] = np.nan
    return df


def _golden_config(
    model_path: Path,
    *,
    oversampling: bool,
    parallel_mode: str = "threads",
    max_workers: int | None = 1,
) -> dict[str, Any]:
    """The golden-run config: 4 algorithms, refine, sanity-gate select."""
    return {
        "general": {
            "comparison_metric": "rmse",
            "path_to_model": str(model_path),
            "random_state": 7,
            "parallel_mode": parallel_mode,
            "max_workers": max_workers,
            "feature_selection": {"mode": "auto"},
            "phases": [
                {"name": "search", "n_trials": 4, "action": "all_algorithms"},
                {"name": "polish", "n_trials": 3, "action": "refine"},
                {
                    "name": "gate",
                    "action": "select",
                    "strategy": "sanity_gate",
                    "output": "top_k",
                    "k": 2,
                },
                {
                    "name": "final",
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        # Oversampling is a global setting of the engine (every algorithm of
        # the run receives it).
        "oversampling": {"enable": oversampling, "multiplier": 1.5},
        "algorithms": {
            "decision_tree": {},
            "ridge": {},
            "isotonic_regression": {},
            "poissonregressor": {},
        },
    }


def _golden_summary(result: dict[str, Any], model_path: Path, frame: pd.DataFrame):
    """Collect the snapshot fields of a golden run."""
    loaded = ModelTrainer.load(model_path)
    preds = np.asarray(loaded.predict(frame), dtype=float).reshape(-1)
    keys = (
        "algorithm",
        "params",
        "score",
        "oof_score",
        "holdout_score",
        "bbc_score",
        "n_trials_evaluated",
    )
    summary = {key: result[key] for key in keys if key in result}
    summary["predictions"] = preds
    return summary


_GOLDEN_CASES = {
    "golden_numeric_bbc": (_numeric_dataset, True),
    "golden_mixed_holdout": (_mixed_dataset, False),
}


@pytest.mark.parametrize("case", sorted(_GOLDEN_CASES))
def test_golden_run_sequential(case: str, tmp_path: Path) -> None:
    """A sequential seeded run reproduces the committed snapshot exactly."""
    make_df, oversampling = _GOLDEN_CASES[case]
    df = make_df()
    model_path = tmp_path / "model.pkl"
    result = train_best_model(
        config=_golden_config(model_path, oversampling=oversampling),
        df=df,
        target="target",
    )
    if case == "golden_numeric_bbc":
        assert "bbc_score" in result and "holdout_score" not in result
    else:
        assert "holdout_score" in result and "bbc_score" not in result
    frame = df.drop(columns=["target"]).iloc[:12]
    _check_snapshot(case, _golden_summary(result, model_path, frame))


def _run_processes_mode(case: str, model_path: str) -> dict[str, Any]:
    """Run the golden case in ``processes`` mode; return the winner and params."""
    make_df, oversampling = _GOLDEN_CASES[case]
    result = train_best_model(
        config=_golden_config(
            Path(model_path),
            oversampling=oversampling,
            parallel_mode="processes",
            max_workers=2,
        ),
        df=make_df(),
        target="target",
    )
    return {"algorithm": result["algorithm"], "params": _to_jsonable(result["params"])}


@pytest.mark.parametrize("case", sorted(_GOLDEN_CASES))
def test_golden_run_processes_mode(case: str, tmp_path: Path) -> None:
    """The same run in ``processes`` mode completes with the same winner/params.

    The run happens in a fresh interpreter: forking worker processes from the
    multi-threaded pytest process (after other runs started thread pools) can
    deadlock regardless of the engine code.
    """
    if UPDATE_SNAPSHOTS:
        pytest.skip("snapshots are written by the sequential run")
    repo_root = Path(__file__).resolve().parent.parent
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from tests.test_refactor_safety_net import _run_processes_mode\n"
        f"out = _run_processes_mode({case!r}, {str(tmp_path / 'model.pkl')!r})\n"
        "print('RESULT ' + json.dumps(out))\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    result = json.loads(line[len("RESULT ") :])
    expected = json.loads((SNAPSHOT_DIR / f"{case}.json").read_text())
    assert result["algorithm"] == expected["algorithm"]
    _assert_matches(result["params"], expected["params"], "params")


# ──────────────────────────────────────────────────────────────────────────────
#  2. Pickle compatibility
# ──────────────────────────────────────────────────────────────────────────────

_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}
_GET_OPS = {"BINGET", "LONG_BINGET", "GET"}
_NON_PUSH_OPS = {"FRAME", "PROTO", "MEMOIZE", "STOP", "PUT", "BINPUT", "LONG_BINPUT"}


def _pickled_globals(obj: Any) -> list[str]:
    """Collect the ``configurable_automl_engine.*`` globals referenced by a pickle."""
    data = pickle.dumps(obj)
    found: set[str] = set()
    pushed: list[Any] = []
    memo: dict[int, Any] = {}
    for op, arg, _pos in pickletools.genops(data):
        name = op.name
        if name in _STRING_OPS:
            pushed.append(arg)
        elif name == "MEMOIZE":
            memo[len(memo)] = pushed[-1] if pushed else None
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            memo[int(arg)] = pushed[-1] if pushed else None
        elif name in _GET_OPS:
            pushed.append(memo.get(int(arg)))
        elif name == "STACK_GLOBAL":
            module, qualname = pushed[-2], pushed[-1]
            found.add(f"{module}.{qualname}")
            pushed.append(None)
        elif name == "GLOBAL":
            module, qualname = str(arg).split(" ", 1)
            found.add(f"{module}.{qualname}")
            pushed.append(None)
        elif name not in _NON_PUSH_OPS:
            pushed.append(None)
    return sorted(g for g in found if g.startswith(f"{PACKAGE}."))


def _pickle_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(11)
    n = 60
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "color": rng.choice(["red", "green", "blue"], size=n),
        }
    )
    X.loc[[3, 17, 41], "b"] = np.nan
    y = pd.Series(3.0 * X["a"] - X["c"] + rng.normal(scale=0.2, size=n), name="y")
    return X, y


def test_pickle_compatibility_snapshot() -> None:
    """Fitted trainers pickle the same package globals and attribute names."""
    X, y = _pickle_frame()
    trainers = {
        # Preprocessing (categorical + numeric with missing values), feature
        # selection and oversampling.
        "ridge_fs_oversampling": ModelTrainer(
            algorithm="ridge",
            hyperparams={"alpha": 1.0},
            metric="rmse",
            random_state=3,
            data_oversampling=True,
            data_oversampling_multiplier=1.5,
            feature_selection_cfg={"mode": "always"},
            validation_strategy="k_fold",
            n_folds=3,
        ),
        # The isotonic data adapter (IsotonicDataTransformer) plus oversampling.
        "isotonic_oversampling": ModelTrainer(
            algorithm="isotonic_regression",
            hyperparams={"feature_index": 0},
            metric="rmse",
            random_state=3,
            data_oversampling=True,
            data_oversampling_multiplier=1.5,
            validation_strategy="k_fold",
            n_folds=3,
        ),
    }
    snapshot: dict[str, Any] = {}
    for name, trainer in trainers.items():
        trainer.fit(X, y)
        snapshot[name] = {
            "globals": _pickled_globals(trainer),
            "attributes": sorted(trainer.__dict__),
        }
        restored = pickle.loads(pickle.dumps(trainer))
        assert sorted(restored.__dict__) == sorted(trainer.__dict__)
    _check_snapshot("pickle_compatibility", snapshot)


# ──────────────────────────────────────────────────────────────────────────────
#  3. Public API
# ──────────────────────────────────────────────────────────────────────────────

PUBLIC_IMPORTS = {
    "configurable_automl_engine": [
        "ModelTrainer",
        "train_best_model",
        "Config",
        "__version__",
    ],
    "configurable_automl_engine.tuner": [
        "optimize",
        "OptimizeResult",
        "HyperoptError",
        "InvalidAlgorithmError",
        "InvalidDataError",
        "HPOHistoryError",
        "WORST_SCORE_THRESHOLD",
        "HPO_WORST_SCORE",
    ],
    "configurable_automl_engine.trainer": [
        "ModelTrainer",
        "train_model",
        "TrainingError",
        "IsotonicDataTransformer",
    ],
}

#: The names that ``training_engine/component.py`` re-exported up to 0.4.x
#: and the modules that define them (0.5.0 removed the re-exports: the
#: component exports only ``execute_phases`` and ``train_best_model``).
PIPELINE_NAMES = {
    "FAMILY_DIVERSITY_MULTIPLIER_DEFAULT": "finalists",
    "FinalistCandidate": "state",
    "PhaseContext": "state",
    "PhaseResult": "state",
    "PreparedData": "state",
    "RobustWinnerResult": "state",
    "execute_phases": "component",
    "is_valid_winner_score": "finalists",
    "load_config": "data_prep",
    "persist_artifact": "artifact",
    "prepare_dataset": "data_prep",
    "select_finalists": "finalists",
    "select_robust_winner": "finalists",
    "select_winner": "finalists",
    "train_best_model": "component",
}


@pytest.mark.parametrize(
    ("module", "name"),
    [(m, n) for m, names in PUBLIC_IMPORTS.items() for n in names]
    + [
        (f"configurable_automl_engine.training_engine.{m}", n)
        for n, m in PIPELINE_NAMES.items()
    ],
)
def test_public_import(module: str, name: str) -> None:
    """Every documented public import path keeps working."""
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module}.{name} is missing"


def test_component_exports_only_the_pipeline_entry_points() -> None:
    """``component`` exports the two entry points; the steps live elsewhere."""
    from configurable_automl_engine.training_engine import component

    assert sorted(component.__all__) == ["execute_phases", "train_best_model"]
    for name, module in PIPELINE_NAMES.items():
        path = f"{PACKAGE}.training_engine.{module}"
        obj = getattr(importlib.import_module(path), name)
        if callable(obj):  # functions and classes are defined there
            assert obj.__module__ == path, name


def test_pickled_classes_stay_defined_in_their_modules() -> None:
    """The classes of the saved trainer state live at their documented paths.

    A pickle references every class by its module path, so the path is part
    of the artifact format: ``ModelTrainer`` in ``trainer``,
    ``IsotonicDataTransformer`` in ``isotonic``, the state groups in ``trainer_state`` and the setting groups
    in ``common.settings``.
    """
    from configurable_automl_engine import isotonic, trainer, trainer_state
    from configurable_automl_engine.common import settings

    assert trainer.ModelTrainer.__module__ == f"{PACKAGE}.trainer"
    assert trainer.IsotonicDataTransformer is isotonic.IsotonicDataTransformer
    assert isotonic.IsotonicDataTransformer.__module__ == f"{PACKAGE}.isotonic"
    for cls in (
        trainer_state.TrainerSettings,
        trainer_state.PreparedFeatures,
        trainer_state.ValidationScores,
        trainer_state.FeatureSelectionResult,
    ):
        assert cls.__module__ == f"{PACKAGE}.trainer_state"
    for cls in (
        settings.EncodingSettings,
        settings.InputPolicies,
        settings.OversamplingSettings,
    ):
        assert cls.__module__ == f"{PACKAGE}.common.settings"


# ──────────────────────────────────────────────────────────────────────────────
#  4. Imports
# ──────────────────────────────────────────────────────────────────────────────


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_package_import_stays_lightweight() -> None:
    """``import configurable_automl_engine`` does not pull heavy/lazy modules."""
    code = (
        "import sys, json\n"
        "import configurable_automl_engine\n"
        "names = ['optuna', 'configurable_automl_engine.tuner',\n"
        "         'configurable_automl_engine.training_engine.component']\n"
        "print(json.dumps([n for n in names if n in sys.modules]))\n"
    )
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


def _package_modules() -> list[str]:
    root = Path(configurable_automl_engine.__file__).parent
    names = [PACKAGE]
    for info in pkgutil.walk_packages([str(root)], prefix=f"{PACKAGE}."):
        names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("module", _package_modules())
def test_module_imports_on_its_own(module: str) -> None:
    """Every module of the package imports first in a fresh interpreter."""
    proc = _run_python(f"import {module}")
    assert proc.returncode == 0, f"import {module} failed:\n{proc.stderr}"
