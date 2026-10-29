"""Safety net for the structural refactor of the engine modules.

The tests pin properties of the package's behavior that do not depend on the
installed library versions (no stored expected outputs):

1. Parallel mode — ``train_best_model`` on two small fixed datasets (the
   BBC-CV path and the hold-out path) gives the same winner, parameters,
   scores and predictions in sequential and in ``processes`` mode.
2. Saved models — a fitted ``ModelTrainer`` saved and loaded in the same
   process predicts identically; the classes of the saved state live at
   their documented module paths (the pickle records them). An artifact of
   an older format is refused with the "retrain the model" error
   (``tests/test_artifact_metadata.py``).
3. Public API — the documented import paths keep working.
4. Imports — ``import configurable_automl_engine`` stays lightweight, and
   every module of the package imports on its own (no import cycles).

That a change keeps the results exactly is checked by
``scripts/compare_with_base.py``: it runs the same scenarios on the base
commit and on the current code, in the same environment, and compares the
outputs (nothing is stored, so library versions do not matter).
"""

from __future__ import annotations

import importlib
import json
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

PACKAGE = "configurable_automl_engine"


# ──────────────────────────────────────────────────────────────────────────────
#  1. Sequential vs processes mode
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


def _pipeline_config(
    model_path: Path,
    *,
    oversampling: bool,
    parallel_mode: str = "threads",
    max_workers: int | None = 1,
) -> dict[str, Any]:
    """The run config: 4 algorithms, refine, sanity-gate select."""
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


_CASES = {
    "numeric_bbc": (_numeric_dataset, True),
    "mixed_holdout": (_mixed_dataset, False),
}

_SUMMARY_KEYS = (
    "algorithm",
    "params",
    "score",
    "oof_score",
    "holdout_score",
    "bbc_score",
    "n_trials_evaluated",
    "disqualified_algorithms",
)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and containers into JSON-compatible values.

    Floats pass through ``json`` unchanged (``repr`` round-trips), so the
    comparison of the two modes stays exact.
    """
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


def _run_case(case: str, model_path: str, parallel_mode: str) -> dict[str, Any]:
    """Run one case; return the reported fields and the saved model's predictions."""
    make_df, oversampling = _CASES[case]
    df = make_df()
    result = train_best_model(
        config=_pipeline_config(
            Path(model_path),
            oversampling=oversampling,
            parallel_mode=parallel_mode,
            max_workers=2 if parallel_mode == "processes" else 1,
        ),
        df=df,
        target="target",
    )
    summary = {key: result[key] for key in _SUMMARY_KEYS if key in result}
    frame = df.drop(columns=["target"]).iloc[:12]
    loaded = ModelTrainer.load(model_path)
    summary["predictions"] = np.asarray(loaded.predict(frame), dtype=float)
    return _to_jsonable(summary)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_sequential_and_processes_mode_agree(case: str, tmp_path: Path) -> None:
    """Sequential and ``processes`` mode give the same winner, scores, predictions.

    Each run happens in a fresh interpreter: forking worker processes from
    the multi-threaded pytest process (after other runs started thread
    pools) can deadlock regardless of the engine code.
    """
    repo_root = Path(__file__).resolve().parent.parent
    results = {}
    for mode in ("threads", "processes"):
        code = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(repo_root)!r})\n"
            "from tests.test_refactor_safety_net import _run_case\n"
            f"out = _run_case({case!r}, {str(tmp_path / f'{mode}.pkl')!r}, "
            f"{mode!r})\n"
            "print('RESULT ' + json.dumps(out))\n"
        )
        proc = _run_python(code)
        assert proc.returncode == 0, proc.stderr[-4000:]
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        results[mode] = line[-1][len("RESULT ") :]
    sequential = json.loads(results["threads"])
    if case == "numeric_bbc":
        assert "bbc_score" in sequential and "holdout_score" not in sequential
    else:
        assert "holdout_score" in sequential and "bbc_score" not in sequential
    # Compared as JSON text: exact floats, and NaN equals NaN.
    assert results["threads"] == results["processes"]


# ──────────────────────────────────────────────────────────────────────────────
#  2. Saved models
# ──────────────────────────────────────────────────────────────────────────────


def _trainer_frame() -> tuple[pd.DataFrame, pd.Series]:
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


_ROUND_TRIP_TRAINERS: dict[str, dict[str, Any]] = {
    # Preprocessing (categorical + numeric with missing values), feature
    # selection and oversampling.
    "ridge_fs_oversampling": {
        "algorithm": "ridge",
        "hyperparams": {"alpha": 1.0},
        "data_oversampling": True,
        "data_oversampling_multiplier": 1.5,
        "feature_selection_cfg": {"mode": "always"},
    },
    # The isotonic data adapter (IsotonicDataTransformer) plus oversampling.
    "isotonic_oversampling": {
        "algorithm": "isotonic_regression",
        "hyperparams": {"feature_index": 0},
        "data_oversampling": True,
        "data_oversampling_multiplier": 1.5,
    },
    # The categorical encoders of the preprocessing.
    "target_encoding": {
        "algorithm": "decision_tree",
        "hyperparams": {"max_depth": 3},
        "encoding_strategy": "target",
    },
    "frequency_hashing": {
        "algorithm": "ridge",
        "hyperparams": {"alpha": 0.5},
        "encoding_strategy": "frequency",
        "high_cardinality_threshold": 2,
        "high_cardinality_encoding": "hashing",
        "hashing_n_components": 4,
    },
    # SMOTE with noise (DataOversampler with a non-default algorithm).
    "svr_smote_noise": {
        "algorithm": "svr",
        "hyperparams": {"C": 2.0},
        "data_oversampling": True,
        "data_oversampling_multiplier": 2.0,
        "data_oversampling_algorithm": "smote",
        "data_oversampling_add_noise": True,
    },
}


@pytest.mark.parametrize("name", sorted(_ROUND_TRIP_TRAINERS))
def test_save_load_round_trip_predicts_identically(name: str, tmp_path: Path) -> None:
    """A fitted trainer saved and loaded in the same process predicts identically."""
    X, y = _trainer_frame()
    trainer = ModelTrainer(
        metric="rmse",
        random_state=3,
        validation_strategy="k_fold",
        n_folds=3,
        **_ROUND_TRIP_TRAINERS[name],
    )
    trainer.fit(X, y)
    path = tmp_path / "model.pkl"
    trainer.save(path)
    restored = ModelTrainer.load(path)
    assert sorted(restored.__dict__) == sorted(trainer.__dict__)
    expected = np.asarray(trainer.predict(X), dtype=float)
    actual = np.asarray(restored.predict(X), dtype=float)
    np.testing.assert_array_equal(actual, expected)


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
