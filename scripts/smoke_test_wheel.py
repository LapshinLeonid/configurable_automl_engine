"""Smoke-test the wheel installed from the built distribution.

Run from a directory outside the repository so imports resolve to the
installed package. The script checks the distribution metadata and the
version derived from the release tag, then trains a tiny model through the
public API, reloads the saved artifact and validates its predictions. Any
failure exits non-zero, which blocks publication.

Usage::

    python scripts/smoke_test_wheel.py --expected-version v1.2.3 \\
        --forbid-source-root /path/to/repository
"""

from __future__ import annotations

import argparse
import importlib.metadata
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import configurable_automl_engine as caml
from configurable_automl_engine import ModelTrainer

_DISTRIBUTION = "configurable-automl-engine"
_N_PREDICTIONS = 5


def _parse_args() -> argparse.Namespace:
    """Parse the command line used by the release workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        help="release tag the wheel was built from (a leading 'v' is ignored)",
    )
    parser.add_argument(
        "--forbid-source-root",
        help="repository root whose sources must never be the imported package",
    )
    return parser.parse_args()


def _fail(message: str) -> None:
    """Abort the smoke test with a message that names the failed check."""
    raise SystemExit(f"smoke test failed: {message}")


def _check_origin(forbidden_root: str | None) -> None:
    """Reject an import that resolves to the repository sources, not the wheel."""
    package_path = Path(caml.__file__).resolve()
    if forbidden_root is None:
        return
    root = Path(forbidden_root).resolve()
    if package_path.is_relative_to(root):
        _fail(f"imported repository sources at {package_path}, not the installed wheel")


def _check_versions(expected_tag: str | None) -> str:
    """Check package, distribution and release-tag versions agree."""
    installed = importlib.metadata.version(_DISTRIBUTION)
    if installed != caml.__version__:
        _fail(
            f"distribution metadata version {installed!r} differs from "
            f"package __version__ {caml.__version__!r}"
        )
    if expected_tag is not None:
        expected = expected_tag.removeprefix("v").removeprefix("V")
        if caml.__version__ != expected:
            _fail(
                f"wheel version {caml.__version__!r} does not match "
                f"release tag {expected_tag!r}"
            )
    return installed


def _dataset(n_rows: int = 120) -> pd.DataFrame:
    """Build a small deterministic regression dataset."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({f"f{i}": rng.normal(size=n_rows) for i in range(3)})
    frame["target"] = (
        2.0 * frame["f0"] - frame["f1"] + rng.normal(scale=0.05, size=n_rows)
    )
    return frame


def _config(model_path: Path) -> dict[str, Any]:
    """Return a two-trial pipeline that exercises search, selection and export."""
    return {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "train_test_split",
            "holdout_fraction": 0.25,
            "random_state": 0,
            "path_to_model": str(model_path),
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {"ridge": {}},
    }


def _train_and_reload(model_path: Path, df: pd.DataFrame) -> float:
    """Train, export, reload and predict; return the honest hold-out score."""
    result = caml.train_best_model(config=_config(model_path), df=df, target="target")
    if result.get("algorithm") != "ridge":
        _fail(f"unexpected winner {result.get('algorithm')!r}")
    score = float(result["holdout_score"])
    if not np.isfinite(score):
        _fail(f"hold-out score is not finite: {score!r}")

    trainer = ModelTrainer.load(model_path)
    features = df.drop(columns=["target"]).head(_N_PREDICTIONS)
    predictions = np.asarray(trainer.predict(features), dtype=float)
    if predictions.shape != (_N_PREDICTIONS,):
        _fail(f"unexpected prediction shape {predictions.shape}")
    if not np.all(np.isfinite(predictions)):
        _fail("model produced non-finite predictions")
    return score


def main() -> None:
    """Run all smoke checks against the installed package."""
    args = _parse_args()
    _check_origin(args.forbid_source_root)
    version = _check_versions(args.expected_version)

    df = _dataset()
    with tempfile.TemporaryDirectory(prefix="caml-wheel-smoke-") as tmp:
        model_path = Path(tmp) / "model.pkl"
        score = _train_and_reload(model_path, df)

    print(f"smoke test passed: version={version}, winner=ridge, holdout_r2={score:.4f}")


if __name__ == "__main__":
    main()
