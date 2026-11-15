"""Smoke-test an installed contract in an environment without the engine."""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.metadata import metadata, requires
from importlib.resources import files

import automl_contracts

FORBIDDEN_MODULES = (
    "configurable_automl_engine",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "optuna",
    "yaml",
    "joblib",
    "imblearn",
    "pyarrow",
    "mlflow",
    "shap",
    "xgboost",
)


def main() -> int:
    """Fail when the installed contract needs the engine or heavy packages."""
    loaded = [name for name in FORBIDDEN_MODULES if name in sys.modules]
    if loaded:
        raise SystemExit(f"contract import loaded forbidden modules: {loaded}")
    installed = [
        name for name in FORBIDDEN_MODULES if importlib.util.find_spec(name) is not None
    ]
    if installed:
        raise SystemExit(
            f"isolated environment contains forbidden modules: {installed}"
        )

    package = files("automl_contracts")
    for relative in ("py.typed", "schemas/config.schema.json"):
        if not package.joinpath(relative).is_file():
            raise SystemExit(f"installed contract is missing {relative}")

    meta = metadata("automl-contracts")
    requirements = requires("automl-contracts") or []
    allowed = {"pydantic", "pydantic-core"}
    for requirement in requirements:
        name = requirement.split(";", 1)[0]
        name = name.split("[", 1)[0]
        name = name.split("<", 1)[0]
        name = name.split(">", 1)[0]
        name = name.split("=", 1)[0]
        name = name.split("!", 1)[0]
        name = name.split("~", 1)[0].strip().lower().replace("_", "-")
        if name not in allowed:
            raise SystemExit(f"contract requires unexpected package {name!r}")
    if not automl_contracts.__all__:
        raise SystemExit("contract exports no public names")
    if not meta["Version"]:
        raise SystemExit("contract has no version")
    print(
        json.dumps(
            {
                "distribution": meta["Name"],
                "version": meta["Version"],
                "requirements": requirements,
                "public_names": len(automl_contracts.__all__),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
