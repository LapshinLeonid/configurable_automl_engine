"""Build the stable public-API snapshot of ``automl_contracts``.

The snapshot is the golden API surface of the contract: the exact module
exports, the kind and canonical module of every public symbol, the fields and
configuration of every Pydantic model, enum members, Protocol signatures,
function signatures and constant values.

Only the standard library, Pydantic and the contract itself are imported, so
the snapshot can be rebuilt in the contract-only environment.

The golden file is intentionally versioned together with ``automl-contracts``
(``contract_version`` and ``changelog_version``).  A public-API change therefore
requires an explicit golden update, a ``CHANGELOG.md`` entry and a version bump.
"""

from __future__ import annotations

import ast
import enum
import hashlib
import importlib
import inspect
import json
import re
import types
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = REPO_ROOT / "automl-contracts"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
PUBLIC_API_GOLDEN = GOLDEN_DIR / "public_api.json"
RELEASE_HISTORY = GOLDEN_DIR / "releases.json"

CONTRACT_PYPROJECT = CONTRACT_ROOT / "pyproject.toml"
CONTRACT_CHANGELOG = CONTRACT_ROOT / "CHANGELOG.md"
API_REFERENCE_DOC = REPO_ROOT / "API_REFERENCE.md"

#: Modules whose contents are part of the documented public import surface.
#: The root package is represented by ``__all__``.
PUBLIC_SUBMODULES = (
    "automl_contracts.definitions",
    "automl_contracts.errors",
    "automl_contracts.metrics",
    "automl_contracts.custom_metrics",
    "automl_contracts.protocols",
    "automl_contracts.search_space",
    "automl_contracts.feature_selection",
    "automl_contracts.preprocessing",
    "automl_contracts.result",
    "automl_contracts.config",
    "automl_contracts.config.general",
    "automl_contracts.config.phases",
    "automl_contracts.config.sanity_gate",
    "automl_contracts.config.complexity_guard",
    "automl_contracts.config.ensemble",
    "automl_contracts.config.algorithms",
)

#: Public symbols of the submodules that do not declare ``__all__``.  The
#: lists mirror the documented public surface instead of everything imported
#: into the module namespace.
DOCUMENTED_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "automl_contracts.errors": ("TrainingError", "InvalidPredictionError"),
    "automl_contracts.search_space": (
        "SearchSpaceEntry",
        "BaseDistribution",
        "CategoricalSpace",
        "NumericSpace",
        "FloatSpace",
        "IntSpace",
    ),
    "automl_contracts.feature_selection": (
        "FeatureSelectionMode",
        "FeatureSelectionMethod",
        "FeatureSelectionCfg",
    ),
    "automl_contracts.config.general": (
        "GeneralCfg",
        "ComparisonMetric",
        "AUTO_HOLDOUT_FRACTION",
        "AUTO_HOLDOUT_MIN_ROWS",
        "InfPolicy",
        "DatetimePolicy",
        "SplitAggregation",
    ),
    "automl_contracts.config.phases": (
        "HPOPhaseAction",
        "SelectStrategy",
        "SelectOutput",
        "PruningCfg",
        "PruningStrategy",
        "HPOPhaseCfg",
        "DEFAULT_PHASE_N_TRIALS",
        "ERR_REMOVED_REFINE_WINNER",
    ),
    "automl_contracts.config.sanity_gate": (
        "SanityGateCfg",
        "CorridorMode",
        "GeneralizationGapMode",
        "ERR_REMOVED_SANITY_GATE_MODE",
    ),
    "automl_contracts.config.complexity_guard": (
        "ComplexityGuardCfg",
        "ComplexityGuardMode",
        "ComplexityGuardUserRangePolicy",
        "ComplexityGuardSizeBasis",
    ),
    "automl_contracts.config.ensemble": (
        "EnsembleCfg",
        "EnsembleWeightMethod",
    ),
}

_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")
_API_VERSION_PATTERN = re.compile(r"^## \[([^\]]+)\]", re.MULTILINE)

#: Version of the snapshot representation itself. Bump it when the shape of
#: the golden payload changes without changing the contract API; release
#: history entries with another schema version may be re-pinned.
SNAPSHOT_SCHEMA_VERSION = 2


def read_contract_version() -> str:
    """Return the static version from ``automl-contracts/pyproject.toml``."""
    import tomllib

    data = tomllib.loads(CONTRACT_PYPROJECT.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def read_changelog_latest_version() -> str:
    """Return the newest released version from the contract changelog."""
    text = CONTRACT_CHANGELOG.read_text(encoding="utf-8")
    for match in _API_VERSION_PATTERN.finditer(text):
        version = match.group(1)
        if version.lower() != "unreleased":
            return version
    raise AssertionError("automl-contracts/CHANGELOG.md has no released version")


def module_exports(module_name: str) -> list[str]:
    """Return the documented public exports of one contract module."""
    module = importlib.import_module(module_name)
    if hasattr(module, "__all__"):
        return list(module.__all__)
    if module_name in DOCUMENTED_SUBMODULE_EXPORTS:
        return list(DOCUMENTED_SUBMODULE_EXPORTS[module_name])
    defined = [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module_name
    ]
    return sorted(defined)


def _format_metadata(metadata: Any) -> str:
    """Format a Pydantic/typing annotation metadata object deterministically."""
    func = getattr(metadata, "func", None)
    if callable(func):
        module = getattr(func, "__module__", "")
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", "?"))
        return f"{type(metadata).__name__}({module}.{qualname})"
    discriminator = getattr(metadata, "discriminator", None)
    if discriminator is not None:
        name = getattr(discriminator, "__qualname__", repr(discriminator))
        return f"{type(metadata).__name__}({name})"
    return _ADDRESS_RE.sub("", repr(metadata))


def format_annotation(annotation: Any) -> str:
    """Render a type annotation without memory addresses or module-object noise."""
    if annotation is inspect.Parameter.empty:
        return "<empty>"
    if annotation is None or annotation is type(None):
        return "None"
    if annotation is Any:
        return "Any"
    if isinstance(annotation, TypeVar):
        return str(annotation)
    if isinstance(annotation, type):
        if annotation.__module__ == "builtins":
            return annotation.__qualname__
        return f"{annotation.__module__}.{annotation.__qualname__}"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return "Literal[" + ", ".join(repr(value) for value in args) + "]"
    if origin is Annotated:
        metadata = ", ".join(_format_metadata(item) for item in args[1:])
        return f"Annotated[{format_annotation(args[0])}, {metadata}]"
    if origin in (Union, types.UnionType):
        return " | ".join(format_annotation(item) for item in args)
    if origin is not None:
        name = getattr(origin, "__qualname__", str(origin))
        if origin.__module__ not in ("builtins", "typing", "types"):
            name = f"{origin.__module__}.{name}"
        elif origin.__module__ == "types" and name in ("UnionType",):
            name = "Union"
        return f"{name}[{', '.join(format_annotation(item) for item in args)}]"
    text = repr(annotation)
    return _ADDRESS_RE.sub("", text)


def format_value(value: Any) -> Any:
    """Convert a default/constant value to a deterministic JSON-safe form."""
    if value is PydanticUndefined:
        return "<PydanticUndefined>"
    if isinstance(value, enum.Enum):
        return {
            "__enum__": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if isinstance(value, BaseModel):
        return {
            "__model__": f"{type(value).__module__}.{type(value).__qualname__}",
            "data": value.model_dump(mode="json"),
        }
    if isinstance(value, Path):
        return {"__path__": value.as_posix()}
    if isinstance(value, tuple):
        return [format_value(item) for item in value]
    if isinstance(value, list):
        return [format_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): format_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _ADDRESS_RE.sub("", repr(value))


def _format_signature(obj: Any) -> str:
    """Render a callable signature with deterministic annotations and kinds."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):  # pragma: no cover - builtins without signature
        return "<no signature>"
    parts: list[str] = []
    positional_only: list[int] = []
    star_emitted = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not star_emitted:
            parts.append("*")
            star_emitted = True
        prefix = ""
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
            star_emitted = True
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            prefix = "**"
        text = f"{prefix}{parameter.name}"
        if parameter.annotation is not inspect.Parameter.empty:
            text += f": {format_annotation(parameter.annotation)}"
        if parameter.default is not inspect.Parameter.empty:
            text += f" = {format_value(parameter.default)!r}"
        parts.append(text)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only.append(len(parts) - 1)
    if positional_only:
        parts.insert(positional_only[-1] + 1, "/")
    result = f"({', '.join(parts)})"
    if signature.return_annotation is not inspect.Signature.empty:
        result += f" -> {format_annotation(signature.return_annotation)}"
    return result


def _model_snapshot(model: type[BaseModel]) -> dict[str, Any]:
    """Describe fields and configuration of a Pydantic model."""
    fields: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        factory = field.default_factory
        fields[name] = {
            "annotation": format_annotation(field.annotation),
            "required": field.is_required(),
            "default": format_value(field.default),
            "default_factory": getattr(factory, "__qualname__", None)
            if factory is not None
            else None,
            "alias": field.alias,
            "validation_alias": _ADDRESS_RE.sub("", repr(field.validation_alias))
            if field.validation_alias is not None
            else None,
        }
    config: dict[str, Any] = {}
    for key in (
        "extra",
        "frozen",
        "populate_by_name",
        "validate_assignment",
        "str_strip_whitespace",
        "use_enum_values",
        "arbitrary_types_allowed",
    ):
        if key in model.model_config:
            config[key] = format_value(model.model_config[key])
    return {"fields": fields, "config": config}


def _symbol_snapshot(name: str, value: Any) -> dict[str, Any]:
    """Describe one public symbol (kind, canonical module and details)."""
    entry: dict[str, Any] = {
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
    }
    if inspect.isclass(value):
        if getattr(value, "_is_protocol", False):
            entry["kind"] = "protocol"
            entry["call"] = _format_signature(value.__call__)
        elif issubclass(value, BaseModel):
            entry["kind"] = "pydantic_model"
            entry.update(_model_snapshot(value))
        elif issubclass(value, enum.Enum):
            entry["kind"] = "enum"
            entry["members"] = {member.name: member.value for member in value}
        elif issubclass(value, BaseException):
            entry["kind"] = "exception"
            entry["mro"] = [
                f"{base.__module__}.{base.__qualname__}" for base in value.__mro__
            ]
        else:
            entry["kind"] = "class"
    elif get_origin(value) is Literal:
        entry["kind"] = "literal"
        entry["values"] = list(get_args(value))
    elif (
        get_origin(value) is Annotated
        or get_origin(value)
        in (
            Union,
            types.UnionType,
        )
        or isinstance(value, type)
    ):
        entry["kind"] = "type_alias"
        entry["annotation"] = format_annotation(value)
    elif inspect.isfunction(value) or inspect.ismethod(value):
        entry["kind"] = "function"
        entry["signature"] = _format_signature(value)
    elif callable(value):
        entry["kind"] = "callable"
        entry["signature"] = _format_signature(value)
    else:
        entry["kind"] = "constant"
        entry["value"] = format_value(value)
    return entry


def build_public_api() -> dict[str, Any]:
    """Build the versioned public-API payload (without metadata fields)."""
    import automl_contracts

    modules: dict[str, list[str]] = {"automl_contracts": list(automl_contracts.__all__)}
    for module_name in PUBLIC_SUBMODULES:
        modules[module_name] = module_exports(module_name)

    symbols: dict[str, dict[str, Any]] = {}
    for module_name, names in modules.items():
        module = importlib.import_module(module_name)
        for name in names:
            if name == "__version__":
                continue
            if name in symbols:
                continue
            if not hasattr(module, name):
                raise AssertionError(f"{module_name}.{name} is exported but missing")
            symbols[name] = _symbol_snapshot(name, getattr(module, name))

    return {"modules": modules, "symbols": dict(sorted(symbols.items()))}


def api_digest(api: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest of the public-API payload."""
    payload = json.dumps(api, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_snapshot() -> dict[str, Any]:
    """Build the complete golden snapshot including release metadata."""
    import automl_contracts

    api = build_public_api()
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract_version": automl_contracts.__version__,
        "changelog_version": read_changelog_latest_version(),
        "pyproject_version": read_contract_version(),
        "api_digest": api_digest(api),
        **api,
    }


def load_golden() -> dict[str, Any]:
    """Read the committed golden snapshot."""
    return json.loads(PUBLIC_API_GOLDEN.read_text(encoding="utf-8"))


def load_release_history() -> dict[str, dict[str, Any]]:
    """Read the released-version -> public-API entry history.

    Older single-digest entries are read as schema version 1 and are
    re-pinned on the next snapshot regeneration.
    """
    if not RELEASE_HISTORY.is_file():
        return {}
    raw = json.loads(RELEASE_HISTORY.read_text(encoding="utf-8"))
    return {
        version: (
            entry
            if isinstance(entry, dict)
            else {"schema_version": 1, "api_digest": str(entry)}
        )
        for version, entry in raw.items()
    }


def write_golden() -> Path:
    """Regenerate the public-API snapshot and its release history entry.

    The contract version, the ``pyproject.toml`` version and the newest
    changelog release must already agree.  A public-API change for an existing
    version is rejected: bump the version (MINOR or MAJOR, never PATCH) and add
    a changelog entry first.
    """
    snapshot = build_snapshot()
    version = snapshot["contract_version"]
    if not (version == snapshot["pyproject_version"] == snapshot["changelog_version"]):
        raise SystemExit(
            "Bump automl-contracts/pyproject.toml and add the CHANGELOG.md "
            "release entry before regenerating the public-API snapshot."
        )
    history = load_release_history()
    previous = history.get(version)
    if (
        previous is not None
        and previous["schema_version"] == snapshot["schema_version"]
        and previous["api_digest"] != snapshot["api_digest"]
    ):
        raise SystemExit(
            f"The public API changed but the contract version is still "
            f"{version}. Bump the version (MINOR or MAJOR) and add a "
            "CHANGELOG.md entry, then regenerate the snapshot."
        )
    history[version] = {
        "schema_version": snapshot["schema_version"],
        "api_digest": snapshot["api_digest"],
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_API_GOLDEN.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    RELEASE_HISTORY.write_text(
        json.dumps(dict(sorted(history.items())), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PUBLIC_API_GOLDEN


def _api_reference_section(title: str) -> list[str]:
    """Return the lines of one top-level section of ``API_REFERENCE.md``."""
    lines = API_REFERENCE_DOC.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(title)
    except ValueError as exc:
        raise AssertionError(f"API_REFERENCE.md has no {title!r} section") from exc
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("# "):
            end = index
            break
    return lines[start + 1 : end]


def parse_documented_public_modules() -> set[str]:
    """Return the public modules documented in ``API_REFERENCE.md``.

    The section "Public imports" contains one row per stable module. The root
    package is documented by the top-level import block and is always included.
    """
    modules = {"automl_contracts"}
    for line in _api_reference_section("## Public imports"):
        if not line.startswith("| `automl_contracts"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        modules.add(cells[0].strip("`"))
    return modules


def parse_documented_public_symbols() -> dict[str, str]:
    """Map documented public symbols to the module that documents them.

    Both the top-level import block and the stable-submodule table of
    ``API_REFERENCE.md`` are used. A symbol is therefore checked against the
    module the user is told to import it from.
    """
    text = API_REFERENCE_DOC.read_text(encoding="utf-8")
    start = text.index("from automl_contracts import (")
    end = text.index(")\n```", start) + 1
    import_node = ast.parse(text[start:end]).body[0]
    assert isinstance(import_node, ast.ImportFrom)
    result = {
        alias.asname or alias.name: "automl_contracts" for alias in import_node.names
    }
    for line in _api_reference_section("## Public imports"):
        if not line.startswith("| `automl_contracts"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        module = cells[0].strip("`")
        for symbol in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", cells[1]):
            result[symbol] = module
    return result


def parse_documented_engine_paths() -> set[str]:
    """Return the old engine import paths documented in ``API_REFERENCE.md``."""
    paths: set[str] = set()
    for line in _api_reference_section("## Migration from engine imports"):
        match = re.match(r"\|\s*`(configurable_automl_engine[^`]*)`", line)
        if match:
            paths.add(match.group(1))
    return paths


def contract_public_names() -> set[str]:
    """Return the documented public names of the contract package.

    ``ScalingType`` is a deprecated submodule alias and is not part of the
    root ``__all__``; the contract Protocols are documented in the public API
    reference.  ``__version__`` is metadata, not an API symbol.
    """
    from automl_contracts import __all__ as root_all

    names = set(root_all) - {"__version__"}
    names.add("ScalingType")
    names.update({"MetricFunction", "ScorerCallable"})
    return names


def read_engine_contract_version_spec() -> str:
    """Return the engine dependency specifier for ``automl-contracts``."""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for dependency in data["project"]["dependencies"]:
        if str(dependency).startswith("automl-contracts"):
            return str(dependency)
    raise AssertionError("the engine does not declare an automl-contracts dependency")
