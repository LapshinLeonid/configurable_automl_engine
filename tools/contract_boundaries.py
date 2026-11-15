"""Static guards for the contract/engine architecture boundary.

The module is shared by the pytest guards and the CI entry points.  It keeps
the checks independent from the engine runtime: contract imports are inspected
with :mod:`ast`, metadata is read from ``pyproject.toml``, and distribution
artifacts are inspected with :mod:`zipfile`/:mod:`tarfile`.
"""

from __future__ import annotations

import ast
import re
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT / "automl-contracts"
CONTRACT_SRC = CONTRACT_ROOT / "src" / "automl_contracts"
CONTRACT_PYPROJECT = CONTRACT_ROOT / "pyproject.toml"
ENGINE_SRC = REPO_ROOT / "src" / "configurable_automl_engine"
ENGINE_PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_CONFIG = REPO_ROOT / ".sourcecraft" / "ci.yaml"

ENGINE_MODULE = "configurable_automl_engine"
ALLOWED_CONTRACT_DEPENDENCIES = frozenset({"pydantic", "pydantic-core"})
ALLOWED_THIRD_PARTY_ROOTS = frozenset({"pydantic", "pydantic_core"})
FORBIDDEN_LIBRARY_ROOTS = frozenset(
    {
        "imblearn",
        "joblib",
        "matplotlib",
        "mlflow",
        "numpy",
        "optuna",
        "pandas",
        "pyarrow",
        "scipy",
        "seaborn",
        "shap",
        "sklearn",
        "tensorflow",
        "torch",
        "xgboost",
        "yaml",
    }
)
DYNAMIC_IMPORT_CALLS = frozenset(
    {
        "__import__",
        "import_module",
        "importlib.__import__",
        "importlib.import_module",
        "importlib.util.find_spec",
        "importlib.util.spec_from_file_location",
        "find_spec",
        "spec_from_file_location",
    }
)
RUNTIME_CALL_PREFIXES = (
    "builtins.open",
    "ctypes.",
    "io.open",
    "multiprocessing.",
    "os.access",
    "os.chdir",
    "os.chmod",
    "os.chown",
    "os.cpu_count",
    "os.getcwd",
    "os.getenv",
    "os.getenvb",
    "os.listdir",
    "os.makedirs",
    "os.mkdir",
    "os.putenv",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.rmdir",
    "os.stat",
    "os.symlink",
    "os.uname",
    "os.walk",
    "os.path.",
    "pathlib.Path.exists",
    "pathlib.Path.glob",
    "pathlib.Path.is_dir",
    "pathlib.Path.is_file",
    "pathlib.Path.open",
    "pathlib.Path.read_bytes",
    "pathlib.Path.read_text",
    "pathlib.Path.write_bytes",
    "pathlib.Path.write_text",
    "platform.",
    "psutil.",
    "resource.",
    "shutil.copy",
    "shutil.copy2",
    "shutil.move",
    "shutil.rmtree",
    "shutil.which",
    "socket.",
    "subprocess.",
)
RUNTIME_ATTRIBUTES = frozenset(
    {
        "os.environ",
        "os.name",
        "sys.getwindowsversion",
        "sys.implementation",
        "sys.platform",
        "sys.version_info",
    }
)
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


@dataclass(frozen=True)
class Violation:
    """One architecture-boundary violation."""

    rule: str
    path: Path
    line: int
    message: str

    def format(self, root: Path = REPO_ROOT) -> str:
        """Render the violation as a stable, repository-relative message."""
        try:
            relative = self.path.relative_to(root)
        except ValueError:
            relative = self.path
        return f"{relative}:{self.line}: [{self.rule}] {self.message}"


def python_files(root: Path) -> list[Path]:
    """Return all Python source files below ``root`` in deterministic order."""
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and ".mypy_cache" not in path.parts
        and ".ruff_cache" not in path.parts
    )


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node: ast.AST) -> str | None:
    """Return the dotted spelling of a Name/Attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _resolve_callable_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a call target through local import aliases."""
    name = _dotted_name(node)
    if name is None:
        return None
    head, dot, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if dot else resolved


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local import names to their fully qualified module paths."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _absolute_imports(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yield line numbers and modules of absolute imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def _stdlib_roots() -> frozenset[str]:
    return frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def _is_allowed_contract_root(module: str) -> bool:
    root = module.split(".")[0]
    if root in _stdlib_roots():
        return True
    if root in ALLOWED_THIRD_PARTY_ROOTS:
        return True
    return root == "automl_contracts"


def scan_contract_imports(contract_src: Path = CONTRACT_SRC) -> list[Violation]:
    """Reject engine and non-contract libraries imported by the contract."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        for line, module in _absolute_imports(tree):
            root = module.split(".")[0]
            if root == ENGINE_MODULE:
                violations.append(
                    Violation(
                        "dependency-direction",
                        path,
                        line,
                        "the contract must not import "
                        f"{ENGINE_MODULE}: dependencies point from the engine "
                        "to the contract, never back",
                    )
                )
            elif not _is_allowed_contract_root(module):
                violations.append(
                    Violation(
                        "contract-dependency",
                        path,
                        line,
                        f"the contract may only use the standard library and "
                        f"{sorted(ALLOWED_CONTRACT_DEPENDENCIES)}, found "
                        f"{module!r}",
                    )
                )
    return violations


def _annotation_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        annotation: ast.AST | None = None
        if isinstance(node, (ast.AnnAssign, ast.arg)):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        if annotation is not None:
            yield annotation


def _type_strings(node: ast.AST) -> Iterator[ast.Constant]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child


def _string_type_violation(
    path: Path, node: ast.Constant, rule: str
) -> Violation | None:
    value = str(node.value)
    if re.search(rf"\b{re.escape(ENGINE_MODULE)}\b", value):
        return Violation(
            rule,
            path,
            node.lineno,
            f"a string type annotation refers to {ENGINE_MODULE}; the "
            "contract cannot depend on the engine, including forward "
            "references",
        )
    for root in FORBIDDEN_LIBRARY_ROOTS:
        if re.search(rf"\b{re.escape(root)}\b", value):
            return Violation(
                rule,
                path,
                node.lineno,
                f"a string type annotation refers to forbidden library {root!r}",
            )
    return None


def _assigned_type_strings(path: Path, tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) or (
            isinstance(node, ast.AnnAssign) and node.value is not None
        ):
            values = [node.value]
        else:
            values = []
        for value in values:
            if isinstance(value, ast.Call):
                continue
            for constant in _type_strings(value):
                violation = _string_type_violation(path, constant, "string-annotation")
                if violation is not None:
                    violations.append(violation)
    return violations


def _call_type_strings(path: Path, tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name is None:
            continue
        if name.rsplit(".", 1)[-1] not in {
            "ForwardRef",
            "NewType",
            "TypeVar",
            "cast",
        }:
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        for argument in arguments:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                violation = _string_type_violation(path, argument, "string-annotation")
                if violation is not None:
                    violations.append(violation)
    return violations


def scan_contract_type_strings(contract_src: Path = CONTRACT_SRC) -> list[Violation]:
    """Reject engine/heavy references hidden in string annotations."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        for annotation in _annotation_nodes(tree):
            for constant in _type_strings(annotation):
                violation = _string_type_violation(path, constant, "string-annotation")
                if violation is not None:
                    violations.append(violation)
        violations.extend(_assigned_type_strings(path, tree))
        violations.extend(_call_type_strings(path, tree))
    return violations


def scan_contract_dynamic_imports(contract_src: Path = CONTRACT_SRC) -> list[Violation]:
    """Reject dynamic import calls in the contract."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        aliases = _import_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _resolve_callable_name(node.func, aliases)
            if name in DYNAMIC_IMPORT_CALLS:
                violations.append(
                    Violation(
                        "dynamic-import",
                        path,
                        node.lineno,
                        f"dynamic import through {name!r} is forbidden in the "
                        "contract; use a static import",
                    )
                )
    return violations


def scan_contract_runtime_access(
    contract_src: Path = CONTRACT_SRC,
) -> list[Violation]:
    """Reject environment, filesystem and process access in the contract."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        aliases = _import_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                name = _resolve_callable_name(node, aliases)
                if name in RUNTIME_ATTRIBUTES:
                    violations.append(
                        Violation(
                            "runtime-access",
                            path,
                            node.lineno,
                            f"the contract must not read {name!r}: runtime "
                            "environment checks belong to the engine",
                        )
                    )
                continue
            if not isinstance(node, ast.Call):
                continue
            name = _resolve_callable_name(node.func, aliases)
            if name is None:
                continue
            if name == "open" or name.startswith(RUNTIME_CALL_PREFIXES):
                violations.append(
                    Violation(
                        "runtime-access",
                        path,
                        node.lineno,
                        f"the contract must not use {name!r}: runtime and "
                        "environment access belong to the engine",
                    )
                )
    return violations


def _requirement_name(requirement: str) -> str:
    without_marker = requirement.split(";", 1)[0].strip()
    name = re.split(r"[<>=!~\[\s(]", without_marker, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def load_pyproject(path: Path) -> dict[str, object]:
    """Read a TOML project file."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _project_dependencies(project: dict[str, object]) -> list[str]:
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        return []
    return [str(item) for item in dependencies]


def _optional_dependencies(project: dict[str, object]) -> list[str]:
    groups = project.get("optional-dependencies", {})
    if not isinstance(groups, dict):
        return []
    result: list[str] = []
    for values in groups.values():
        if isinstance(values, list):
            result.extend(str(item) for item in values)
    return result


def scan_contract_metadata(
    pyproject: Path = CONTRACT_PYPROJECT,
) -> list[Violation]:
    """Verify the declarative dependency and type-checking policy."""
    violations: list[Violation] = []
    document = load_pyproject(pyproject)
    project = document.get("project", {})
    if not isinstance(project, dict):
        return [Violation("contract-metadata", pyproject, 1, "missing [project]")]
    for requirement in _project_dependencies(project) + _optional_dependencies(project):
        name = _requirement_name(requirement)
        if name not in ALLOWED_CONTRACT_DEPENDENCIES:
            violations.append(
                Violation(
                    "contract-metadata",
                    pyproject,
                    1,
                    f"runtime dependency {name!r} is not allowed; only "
                    f"{sorted(ALLOWED_CONTRACT_DEPENDENCIES)} are permitted "
                    "without an ADR exception",
                )
            )
    version = project.get("version")
    if not isinstance(version, str) or not SEMVER_RE.match(version):
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                f"contract version {version!r} is not a valid semantic version",
            )
        )
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        tool = {}
    mypy = tool.get("mypy", {})
    if not isinstance(mypy, dict) or mypy.get("strict") is not True:
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                "the contract mypy section must enable strict mode",
            )
        )
    importlinter = tool.get("importlinter", {})
    if not isinstance(importlinter, dict):
        importlinter = {}
    if importlinter.get("exclude_type_checking_imports") is not False:
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                "import-linter must inspect TYPE_CHECKING imports "
                "(exclude_type_checking_imports = false)",
            )
        )
    contracts = importlinter.get("contracts", [])
    forbidden_engine = False
    if isinstance(contracts, list):
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            if contract.get("type") != "forbidden":
                continue
            forbidden = contract.get("forbidden_modules", [])
            if isinstance(forbidden, list) and ENGINE_MODULE in forbidden:
                forbidden_engine = True
    if not forbidden_engine:
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                "import-linter must declare a forbidden-modules contract for "
                f"{ENGINE_MODULE}",
            )
        )
    return violations


def _ruff_policy(pyproject: Path) -> dict[str, object]:
    document = load_pyproject(pyproject)
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        return {}
    ruff = tool.get("ruff", {})
    if not isinstance(ruff, dict):
        return {}
    lint = ruff.get("lint", {})
    if not isinstance(lint, dict):
        lint = {}
    mccabe = lint.get("mccabe", {})
    if not isinstance(mccabe, dict):
        mccabe = {}
    fmt = ruff.get("format", {})
    if not isinstance(fmt, dict):
        fmt = {}
    return {
        "line-length": ruff.get("line-length"),
        "target-version": ruff.get("target-version"),
        "extend-select": lint.get("extend-select"),
        "max-complexity": mccabe.get("max-complexity"),
        "quote-style": fmt.get("quote-style"),
        "indent-style": fmt.get("indent-style"),
        "line-ending": fmt.get("line-ending"),
    }


def scan_ruff_policy(
    engine_pyproject: Path = ENGINE_PYPROJECT,
    contract_pyproject: Path = CONTRACT_PYPROJECT,
) -> list[Violation]:
    """Verify that both distributions use the same Ruff rules."""
    engine = _ruff_policy(engine_pyproject)
    contract = _ruff_policy(contract_pyproject)
    violations: list[Violation] = []
    for key, value in engine.items():
        if contract.get(key) != value:
            violations.append(
                Violation(
                    "ruff-policy",
                    contract_pyproject,
                    1,
                    f"contract Ruff option {key!r} differs from the engine: "
                    f"expected {value!r}, found {contract.get(key)!r}",
                )
            )
    return violations


def _contract_names_from_file(path: Path) -> set[str]:
    tree = _parse(path)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            names.add(node.name)
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            continue
        for element in node.value.elts:
            if (
                isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                and element.value not in {"__all__", "__version__"}
            ):
                names.add(element.value)
    return names


def contract_public_names(contract_src: Path = CONTRACT_SRC) -> set[str]:
    """Return public contract classes, functions and constants."""
    names: set[str] = set()
    for path in python_files(contract_src):
        names.update(_contract_names_from_file(path))
    return names


def scan_engine_duplicates(
    engine_src: Path = ENGINE_SRC,
    contract_src: Path = CONTRACT_SRC,
) -> list[Violation]:
    """Reject engine definitions that duplicate contract names."""
    contract_names = contract_public_names(contract_src)
    violations: list[Violation] = []
    for path in python_files(engine_src):
        tree = _parse(path)
        for node in tree.body:
            duplicate = False
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                duplicate = node.name in contract_names
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                duplicate = any(
                    isinstance(target, ast.Name) and target.id in contract_names
                    for target in targets
                )
            if duplicate:
                name = (
                    node.name
                    if isinstance(
                        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    else next(
                        target.id
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                        if isinstance(target, ast.Name)
                    )
                )
                violations.append(
                    Violation(
                        "engine-duplicate",
                        path,
                        node.lineno,
                        f"the engine defines {name!r}, which belongs to the "
                        "contract; import and re-export it from "
                        "automl_contracts instead",
                    )
                )
    return violations


def scan_engine_contract_usage(
    engine_src: Path = ENGINE_SRC,
    engine_pyproject: Path = ENGINE_PYPROJECT,
) -> list[Violation]:
    """Verify that the engine depends on and imports the contract."""
    violations: list[Violation] = []
    document = load_pyproject(engine_pyproject)
    project = document.get("project", {})
    dependencies = _project_dependencies(project) if isinstance(project, dict) else []
    dependency_names = {_requirement_name(item) for item in dependencies}
    if "automl-contracts" not in dependency_names:
        violations.append(
            Violation(
                "engine-contract-dependency",
                engine_pyproject,
                1,
                "the engine must declare automl-contracts as a runtime dependency",
            )
        )
    imports_contract = False
    for path in python_files(engine_src):
        tree = _parse(path)
        for _, module in _absolute_imports(tree):
            if module == "automl_contracts" or module.startswith("automl_contracts."):
                imports_contract = True
                break
        if imports_contract:
            break
    if not imports_contract:
        violations.append(
            Violation(
                "engine-contract-dependency",
                engine_src,
                1,
                "the engine does not import automl_contracts; shared "
                "definitions must come from the contract",
            )
        )
    return violations


def _package_data(document: dict[str, object]) -> dict[str, object]:
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        return {}
    setuptools = tool.get("setuptools", {})
    if not isinstance(setuptools, dict):
        return {}
    package_data = setuptools.get("package-data", {})
    return package_data if isinstance(package_data, dict) else {}


def scan_contract_package_data(
    contract_root: Path = CONTRACT_ROOT,
    pyproject: Path = CONTRACT_PYPROJECT,
) -> list[Violation]:
    """Verify that py.typed and the JSON Schema are packaged."""
    violations: list[Violation] = []
    document = load_pyproject(pyproject)
    package_data = _package_data(document)
    declared = package_data.get("automl_contracts", [])
    if not isinstance(declared, list):
        declared = []
    for required in ("py.typed", "schemas/*.json"):
        if required not in declared:
            violations.append(
                Violation(
                    "package-data",
                    pyproject,
                    1,
                    f"package data must include {required!r}",
                )
            )
    for required_path in (
        contract_root / "src" / "automl_contracts" / "py.typed",
        contract_root / "src" / "automl_contracts" / "schemas" / "config.schema.json",
    ):
        if not required_path.is_file():
            violations.append(
                Violation(
                    "package-data",
                    required_path,
                    1,
                    "required contract package data is missing",
                )
            )
    return violations


def _metadata_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line or line.startswith(" "):
            continue
        key, separator, value = line.partition(": ")
        if not separator:
            continue
        sections.setdefault(key, []).append(value)
    return sections


def _find_member(names: Sequence[str], suffix: str) -> str | None:
    for name in names:
        if name.endswith(suffix):
            return name
    return None


def _check_distribution_dependencies(
    path: Path, requirements: Iterable[str]
) -> list[Violation]:
    violations: list[Violation] = []
    for requirement in requirements:
        name = _requirement_name(requirement)
        if name not in ALLOWED_CONTRACT_DEPENDENCIES:
            violations.append(
                Violation(
                    "artifact-metadata",
                    path,
                    1,
                    f"contract metadata requires forbidden dependency {name!r}",
                )
            )
    return violations


def inspect_contract_wheel(
    wheel: Path, expected_version: str | None = None
) -> list[Violation]:
    """Inspect a built contract wheel for contents and metadata."""
    violations: list[Violation] = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    for name in names:
        if ENGINE_MODULE in name:
            violations.append(
                Violation(
                    "artifact-content",
                    wheel,
                    1,
                    f"contract wheel contains engine file {name!r}",
                )
            )
        if name.endswith(".py") and not name.startswith("automl_contracts/"):
            violations.append(
                Violation(
                    "artifact-content",
                    wheel,
                    1,
                    f"contract wheel contains non-contract Python module {name!r}",
                )
            )
    for required in (
        "automl_contracts/__init__.py",
        "automl_contracts/py.typed",
        "automl_contracts/schemas/config.schema.json",
    ):
        if required not in names:
            violations.append(
                Violation(
                    "artifact-content",
                    wheel,
                    1,
                    f"contract wheel is missing {required!r}",
                )
            )
    metadata_name = _find_member(names, ".dist-info/METADATA")
    if metadata_name is None:
        return violations + [
            Violation(
                "artifact-metadata",
                wheel,
                1,
                "contract wheel has no .dist-info/METADATA",
            )
        ]
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(metadata_name).decode("utf-8")
    sections = _metadata_sections(metadata)
    names_meta = sections.get("Name", [])
    versions = sections.get("Version", [])
    if names_meta != ["automl-contracts"]:
        violations.append(
            Violation(
                "artifact-metadata",
                wheel,
                1,
                f"contract wheel name must be 'automl-contracts', found {names_meta!r}",
            )
        )
    if not versions or not SEMVER_RE.match(versions[0]):
        violations.append(
            Violation(
                "artifact-metadata",
                wheel,
                1,
                f"contract wheel version {versions!r} is not semver",
            )
        )
    elif expected_version is not None and versions[0] != expected_version:
        violations.append(
            Violation(
                "artifact-metadata",
                wheel,
                1,
                f"contract wheel version {versions[0]!r} does not match "
                f"pyproject version {expected_version!r}",
            )
        )
    violations.extend(
        _check_distribution_dependencies(wheel, sections.get("Requires-Dist", []))
    )
    return violations


def inspect_contract_sdist(
    sdist: Path, expected_version: str | None = None
) -> list[Violation]:
    """Inspect a built contract sdist for contents and metadata."""
    violations: list[Violation] = []
    with tarfile.open(sdist, "r:*") as archive:
        names = archive.getnames()
    for name in names:
        if ENGINE_MODULE in name:
            violations.append(
                Violation(
                    "artifact-content",
                    sdist,
                    1,
                    f"contract sdist contains engine file {name!r}",
                )
            )
    for required in (
        "src/automl_contracts/__init__.py",
        "src/automl_contracts/py.typed",
        "src/automl_contracts/schemas/config.schema.json",
    ):
        if not any(name.endswith(required) for name in names):
            violations.append(
                Violation(
                    "artifact-content",
                    sdist,
                    1,
                    f"contract sdist is missing {required!r}",
                )
            )
    pyproject = next((name for name in names if name.endswith("/pyproject.toml")), None)
    if pyproject is None:
        return violations + [
            Violation(
                "artifact-metadata",
                sdist,
                1,
                "contract sdist has no pyproject.toml",
            )
        ]
    if expected_version is not None:
        with tarfile.open(sdist, "r:*") as archive:
            payload = archive.extractfile(pyproject)
            if payload is None:
                raise ValueError(f"cannot read {pyproject} from {sdist}")
            document = tomllib.loads(payload.read().decode("utf-8"))
        version = document.get("project", {}).get("version")
        if version != expected_version:
            violations.append(
                Violation(
                    "artifact-metadata",
                    sdist,
                    1,
                    f"contract sdist version {version!r} does not match "
                    f"pyproject version {expected_version!r}",
                )
            )
    return violations


def inspect_engine_artifacts(
    wheel: Path | None = None, sdist: Path | None = None
) -> list[Violation]:
    """Verify that engine artifacts do not contain contract sources."""
    violations: list[Violation] = []
    if wheel is not None:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if "automl_contracts" in name:
                    violations.append(
                        Violation(
                            "artifact-separation",
                            wheel,
                            1,
                            f"engine wheel contains contract file {name!r}",
                        )
                    )
    if sdist is not None:
        with tarfile.open(sdist, "r:*") as archive:
            for name in archive.getnames():
                if "automl_contracts" in name:
                    violations.append(
                        Violation(
                            "artifact-separation",
                            sdist,
                            1,
                            f"engine sdist contains contract file {name!r}",
                        )
                    )
    return violations


def scan_ci_configuration(ci_path: Path = CI_CONFIG) -> list[Violation]:
    """Verify that the CI workflow contains the contract gates."""
    if not ci_path.is_file():
        return [
            Violation("ci-configuration", ci_path, 1, "CI configuration is missing")
        ]
    text = ci_path.read_text(encoding="utf-8")
    required = {
        "contracts-guarantees": "contracts-guarantees task",
        "scripts/check_contract_boundaries.py": "static boundary check",
        "scripts/verify_contract_distribution.py": "artifact verification",
        "python3.11 -m build automl-contracts": "contract build",
        "twine check": "twine metadata check",
        "python3.11 -m venv": "isolated environment",
        "pytest tests/contracts": "contract tests",
        "mypy src": "contract mypy",
        "ruff check src/ automl-contracts/src/": "ruff for both packages",
    }
    violations: list[Violation] = []
    for token, description in required.items():
        if token not in text:
            violations.append(
                Violation(
                    "ci-configuration",
                    ci_path,
                    1,
                    f"CI configuration has no {description} ({token!r})",
                )
            )
    return violations


def run_static_checks(repo_root: Path = REPO_ROOT) -> list[Violation]:
    """Run every static guard against a repository checkout."""
    contract_root = repo_root / "automl-contracts"
    contract_src = contract_root / "src" / "automl_contracts"
    contract_pyproject = contract_root / "pyproject.toml"
    engine_src = repo_root / "src" / "configurable_automl_engine"
    engine_pyproject = repo_root / "pyproject.toml"
    ci_path = repo_root / ".sourcecraft" / "ci.yaml"
    violations: list[Violation] = []
    violations.extend(scan_contract_imports(contract_src))
    violations.extend(scan_contract_type_strings(contract_src))
    violations.extend(scan_contract_dynamic_imports(contract_src))
    violations.extend(scan_contract_runtime_access(contract_src))
    violations.extend(scan_contract_metadata(contract_pyproject))
    violations.extend(scan_contract_package_data(contract_root, contract_pyproject))
    violations.extend(scan_ruff_policy(engine_pyproject, contract_pyproject))
    violations.extend(scan_engine_duplicates(engine_src, contract_src))
    violations.extend(scan_engine_contract_usage(engine_src, engine_pyproject))
    violations.extend(scan_ci_configuration(ci_path))
    return violations
