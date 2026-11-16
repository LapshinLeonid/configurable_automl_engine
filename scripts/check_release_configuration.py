"""Guard the CI configuration of the ``automl-contracts`` release pipeline.

The checks parse both CI files and keep publication attached to the
``contracts-v*`` tag. Every quality gate must run in its own cube or step
before the upload, the uploader is scanned for forbidden flags, and secrets
must come from the CI secret store rather than from committed literals.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCECRAFT_CI = REPO_ROOT / ".sourcecraft" / "ci.yaml"
GITHUB_CI = REPO_ROOT / ".github" / "workflows" / "CI.yaml"
GITHUB_PUBLISH = REPO_ROOT / ".github" / "workflows" / "publish-contracts.yaml"

SOURCE_CUBES = (
    "preflight",
    "quality-gates",
    "build-and-verify",
    "publish",
    "verify-from-registry",
)
RELEASE_WORKFLOWS = ("publish-contracts", "sync-to-github")
PAT_NAME = "PYPI_REGISTRY_PAT"
PAT_SECRET = "${{ secrets.PYPI_REGISTRY_PAT }}"
REGISTRY_ID_NAME = "SOURCECRAFT_PYPI_REGISTRY_ID"
PAT_SCOPED_CUBES = ("publish", "verify-from-registry")
SOURCE_CUBE_MARKERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "preflight": (
        (
            "scripts/contract_release.py check-tag",
            '--ref "${SOURCECRAFT_COMMIT_REF}"',
        ),
    ),
    "quality-gates": (
        ("scripts/check_contract_boundaries.py",),
        ("python3.11 scripts/check_release_configuration.py",),
        ("python3.11 -m ruff check src/ automl-contracts/src/",),
        ("python3.11 -m mypy scripts/contract_release.py",),
        ("importlinter.cli",),
        ("pytest tests/contracts",),
    ),
    "build-and-verify": (
        ("python3.11 -m build automl-contracts",),
        ("twine check",),
        ("scripts/verify_contract_distribution.py",),
        ("scripts/smoke_contract_install.py",),
        ('"$REPO_DIR/scripts/e2e_contract_consumer.py"',),
    ),
    "publish": (
        (
            "scripts/contract_release.py check-tag",
            '--ref "${SOURCECRAFT_COMMIT_REF}"',
        ),
        ("scripts/contract_release.py publish",),
    ),
    "verify-from-registry": (
        ("scripts/contract_release.py install",),
        ('"$REPO_DIR/scripts/e2e_contract_consumer.py"',),
    ),
}
REGULAR_CHECK_MARKERS = (
    "python3.11 scripts/check_release_configuration.py",
    "scripts/smoke_contract_install.py",
    '"$REPO_DIR/scripts/smoke_engine_install.py"',
    "pytest tests/contracts",
)
GITHUB_STEPS = (
    "Check the release tag",
    "Run release quality gates",
    "Build and verify the distribution",
    "Install the exact artifact in a clean environment",
)
GITHUB_STEP_MARKERS = {
    "Check the release tag": ("scripts/contract_release.py check-tag",),
    "Run release quality gates": (
        "scripts/check_contract_boundaries.py",
        "python scripts/check_release_configuration.py",
        "ruff check src/ automl-contracts/src/",
        "mypy scripts/contract_release.py",
        "lint-imports",
        "pytest tests/contracts",
    ),
    "Build and verify the distribution": (
        "python -m build automl-contracts",
        "twine check",
        "scripts/verify_contract_distribution.py",
    ),
    "Install the exact artifact in a clean environment": (
        "scripts/smoke_contract_install.py",
    ),
}
_LITERAL_CREDENTIAL = re.compile(r"iam:(?!\$|\{|\*)")
_LITERAL_SECRET = re.compile(r"--password(?:=|\s+|\",\s*)?(?![\"']?\$)")
_SECRET_ENV = re.compile(
    r"^\s*(?P<name>[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PAT))\s*:\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)
_LITERAL_SECRET_ASSIGNMENT = re.compile(
    r"^\s*[\"']?(?:PYPI_REGISTRY_PAT|TWINE_PASSWORD)[\"']?\s*[:=]\s*[\"'][^\"']+[\"']",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Violation:
    """A release configuration rule violation.

    Args:
        path: File that violates the rule.
        line: One-based line number, or 1 when the rule is file-wide.
        message: Human-readable explanation.
    """

    path: Path
    line: int
    message: str

    def format(self, root: Path) -> str:
        """Render the violation relative to ``root``.

        Args:
            root: Repository root used for the relative path.

        Returns:
            A ``path:line: message`` diagnostic.
        """
        try:
            name = self.path.relative_to(root)
        except ValueError:
            name = self.path
        return f"{name}:{self.line}: {self.message}"


def _uncommented(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _line_of(lines: list[str], token: str) -> int:
    for index, line in enumerate(lines, start=1):
        if token in line:
            return index
    return 1


def _load_yaml(path: Path, text: str) -> dict[str, Any] | None:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return document if isinstance(document, dict) else None


def _on(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("on")
    if value is None:
        value = document.get(True)  # type: ignore[call-overload]
    return value if isinstance(value, dict) else {}


_YAML_FORM = "__yaml_form__"


def _entries(on_data: dict[str, Any], trigger: str) -> list[dict[str, Any]]:
    raw = on_data.get(trigger, [])
    if raw is None:
        return [{_YAML_FORM: "all workflows", "workflows": []}]
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            item = {"workflows": [item]}
            form = "workflow string"
        elif isinstance(item, list):
            item = {"workflows": [name for name in item if isinstance(name, str)]}
            form = "nested workflow list"
        elif isinstance(item, dict):
            item = dict(item)
            form = "mapping"
        else:
            continue
        item[_YAML_FORM] = form
        entries.append(item)
    return entries


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int))]
    return []


def _filter(entry: dict[str, Any], key: str) -> list[str]:
    raw = entry.get("filter")
    return _names(raw.get(key)) if isinstance(raw, dict) else []


def _workflow_names(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("workflows")
    if isinstance(raw, dict):
        return [str(key) for key in raw]
    if isinstance(raw, list):
        names: list[str] = []
        for item in raw:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.extend(str(key) for key in item)
        return names
    return _names(raw)


def _entry_covers_all_workflows(entry: dict[str, Any]) -> bool:
    return not _workflow_names(entry)


def _entry_runs(entry: dict[str, Any], workflow: str) -> bool:
    return _entry_covers_all_workflows(entry) or workflow in _workflow_names(entry)


def _has_only_contract_tag_filter(entry: dict[str, Any]) -> bool:
    tags = _filter(entry, "tags")
    branches = _filter(entry, "branches")
    paths = _filter(entry, "paths")
    return tags == ["contracts-v*"] and not branches and not paths


def _has_tag_entry(
    push_entries: list[dict[str, Any]], workflow: str, *, only_entry: bool
) -> bool:
    matching = [entry for entry in push_entries if workflow in _workflow_names(entry)]
    if only_entry and len(matching) != 1:
        return False
    return any(_has_only_contract_tag_filter(entry) for entry in matching)


def _document_tasks(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = document.get("tasks")
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                result[item["name"]] = item
    elif isinstance(raw, dict):
        for name, item in raw.items():
            if isinstance(item, dict):
                task = dict(item)
                task.setdefault("name", str(name))
                result[str(name)] = task
    return result


def _tasks(
    workflow: dict[str, Any], document: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    tasks = workflow.get("tasks")
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list):
        return []
    referenced = _document_tasks(document) if document is not None else {}
    result: list[dict[str, Any]] = []
    for task in tasks:
        if isinstance(task, str):
            resolved = referenced.get(task)
            if resolved is not None:
                result.append(resolved)
        elif isinstance(task, dict):
            result.append(task)
    return result


def _task_reference_violations(
    path: Path, workflow: dict[str, Any], document: dict[str, Any]
) -> list[Violation]:
    tasks = workflow.get("tasks")
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list):
        return []
    referenced = _document_tasks(document)
    return [
        Violation(
            path,
            1,
            f"workflow references unknown task {task!r}",
        )
        for task in tasks
        if isinstance(task, str) and task not in referenced
    ]


def _task_name(task: dict[str, Any], index: int) -> str:
    name = task.get("name")
    return name if isinstance(name, str) and name else f"task-{index}"


def _task_cube_names(task: dict[str, Any]) -> list[str]:
    raw_cubes = task.get("cubes")
    if not isinstance(raw_cubes, list):
        return []
    return [
        cube["name"]
        for cube in raw_cubes
        if isinstance(cube, dict) and isinstance(cube.get("name"), str)
    ]


def _cubes(
    workflow: dict[str, Any], document: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task in _tasks(workflow, document):
        raw_cubes = task.get("cubes")
        if not isinstance(raw_cubes, list):
            continue
        for cube in raw_cubes:
            if isinstance(cube, dict) and isinstance(cube.get("name"), str):
                result[cube["name"]] = cube
    return result


def _cube_tasks(
    workflow: dict[str, Any], document: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task in _tasks(workflow, document):
        for name in _task_cube_names(task):
            result[name] = task
    return result


def _script(cube: dict[str, Any]) -> str:
    raw = cube.get("script")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(str(item) for item in raw)
    return ""


_HEREDOC = re.compile(
    r"<<-?\s*(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)
_SHELL_BLOCK_OPEN = re.compile(r"^(?:if|while|until|for|case)\b")
_SHELL_BLOCK_END = re.compile(r"^(?:fi|done|esac)\b")
_SHELL_BLOCK_BRANCH = re.compile(r"^(?:elif|else)\b")
_SHELL_CONDITION_LINE = re.compile(r"^(?:if|elif|while|until)\b")
_BRACE_OPEN = re.compile(r"(?:^|\s)\{\s*$")
_BRACE_CLOSE = re.compile(r"^\}")
_FALSE_CONDITION = re.compile(
    r"^(?:if|elif|while|until)\s+(?P<condition>.*?)(?:;\s*(?:then|do)\b|$)"
)
_FUNCTION_DECLARATION = re.compile(
    r"^(?:function\s+[A-Za-z_][A-Za-z0-9_-]*\s*(?:\(\s*\))?"
    r"|[A-Za-z_][A-Za-z0-9_-]*\s*\(\s*\))\s*[{\(]"
)
_ERREXIT = re.compile(r"^set\s+(?:-[A-Za-z]*e[A-Za-z]*\b|-o\s+errexit\b)")
_DISABLE_ERREXIT = re.compile(r"^set\s+(?:\+[A-Za-z]*e[A-Za-z]*\b|\+o\s+errexit\b)")
_ASSIGNMENT = re.compile(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*(?:\+)?=")


def _strip_shell_comment(line: str) -> str:
    quote: str | None = None
    for index, character in enumerate(line):
        if quote is not None:
            if character == quote:
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _condition_is_false(line: str) -> bool:
    match = _FALSE_CONDITION.match(line)
    if match is None:
        return False
    condition = match.group("condition").strip()
    return bool(re.fullmatch(r"(?:false|0|!\s*true|\[\s*false\s*\])", condition))


def _frames_dead(frames: list[tuple[bool, bool]]) -> bool:
    if not frames:
        return False
    parent_dead, branch_dead = frames[-1]
    return parent_dead or branch_dead


def _shell_line_dead(line: str, frames: list[tuple[bool, bool]]) -> bool:
    """Update the block stack for ``line`` and report whether it may be skipped."""
    if (_SHELL_BLOCK_END.match(line) or _BRACE_CLOSE.match(line)) and frames:
        frames.pop()
    if _SHELL_BLOCK_BRANCH.match(line) and frames:
        parent_dead, _ = frames[-1]
        branch_dead = parent_dead
        if line.startswith("elif"):
            branch_dead = parent_dead or _condition_is_false(line)
        frames[-1] = (parent_dead, branch_dead)
    dead = _frames_dead(frames) or bool(_SHELL_CONDITION_LINE.match(line))
    if _SHELL_BLOCK_OPEN.match(line) and not re.search(r"\b(?:fi|done|esac)\b", line):
        parent_dead = _frames_dead(frames)
        frames.append((parent_dead, parent_dead or _condition_is_false(line)))
    elif _BRACE_OPEN.search(line):
        parent_dead = _frames_dead(frames)
        frames.append((parent_dead, True))
    return dead


def _logical_commands(script: str) -> list[tuple[str, bool]]:
    """Return executable shell lines and whether they may be skipped."""
    raw_lines = script.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    commands: list[tuple[str, bool]] = []
    heredoc: str | None = None
    buffer = ""
    frames: list[tuple[bool, bool]] = []
    for raw_line in raw_lines:
        if heredoc is not None:
            if raw_line.strip() == heredoc:
                heredoc = None
            continue
        match = _HEREDOC.search(raw_line)
        if match is not None:
            heredoc = match.group("delimiter")
        line = _strip_shell_comment(raw_line).rstrip()
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        if line.endswith(("||", "&&", "|")):
            buffer += line + " "
            continue
        line = (buffer + line).strip()
        buffer = ""
        if not line:
            continue
        commands.append((line, _shell_line_dead(line, frames)))
    if buffer.strip():
        commands.append((buffer.strip(), _frames_dead(frames)))
    return commands


def _segments(command: str) -> list[tuple[str, str | None]]:
    """Split a shell command on top-level ``;``, ``&&``, ``||``, ``|`` and ``&``."""
    segments: list[tuple[str, str | None]] = []
    buffer: list[str] = []
    quote: str | None = None
    connector: str | None = None
    index = 0
    while index < len(command):
        character = command[index]
        if quote is not None:
            buffer.append(character)
            if character == quote:
                quote = None
            index += 1
            continue
        if character in "\"'":
            quote = character
            buffer.append(character)
            index += 1
            continue
        if command.startswith(("&&", "||"), index):
            segments.append(("".join(buffer).strip(), connector))
            connector = command[index : index + 2]
            buffer = []
            index += 2
            continue
        if character in ";|&":
            segments.append(("".join(buffer).strip(), connector))
            connector = character
            buffer = []
            index += 1
            continue
        buffer.append(character)
        index += 1
    segments.append(("".join(buffer).strip(), connector))
    return segments


def _segment_can_run(
    text: str,
    connector: str | None,
    next_connector: str | None,
    dead: bool,
    blocked: bool,
) -> bool:
    if not text or dead or blocked or connector in {"||", "|", "&"}:
        return False
    if next_connector in {"||", "&&", "|", "&"}:
        return False
    first = text.split(None, 1)[0]
    if first in {
        "echo",
        "printf",
        "if",
        "elif",
        "while",
        "until",
        "!",
        "false",
        "true",
        ":",
        "exit",
        "return",
    }:
        return False
    if text.startswith("!") or _FUNCTION_DECLARATION.match(text):
        return False
    return not (_ASSIGNMENT.match(text) and "$(" not in text)


def _requirement_executed(
    commands: list[tuple[str, bool]], markers: tuple[str, ...]
) -> bool:
    blocked = False
    for command, dead in commands:
        segments = _segments(command)
        for index, (text, connector) in enumerate(segments):
            if not text:
                continue
            first = text.split(None, 1)[0]
            if (
                not dead
                and connector not in {"||", "&&", "|", "&"}
                and first in {"false", "exit", "return"}
            ):
                blocked = True
            if not all(marker in text for marker in markers):
                continue
            next_connector = (
                segments[index + 1][1] if index + 1 < len(segments) else None
            )
            if _segment_can_run(text, connector, next_connector, dead, blocked):
                return True
    return False


def _shell_violations(
    path: Path,
    cube_name: str,
    script: str,
    markers: tuple[tuple[str, ...], ...],
) -> list[Violation]:
    commands = _logical_commands(script)
    violations: list[Violation] = []
    first_command: tuple[str, bool] | None = None
    for command, dead in commands:
        for text, _ in _segments(command):
            if text and first_command is None:
                first_command = (text, dead)
            if _DISABLE_ERREXIT.match(text):
                violations.append(
                    Violation(
                        path,
                        1,
                        f"cube {cube_name!r} disables error handling with "
                        "'set +e' or 'set +o errexit'",
                    )
                )
        if re.search(r"(?:\|\||;|&&)\s*(?:true|:)\s*(?:$|[;&|])", command):
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {cube_name!r} swallows a failure with a 'true' fallback",
                )
            )
        if re.match(r"^!\s+\S", command):
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {cube_name!r} negates a command with '!'",
                )
            )
    if (
        first_command is None
        or first_command[1]
        or not _ERREXIT.match(first_command[0])
    ):
        violations.append(
            Violation(
                path,
                1,
                f"cube {cube_name!r} must enable 'set -e' as its first "
                "executable command",
            )
        )
    return violations


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


def _check_cube(
    path: Path,
    cube_name: str,
    cube: dict[str, Any],
    markers: tuple[tuple[str, ...], ...],
    task: dict[str, Any] | None = None,
) -> list[Violation]:
    script = _script(cube)
    commands = _logical_commands(script)
    violations: list[Violation] = []
    if _flag(cube.get("allow_failure")):
        violations.append(
            Violation(
                path,
                1,
                f"cube {cube_name!r} must not set allow_failure: true",
            )
        )
    if "if" in cube:
        violations.append(
            Violation(path, 1, f"cube {cube_name!r} must not have a conditional 'if'")
        )
    if task is not None:
        task_name = task.get("name", "unnamed task")
        if _flag(task.get("allow_failure")):
            violations.append(
                Violation(
                    path,
                    1,
                    f"task {task_name!r} must not set allow_failure: true",
                )
            )
        if "if" in task:
            violations.append(
                Violation(
                    path,
                    1,
                    f"task {task_name!r} must not have a conditional 'if'",
                )
            )
    violations.extend(_shell_violations(path, cube_name, script, markers))
    violations.extend(
        Violation(
            path,
            1,
            f"cube {cube_name!r} has no executable "
            + " and ".join(repr(marker) for marker in requirement),
        )
        for requirement in markers
        if not _requirement_executed(commands, requirement)
    )
    return violations


def _order_violation(
    path: Path, names: tuple[str, ...], actual: list[str], label: str
) -> list[Violation]:
    indexes = [actual.index(name) for name in names if name in actual]
    if len(indexes) != len(names):
        return []
    expected = sorted(indexes)
    if indexes == expected:
        return []
    return [
        Violation(
            path,
            1,
            f"{label} order is {actual}, expected {list(names)}",
        )
    ]


def _needs(task: dict[str, Any]) -> list[str]:
    return _names(task.get("needs"))


def _cycle_nodes(edges: dict[str, set[str]]) -> set[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            cycle.add(node)
            return True
        if node in visited:
            return False
        visiting.add(node)
        found = any(visit(child) for child in edges.get(node, set()))
        visiting.discard(node)
        visited.add(node)
        return found

    for node in edges:
        visit(node)
    return cycle


def _reachable(edges: dict[str, set[str]], start: str, target: str) -> bool:
    pending = [start]
    seen: set[str] = set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        pending.extend(edges.get(node, set()))
    return False


def _collect_task_graph(
    path: Path, tasks: list[dict[str, Any]]
) -> tuple[
    list[Violation],
    dict[str, set[str]],
    dict[str, str],
    dict[str, list[str]],
    set[str],
]:
    violations: list[Violation] = []
    edges: dict[str, set[str]] = {}
    owner: dict[str, str] = {}
    task_cubes: dict[str, list[str]] = {}
    task_names: set[str] = set()
    for index, task in enumerate(tasks):
        name = _task_name(task, index)
        task_names.add(name)
        cubes = _task_cube_names(task)
        task_cubes[name] = cubes
        for cube in cubes:
            if cube in owner:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"cube {cube!r} is declared in more than one task",
                    )
                )
            owner[cube] = name
            edges.setdefault(cube, set())
        for first, second in pairwise(cubes):
            edges.setdefault(first, set()).add(second)
    return violations, edges, owner, task_cubes, task_names


def _task_dependency_violations(
    path: Path,
    tasks: list[dict[str, Any]],
    edges: dict[str, set[str]],
    owner: dict[str, str],
    task_cubes: dict[str, list[str]],
    task_names: set[str],
) -> list[Violation]:
    violations: list[Violation] = []
    for index, task in enumerate(tasks):
        name = _task_name(task, index)
        cubes = task_cubes[name]
        for dependency in _needs(task):
            if dependency == "-":
                violations.append(
                    Violation(
                        path,
                        1,
                        f'task {name!r} uses needs: ["-"], which permits '
                        "parallel execution",
                    )
                )
                continue
            if dependency not in task_names:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"task {name!r} needs unknown task {dependency!r}",
                    )
                )
                continue
            dependency_cubes = task_cubes.get(dependency, [])
            if dependency_cubes and cubes:
                edges.setdefault(dependency_cubes[-1], set()).add(cubes[0])
        raw_cubes = task.get("cubes")
        if not isinstance(raw_cubes, list):
            continue
        for cube in raw_cubes:
            if not isinstance(cube, dict) or not isinstance(cube.get("name"), str):
                continue
            cube_name = cube["name"]
            for dependency in _names(cube.get("needs")):
                if dependency == "-":
                    violations.append(
                        Violation(
                            path,
                            1,
                            f'cube {cube_name!r} uses needs: ["-"], which '
                            "permits parallel execution",
                        )
                    )
                elif dependency not in owner:
                    violations.append(
                        Violation(
                            path,
                            1,
                            f"cube {cube_name!r} needs unknown cube {dependency!r}",
                        )
                    )
                else:
                    edges.setdefault(dependency, set()).add(cube_name)
    return violations


def _task_graph_violations(path: Path, tasks: list[dict[str, Any]]) -> list[Violation]:
    """Check that tasks and needs enforce the publication order."""
    violations, edges, owner, task_cubes, task_names = _collect_task_graph(path, tasks)
    violations.extend(
        _task_dependency_violations(path, tasks, edges, owner, task_cubes, task_names)
    )
    cycle = _cycle_nodes(edges)
    if cycle:
        names = ", ".join(sorted(cycle))
        violations.append(
            Violation(
                path,
                1,
                f"publish-contracts task order is cyclic around {names}",
            )
        )
        return violations
    for first, second in pairwise(SOURCE_CUBES):
        if first in owner and second in owner and not _reachable(edges, first, second):
            violations.append(
                Violation(
                    path,
                    1,
                    f"publish-contracts cube order is invalid: {first!r} must "
                    f"run before {second!r}; tasks run in parallel unless "
                    "needs chains them",
                )
            )
    return violations


def _secret_violations(path: Path, lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for index, line in enumerate(lines, start=1):
        match = _SECRET_ENV.search(line)
        if match is None:
            continue
        value = match.group("value").strip()
        if value.startswith("${{") or value in {"", '""', "''"}:
            continue
        violations.append(
            Violation(
                path,
                index,
                f"{match.group('name')} stores a literal secret value",
            )
        )
    return violations


def _credential_violations(path: Path, lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for index, line in enumerate(lines, start=1):
        if _LITERAL_CREDENTIAL.search(line):
            violations.append(
                Violation(
                    path,
                    index,
                    "the registry URL contains a literal credential instead of a variable",
                )
            )
        if _LITERAL_SECRET.search(line):
            violations.append(Violation(path, index, "a password literal is present"))
    return violations


def _check_uploader(path: Path, text: str) -> list[Violation]:
    lines = text.splitlines()
    violations: list[Violation] = []
    if "--skip-existing" in text:
        violations.append(
            Violation(
                path,
                _line_of(lines, "--skip-existing"),
                "publication must fail on an existing version, not skip it",
            )
        )
    for index, line in enumerate(lines, start=1):
        if "--password" in line:
            violations.append(
                Violation(path, index, "the uploader passes a password literal")
            )
        if _LITERAL_SECRET_ASSIGNMENT.search(line):
            violations.append(
                Violation(path, index, "the uploader embeds a literal PAT")
            )
        if "https://iam:" in line and "${" not in line:
            violations.append(
                Violation(path, index, "the uploader embeds a literal credential URL")
            )
    return violations


def _entry_label(entry: dict[str, Any]) -> str:
    form = entry.get(_YAML_FORM, "mapping")
    workflows = _workflow_names(entry)
    if not workflows:
        return f"{form} entry, all workflows"
    return f"{form} entry, workflows: {workflows!r}"


def _pull_request_trigger_violations(
    path: Path, entries: list[dict[str, Any]]
) -> list[Violation]:
    violations: list[Violation] = []
    if not any(_entry_runs(entry, "quality-checks") for entry in entries):
        violations.append(Violation(path, 1, "pull requests do not run quality-checks"))
    for index, entry in enumerate(entries, start=1):
        names = _workflow_names(entry)
        label = _entry_label(entry)
        if not names:
            violations.append(
                Violation(
                    path,
                    1,
                    f"pull_request entry {index} ({label}) applies to all "
                    "workflows, including release workflows; only "
                    "quality-checks is allowed on pull_request",
                )
            )
            continue
        release = [name for name in names if name in RELEASE_WORKFLOWS]
        if release:
            violations.append(
                Violation(
                    path,
                    1,
                    f"pull_request entry {index} ({label}) runs release "
                    f"workflow(s) {release!r}; only quality-checks is allowed "
                    "on pull_request",
                )
            )
        extra = [name for name in names if name != "quality-checks"]
        if extra and not release:
            violations.append(
                Violation(
                    path,
                    1,
                    f"pull_request entry {index} ({label}) runs workflow(s) "
                    f"{extra!r}; only quality-checks is allowed on pull_request",
                )
            )
    return violations


def _push_trigger_violations(
    path: Path, entries: list[dict[str, Any]]
) -> list[Violation]:
    violations: list[Violation] = []
    if not _has_tag_entry(entries, "publish-contracts", only_entry=True):
        violations.append(
            Violation(
                path,
                1,
                "publish-contracts must have exactly one push entry with only "
                "the contracts-v* tag filter",
            )
        )
    if not _has_tag_entry(entries, "sync-to-github", only_entry=False):
        violations.append(
            Violation(
                path,
                1,
                "sync-to-github has no push entry with the contracts-v* tag filter",
            )
        )
    for index, entry in enumerate(entries, start=1):
        names = _workflow_names(entry)
        label = _entry_label(entry)
        if not names:
            violations.append(
                Violation(
                    path,
                    1,
                    f"push entry {index} ({label}) applies to all workflows, "
                    "including release workflows; list workflows explicitly",
                )
            )
            continue
        if "publish-contracts" in names and not _has_only_contract_tag_filter(entry):
            violations.append(
                Violation(
                    path,
                    1,
                    f"push entry {index} ({label}) runs publish-contracts "
                    "without only the contracts-v* tag filter",
                )
            )
    return violations


def _other_trigger_violations(path: Path, on_data: dict[str, Any]) -> list[Violation]:
    violations: list[Violation] = []
    for trigger in on_data:
        if not isinstance(trigger, str) or trigger in {"push", "pull_request"}:
            continue
        for index, entry in enumerate(_entries(on_data, trigger), start=1):
            names = _workflow_names(entry)
            label = _entry_label(entry)
            if not names:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"trigger {trigger!r} entry {index} ({label}) applies "
                        "to all workflows, including release workflows",
                    )
                )
                continue
            release = [name for name in names if name in RELEASE_WORKFLOWS]
            if release:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"trigger {trigger!r} entry {index} ({label}) runs "
                        f"release workflow(s) {release!r}; release workflows "
                        "are allowed only in the push contracts-v* tag entry",
                    )
                )
    return violations


def _trigger_violations(
    path: Path, text: str, document: dict[str, Any]
) -> list[Violation]:
    on_data = _on(document)
    violations = _pull_request_trigger_violations(
        path, _entries(on_data, "pull_request")
    )
    violations.extend(_push_trigger_violations(path, _entries(on_data, "push")))
    violations.extend(_other_trigger_violations(path, on_data))
    if "refs/tags/" not in _uncommented(text):
        violations.append(
            Violation(path, 1, "the GitHub sync workflow never pushes a tag ref")
        )
    return violations


def _env(container: dict[str, Any]) -> dict[str, Any]:
    value = container.get("env")
    return value if isinstance(value, dict) else {}


def _pat_cube_violations(
    path: Path, workflow_name: str, cube: dict[str, Any]
) -> list[Violation]:
    name = cube.get("name")
    if not isinstance(name, str):
        return []
    allowed = workflow_name == "publish-contracts" and name in PAT_SCOPED_CUBES
    violations: list[Violation] = []
    if PAT_NAME in _env(cube):
        if not allowed:
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {name!r} in workflow {workflow_name!r} must not "
                    f"receive {PAT_NAME}; it is not a publishing cube",
                )
            )
        elif _env(cube).get(PAT_NAME) != PAT_SECRET:
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {name!r} does not read {PAT_NAME} from the "
                    "repository secret",
                )
            )
    if PAT_NAME in _uncommented(_script(cube)) and not allowed:
        violations.append(
            Violation(
                path,
                1,
                f"cube {name!r} in workflow {workflow_name!r} must not use {PAT_NAME}",
            )
        )
    return violations


def _pat_workflow_violations(
    path: Path,
    workflow_name: str,
    workflow: dict[str, Any],
    document: dict[str, Any],
) -> list[Violation]:
    violations: list[Violation] = []
    if PAT_NAME in _env(workflow):
        violations.append(
            Violation(
                path,
                1,
                f"{PAT_NAME} must not be defined at workflow level "
                f"({workflow_name!r}); scope it to the publishing cubes",
            )
        )
    for index, task in enumerate(_tasks(workflow, document)):
        if PAT_NAME in _env(task):
            violations.append(
                Violation(
                    path,
                    1,
                    f"task {_task_name(task, index)!r} must not receive "
                    f"{PAT_NAME}; scope it to the publishing cubes",
                )
            )
        raw_cubes = task.get("cubes")
        if not isinstance(raw_cubes, list):
            continue
        for cube in raw_cubes:
            if isinstance(cube, dict):
                violations.extend(_pat_cube_violations(path, workflow_name, cube))
    return violations


def _pat_scope_violations(
    path: Path, document: dict[str, Any], publish_cubes: dict[str, dict[str, Any]]
) -> list[Violation]:
    """Keep the registry PAT on the cubes that publish or install."""
    violations: list[Violation] = []
    if PAT_NAME in _env(document):
        violations.append(
            Violation(
                path,
                1,
                f"{PAT_NAME} must not be defined at document level; scope it "
                "to the publish and verify-from-registry cubes",
            )
        )
    workflows = document.get("workflows")
    if isinstance(workflows, dict):
        for workflow_name, workflow in workflows.items():
            if isinstance(workflow_name, str) and isinstance(workflow, dict):
                violations.extend(
                    _pat_workflow_violations(path, workflow_name, workflow, document)
                )
    for name in PAT_SCOPED_CUBES:
        cube = publish_cubes.get(name)
        if cube is not None and PAT_NAME not in _env(cube):
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {name!r} has no {PAT_NAME} environment variable",
                )
            )
    return violations


def _tag_guard_violations(
    path: Path, cube: dict[str, Any], cube_name: str
) -> list[Violation]:
    """Reject an advisory tag check that cannot stop publication."""
    script = _script(cube)
    if re.search(r"git\s+tag\s+--points-at\s+HEAD\s*\|\s*grep", script) is not None:
        return [
            Violation(
                path,
                1,
                f"cube {cube_name!r} uses an advisory tag check; publication "
                "must fail before the upload when the tag is not on HEAD",
            )
        ]
    return []


def _publish_violations(
    path: Path, text: str, document: dict[str, Any]
) -> list[Violation]:
    violations: list[Violation] = []
    workflows = document.get("workflows")
    if not isinstance(workflows, dict):
        return [Violation(path, 1, "CI configuration has no workflows")]
    publish = workflows.get("publish-contracts")
    if not isinstance(publish, dict):
        return [
            Violation(path, 1, "CI configuration has no publish-contracts workflow")
        ]
    tasks = _tasks(publish, document)
    cubes = _cubes(publish, document)
    cube_tasks = _cube_tasks(publish, document)
    violations.extend(_task_reference_violations(path, publish, document))
    actual_order = [name for name in cubes if name in SOURCE_CUBES]
    for name in SOURCE_CUBES:
        if name not in cubes:
            violations.append(
                Violation(path, 1, f"publish-contracts has no {name!r} cube")
            )
        else:
            violations.extend(
                _check_cube(
                    path,
                    name,
                    cubes[name],
                    SOURCE_CUBE_MARKERS[name],
                    task=cube_tasks.get(name),
                )
            )
    violations.extend(
        _order_violation(path, SOURCE_CUBES, actual_order, "publish-contracts")
    )
    violations.extend(_task_graph_violations(path, tasks))
    for name, cube in cubes.items():
        script = _uncommented(_script(cube))
        if name != "publish" and "scripts/contract_release.py publish" in script:
            violations.append(Violation(path, 1, f"cube {name!r} runs the uploader"))
        if name != "verify-from-registry" and (
            "scripts/contract_release.py install" in script
        ):
            violations.append(
                Violation(path, 1, f"cube {name!r} installs from the registry")
            )
        if name == "publish":
            violations.extend(_tag_guard_violations(path, cube, name))
    env = _env(publish)
    if REGISTRY_ID_NAME not in env:
        violations.append(
            Violation(path, 1, f"publish-contracts has no {REGISTRY_ID_NAME}")
        )
    violations.extend(_pat_scope_violations(path, document, cubes))
    violations.extend(_check_uploader(path, text))
    return violations


def _quality_violations(
    path: Path, workflows: dict[str, Any], document: dict[str, Any]
) -> list[Violation]:
    quality = workflows.get("quality-checks")
    if not isinstance(quality, dict):
        return [Violation(path, 1, "CI configuration has no quality-checks workflow")]
    cubes = _cubes(quality, document)
    all_scripts = _uncommented("\n".join(_script(cube) for cube in cubes.values()))
    return [
        Violation(path, 1, f"quality-checks workflow has no {marker!r}")
        for marker in REGULAR_CHECK_MARKERS
        if marker not in all_scripts
    ]


def check_sourcecraft_ci(text: str, path: Path = SOURCECRAFT_CI) -> list[Violation]:
    """Check the SourceCraft triggers and the publication workflow.

    Args:
        text: Content of ``.sourcecraft/ci.yaml``.
        path: File path used in diagnostics.

    Returns:
        Every violated release rule.
    """
    document = _load_yaml(path, text)
    if document is None:
        return [Violation(path, 1, "SourceCraft CI is not valid YAML")]
    workflows = document.get("workflows")
    if not isinstance(workflows, dict):
        return [Violation(path, 1, "CI configuration has no workflows")]
    lines = text.splitlines()
    violations = _trigger_violations(path, text, document)
    violations.extend(_publish_violations(path, text, document))
    violations.extend(_credential_violations(path, lines))
    violations.extend(_secret_violations(path, lines))
    violations.extend(_quality_violations(path, workflows, document))
    return violations


def check_github_publish(text: str, path: Path = GITHUB_PUBLISH) -> list[Violation]:
    """Check the GitHub verification mirror of the release workflow.

    Args:
        text: Content of ``.github/workflows/publish-contracts.yaml``.
        path: File path used in diagnostics.

    Returns:
        Every violated release rule.
    """
    lines = text.splitlines()
    document = _load_yaml(path, text)
    if document is None:
        return [Violation(path, 1, "GitHub release workflow is not valid YAML")]
    on_data = _on(document)
    push = on_data.get("push")
    if not isinstance(push, dict):
        return [
            Violation(path, 1, "GitHub release workflow is not limited to contracts-v*")
        ]
    tags = _names(push.get("tags"))
    branches = _names(push.get("branches"))
    paths = _names(push.get("paths"))
    violations: list[Violation] = []
    if tags != ["contracts-v*"] or branches or paths:
        violations.append(
            Violation(path, 1, "GitHub release workflow is not limited to contracts-v*")
        )
    jobs = document.get("jobs")
    job = next(iter(jobs.values())) if isinstance(jobs, dict) and jobs else None
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list):
        violations.append(Violation(path, 1, "GitHub release job has no steps"))
        return violations
    by_name: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        run = step.get("run")
        if isinstance(name, str) and isinstance(run, str):
            by_name[name] = run
    actual_order = [name for name in by_name if name in GITHUB_STEPS]
    for name in GITHUB_STEPS:
        if name not in by_name:
            violations.append(
                Violation(path, 1, f"GitHub release job has no {name!r} step")
            )
            continue
        clean = _uncommented(by_name[name])
        for marker in GITHUB_STEP_MARKERS[name]:
            if marker not in clean:
                violations.append(
                    Violation(path, 1, f"step {name!r} has no {marker!r}")
                )
    violations.extend(
        _order_violation(path, GITHUB_STEPS, actual_order, "GitHub release job")
    )
    for name, script in by_name.items():
        if "scripts/contract_release.py publish" in script:
            violations.append(
                Violation(path, 1, f"step {name!r} publishes from the GitHub mirror")
            )
    violations.extend(_check_uploader(path, text))
    violations.extend(_credential_violations(path, lines))
    violations.extend(_secret_violations(path, lines))
    return violations


def check_github_ci(text: str, path: Path = GITHUB_CI) -> list[Violation]:
    """Check that GitHub CI runs on pull requests and release guards.

    Args:
        text: Content of ``.github/workflows/CI.yaml``.
        path: File path used in diagnostics.

    Returns:
        Every violated release rule.
    """
    clean = _uncommented(text)
    violations: list[Violation] = []
    if re.search(r"^  pull_request:", clean, re.MULTILINE) is None:
        violations.append(
            Violation(path, 1, "GitHub CI is not triggered on pull requests")
        )
    for marker in (
        "python scripts/check_release_configuration.py",
        '"$GITHUB_WORKSPACE/scripts/smoke_engine_install.py"',
        "/tmp/contracts-venv/bin/python scripts/smoke_contract_install.py",
    ):
        if marker not in clean:
            violations.append(Violation(path, 1, f"GitHub CI has no {marker!r}"))
    return violations


def check_scripts(root: Path = REPO_ROOT) -> list[Violation]:
    """Check the helper scripts that implement the publication rules.

    Args:
        root: Repository root that contains the scripts.

    Returns:
        Every violated release rule.
    """
    violations: list[Violation] = []
    scripts = {
        "release": root / "scripts" / "contract_release.py",
        "e2e": root / "scripts" / "e2e_contract_consumer.py",
        "engine": root / "scripts" / "smoke_engine_install.py",
        "guard": root / "scripts" / "check_release_configuration.py",
    }
    for path in scripts.values():
        if not path.is_file():
            violations.append(Violation(path, 1, "release helper script is missing"))
    release = scripts["release"]
    if not release.is_file():
        return violations
    release_text = release.read_text(encoding="utf-8")
    for marker in (
        "SOURCECRAFT_PYPI_REGISTRY_ID",
        "PYPI_REGISTRY_PAT",
        "legacy",
        "simple",
        "twine",
        "already exists",
        "--no-deps",
    ):
        if marker not in release_text:
            violations.append(
                Violation(release, 1, f"release helper has no {marker!r}")
            )
    violations.extend(_check_uploader(release, release_text))
    for path in scripts.values():
        if path.is_file():
            violations.extend(
                _secret_violations(path, path.read_text(encoding="utf-8").splitlines())
            )
    return violations


def run_checks(root: Path = REPO_ROOT) -> list[Violation]:
    """Run every release configuration guard against ``root``.

    Args:
        root: Repository root to inspect.

    Returns:
        Every violated release rule.
    """
    sourcecraft = root / ".sourcecraft" / "ci.yaml"
    github_ci = root / ".github" / "workflows" / "CI.yaml"
    github_publish = root / ".github" / "workflows" / "publish-contracts.yaml"
    violations: list[Violation] = []
    if not sourcecraft.is_file():
        violations.append(Violation(sourcecraft, 1, "SourceCraft CI file is missing"))
    else:
        violations.extend(
            check_sourcecraft_ci(sourcecraft.read_text(encoding="utf-8"), sourcecraft)
        )
    if not github_publish.is_file():
        violations.append(
            Violation(github_publish, 1, "GitHub release workflow is missing")
        )
    else:
        violations.extend(
            check_github_publish(
                github_publish.read_text(encoding="utf-8"), github_publish
            )
        )
    if not github_ci.is_file():
        violations.append(Violation(github_ci, 1, "GitHub CI file is missing"))
    else:
        violations.extend(
            check_github_ci(github_ci.read_text(encoding="utf-8"), github_ci)
        )
    violations.extend(check_scripts(root))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Print every violation and return a non-zero status when one exists.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    Returns:
        ``0`` when the configuration is wired, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    arguments = parser.parse_args(argv)
    violations = run_checks(arguments.repo_root)
    if not violations:
        print("OK: contract release configuration is wired")
        return 0
    for violation in violations:
        print(violation.format(arguments.repo_root), file=sys.stderr)
    print(
        f"FAIL: {len(violations)} release configuration violation(s)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
