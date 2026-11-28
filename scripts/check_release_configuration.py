"""Guard the release configuration of the engine and its contracts.

The checks parse both CI files and keep publication attached to the
``contracts-v*`` tag. Every quality gate must run in its own cube or step
before the upload, the public PyPI upload works through Trusted Publishing,
the uploader is scanned for forbidden flags, and secrets must come from the CI
secret store rather than from committed literals. The engine version
configuration must ignore the ``contracts-v*`` tag family.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import shlex
import sys
import tomllib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
SOURCECRAFT_CI = REPO_ROOT / ".sourcecraft" / "ci.yaml"
GITHUB_CI = REPO_ROOT / ".github" / "workflows" / "CI.yaml"
GITHUB_PUBLISH = REPO_ROOT / ".github" / "workflows" / "publish-contracts.yaml"

ENGINE_TAG = "v1.2.3"
CONTRACT_TAG = "contracts-v0.1.0"
SETUPTOOLS_SCM_MIN_VERSION = Version("10.3.4")

SOURCE_CUBES = (
    "preflight",
    "quality-gates",
    "build-and-verify",
    "publish",
    "verify-from-registry",
)
RELEASE_WORKFLOWS = ("publish-contracts", "sync-to-github")
PAT_NAME = "PYPI_REGISTRY_PAT"
PAT_SECRET_EXPRESSION = re.compile(r"\$\{\{\s*secrets\.PYPI_REGISTRY_PAT\s*\}\}")
REGISTRY_ID_NAME = "SOURCECRAFT_PYPI_REGISTRY_ID"
PAT_SCOPED_CUBES = ("publish", "verify-from-registry")
PYTHON_SETUP_SCRIPT = "scripts/setup_python311.sh"
SUPPORTED_PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")
SETUP_PYTHON_ACTION = "actions/setup-python@v5"
MATRIX_VERSION_EXPRESSION = "${{ matrix.python-version }}"
_PYTHON311_INSTALL = re.compile(
    r"(?:add-apt-repository[^\n]*ppa:deadsnakes"
    r"|\bapt(?:-get)?\b[^\n]*\binstall\b[^\n]*python3\.11)"
)
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
        ("python3.11 -m ruff check src/ configurable-automl-contracts/src/",),
        ("python3.11 -m mypy scripts/contract_release.py",),
        ("importlinter.cli",),
        ("pytest tests/contracts",),
    ),
    "build-and-verify": (
        ("python3.11 -m build configurable-automl-contracts",),
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
GITHUB_PYPI_PUBLISH_STEP = "Publish to PyPI"
GITHUB_PYPI_VERIFY_STEP = "Verify installation from PyPI"
GITHUB_PYPI_SDIST_VERIFY_STEP = "Verify sdist installation from PyPI"
GITHUB_PYPI_ACTION = "pypa/gh-action-pypi-publish@release/v1"
GITHUB_PYPI_PACKAGES_DIR = "configurable-automl-contracts/dist"
GITHUB_PYPI_PUBLISH_INPUTS = frozenset({"packages-dir"})
GITHUB_PYPI_INSTALL_COMMAND = "configurable-automl-contracts==${VERSION}"
GITHUB_PYPI_INSTALL_MARKERS = (
    "--isolated",
    "PIP_CONFIG_FILE=/dev/null",
    "--index-url https://pypi.org/simple",
    "--no-cache-dir",
)
GITHUB_PYPI_VERIFY_MARKERS = (
    "scripts/smoke_contract_install.py",
    '"$GITHUB_WORKSPACE/scripts/e2e_contract_consumer.py"',
)
GITHUB_PYPI_SDIST_INSTALL_MARKERS = (
    "--no-binary configurable-automl-contracts",
    *GITHUB_PYPI_INSTALL_MARKERS,
)
GITHUB_STEP_MARKERS = {
    "Check the release tag": ("scripts/contract_release.py check-tag",),
    "Run release quality gates": (
        "scripts/check_contract_boundaries.py",
        "python scripts/check_release_configuration.py",
        "ruff check src/ configurable-automl-contracts/src/",
        "mypy scripts/contract_release.py",
        "lint-imports",
        "pytest tests/contracts",
    ),
    "Build and verify the distribution": (
        "python -m build configurable-automl-contracts",
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
_FALSE_CONDITION = re.compile(
    r"^(?:if|elif|while|until)\s+(?P<condition>.*?)(?:;\s*(?:then|do)\b|$)"
)
_FUNCTION_NAME = r"[^\s(){;]+"
_FUNCTION_DECLARATION = re.compile(
    rf"^(?:function\s+{_FUNCTION_NAME}(?:\s*\(\s*\))?"
    rf"|{_FUNCTION_NAME}\s*\(\s*\))\s*[{{(]"
)
_SHELL_EVENT = re.compile(
    r"(?:^|(?<=[;|&({}`]))\s*"
    r"(?:(?:time\s+(?:-p\s+)?)?(?:!\s+)?|(?:then|do)\s+)?"
    r"(?:"
    rf"(?P<function>(?:function\s+{_FUNCTION_NAME}(?:\s*\(\s*\))?"
    rf"|{_FUNCTION_NAME}\s*\(\s*\)))"
    r"(?P<body>\s*[{(])?"
    r"|(?P<open>if|while|until|for|select|case)\b"
    r"|(?P<close>fi|done|esac)\b"
    r"|(?P<branch>elif|else)\b"
    r"|(?P<brace>[{}])"
    r"|(?P<paren>[()])"
    r")"
)
_ERREXIT = re.compile(r"^set\s+(?:-[A-Za-z]*e[A-Za-z]*\b|-o\s+errexit\b)")
_DISABLE_ERREXIT = re.compile(r"^set\s+(?:\+[A-Za-z]*e[A-Za-z]*\b|\+o\s+errexit\b)")
_ASSIGNMENT = re.compile(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*(?:\+)?=")


_ESCAPING_QUOTES = frozenset({'"', "$'"})


def _starts_comment(line: str, index: int) -> bool:
    """Return whether ``#`` at ``index`` begins a shell comment."""
    if index == 0:
        return True
    previous = line[index - 1]
    if previous.isspace() or previous in ";|&()<>}":
        return True
    if previous == "{":
        return not (index >= 2 and line[index - 2] == "$")
    return False


def _strip_shell_comment(line: str) -> str:
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if quote is not None:
            if quote in _ESCAPING_QUOTES and character == "\\":
                index += 2
                continue
            if character == quote[-1]:
                quote = None
        elif line.startswith("$'", index):
            quote = "$'"
            index += 2
            continue
        elif character in "\"'":
            quote = character
        elif character == "#" and _starts_comment(line, index):
            return line[:index]
        index += 1
    return line


def _heredoc_delimiter(line: str) -> str | None:
    """Return a heredoc delimiter outside quotes and arithmetic."""
    index = 0
    quote: str | None = None
    arithmetic = 0
    while index < len(line):
        character = line[index]
        if quote is not None:
            if quote in _ESCAPING_QUOTES and character == "\\":
                index += 2
                continue
            if character == quote[-1]:
                quote = None
            index += 1
            continue
        if line.startswith("$'", index):
            quote = "$'"
            index += 2
            continue
        if character in "\"'":
            quote = character
            index += 1
            continue
        if line.startswith("$((", index):
            arithmetic += 1
            index += 3
            continue
        if line.startswith("((", index):
            arithmetic += 1
            index += 2
            continue
        if arithmetic:
            if character == "(":
                arithmetic += 1
            elif character == ")":
                arithmetic -= 1
            index += 1
            continue
        if line.startswith("<<", index):
            match = _HEREDOC.match(line, index)
            if match is not None:
                return match.group("delimiter")
            index += 2
            continue
        index += 1
    return None


def _unclosed_quote(line: str) -> str | None:
    """Return the quote character left open at the end of a line."""
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if quote is not None:
            if quote in _ESCAPING_QUOTES and character == "\\":
                index += 2
                continue
            if character == quote[-1]:
                quote = None
        elif line.startswith("$'", index):
            quote = "$'"
            index += 2
            continue
        elif character in "\"'":
            quote = character
        index += 1
    return quote


def _condition_is_false(line: str) -> bool:
    match = _FALSE_CONDITION.match(line)
    if match is None:
        return False
    condition = match.group("condition").strip()
    return bool(re.fullmatch(r"(?:false|0|!\s*true|\[\s*false\s*\])", condition))


_Frame = tuple[bool, bool, str]


def _frames_dead(frames: list[_Frame]) -> bool:
    if not frames:
        return False
    parent_dead, branch_dead, _kind = frames[-1]
    return parent_dead or branch_dead


def _unquoted_shell(line: str) -> str:
    """Blank out quoted text, keeping shell command separators."""
    characters: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if quote is not None:
            characters.append(" ")
            if (
                quote in _ESCAPING_QUOTES
                and character == "\\"
                and index + 1 < len(line)
            ):
                characters.append(" ")
                index += 2
                continue
            if character == quote[-1]:
                quote = None
        elif line.startswith("$'", index):
            quote = "$'"
            characters.extend((" ", " "))
            index += 2
            continue
        elif character in "\"'":
            quote = character
            characters.append(" ")
        else:
            characters.append(character)
        index += 1
    return "".join(characters)


def _shell_block_events(line: str) -> list[tuple[str, str, str]]:
    """Return block events in command order for one logical shell line."""
    events: list[tuple[str, str, str]] = []
    for match in _SHELL_EVENT.finditer(_unquoted_shell(line)):
        tail = line[match.start() :].lstrip()
        tail = re.sub(
            r"^(?:time\s+(?:-p\s+)?)?(?:!\s+)?(?:then\s+|do\s+)?",
            "",
            tail,
        )
        body = (match.group("body") or "").strip()
        if match.group("function"):
            if body == "{":
                events.append(("open", "brace", tail))
            elif body == "(":
                events.append(("open", "paren", tail))
            else:
                events.append(("pending", "function", tail))
        elif match.group("open"):
            events.append(("open", "block", tail))
        elif match.group("close"):
            events.append(("close", "block", tail))
        elif match.group("branch"):
            events.append(("branch", "block", tail))
        elif match.group("brace") == "{":
            events.append(("open", "brace", tail))
        elif match.group("brace") == "}":
            events.append(("close", "brace", tail))
        elif match.group("paren") == "(":
            prefix = _unquoted_shell(line[: match.start()]).rstrip()
            connector = ""
            if prefix.endswith(("&&", "||")):
                connector = prefix[-2:]
            elif prefix.endswith(("|", "&")):
                connector = prefix[-1]
            events.append(("open_paren", "paren", connector))
        elif match.group("paren") == ")":
            events.append(("close", "paren", tail))
    return events


def _unclosed_command_substitutions(line: str) -> int:
    """Return the number of ``$()`` or arithmetic groups left open."""
    sanitized = _unquoted_shell(line)
    depth = 0
    index = 0
    while index < len(sanitized):
        if sanitized.startswith("$((", index):
            depth += 1
            index += 3
            continue
        if sanitized.startswith("$(", index):
            depth += 1
            index += 2
            continue
        if sanitized.startswith("((", index):
            depth += 1
            index += 2
            continue
        if depth:
            if sanitized[index] == "(":
                depth += 1
            elif sanitized[index] == ")":
                depth -= 1
        index += 1
    return depth


def _shell_line_state(
    line: str,
    frames: list[_Frame],
    function_body: bool = False,
) -> tuple[bool, bool, bool]:
    """Update the block stack for ``line``.

    Args:
        line: One shell line with comments and continuations already removed.
        frames: Mutable stack of ``(parent_dead, branch_dead, kind)`` frames.
        function_body: Whether ``line`` opens a function body after ``foo()``.

    Returns:
        ``(dead, conditional, pending_function)``: whether the line may be
        skipped, whether it belongs to a non-linear block, and whether a
        function body is expected on a following line.
    """
    events = _shell_block_events(line)
    conditional = bool(frames) or bool(events)
    pending_function = False
    for action, kind, text in events:
        if action == "pending":
            pending_function = True
        elif action == "open":
            parent_dead = _frames_dead(frames)
            branch_dead = (
                True
                if kind in {"brace", "paren"}
                else parent_dead or _condition_is_false(text)
            )
            frames.append((parent_dead, branch_dead, kind))
        elif action == "open_paren":
            if function_body or any(kind == "paren" for _, _, kind in frames):
                parent_dead = _frames_dead(frames)
                frames.append((parent_dead, True, "paren"))
            elif text:
                parent_dead = _frames_dead(frames)
                frames.append((parent_dead, True, "subshell"))
        elif action == "close":
            closable = {"cmdsub", "subshell"} if kind == "paren" else {kind}
            if frames and frames[-1][2] in closable:
                frames.pop()
        elif action == "branch" and frames:
            parent_dead, _, top_kind = frames[-1]
            branch_dead = parent_dead
            if text.startswith("elif"):
                branch_dead = parent_dead or _condition_is_false(text)
            frames[-1] = (parent_dead, branch_dead, top_kind)
    opens = _unclosed_command_substitutions(line)
    if opens:
        parent_dead = _frames_dead(frames)
        frames.extend((parent_dead, True, "cmdsub") for _ in range(opens))
    return _frames_dead(frames), conditional, pending_function


def _logical_commands(script: str) -> list[tuple[str, bool, bool]]:
    """Return executable shell lines, skip state and block membership."""
    raw_lines = script.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    commands: list[tuple[str, bool, bool]] = []
    heredoc: str | None = None
    buffer = ""
    frames: list[_Frame] = []
    pending_function = False
    multiline_quote: str | None = None
    for raw_line in raw_lines:
        if heredoc is not None:
            if raw_line.strip() == heredoc:
                heredoc = None
            continue
        if multiline_quote is not None:
            if buffer.endswith("\\"):
                buffer = buffer[:-1]
            else:
                buffer += " "
            buffer += raw_line
            multiline_quote = _unclosed_quote(buffer)
            if multiline_quote is not None:
                continue
            line = _strip_shell_comment(buffer).rstrip()
            buffer = ""
        else:
            line = _strip_shell_comment(raw_line).rstrip()
            delimiter = _heredoc_delimiter(line)
            if delimiter is not None:
                heredoc = delimiter
            if not line and not buffer.rstrip().endswith(("||", "&&", "|")):
                line = buffer.strip()
                buffer = ""
            if not line:
                continue
            if line.endswith("\\"):
                buffer += line[:-1]
                continue
            if line.endswith(("||", "&&", "|")):
                buffer += line + " "
                continue
            line = (buffer + line).strip()
            buffer = ""
            multiline_quote = _unclosed_quote(line)
            if multiline_quote is not None:
                buffer = line
                continue
        dead, conditional, pending_function = _shell_line_state(
            line, frames, pending_function
        )
        commands.append((line, dead, conditional))
    if buffer.strip():
        text = buffer.strip()
        dead, conditional, _pending = _shell_line_state(text, frames, pending_function)
        commands.append((text, dead, conditional))
    unsafe = _unsafe_paren_commands(commands)
    if unsafe:
        commands = [
            (line, dead, conditional or index in unsafe)
            for index, (line, dead, conditional) in enumerate(commands)
        ]
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
            if (
                quote in _ESCAPING_QUOTES
                and character == "\\"
                and index + 1 < len(command)
            ):
                buffer.append(command[index + 1])
                index += 2
                continue
            if character == quote[-1]:
                quote = None
            index += 1
            continue
        if command.startswith("$'", index):
            quote = "$'"
            buffer.extend(("$", "'"))
            index += 2
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


def _unsafe_paren_commands(
    commands: list[tuple[str, bool, bool]],
) -> set[int]:
    """Return commands inside a subshell that is chained after its closing paren."""
    unsafe: set[int] = set()
    stack: list[int] = []
    for index, (line, _dead, _conditional) in enumerate(commands):
        sanitized = _unquoted_shell(line)
        for position, character in enumerate(sanitized):
            if character == "(":
                stack.append(index)
            elif character == ")" and stack:
                opened = stack.pop()
                if re.search(r"[&|]", sanitized[position + 1 :]):
                    unsafe.update(range(opened, index + 1))
    return unsafe


def _substitution_end(text: str, start: int) -> int | None:
    """Return the index of the closing paren of a ``$(...)`` body."""
    cursor = start
    depth = 1
    quote: str | None = None
    while cursor < len(text):
        current = text[cursor]
        if quote is not None:
            if quote in _ESCAPING_QUOTES and current == "\\":
                cursor += 2
                continue
            if current == quote[-1]:
                quote = None
            cursor += 1
            continue
        if text.startswith("$'", cursor):
            quote = "$'"
            cursor += 2
            continue
        if current in "\"'":
            quote = current
            cursor += 1
            continue
        if text.startswith("$(", cursor):
            depth += 1
            cursor += 1
            continue
        if current == ")":
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None


def _command_substitutions(text: str) -> list[str]:
    """Return top-level ``$(...)`` bodies outside quoted shell text."""
    bodies: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(text):
        character = text[index]
        if quote is not None:
            if quote in _ESCAPING_QUOTES and character == "\\":
                index += 2
                continue
            if character == quote[-1]:
                quote = None
            index += 1
            continue
        if text.startswith("$'", index):
            quote = "$'"
            index += 2
            continue
        if character in "\"'":
            quote = character
            index += 1
            continue
        if not text.startswith("$(", index):
            index += 1
            continue
        end = _substitution_end(text, index + 2)
        if end is None:
            break
        bodies.append(text[index + 2 : end])
        index = end + 1
    return bodies


_WRAPPER_HEADS = frozenset(
    {
        "builtin",
        "command",
        "doas",
        "env",
        "nice",
        "nohup",
        "setsid",
        "stdbuf",
        "sudo",
        "time",
        "timeout",
    }
)


def _shell_tokens(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _skip_options(
    tokens: list[str], index: int, options_with_value: frozenset[str]
) -> int:
    """Skip leading option tokens, including options that take an argument."""
    while index < len(tokens) and tokens[index].startswith("-"):
        if tokens[index] in options_with_value and index + 1 < len(tokens):
            index += 2
        else:
            index += 1
    return index


def _is_detaching_option(token: str) -> bool:
    if token.startswith("--"):
        name = token.split("=", 1)[0]
        return "--fork".startswith(name) or "--background".startswith(name)
    return False


def _wrapper_has_detaching_option(tokens: list[str], index: int, token: str) -> bool:
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        if _is_detaching_option(option):
            return True
        if not option.startswith("--"):
            if token == "setsid" and "f" in option[1:]:
                return True
            if token == "sudo" and "b" in option[1:]:
                return True
        index += 1
    return False


def _skip_wrapper_options(
    tokens: list[str], index: int, options_with_value: frozenset[str]
) -> int | None:
    """Skip wrapper options, rejecting options that terminate the wrapper."""
    while index < len(tokens) and tokens[index].startswith("-"):
        if tokens[index] in _TERMINAL_WRAPPER_OPTIONS:
            return None
        if tokens[index] in _DETACHING_WRAPPER_OPTIONS:
            return None
        if tokens[index] in options_with_value and index + 1 < len(tokens):
            index += 2
        else:
            index += 1
    return index


_WRAPPER_OPTIONS_WITH_VALUE: dict[str, frozenset[str]] = {
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "nice": frozenset({"-n"}),
    "sudo": frozenset(
        {
            "-u",
            "--user",
            "-g",
            "--group",
            "-p",
            "--prompt",
            "-r",
            "--role",
            "-t",
            "--type",
        }
    ),
    "doas": frozenset({"-u"}),
    "time": frozenset({"-f", "--format", "-o"}),
}
_WRAPPER_EXTRA_ARGS = {"timeout": 1}


def _env_option_is_split(token: str) -> bool:
    if token.startswith("--"):
        name = token.split("=", 1)[0]
        return len(name) > 2 and "--split-string".startswith(name)
    return token.startswith("-") and not token.startswith("--") and "S" in token[1:]


def _skip_env_options(tokens: list[str], index: int) -> int | None:
    while index < len(tokens):
        current = tokens[index]
        if current in _TERMINAL_WRAPPER_OPTIONS or _env_option_is_split(current):
            return None
        if current.startswith(("PATH=", "PYTHONPATH=")):
            return None
        if current in {"-u", "--unset"}:
            index += 2
        elif current.startswith("-") or "=" in current:
            index += 1
        else:
            break
    return index


def _command_lookup_option(tokens: list[str], index: int) -> bool:
    """Return whether command options request a path lookup instead of a run."""
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        if option in _TERMINAL_OPTIONS:
            return True
        if option in {"-v", "-V"}:
            return True
        if not option.startswith("--") and ("v" in option[1:] or "V" in option[1:]):
            return True
        index += 1
    return False


def _skip_wrapper(tokens: list[str], index: int, token: str) -> int | None:
    """Skip a command wrapper and its options, returning the next index."""
    if _wrapper_has_detaching_option(tokens, index, token):
        return None
    if token == "command":
        if _command_lookup_option(tokens, index):
            return None
        return _skip_wrapper_options(tokens, index, frozenset())
    if token == "builtin":
        return index
    if token == "env":
        return _skip_env_options(tokens, index)
    next_index = _skip_wrapper_options(
        tokens, index, _WRAPPER_OPTIONS_WITH_VALUE.get(token, frozenset())
    )
    if next_index is None:
        return None
    extra = _WRAPPER_EXTRA_ARGS.get(token, 0)
    return next_index + extra if next_index + extra <= len(tokens) else next_index


def _effective_command(text: str) -> tuple[str | None, list[str]]:
    """Return the first token and arguments after unwrapping prefixes."""
    tokens = _shell_tokens(text)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in _WRAPPER_HEADS:
            return token, tokens[index:]
        next_index = _skip_wrapper(tokens, index + 1, token)
        if next_index is None:
            return None, []
        index = next_index
    return None, []


def _effective_first_token(text: str) -> str | None:
    """Return the first token of the command after unwrapping prefixes."""
    return _effective_command(text)[0]


_NON_EXECUTING_FIRST_TOKENS = frozenset(
    {
        "[",
        "alias",
        "break",
        "case",
        "continue",
        "declare",
        "echo",
        "elif",
        "exit",
        "export",
        "false",
        "for",
        "function",
        "grep",
        "hash",
        "if",
        "local",
        "printf",
        "readonly",
        "return",
        "set",
        "shift",
        "test",
        "trap",
        "true",
        "typeset",
        "unalias",
        "unset",
        "until",
        "while",
        "!",
        ":",
    }
)


def _segment_can_run(
    text: str,
    connector: str | None,
    next_connector: str | None,
    dead: bool,
    blocked: bool,
) -> bool:
    if not text or dead or blocked or connector in {"||", "&&", "|", "&"}:
        return False
    if next_connector in {"||", "&&", "|", "&"}:
        return False
    first = _effective_first_token(text)
    if first is None or first in _NON_EXECUTING_FIRST_TOKENS:
        return False
    if text.startswith("!") or _FUNCTION_DECLARATION.match(text):
        return False
    return not (_ASSIGNMENT.match(text) and "$(" not in text)


def _dynamic_token(token: str) -> bool:
    """Return whether a token is produced by shell expansion."""
    return "$" in token or "`" in token


def _dynamic_command_word(text: str) -> bool:
    """Return whether the command name is produced by shell expansion."""
    tokens = _shell_tokens(text)
    index = 0
    while index < len(tokens) and _ASSIGNMENT.match(tokens[index]):
        index += 1
    if index >= len(tokens):
        return False
    if tokens[index] in _WRAPPER_HEADS:
        effective = _effective_first_token(" ".join(tokens[index:]))
        return effective is not None and _dynamic_token(effective)
    return _dynamic_token(tokens[index])


_SHELL_INTERPRETERS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_SCRIPT_INTERPRETERS = frozenset({"node", "perl", "php", "ruby"})
_TERMINAL_OPTIONS = frozenset({"--version", "--help", "-h", "-V", "-v"})
_TERMINAL_WRAPPER_OPTIONS = frozenset({"--version", "--help", "-V", "-h"})
_DETACHING_WRAPPER_OPTIONS = frozenset({"-f", "--fork", "-b", "--background"})
_SCRIPT_OPTION_VALUES: dict[str, frozenset[str]] = {
    "scripts/contract_release.py": frozenset(
        {
            "--ref",
            "--tag",
            "--pyproject",
            "--dist",
            "--expected-version",
            "--python",
            "--version",
            "--with-jsonschema",
        }
    ),
    "scripts/verify_contract_distribution.py": frozenset(
        {
            "--contract-wheel",
            "--contract-sdist",
            "--engine-wheel",
            "--engine-sdist",
            "--expected-version",
        }
    ),
    "scripts/e2e_contract_consumer.py": frozenset(
        {
            "--expected-version",
            "--reference-config",
            "--forbid-source-root",
        }
    ),
    "scripts/smoke_engine_install.py": frozenset(
        {
            "--expected-contract-version",
            "--import-matrix",
            "--forbid-source-root",
        }
    ),
}
_SCRIPT_FLAG_OPTIONS = frozenset({"--with-jsonschema"})
_SCRIPT_POSITIONALS: dict[str, frozenset[str]] = {
    "scripts/contract_release.py": frozenset(
        {"check-tag", "install", "publish", "show-endpoints"}
    ),
}


def _normalized_shell_text(text: str) -> str:
    return text.replace('"', "").replace("'", "")


def _terminal_option_before(tail: list[str], stop: int) -> bool:
    return any(token in _TERMINAL_OPTIONS for token in tail[1:stop])


def _is_python_interpreter(name: str) -> bool:
    return name.startswith("python")


def _linter_result_argument(
    argument: ast.expr, imported_names: set[str], marker: str
) -> bool:
    """Return whether an exit argument is the linter call itself."""
    if not isinstance(argument, ast.Call):
        return False
    function = argument.func
    if isinstance(function, ast.Name) and function.id in imported_names:
        return True
    if isinstance(function, ast.Attribute):
        text = ast.unparse(function)
        return text.startswith(f"{marker}.") and text.rsplit(".", 1)[-1] in {
            "lint_imports",
            *imported_names,
        }
    return False


def _python_exit_argument(tree: ast.Module) -> ast.expr | None:
    """Return the argument of a final top-level ``exit`` or ``SystemExit``."""
    if not tree.body:
        return None
    last = tree.body[-1]
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call):
        call = last.value
        if ast.unparse(call.func) in {"exit", "sys.exit"} and call.args:
            return call.args[0]
    if isinstance(last, ast.Raise) and isinstance(last.exc, ast.Call):
        raise_call = last.exc
        if ast.unparse(raise_call.func) == "SystemExit" and raise_call.args:
            return raise_call.args[0]
    return None


def _python_rebinds_linter(
    tree: ast.Module, imported_names: set[str], marker: str
) -> bool:
    """Return whether Python code rebinds the exit or linter entry points."""
    names = {"sys", "exit", "SystemExit"} | imported_names
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                return True
            if isinstance(target, ast.Attribute):
                text = ast.unparse(target)
                if text.startswith(("sys.", f"{marker}.")):
                    return True
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in names
        ):
            return True
    return False


def _python_exits_with_linter(
    tree: ast.Module, imported_names: set[str], marker: str
) -> bool:
    """Return whether Python code propagates a linter result to the exit status."""
    if _python_rebinds_linter(tree, imported_names, marker):
        return False
    argument = _python_exit_argument(tree)
    return argument is not None and _linter_result_argument(
        argument, imported_names, marker
    )


def _python_code_executes_marker(code: str, marker: str) -> bool:
    """Return whether Python code imports or calls the marker."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    imported_names: set[str] = set()
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if marker in alias.name:
                    imported_names.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if marker in (node.module or ""):
                imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    calls.append(child.id)
                elif isinstance(child, ast.Attribute):
                    calls.append(ast.unparse(child))
    if marker == "importlinter.cli":
        return _python_exits_with_linter(tree, imported_names, marker)
    if any(marker in call for call in calls):
        return True
    return any(name in call for name in imported_names for call in calls)


def _module_marker_matches(marker: str, module: str) -> bool:
    return (
        marker == module or marker.startswith(module + " ") or f"-m {module}" in marker
    )


def _script_arguments_allowed(script: str, arguments: list[str]) -> bool:
    allowed: frozenset[str] = frozenset()
    positionals: frozenset[str] = frozenset()
    for suffix, options in _SCRIPT_OPTION_VALUES.items():
        if script.endswith(suffix):
            allowed = options
            break
    for suffix, values in _SCRIPT_POSITIONALS.items():
        if script.endswith(suffix):
            positionals = values
            break
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("-"):
            if token not in positionals:
                return False
            index += 1
            continue
        if token not in allowed:
            return False
        if token in _SCRIPT_FLAG_OPTIONS:
            index += 1
            continue
        if index + 1 >= len(arguments) or arguments[index + 1] in _TERMINAL_OPTIONS:
            return False
        index += 2
    return True


def _interpreter_script_matches(
    tail: list[str],
    normalized_markers: list[str],
    options_with_value: frozenset[str],
) -> bool:
    index = _skip_options(tail, 1, options_with_value)
    if index >= len(tail):
        return False
    script = _normalized_shell_text(tail[index])
    interpreter = _normalized_shell_text(tail[0])
    command_text = _normalized_shell_text(" ".join(tail))
    if not _script_arguments_allowed(script, tail[index + 1 :]):
        return False
    return all(marker in command_text for marker in normalized_markers) and any(
        marker.startswith(script)
        or (marker.startswith(interpreter + " ") and script in marker)
        for marker in normalized_markers
    )


def _python_executes_markers(tail: list[str], normalized_markers: list[str]) -> bool:
    if "-c" in tail:
        index = tail.index("-c")
        if _terminal_option_before(tail, index) or index + 1 >= len(tail):
            return False
        code = tail[index + 1]
        return all(
            _python_code_executes_marker(code, marker) for marker in normalized_markers
        )
    if "-m" in tail:
        index = tail.index("-m")
        if _terminal_option_before(tail, index) or index + 1 >= len(tail):
            return False
        if any(token in _TERMINAL_OPTIONS for token in tail[index + 2 :]):
            return False
        module = tail[index + 1]
        actual = tail[index + 1 :]
        expected_commands: list[list[str]] = []
        for marker in normalized_markers:
            if f"-m {module}" in marker:
                suffix = marker.split(f"-m {module}", 1)[1].strip()
                expected = module + (f" {suffix}" if suffix else "")
                expected_commands.append(expected.split())
            elif marker.split()[0] == module:
                expected_commands.append(marker.split())
        if module == "pytest":
            return actual in expected_commands
        if module == "importlinter.cli":
            return False
        if module in {"mypy", "ruff"}:
            return any(
                actual[: len(expected)] == expected
                and all(
                    not token.startswith(("-", "@"))
                    for token in actual[len(expected) :]
                )
                for expected in expected_commands
            )
        if module in {"build", "twine"}:
            return any(
                actual[: len(expected)] == expected for expected in expected_commands
            ) and all(
                _module_marker_matches(marker, module) for marker in normalized_markers
            )
        return all(
            _module_marker_matches(marker, module) for marker in normalized_markers
        )
    options_with_value = frozenset({"-W", "-X", "-Q"})
    script_index = _skip_options(tail, 1, options_with_value)
    if _terminal_option_before(tail, script_index):
        return False
    return _interpreter_script_matches(tail, normalized_markers, options_with_value)


def _script_interpreter_executes(
    tail: list[str], normalized_markers: list[str]
) -> bool:
    if any(option in tail for option in ("-e", "-E", "--eval")):
        return False
    script_index = _skip_options(tail, 1, frozenset())
    if _terminal_option_before(tail, script_index):
        return False
    return _interpreter_script_matches(tail, normalized_markers, frozenset())


def _marker_is_executed(text: str, markers: tuple[str, ...]) -> bool:
    """Return whether the marker appears where a command would execute it."""
    tokens = _shell_tokens(text)
    first, tail = _effective_command(text)
    if first is None:
        return False
    if _ASSIGNMENT.match(text) and "$(" in text:
        return _assignment_runs_markers(text, markers)
    name = Path(first).name
    if name == "eval":
        return _requirement_executed(_logical_commands(" ".join(tokens[1:])), markers)
    if name in _SHELL_INTERPRETERS:
        if "-c" not in tail:
            return False
        index = tail.index("-c")
        if _terminal_option_before(tail, index) or index + 1 >= len(tail):
            return False
        return _requirement_executed(_logical_commands(tail[index + 1]), markers)
    normalized_markers = [_normalized_shell_text(marker) for marker in markers]
    if _is_python_interpreter(name):
        return _python_executes_markers(tail, normalized_markers)
    if name in _SCRIPT_INTERPRETERS:
        return _script_interpreter_executes(tail, normalized_markers)
    if any(token in _TERMINAL_OPTIONS for token in tail[1:]):
        return False
    command_text = _normalized_shell_text(" ".join(tail))
    if not all(marker in command_text for marker in normalized_markers):
        return False
    normalized_tail = [_normalized_shell_text(token) for token in tail]
    for marker in normalized_markers:
        expected = marker.split()
        if not expected or normalized_tail[: len(expected)] != expected:
            continue
        if all(
            not token.startswith(("-", "@"))
            for token in normalized_tail[len(expected) :]
        ):
            return True
    return False


def _assignment_runs_markers(text: str, markers: tuple[str, ...]) -> bool:
    """Return whether an assignment runs the marker in a command substitution."""
    return any(
        all(marker in body for marker in markers)
        and _segment_can_run(body.strip(), None, None, False, False)
        and _marker_is_executed(body.strip(), markers)
        for body in _command_substitutions(text)
    )


def _standalone_segment(segments: list[tuple[str, str | None]], index: int) -> bool:
    """Return whether a segment is not chained to another command."""
    if segments[index][1] is not None:
        return False
    return all(
        not text and connector == ";" for text, connector in segments[index + 1 :]
    )


def _requirement_executed(
    commands: list[tuple[str, bool, bool]], markers: tuple[str, ...]
) -> bool:
    blocked = False
    for command, dead, conditional in commands:
        segments = _segments(command)
        for index, (text, connector) in enumerate(segments):
            if not text:
                continue
            first = _effective_first_token(text)
            if (
                not dead
                and connector not in {"||", "&&", "|", "&"}
                and first in {"false", "exit", "return"}
            ):
                blocked = True
            if conditional or not all(marker in text for marker in markers):
                continue
            next_connector = (
                segments[index + 1][1] if index + 1 < len(segments) else None
            )
            if not _standalone_segment(segments, index):
                continue
            if not _segment_can_run(text, connector, next_connector, dead, blocked):
                continue
            if _marker_is_executed(text, markers):
                return True
    return False


def _errexit_effective_command(text: str) -> tuple[str | None, list[str]]:
    """Unwrap only prefixes that can run a builtin in the current shell."""
    tokens = _shell_tokens(text)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "command":
            if _command_lookup_option(tokens, index + 1):
                return None, []
            index = _skip_options(tokens, index + 1, frozenset())
            continue
        if token == "builtin":
            index += 1
            continue
        if token == "time":
            index += 2 if index + 1 < len(tokens) and tokens[index + 1] == "-p" else 1
            continue
        return token, tokens[index:]
    return None, []


def _errexit_matches(
    text: str,
    pattern: re.Pattern[str],
    *,
    recurse_shell: bool = True,
) -> bool:
    """Match an errexit change after unwrapping current-shell prefixes."""
    first, tail = _errexit_effective_command(text)
    if first is None:
        return False
    name = Path(first).name
    if name == "eval":
        body = " ".join(tail[1:])
        for command, _dead, _conditional in _logical_commands(body):
            segments = _segments(command)
            for segment, connector in segments:
                if pattern is _ERREXIT and connector not in {None, ";"}:
                    continue
                if _errexit_matches(segment, pattern, recurse_shell=recurse_shell):
                    return True
        return False
    if name in _SHELL_INTERPRETERS and "-c" in tail:
        if not recurse_shell:
            return False
        index = tail.index("-c")
        if _terminal_option_before(tail, index):
            return False
        return index + 1 < len(tail) and _errexit_matches(tail[index + 1], pattern)
    return bool(pattern.match(" ".join(tail)))


def _function_token_name(token: str) -> str | None:
    match = re.match(r"([^\s(){;]+)\s*\(\s*\)", token)
    return match.group(1) if match is not None else None


def _function_declaration_name(tokens: list[str], index: int) -> str | None:
    name = _function_token_name(tokens[index])
    if name is not None:
        return name
    if index + 1 < len(tokens) and tokens[index + 1] == "()":
        return tokens[index]
    if (
        index + 2 < len(tokens)
        and tokens[index + 1] == "("
        and tokens[index + 2] == ")"
    ):
        return tokens[index]
    return None


def _shadowing_names(commands: list[tuple[str, bool, bool]]) -> set[str]:
    """Find functions, aliases and hash overrides that can shadow commands."""
    names: set[str] = set()
    for command, _dead, _conditional in commands:
        tokens = _shell_tokens(command)
        for index, token in enumerate(tokens):
            name = _function_declaration_name(tokens, index)
            if name is not None:
                names.add(name)
            if token == "function" and index + 1 < len(tokens):
                match = re.match(r"([^\s(){;]+)", tokens[index + 1])
                if match is not None:
                    names.add(match.group(1))
            if token == "alias":
                cursor = index + 1
                while cursor < len(tokens) and tokens[cursor].startswith("-"):
                    cursor += 1
                if cursor < len(tokens) and "=" in tokens[cursor]:
                    names.add(tokens[cursor].split("=", 1)[0])
            if token == "hash" and index + 1 < len(tokens):
                option = tokens[index + 1]
                if option == "-p":
                    cursor = index + 3
                    if cursor < len(tokens) and tokens[cursor] == "--":
                        cursor += 1
                    if cursor < len(tokens):
                        names.add(tokens[cursor])
                elif option.startswith("-p") and len(option) > 2:
                    names.add(tokens[index + 2])
            if token == "eval" and index + 1 < len(tokens):
                body = " ".join(tokens[index + 1 :])
                names.update(_shadowing_names(_logical_commands(body)))
    return names


_GATE_ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "BASH_ENV",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
)


_COMMAND_PREFIX_KEYWORDS = frozenset(
    {
        "then",
        "do",
        "else",
        "elif",
        "if",
        "while",
        "until",
        "for",
        "select",
        "case",
        "{",
        "(",
        "!",
    }
)


def _command_word(part: str) -> str | None:
    """Return the command word after assignment, keyword and wrapper prefixes."""
    tokens = _shell_tokens(part)
    index = 0
    while index < len(tokens) and (
        _ASSIGNMENT.match(tokens[index]) or tokens[index] in _COMMAND_PREFIX_KEYWORDS
    ):
        index += 1
    if index >= len(tokens):
        return None
    return _effective_first_token(" ".join(tokens[index:]))


def _forbidden_constructs(command: str) -> set[str]:
    issues: set[str] = set()
    for part, _connector in _segments(command):
        if _dynamic_command_word(part):
            issues.add("dynamic_command")
        if _command_word(part) in {"source", "."}:
            issues.add("source")
        tokens = _shell_tokens(part)
        for index, token in enumerate(tokens):
            following = tokens[index + 1 :]
            if token == "set" and (
                following[:1] == ["+e"]
                or following[:2] == ["+o", "errexit"]
                or (following and any(_dynamic_token(arg) for arg in following))
            ):
                issues.add("errexit")
            if token == "trap":
                issues.add("trap")
            if token == "shopt" and _shopt_disables_errexit(following):
                issues.add("errexit")
            if token == "enable" and following[:1] == ["-n"]:
                issues.add("enable")
            if token == "eval" and following:
                body = " ".join(following)
                if _command_substitutions(body) or "`" in body:
                    issues.add("dynamic_eval")
                for inner, _dead, _conditional in _logical_commands(body):
                    issues.update(_forbidden_constructs(inner))
    return issues


def _shopt_disables_errexit(following: list[str]) -> bool:
    """Return whether ``shopt`` options disable the ``errexit`` shell option."""
    options = [
        option for option in following if option.startswith("-") and option != "-o"
    ]
    disabling = any(
        not option.startswith("--") and "u" in option[1:] for option in options
    )
    if not disabling or "-o" not in following:
        return False
    target = following[following.index("-o") + 1 :]
    return bool(target) and (target[0] == "errexit" or _dynamic_token(target[0]))


_REDIRECTION_TOKEN = re.compile(r"^(?:\d+|\{[^}]+\})?(?:[<>]|&>)")
_REDIRECTION_OPERATORS = frozenset(
    {">", ">>", "<", "<<", "<<<", "&>", "&>>", "<>", ">|"}
)


def _exec_has_command(arguments: list[str]) -> bool:
    """Return whether ``exec`` arguments contain a command, not redirects."""
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token in _REDIRECTION_OPERATORS:
            index += 2
            continue
        if _REDIRECTION_TOKEN.match(token):
            index += 1
            continue
        return True
    return False


def _has_early_termination(command: str) -> bool:
    for part, _connector in _segments(command):
        tokens = _shell_tokens(part)
        for index, token in enumerate(tokens):
            following = tokens[index + 1 :]
            if token in {"exit", "return"} and following:
                return True
            if token == "exec" and _exec_has_command(following):
                return True
            if token == "eval" and following:
                body = " ".join(following)
                if any(
                    _has_early_termination(inner)
                    for inner, _dead, _conditional in _logical_commands(body)
                ):
                    return True
    return False


def _has_gate_env_override(command: str) -> bool:
    tokens = _shell_tokens(command)
    if any(token == key for token in tokens for key in _GATE_ENV_KEYS) and any(
        token
        in {"export", "declare", "typeset", "readonly", "read", "mapfile", "readarray"}
        for token in tokens
    ):
        return True
    for token in tokens:
        assignment_key, separator, _value = token.partition("=")
        if separator and ("$" in assignment_key or "`" in assignment_key):
            return True
        if any(token.startswith((f"{key}=", f"{key}+=")) for key in _GATE_ENV_KEYS):
            return True
    if any(
        token
        in {"export", "declare", "typeset", "readonly", "read", "mapfile", "readarray"}
        for token in tokens
    ) and any(("$" in token or "`" in token) and "=" not in token for token in tokens):
        return True
    if re.search(
        r"(?:^|[\s;|&(])(?:" + "|".join(_GATE_ENV_KEYS) + r")(?:\+)?=",
        command,
    ):
        return True
    for index, token in enumerate(tokens):
        if token == "printf" and index + 2 < len(tokens) and tokens[index + 1] == "-v":
            target = tokens[index + 2]
            if target in _GATE_ENV_KEYS or "$" in target or "`" in target:
                return True
        if token == "eval" and index + 1 < len(tokens):
            body = " ".join(tokens[index + 1 :])
            if any(
                _has_gate_env_override(inner)
                for inner, _dead, _conditional in _logical_commands(body)
            ):
                return True
    return False


_FORBIDDEN_CONSTRUCT_MESSAGES = {
    "errexit": "disables error handling with 'set +e' or 'set +o errexit'",
    "trap": "uses a trap that can mask gate failures",
    "source": "sources another script, which can shadow required commands",
    "enable": "disables a shell builtin",
    "dynamic_eval": "evaluates a command substitution with 'eval'",
    "dynamic_command": ("runs a command name produced by a variable or substitution"),
}


def _shadowing_violations(
    path: Path, cube_name: str, commands: list[tuple[str, bool, bool]]
) -> list[Violation]:
    """Return violations for every declaration that can shadow a command."""
    return [
        Violation(
            path,
            1,
            f"cube {cube_name!r} declares a function, alias or hash that "
            f"shadows required command or guard builtin {name!r}",
        )
        for name in sorted(_shadowing_names(commands))
    ]


def _last_marker_index(
    commands: list[tuple[str, bool, bool]],
    markers: tuple[tuple[str, ...], ...],
) -> int:
    """Return the index of the last command that runs a required marker."""
    index = -1
    for command_index, (command, _dead, _conditional) in enumerate(commands):
        if any(all(marker in command for marker in group) for group in markers):
            index = command_index
    return index


def _shell_violations(
    path: Path,
    cube_name: str,
    script: str,
    markers: tuple[tuple[str, ...], ...],
) -> list[Violation]:
    commands = _logical_commands(script)
    violations = _shadowing_violations(path, cube_name, commands)
    last_marker_index = _last_marker_index(commands, markers)
    first_command: tuple[str, bool, bool] | None = None
    reported_constructs: set[str] = set()
    path_override = False
    for command_index, (command, dead, _conditional) in enumerate(commands):
        constructs = _forbidden_constructs(command)
        for construct, message in _FORBIDDEN_CONSTRUCT_MESSAGES.items():
            if construct in constructs and construct not in reported_constructs:
                reported_constructs.add(construct)
                violations.append(Violation(path, 1, f"cube {cube_name!r} {message}"))
        if not path_override and _has_gate_env_override(command):
            path_override = True
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {cube_name!r} overrides PATH or PYTHONPATH",
                )
            )
        if command_index < last_marker_index and _has_early_termination(command):
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {cube_name!r} can exit before required gates run",
                )
            )
        segments = _segments(command)
        for index, (text, _connector) in enumerate(segments):
            if text and first_command is None:
                first_command = (
                    text,
                    dead,
                    _standalone_segment(segments, index),
                )
            if "errexit" not in reported_constructs and _errexit_matches(
                text, _DISABLE_ERREXIT
            ):
                reported_constructs.add("errexit")
                violations.append(
                    Violation(
                        path,
                        1,
                        f"cube {cube_name!r} "
                        f"{_FORBIDDEN_CONSTRUCT_MESSAGES['errexit']}",
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
        if re.search(
            r"(?:^|[;|&({}])\s*(?:time\s+(?:-p\s+)?)?!\s+\S",
            command,
        ):
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
        or not first_command[2]
        or not _errexit_matches(first_command[0], _ERREXIT, recurse_shell=False)
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


def _env_entries(container: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return ``env`` entries in mapping, list and string forms."""
    raw = container.get("env")
    if isinstance(raw, dict):
        return [(str(key), value) for key, value in raw.items()]
    entries: list[tuple[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and "=" in item:
                key, _, value = item.partition("=")
                entries.append((key.strip(), value.strip()))
            elif isinstance(item, dict):
                entries.extend((str(key), value) for key, value in item.items())
    elif isinstance(raw, str) and "=" in raw:
        key, _, value = raw.partition("=")
        entries.append((key.strip(), value.strip()))
    return entries


def _env(container: dict[str, Any]) -> dict[str, Any]:
    return dict(_env_entries(container))


def _env_shape_violations(
    path: Path, container: dict[str, Any], label: str
) -> list[Violation]:
    """Reject an ``env`` that is not a mapping, as documented by SourceCraft."""
    if "env" not in container:
        return []
    raw = container.get("env")
    if raw is None or isinstance(raw, dict):
        return []
    return [
        Violation(
            path,
            1,
            f"{label} env must be a mapping, not {type(raw).__name__}",
        )
    ]


def _gate_env_key_violations(
    path: Path, container: dict[str, Any], label: str
) -> list[Violation]:
    violations: list[Violation] = []
    for key, _value in _env_entries(container):
        if key in _GATE_ENV_KEYS:
            violations.append(
                Violation(
                    path,
                    1,
                    f"{label} env must not set {key}",
                )
            )
    return violations


def _references_pat_secret(value: Any) -> bool:
    return isinstance(value, str) and PAT_SECRET_EXPRESSION.search(value) is not None


def _pat_cube_violations(
    path: Path, workflow_name: str, cube: dict[str, Any]
) -> list[Violation]:
    name = cube.get("name")
    if not isinstance(name, str):
        return []
    allowed = workflow_name == "publish-contracts" and name in PAT_SCOPED_CUBES
    label = f"cube {name!r} in workflow {workflow_name!r}"
    violations = _env_shape_violations(path, cube, label)
    violations.extend(_gate_env_key_violations(path, cube, label))
    for key, value in _env_entries(cube):
        references_pat = _references_pat_secret(value)
        if allowed:
            if key == PAT_NAME:
                if not references_pat:
                    violations.append(
                        Violation(
                            path,
                            1,
                            f"cube {name!r} does not read {PAT_NAME} from the "
                            "repository secret",
                        )
                    )
            elif references_pat:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"cube {name!r} must not receive {PAT_NAME} under "
                        f"environment key {key!r}; use only {PAT_NAME}",
                    )
                )
        elif key == PAT_NAME or references_pat:
            violations.append(
                Violation(
                    path,
                    1,
                    f"cube {name!r} in workflow {workflow_name!r} must not "
                    f"receive {PAT_NAME} under environment key {key!r}; it is "
                    "not a publishing cube",
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
    workflow_label = f"workflow {workflow_name!r}"
    violations = _env_shape_violations(path, workflow, workflow_label)
    violations.extend(_gate_env_key_violations(path, workflow, workflow_label))
    for key, value in _env_entries(workflow):
        if key == PAT_NAME or _references_pat_secret(value):
            violations.append(
                Violation(
                    path,
                    1,
                    f"{PAT_NAME} must not be defined at workflow level "
                    f"({workflow_name!r}) under environment key {key!r}; scope "
                    "it to the publishing cubes",
                )
            )
    for index, task in enumerate(_tasks(workflow, document)):
        task_name = _task_name(task, index)
        task_label = f"task {task_name!r}"
        violations.extend(_env_shape_violations(path, task, task_label))
        violations.extend(_gate_env_key_violations(path, task, task_label))
        for key, value in _env_entries(task):
            if key == PAT_NAME or _references_pat_secret(value):
                violations.append(
                    Violation(
                        path,
                        1,
                        f"task {task_name!r} must not receive {PAT_NAME} under "
                        f"environment key {key!r}; scope it to the publishing "
                        "cubes",
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
    violations = _env_shape_violations(path, document, "document")
    violations.extend(_gate_env_key_violations(path, document, "document"))
    for key, value in _env_entries(document):
        if key == PAT_NAME or _references_pat_secret(value):
            violations.append(
                Violation(
                    path,
                    1,
                    f"{PAT_NAME} must not be defined at document level under "
                    f"environment key {key!r}; scope it to the publish and "
                    "verify-from-registry cubes",
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
    commands: list[tuple[str, bool, bool]] = []
    violations: list[Violation] = []
    markers = tuple((marker,) for marker in REGULAR_CHECK_MARKERS)
    for cube_name, cube in cubes.items():
        script = _script(cube)
        commands.extend(_logical_commands(script))
        violations.extend(_shell_violations(path, cube_name, script, markers))
    violations.extend(
        Violation(
            path,
            1,
            f"quality-checks workflow has no executable {marker!r}",
        )
        for marker in REGULAR_CHECK_MARKERS
        if not _requirement_executed(commands, (marker,))
    )
    return violations


def _python_setup_violations(
    path: Path, workflows: dict[str, Any], document: dict[str, Any]
) -> list[Violation]:
    """Reject duplicated Python 3.11 installations in workflow tasks."""
    violations: list[Violation] = []
    for workflow_name, workflow in workflows.items():
        if not isinstance(workflow_name, str) or not isinstance(workflow, dict):
            continue
        for index, task in enumerate(_tasks(workflow, document)):
            task_name = _task_name(task, index)
            raw_cubes = task.get("cubes")
            if not isinstance(raw_cubes, list):
                continue
            cubes = {
                cube["name"]: cube
                for cube in raw_cubes
                if isinstance(cube, dict) and isinstance(cube.get("name"), str)
            }
            scripts = {
                cube_name: _uncommented(_script(cube))
                for cube_name, cube in cubes.items()
            }
            for cube_name, script in scripts.items():
                if _PYTHON311_INSTALL.search(script):
                    violations.append(
                        Violation(
                            path,
                            1,
                            f"cube {cube_name!r} in workflow {workflow_name!r} "
                            f"installs Python 3.11 inline; call "
                            f"{PYTHON_SETUP_SCRIPT} instead",
                        )
                    )
            uses_python = any("python3.11" in script for script in scripts.values())
            delegates = any(
                PYTHON_SETUP_SCRIPT in script for script in scripts.values()
            )
            if uses_python and not delegates:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"task {task_name!r} in workflow {workflow_name!r} "
                        f"installs Python 3.11 without {PYTHON_SETUP_SCRIPT}",
                    )
                )
    return violations


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
    violations.extend(_python_setup_violations(path, workflows, document))
    return violations


def _has_write_permission(
    document: dict[str, Any], job: dict[str, Any], permission: str
) -> bool:
    """Return whether a workflow grants ``permission: write`` to a job.

    A job-level ``permissions`` block replaces the workflow-level block
    entirely, so an unlisted scope there is ``none`` even when the workflow
    grants it. The ``write-all`` shorthand applies to the whole block.
    """
    job_permissions = job.get("permissions")
    if isinstance(job_permissions, dict):
        return job_permissions.get(permission) == "write"
    if job_permissions is not None:
        return isinstance(job_permissions, str) and job_permissions == "write-all"
    workflow_permissions = document.get("permissions")
    if isinstance(workflow_permissions, dict):
        return workflow_permissions.get(permission) == "write"
    return isinstance(workflow_permissions, str) and workflow_permissions == "write-all"


def _shell_commands(text: str) -> list[str]:
    """Join backslash continuations into complete shell commands."""
    commands: list[str] = []
    current: list[str] = []
    for raw_line in _uncommented(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("\\"):
            current.append(line[:-1].rstrip())
            continue
        current.append(line)
        commands.append(" ".join(current))
        current = []
    if current:
        commands.append(" ".join(current))
    return commands


def _verify_step_violations(
    path: Path,
    name: str,
    step: Any,
    markers: tuple[str, ...],
    install_markers: tuple[str, ...],
) -> list[Violation]:
    """Require a post-upload verification step with all of its markers.

    The isolation and index markers must be on the command that resolves
    ``configurable-automl-contracts==${VERSION}``, not merely somewhere in the step.
    """
    if not isinstance(step, dict):
        return [Violation(path, 1, f"GitHub release job has no {name!r} step")]
    run = step.get("run")
    if not isinstance(run, str):
        return [Violation(path, 1, f"step {name!r} has no run script")]
    clean = _uncommented(run)
    violations = [
        Violation(path, 1, f"step {name!r} has no {marker!r}")
        for marker in markers
        if marker not in clean
    ]
    install_commands = [
        command
        for command in _shell_commands(run)
        if GITHUB_PYPI_INSTALL_COMMAND in command
    ]
    if not install_commands:
        violations.append(
            Violation(
                path,
                1,
                f"step {name!r} has no install command for "
                f"{GITHUB_PYPI_INSTALL_COMMAND!r}",
            )
        )
        return violations
    for command in install_commands:
        for marker in install_markers:
            if marker not in command:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"install command in step {name!r} has no {marker!r}",
                    )
                )
    return violations


def _github_pypi_violations(
    path: Path,
    document: dict[str, Any],
    job: dict[str, Any],
    steps: list[Any],
) -> list[Violation]:
    """Require an OIDC PyPI publication and a post-upload consumer check."""
    violations: list[Violation] = []
    steps_by_name = {
        step["name"]: step
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }
    ordered_names = [
        step["name"]
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    ]
    publish_step = steps_by_name.get(GITHUB_PYPI_PUBLISH_STEP)
    if not isinstance(publish_step, dict):
        violations.append(
            Violation(
                path,
                1,
                f"GitHub release job has no {GITHUB_PYPI_PUBLISH_STEP!r} step",
            )
        )
    else:
        uses = publish_step.get("uses")
        if uses != GITHUB_PYPI_ACTION:
            violations.append(
                Violation(
                    path,
                    1,
                    f"step {GITHUB_PYPI_PUBLISH_STEP!r} must use {GITHUB_PYPI_ACTION}",
                )
            )
        options = publish_step.get("with")
        if not isinstance(options, dict):
            violations.append(
                Violation(
                    path,
                    1,
                    f"step {GITHUB_PYPI_PUBLISH_STEP!r} has no input mapping",
                )
            )
        else:
            unexpected = sorted(set(options) - GITHUB_PYPI_PUBLISH_INPUTS)
            if unexpected:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"step {GITHUB_PYPI_PUBLISH_STEP!r} has unsupported "
                        f"inputs {unexpected!r}",
                    )
                )
            if options.get("packages-dir") != GITHUB_PYPI_PACKAGES_DIR:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"step {GITHUB_PYPI_PUBLISH_STEP!r} must publish "
                        f"{GITHUB_PYPI_PACKAGES_DIR}",
                    )
                )
    violations.extend(
        _verify_step_violations(
            path,
            GITHUB_PYPI_VERIFY_STEP,
            steps_by_name.get(GITHUB_PYPI_VERIFY_STEP),
            GITHUB_PYPI_VERIFY_MARKERS,
            GITHUB_PYPI_INSTALL_MARKERS,
        )
    )
    violations.extend(
        _verify_step_violations(
            path,
            GITHUB_PYPI_SDIST_VERIFY_STEP,
            steps_by_name.get(GITHUB_PYPI_SDIST_VERIFY_STEP),
            GITHUB_PYPI_VERIFY_MARKERS,
            GITHUB_PYPI_SDIST_INSTALL_MARKERS,
        )
    )
    required_order = (
        *GITHUB_STEPS,
        GITHUB_PYPI_PUBLISH_STEP,
        GITHUB_PYPI_VERIFY_STEP,
        GITHUB_PYPI_SDIST_VERIFY_STEP,
    )
    present = [name for name in ordered_names if name in required_order]
    if present != list(required_order):
        violations.append(
            Violation(
                path,
                1,
                "GitHub release job must run the release gates, build, install, "
                "publish to PyPI and verify the published wheel and sdist in "
                "that order",
            )
        )
    if not _has_write_permission(document, job, "id-token"):
        violations.append(
            Violation(
                path,
                1,
                "GitHub release job needs id-token: write for PyPI Trusted Publishing",
            )
        )
    return violations


def check_github_publish(text: str, path: Path = GITHUB_PUBLISH) -> list[Violation]:
    """Check the public PyPI publication workflow for the contract.

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
    if not isinstance(job, dict):
        violations.append(Violation(path, 1, "GitHub release job has no steps"))
        return violations
    steps = job.get("steps")
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
    violations.extend(_github_pypi_violations(path, document, job, steps))
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


def _job_steps(job: Any) -> list[dict[str, Any]]:
    """Return the mapping steps of a GitHub Actions job."""
    if not isinstance(job, dict):
        return []
    steps = job.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _setup_python_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the ``setup-python`` steps among ``steps``."""
    return [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]


def _setup_python_version(step: dict[str, Any]) -> str | None:
    """Return the configured ``python-version`` of a setup step."""
    options = step.get("with")
    if not isinstance(options, dict):
        return None
    version = options.get("python-version")
    return None if version is None else str(version).strip()


def _check_main_python_version(jobs: dict[str, Any], path: Path) -> list[Violation]:
    """Require the main job to pin the oldest supported interpreter."""
    job = jobs.get("test")
    if not isinstance(job, dict):
        return [Violation(path, 1, "GitHub CI has no 'test' job")]
    setups = _setup_python_steps(_job_steps(job))
    if not setups:
        return [Violation(path, 1, "the main CI job does not set up Python")]
    step = setups[0]
    violations: list[Violation] = []
    if str(step.get("uses", "")) != SETUP_PYTHON_ACTION:
        violations.append(
            Violation(path, 1, f"the main CI job must use {SETUP_PYTHON_ACTION}")
        )
    if _setup_python_version(step) != SUPPORTED_PYTHON_VERSIONS[0]:
        violations.append(
            Violation(
                path,
                1,
                "the main CI job must pin python-version to "
                f'"{SUPPORTED_PYTHON_VERSIONS[0]}"',
            )
        )
    return violations


def _python_matrix_job(
    jobs: dict[str, Any],
) -> tuple[str, dict[str, Any], list[str]] | None:
    """Return the job whose matrix enumerates the supported interpreters."""
    expected = list(SUPPORTED_PYTHON_VERSIONS)
    for name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        strategy = job.get("strategy")
        matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
        versions = matrix.get("python-version") if isinstance(matrix, dict) else None
        if not isinstance(versions, list):
            continue
        normalized = [str(version).strip() for version in versions]
        if normalized == expected:
            return str(name), job, normalized
    return None


def _pytest_step(steps: list[dict[str, Any]], operator: str) -> dict[str, Any] | None:
    """Return the ``pytest`` step gated by a ``github.ref_type`` comparison."""
    for step in steps:
        condition = str(step.get("if", ""))
        if "github.ref_type" not in condition or operator not in condition:
            continue
        if "pytest" in str(step.get("run", "")):
            return step
    return None


def _runs_full_suite(step: dict[str, Any]) -> bool:
    """Tell whether a step invokes pytest without selecting test paths."""
    arguments = str(step.get("run", "")).split()
    if not arguments or arguments[0] != "pytest":
        return False
    return not any(
        argument == "tests" or argument.startswith("tests/")
        for argument in arguments[1:]
    )


def _check_python_matrix(jobs: dict[str, Any], path: Path) -> list[Violation]:
    """Require the interpreter matrix to split the fast and release suites."""
    expected = list(SUPPORTED_PYTHON_VERSIONS)
    found = _python_matrix_job(jobs)
    if found is None:
        return [
            Violation(
                path,
                1,
                f"GitHub CI has no Python version matrix covering {expected!r}",
            )
        ]
    name, job, _ = found
    steps = _job_steps(job)
    violations: list[Violation] = []
    setups = _setup_python_steps(steps)
    if not setups:
        violations.append(
            Violation(
                path, 1, f"the Python version matrix job {name!r} sets up no Python"
            )
        )
    else:
        step = setups[0]
        if str(step.get("uses", "")) != SETUP_PYTHON_ACTION:
            violations.append(
                Violation(
                    path,
                    1,
                    f"the Python version matrix job must use {SETUP_PYTHON_ACTION}",
                )
            )
        if _setup_python_version(step) != MATRIX_VERSION_EXPRESSION:
            violations.append(
                Violation(
                    path,
                    1,
                    "the Python version matrix job must set python-version to "
                    f'"{MATRIX_VERSION_EXPRESSION}"',
                )
            )
    fast = _pytest_step(steps, "!=")
    if fast is None:
        violations.append(
            Violation(
                path,
                1,
                "the Python version matrix has no fast test subset for pushes "
                "and pull requests",
            )
        )
    elif not any(
        argument.startswith("tests/") for argument in str(fast.get("run", "")).split()
    ):
        violations.append(
            Violation(
                path, 1, "the fast test subset of the Python version matrix is empty"
            )
        )
    full = _pytest_step(steps, "==")
    if full is None:
        violations.append(
            Violation(
                path,
                1,
                "the Python version matrix has no full test suite for releases",
            )
        )
    elif not _runs_full_suite(full):
        violations.append(
            Violation(
                path,
                1,
                "the release test step of the Python version matrix must run "
                "the full pytest suite",
            )
        )
    return violations


def check_github_ci_python_versions(
    text: str, path: Path = GITHUB_CI
) -> list[Violation]:
    """Check the supported-interpreter matrix and the release test scope.

    The main job is pinned to the oldest supported interpreter. A matrix job
    runs the fast test subset on every push and pull request, and a release
    tag runs the full suite on every supported version.

    Args:
        text: Content of ``.github/workflows/CI.yaml``.
        path: File path used in diagnostics.

    Returns:
        Every violated interpreter-matrix rule.
    """
    document = _load_yaml(path, text)
    if document is None:
        return [Violation(path, 1, "GitHub CI is not valid YAML")]
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return [Violation(path, 1, "GitHub CI defines no jobs")]
    return [
        *_check_main_python_version(jobs, path),
        *_check_python_matrix(jobs, path),
    ]


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _describe_match_globs(describe_command: Any) -> list[str] | None:
    """Return every ``--match`` glob of a ``git describe`` command.

    ``None`` means the command cannot be parsed safely, while an empty list
    means that a custom command was configured without any ``--match``.
    """
    if isinstance(describe_command, str):
        try:
            parts = shlex.split(describe_command)
        except ValueError:
            return None
    elif isinstance(describe_command, list):
        parts = [str(part) for part in describe_command]
    else:
        return None
    globs: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "--match":
            if index + 1 >= len(parts):
                return None
            globs.append(parts[index + 1])
            index += 2
            continue
        if part.startswith("--match="):
            value = part.partition("=")[2]
            if not value:
                return None
            globs.append(value)
        index += 1
    return globs


def _tag_filter_violations(match_globs: list[str], path: Path) -> list[Violation]:
    """Require the ``--match`` globs to select engine tags only."""
    violations: list[Violation] = []
    if not any(
        fnmatch.fnmatchcase(ENGINE_TAG, match_glob) for match_glob in match_globs
    ):
        violations.append(
            Violation(
                path,
                1,
                f"engine tag filters {match_globs!r} do not accept {ENGINE_TAG!r}",
            )
        )
    colliding = [
        match_glob
        for match_glob in match_globs
        if fnmatch.fnmatchcase(CONTRACT_TAG, match_glob)
    ]
    if colliding:
        violations.append(
            Violation(
                path,
                1,
                f"engine tag filters {colliding!r} accept contract tag "
                f"{CONTRACT_TAG!r}; a contract release tag would break the "
                "engine build",
            )
        )
    return violations


def _requirement_lower_bound(requirement: Requirement) -> Version | None:
    """Return the lowest version allowed by a requirement's lower-bound specs."""
    lower: Version | None = None
    for specifier in requirement.specifier:
        if specifier.operator not in (">=", ">", "==", "===", "~="):
            continue
        candidate = specifier.version.removesuffix(".*")
        try:
            bound = Version(candidate)
        except InvalidVersion:
            return None
        if lower is None or bound < lower:
            lower = bound
    return lower


def _scm_build_requirement_violations(
    document: dict[str, Any], path: Path
) -> list[Violation]:
    """Check that the build requirements support the configured tag table."""
    requires = _as_mapping(document.get("build-system")).get("requires")
    if not isinstance(requires, list):
        return [Violation(path, 1, "engine build-system.requires is missing")]
    scm_requirement: Requirement | None = None
    for raw in requires:
        if not isinstance(raw, str):
            continue
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as error:
            return [
                Violation(
                    path,
                    1,
                    f"engine build requirement {raw!r} is invalid: {error}",
                )
            ]
        if canonicalize_name(requirement.name) == "setuptools-scm":
            scm_requirement = requirement
            break
    if scm_requirement is None:
        return [
            Violation(
                path,
                1,
                "engine build requirements do not declare setuptools-scm",
            )
        ]
    lower_bound = _requirement_lower_bound(scm_requirement)
    if lower_bound is None or lower_bound < SETUPTOOLS_SCM_MIN_VERSION:
        return [
            Violation(
                path,
                1,
                "engine build requirements must require "
                f"setuptools-scm>={SETUPTOOLS_SCM_MIN_VERSION} so that the "
                "[tool.setuptools_scm.tag] configuration is supported; found "
                f"{scm_requirement!s}",
            )
        ]
    return []


def check_engine_version_tags(
    text: str, path: Path = ROOT_PYPROJECT
) -> list[Violation]:
    """Check that engine version inference ignores contract release tags.

    The engine and ``configurable-automl-contracts`` share one repository but have separate
    tag families. ``setuptools-scm`` must consider only ``v<version>`` engine
    tags: a ``contracts-v<version>`` tag at HEAD must neither be parsed as an
    engine version nor make the build fail. The build requirements must also
    declare a ``setuptools-scm`` version that supports the tag configuration.

    Args:
        text: Content of the root ``pyproject.toml``.
        path: File path used in diagnostics.

    Returns:
        Every violated release rule.
    """
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        return [Violation(path, 1, f"engine pyproject.toml is not valid TOML: {error}")]
    violations = _scm_build_requirement_violations(document, path)
    scm = _as_mapping(_as_mapping(document.get("tool")).get("setuptools_scm"))
    tag = _as_mapping(scm.get("tag"))
    prefix = tag.get("prefix")
    prefix = prefix if isinstance(prefix, str) else ""

    regex_text = tag.get("regex")
    if not isinstance(regex_text, str):
        legacy_regex = scm.get("tag_regex")
        regex_text = legacy_regex if isinstance(legacy_regex, str) else None
    if regex_text is None:
        violations.append(
            Violation(path, 1, "engine setuptools-scm config has no tag regex")
        )
    else:
        try:
            tag_regex = re.compile(regex_text)
        except re.error as error:
            violations.append(
                Violation(path, 1, f"engine tag regex is invalid: {error}")
            )
        else:
            if tag_regex.match(ENGINE_TAG.removeprefix(prefix)) is None:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"engine tag regex does not accept {ENGINE_TAG!r}",
                    )
                )
            if tag_regex.match(CONTRACT_TAG.removeprefix(prefix)) is not None:
                violations.append(
                    Violation(
                        path,
                        1,
                        f"engine tag regex accepts contract tag {CONTRACT_TAG!r}",
                    )
                )

    git = _as_mapping(_as_mapping(scm.get("scm")).get("git"))
    describe = git.get("describe_command")
    if describe is None:
        describe = scm.get("git_describe_command")
    if describe is None:
        violations.extend(_tag_filter_violations([f"{prefix}*[0-9]*"], path))
    else:
        match_globs = _describe_match_globs(describe)
        if match_globs is None:
            violations.append(
                Violation(
                    path,
                    1,
                    f"engine git describe command {describe!r} cannot be parsed",
                )
            )
        elif not match_globs:
            violations.append(
                Violation(
                    path,
                    1,
                    f"engine git describe command {describe!r} has no --match; "
                    "its tag filter cannot be proven safe",
                )
            )
        else:
            violations.extend(_tag_filter_violations(match_globs, path))
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
        "python": root / PYTHON_SETUP_SCRIPT,
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
    python_setup = scripts["python"]
    if python_setup.is_file():
        python_setup_text = python_setup.read_text(encoding="utf-8")
        for marker in (
            "Acquire::Retries",
            "add-apt-repository -y ppa:deadsnakes/ppa",
            "PYTHON311_SETUP_RETRY",
            "PYTHON311_SETUP_FAILED",
        ):
            if marker not in python_setup_text:
                violations.append(
                    Violation(python_setup, 1, f"Python setup script has no {marker!r}")
                )
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
    root_pyproject = root / "pyproject.toml"
    violations: list[Violation] = []
    if not root_pyproject.is_file():
        violations.append(
            Violation(root_pyproject, 1, "engine pyproject.toml is missing")
        )
    else:
        violations.extend(
            check_engine_version_tags(
                root_pyproject.read_text(encoding="utf-8"), root_pyproject
            )
        )
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
        github_ci_text = github_ci.read_text(encoding="utf-8")
        violations.extend(check_github_ci(github_ci_text, github_ci))
        violations.extend(check_github_ci_python_versions(github_ci_text, github_ci))
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
