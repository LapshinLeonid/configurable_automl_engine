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
SOURCE_CUBE_MARKERS = {
    "preflight": ("scripts/contract_release.py check-tag",),
    "quality-gates": (
        "scripts/check_contract_boundaries.py",
        "python3.11 scripts/check_release_configuration.py",
        "python3.11 -m ruff check src/ automl-contracts/src/",
        "python3.11 -m mypy scripts/contract_release.py",
        "importlinter.cli",
        "pytest tests/contracts",
    ),
    "build-and-verify": (
        "python3.11 -m build automl-contracts",
        "twine check",
        "scripts/verify_contract_distribution.py",
        "scripts/smoke_contract_install.py",
    ),
    "publish": ("scripts/contract_release.py publish",),
    "verify-from-registry": (
        "scripts/contract_release.py install",
        '"$REPO_DIR/scripts/e2e_contract_consumer.py"',
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


def _entries(on_data: dict[str, Any], trigger: str) -> list[dict[str, Any]]:
    raw = on_data.get(trigger, [])
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            item = {"workflows": [item]}
        if isinstance(item, dict):
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
    return _names(entry.get("workflows"))


def _has_tag_entry(
    push_entries: list[dict[str, Any]], workflow: str, *, only_entry: bool
) -> bool:
    matching = [entry for entry in push_entries if workflow in _workflow_names(entry)]
    if only_entry and len(matching) != 1:
        return False
    for entry in matching:
        tags = _filter(entry, "tags")
        branches = _filter(entry, "branches")
        paths = _filter(entry, "paths")
        if tags == ["contracts-v*"] and not branches and not paths:
            return True
    return False


def _has_trigger_workflow(entries: list[dict[str, Any]], workflow: str) -> bool:
    return any(workflow in _workflow_names(entry) for entry in entries)


def _cubes(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    tasks = workflow.get("tasks")
    if not isinstance(tasks, list):
        return result
    for task in tasks:
        if not isinstance(task, dict):
            continue
        raw_cubes = task.get("cubes")
        if not isinstance(raw_cubes, list):
            continue
        for cube in raw_cubes:
            if isinstance(cube, dict) and isinstance(cube.get("name"), str):
                result[cube["name"]] = cube
    return result


def _script(cube: dict[str, Any]) -> str:
    raw = cube.get("script")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(str(item) for item in raw)
    return ""


def _check_cube(
    path: Path, cube_name: str, cube: dict[str, Any], markers: tuple[str, ...]
) -> list[Violation]:
    text = _uncommented(_script(cube))
    return [
        Violation(path, 1, f"cube {cube_name!r} has no {marker!r}")
        for marker in markers
        if marker not in text
    ]


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


def _trigger_violations(
    path: Path, text: str, document: dict[str, Any]
) -> list[Violation]:
    on_data = _on(document)
    violations: list[Violation] = []
    if not _has_trigger_workflow(_entries(on_data, "pull_request"), "quality-checks"):
        violations.append(Violation(path, 1, "pull requests do not run quality-checks"))
    push_entries = _entries(on_data, "push")
    if not _has_tag_entry(push_entries, "publish-contracts", only_entry=True):
        violations.append(
            Violation(
                path,
                1,
                "publish-contracts must have exactly one push entry with only "
                "the contracts-v* tag filter",
            )
        )
    if not _has_tag_entry(push_entries, "sync-to-github", only_entry=False):
        violations.append(
            Violation(
                path,
                1,
                "sync-to-github has no push entry with the contracts-v* tag filter",
            )
        )
    if "refs/tags/" not in _uncommented(text):
        violations.append(
            Violation(path, 1, "the GitHub sync workflow never pushes a tag ref")
        )
    return violations


def _publish_violations(
    path: Path, text: str, workflows: dict[str, Any]
) -> list[Violation]:
    violations: list[Violation] = []
    publish = workflows.get("publish-contracts")
    if not isinstance(publish, dict):
        return [
            Violation(path, 1, "CI configuration has no publish-contracts workflow")
        ]
    cubes = _cubes(publish)
    actual_order = [name for name in cubes if name in SOURCE_CUBES]
    for name in SOURCE_CUBES:
        if name not in cubes:
            violations.append(
                Violation(path, 1, f"publish-contracts has no {name!r} cube")
            )
        else:
            violations.extend(
                _check_cube(path, name, cubes[name], SOURCE_CUBE_MARKERS[name])
            )
    violations.extend(
        _order_violation(path, SOURCE_CUBES, actual_order, "publish-contracts")
    )
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
    env = publish.get("env")
    keys = env if isinstance(env, dict) else {}
    if "SOURCECRAFT_PYPI_REGISTRY_ID" not in keys:
        violations.append(
            Violation(path, 1, "publish-contracts has no SOURCECRAFT_PYPI_REGISTRY_ID")
        )
    if "PYPI_REGISTRY_PAT" not in keys:
        violations.append(
            Violation(path, 1, "publish-contracts has no PYPI_REGISTRY_PAT")
        )
    elif keys.get("PYPI_REGISTRY_PAT") != "${{ secrets.PYPI_REGISTRY_PAT }}":
        violations.append(
            Violation(
                path,
                1,
                "publish-contracts does not read PYPI_REGISTRY_PAT from a secret",
            )
        )
    violations.extend(_check_uploader(path, text))
    return violations


def _quality_violations(path: Path, workflows: dict[str, Any]) -> list[Violation]:
    quality = workflows.get("quality-checks")
    if not isinstance(quality, dict):
        return [Violation(path, 1, "CI configuration has no quality-checks workflow")]
    cubes = _cubes(quality)
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
    violations.extend(_publish_violations(path, text, workflows))
    violations.extend(_credential_violations(path, lines))
    violations.extend(_secret_violations(path, lines))
    violations.extend(_quality_violations(path, workflows))
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
