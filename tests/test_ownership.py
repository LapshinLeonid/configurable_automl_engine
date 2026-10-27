"""Ownership rule of the mutable state objects (a static AST check).

Only the owner mutates its state:

- ``_OptimizeRunner`` (``hpo/runner.py``) — only its own methods;
- ``ModelTrainer`` (``trainer.py``) — only its own methods;
- ``PhaseContext`` (``training_engine/state.py``) — only
  ``training_engine/component.py``.

Everywhere else in the package there is no attribute assignment, subscript
assignment, ``del`` or in-place mutating call (``append``, ``extend``,
``update``, ``pop``, ``clear``, ``add``, ``setdefault``, ``insert``,
``remove``) on such an object or on anything reached through its attributes,
and no ``setattr`` on it. The objects are recognized by the parameter
annotation (``runner: _OptimizeRunner``, ``trainer: ModelTrainer``,
``ctx: PhaseContext``) and by the names ``runner``, ``trainer`` and ``ctx``.

``ALLOWED_VIOLATIONS`` lists the functions that still break the rule; it may
only shrink. A violation that is not listed fails the test, and so does a
listed entry that no longer violates the rule (remove it from the list).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

import configurable_automl_engine

PACKAGE_ROOT = Path(configurable_automl_engine.__file__).parent

#: Tracked class → (owner module, owner class or ``None`` for the whole module).
OWNERS: dict[str, tuple[str, str | None]] = {
    "_OptimizeRunner": ("hpo/runner.py", "_OptimizeRunner"),
    "ModelTrainer": ("trainer.py", "ModelTrainer"),
    "PhaseContext": ("training_engine/component.py", None),
}

#: Conventional variable name → tracked class.
NAMES: dict[str, str] = {
    "runner": "_OptimizeRunner",
    "trainer": "ModelTrainer",
    "ctx": "PhaseContext",
}

MUTATING_METHODS = frozenset(
    {
        "append",
        "extend",
        "update",
        "pop",
        "clear",
        "add",
        "setdefault",
        "insert",
        "remove",
    }
)

#: (module path, qualified function name, tracked class) of the functions
#: that still mutate an object they do not own. May only shrink.
ALLOWED_VIOLATIONS: set[tuple[str, str, str]] = {
    ("trainer_data.py", "_apply_input_diagnostics", "ModelTrainer"),
    ("trainer_data.py", "_build_preprocessor", "ModelTrainer"),
    ("trainer_data.py", "_detect_feature_types", "ModelTrainer"),
    ("trainer_data.py", "_drop_uninformative_features", "ModelTrainer"),
    ("trainer_data.py", "_prepare_data", "ModelTrainer"),
    ("trainer_data.py", "_prepare_training_features", "ModelTrainer"),
    ("trainer_data.py", "_reset_fit_state", "ModelTrainer"),
    ("trainer_scoring.py", "_finalize_oof", "ModelTrainer"),
    ("trainer_scoring.py", "_score_on_validation_splits", "ModelTrainer"),
    ("training_engine/hpo_phases.py", "_apply_target_domain_guard", "PhaseContext"),
    ("training_engine/hpo_phases.py", "_consume_worker_results", "PhaseContext"),
    (
        "training_engine/select_steps.py",
        "_drop_kept_from_circuit_breaker",
        "PhaseContext",
    ),
    ("training_engine/select_steps.py", "_execute_select_cv_best", "PhaseContext"),
    (
        "training_engine/select_steps.py",
        "_execute_select_sanity_gate",
        "PhaseContext",
    ),
}


@dataclass(frozen=True)
class Violation:
    """One function that mutates an object it does not own."""

    module: str
    function: str
    tracked: str
    line: int


def _root_name(node: ast.AST) -> str | None:
    """The name at the root of an ``a.b[c].d`` chain (``None`` if not a name)."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _annotated_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Parameters annotated with a tracked class: name → tracked class."""
    tracked: dict[str, str] = {}
    args = func.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        if arg.annotation is None:
            continue
        words = set(re.findall(r"\w+", ast.unparse(arg.annotation)))
        for cls in OWNERS:
            if cls in words:
                tracked[arg.arg] = cls
    return tracked


class _Scanner(ast.NodeVisitor):
    """Collect the mutations of tracked objects, function by function."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.classes: list[str] = []
        self.scopes: list[dict[str, str]] = [{}]
        self.found: list[Violation] = []

    # ── scopes ───────────────────────────────────────────────────────────
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()
        self.stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.scopes.append({**self.scopes[-1], **_annotated_params(node)})
        self.generic_visit(node)
        self.scopes.pop()
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # ── helpers ──────────────────────────────────────────────────────────
    def _tracked(self, name: str | None) -> str | None:
        if name is None:
            return None
        return self.scopes[-1].get(name) or NAMES.get(name)

    def _owned(self, tracked: str) -> bool:
        module, cls = OWNERS[tracked]
        if module != self.module:
            return False
        return cls is None or cls in self.classes

    def _report(self, tracked: str | None, node: ast.AST) -> None:
        if tracked is None or self._owned(tracked) or not self.stack:
            return
        self.found.append(
            Violation(
                self.module,
                ".".join(self.stack),
                tracked,
                getattr(node, "lineno", 0),
            )
        )

    def _check_target(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._check_target(elt)
            return
        if isinstance(target, ast.Starred):
            self._check_target(target.value)
            return
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            self._report(self._tracked(_root_name(target)), target)

    # ── mutations ────────────────────────────────────────────────────────
    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in MUTATING_METHODS
            and isinstance(func.value, (ast.Attribute, ast.Subscript))
        ):
            # ``ctx.oof_pool.append(...)``: a mutation of the object's state.
            self._report(self._tracked(_root_name(func.value)), node)
        if (
            isinstance(func, ast.Name)
            and func.id in {"setattr", "delattr"}
            and node.args
        ):
            self._report(self._tracked(_root_name(node.args[0])), node)
        self.generic_visit(node)


def _modules() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if p.name != "_version.py")


def find_violations() -> list[Violation]:
    """Scan the package for mutations of objects outside their owners."""
    found: list[Violation] = []
    for path in _modules():
        module = path.relative_to(PACKAGE_ROOT).as_posix()
        scanner = _Scanner(module)
        scanner.visit(ast.parse(path.read_text(encoding="utf-8")))
        found.extend(scanner.found)
    return found


def test_only_the_owner_mutates_its_state() -> None:
    """No function outside the owner mutates a runner / trainer / context."""
    found = find_violations()
    keys = {(v.module, v.function, v.tracked) for v in found}
    new = sorted(keys - ALLOWED_VIOLATIONS)
    details = [
        f"{v.module}:{v.line} {v.function} mutates a {v.tracked}"
        for v in found
        if (v.module, v.function, v.tracked) in set(new)
    ]
    assert not new, (
        "Only the owner may mutate its state (see agents.md); return a result "
        "and let the owner apply it:\n" + "\n".join(details)
    )
    stale = sorted(ALLOWED_VIOLATIONS - keys)
    assert not stale, f"Remove the fixed entries from ALLOWED_VIOLATIONS: {stale}"


# ── self-tests of the scanner ───────────────────────────────────────────────


def _scan(source: str, module: str = "some/module.py") -> set[tuple[str, str]]:
    scanner = _Scanner(module)
    scanner.visit(ast.parse(source))
    return {(v.function, v.tracked) for v in scanner.found}


def test_scanner_detects_every_mutation_form() -> None:
    """Attribute, subscript, augmented, del, setattr and mutating calls."""
    source = '''
def by_name(runner, trainer, ctx):
    runner.X = 1
    trainer.scores["r2"] = 0.5
    ctx.oof_pool.append(1)
    ctx.full_fit_trainers.clear()
    del ctx.disqualified_algorithms["a"]
    setattr(trainer, "x", 1)

def by_annotation(r: _OptimizeRunner, t: "ModelTrainer", c: PhaseContext | None):
    r.n += 1
    t.a, t.b = 1, 2
    c.items.update({})

def nested(ctx):
    def inner():
        ctx.winner_algo = "x"
    return inner
'''
    assert _scan(source) == {
        ("by_name", "_OptimizeRunner"),
        ("by_name", "ModelTrainer"),
        ("by_name", "PhaseContext"),
        ("by_annotation", "_OptimizeRunner"),
        ("by_annotation", "ModelTrainer"),
        ("by_annotation", "PhaseContext"),
        ("nested.inner", "PhaseContext"),
    }


def test_scanner_ignores_reads_and_owner_writes() -> None:
    """Reads, rebinding a name and the owner's own writes are allowed."""
    reads = '''
def reads(runner, trainer, ctx):
    x = runner.X
    trainer.fit(x, x)
    items = list(ctx.working_set)
    runner = None
    other = {}
    other["a"] = 1
'''
    assert _scan(reads) == set()
    owner = '''
class _OptimizeRunner:
    def run(self, runner):
        runner.X = 1
'''
    assert _scan(owner, "hpo/runner.py") == set()
    component = '''
def apply(ctx):
    ctx.working_set = {}
'''
    assert _scan(component, "training_engine/component.py") == set()


# ── the pure modules are decoupled from the owner classes ────────────────────

#: Modules of pure functions: they must not import the owner classes, not
#: even under ``TYPE_CHECKING``.
PURE_MODULES = (
    "hpo/search_space.py",
    "hpo/evaluation.py",
    "hpo/trial_pipeline.py",
    "hpo/pruning.py",
)

FORBIDDEN_IMPORTS = {"_OptimizeRunner", "ModelTrainer"}


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[-1] == "runner" and node.level >= 1:
                names.add("hpo.runner")
            if module.endswith("hpo.runner"):
                names.add("hpo.runner")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("module", PURE_MODULES)
def test_pure_modules_do_not_import_the_owner_classes(module: str) -> None:
    """The pure modules take explicit inputs, not the owner objects."""
    tree = ast.parse((PACKAGE_ROOT / module).read_text(encoding="utf-8"))
    imported = _imported_names(tree)
    assert not imported & (FORBIDDEN_IMPORTS | {"hpo.runner"}), (
        f"{module} imports {sorted(imported & (FORBIDDEN_IMPORTS | {'hpo.runner'}))}"
    )
