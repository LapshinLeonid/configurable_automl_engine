---
name: docs-standards-guard
description: Pre-commit self-check of docstrings and comments against the repository documentation review checklist (no change history, no task codes, no duplicated types in Args:, short docstrings for private functions, "why" comments, compact module docstrings, English Google style). Run on every touched file in src/ and tests/ before committing.
---

# docs-standards-guard

Self-check pass on docstrings and comments before committing. Its purpose is to
catch the recurring documentation review findings that ruff and mypy cannot see
(docstrings and comments are prose, not parsed code), so they do not surface as
review rounds later.

When to run: after implementation and formatting are done, immediately before
`git commit`. Apply the checklist below to every touched file in `src/` and
`tests/` — both new files and edited fragments (read a few lines of context
around each edited hunk, not only the changed lines).

## Checklist

Go through each item for every touched file. All items must pass.

1. **No change-history wording.** Docstrings and comments describe current
   behavior only. Reject: "was", "previously", "no longer", "as before",
   "like before", "legacy", "removed", "used to", "instead of the old", and any
   other before/after framing — including Russian equivalents ("было",
   "раньше", "больше не"). If a docstring says what a parameter *used* to mean,
   rewrite it to state what it *means*.
2. **No task codes or issue references.** Reject "B3", "S3" and similar
   milestone markers, "issue #NNN", "the issue requirement", "per the task".
   State the rationale in plain words instead ("Keep the context alive so the
   callback can fire after the runner exits").
3. **No duplicated types in `Args:`.** Annotations are the single source of
   truth (mypy strict). Write `name: description`, never `name (str):
   description`. The same applies to `Returns:` and `Yields:`.
4. **Private functions keep a short docstring.** Every `_name` function gets a
   one-to-two-line docstring: what it does and why it exists. Full `Args:` /
   `Returns:` sections only when semantics are non-obvious — units, array
   shapes, metric sign, NaN handling, in-place vs copy. Do not add boilerplate
   sections that restate the signature.
5. **Comments answer "why", not "what".** A comment must explain a non-obvious
   decision: a numeric tolerance, a workaround for a library quirk, thread or
   process safety, a paper or spec reference, a performance trade-off. Delete a
   comment that restates the next line of code.
6. **Module docstrings stay compact.** Guideline: ~15 lines maximum, covering
   the module purpose and key invariants. User-facing behavior belongs in
   `API_REFERENCE.md`, not duplicated in code. Reject multi-page tutorials,
   exhaustive parameter tables, and usage walkthroughs inside the module.
7. **English, Google style.** All docstrings and comments are in English,
   Google format (`Args:`, `Returns:`, `Raises:` sections) for every public
   module, class, and function. Reject Russian text and reST/NumPy/EPYDOC
   markup.
8. **Complexity, not accretion.** When a touched block already carries several
   branches and helpers, do not bolt new intricate logic onto it — split the
   new block out instead (follows the "reduce complexity hot spots" direction
   from the architecture work). Flag such spots in the self-check even if the
   code itself is correct.

## Acceptance

Before committing, reread the full diff as a reviewer would, using this
checklist as the review rubric. For every finding, fix the text in place.

Where behavior has changed, describe the *actual current state* ("Deduplicates
candidates against the fitted selector") — never the delta ("Now also
deduplicates, previously it did not"). If you cannot phrase a docstring without
history wording, the text is describing the wrong thing: describe what the code
does today.

Only commit when every checklist item passes for every touched file.
