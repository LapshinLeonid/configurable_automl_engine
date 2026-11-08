---
name: pre-submit-verification
description: "Final \"reviewer's-eye\" verification of a PR candidate before submission. Run before creating a PR and before every push to an open PR branch: full local quality gates in a clean environment, mutation checks of new tests, factual verification of every claim in the PR description, stability checks for stochastic gates, and a fresh-checkout re-run. Submit for review only when every item passes."
---

# Pre-submit verification

## Purpose

Verify a PR candidate the way the reviewer will verify it — **before** the PR
is created or pushed. Blocking review findings almost always trace back to one
root cause: an agent ships a PR with unverified claims (a test that does not
actually catch the claimed regression, a gate profile that fails on the very
edits it was proposed for). The reviewer then has to spend the review proving
the text wrong instead of reviewing the change.

Use this skill as the final step after the work itself is done:

- **before creating a new pull request**, and
- **before every push to the branch of an already-open pull request**.

The rule: every statement that will be read as a fact by the reviewer must have
been established by an actual run, not by expectation. Only when all items
below pass is the PR ready for review (`wait_review` / review request).

## Instructions

Run all items, in order, each time. Do not skip items because a previous run
passed — any new commit invalidates all previous verification.

### 1. Full local check suite in a clean environment

"A clean environment" means: the same fresh-worktree recipe as in item 5, with
its own virtual environment — not the current working directory, whose
uncommitted leftovers, stale artifacts, and editable install may be silently
compensating for problems in the branch.

Create a fresh worktree of the branch and a fresh environment in it (full
recipe in item 5):

```bash
WORKTREE=$(mktemp -d /tmp/pr-verify-XXXXXX)
git worktree add "$WORKTREE" HEAD
python3.11 -m venv "$WORKTREE/.venv"
. "$WORKTREE/.venv/bin/activate"
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" pip install -e ".[all,test]"
python -c "import configurable_automl_engine; print(configurable_automl_engine.__file__)"
# The printed path must be inside "$WORKTREE".
```

Then, from the worktree root with that environment activated:

```bash
ruff format --check src/
ruff check src/
lint-imports
mypy src --install-types --non-interactive
pytest --cov=src
```

All commands must exit successfully. A failure anywhere is a blocker: fix and
restart the whole suite. Do not interpret, silence, or `xfail` a failing check
to make the suite green — that is exactly the kind of unverified claim this
skill exists to prevent.

Never run bare `ruff format src/` (or any other tree-mutating command) during
verification: if `ruff format --check` reports a diff, treat it as a failure —
auto-format, commit, and restart the checklist. The checked tree must stay
identical to the committed branch; auto-formatting belongs to the fix phase,
not the verification phase.

### 2. Mutation check for every new test

For each test added in this PR, prove it actually catches the defect it claims
to catch:

- Reconstruct the **pre-fix** behavior (revert the production change under
  test, or check out the base commit).
- Run the new test on that version: **it must fail**.
- Restore the fix and run the test again: **it must pass**.

A new test that passes both before and after the fix does not test anything;
rewrite it before submitting. The same applies to any claim of the form "this
test guards against a regression" — the guard is proven by the mutation run,
not by reading the test.

Check first whether a test-design skill exists in `skills/` at the time of the
run (for example, `meaningful-test-design`, which may not exist yet). If it
exists, follow it when writing and rewriting the test; if it does not, apply
the rule above — the mutation run is the only proof that the test guarantees
anything.

### 3. Verify every claim in the PR description

Go through the PR description statement by statement. Every claim that can be
checked by running something must be checked by running it:

- "tests failed on main / on the base commit" — actually run the relevant
  tests on the base commit and confirm the failure;
- "this test fixes the regression" — actually run the mutation check from
  item 2;
- "the gate profile passes" — actually run the benchmark/gate on the final
  branch state, not on an earlier draft;
- "quality checks pass", "coverage is X", "benchmark improved by Y" — take the
  numbers from the final run's actual output, never from memory or an
  intermediate run.

If any claim cannot be verified by a run, remove it or rephrase it as
expectation (explicitly marked as such) instead of leaving it as fact.

### 4. Stability of stochastic gates

If the PR adds or changes a benchmark, gate, or threshold that depends on
seeds, repeats, or sampling:

- Run the gate on **multiple seed windows** (several distinct seed values or
  seed ranges), not just one run.
- Record the observed spread of the metric across those windows.
- Choose the gate tolerance from the observed spread with a margin — do not
  fit the tolerance to a cherry-picked run. Known failure pattern: median
  per-seed gaps of ±0.068 against a tolerance of 0.02; the profile passed on
  the seeds it was tuned on and failed on the already-made edits.
- State in the PR description which windows were checked and what spread was
  observed.

A tolerance that holds only on the seeds it was tuned on is not verified.

### 5. Fresh checkout

Re-run the check suite (item 1) and the tests relevant to the change in the
**fresh worktree of the PR branch** you created in item 1 — the way the
reviewer will — not in a working directory with uncommitted leftovers, stale
artifacts, or local tweaks. If you skipped the worktree setup in item 1,
create it now using the recipe below:

```bash
WORKTREE=$(mktemp -d /tmp/pr-verify-XXXXXX)
git worktree add "$WORKTREE" HEAD

# Fresh environment: without this step pytest inside the worktree still
# imports the package through the editable install of the ORIGINAL checkout,
# which silently tests the wrong (possibly dirty) code — a false green.
python3.11 -m venv "$WORKTREE/.venv"
. "$WORKTREE/.venv/bin/activate"
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" pip install -e ".[all,test]"

# Prove the interpreter actually resolves the package from the worktree:
python -c "import configurable_automl_engine; print(configurable_automl_engine.__file__)"
# The printed path must be inside "$WORKTREE". If it is not, stop and fix the
# environment before running any checks — the results would be meaningless.

# run the suite inside the fresh worktree, then:
# git worktree remove <path>
```

If verification only passes in the dirty working directory, the PR is not
done: find what the working directory was compensating for and make the
branch self-contained.

### 6. Re-check the diff against the description

Derive the base branch from the PR's target branch, not a hardcoded default
(the PR may target a non-default branch):

```bash
# <target> is the target branch of this PR (for SourceCraft, "target_branch"
# from `src pr get <number> --json`); for the common case it is `main`:
git diff <target>...HEAD --stat
git diff <target>...HEAD
```

- Recalculate the diff and confirm every file/function it touches matches
  what the description says the PR does.
- If the description is stale (the change grew, tests were renamed, scope
  changed), update it before submission — for an open SourceCraft PR use
  `src pr edit <number> --title ... --description ...` so the description
  matches the fact, and leave a reviewer's notes untouched.

## Definition of done

The PR is ready for review **only when every item above has actually been run
and passed on the final commit of the branch**:

- [ ] full local check suite green in a clean environment;
- [ ] every new test fails pre-fix and passes post-fix;
- [ ] every claim in the PR description backed by an actual run;
- [ ] stochastic gates pass on multiple seed windows with a justified
      tolerance;
- [ ] checks green on a fresh checkout of the branch;
- [ ] `git diff <target>...HEAD` (where `<target>` is the PR's target branch)
      matches the PR description (`src pr edit` applied if not); the diff
      source of truth for an open SourceCraft PR is `src pr diff <number>`.

Then — and only then — submit for review (`wait_review` / review request). If
any item fails, fix, commit, and start this checklist over from item 1 on the
new commit.
