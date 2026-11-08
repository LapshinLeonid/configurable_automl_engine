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

From the repository root, with the environment matching the project's quality
baseline (Python 3.11+, dependencies installed via `pip install -e ".[all,test]"`):

```bash
ruff format src/
ruff check src/
ruff format --check src/
lint-imports
mypy src --install-types --non-interactive
pytest --cov=src
```

All commands must exit successfully. A failure anywhere is a blocker: fix and
restart the whole suite. Do not interpret, silence, or `xfail` a failing check
to make the suite green — that is exactly the kind of unverified claim this
skill exists to prevent.

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

If the project has a related skill for test design (for example,
`meaningful-test-design`), follow it when writing and rewriting the test.

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

Re-run the check suite (item 1) and the tests relevant to the change on a
**clean checkout of the PR branch** — the way the reviewer will — not in a
working directory with uncommitted leftovers, stale artifacts, or local
tweaks:

```bash
git worktree add /tmp/pr-verify-$(date +%s) HEAD
# run the suite inside the fresh worktree, then:
# git worktree remove <path>
```

If verification only passes in the dirty working directory, the PR is not
done: find what the working directory was compensating for and make the
branch self-contained.

### 6. Re-check the diff against the description

```bash
git diff main...HEAD --stat
git diff main...HEAD
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
- [ ] `git diff main...HEAD` matches the PR description (`src pr edit` applied
      if not).

Then — and only then — submit for review (`wait_review` / review request). If
any item fails, fix, commit, and start this checklist over from item 1 on the
new commit.
