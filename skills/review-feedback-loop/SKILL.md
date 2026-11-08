---
name: review-feedback-loop
description: Discipline for processing code-review feedback on a pull request so that every finding from every review round is resolved, verified, and answered in a single iteration before pushing the next update.
---

# review-feedback-loop

## Purpose

Use this skill when a pull request receives review comments and a new iteration is being prepared. It enforces a closed feedback loop: every finding is either fixed with proof or explicitly declined with justification — nothing is silently dropped, half-fixed, or answered without verification. This prevents repeat `block` verdicts on the same points, partial fixes, and follow-up tasks filed after merge.

The trigger for this skill is any set of review findings on a PR assigned to the agent: a review round verdict (`block` / `approve`) with inline comments, individual review comments, or CI findings the reviewer forwarded.

## Step 1 — Collect the full findings list

1. Read **all** comments on the PR: every review round, every inline thread, and the review verdict summaries — not only the latest round. A finding from an earlier round that is not addressed in a later round is still open.
2. Extract each finding as a separate item. One comment may contain several findings — split them.
3. Classify each finding:
   - **blocking** — verdict-level or explicitly marked blocking; the PR must not be pushed until these are fixed with proof or resolved with the reviewer;
   - **minor** — small concrete issues (naming, missing test case, formatting): fix them in the same iteration, they are cheap;
   - **non-blocking** — suggestions, style preferences, optional improvements: either fix if trivially cheap, or explicitly decline with a reason.
4. If the same point appears in several rounds, keep only one item, but track which rounds mentioned it.

Nothing may be deleted from this list during the iteration. An item is only ever moved to `fixed`, `declined`, or `question` — with the corresponding evidence.

## Step 2 — Resolve each finding

For every finding, exactly one of three outcomes is allowed:

### Fixed — with proof

Fixing means proving the fix, not announcing it. Before writing "done" anywhere, produce evidence:

- **Code findings:** quote the changed code (file, lines) in the summary comment, and show the relevant check passing (`ruff`, `mypy`, `lint-imports`, `pytest`) locally.
- **Test findings (weak, invalid, or non-deterministic test):** after rewriting the test, run a **mutation check** — temporarily break the production code so the new test must fail, run it, observe the failure, revert the breakage. A test that cannot be made to fail by breaking the code does not "fix" the finding.
- **Claim findings ("the PR says X, but the code does Y"):** fix whichever side is wrong. If the description is wrong, update the PR description (see below); if the code is wrong, fix the code and rerun the checks.

A response of "accepted, fixing" without a subsequent proof is not a resolution.

### Declined — with justification

If a finding is not fixed, the reason must be written out: why the current behavior is correct, which requirement or standard supports it, or why the change would be harmful. "Ignored" or no answer is not allowed. Declines of blocking findings must be visible in the PR thread before the next review round, not buried in a commit message.

### Question — ask, do not stall

If a finding is unclear, contradictory, infeasible, or requires an owner's decision (product choice, breaking change, scope beyond the task), ask the reviewer a concrete question **in the thread of that finding**. Do not silently skip the finding and do not guess: an unanswered blocking question becomes a repeat `block` verdict. While a question is pending, mark the finding as `question` and do not push claiming full resolution.

## Step 3 — Sync the PR description with the actual diff

After all fixes are applied, compare the PR description against the actual diff:

- Every behavior claim in the description ("the test catches regression X", "the gate uses median", "this covers case Y") must be verifiable in the current diff. If a review finding proved a claim wrong, correct the description so it matches the code.
- Success criteria in the linked task are part of this check: if a criterion is factually not met by the diff, do not claim it — either change the code until it is met or reflect the real state.
- Update the description with `src pr edit <number> --title ... --description ...` before pushing. Text–code mismatch on a PR level is itself a review blocker.

## Step 4 — Pre-push checklist

Before pushing the iteration, verify item by item:

- [ ] Every finding from **every** round has status `fixed`, `declined`, or `question`.
- [ ] Every **blocking** finding is `fixed` with evidence (check run, mutation check for tests, code quote) or has an open explicit question to the reviewer — never `declined` without discussion.
- [ ] All quality gates pass locally after the fixes: `ruff format --check src/`, `ruff check src/`, `mypy src`, `lint-imports`, `pytest`.
- [ ] Every changed claim in the PR description now matches the diff; the description was updated via `src pr edit` if needed.
- [ ] The summary comment (see template) is posted **before or together with** the push, not after review requests.

If any checklist item fails, do not push.

## Step 5 — Post one summary comment

Answer the review with **one consolidated comment** — not a scatter of replies per finding. Template:

```markdown
## Review round N — findings summary

| # | Finding (round, severity) | Action | Proof |
|---|---------------------------|--------|-------|
| 1 | <short finding text> (round 2, blocking) | fixed | <pytest run / mutation check result / link to commit & code quote> |
| 2 | <short finding text> (round 2, minor) | fixed | <check output / code quote> |
| 3 | <short finding text> (round 1, non-blocking) | declined | <one-sentence justification> |
| 4 | <short finding text> (round 2, blocking) | question | <concrete question to the reviewer> |

All blocking findings from previous rounds: <count> — <status breakdown>.
PR description updated: <yes/no, what changed>.
Quality gates: <ruff / mypy / lint-imports / pytest — all passing locally>.
```

Rules for the comment:

- Sort rows by severity: blocking first, then minor, then non-blocking.
- The "Proof" column must contain something checkable: a command result, a code quote, a commit link — not "done" or "fixed".
- If the table has any `question` rows, state explicitly at the top which findings await the reviewer and that the PR is not claimed fully resolved.

## Expected effect

- No repeat `block` verdicts on points that were claimed fixed but were not.
- No review "debt" leaking into follow-up tasks after merge.
- Fewer review rounds per PR and shorter time to merge.
