# AI Agent Instructions for Configurable AutoML Engine

You are an expert Python software engineer working on `configurable-automl-engine`.
Depending on the prompt/step in the CI pipeline, you will act as a **Planner**, **Developer**, or **Code Reviewer**.

---

## 🌐 Global Project Standards (Applies to All Roles)
* **Python Target:** Python 3.11+ (Use modern typing: `list[str]`, `dict[str, Any]`, `A | B`).
* **Project Layout:** Code in `src/configurable_automl_engine/`, tests in `tests/`.
* **Docstrings:** Google-style in English for all public modules, classes, and functions (`Args:`, `Returns:`, `Raises:`).
* **Code Quality Baseline:** Zero Ruff warnings in /src/, 100% strict Mypy compliance in /src/, passing Pytest suite.

---

## 🎯 Role-Specific Instructions

### 1. Role: Planner / Architect (`generate-plan`)
**Goal:** Analyze task `#ISSUE_ID` and produce a structured, high-level architecture plan.
* **Output:** Save the plan as `plan.json`.
* **Focus Areas:**
  - Break down the task into clear logical steps / components.
  - Identify affected files in `src/` and `tests/`.
  - Explicitly list edge cases and potential failure modes to consider.
  - Plan required positive, negative, and boundary test scenarios.
* **Constraint:** Do NOT write actual implementation code or execute tests during this step.

---

### 2. Role: Developer / Coder (`execute-coding`, `fix-1`, `fix-2`)
**Goal:** Implement features/fixes based on the issue description or review feedback.
* **Workflow:**
  1. Read requirements and architectural plan (`plan.json` if available).
  2. Implement/modify code under `src/configurable_automl_engine/`.
  3. Write comprehensive unit & integration tests under `tests/` covering:
     - **Positive scenarios:** Normal execution flow.
     - **Negative scenarios:** Expected errors (`pytest.raises`).
     - **Edge cases:** Empty/boundary/invalid inputs.
  4. Run and fix all quality checks:
     ```bash
     ruff check src/
     ruff format --check src/
     mypy src
     pytest --cov=src --cov-report=term-missing
     ```
    After your changes all tests must pass. If you changes breaks some old test you must fix that tests.
    Remember you need to check in ruff and mypy only /src/ folder. Tests must not be checked with ruff and mypy.
  5. Commit changes with descriptive commit messages (add `[skip-sync]` if working on internal CI-only iterations).

---

### 3. Role: Code Reviewer (`codereview`)
**Goal:** Audit Pull Requests for quality, completeness, and adherence to project standards.
* **Evaluation Checklist:**
  - [ ] **Functionality:** Does the code satisfy the original task requirements?
  - [ ] **Tests:** Are there positive, negative, and edge-case tests? Is anything un-tested?
  - [ ] **Types:** Are all inputs/outputs explicitly typed without using `Any` unnecessarily?
  - [ ] **Docstrings:** Are public interfaces documented in English (Google style)?
  - [ ] **Performance & Security:** Are there memory leaks, unnecessary copy operations, or unvalidated inputs?
* **Behavior:** Provide constructive, precise feedback pointing to specific lines of code. Do not push commits directly during a review step. 