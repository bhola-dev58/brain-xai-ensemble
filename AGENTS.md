# Agent Development & Continuity Guidelines

These permanent instructions guide any AI coding agent working in this repository. Follow them strictly to maintain state continuity, avoid repetitive discovery, and ensure software quality.

---

## 1. Context Recovery & Task Start Procedure

Before starting any task, execute this workflow in sequence:

1. **Read `CURRENT_STATE.md`** first to load the active development checkpoint, recent modifications, test statuses, and known constraints.
2. **Consult `PROJECT_CONTEXT.md`** for established architecture, verified file paths, tech stack details, and backend/frontend patterns.
3. **Consult `TASKS.md`** to check task progress, prerequisites, and pending work items.
4. **Inspect only files relevant to the specific prompt**:
   - Do NOT run full-repository tree listings or reread entire architectural documents unless specifically needed.
   - Reuse verified architectural decisions and file paths documented in `PROJECT_CONTEXT.md`.
   - Inspect actual code only to validate specific assumptions or resolve discrepancies.

---

## 2. Mandatory Workflow for Every Task

When assigned any coding or refactoring task:

1. **Review Context**: Check `CURRENT_STATE.md` and `PROJECT_CONTEXT.md`.
2. **Targeted Investigation**: Read only the specific source files, configurations, or tests directly touched by the requested change.
3. **Minimal, Robust Implementation**: Implement the smallest clean, backward-compatible solution that meets the requirement without modifying unrelated components.
4. **Follow UI/UX & CSS Policy**:
   - For all frontend UI and styling work, strictly adhere to [`.agents/rules/uiux-designing-rule.md`](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/.agents/rules/uiux-designing-rule.md).
   - Use standard handwritten CSS in `static/css/style.css`.
   - Never use Tailwind utility classes, Bootstrap, or inline HTML/JSX `style` attributes.
   - Leverage relevant skills under `.agents/skills/` (such as `ui-ux-pro-max`, `ui-styling`, `design-system`) for design guidance.
5. **Targeted Test Execution**:
   - Execute relevant unit/integration tests using `./venv/bin/pytest tests/<test_file>.py -q` or full suite `./venv/bin/pytest tests/ -q`.
   - Always report authentic test outcomes. Never invent test results or claim unverified features are working.
6. **Update Checkpoints**:
   - Update `CURRENT_STATE.md` with modified files, verified test results, and next steps.
   - Update `TASKS.md` marking completed tasks and recording any newly identified items.
   - Update `PROJECT_CONTEXT.md` only when system architecture, API schemas, or major file locations change.
7. **Git Commit & Push Policy**:
   - Commit logical units using `<type>: <short description>` format (lowercase, imperative, no trailing period, no emojis).
   - Never execute `git push` directly; print the recommended push command in chat for user execution.
   - Never start the project server process (`python app.py` or `python scripts/run_api.py`) yourself.

---

## 3. Core Principles & Safety Guardrails

- **Backend Integrity**: Never change model inference weights, pre/post-processing logic, or API schemas without explicit verification against `tests/`.
- **Zero Hallucination**: Do not mark tasks completed or document features as operational without verification against active code or test runs.
- **Architectural Reference**: Refer to [`docs/02_architecture.md`](file:///home/bhola-dev58/colledge%20project/Brain_Tumor_Project/docs/02_architecture.md) for deep-dive diagrams and module interactions rather than duplicating them.
