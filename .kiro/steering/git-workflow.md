---
inclusion: always
---

# Git workflow

- Never work, commit or push on `main`. One branch per task group: `feat/<area>-<short>`, `docs/<short>`, `fix/<short>`.
- Commits, pushes, PRs, merges and deletions require the owner's explicit approval.
- Every PR: summary, REQ ids covered, what was tested, evidence produced, traceability rows updated.
- A task in `.kiro/specs/cora/tasks.md` is ticked only when its evidence exists in the repo.
- Stage specific files; run `ruff` and targeted tests before committing.
