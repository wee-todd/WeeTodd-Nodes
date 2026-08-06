---
name: python-environment-preflight
description: Validate a project's Python interpreter, requires-python constraint, virtual environment ownership, platform, and installer capability before creating, replacing, or installing into a venv. Use before pip installs, dependency upgrades, MLX setup, ComfyUI environment changes, or recovery from an incompatible Python environment.
---

# Python environment preflight

Run the bundled preflight before any package installation or virtual-environment mutation.

## Workflow

1. Read `references/policy.md` when an existing environment is incompatible or replacement is
   required.
2. Run `scripts/preflight.py --project <project> --python <candidate>` for each candidate
   interpreter. Add `--require-architecture arm64` for MLX work.
3. Stop if the candidate does not satisfy `project.requires-python`.
4. Inspect the reported executable, version, architecture, platform, venv path, and pip version.
5. Select a compatible interpreter before creating or replacing a venv.
6. Preserve an incompatible venv outside the repository unless the user explicitly authorizes
   deletion.
7. Create the replacement venv with the selected interpreter.
8. Run the preflight against the replacement venv before installing packages.
9. Install dependencies only after both preflight runs pass.
10. Run `python -m pip check` and the project validation suite.

## Guardrails

- Do not assume that `.venv/bin/python` matches the project requirement.
- Do not upgrade pip as a substitute for an incompatible Python interpreter.
- Do not install system-wide packages when a project or host environment is intended.
- Do not leave backup environments, wheels, caches, or machine-specific paths inside the project.
- Use `python -m pip`, not an unqualified `pip` command.
- Treat ComfyUI's Python environment as host-owned. Do not replace it without explicit authority.
- On Apple Silicon, verify `arm64` before installing MLX.
