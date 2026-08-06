---
type: Procedure
title: Python environment preflight
description: Prevent incompatible Python and venv changes before dependency or MLX installation.
resource: ../docs/PYTHON_ENVIRONMENT.md
tags: [python, mlx, dependencies, validation]
status: stable
generated:
  by: process:codex-review
  at: 2026-08-05T21:00:00-07:00
sources:
  - id: project-metadata
    resource: ../pyproject.toml
    title: Project Python and dependency metadata
  - id: environment-skill
    resource: ../.agents/skills/python-environment-preflight/SKILL.md
    title: Python environment preflight skill
---

# Requirement

Run the deterministic environment preflight before any pip, Python, venv, MLX, or dependency
change. Confirm the interpreter version, executable, architecture, platform, and environment prefix.

# Incompatible environment

Do not mutate an incompatible venv. A pip upgrade cannot change the Python interpreter version.

1. Select a compatible interpreter.
2. Preserve the incompatible venv outside the repository.
3. Create a replacement venv.
4. Run preflight against the replacement venv.
5. Install dependencies.
6. Run `python -m pip check` and project tests.

# ComfyUI boundary

Treat the ComfyUI Python environment as host-owned. Do not replace the host environment or upgrade
unrelated packages without explicit authorization.

# MLX boundary

Confirm Apple Silicon `arm64` and compatible Python before MLX installation. Verify the MLX import
without downloading weights or starting generation.
