---
type: Decision
title: Initial technical-lead audit
description: The ranked findings and boundary decisions governing scaffold hardening.
resource: ../docs/AUDIT_2026-08-05.md
tags: [audit, architecture, comfyui, mlx]
status: stable
generated:
  by: process:codex-review
  at: 2026-08-05T17:10:28-07:00
sources:
  - id: audit
    resource: ../docs/AUDIT_2026-08-05.md
    title: Technical-lead audit
---

# Decision

Keep the MLX engine independent from the ComfyUI adapter. Engine packing, schedules, neural modules,
and synchronized decoding remain framework-neutral.[^audit]

Keep paths, node schemas, progress, cancellation, publication, and host lifecycle policy in the
ComfyUI adapter.[^audit]

# Required work

Before expanding conditioning, harden task-aware component validation and schedule-safe adaptive
layer normalization reuse. Also harden memory estimation, output contracts, metadata, and clean
installation.

# Validation constraint

Treat checkpoint conversion and generation as expensive, opt-in validation activities.[^audit]

[^audit]: Technical-lead audit
