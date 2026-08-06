---
type: Playbook
title: H3 node implementation plan
description: Prioritized implementation sequence for a composable MLX-native H3 node suite.
resource: ../docs/ROADMAP.md
tags: [roadmap, nodes, conditioning, quantization]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T17:10:28-07:00
sources:
  - id: roadmap
    resource: ../docs/ROADMAP.md
    title: Prioritized implementation plan
---

# Sequence

1. Stabilize text-to-video-plus-audio generation with component manifests, task validation, memory
   estimates, schedule-safe adaptive layer normalization behavior, progress, cancellation, output
   naming, metadata, and tiny-model tests.
2. Add component and quantized loaders. Then add first-frame, last-frame, and combined first/last-frame
   conditioning.
3. Add ordered image, video, and audio reference conditioning with official limits and prompt labels.
4. Add previews, selective unloading, workflow examples, and Comfy Registry packaging.

Detailed acceptance criteria and deferred experiments remain in the canonical [roadmap](../docs/ROADMAP.md).[^roadmap]

[^roadmap]: Prioritized implementation plan
