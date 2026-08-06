---
type: Contract
title: H3 resolution selection
description: Preset-first H3 canvas selection with one configuration for preflight and generation.
resource: ../docs/reference/H3_RESOLUTION_SELECTION.md
tags: [minimax-h3, comfyui, resolution, workflow]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T20:51:00-07:00
---

# Contract

Select a quality tier and an aspect ratio in the generation configuration. Use custom mode only for
exact dimensions. Resolve every preset to the H3 32-pixel canvas grid.

# Tiers

Provide fast-smoke 384P, balanced 512P, native-quality 768P, and experimental 2K tiers. Label the 2K
tier as a very-high-memory experiment because dense attention makes the tier expensive.

# Single source

Pass the same generation configuration to Preflight, sampling, and publication. Do not request a
second width or height from Preflight.
