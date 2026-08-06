---
type: Decision
title: BF16-class first T2VA baseline
description: Keep the first T2VA smoke test unquantized while preserving audited FP32 exceptions.
resource: ../docs/ROADMAP.md
tags: [minimax-h3, mlx, bf16, precision, performance]
status: stable
generated:
  by: process:codex-review
  at: 2026-08-05T23:00:00-07:00
sources:
  - id: engine-loader
    resource: ../src/minimax_h3_mlx/load.py
    title: MiniMax H3 MLX transformer loader
---

# Decision

Use the unquantized reference precision policy for the first working T2VA generation. Most
transformer weights and Qwen3-VL execution use BF16.

Preserve the reference FP32 patch projections, timestep path, and output heads. These small modules
remain FP32 because their coherent rounding error can accumulate through the denoising trajectory.

# Deferred work

Defer weight quantization, activation quantization, and mixed-precision allocation until the first
ComfyUI T2VA smoke generation succeeds.

# Current performance work

Continue optimizations that preserve the baseline numerical contract. Priorities include staged
component residency, exact text-encoder truncation, AdaLN projection release, bounded video VAE
batches, reduced allocation, and measured kernel improvements.
