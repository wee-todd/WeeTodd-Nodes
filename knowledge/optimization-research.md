---
type: Research Summary
title: Apple Silicon H3 optimization research
description: Evidence and experiment boundaries for H3 quantization and attention optimization.
resource: ../docs/reference/PHOSPHENE_OPT_VERDICT.md
tags: [apple-silicon, quantization, attention, performance]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T17:10:28-07:00
sources:
  - id: engine-benchmarks
    resource: ../docs/reference/PHOSPHENE_OPT_BENCHMARKS.md
    title: MiniMax H3 MLX speed campaign benchmarks
  - id: metal-int8
    resource: ../docs/reference/METAL_QUANTIZED_ATTENTION_RESEARCH.md
    title: Metal quantized attention research note
  - id: optiq
    resource: ../docs/reference/OPTIQ_MIXED_PRECISION_RESEARCH.md
    title: OptiQ mixed-precision research note
---

# Weight quantization

## Claim

Low-bit H3 weights are currently a memory optimization, not a demonstrated speed optimization.

## Evidence

On the measured M4 Max workload, brain floating-point 16-bit (BF16) H3 approached local dense
general matrix multiplication throughput. Affine 4-bit and 8-bit weight matrix multiplications were
slower.[^engine-benchmarks]

## Limitation

The result applies to the measured hardware, runtime, checkpoint, and workload.

## Required validation

Run a hardware-specific benchmark before claiming that a low-bit checkpoint improves generation
speed.

# Quantized attention

## Claim

Dynamic 8-bit integer attention can reduce attention-memory traffic at long packed sequence lengths.

## Basis

The expected opportunity increases with packed sequence length.[^metal-int8]

## Limitation

The project does not have an independently validated fused kernel.[^metal-int8]

## Required validation

Validate quantization after root mean square normalization and rotary positional embedding. Also
validate stable online softmax, hardware gating, and deterministic audio-video quality before
enabling this backend.

# Mixed precision

## Claim

Mixed per-layer precision is a candidate memory optimization for H3.

## Limitation

Language-model logit Kullback–Leibler divergence does not transfer directly to H3.[^optiq]

## Required validation

Use cached teacher inputs and single-block replay across timesteps and modalities. Rank candidate
precisions, then validate the selected allocation over complete denoising trajectories.

[^engine-benchmarks]: MiniMax H3 MLX speed campaign benchmarks
[^metal-int8]: Metal quantized attention research note
[^optiq]: OptiQ mixed-precision research note
