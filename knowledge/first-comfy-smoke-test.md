---
type: Playbook
title: First ComfyUI H3 smoke test
description: Gated path to one synchronized MiniMax H3 generation through ComfyUI and MLX.
resource: ../docs/SMOKE_TEST_PLAN.md
tags: [comfyui, smoke-test, minimax-h3, mlx]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T19:10:07-07:00
sources:
  - id: smoke-test-plan
    resource: ../docs/SMOKE_TEST_PLAN.md
    title: First ComfyUI smoke-test plan
---

# Goal

Produce one playable five-second MiniMax H3 output with synchronized video and stereo audio through
a ComfyUI workflow backed by MLX.

# Gates

1. Prove clean import, registration, workflow loading, and missing-component errors without weights.
2. Validate every component and estimate staged peak memory from headers before allocation. The
   component loader and preflight implementation now satisfy the code-level part of this gate.
3. Pass tiny-model tests for packing, schedules, progress, cancellation, cleanup, and output safety.
4. Probe Qwen3-VL, transformer, video VAE, and audio VAE independently with approved local artifacts.
5. Run the smallest validated text-only generation after the operator approves its expected cost.

# Completed code contracts

Weight-free tests import the node module and repository entrypoint without importing MLX. The
component loader, preflight node, text-only Qwen3-VL encode node, and Qwen3-VL unload node are
registered.

The Qwen3-VL cache supports compatible reuse, replacement, unload-after-encode, and failure cleanup.
Text-only encoding omits the vision tower and accepts independent processor and tokenizer paths.

The transformer-only sampler produces synchronized undecoded video and audio latents. It reports
each evaluation, supports cancellation callbacks, unloads on failure, and reloads when a changed
schedule would make dropped AdaLN projections unsafe to reuse.

Independent video and audio VAE nodes decode final ComfyUI `IMAGE` and `AUDIO` values. The output
node validates synchronized timing and atomically publishes MP4 and JSON files under ComfyUI.

# Verified host and component state

A clean ComfyUI 0.30.0 host loaded the eight-node T2VA workflow without weights. The host registered
all fifteen WeeTodd nodes. Live schema validation accepted every required input and link type.

Strict isolated loader probes passed for the audio VAE, video VAE, Q8 Qwen3-VL text encoder, and
pruned transformer. Each process released its model state after the probe. No inference forward or
generation ran.

The selected five-second wiring-test configuration has a 42.388 GB header-based staged peak
estimate. The selected local stack combines a BF16/FP32 transformer, Q8 text encoder, FP16 video
VAE, and FP32 audio VAE. The mixed stack differs from the preferred unquantized BF16-class baseline.

# Next gate

Select or approve the mixed component identities. Then run one short prompt encode and bounded
forward/decode probes if required. Start the five-second generation only after explicit operator
approval.

# Success criteria

The output is playable and synchronized. Progress and cancellation work. Metadata identifies every
component and resolved generation value. Output paths remain under ComfyUI. Explicit unload clears
the process-local pipeline and MLX cache.

# Stop conditions

Do not generate when memory headroom is unsafe, component identity is ambiguous, task metadata is
incompatible, cancellation does not work, or either final decoder has not passed its probe.
