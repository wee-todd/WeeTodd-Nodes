---
type: Contract
title: H3 transformer sampling contract
description: Inputs, outputs, lifecycle, and safety rules for transformer-only synchronized sampling.
resource: ../docs/ARCHITECTURE.md
tags: [minimax-h3, mlx, sampling, audio-video]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T18:20:36-07:00
sources:
  - id: engine-contract
    resource: engine-contract.md
    title: MiniMax H3 MLX engine contract
  - id: smoke-test-plan
    resource: ../docs/SMOKE_TEST_PLAN.md
    title: First ComfyUI smoke-test plan
---

# Input

The first sampler accepts a validated T2VA component set, text-only Qwen3-VL conditioning, and a
validated generation configuration. Conditioning contains live MLX embeddings and one text modality
tag for each row. The sampler rejects conditioning whose text encoder, processor, or tokenizer
identity differs from the component set selected for the transformer.

# Execution

The sampler creates normalized video and stereo-audio noise. It packs text, target audio, and target
video rows into one sequence. Video and audio use separate shifted schedules and one shared
transformer evaluation for each schedule interval.

The adapter reports each evaluation and checks ComfyUI cancellation at every callback. Cancellation
or failure releases the transformer-only runtime.

# Output

The sampler returns synchronized normalized video and audio latents. It does not load or execute the
video VAE or audio VAE. The latent contract retains both the transformer component identity and the
validated generation configuration so downstream decoders and metadata nodes can verify provenance.

# Reuse constraint

Warm reuse requires equal component identity, sampling-step count, and AdaLN-drop policy. A changed
schedule reloads the transformer because a previous run may have removed the AdaLN projection
weights needed to build another modulation cache.

# Validation status

Synthetic lifecycle and node-contract tests pass without MLX. A tiny-config MLX shape test exists
but has not run in the current environment. No real checkpoint has been loaded.
