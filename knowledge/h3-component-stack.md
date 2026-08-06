---
type: Reference
title: MiniMax H3 component stack
description: Required weighted and unweighted components for synchronized H3 generation.
resource: ../docs/SMOKE_TEST_PLAN.md
tags: [minimax-h3, components, memory, mlx]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T23:30:00-07:00
sources:
  - id: minimax-h3-manifest
    resource: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/FL2VA/model_index.json
    title: MiniMax H3 FL2VA model manifest
  - id: minimax-h3-mlx
    resource: https://github.com/mrbizarro/minimax-h3-mlx
    title: MiniMax H3 MLX engine
---

# Required weighted components

1. The transformer jointly denoises packed video and audio rows.
2. The Qwen3-VL encoder produces prompt and visual conditioning states.
3. The video variational autoencoder encodes conditions and decodes final video frames.
4. The audio variational autoencoder encodes references and decodes final stereo audio.

# Required unweighted components

The pipeline also requires processor and tokenizer assets, scheduler logic, sigma mapping, packed
latent construction, media conversion, output publication, progress, cancellation, and cleanup.

# Memory decision

Estimate and manage every component independently. Prefer staged residency: encode conditioning,
release Qwen3-VL when necessary, sample with the transformer, then decode with the final VAEs.

The default residency order is:

```text
load Qwen3-VL -> encode -> release Qwen3-VL
load transformer -> sample -> release transformer
load video VAE -> decode -> release video VAE
load audio VAE -> decode -> release audio VAE
publish host media
```

Every weighted node defaults to unload after use. A keep-warm choice must be explicit and must
report the resident state. Failure or cancellation releases the active component when staged
unloading is selected. The pipeline must not duplicate live conditioning, latents, frames, or audio
for metadata and previews.

# Implementation status

The component loader and preflight nodes are implemented. Preflight reads configuration files and
safetensors headers without reading tensor payloads. It validates task metadata, required assets,
indexed shards, latent dimensions, and supported MLX affine quantization recipes.

Preflight rejects non-MLX single-file components that lack the engine's self-describing metadata
or pruned transformer table. A native ComfyUI or Kijai safetensors artifact is not automatically an
MLX component.

The memory report separates Qwen3-VL encoding, transformer loading, transformer sampling, video
decoding, and audio decoding. Kernel workspaces and allocator fragmentation remain estimates until
real-component probes calibrate them.

# Optimization constraint

Do not describe a transformer artifact as a complete pipeline size. Validate component format,
precision, quantization recipe, task metadata, quality, and MLX compatibility independently.
