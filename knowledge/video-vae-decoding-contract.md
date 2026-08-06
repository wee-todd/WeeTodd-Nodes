---
type: Contract
title: H3 video VAE decoding contract
description: Defines final video decoding, provenance checks, output layout, and staged memory cleanup.
resource: ../docs/ARCHITECTURE.md
tags: [minimax-h3, mlx, video-vae, decoding]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T20:00:00-07:00
sources:
  - id: engine-contract
    resource: engine-contract.md
    title: MiniMax H3 MLX engine contract
  - id: transformer-sampling-contract
    resource: transformer-sampling-contract.md
    title: H3 transformer sampling contract
---

# Input

The decoder accepts synchronized H3 latents and an explicit video VAE specification. The decoder
rejects latents that name a different video VAE.

# Execution

The decoder loads only the final video VAE. The decoder reverses H3 latent normalization before VAE
execution. The decoder reverses ImageNet pixel normalization after VAE execution.

The decoder checks ComfyUI cancellation before and after VAE execution. A cancellation or failure
releases the process-local video VAE and clears the MLX cache.

# Output

The decoder returns contiguous float RGB frames with shape `(frames, height, width, 3)` and values
from zero through one. The adapter converts the array to a ComfyUI `IMAGE` tensor. The synchronized
audio latents remain available in the original latent contract.

# Limitation

The installed MLX runtime permits synthetic engine validation. Final decoder execution still needs
approved local H3 video VAE weights.
