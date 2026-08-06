---
type: Contract
title: H3 audio VAE decoding contract
description: Defines final stereo audio decoding, timing metadata, provenance checks, and cleanup.
resource: ../docs/ARCHITECTURE.md
tags: [minimax-h3, mlx, audio-vae, decoding, comfyui]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T22:00:00-07:00
sources:
  - id: engine-contract
    resource: engine-contract.md
    title: MiniMax H3 MLX engine contract
  - id: transformer-sampling-contract
    resource: transformer-sampling-contract.md
    title: H3 transformer sampling contract
  - id: comfyui-audio-contract
    resource: https://github.com/Comfy-Org/ComfyUI/blob/15989f87ca89bfe2e7c47763252c559e96d97551/comfy_extras/nodes_audio.py
    title: ComfyUI audio nodes
---

# Input

The decoder accepts synchronized H3 latents and an explicit audio VAE specification. The decoder
rejects latents that name a different audio VAE.

The audio VAE sampling rate must equal the latent contract sampling rate. The first supported H3
contract uses 32 kHz stereo audio.

# Execution

The decoder loads only the final audio VAE. The decoder reverses H3 audio-latent normalization
before VAE execution. The mono VAE decodes the two packed channel batches into synchronized left and
right waveforms.

The decoder checks ComfyUI cancellation before and after VAE execution. A cancellation, mismatch,
or failure releases the process-local audio VAE and clears the MLX cache.

# Output

The engine returns contiguous float audio with shape `(channels, samples)`. The adapter returns the
current ComfyUI `AUDIO` mapping:

```text
waveform: (1, 2, samples)
sample_rate: 32000
```

Metadata retains the channel count, sample count, audio duration, video frame count, video frame
rate, decode duration, and residency state.

# Limitation

Synthetic lifecycle tests validate the adapter boundary. Final decoder execution still needs
approved local H3 audio VAE weights.
