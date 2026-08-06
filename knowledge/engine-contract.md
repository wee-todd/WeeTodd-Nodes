---
type: Reference
title: MiniMax H3 MLX engine contract
description: Stable timing, geometry, conditioning, and lifecycle invariants for the MLX engine.
resource: ../.agents/skills/wee-todd-h3-mlx/references/engine-contract.md
tags: [minimax-h3, mlx, engine, contract]
status: stable
generated:
  by: process:codex-review
  at: 2026-08-05T17:10:28-07:00
sources:
  - id: minimax-h3
    resource: https://huggingface.co/MiniMaxAI/MiniMax-H3
    title: MiniMax H3 model repository
  - id: mlx-engine
    resource: https://github.com/mrbizarro/minimax-h3-mlx
    title: MiniMax H3 MLX engine
---

# Model behavior

H3 jointly denoises video and stereo-audio rows in one packed sequence.[^minimax-h3]

Video is 24 frames per second. Audio latents run at 40 Hz. Decoded audio is 32 kHz stereo.

The released weights use classifier-free guidance (CFG) distillation. The runtime therefore has no
conventional negative prompt or guidance pass.[^minimax-h3]

# Input constraints

The engine accepts durations from 5 through 15 seconds. The width and height must be divisible by
32. Native-quality geometry has a 768-pixel short edge. Smaller canvases are wiring tests.

# Runtime requirements

Preserve packed ordering and synchronized decoding as one generation contract.[^minimax-h3]

Keep model state process-local, lazily loaded, and explicitly unloadable.[^mlx-engine]

# Lifecycle limitation

Adaptive layer normalization (AdaLN) schedule precomputation can release approximately 13 billion
parameters. Dropping the projection parameters makes warm pipeline reuse schedule-sensitive.[^mlx-engine]

[^minimax-h3]: MiniMax H3 model repository
[^mlx-engine]: MiniMax H3 MLX engine
