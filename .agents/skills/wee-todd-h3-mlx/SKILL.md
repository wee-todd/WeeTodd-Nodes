---
name: wee-todd-h3-mlx
description: Implement, review, test, or optimize WeeTodd Nodes for MiniMax H3 generation through MLX in ComfyUI on Apple Silicon. Use for H3 loaders, generation and conditioning nodes, packed audio-video latents, MLX memory lifecycle, checkpoint conversion or quantization, parity checks, previews, and Kijai-style composable workflow design.
---

# WeeTodd H3 MLX

Work only inside the MiniMax H3, MLX, and ComfyUI boundary defined by `AGENTS.md`.

## Workflow

1. Read `AGENTS.md` and classify unknown runtime assumptions before editing.
2. Decide whether the change belongs to the MLX engine (`src/minimax_h3_mlx`) or ComfyUI adapter (`src/wee_todd_nodes`).
3. Read `references/engine-contract.md` before changing runtime behavior.
4. Read `references/ecosystem.md` when designing node interfaces or studying third-party behavior.
5. Keep imports lightweight and defer checkpoint loading until graph execution.
6. Validate paths, duration, dimensions, task support, and memory options before inference.
7. Add focused unit coverage. Run parity or real-checkpoint tests only when the changed layer requires them.
8. Run the validation commands in `AGENTS.md` and report expensive tests not run.

## Guardrails

- Never add Draw Things material.
- Never commit weights, media, caches, tokens, or absolute personal paths.
- Do not copy GPL Spectrum or KJNodes implementation code into Apache-2.0 files.
- Preserve synchronized audio/video timing and packed-sequence ordering.
- Do not imply native-resolution H3 is fast on current Apple Silicon.
- Avoid full-media PyTorch round-trips unless a ComfyUI contract requires them.
