---
name: wee-todd-h3-mlx
description: Implement, review, test, or optimize WeeTodd Nodes for MiniMax H3 generation through MLX in ComfyUI on Apple Silicon. Use for H3 loaders, generation and conditioning nodes, packed audio-video latents, MLX memory lifecycle, checkpoint conversion or quantization, parity checks, previews, and composable workflow design.
---

# WeeTodd H3 MLX

Work only inside the MiniMax H3, MLX, and ComfyUI boundary defined by `AGENTS.md`.

## Workflow

1. Read `AGENTS.md` and classify unknown runtime assumptions before editing.
2. Before dependency or venv changes, use `../python-environment-preflight/SKILL.md` and run its
   preflight script. Do not trust an existing `.venv` until the check passes.
3. Decide whether the change belongs to the MLX engine (`src/minimax_h3_mlx`) or ComfyUI adapter (`src/wee_todd_nodes`).
4. Read `references/engine-contract.md` before changing runtime behavior.
5. Read `references/ecosystem.md` when designing node interfaces or studying third-party behavior.
6. Keep imports lightweight and defer checkpoint loading until graph execution.
7. Validate paths, duration, dimensions, task support, and memory options before inference.
8. Preserve staged residency across the full graph. Default every weighted stage to unload after
   use. Permit keep-warm only through an explicit user control and report the resident state.
9. Add focused unit coverage. Run parity or real-checkpoint tests only when the changed layer requires them.
10. Use concise, controlled language for node text, errors, procedures, and workflows.
11. Run the validation commands in `AGENTS.md` and report expensive tests not run.

## Guardrails

- Draw Things documentation and models may be studied as design references.
- Do not copy Draw Things code, binaries, model weights, recipes, or project-specific implementation into WeeTodd Nodes.
- Independently validate any derived MLX/H3 optimization and retain provenance in local research notes.
- Never commit weights, media, caches, tokens, or absolute personal paths.
- Do not copy incompatible or unlicensed third-party implementation code into Apache-2.0 files.
- Preserve synchronized audio/video timing and packed-sequence ordering.
- Do not imply native-resolution H3 is fast on current Apple Silicon.
- Avoid full-media PyTorch round-trips unless a ComfyUI contract requires them.
- Do not load a downstream weighted component before the upstream component is releasable.
- On success, failure, or cancellation, release the active component when staged unloading is
  selected. Clear Python references before `gc.collect()` and `mx.clear_cache()`.
- Keep live conditioning and synchronized latents only as long as their downstream consumers need
  them. Do not duplicate large MLX or host tensors for metadata or previews.
