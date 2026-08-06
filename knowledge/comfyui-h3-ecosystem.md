---
type: Research Summary
title: ComfyUI MiniMax H3 ecosystem
description: Native ComfyUI, Kijai, and Spectrum contracts relevant to an MLX-native H3 node suite.
resource: ../docs/reference/COMFYUI_H3_ECOSYSTEM.md
tags: [comfyui, minimax-h3, kijai, spectrum]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T17:21:42-07:00
sources:
  - id: comfyui-h3-nodes
    resource: https://github.com/Comfy-Org/ComfyUI/blob/7972b5ba7f1597f68261be33c912f5e5dba8b9c0/comfy_extras/nodes_minimax_h3.py
    title: Native ComfyUI MiniMax H3 nodes
  - id: comfyui-h3-workflows
    resource: https://docs.comfy.org/tutorials/video/minimax/minimax-h3
    title: Official ComfyUI MiniMax H3 workflows
  - id: kijai-h3-experimental
    resource: https://huggingface.co/Kijai/MiniMax-H3-experimental
    title: Kijai experimental MiniMax H3 artifacts
  - id: spectrum-h3
    resource: https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3
    title: Spectrum MiniMax H3
---

# Native ComfyUI

Native ComfyUI defines synchronized audio-video latents, first/last-frame conditioning, multimodal
reference conditioning, and coordinated video/audio sigma shifts. These contracts are the primary
semantic reference for WeeTodd nodes.

# Kijai

Kijai's 12.54 GB experimental W4A8 artifact is a transformer, not a complete H3 pipeline. The
pipeline also requires Qwen3-VL, processor and tokenizer assets, a video VAE, and an audio VAE.
KJNodes provides useful preview and operator-feedback references. Its GPL implementation must not
be copied.

# Spectrum

Spectrum demonstrates guarded transformer-evaluation forecasting, compatibility checks, bounded
history, fallbacks, and teardown. Forecasting can change motion and detail. Its GPL implementation
must not be copied.

# Decision

Implement independent MLX-native nodes. Do not require native H3, KJNodes, or Spectrum nodes. Use
their published contracts and observed behavior as attributed design evidence.
