# Ecosystem references

- ComfyUI native H3: current `comfy_extras/nodes_minimax_h3.py`. Study contracts and conditioning semantics; translate deliberately to MLX.
- Spectrum H3: `xmarre/ComfyUI-Spectrum-MiniMax-H3`, GPL-3.0. Conceptual and benchmark reference only.
- KJNodes: `kijai/ComfyUI-KJNodes`. Study composability, controls, previews, and workflow ergonomics; do not copy code.
- MLX provenance: `mrbizarro/minimax-h3-mlx`, Apache-2.0.

Prefer small nodes with stable typed connections. Separate model selection, configuration, conditioning, sampling, decoding, preview, saving, and unloading when independently useful.
