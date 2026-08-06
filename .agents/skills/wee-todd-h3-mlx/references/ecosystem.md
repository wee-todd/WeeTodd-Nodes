# Ecosystem references

- ComfyUI native H3: current `comfy_extras/nodes_minimax_h3.py`. Study synchronized latent,
  first/last-frame, reference, sigma-shift, v3 API, and workflow contracts; translate deliberately
  to MLX.
- Spectrum H3: `xmarre/ComfyUI-Spectrum-MiniMax-H3`, GPL-3.0. Study compatibility checks, bounded
  history, exact fallbacks, teardown, and quality gates. Do not copy code.
- KJNodes: `kijai/ComfyUI-KJNodes`, GPL-3.0. Study composability, controls, previews, model patching,
  and workflow ergonomics; do not copy code.
- Kijai H3 artifacts: treat the experimental W4A8 transformer, quantized video VAE, and tiny preview
  autoencoder as separate component references. They do not form a complete pipeline and are not
  automatically MLX-compatible.
- MLX provenance: `mrbizarro/minimax-h3-mlx`, Apache-2.0.

Prefer small nodes with stable typed connections. Separate model selection, configuration, conditioning, sampling, decoding, preview, saving, and unloading when independently useful.
