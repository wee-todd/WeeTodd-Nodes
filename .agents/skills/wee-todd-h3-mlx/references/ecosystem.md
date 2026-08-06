# Ecosystem references

- ComfyUI native H3: current `comfy_extras/nodes_minimax_h3.py`. Study synchronized latent,
  first/last-frame, reference, sigma-shift, v3 API, and workflow contracts; translate deliberately
  to MLX.
- MiniMax H3 model materials: use published model cards, configuration, and checkpoint metadata for
  architecture and task contracts.
- MLX: use the current framework documentation and independently verify Apple Silicon behavior.
- Third-party implementations may inform research questions, but public project design and behavior
  must be justified by primary sources and WeeTodd's own test evidence.

Prefer small nodes with stable typed connections. Separate model selection, configuration, conditioning, sampling, decoding, preview, saving, and unloading when independently useful.
