# ComfyUI MiniMax H3 ecosystem

Research date: 2026-08-05.

This document records interface and design evidence. WeeTodd Nodes does not copy implementation
code from the projects listed here.

## Native ComfyUI

Current ComfyUI provides four MiniMax H3 nodes through its v3 extension API:

- `EmptyMiniMaxH3LatentAV` creates synchronized video and audio latent shapes.
- `MiniMaxH3ImageToVideo` builds text, first-frame, last-frame, or first/last-frame conditioning.
- `MiniMaxH3ReferenceToVideo` builds ordered image, video, soundtrack, and audio references.
- `MiniMaxH3SigmaShift` applies coordinated video and audio flow shifts.

The native contract packs text, condition rows, target audio rows, and target video rows. Native
latents use 24 video channels and 32 audio channels. Video runs at 24 frames per second, audio
latents run at 40 Hz, and frame counts use the `17k+5` grid. The reference node presents images,
videos, and audio with `<Picture i>`, `<Video k>`, and `<Audio j>` labels.

The official workflow separates four component artifacts:

1. A diffusion transformer.
2. A Qwen3-VL-32B text and vision encoder.
3. A video variational autoencoder (VAE).
4. An audio VAE.

The workflow stores the transformer under `models/diffusion_models`, Qwen3-VL under
`models/text_encoders`, and both VAEs under `models/vae`. The processor, tokenizer, sampler,
scheduler, preview, and media-combine behavior are additional runtime dependencies even though
they are not all weight files.

Sources:

- [ComfyUI MiniMax H3 node source](https://github.com/Comfy-Org/ComfyUI/blob/7972b5ba7f1597f68261be33c912f5e5dba8b9c0/comfy_extras/nodes_minimax_h3.py)
- [Official ComfyUI MiniMax H3 workflows](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [Comfy-Org MiniMax H3 artifacts](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/0bd506d2e895983a9663037febda27aa3948cf48)

## Kijai references

Kijai's H3 work currently supplements the native ComfyUI pipeline rather than providing a separate
complete H3 node suite.

The experimental model repository contains a 12.54 GB mixed W4A8 FL2VA transformer and a 3.17 GB
Int8 convolution-rotation video VAE. The repository describes both files as work in progress. The
transformer depends on an experimental `comfy-kitchen` change, and the video VAE depends on a
ComfyUI pull request. The repository does not contain the Qwen3-VL encoder, audio VAE, processor, or
tokenizer. Therefore, “12 GB” describes one transformer artifact, not a complete runnable pipeline.

KJNodes also provides preview override support for an H3 tiny autoencoder. The tiny autoencoder is
appropriate for sampling previews only. It cannot replace the final video VAE decode.

KJNodes is GPL-3.0. WeeTodd Nodes can study node boundaries, preview behavior, model patching,
operator feedback, and failure handling. WeeTodd Nodes must not copy KJNodes implementation code.
Kijai model artifacts also require independent format, license, quality, and MLX compatibility
review before conversion or redistribution.

Sources:

- [Kijai experimental H3 artifacts](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/8b48334e6263a39b34eef85f9f5e271ba4506945)
- [Kijai H3 tiny autoencoder](https://huggingface.co/Kijai/MiniMax-H3-TAE/tree/a213ac8bf2f148b4f32372279a7f207846978900)
- [KJNodes H3 preview commit](https://github.com/kijai/ComfyUI-KJNodes/commit/0f8f8d7c3e14e855b61261916ee361f6578bff75)

## Spectrum reference

Spectrum adds an optional model patch to native ComfyUI H3. It forecasts post-transformer target
features and skips selected transformer evaluations. Actual steps use the native model. Forecast
steps can change motion trajectories or degrade short-lived details.

Useful design lessons are explicit compatibility checks, conservative defaults, sampler-aware
fallbacks, bounded history, per-run teardown, and same-seed comparison against the unmodified
model. Spectrum is GPL-3.0. WeeTodd Nodes must not copy its implementation. A future MLX trajectory
forecast experiment must be independently designed and must remain behind exact fallback and
quality gates.

Source: [Spectrum MiniMax H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/tree/85ec1da66277e893079ecd46e32cc865c56cfe53)

## Boundary decision

WeeTodd Nodes will implement MLX-native component loaders and generation nodes. It will not require
native H3, KJNodes, or Spectrum nodes. The native ComfyUI contracts remain the primary semantic and
workflow reference. Kijai and Spectrum remain design and benchmark references only.
