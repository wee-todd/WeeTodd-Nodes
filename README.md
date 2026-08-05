# WeeTodd Nodes

Experimental ComfyUI nodes for running MiniMax H3 natively through MLX on Apple Silicon.

This project begins with the Apache-2.0 MiniMax H3 MLX engine developed alongside Phosphene, then exposes it through small, composable ComfyUI nodes. The goal is a Kijai-style H3 suite for Mac users: explicit loaders, reusable model state, generation controls, conditioning tools, memory controls, previews, quantization, and workflow examples.

## Current nodes

- **WeeTodd H3 Model Loader (MLX)** describes a full or quantized checkpoint and loads it lazily.
- **WeeTodd H3 Generation Config** validates duration, steps, seed, dimensions, and AdaLN behavior.
- **WeeTodd H3 Generate Video + Audio** produces a synchronized MP4 and JSON sidecar.
- **WeeTodd H3 Unload MLX Runtime** releases the warm pipeline and cached MLX allocations.

The first release supports text-to-video-plus-audio. First/last-frame and multi-reference nodes are next; the engine already contains much of the keyframe machinery.

## Install for development

Clone into `ComfyUI/custom_nodes/WeeTodd-Nodes`, then use the same Python environment as ComfyUI:

```bash
python -m pip install -e .
```

Restart ComfyUI and look under `WeeTodd/H3`.

## Reality check

H3 is a 33B joint audio/video diffusion transformer plus a large text encoder and two VAEs. MiniMax's initial open release uses dense attention. Native-resolution generation is extremely compute-intensive even on large Apple Silicon systems. Start with `640x384`, 5 seconds, and a low step count to verify wiring before increasing quality.

Models are never bundled. See [Architecture](docs/ARCHITECTURE.md), [Roadmap](docs/ROADMAP.md), and [Attribution](docs/ATTRIBUTION.md).

## Status

Experimental and pre-release. APIs, node names, and workflow compatibility may change.
