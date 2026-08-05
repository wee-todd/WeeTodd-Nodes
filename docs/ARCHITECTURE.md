# Architecture

ComfyUI owns graph scheduling and output locations. WeeTodd Nodes owns the MLX pipeline and returns paths plus structured generation metadata. No PyTorch model is created or patched.

```text
ComfyUI graph
  -> model specification
  -> generation configuration
  -> lazy process-local MLX pipeline
  -> Qwen3-VL + packed H3 DiT + video/audio VAEs
  -> synchronized MP4 and JSON sidecar
```

The immutable model specification is cheap to pass through graphs. One compatible pipeline stays warm until explicitly unloaded. Media is written directly to avoid duplicating full results through PyTorch tensors in unified memory. Engine code remains separate from ComfyUI adapters so parity tests stay framework-independent.

Near-term extension points are first/last-frame conditioning, multi-reference packing, component loaders, quantized transformers, progress and cancellation, preview decode, and long-video stitching.
