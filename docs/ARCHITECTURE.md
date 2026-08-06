# Architecture

ComfyUI owns graph scheduling and output locations. WeeTodd Nodes owns the MLX pipeline and returns paths plus structured generation metadata. No PyTorch model is created or patched.

```text
ComfyUI graph
  -> component specification
  -> header-only preflight
  -> Qwen3-VL text conditioning
  -> generation configuration
  -> lazy process-local MLX pipeline
  -> Qwen3-VL + packed H3 DiT + video/audio VAEs
  -> synchronized MP4 and JSON sidecar
```

The immutable model specification is cheap to pass through graphs. One compatible pipeline stays warm until explicitly unloaded. Media is written directly to avoid duplicating full results through PyTorch tensors in unified memory. Engine code remains separate from ComfyUI adapters so parity tests stay framework-independent.

The composable path defaults to staged residency, not full-pipeline residency. The normal order is
Qwen3-VL encode, transformer sampling, video VAE decode, audio VAE decode, and publication. Each
weighted node defaults to unload after use and reports whether its process-local cache remains
resident. An operator may keep a compatible component warm explicitly for repeated work. Failure or
cancellation releases the active component when staged unloading is selected. Downstream stages
must not load before the upstream component is releasable.

The component specification names the transformer, Qwen3-VL encoder, processor, tokenizer, video
VAE, and audio VAE independently. The adapter validates their manifests and safetensors headers
without importing MLX or reading tensor payloads. The engine remains responsible for constructing
and executing compatible MLX components after preflight.

Text conditioning has its own process-local cache. A text-only request constructs Qwen3-VL without
the vision tower, reads the unnormalized layer-50 state, materializes the conditioning embeddings,
and can release the encoder before transformer sampling. Processor and tokenizer directories are
explicit component inputs rather than assumed machine-specific siblings.

Transformer sampling has a separate process-local cache. The sampler consumes live conditioning,
constructs target video and stereo-audio noise, builds both shifted schedules, runs one packed
transformer evaluation per schedule interval, and returns normalized MLX latents. It does not load
either VAE. Warm reuse is keyed by component identity, sampling steps, and AdaLN-drop policy. A
changed schedule reloads the transformer before a dropped AdaLN projection can be reused incorrectly.
Conditioning must name the same Qwen3-VL encoder, processor, and tokenizer as the selected component
set. Returned latents retain transformer and generation provenance for downstream verification.

Final video decoding has a separate process-local cache. The decoder loads only the selected video
VAE, checks latent provenance, reverses H3 latent and pixel normalization, and returns float RGB
frames in ComfyUI layout. The decoder releases its model and MLX cache after success, cancellation,
or failure when staged unloading is enabled. The synchronized audio latents remain in the original
latent contract for the audio decoder.

Final audio decoding has an independent process-local cache. The decoder loads only the selected
audio VAE, checks latent provenance and sample-rate compatibility, and reverses H3 latent
normalization. The adapter returns the current ComfyUI `AUDIO` mapping with a float waveform shaped
as `(batch, channels, samples)` and an integer sample rate. H3 output uses one batch, two channels,
and 32 kHz. Timing metadata retains the audio sample count, duration, video frame count, and video
frame rate. Cancellation or failure releases the audio VAE and clears the MLX cache.

Publication runs after both VAE stages. The adapter converts ComfyUI image tensors directly to
contiguous byte RGB and converts the audio tensor to contiguous float audio. The publisher rejects
incorrect dimensions, frame rate, sample rate, channels, non-finite audio, unsafe paths, and more
than 25 ms of audio-video drift. The 25 ms limit equals one H3 audio-latent interval. FFmpeg writes
to a temporary target. The publisher atomically moves successful media and metadata into the
ComfyUI output directory and removes temporary WAV data on every exit path.

The first T2VA smoke test uses the unquantized reference precision policy. The transformer remains
BF16 except for the audited FP32 patch projections, timestep path, and output heads. Qwen3-VL uses
BF16. Final decoded media uses the host formats required by ComfyUI and FFmpeg. Weight and activation
quantization remain deferred performance experiments.

Near-term extension points are first/last-frame conditioning, multi-reference packing, component loaders, quantized transformers, progress and cancellation, preview decode, and long-video stitching.
