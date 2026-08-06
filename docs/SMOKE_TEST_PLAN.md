# First ComfyUI smoke-test plan

The first smoke test proves that ComfyUI can execute one synchronized MiniMax H3 video-and-audio
generation through the MLX engine. The smoke test is a wiring and lifecycle test, not a quality or
speed benchmark.

No gate requires a model download or expensive generation until the operator explicitly approves
the checkpoint, storage use, expected memory, resolution, duration, and sampling-step count.

## Complete component stack

| Component | First-test responsibility | Initial optimization policy |
| --- | --- | --- |
| Transformer | Jointly denoise packed video and audio rows. | Use the unquantized reference BF16/FP32 checkpoint policy. Precompute the adaptive layer normalization schedule and release its projection weights when reuse is safe. |
| Qwen3-VL encoder | Produce the layer-50 conditioning state from the prompt. | Load only required layers and heads. Omit the vision tower for text-only generation. Unload the encoder after conditioning when memory requires it. |
| Processor and tokenizer | Build the Qwen3-VL prompt representation. | Keep these assets on the CPU. Validate their files before loading weights. |
| Video VAE | Decode video latents to frames. | Keep final decode exact. Use tiled and bounded-batch decoding. Load after sampling or unload before sampling when staged residency reduces peak memory. |
| Audio VAE | Decode audio latents to 32 kHz stereo audio. | Keep the first test at the audited precision. Stage the component because its residency is small relative to the transformer and Qwen3-VL. |
| Scheduler and sigma mapping | Produce coordinated video and audio denoising schedules. | Keep arithmetic exact. Do not treat a sampling step as identical to a transformer evaluation. |
| Packed latent contract | Preserve text, condition, target-audio, and target-video ordering. | Validate row counts and timing before allocation. |
| Media writer | Mux synchronized frames and audio into an output file. | Write directly to the ComfyUI output directory. Avoid a full-media PyTorch copy. |
| ComfyUI adapter | Register nodes, report progress, handle cancellation, publish output, and release memory. | Keep imports lightweight. Convert host events at the adapter boundary. |

The FL2VA and Ref2VA release directories contain transformer, Qwen3-VL encoder, processor,
tokenizer, video VAE, and audio VAE components. Their weight payloads are byte-identical in the
official diffusers release; task metadata selects the supported conditioning contract. Converted or
quantized artifacts must retain the task and component metadata needed to reject incompatible
combinations.

## Gate 0 — freeze the first contract

Use text-to-video-plus-audio only. Use no image, video, or audio references. Resolve the exact
ComfyUI version, Apple Silicon hardware class, macOS version, Python version, MLX version, and model
artifact identities in the run record. Do not store machine-specific paths in the repository.

Success requires one workflow with these logical nodes:

```text
H3 component specification
  -> H3 component validation and memory estimate
  -> H3 text conditioning
  -> H3 synchronized sampling
  -> H3 video and audio decode
  -> H3 output publication
  -> H3 unload
```

## Gate 1 — import and graph validation

1. Install the custom node in a clean ComfyUI environment without model files.
2. Start ComfyUI.
3. Confirm that node import does not load MLX weights.
4. Confirm that every WeeTodd node is registered once.
5. Load the smoke-test workflow.
6. Confirm that missing components produce specific corrective errors before allocation.
7. Confirm that ComfyUI Manager installation and dependency resolution do not alter unrelated host packages.

Implementation status: automated tests import both the internal node module and the repository
ComfyUI entrypoint without importing MLX. The tests confirm unique registration for all fifteen
current nodes. A clean ComfyUI 0.30.0 checkout at commit `15989f8` loaded the eight-node example
without weights. Live `/object_info` validation confirmed all node types, required inputs, and link
types. The workflow files are `examples/t2va_smoke_workflow.json` and
`examples/t2va_smoke_api.json`.

## Gate 2 — component manifest and memory preflight

Implement immutable specifications for the transformer, Qwen3-VL encoder, processor/tokenizer,
video VAE, and audio VAE. Accept ComfyUI model roots and repository-relative component names. Do not
embed absolute paths.

Read safetensors headers and configuration files without mapping tensor payloads. Report:

- Component identity, precision, quantization recipe, task family, and file completeness.
- On-disk bytes and estimated resident bytes for each component.
- Adaptive layer normalization cache bytes.
- Packed text, audio, and video row counts.
- Estimated attention and VAE workspace.
- The staged-residency order and estimated peak unified memory.
- Available-memory headroom when the host exposes a dependable value.

Reject an incomplete, ambiguous, task-incompatible, or unsupported component set before loading
weights.

Implementation status: the component specification and header-only preflight nodes are implemented.
The current estimator reports component storage, removable adaptive layer normalization bytes,
packed rows, schedule-cache size, decode workspace, staged peaks, and optional supplied-memory
headroom. The estimate is conservative guidance, not a Metal allocation guarantee. Real-component
probe measurements must calibrate the workspace factors before the first generation.

Preflight rejects a native ComfyUI or experimental single-file artifact unless it contains the
self-describing metadata and tensor contract required by the MLX loader. A safetensors extension
alone does not establish MLX compatibility.

## Gate 3 — inexpensive engine tests

Use tiny synthetic modules and temporary files. Test component parsing, schedule construction,
packed shapes, callback cadence, cancellation, failure cleanup, cache reuse, output confinement,
and explicit unloading. Test text conditioning, video decode, and audio decode independently with
tiny configurations.

Do not use a real H3 checkpoint for this gate.

Implementation status: the text-only Qwen3-VL specification, process-local cache, encode node, and
explicit unload node are implemented. Synthetic tests cover compatible reuse, incompatible
replacement, unload-after-encode, failure cleanup, empty-prompt rejection, and weight-free imports.
The encoder accepts independent processor and tokenizer paths. Real MLX output-shape and memory
tests remain part of Gate 4.

The transformer-only sampler and unload node are also implemented. The sampler accepts text-only
conditioning, creates synchronized target video and audio rows, uses distinct video and audio sigma
shifts, reports every transformer evaluation, checks cancellation through the adapter callback, and
returns undecoded normalized latents. Synthetic lifecycle tests cover progress, schedule-safe reuse,
automatic unloading, failure cleanup, and task rejection. A tiny-config MLX sampling-shape test is
present and runs in the local Python 3.12 development environment.

The final video VAE decoder and unload node are implemented. The decoder accepts normalized H3
video latents, verifies the selected VAE identity, returns ComfyUI float RGB frames, and preserves
the synchronized audio stream in the original latent contract. Synthetic tests cover cache reuse,
unload-after-decode, provenance rejection, and failure cleanup. MLX 0.32.0 is installed in the local
Python 3.12 development environment. The selected compact video VAE passed a strict loader probe
with 24 latent channels and a 17-frame clip structure.

The final audio VAE decoder and unload node are implemented. The decoder verifies audio VAE
provenance and the 32 kHz sample-rate contract. The adapter returns current ComfyUI `AUDIO` data with
shape `(1, 2, samples)`. Metadata retains sample count, audio duration, video frame count, and video
frame rate. Synthetic tests cover compatible reuse, automatic unloading, provenance rejection,
sample-rate rejection, cancellation, and failure cleanup. The selected compact audio VAE passed a
strict loader probe with 32 latent channels and a 32 kHz sample rate.

The synchronized publication node is implemented. The node accepts decoded ComfyUI `IMAGE` and
`AUDIO`, validates the H3 24 fps and 32 kHz timing contract, permits at most one 40 Hz audio-latent
interval of drift, and writes collision-safe MP4 and JSON outputs under ComfyUI. Publication uses
temporary files and removes partial video, metadata, and WAV data after failure or cancellation.
Synthetic publication tests do not require model weights.

## Gate 4 — real-component probes

Run probes only after the operator approves the selected local artifacts.

1. Load and encode one short text prompt with Qwen3-VL. Record peak memory and output shape.
2. Release Qwen3-VL if staged residency is selected.
3. Load the transformer and run one deterministic forward-shape probe.
4. Build the full sampling schedule and validate adaptive layer normalization reuse.
5. Decode tiny synthetic video latents with the final video VAE.
6. Decode tiny synthetic audio latents with the final audio VAE.
7. Release each component and confirm MLX cache cleanup.

Each probe must be independently cancellable. A probe failure must not force a complete generation.

Implementation status: isolated strict loader probes passed for the compact audio VAE, compact
video VAE, compact Q8 Qwen3-VL text encoder, and pruned BF16-class transformer. The Qwen3-VL probe
loaded 50 of 64 layers without the vision tower. The compact encoder requires the authentic nested
Qwen3-VL architecture config from the selected checkpoint metadata; the adapter now resolves that
config explicitly. The transformer probe confirmed 50 layers, hidden size 5376, and a 1001-point
AdaLN curve. No prompt encoding, transformer forward, VAE decode, or generation ran.

For the five-second, eight-step, 640 by 384 wiring test, header preflight estimates a 42.388 GB
staged peak. The selected local component set is mixed precision: a pruned BF16/FP32 transformer,
Q8 compact Qwen3-VL weights, FP16 video VAE weights, and FP32 audio VAE weights. This set does not
satisfy the preferred unquantized BF16-class first-baseline policy. Operator approval must address
that difference before generation.

## Gate 5 — first ComfyUI generation

Use a five-second text-only request, the smallest validated canvas, and the smallest validated
sampling schedule. A sub-native canvas is acceptable for this smoke test when the workflow labels
it as an off-distribution wiring test. Explain the quality limitation before execution.

Success requires:

- ComfyUI progress updates for every transformer evaluation.
- Immediate interruption at the next safe callback.
- One playable output with synchronized video and stereo audio.
- Collision-safe output naming under the ComfyUI output directory.
- A metadata sidecar with component identities, quantization recipes, resolved dimensions, frame
  count, duration, seed, schedules, timings, software versions, and cleanup policy.
- No stale temporary media after success, cancellation, or failure.
- Explicit unload followed by observed MLX cache cleanup.

## Optimization order

1. Reduce Qwen3-VL residency through exact layer truncation, optional vision loading, quantized
   weights, and post-encode unloading.
2. Reduce transformer residency through safe adaptive layer normalization pruning and independently
   validated MLX weight quantization.
3. Reduce peak memory through staged component residency before changing final-decode arithmetic.
4. Tune exact video VAE tiling and decode batch size. Treat quantized video VAE weights as an
   experimental quality path.
5. Keep the audio VAE at the audited precision for the first generation. Quantize it only after
   waveform and synchronization parity tests exist.
6. Add a tiny video autoencoder for previews only. Never use it for final output.
7. Benchmark fused or quantized attention after the exact dense path passes the first generation.
8. Consider trajectory forecasting only after deterministic baselines and quality gates exist.

## Stop conditions

Stop before generation when estimated peak memory lacks safe headroom, a component identity is
ambiguous, task metadata does not match text-to-video-plus-audio, cancellation is not functional,
or final video and audio decoding have not passed independent probes.
