# Prioritized implementation plan

The gated execution sequence is defined in the [first ComfyUI smoke-test plan](SMOKE_TEST_PLAN.md).

The first working T2VA target uses the unquantized reference BF16/FP32 precision policy. Preserve
the reference FP32 stability exceptions. Defer weight and activation quantization until the first
ComfyUI smoke generation succeeds. Continue performance work that does not change numerical
precision, including staged residency, exact layer truncation, bounded VAE batches, and allocation
reduction.

## P0 — make the existing T2VA contract dependable

1. Add a component manifest/specification that validates `model_index.json`, the task family,
   tokenizer/processor, text encoder, transformer, visual VAE, and audio VAE before allocation.
   Treat processor/tokenizer assets, scheduler logic, packing, and media publication as required
   pipeline components even though they are not model weights.
2. Add explicit pipeline state reporting and memory estimates from tensor headers, including peak
   load headroom, AdaLN cache size, packed row count, and estimated decode workspace.
3. Make AdaLN-drop reuse correct: reload automatically when a new schedule cannot be built after
   projections were dropped, or introduce a schedule-independent pruned-AdaLN component contract.
4. Complete ComfyUI lifecycle integration: progress, cancellation, preview events, failure cleanup,
   and collision-safe output naming with workflow metadata embedded or stored beside the media.
5. Add tiny-model engine tests for callback cadence, cancellation, cache reuse, task rejection, and
   synchronized output lengths. Keep real-checkpoint parity opt-in.

## P1 — composable loading and FL2VA

1. Implement separate immutable specs for the transformer, text encoder/processor, visual VAE, and
   audio VAE; compose them in an MLX H3 pipeline loader without loading weights during node import.
2. Add a quantized transformer loader that reads and validates `quant_config.json`; report its
   recipe and reject incompatible or ambiguous single-file exports before allocation.
3. Add first-frame and last-frame conditioning builders, followed by a combined first/last builder.
   Preserve the first-frame stretch, last-frame cover-crop, seeded posterior sampling, float16
   round-trip, noise augmentation, and anchor positions already defined by the engine contract.
4. Split generation into conditioning, sampling, decode, preview/save, and unload nodes while
   keeping one synchronized audio/video result object across those connections.

## P2 — Ref2VA and operator feedback

1. Extend the engine packing contract for ordered image, video, video-with-audio, and audio reference
   blocks. Enforce the official counts, 2–15 second reference limits, total-duration limits, and the
   rule that audio cannot be the sole reference.
2. Match MiniMax presentation ordering and prompt labels (`<Picture i>`, `<Video k>`, `<Audio j>`),
   including 2 fps visual sampling for the text encoder and 24 fps VAE input.
3. Add periodic low-cost preview decoding behind an explicit interval/quality control. Preview data
   should be disposable and must not force the final media through PyTorch.
4. Add memory-pressure policies: keep warm, unload selected components after encode/decode, or full
   release. Report actual MLX cache cleanup and distinguish resident weights from peak workspace.

## P3 — packaging and workflows

1. Add generation metadata covering checkpoint identity, component/quant recipes, task, prompt,
   resolved frame count and duration, dimensions, schedules, seed, timings, and software versions.
2. Ship minimal T2VA, first-frame, last-frame, first/last, image-reference, video-reference, and
   audio+visual-reference workflow examples after their node contracts stabilize.
3. Register a real Comfy Registry publisher, add valid `[tool.comfy]` metadata and a publish ignore
   list, then test clean installation through current ComfyUI Manager on Apple Silicon.
4. Add compatibility CI for the oldest supported Python/MLX set and a current ComfyUI checkout.

## Deferred experimental work

- Step-cache and long-video experiments remain behind exact fallbacks and explicit quality labels.
- Benchmark dynamic Int8 attention as a distinct backend experiment. Quantize normalized/rotated
  BF16 Q and K with row-group scales, V with row-wise affine parameters, retain stable online
  softmax accumulation, and dequantize only the accumulated output. Start with synthetic H3 shapes
  and require parity plus end-to-end quality gates before exposing any node control.
- Benchmark MLX `mxfp8` and `nvfp4` input quantization separately from the existing affine Q4/Q8
  weight recipes. Record hardware/OS gating, peak memory, compile cost, and per-layer exclusions;
  never label an artifact “W4A8” unless both weight and activation formats are specified precisely.
- Develop H3-specific mixed-precision allocation inspired by sensitivity-driven quantization. Use
  cached block inputs and teacher outputs at representative denoising timesteps/modalities instead
  of autoregressive logit KL or a full generation per layer. Allocate a target bits-per-weight
  budget only after whole-trajectory and synchronized audio/video quality validation.
- Sparse attention waits for a compatible published MiniMax implementation; dense-attention cost
  must remain visible in estimates and documentation.
