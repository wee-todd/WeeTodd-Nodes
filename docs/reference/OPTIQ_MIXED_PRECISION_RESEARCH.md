# OptiQ mixed-precision research note

Date: 2026-08-05

This note evaluates MLX OptiQ as a design reference for MiniMax H3. OptiQ is not added as a runtime
dependency, and its implementation or LLM-specific recipes are not copied.

## What OptiQ contributes

OptiQ measures the effect of quantizing one layer at a time, ranks candidates by output-distribution
KL divergence, then uses a bit-budget allocator to assign mixed per-layer precision. Its default
candidate set is 4/8-bit affine weights with group size 64 and a target average bits per weight. It
also records the assignment and sensitivity table in model metadata, builds a uniform baseline,
and evaluates the result rather than treating successful conversion as proof of quality.

Those product choices are directly useful to WeeTodd:

- Separate measurement, allocation, conversion, and evaluation.
- Compare mixed precision with a uniform quant at the same size.
- Preserve a machine-readable per-module recipe and the sensitivity evidence that produced it.
- Protect structurally sensitive and inexpensive modules instead of forcing every matrix to the
  lowest width.
- Optimize for a target resident-size/BPW budget, not a misleading “4-bit model” label.

## Why OptiQ is not directly applicable

OptiQ's converter and metric assume an autoregressive `mlx-lm` model with token logits. H3 is a
joint audio/video diffusion transformer whose output is a velocity prediction over packed modality
rows. It has no KV cache during denoising and no language-model output distribution on which to
compute the same KL score.

A direct layer sweep would also be economically wrong. H3 has 50 large blocks, and one native
forward can take tens to hundreds of seconds depending on packed length. Running every candidate
bit width across multiple samples and timesteps as full-model probes would take days or weeks and
would repeatedly allocate a model whose BF16 form is already difficult to hold.

The text encoder is Qwen3-VL-derived, but H3 consumes its layer-50 hidden state rather than its LM
head logits. An LLM quant that preserves text-generation accuracy is therefore not automatically a
valid H3 conditioner quant. Vision-token and prompt hidden-state parity must be measured at the
actual truncation boundary.

## H3-specific sensitivity design

### Calibration corpus

Use a small, redistribution-safe manifest spanning the distributions H3 must preserve:

- Text-only prompts with dialogue, motion, multiple subjects, camera movement, and sound events.
- First-frame and first/last-frame cases once FL2VA is exposed.
- Image, video, and audio references once Ref2VA exists.
- Short and long packed sequences, since attention share and error propagation change with length.

Store prompts and synthetic/reference identifiers, not generated media or private inputs.

### Probe points

Sample the actual denoising trajectory rather than only the initial noise:

- Early/high-noise, middle, and late/low-noise steps.
- Video, audio, and conditioning rows separately.
- At least two packed lengths and representative spatial geometries.

AdaLN requires special handling. Its large projections are precomputed and dropped, so their
sensitivity should be measured on the resulting modulation tables across the whole schedule. The
existing evidence that 8-bit AdaLN is acceptable remains a separate result from core DiT weights.

### Cheap block replay

Run a small number of BF16 teacher forwards and capture each block's input, modulation tensors, and
BF16 output at the selected probe points. Then replay one block locally with one candidate module
quantized at a time. This changes calibration cost from repeated full 50-block generations to many
single-block evaluations and makes 4/6/8-bit comparisons practical.

For each candidate record:

- Normalized MSE and cosine similarity of the block residual output.
- Q/K/V and FFN branch error before the residual addition.
- Error split by text, condition-video, target-video, and audio rows.
- Maximum/outlier error, not only a mean that can hide audio or reference failures.
- Resident bytes and measured block time at the real H3 row counts.

Block replay is a ranking signal, not final validation. Residual errors can compound coherently over
50 blocks and repeated denoising steps.

### Allocation

Start from the lowest candidate precision and upgrade the module with the largest reduction in
weighted sensitivity per added byte until the target resident budget is reached. The weighting must
include modality and timestep coverage. Enforce structural floors initially:

- Keep timestep embedding, input patch projections, condition projection, Q/K norms, final norm,
  and video/audio output heads at their audited precision.
- Treat QKV, attention output, FFN input, and FFN output as separate choices rather than assigning
  one width to an entire block.
- Protect the first and last blocks until measurements demonstrate otherwise.
- Keep the text encoder, DiT, video VAE, and audio VAE as independent quantization domains.

The first useful target is not the smallest artifact. It is a mixed recipe that fits the complete
T2VA pipeline on a 32 GB Mac with enough working headroom for a 5-second wiring/quality run.

## Required final validation

Every proposed recipe must be compared with BF16 and a uniform quant at equal or lower resident
size. Validation must include:

- Transformer/block numerical probes across the stored calibration manifest.
- Deterministic short T2VA generations followed by FL2VA and Ref2VA coverage.
- Faces, object count/permanence, motion, conditioning adherence, speech clarity, audio level, and
  audio/video synchronization.
- Cold-load peak, warm resident size, seconds per step, decode workspace, and unload behavior.
- Exact source revisions, MLX version, bit map, group sizes, excluded modules, and calibration hash.

Because existing M4 Max measurements show affine Q4/Q8 matmuls slower than BF16 at H3's large-M
shapes, mixed precision is initially a memory product. Speed claims require independent end-to-end
measurement on each supported Apple GPU generation.
