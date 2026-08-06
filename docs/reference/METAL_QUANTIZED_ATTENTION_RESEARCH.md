# Metal quantized attention research note

Date: 2026-08-05

This note records design research only. It does not import Draw Things code, binaries, model
recipes, or weights.

## What the Draw Things release describes

Draw Things reports two separate optimizations on M5-class Apple Silicon:

1. A fused attention operator accepts FP16/BF16 Q, K, and V, dynamically quantizes Q/K with
   row-group scales and V with row-wise affine parameters, performs most attention arithmetic in
   Int8, and dequantizes after accumulation. They report a 1.24–1.41x improvement over their own
   Metal Flash Attention implementation.
2. A fused dynamic-activation-quantization plus Int8 GEMM path is enabled only for their “8-bit S”
   models. They report 1.61–1.87x over their FP16 GEMM baseline for that kernel.

The release reports M5 results. It does not establish equivalent gains on M1–M4, in MLX, or for
MiniMax H3's exact head count, head width, and packed sequence distribution. Its end-to-end numbers
also combine both mechanisms on eligible models, so they cannot be transferred directly to H3.

## What maps to H3

H3 uses full self-attention over one packed text, conditioning, stereo-audio, and video sequence.
After Q/K RMSNorm and MM-RoPE, its attention tensors are `(B, 56, S, 128)`. This regular head shape
is a plausible target for row-scaled dynamic quantization, and H3 has no causal mask to complicate
an initial dense kernel. The quantization must occur after normalization and rotary application;
moving it earlier changes the operation being approximated.

The opportunity grows with duration. Existing M4 Max measurements attribute about 24% of a
5-second step to attention, but about 51% at the measured 10-second packed length. Even a 1.4x
attention-only gain therefore has an approximate Amdahl ceiling of 1.08x at 5 seconds and 1.17x at
10 seconds, before launch/fusion effects. It is valuable but does not by itself make native H3 fast.

The second mechanism does not map to the project's existing affine Q4/Q8 implementation. MLX
affine `QuantizedLinear` is weight-only by default and was measured slower than BF16 at H3's large-M
shapes on M4 Max. Current MLX separately supports activation quantization only for `mxfp8` and
`nvfp4` linear modes. Those formats require their own conversion, compatibility, accuracy, and
hardware benchmark; they are not evidence that affine W4A8 will be fast.

## Proposed independent experiment

### Phase 0: capability and ceiling

- Record chip generation, GPU cores, OS, MLX version, supported quantization modes, and thermal
  state. Never generalize M5 Neural Accelerator results to earlier chips.
- Benchmark current `mx.fast.scaled_dot_product_attention` at H3 shapes spanning approximately
  5.5k, 7.7k, 13k, and 25k packed rows.
- Measure quantize/dequantize bandwidth separately. Reject the idea early if those passes cannot be
  fused or already consume the projected attention saving.

### Phase 1: numerical reference

- Implement a slow MLX reference that quantizes post-RMSNorm/post-RoPE Q/K per row group and V per
  row, accumulates scores and softmax in a stable floating-point reference, and reports maximum
  error, cosine similarity, attention-output SNR, and per-head outliers.
- Sweep symmetric versus affine V, Q/K group sizes, scale precision, and accumulator precision.
- Test modality slices separately because audio, text, and video share attention but may have
  different activation distributions.

### Phase 2: fused prototype

- Only after the reference passes, build an MLX custom Metal kernel behind an internal backend
  interface. Preserve a BF16 fallback and runtime capability check.
- Use online softmax so the full `S x S` score matrix is never materialized. Quantization,
  score accumulation, softmax statistics, V accumulation, and output dequantization should be
  evaluated as one fusion problem; separate MLX operations are unlikely to reproduce the claimed
  benefit.
- Benchmark warm execution, compilation latency, peak unified memory, and sustained multi-minute
  behavior. H3 steps are long enough that thermal behavior matters.

### Phase 3: generation gate

- Compare deterministic BF16 and quantized-attention outputs at matched seeds for T2VA and FL2VA,
  then Ref2VA when implemented.
- Evaluate faces, fine motion, object count/permanence, lip/audio synchronization, dialogue level,
  and first/last-frame adherence. Tensor similarity alone is insufficient across repeated denoising
  steps.
- Expose the backend only when it wins end to end on a named hardware class. Default to `auto`, with
  an explicit BF16 fallback and metadata recording the selected kernel and quantization parameters.

## New architectural implication

Attention backends and checkpoint quantization should be independent specifications. A model loader
should describe stored weights and linear compute; a sampler/backend node should select dense BF16,
dynamic Int8 attention, or a future verified sparse implementation. This prevents filenames such as
“8-bit” from obscuring whether the weights, activations, attention operands, or all three are
quantized.
