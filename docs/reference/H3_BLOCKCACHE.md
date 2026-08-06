# H3 BlockCache

## Purpose

WeeTodd BlockCache reduces repeated MiniMax H3 transformer work while retaining the graph-based
ComfyUI workflow contract. The user selects BlockCache with a separate node and connects the node
to the H3 sampler. EasyCache and BlockCache are mutually exclusive.

## Engine contract

BlockCache always evaluates transformer block zero. The engine measures the block-zero residual
for video and audio target rows separately. The larger relative-change score controls reuse, so a
stable video stream cannot hide a changing audio stream and a stable audio stream cannot hide a
changing video stream.

On a cache miss, the engine evaluates the remaining transformer blocks. The engine caches the
post-block-zero to post-stack residual for the video and audio target rows. On a cache hit, the
engine adds those residuals to the current block-zero output. The engine then evaluates the current
final normalization and output heads. Both H3 schedulers advance after every sampling step.

The cache belongs to one sampling request. The sampler releases the cache with the transformer on
success, failure, or cancellation. Generation metadata records the policy, resolved threshold,
hit count, and cache size.

## Automatic policies

Conservative auto allows no consecutive hits and limits hits to 25 percent of the sampling steps.
Balanced auto allows no consecutive hits and limits hits to 35 percent. Speed auto allows two
consecutive hits and limits hits to 50 percent. Each policy protects the final sampling step.

The automatic threshold uses the first eligible live change score. Each policy applies a bounded
multiplier and records the resolved threshold. These controls bound reuse frequency; they do not
prove perceptual equivalence.

## 384P benchmark

The benchmark uses five seconds, 640 by 384 pixels, seed zero, the approved H3 audiovisual prompt,
the BF16 pruned transformer, the Q8 text encoder, and the final video and audio variational
autoencoders (VAEs). Each weighted stage unloads after use. The matrix covers conservative,
balanced, and speed policies at 8, 12, 16, and 20 requested sampling steps. The existing no-cache
measurements provide the baseline because all generation inputs and component identities match.

![BlockCache policy and step-scaling chart](../../benchmarks/artifacts/charts/h3_blockcache_policy_step_scaling_384p.svg)

| Policy | Steps | Full evaluations | BlockCache hits | Sampling time | Workflow time |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 8 | 7 | 0 | 165.56 s | 191.51 s |
| None | 12 | 11 | 0 | 264.03 s | 292.02 s |
| None | 16 | 15 | 0 | 369.94 s | 396.37 s |
| None | 20 | 19 | 0 | 456.25 s | 482.74 s |
| Conservative | 8 | 6 | 1 | 139.55 s | 166.48 s |
| Conservative | 12 | 9 | 2 | 213.21 s | 240.89 s |
| Conservative | 16 | 12 | 3 | 285.10 s | 312.98 s |
| Conservative | 20 | 15 | 4 | 356.74 s | 384.86 s |
| Balanced | 8 | 5 | 2 | 119.61 s | 146.18 s |
| Balanced | 12 | 8 | 3 | 192.21 s | 218.58 s |
| Balanced | 16 | 10 | 5 | 242.37 s | 269.01 s |
| Balanced | 20 | 13 | 6 | 318.64 s | 345.24 s |
| Speed | 8 | 4 | 3 | 99.05 s | 125.76 s |
| Speed | 12 | 6 | 5 | 147.61 s | 174.10 s |
| Speed | 16 | 9 | 6 | 221.51 s | 247.88 s |
| Speed | 20 | 10 | 9 | 246.84 s | 273.23 s |

At 20 steps, conservative reduced sampling time by 21.8 percent. Balanced reduced sampling time by
30.2 percent. Speed reduced sampling time by 45.9 percent. The corresponding complete-workflow
savings were 20.3, 28.5, and 43.4 percent.

A full transformer evaluation took approximately 23.2 through 24.4 seconds. A BlockCache hit took
approximately 0.5 seconds because block zero and the current output heads still ran. The cached
video and audio block-tail residuals used 99,929,088 bytes, or approximately 95.3 MiB.

All 12 BlockCache generations produced one H.264 video stream and one Advanced Audio Coding (AAC)
stream. Every output contained 124 frames, 32 kHz stereo audio, and 0.0083 seconds of reported
audio-video drift. The metadata preserved the exact prompt, component identities, policy, resolved
threshold, hit count, and cache size.

The endpoint contact sheet contains the requested red robot and raised-hand final state across the
matrix. Endpoint inspection does not establish full motion, detail, or audio equivalence.

## Limitations

The 640 by 384 canvas is an off-distribution smoke-test resolution. One prompt and one seed cannot
establish visual, motion, audio, or synchronization equivalence. Queue order is not randomized, so
thermal state can affect runtime comparisons.

The BlockCache runs selected the explicit custom 640 by 384 canvas. The no-cache baseline selected
the 384P preset and 5:3 ratio, which resolves to the same 640 by 384 engine dimensions. The selector
metadata differs, but the sampler receives the same width and height.

## Design provenance

The unlicensed T8mars BlockCache repository was used only as a design reference. The independent
WeeTodd implementation uses MLX arrays, the existing H3 packed-sequence boundary, and the project
lifecycle contract. See [attribution](../ATTRIBUTION.md).
