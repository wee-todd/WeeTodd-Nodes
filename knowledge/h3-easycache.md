---
type: Experiment
title: H3 EasyCache
description: Joint MLX residual reuse for MiniMax H3 video and audio transformer evaluations.
resource: ../docs/reference/H3_EASYCACHE.md
tags: [minimax-h3, mlx, easycache, performance]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-06T06:50:00-07:00
sources:
  - id: comfyui-easycache
    resource: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_easycache.py
    title: ComfyUI EasyCache implementation
  - id: h3-easycache-policy-step-scaling-report
    resource: ../docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING.md
    title: H3 EasyCache policy and step scaling
  - id: h3-easycache-policy-step-scaling-data
    resource: ../benchmarks/h3_easycache_policy_step_scaling.csv
    title: H3 EasyCache policy and step-scaling measurements
  - id: h3-easycache-policy-step-scaling-768p-report
    resource: ../docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING_768P.md
    title: H3 EasyCache policy and step scaling at 768P
  - id: h3-easycache-policy-step-scaling-768p-data
    resource: ../benchmarks/h3_easycache_policy_step_scaling_768p.csv
    title: H3 EasyCache 768P policy and step-scaling measurements
---

# Contract

Cache video and audio prediction residuals together. Use one joint reuse decision. Advance both H3
schedulers when a transformer evaluation is skipped.

# Lifecycle

Create cache state for one sampling request. Release cache state with the transformer on success,
failure, or cancellation.

# Automatic policies

Protect two calibration evaluations and the final evaluation. Derive a bounded threshold from the
live joint change estimate. Record the policy, resolved threshold, and skipped count in metadata.

Conservative auto permits no consecutive skips and uses the configured skip cap. Balanced auto
permits no consecutive skips and up to 35 percent skipped evaluations. Speed auto permits two
consecutive skips and up to 50 percent skipped evaluations. The user must select each policy.

# First result

The official threshold 0.20 skipped zero of seven real transformer evaluations. The cached and
uncached MP4 files were byte-identical. The test does not show a speed improvement.

# Automatic results

Conservative auto skipped one of seven evaluations and sampled in 141.00 seconds. Balanced auto
skipped two evaluations and sampled in 116.58 seconds. Speed auto skipped three evaluations and
sampled in 100.15 seconds. The uncached sampling time was 165.82 seconds for the same request.

The balanced-auto and speed-auto contact sheets contained the requested action sequence. Visual
inspection does not prove motion or audio parity.

# Balanced step scaling

The [balanced step-scaling benchmark](../docs/reference/H3_EASYCACHE_BALANCED_STEP_SCALING.md)
measured 8, 12, 16, and 20 requested steps. Balanced auto cached 27 through 33 percent of scheduled
transformer opportunities. Sampling time increased approximately 16.20 seconds per requested step.

The AdaLN schedule cache increased from 126 MB at eight steps to 358 MB at 20 steps. The resolved
reuse threshold decreased as the schedule became denser.

# Policy step scaling

The [four-policy scaling benchmark](../docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING.md) compared
no cache, conservative, balanced, and speed at 8, 12, 16, and 20 requested steps. The sampling-time
slopes were 24.45, 18.02, 16.20, and 11.59 seconds per requested step, respectively.

Conservative saved 13 through 24 percent of uncached sampling time. Balanced saved 26 through 32
percent. Speed saved 40 through 49 percent. One run produced each point, so the measurements do not
establish perceptual quality or statistical variance.

## Benchmark conditions

Each generation used five seconds, 640 by 384 pixels, seed zero, the same H3 audiovisual prompt,
the BF16 pruned transformer, the Q8 text encoder, and the same final video and audio VAEs. Each
weighted stage unloaded after use. The benchmark ran no cache and all three automatic policies at
8, 12, 16, and 20 requested sampling steps.

## Complete measurements

| Policy | Steps | Transformer evaluations | Cached evaluations | Sampling time | Workflow time |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 8 | 7 | 0 | 165.56 s | 191.51 s |
| None | 12 | 11 | 0 | 264.03 s | 292.02 s |
| None | 16 | 15 | 0 | 369.94 s | 396.37 s |
| None | 20 | 19 | 0 | 456.25 s | 482.74 s |
| Conservative | 8 | 6 | 1 | 141.00 s | 174.74 s |
| Conservative | 12 | 9 | 2 | 229.47 s | 256.55 s |
| Conservative | 16 | 12 | 3 | 282.81 s | 308.76 s |
| Conservative | 20 | 15 | 4 | 363.49 s | 390.38 s |
| Balanced | 8 | 5 | 2 | 116.58 s | 161.14 s |
| Balanced | 12 | 8 | 3 | 196.22 s | 225.50 s |
| Balanced | 16 | 10 | 5 | 259.09 s | 288.91 s |
| Balanced | 20 | 13 | 6 | 311.59 s | 339.61 s |
| Speed | 8 | 4 | 3 | 100.15 s | 128.94 s |
| Speed | 12 | 6 | 5 | 151.82 s | 179.10 s |
| Speed | 16 | 8 | 7 | 187.62 s | 213.61 s |
| Speed | 20 | 10 | 9 | 242.78 s | 269.65 s |

At 20 steps, conservative reduced sampling time by 20.3 percent, balanced reduced sampling time by
31.7 percent, and speed reduced sampling time by 46.8 percent. The corresponding complete-workflow
savings were 19.1, 29.6, and 44.1 percent.

The AdaLN schedule cache used approximately 126 MB at 8 steps, 203 MB at 12 steps, 271 MB at 16
steps, and 358 MB at 20 steps.

# Native 768P policy step scaling

The [native 768P benchmark](../docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING_768P.md) repeated the
complete matrix at 1344 by 768 pixels. All other generation conditions remained equal to the smoke-
resolution benchmark.

| Policy | Steps | Transformer evaluations | Cached evaluations | Sampling time | Workflow time |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 8 | 7 | 0 | 1322.97 s | 1448.11 s |
| None | 12 | 11 | 0 | 1818.87 s | 1928.98 s |
| None | 16 | 15 | 0 | 2480.15 s | 2590.66 s |
| None | 20 | 19 | 0 | 3139.98 s | 3256.57 s |
| Conservative | 8 | 6 | 1 | 1001.85 s | 1111.53 s |
| Conservative | 12 | 9 | 2 | 1491.06 s | 1598.02 s |
| Conservative | 16 | 12 | 3 | 1980.13 s | 2088.79 s |
| Conservative | 20 | 15 | 4 | 2494.27 s | 2604.63 s |
| Balanced | 8 | 5 | 2 | 810.47 s | 918.39 s |
| Balanced | 12 | 8 | 3 | 1329.48 s | 1435.85 s |
| Balanced | 16 | 10 | 5 | 1643.93 s | 1753.07 s |
| Balanced | 20 | 13 | 6 | 2175.15 s | 2281.12 s |
| Speed | 8 | 4 | 3 | 666.57 s | 776.06 s |
| Speed | 12 | 6 | 5 | 984.29 s | 1093.20 s |
| Speed | 16 | 8 | 7 | 1317.57 s | 1434.05 s |
| Speed | 20 | 10 | 9 | 1668.15 s | 1776.75 s |

At 20 steps, conservative reduced sampling time by 20.6 percent, balanced reduced sampling time by
30.7 percent, and speed reduced sampling time by 46.9 percent. The corresponding complete-workflow
savings were 20.0, 30.0, and 45.4 percent.

After the first cold run, a native-resolution transformer evaluation took approximately 162
through 171 seconds. An evaluation at 1344 by 768 cost approximately 6.8 times an evaluation at 640
by 384, although the native canvas contains approximately 4.2 times as many pixels. Dense attention
produces a superlinear resolution cost.

Endpoint inspection found the requested subject and final-wave state in all 16 native-resolution
outputs. Endpoint inspection does not establish full motion, detail, or audio equivalence.

## Validation evidence

The validation process matched all 16 CSV rows to published metadata sidecars. FFprobe found one
video stream and one audio stream in every published MP4 file. The project validation passed 84
inexpensive tests, OKF validation, Markdown lint, Ruff, Python compilation, and Git whitespace
checks. The Torch-dependent packing-parity test did not run in the MLX project environment.

# Required validation

Compare all policies across more prompts and seeds. Measure video motion, audio quality,
synchronization, skipped evaluations, and total time. Randomize queue order to control for cold-run
and thermal effects.
