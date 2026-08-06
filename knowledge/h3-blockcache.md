---
type: Experiment
title: H3 BlockCache
description: Block-level residual reuse for the joint MiniMax H3 MLX transformer.
resource: ../docs/reference/H3_BLOCKCACHE.md
tags: [minimax-h3, mlx, blockcache, performance]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-06T11:35:00-07:00
sources:
  - id: t8mars-h3-blockcache
    resource: https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8
    title: T8mars MiniMax H3 BlockCache
  - id: h3-blockcache-report
    resource: ../docs/reference/H3_BLOCKCACHE.md
    title: WeeTodd H3 BlockCache design and benchmark
  - id: h3-blockcache-data
    resource: ../benchmarks/h3_blockcache_policy_step_scaling_384p.csv
    title: H3 BlockCache 384P policy and step-scaling measurements
---

# Contract

Evaluate transformer block zero on every sampling step. Measure video and audio target-row changes
separately. Use the larger score for the cache decision. Evaluate the current output heads after a
cache hit.

# Lifecycle

Create BlockCache state for one sampling request. Release BlockCache state with the transformer on
success, failure, or cancellation. Reject a graph that connects EasyCache and BlockCache together.

# Evidence

The focused tests verify separate video and audio residual reconstruction, worst-modality change
selection, policy bounds, cache-size reporting, and the mutually exclusive node contract.

The 640 by 384 matrix covered conservative, balanced, and speed policies at 8, 12, 16, and 20
requested sampling steps. The existing no-cache measurements provided the baseline under the same
generation contract.

At 20 steps, conservative reduced sampling time by 21.8 percent. Balanced reduced sampling time by
30.2 percent. Speed reduced sampling time by 45.9 percent. The video and audio cache used
99,929,088 bytes.

All 12 BlockCache outputs contained synchronized H.264 video and AAC audio. Metadata validation
confirmed the exact prompt, dimensions, frame count, component identities, policy, hit count, and
cache size.

# Limitations

BlockCache approximates the skipped transformer blocks. A bounded hit count does not prove output
equivalence. Validate motion, detail, audio quality, synchronization, and runtime across more
prompts and seeds.

The BlockCache runs used explicit custom dimensions. The no-cache baseline used the preset selector
that resolves to the same 640 by 384 sampler dimensions.
