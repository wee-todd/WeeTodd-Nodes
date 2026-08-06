# WeeTodd Nodes

Experimental ComfyUI nodes for running MiniMax H3 natively through MLX on Apple Silicon.

This project begins with the Apache-2.0 MiniMax H3 MLX engine developed alongside Phosphene, then exposes it through small, composable ComfyUI nodes. The goal is a Kijai-style H3 suite for Mac users: explicit loaders, reusable model state, generation controls, conditioning tools, memory controls, previews, quantization, and workflow examples.

## Current nodes

- **WeeTodd H3 Component Loader** creates a lazy specification for the transformer, Qwen3-VL,
  processor, tokenizer, video VAE, and audio VAE.
- **WeeTodd H3 Component Preflight** validates the manifest, component files, task, configuration,
  and quantization recipe. It estimates staged memory without loading tensor payloads.
- **WeeTodd H3 Text Encode (Qwen3-VL)** produces text-only conditioning from the required
  unnormalized layer-50 state. It omits the vision tower and can unload Qwen3-VL after encoding.
- **WeeTodd H3 Unload Qwen3-VL** explicitly releases a warm process-local conditioner.
- **WeeTodd H3 Sample Video + Audio Latents** runs the transformer over one synchronized packed
  sequence and returns undecoded MLX video and audio latents.
- **WeeTodd H3 Unload Transformer** releases the transformer-only sampler and MLX cache.
- **WeeTodd H3 Decode Video VAE** converts normalized video latents to ComfyUI `IMAGE` frames. It
  checks component provenance and can unload the final video VAE after decoding.
- **WeeTodd H3 Unload Video VAE** explicitly releases a warm process-local video decoder.
- **WeeTodd H3 Decode Audio VAE** converts normalized audio latents to ComfyUI `AUDIO` with one
  batch, two channels, and a 32 kHz sample rate. It retains synchronized timing metadata.
- **WeeTodd H3 Unload Audio VAE** explicitly releases a warm process-local audio decoder.
- **WeeTodd H3 Publish Video + Audio** validates synchronized ComfyUI images and audio, writes a
  collision-safe MP4 atomically, removes temporary files, and emits a JSON metadata sidecar.
- **WeeTodd H3 Model Loader (MLX)** describes a full or quantized checkpoint and loads it lazily.
- **WeeTodd H3 Generation Config** selects a resolution tier and aspect ratio, then validates the
  resolved canvas, duration, steps, seed, and AdaLN behavior. Exact custom dimensions remain
  available as an advanced option.
- **WeeTodd H3 Generate Video + Audio** produces a synchronized MP4 and JSON sidecar.
- **WeeTodd H3 Unload MLX Runtime** releases the warm pipeline and cached MLX allocations.

The first release supports text-to-video-plus-audio. First/last-frame and multi-reference nodes are next; the engine already contains much of the keyframe machinery.

## Install for development

Clone into `ComfyUI/custom_nodes/WeeTodd-Nodes`, then use the same Python environment as ComfyUI:

```bash
python .agents/skills/python-environment-preflight/scripts/preflight.py \
  --project . --python python --require-architecture arm64
python -m pip install -e .
```

Read the [Python environment policy](docs/PYTHON_ENVIRONMENT.md) before replacing a venv or changing
ComfyUI dependencies.

Restart ComfyUI and look under `WeeTodd/H3`.

## Reality check

H3 is a 33B joint audio/video diffusion transformer plus a large text encoder and two VAEs. MiniMax's initial open release uses dense attention. Native-resolution generation is extremely compute-intensive even on large Apple Silicon systems. Start with the `384P (fast smoke)` tier, 5 seconds, and a low step count to verify wiring before increasing quality.

Resolution selection follows the hosted-node style while resolving to local H3 canvases. Choose a
quality tier (`384P`, `512P`, `768P`, or experimental `2K`) and then choose an aspect ratio from
`21:9` through `9:21`. The `768P (native quality)` and `16:9` selection resolves to H3's established
1344 by 768 canvas. The experimental `2K` tier has a much larger attention and memory cost. Use
Preflight before generation.

The first T2VA baseline uses the unquantized reference precision policy. Most transformer weights
use BF16. The small patch, timestep, and output modules remain FP32 because the audited reference
keeps those stability-sensitive modules in FP32. Quantized execution remains deferred until the
BF16-class smoke test works.

## EasyCache benchmark results

WeeTodd tested no cache, conservative auto, balanced auto, and speed auto at 8, 12, 16, and 20
requested steps. Every run used the same five-second prompt, seed, model components, and staged
unloading policy. The first matrix used a 640 by 384 smoke-test canvas. The second matrix used the
native-quality 1344 by 768 canvas.

At 20 steps and 1344 by 768, conservative reduced sampling time by 20.6 percent, balanced reduced
sampling time by 30.7 percent, and speed reduced sampling time by 46.9 percent. Speed reduced the
complete workflow from 54.3 minutes to 29.6 minutes. Balanced remained a distinct middle policy at
38.0 minutes.

![H3 EasyCache scaling at native 768P](benchmarks/artifacts/charts/h3_easycache_policy_step_scaling_768p.svg)

A complete transformer evaluation stabilized near 165 through 167 seconds at 1344 by 768, compared
with approximately 23 through 26 seconds at 640 by 384. The native canvas has 4.2 times as many
pixels, but an evaluation cost approximately 6.8 times as much. H3's dense attention therefore
makes resolution substantially more expensive than pixel count alone suggests.

These are single-seed performance measurements. Endpoint inspection found the requested robot and
final-wave state in all native-resolution outputs, but the benchmark does not prove motion, detail,
or audio equivalence. Read the [native 768P report](docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING_768P.md),
the [smoke-resolution report](docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING.md), or the
[local artifact bundle](benchmarks/artifacts/README.md).

Models are never bundled. See [Architecture](docs/ARCHITECTURE.md), [Roadmap](docs/ROADMAP.md),
[Smoke-test plan](docs/SMOKE_TEST_PLAN.md),
[Attribution](docs/ATTRIBUTION.md), and the agent-facing [OKF knowledge bundle](knowledge/index.md).

## Status

Experimental and pre-release. APIs, node names, and workflow compatibility may change.
