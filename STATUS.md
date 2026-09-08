# WeeTodd implementation status

Reconciled 2026-09-07 against the local source and saved acceptance evidence.

## Shared renderer and ComfyUI

The shared Python/MLX backend runs through ComfyUI or `scripts/render_headless.py`.
The headless process blocks ComfyUI and node-catalog imports. It accepts versioned JSON
recipes and records resolved assets, effective conditioning, results, and runtime unloading.
The catalog contains 119 nodes and 42 UI workflows. Static validation establishes portable
contracts; it does not establish that every workflow has local models and selected input media.

| Engine | Implemented | Qualification limits |
| --- | --- | --- |
| H3 | T2V, endpoint/timed frames, multimodal Ref2VA, audio-driven Ref2VA, external extension, generic LoRAs, FastH3 and VDN variants | Native Ref2VA A2V and extension have real renders. Extension visual quality needs further qualification. Accelerator/task combinations are gated. A2V generates a new soundtrack. |
| H3 Fun ControlNet | Loader, preprocessing boundary, resident/paged execution, nodes and headless transport | Synthetic and checkpoint-header checks only. No real control render qualification. Checkpoint availability and applicable terms remain separate. |
| LTX 2.3 | T2V, keyframes, A2V, Ingredients, Union/Motion controls, generic LoRAs, Dev/distilled video extension | Conditioned renders and longer distilled extension have evidence. Generic LoRAs support resident and streamed paths; specialized control combinations remain separately gated. |
| LTX 2.5 | T2V, keyframes, A2V, Ingredients/MSR, IC controls, external extension, refinement/upscaling, LoRAs | Full-length MSR and short extension have evidence. Temporal DFR remains diagnostic. Every adapter/precision/task combination is not qualified. |

Ten saved H3/FastH3/VDN/LTX candidate pairs were rehashed on 2026-09-07: all headless MP4s
match their ComfyUI controls byte for byte. These short fixtures establish local adapter parity,
not publisher parity, all-task coverage, perceptual quality, or universal performance.
Newer conditioning integrations have their own evidence and do not inherit this certificate.

The local model library supports inventory, a persistent registry, shared asset references,
recipe import, supported LoRA normalization, and preflight. It does not implement universal
model detection, automatic downloads/conversions, DoRA/LyCORIS, or arbitrary missing-file discovery.

## Standalone graphical interface

WeeTodd Studio now lives in `studio/` as a native Swift macOS application around the shared
renderer. It includes the requested editor layout, full-window prompt editor, native Light/Dark
appearance, clip-state colors and prioritized Actions, three collapsible asset stores, movie/still/
sequence import, multiple audio tracks, titles/transitions, versions, split/extension/bridge tools,
project save/recovery and Collect Media. Movie settings resolve at clip level during finishing.

Movie and clip headless-job export embeds generation and finishing plans. `WeeToddCLI` or
`render_headless.py --job` executes them sequentially with preflight, cancellation, integrity checks
and resumable render/finishing stages. Studio can install a private native Python/MLX runtime using
pinned, hash-verified dependencies. It preserves other environments and shared model files.

The app is an initial development build. Model recipes, FFmpeg/FFprobe, and optional RIFE still need
configuration; automatic model downloads, retail signing/notarization and clean-Mac qualification
remain release work. MetalFX interpolation is experimental and requires explicit depth/motion/camera
guides. See [Studio usage and limits](studio/README.md) for the exact implementation boundary.

## Checkpoint validation and remaining work

- 1,304 tests passed and one skipped in the suite excluding optional algorithm-search tests.
- The focused node/runtime/headless/library/workflow review passed 468 tests.
- README catalog and portable H3 API preflight passed.
- Full publisher-checkpoint parity and fresh expensive model renders were not run in this review.
- The backend checkpoint is `e31a27c`; its validation figures above retain their original scope.
- Studio adds 10 Swift document tests and 13 Python bridge/job tests, including real media exports.
  The current full Python suite passed 1,318 tests with one skipped (optional algorithm search excluded).
- A new LTX 2.5 job completed generation, finishing, title assembly, and verified resume.
- A fresh app-managed native runtime produced byte-identical generated and assembled MP4s for that
  one-second fixture. Both used the existing shared model files.
- MetalFX spatial + RIFE finishing and explicit-guide MetalFX interpolation completed with audio
  and verified output timing/dimensions. No general interpolation-quality claim follows from these tests.
- Complete retail packaging/model onboarding and qualification on clean, lower-memory Macs.
- Qualify additional adapter/task/precision combinations before promoting them in the interface.

Local research and detailed historical reports remain outside version control by project policy.
Historical reports retain their measured scope; this file is the portable current-status entry.
