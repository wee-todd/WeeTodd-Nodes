# WeeTodd implementation status

Reconciled 2026-09-09 against the local source and saved acceptance evidence.

Follow-up 2026-09-11: matched LTX 2.5 Q8-paged T2V at 768×448, 121 frames and the
8+3 schedule measured 64.20 seconds on the repeated baseline versus 64.18 seconds with an
experimental one-block GPU-weight lookahead. Both complete video/audio MP4s were byte-identical;
MLX peak remained 8.06 GiB. The first baseline's 72.65 seconds included additional encoder and
compilation warming, so its apparent improvement is not credited to lookahead. First-frame
conditioning measured 70.37 versus 71.35 seconds, and two-subject MSR measured 149.36 versus
148.32 seconds, again with byte-identical paired movies. The rank-450 adapter stress test at
strength 0.3 measured 110.09 versus 107.72 seconds with the same movie hash; that single 2.2%
difference is not a repeat-qualified speed claim. This does not justify changing LTX
acceleration defaults or porting the candidate to LTX 2.3. Tests used an M3 Ultra with 256 GiB,
fresh renderer processes, and unpurged OS caches; they do not qualify a physical 36 GB machine.
Matched LTX 2.3 comparison is now complete: distilled 1.1 Q8, Q8 QAT Gemma, 768×448,
121 frames, 25 fps, eight evaluations, CFG 1/STG 0 and linear trailing Shift 5. Native DT
measured 103.0 s from a fresh process and 87.0 s warm, with its canvas cleared before each run.
WeeTodd's staged/streamed research route measured 90.2/90.3 s; a resident variant measured
89.3/87.7 s but raised process physical-footprint high-water from 21.3 to 128.8 GiB. It is not
promoted as the default. DT peaked at 31.6 GiB on that physical-footprint counter. These are
single pairs with OS caches intact; DT and MLX quantization/RNG differ, so equal seeds do not
establish cross-engine pixel or quality parity.

The validated route is now available as an explicit LTX 2.3 single-pass distilled 1.1 T2V
preset in Studio, headless setup and ComfyUI Generation Config. It selects the actual versioned
checkpoint, requires no Dev alias or upscaler, keeps staged/streamed memory defaults, and exposes
Steps and Shift with fixed audio/video guidance. Existing two-stage and conditioning routes
remain unchanged. The normal headless client completed the matched case in 91.1 s with the
exact same complete MP4 hash as the research route, eight progress callbacks and no retained
runtime. Process peak RSS was 19.0 GiB and MLX allocator peak was 30.1 GiB (different counters
from physical footprint). Tests cover checkpoint selection, task rejection, controls, generic
LoRA target validation, real sampler evaluation counts and staged cleanup on success/failure/
cancellation. Selected Python tests (658), Swift tests (60), packaging and workflow checks pass.
Full-size saved ComfyUI execution and physical 36 GB qualification remain open.

The same validation found and fixed a headless LTX 2.5 LoRA result-serialization error:
inspection paths are now JSON strings, so valid adapter preflight/generation can save `result.json`.
No weights or strengths change. First-frame and ordinary-LoRA recipe examples now document the
existing shared format. Studio managed runtimes need their source refreshed to receive the fix.

Follow-up 2026-09-10: Studio now displays native stage/evaluation progress, elapsed/last-output
time, and measured statistics for new render versions. LTX 2.5 guided setup includes dedicated
IC-LoRA control, Ingredients and MSR routes with compatible attachment controls. H3 head/FFN chunk
settings propagate into deferred blocks. An experimental 0–16 GB retained raw-page budget is shared
by Studio, headless recipes and the composable ComfyUI sampler; default 0 preserves no retention.
Cache budget, quantized/LoRA parity and success/failure/cancellation cleanup have focused tests.
Full-size cache timing and post-fix chunk performance on a physical 36 GB Mac remain unqualified.

Studio generation selection now separates engine, task and preset. Supported sampling controls and
LoRA strengths are visible; unsupported CFG/Shift or fixed schedules are explained rather than
silently ignored. Preparation and export share a content-based recipe resolver, and Generate runs
preflight automatically. Legacy clips preserve their Custom recipe behavior. H3 acceleration
preferences have per-clip overrides, and headless resident sampling now forwards the selected
policy while retaining staged unloading before decode. A matched 512×512, 124-frame, 19-evaluation H3 test on a 256 GiB M3 Ultra completed in
574.7 s with paged weights and normal working buffers versus 1065.8 s for the saved lower-memory
baseline; movies were byte-identical. MLX stage peak rose from 6.64 to 6.98 GiB. Full residency
reduced total time to 535.1 s but raised the stage peak to 32.32 GiB. Studio exposes both choices;
automatic defaults stay conservative pending broader qualification. This does not qualify either
option on a physical 36 GB Mac. See [measurement conditions](studio/README.md#matched-h3-execution-measurements).

Community report, 36 GB M3 Max (user-measured, source artifacts not independently inspected):
LTX 2.5 T2V at 768×448, 121 frames, 8+3 distilled, paged Q8 transformer/Gemma completed in
164.8 s (reported sampling 148.9 s), with reported complete process peak 9.20 GB. H3 first-frame
generation at 768×448 using the Q8 vision encoder completed in 1750 s (sampling 1628 s), with
reported process peak 7.72 GB, staged peak 6.24 GB and 8 ms AV drift. This establishes a reported
baseline for those recipes, not universal physical-36-GB qualification. The older BF16 comparison
used 672×384, so its 552 s timing is not a matched-resolution paging benchmark.

Follow-up 2026-09-10: an experimental native H3 DT-file adapter now reads original row-int8,
palette8, ezm7 and F16 storage without writing converted weights. Guided setup creates metadata
references for H3 T2V; the shared headless/composable-node path reuses the existing sampler and
staged lifecycle. Qwen language layers load sequentially with bounded embedding-row reads. Video
and audio decoders load from the original combined DT VAE. Image/reference/audio-input modes,
other DT model families, accelerators other than verified MPP projections and 36 GB hardware are
not qualified. The complete
DT-weight headless render completed at 512×512/124 frames/19 Euler evaluations in 924.5 seconds
with an 18.07 GiB process-footprint peak and 8.3 ms A/V duration drift. All weighted runtimes
unloaded. Sampled frames were coherent; broad quality and a saved ComfyUI DT-file render remain
open. Block preparation took 320.7 seconds, including 184.7 seconds decoding DT codecs. No DT app
performance parity is claimed.

The subsequent Metal weight-preparation pass reduced that same DT-weight render to 724.5 seconds
(21.6% less time) with an identical MP4 digest. Process footprint fell to 16.35 GiB; transformer
MLX peak rose from 4.94 to 5.31 GiB, with the overall 12.53 GiB MLX peak unchanged. Original weights
remain read-only, one block stays active, and all runtimes unloaded. All 534 mapped native tensors
and the exhaustive finite-half-scale/int8 rounding check matched the CPU decoder. Block preparation
fell from 320.7 to 134.4 seconds; direct packed-int8 projections and physical 36 GB qualification
remain open. Existing DT-file recipes use this preparation path automatically.

Verified Automatic MPP projections subsequently reduced the same render to 651.4 seconds
(10:51), with byte-identical video/audio and unchanged 16.35 GiB process footprint. All four
projection signatures passed the runtime gate. New normal-memory DT recipes select Automatic;
existing clips retain their settings and can opt in through the Projection control.

The next memory pass retains original F16 video decoder weights with FP32 activations, omits
the unused encoder, and evaluates each decoder block before building the next graph. DT transformer
reads use bounded read-only mappings with ordinary-read fallback; no mapped weight cache is kept.
These paths share the same Studio, headless and composable-node renderer. The matched complete
512×512/124-frame/19-evaluation M3 Ultra (256 GiB) render peaks at **9.48 GiB process footprint**
(42.0% below 16.35 GiB) and **6.58 GiB MLX allocation** (previously 12.53 GiB), with a byte-identical
video/audio MP4 and all weighted runtimes unloaded. Total time is effectively unchanged at
649.7 versus 651.4 seconds. Both runs used fresh renderer processes and fresh prompt caches;
OS file caches were not purged. Block preparation fell from 131.9 to 109.4 seconds, but compute
time increased in this desktop run. Do not claim an overall speed improvement from this pass.

The following DT preparation pass batches one cached-modulation block's GPU decoding and native
layout/dtype conversion, retaining eager fixed-weight and initial AdaLN loading. The same complete
recipe finished in **612.7 seconds (10:13)** versus 649.7 seconds, with a byte-identical video/audio
MP4, 950 batches, all weighted stages unloaded and no retained weight cache. Preparation fell from
109.4 to 91.7 seconds. Process footprint stayed effectively flat at **9.51 versus 9.48 GiB**;
transformer/video MLX peaks remained 5.31/6.58 GiB. Total elapsed time was 5.7% lower in this single
desktop comparison; unchanged block computation also ran faster (449.5 versus 463.6 seconds), so
the full elapsed gain is not attributable to batching alone. This does not extend hardware or
saved ComfyUI DT-file workflow qualification.

The subsequent native-layout pass fuses compatible DT int8 Q/K/V and gate/up decoding, ordering,
and BF16 conversion while preserving intermediate FP16 rounding. Other codecs, rounding modes,
fixed weights and initial modulation loading retain the existing path. All 50 real blocks matched
bitwise. Alternating preparation probes fell from 4.10/4.12 to 3.41/3.39 seconds, with temporary
MLX preparation peaks reduced from 1.72 to 1.08 GiB. The same complete M3 Ultra render finished
in **605.3 seconds (10:05)** versus 612.7 seconds, with a byte-identical video/audio MP4, 1,900
fused groups, all stages unloaded, no retained weight cache and no persistent converted weights.
Preparation fell **91.7 to 78.4 seconds**; unchanged compute took 454.0 versus 449.5 seconds.
The observed total improvement is 1.2%, with peak process footprint unchanged at **9.50 GiB**
and transformer/video MLX peaks unchanged at 5.31/6.58 GiB. Qualification limits above still apply.

The next DT-weight pass overlaps preparation of one next block with current-block computation.
It reuses the existing executor and exact native weight values, with a worker-owned Metal stream
and read-only SQLite handles. Normal-memory, resolved-MPP, single-block, cached-modulation runs
with no retained-page budget enable it; low-memory, standard MLX, selected-block windows and
retained-page caches keep sequential preparation. Failure, cancellation and close drain the worker;
source-change errors release prepared arrays even while the exception traceback remains alive.

The integrated matched M3 Ultra render completed in **546.8 seconds (9:07)** versus 605.3 seconds
(10:05), with a byte-identical video/audio MP4, 931 consumed lookahead blocks and all weighted
stages unloaded. Sampling/setup fell **550.5 to 490.8 seconds**. Process footprint was **9.48 GiB**
and overall MLX peak stayed **6.58 GiB**; transformer MLX peak rose **5.31 to 5.91 GiB**. The observed
elapsed improvement is **9.7%**; a same-workload prototype took 570.7 seconds, so desktop variance
remains material. That prototype had a single 13.56 GiB footprint sample at shutdown; the spike
did not recur in the integrated run. Neither new attention tiles nor FP16 attention inputs improved
the measured baseline and neither was promoted. DT sampler parity, a saved ComfyUI DT-file graph
and physical-36-GB qualification remain open. Six I2V/FFLF setup notes now correctly identify
their selected vision encoder and explain the optional Q8 vision-paged selector change.

## Shared renderer and ComfyUI

The shared Python/MLX backend runs through ComfyUI or `scripts/render_headless.py`.
The headless process blocks ComfyUI and node-catalog imports. It accepts versioned JSON
recipes and records resolved assets, effective conditioning, results, and runtime unloading.
The catalog contains 128 nodes and 46 UI workflows. Static validation establishes portable
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
model detection, arbitrary downloads/conversions, DoRA/LyCORIS, or arbitrary missing-file discovery.
Explicit guided setup now discovers supported component layouts and handles pinned catalog downloads
and selected conversions outside graph execution.

## Standalone graphical interface

Draw Things integration is experimental. Studio includes saved gRPC/cloud connections, Keychain
credentials, image generation into the existing asset stores, a Draw Things clip type, CU preflight,
first-frame image inputs, and compatible server-resident LoRAs/groups. One shared adapter serves
Studio, v3 headless jobs, and ComfyUI. The transport preserves separate video/audio tensors and
refuses silent completion for audiovisual models. Headless jobs run sequentially, verify completed
artifact hashes, and require a deliberate new attempt after an uncertain remote submission or loss
of an already-generated artifact.

Synthetic gRPC image and audiovisual generation passed through Studio's native interface. Real
FFmpeg tests establish transport, timing, failure handling, and media publication. A CLI job with
Studio closed also completed image-to-first-frame video generation and movie assembly with an
existing clip, dissolve, title, and supplementary audio. These fixture-based checks do not establish
model quality or live account compatibility. Offline resume reused both remote artifacts and the
same final movie hash after the fixture server stopped. Direct Cloud uses
a read-only free-request/PAYG check and fresh CU policy; unknown allowance stays blocked. The DT+
App Bridge has no verified free-only billing contract and cannot generate in this implementation.
A real LTX 2.3 cloud response exposed a finalization timing assumption: all 121 video frames and
230,880 stereo samples arrived, but the complete causal audio was shorter than the 24 FPS video.
The helper and Python bridge now recognize and independently validate the exact LTX causal count.
Offline recovery produced a verified 768×448 MP4 with all 121 frames and the original 48 kHz audio;
no cloud resubmission was used. A matching gRPC fixture covers the receipt path. Broader live account
qualification and a fresh post-fix cloud submission remain pending.
Advanced reference/control/audio conditioning and local LoRA conversion/upload remain unsupported.
The optional helper distribution includes dependency source and rebuild/replacement instructions.
See [Draw Things setup and qualification](studio/README.md#draw-things--experimental).

WeeTodd Studio now lives in `studio/` as a native Swift macOS application around the shared
renderer. It includes the requested editor layout, full-window prompt editor, native Light/Dark
appearance, clip-state colors and prioritized Actions, three collapsible asset stores, movie/still/
sequence import, multiple audio tracks, titles/transitions, versions, split/extension/bridge tools,
project save/recovery and Collect Media. Movie settings resolve at clip level during finishing.

Movie and clip headless-job export embeds generation and finishing plans. `WeeToddCLI` or
`render_headless.py --job` executes them sequentially with preflight, cancellation, integrity checks
and resumable render/finishing stages. Studio can install a private native Python/MLX runtime using
pinned, hash-verified dependencies. It preserves other environments and shared model files.

Studio also has a model-filtered linked LoRA library and reusable named groups. LTX 2.5 groups
accept both LTX 2.3 and LTX 2.5 members; H3 and LTX 2.3 stay separate. Applying a group snapshots its
members and strengths into the clip; editing a template cannot mutate existing projects or exported
jobs. The shared renderer retains authoritative adapter and task checks. GUI validation uses synthetic
header fixtures and establishes editing/transport behavior, not visual quality.

Guided setup now provides built-in H3/LTX presets, header-based existing-model reuse, validated
recipe creation, memory-policy advisories and explicit pinned downloads/preparation. Interrupted
downloads resume; SHA-256 and staged output publication protect existing models. Final media
preflight remains required for image/reference clips. Published Vayden releases provide the H3 Q8
vision encoder and complete LTX 2.5 distilled Q8 component set. The catalog pins each release and
every file checksum; source conversion remains optional and applicable model terms are retained.
H3 setup also offers the matching text/image or genuine reference transformer, Q8 video VAE and
official task support files (audio VAE, tokenizer, processor and task manifest). Task-specific downloads
are filtered, and explicit downloaded transformer provenance is checked during discovery and recipe
creation. Each model row has **Import…** for a linked local file/folder and **Download…** for a
compatible package, with size/contents/terms reviewed before transfer.

The app is an initial development build. FFmpeg/FFprobe and optional RIFE still need configuration;
retail signing/notarization and clean-Mac qualification remain release work. MetalFX interpolation is
experimental and requires explicit depth/motion/camera guides. See [Studio usage and limits](studio/README.md) for the exact implementation boundary.

## Experimental clip motion enhancement

Motion Fidelity (De-Roping) is implemented for H3 clips through Studio, v2 movie/clip headless jobs
and two ComfyUI nodes. It is off by default. Adaptive latent analysis or uniform holds expands video
and pitch-preserved conditioning audio; same-resolution partial H3 denoising then recovers source
frame timing and remuxes the original soundtrack. Sources are retained and enhancement settings do
not invalidate base generation. Completed enhancement stages can resume after hash verification.

A dense H3 896×512 boxing reference used 19 source evaluations without FastVideo/VDN. Uniform 2×
refinement recovered exactly 124 frames at 24 fps with audio/video duration drift below 1 ms.
Matching frames retain the subject/action, but changes are subtle and fast-glove blur remains.
This is not general motion-fidelity qualification. Plain H3 T2VA repair
recipes only; LTX, imported-movie UI support, long-clip windows and regional edits remain gated.
See [controls, budgets and limits](studio/README.md#motion-fidelity-de-roping--experimental-h3).

## H3 reference paging

Qwen paging v2 retains vision features and releases vision before sequential language layers.
The shared renderer accepts these pages for Ref2VA and FL2VA, with an experimental genuine Ref2VA
Q8-paged Comfy workflow and Studio/headless recipe preparation command. Existing text-only v1
pages remain T2VA-only. Tiny resident/paged tests cover actual image/video execution in FP32 and
BF16, repeated requests, cancellation/failure cleanup and packed Q8 storage preservation.
Transformer preparation from the genuine 13-shard Ref2VA source measured a 5.86GB complete-process
peak. A one-image 640×384 saved Comfy graph completed 19 dense evaluations and published
124 synchronized frames, with a 21.70GB full-process peak on an M3 Ultra/256 GiB host.
The headless output is byte-identical, with a 21.26GB complete-process peak and all runtimes
released. A Studio clip job exported and passed CLI preflight. The final Python suite excluding
optional algorithm search passed 1,407 tests with one skip; all 14 Swift tests passed and the
release application was rebuilt. Physical 36GB Mac and broad reference-quality qualification
remain open. Full BF16-versus-Q8 model parity was not run.

## Source checkpoints

- `e7ddb75`: guided Studio/CLI model setup and both published preconverted Q8 downloads; includes
  `f3168af` for native setup, model discovery, verified preparation and recipe creation.
- `e31a27c`: shared headless renderer and conditioned ComfyUI workflows.
- `4c56ea2`: native Studio editor, managed runtime installation and resumable movie/clip jobs.
- Publication cleanup: shipped Python preflight no longer depends on ignored agent files. App
  packaging verifies a fresh bundle before replacing the previous build. Project/job exports and
  macOS metadata are excluded from Git; user data and existing runtimes are preserved.

## Checkpoint validation and remaining work

H3 download completion (2026-09-09):

- The genuine Ref2VA Q8-extended transformer is published under Vayden at
  `9f339718e571f181b9a4c3043916ec0f2dc24f00`. All 13 native source hashes, 51 page hashes/headers,
  and 114 remote release files were verified before publication.
- The expanded catalog includes text/image and reference transformers, the Q8 video VAE, and
  task support packages. Source terms remain included. Native task/component filtering has focused
  coverage; downloaded provenance rejects conflicting transformer selection.
- 464 selected Python tests and 32 Swift tests passed. The existing text/image transformer, video
  VAE and both support packages passed installation checks; H3 text/image/reference recipes passed
  component preflight. The published Ref2VA package then passed installation, discovery and recipe
  preflight, including exclusion from the incompatible image preset. The release app was rebuilt
  with all catalog entries. These checks do not establish a new generation or physical 36 GB qualification.
- After the Mac was unlocked, native UI checks passed in Light and Dark modes: per-component
  Download expanded/scrolled to the correct package, reference setup excluded the text/image
  transformer, file/folder Import linked the selected paths, and LTX 2.5 clearly displayed its full
  component bundle. LTX 2.3 kept Import available with unsupported downloads disabled. The original
  System appearance was restored. No new download or generation was started during these UI checks.

Earlier model-setup qualification (2026-09-09, `e7ddb75`):

- 431 focused/required Python tests and 30 Swift tests passed; release app rebuilt with both catalogs.
- All 115 Qwen and 207 LTX release files passed remote size/checksum verification. Both packages
  installed through the shared setup service using verified existing weights and downloaded support files.
- The installed Qwen package was recognized as vision-capable. The installed LTX package exposed
  all five components and produced a validated recipe. Native setup was checked in Light/Dark modes,
  including the zero-recipes starting state, real model discovery and recipe creation.
- README/node/workflow checks passed. These setup checks did not include a new generation run,
  full publisher parity, or qualification on a physical 36 GB Mac.

Earlier checkpoint evidence (retain each figure's original scope):

- 1,304 tests passed and one skipped in the suite excluding optional algorithm-search tests.
- The focused node/runtime/headless/library/workflow review passed 468 tests.
- README catalog and portable H3 API preflight passed.
- Full publisher-checkpoint parity and fresh expensive model renders were not run in this review.
- The backend checkpoint is `e31a27c`; its validation figures above retain their original scope.
- Studio adds 10 Swift document tests and 13 Python bridge/job tests, including real media exports.
  At that checkpoint, the full Python suite passed 1,318 tests with one skipped.
- A new LTX 2.5 job completed generation, finishing, title assembly, and verified resume.
- A fresh app-managed native runtime produced byte-identical generated and assembled MP4s for that
  one-second fixture. Both used the existing shared model files.
- MetalFX spatial + RIFE finishing and explicit-guide MetalFX interpolation completed with audio
  and verified output timing/dimensions. No general interpolation-quality claim follows from these tests.
- Publication cleanup adds six packaging/preflight regressions. The full Python suite passed
  1,324 tests with one skipped (optional algorithm search excluded); all 10 Swift tests passed.
- Motion Fidelity adds shared H3 timing/refinement, Studio and headless integration, and two
  experimental ComfyUI nodes. The full suite passed 1,345 Python tests with one skipped; all 12
  Swift tests passed. Real H3 generation/refinement, packaged CLI movie assembly/resume, and the
  Studio Enhance/source-comparison action completed. Broad motion/identity quality remains open.
- Remaining release work: retail packaging, wider model-setup coverage, and qualification on clean,
  lower-memory Macs.
- Qualify additional adapter/task/precision combinations before promoting them in the interface.

Local research and detailed historical reports remain outside version control by project policy.
Historical reports retain their measured scope; this file is the portable current-status entry.
