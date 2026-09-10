# WeeTodd Nodes

MLX-native MiniMax H3, LTX 2.3, and LTX 2.5 custom nodes for ComfyUI on Apple Silicon.

WeeTodd keeps each model engine behind a separate ComfyUI adapter. H3 generation preserves video
and audio as one synchronized latent contract. Weighted components load only when the graph runs
and can unload between Qwen3-VL, transformer, video VAE, and audio VAE stages.

- 56 composable nodes under `WeeTodd/H3`
- 128 registered nodes across all engines and media utilities; 46 shipped UI workflows

See [implementation status](STATUS.md) for current capabilities and qualification limits.

Studio's [clip generation controls](studio/README.md#clip-generation-controls) now separate engine,
task and preset selection, display supported sampling controls, and validate automatically before
generation. App-level H3 acceleration preferences have per-clip overrides; old clips retain Custom
recipe behavior. Headless jobs carry the resolved settings, including explicit resident sampling.

Studio now shows native render progress and per-version measurements, with guided LTX 2.5
control/Ingredients/MSR setup. H3 has an optional retained-page budget shared by Studio, headless
jobs and the experimental H3 Paging Settings node. It is off by default; full-size 36 GB speed
qualification remains open. See [progress and paging controls](studio/README.md#progress-measurements-and-h3-page-retention).

An optional [Draw Things integration](studio/README.md#draw-things--experimental) adds remote image
assets and audiovisual clips to Studio, portable headless jobs, and ComfyUI through a shared gRPC
adapter. It displays estimated CU and verifies free-only eligibility before submission. Self-hosted
transport is fixture-tested. One real LTX 2.3 cloud response was recovered and verified locally;
full live account and generation qualification is still pending.
The pinned remote SDK supports selected image models and LTX 2/2.3 video; native H3/LTX 2.5 support
remains separate. DT+ App Bridge generation is unavailable until its billing route can be verified.

[WeeTodd Studio](studio/README.md) is the native Swift editor in this repository. It combines generated
and imported clips, titles, transitions and multiple audio tracks, and exports resumable movie/clip
jobs for the shared headless renderer. Its [LoRA library and groups](studio/README.md#loras-and-groups)
filter by clip model, support mixed LTX 2.3/2.5 groups for LTX 2.5, and preserve adjustable strengths
in projects and exported jobs. A managed native Python runtime can be installed from Studio
Settings. Guided model setup provides built-in presets, existing-model discovery, compatible recipe
creation and explicit verified downloads/preparation. Finishing tools still need configuration; see
the Studio guide for build instructions, tested behavior, and consumer-release limitations.

Recent experimental controls include target-frame H3 image, clip, and audio guides; an H3 token
and attention-workspace estimator; independent MLX attention-head and feed-forward row chunking;
LTX 2.5 timed input keyframes; learned generated-keyframe slots; lazy ordered LTX 2.5 LoRA stacks;
and selectable fast-distilled, production-guided, and HQ `res_2s` recipes. An optional modifier
predicts a one-shot duration from the prompt with the official LTX 2.5 duration head. A separate
experimental DFR modifier can add learned temporal refinement at 48 or 96 fps. Its stream and
conditioning contracts pass, but current MLX visual parity is not production-ready.
Existing modifier and conditioning nodes preserve older Generation Config socket order.
The H3 projection backend adds a backward-compatible `auto` choice; saved workflows that explicitly
select `mlx` retain that selection. Eligible projections in paged transformer checkpoints are
wrapped when each block window loads. On an M3 Ultra q8-paged 384p Turbo control, MPP reduced
sampling from 113.35 to 108.49 seconds (4.3%) with the same 7.23 GB MLX peak and a byte-identical
MP4. Add a modifier or conditioning node only when the feature is needed.

## Experimental H3 Motion Fidelity

Studio clips, movie/clip headless jobs and the **H3 Motion Fidelity Settings / Refine** nodes now
support optional De-Roping. The shared native renderer expands source motion, partially refines
joint video/audio latents, and recovers original frame timing with the source soundtrack. The
original is retained. Use a full-schedule H3 T2VA repair recipe, optionally with an explicitly
selected standard LoRA; the feature is off by default and has strict native-frame and memory
budgets. Repair LoRAs affect refinement only, not base generation. A real MLX render establishes
execution and timing, while broad visual improvement remains unqualified. See [Studio controls and limits](studio/README.md#motion-fidelity-de-roping--experimental-h3).
Refinement strength and evaluation count can be set independently for equal-budget comparisons;
older projects and graphs retain their automatic evaluation counts.
Each Studio clip can also override its repair recipe prompt in a full-window editor. The editor
loads the exact resolved prompt, preserves saved nonblank text verbatim, and can reset to the
recipe prompt. The override is embedded in headless jobs and participates only in enhancement
freshness, so changing it leaves the base generation reusable.

## Choose a workflow

Public workflows use this layout:

```text
workflows/<profile>/<task>/<workflow>.json
```

The profile names are intentionally simple:

| Profile | Use it when | H3 sampling policy |
| --- | --- | --- |
| Speed | Fast iteration has priority. | Four real Turbo evaluations at 384p. |
| Balance | Quality, speed, and memory all matter. | Two base evaluations plus four Turbo evaluations at 512p. |
| Performance | Quality-first compute has priority. | Dense 20-point schedule with 19 real evaluations at 512p. |

The core H3 profiles include the same four task graphs. Additional experimental graphs can use
different sizes: the paged Ref2VA candidate starts at 640×384 with the dense schedule.
Media selectors are empty by design. Select
images, video, or audio after loading the workflow.

| Task | Speed | Balance | Performance |
| --- | --- | --- | --- |
| T2V + audio | [Open workflow](workflows/speed/t2v/h3_t2v_speed.json) | [Open workflow](workflows/balance/t2v/h3_t2v_balance.json) | [Open workflow](workflows/performance/t2v/h3_t2v_performance.json) |
| FastH3 T2V + audio | [40-layer candidate](workflows/speed/t2v/h3_fasth3_40layer_candidate.json) | — | [Compact indexed Metal workflow](workflows/performance/t2v/h3_fasth3_compact_vsa_performance.json) |
| VDN-H3 T2V + audio (experimental) | — | [8-step paged workflow](workflows/balance/t2v/h3_vdn_8_step_experimental.json) | [Resident / warm — high memory](workflows/performance/t2v/h3_vdn_8_step_resident_experimental.json) |
| I2V + audio | [Open workflow](workflows/speed/i2v/h3_i2v_speed.json) | [Open workflow](workflows/balance/i2v/h3_i2v_balance.json) | [Open workflow](workflows/performance/i2v/h3_i2v_performance.json) |
| First/last-frame video + audio | [Open workflow](workflows/speed/fflf2va/h3_fflf2va_speed.json) | [Open workflow](workflows/balance/fflf2va/h3_fflf2va_balance.json) | [Open workflow](workflows/performance/fflf2va/h3_fflf2va_performance.json) |
| Ref2VA | [Open workflow](workflows/speed/ref2va/h3_ref2va_speed.json) | [Open workflow](workflows/balance/ref2va/h3_ref2va_balance.json) | [Open workflow](workflows/performance/ref2va/h3_ref2va_performance.json) |

`fflf2va` covers first frame, last frame, and combined first/last-frame conditioning. Use **H3
Frames** when the graph needs numbered middle frames.

### Draw Things workflows · experimental

Build the [optional transport helper](studio/README.md#build-the-optional-connection-runtime),
configure a connection, and use **Draw Things Discover** to obtain an exact compatible server model
ID. Replace the placeholder in **Draw Things Request** before queuing. For a self-hosted endpoint,
confirm that cloud offload is disabled before enabling its confirmation control. The examples are
portable starting graphs; they do not establish a working endpoint or free-tier allowance.

| Workflow | API example | Behavior |
| --- | --- | --- |
| [Estimate only](workflows/balance/t2v/drawthings_estimate_only.json) | [API](examples/drawthings_estimate_only_api.json) | Refresh capability, CU estimate, and billing eligibility without generation. |
| [Image asset](workflows/balance/t2v/drawthings_image.json) | [API](examples/drawthings_image_api.json) | Prepare and generate one image through the shared adapter. |
| [Video + audio](workflows/balance/t2v/drawthings_video.json) | [API](examples/drawthings_video_api.json) | Prepare and generate an audiovisual clip, then finish it with FFmpeg. |

Credentials are supplied through runtime environment references. Generated files use ComfyUI's
configured output directory. Direct Cloud remains subject to strict free-only checks and pending
live qualification; DT+ App Bridge generation is blocked. See [connection policies and limits](studio/README.md#connections-allowance-and-cu).

### LTX workflows

| Profile | Workflow | Purpose |
| --- | --- | --- |
| Balance | [LTX 2.3 two-stage](workflows/balance/t2v/ltx23_two_stage.json) | Standalone LTX 2.3 synchronized generation. |
| Balance | [LTX 2.5 768×512](workflows/balance/t2v/ltx25_768x512_two_stage.json) | Recommended LTX 2.5 starting point with the official 8+3 schedule. |
| Speed | [LTX 2.5 1344×768 Sol paged speed](workflows/speed/t2v/ltx25_1344x768_sol_paged_speed.json) | Eight-forward full-resolution T2V with Q8 paging and fused Sol Attention; verify sparse calls in metadata. |
| Balance | [LTX 2.5 audio-to-video](workflows/balance/ref2va/ltx25_audio_to_video.json) | Freeze one input audio track during both visual stages and publish the original waveform. |
| Performance | [LTX 2.5 Ingredients quality](workflows/performance/ref2va/ltx25_ingredients_reference_sheet_quality.json) | Full 15-forward CFG++ Ingredients generation. |
| Balance | [LTX 2.5 Ingredients balanced](workflows/balance/ref2va/ltx25_ingredients_reference_sheet_balanced.json) | Hybrid 12-forward CFG++ Ingredients generation. |
| Speed | [LTX 2.5 Ingredients + Sol speed](workflows/speed/ref2va/ltx25_ingredients_reference_sheet_speed.json) | Eight-forward Q8 Ingredients generation with compact reference sizing and paged-speed Sol Attention. |
| Speed | [LTX 2.5 MSR speed](workflows/speed/ref2va/ltx25_msr_two_subject_speed.json) | MSR starting graph with Q8 paging, eight real forwards, automatic priority density, and Sol-aligned reference layouts. |
| Balance | [LTX 2.5 MSR balance](workflows/balance/ref2va/ltx25_msr_two_subject_balance.json) | Two-subject Q8 MSR graph with 25-frame dense references and exact attention. |
| Performance | [LTX 2.5 MSR performance](workflows/performance/ref2va/ltx25_msr_two_subject_performance.json) | BF16 quality-first MSR graph with 33-frame full-canvas references and exact attention. |
| Balance | [LTX 2.5 Union Canny](workflows/balance/ref2va/ltx25_union_canny_balanced.json) | MLX-native Canny preprocessing with baked Q8 Union Control. |
| Balance | [LTX 2.5 Union Depth](workflows/balance/ref2va/ltx25_union_depth_balanced.json) | MLX Video Depth Anything preprocessing with baked Q8 Union Control. |
| Balance | [LTX 2.5 Union Pose](workflows/balance/ref2va/ltx25_union_pose_balanced.json) | Preprocessed pose-video control with baked Q8 Union Control. |
| Balance | [LTX 2.5 CrossView Warp + reference](workflows/balance/ref2va/ltx25_crossview_warp_balanced.json) | Create a depth-warped camera view, preview its disocclusion holes, and stack the trained warp/source pair with an Ingredients character-and-scene reference frame. The original soundtrack is preserved. |
| Performance | [LTX 2.5 Motion Track quality](workflows/performance/i2v/ltx25_motion_track_quality.json) | MLX-generated colored trajectories with the dedicated Motion Track IC-LoRA and full 15-forward CFG++. |
| Performance | [LTX 2.5 768×512 guided HQ](workflows/performance/t2v/ltx25_768x512_guided_hq.json) | Development transformer with selectable 30-step guided Euler or 15-step guided `res_2s`, followed by the official distilled-LoRA refinement stage. |
| Balance | [LTX 2.5 768×512 practical DFR](workflows/balance/t2v/ltx25_768x512_dfr_conv_vae.json) | Exact prebaked Q8 DFR sampling with bounded convolutional-VAE publication. |
| Performance | [LTX 2.5 768×512 DFR + Diffusion VAE](workflows/performance/t2v/ltx25_768x512_dfr_diffusion_vae.json) | Experimental full-resolution detail generation with exact prebaked Q8 adapters and one-step pixel-diffusion decode. |
| Performance | [LTX 2.5 768×512 accelerated DFR + Diffusion VAE](workflows/performance/t2v/ltx25_768x512_dfr_diffusion_vae_metal_tiled.json) | Experimental query-tiled Metal decoder with substantially lower runtime and memory than the exact Diffusion VAE. |
| Performance | [LTX 2.5 768×512 DFR temporal 48 fps](workflows/performance/t2v/ltx25_768x512_dfr_temporal_48fps.json) | Diagnostic-only learned temporal refinement with untouched stage-one audio; current MLX visual parity is not production-ready. |
| Performance | [LTX 2.5 1920×1088](workflows/performance/t2v/ltx25_1920x1088_two_stage.json) | High-resolution quality-first generation. |
| Balance | [LTX 2.5 chained timeline](workflows/balance/continuation/ltx25_768p_15s_three_window_chain.json) | Experimental long-timeline continuation and selective regeneration. |
| Balance | [LTX 2.5 video refine](workflows/balance/video-upscale/ltx25_any_video_pixel_spatial_2x.json) | Refine and upscale any source movie while preserving its audio. |
| Balance | [Florence-2 text mask + CorridorKey](workflows/balance/keying/corridorkey_mlx_florence2_text_mask.json) | MLX text grounding on sparse frames, standard mask refinement, and CorridorKey extraction with both previews connected. |

### CorridorKey workflow

Use the [balanced CorridorKey MLX workflow](workflows/balance/keying/corridorkey_mlx_auto_chroma.json)
to extract a straight-color foreground and alpha matte from green-screen footage. The default
auto-chroma node creates the coarse, eroded hint that CorridorKey expects. Any standard ComfyUI
`MASK` from SAM, Florence-2, Impact Pack, or another segmentation node can replace that hint after
passing through **CorridorKey Mask Refine**.

For text-selected subjects, use the [Florence-2 text-mask CorridorKey workflow](workflows/balance/keying/corridorkey_mlx_florence2_text_mask.json).
The tested Q8 Florence model runs through MLX-VLM, evaluates frame 0, each selected stride, and the
final frame, then interpolates one union box across the remaining frames. The default mode refines
each localized region into an image-guided silhouette. The fast mode emits a rectangle. The
default eight-frame stride reduces detector work. Set the stride to one when motion or occlusion
makes tracking more important than preprocessing speed. Review the green contour before running
or accepting the final key. The orange rectangle shows the Florence search region.

## Install

1. Use an arm64 Python 3.11 or later ComfyUI environment.
2. Install WeeTodd Nodes under `ComfyUI/custom_nodes/WeeTodd-Nodes`.
3. Install the package into the active ComfyUI environment.
4. Restart ComfyUI.

```bash
COMFYUI_ROOT=/path/to/ComfyUI

"$COMFYUI_ROOT/.venv/bin/python" -m pip install \
  -e "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes"
```

Install the LTX runtime extra before loading an LTX 2.3 or LTX 2.5 workflow:

```bash
"$COMFYUI_ROOT/.venv/bin/python" -m pip install \
  -e "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes[ltx]"
```

The LTX extra pins the first upstream revision that builds with current Hatchling. Older WeeTodd
checkouts pinned a revision whose subpackage metadata referenced a README outside the package and
could fail during editable installation. Update the repository before troubleshooting that error.

WeeTodd discovers ffmpeg from an explicit node value, `WEETODD_FFMPEG`, the ComfyUI Python
environment, the process PATH, standard Homebrew/MacPorts locations, or `imageio-ffmpeg`. ComfyUI
Desktop does not need to inherit an interactive shell profile and users should not need to add a
symlink inside its `.venv`.

The project requires MLX 0.32.2 or later. Confirm that ComfyUI uses arm64 Python before installing:

```bash
"$COMFYUI_ROOT/.venv/bin/python" -c \
  'import platform, sys; print(sys.version); print(platform.machine())'
```

LTX 2.5 configuration resolves checkpoint-dependent memory controls at execution. A paged
transformer automatically enables low-RAM streaming, and streaming automatically selects the
compatible `reference_fp32` feed-forward path. Resolved settings and every adjustment are written
to generation metadata. Older saved workflows whose feed-forward value shifted into the prompt
context field are repaired when the Generation Config node executes.

### Optional CorridorKey MLX setup

CorridorKey is a separately licensed optional runtime. WeeTodd does not include its source or
checkpoint. Review the [CorridorKey license](https://github.com/nikopueringer/CorridorKey/blob/main/LICENSE)
before installation. The license permits production use but restricts redistribution, competing
products, commercial software integration, and paid inference services.

Install the official MLX runtime in ComfyUI's active Python environment:

```bash
"$COMFYUI_ROOT/.venv/bin/python" -m pip install \
  "corridorkey-mlx @ git+https://github.com/nikopueringer/corridorkey-mlx.git@04503e797060e091f991bc88b85ec61b0b9b862b"
```

Download `corridorkey_mlx.safetensors` from the
[official CorridorKey MLX v1.0.0 release](https://github.com/nikopueringer/corridorkey-mlx/releases/tag/v1.0.0).
Place the checkpoint in `ComfyUI/models/corridorkey/` or a shared `corridorkey` model root.

The loader provides four profiles. Compiled 512 is the fastest preview. Compiled 1024 is the
default balance. Compiled 2048 is the quality-first path. Tiled 512 reduces Metal peak allocation
and preserves the source aspect ratio at the cost of more model passes. On an M3 Ultra synthetic
1,280×720 probe, compiled 1024 took 0.40 seconds after warm-up at a 3.44 GiB MLX peak. Tiled 512
took 0.66 seconds at a 2.03 GiB peak. Treat these measurements as execution checks, not footage
quality benchmarks.

The current official MLX backend supports the green checkpoint. Its public engine returns 8-bit
foreground and alpha arrays, and its despill and despeckle options are placeholders. WeeTodd
therefore performs compositing and generic mask cleanup independently and does not claim float EXR
parity yet. Blue-screen MLX support remains gated until an official blue MLX checkpoint and stable
engine contract are available.

### Optional Florence-2 MLX auto masking

Download the MIT-licensed [Florence-2 base fine-tune Q8 MLX bundle](https://huggingface.co/mlx-community/Florence-2-base-ft-8bit)
and keep every tokenizer, processor, configuration, and safetensors file together under:

```text
ComfyUI/models/florence2/Florence-2-base-ft-8bit/
```

The Florence loader is lazy and does not execute checkpoint-supplied remote Python code. It uses
MLX-VLM's native Florence architecture, the current installed Transformers processor, and the
checkpoint-declared BART tokenizer. The published Q4 conversion is intentionally rejected because local validation found unreliable
coordinate-token decoding. Q8 measured 0.665 GiB MLX peak for one reviewed 768-pixel
reference-sheet silhouette. The clean ComfyUI graph completed in 3.23 seconds and unloaded after
the node completed. This is not a complete CorridorKey graph or complete-ComfyUI peak measurement.

## Model downloads and reuse

Studio users can open **Studio Settings → Model setup**, choose an H3/LTX preset, and select
**Use Existing Models**. Scan an existing ComfyUI model folder or another shared library, resolve
any ambiguous components, then create a validated recipe. Weights remain in their existing locations.
Image/reference presets prepare the components first; required clip media is validated after attachment.
See [guided setup](studio/README.md#guided-model-setup) and the
[portable LTX 2.5 recipe example](examples/headless/ltx25_distilled_q8_t2v.json).

The same setup service is available without Studio:

```bash
python scripts/setup_models.py catalog
python scripts/setup_models.py scan ltx25-text /path/to/ComfyUI/models
python scripts/setup_models.py downloads
```

Downloads are explicit setup actions and never run inside a node graph. The catalog lists pinned
sources, terms and disk requirements. Downloads resume after interruption, verify complete SHA-256
hashes, reuse checksum-matching files from selected roots, and prepare outputs in a new directory.
The [preconverted Q8 vision encoder](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged)
avoids local conversion and is the recommended H3 encoder download. It retains source terms and
requires Hugging Face access. Source conversion remains available as an alternative.
For LTX 2.5, the [preconverted distilled Q8 package](https://huggingface.co/Vayden/LTX-2.5-MLX-Q8-Paged)
includes the paged transformer and Gemma encoder, video/audio VAEs and spatial upscaler. Guided setup
downloads the prepared components directly; local quantization is optional.
The same prepared components can be selected by Studio, the headless runner and existing ComfyUI nodes.

### Reuse Draw Things H3 files · experimental

Studio's **MiniMax H3 · Draw Things models · Text to video** setup preset can read the
original H3 transformer, Qwen encoder and combined VAE checkpoints directly from a local
Draw Things model store. It creates small metadata references; it does not convert, copy,
modify or persistently cache model weights. The same recipe runs headlessly, and its component
paths work with the composable H3 ComfyUI nodes. This is native WeeTodd generation; choose a
Draw Things clip instead to use the DT application/server or cloud service.

The initial route supports the tested H3 row-int8/palette8 transformer, 50-layer Qwen row-int8
encoder and F16 VAE layouts for **text-to-video with generated audio**. Other model families,
image/reference/audio-input conditioning, and arbitrary DT formats are not supported by this
route. Setup validates tensor layouts before loading weights. Keep the original files available;
changing/removing them requires revalidation. LoRAs, resident execution and accelerators other
than verified MPP projections have not been qualified with these weights. Existing native presets
remain the performance default.

Import the three `.ckpt` files and an existing H3 tokenizer folder, or use the small **H3 tokenizer ·
For Draw Things model reuse** download (about 11.51 MB, no weights). See the
[Studio and CLI instructions](studio/README.md#reuse-local-draw-things-h3-models).
The adapter decodes/reorders one transformer block at a time in memory and uses the native
BF16/FP32 arithmetic policy. Packed int8 and palette weights now expand through bounded Metal
kernels, and layout transforms stay on the GPU. Raw/ezm7 tensors retain the CPU reader. This changes
weight preparation, not projection precision or sampler math, and writes no converted weights.
It does not inherit DT's Swift/Metal sampler speed or numerical results.

The matched DT-weight render at 512×512, 124 frames, 24 fps, stereo 32 kHz audio and 19 Euler
evaluations fell from **924.5 to 724.5 seconds (21.6% less time)** on an M3 Ultra with 256 GiB RAM.
The entire MP4 was byte-identical. Process-footprint peak fell from **18.07 to 16.35 GiB**; overall
MLX peak stayed at 12.53 GiB, while transformer-stage MLX peak rose from 4.94 to 5.31 GiB. Qwen
encoded afresh in both runs. Sampling/setup fell from 868.7 to 667.6 seconds; video decode was
31.1 versus 33.0 seconds and was not optimized. All weighted runtimes unloaded.

Block preparation fell from 320.7 to 134.4 seconds, including a reduction in codec decoding from
184.7 to 49.0 seconds. All 534 mapped fixed/block tensors matched the CPU reference byte for byte.
The decoder also passed every finite FP16 scale multiplied by every int8 value. This remains a
single matched development comparison, slower than the earlier native Q8-paged run, not a matched
DT application benchmark or a physical 36 GB qualification. Direct packed-int8 matrix multiplication,
a full saved ComfyUI DT-file render and broad visual-quality qualification remain future work.

New DT recipes using normal working memory select `projection_backend=auto`. The renderer enables
MPP projections only on its qualified GPU architecture and macOS version, verifies each new
projection shape against standard MLX on first use, and falls back on unsupported configurations,
kernel failures or a verification mismatch. Lower-memory recipes retain standard MLX. Existing
recipes are not rewritten: choose **Automatic** under the clip's projection settings, or set
`config.projection_backend` to `auto` in an exported recipe. ComfyUI exposes the same choice in
**H3 Generation Config**. **MLX** remains available for comparison or to disable MPP.

With Automatic projections, the same matched DT-weight render completed in **651.4 seconds
(10:51), down another 10.1% from 724.5 seconds**. The complete MP4 remained byte-identical,
process-footprint peak stayed at **16.35 GiB**, and transformer/overall MLX peaks stayed at
**5.31/12.53 GiB**. Block execution fell from 513.8 to 446.3 seconds; block preparation was
134.4 versus 131.9 seconds. All four projection signatures passed verification, no retained weight
cache was used, and all weighted runtimes unloaded. This is the same M3 Ultra/256 GiB development
comparison, with fresh prompt encoding; it does not qualify other hardware or lower-memory settings.

### Ready-to-use Q8 downloads

Choose **Set Up… → Download or prepare a model** in Studio and select **Preconverted (Recommended)**.
You can also click **Download…** beside an individual component to select its compatible package.
**Import…** beside it links a model already on disk. Some downloads include several components;
review the displayed package contents before starting.
Accept the selected repository's access terms on Hugging Face and configure **Hugging Face access**
in Studio, or use your existing CLI login. Download only the package needed by your chosen engine.

| Package | Download size¹ | Included | Still separate |
| --- | --- | --- | --- |
| [H3 Q8 vision encoder](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged) | 28.22 GB | Q8 language pages, retained vision tower, manifests and support files | Matching H3 transformer, video VAE and task support downloads below |
| [H3 text/image Q8 transformer](https://huggingface.co/Vayden/MiniMax-H3-MLX-q8-extended-paged) | 33.38 GB | Paged Q8-extended FL2VA transformer | Qwen encoder, video VAE and text/image support files |
| [H3 reference Q8 transformer](https://huggingface.co/Vayden/MiniMax-H3-Ref2VA-MLX-q8-extended-paged) | 58.26 GB | Genuine native Ref2VA transformer, paged Q8-extended | Vision-capable Qwen encoder, video VAE and reference support files |
| [H3 Q8 video VAE](https://huggingface.co/Vayden/MiniMax-H3-Video-VAE-MLX-Q8) | 2.94 GB | Directly loadable video VAE | Shared by both H3 component sets |
| H3 text/image or reference support files | 0.63 GB each | Official task manifest, audio VAE, tokenizer and processor | Choose the support package matching your transformer/task |
| [LTX 2.5 distilled Q8](https://huggingface.co/Vayden/LTX-2.5-MLX-Q8-Paged) | 43.49 GB | Q8-paged transformer and Gemma, convolutional video VAE, audio VAE and spatial upscaler | Optional task LoRAs, controls and alternative guided/DFR components |

¹ Decimal download sizes, not RAM requirements. Setup displays required disk space before downloading.
It pins a verified release and each file's SHA-256. Keep the installed directories and manifests intact.

After downloading, scan the returned folder, resolve component choices, and select **Create Recipe**.
Use **Use Recipe for Selected Clip** for a compatible Studio clip, then **Prepare clip** before generation.
CLI users can follow the complete [LTX 2.5 download-to-recipe example](examples/headless/README.md#download-and-create-an-ltx-25-recipe).
Source conversion is optional and remains available in the catalog. For setup errors or an older
managed runtime, see [Studio setup troubleshooting](studio/README.md#model-setup-troubleshooting).

For H3, download four items: **the matching transformer + matching support files + Qwen vision
encoder + Q8 video VAE**. Studio filters task-specific downloads for the selected preset. Scan all
four installed directories together and create the recipe. The encoder and video VAE can be shared
between text/image and reference clips. The support packages use pinned files directly from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3); they do not fetch the complete
BF16 checkpoint or execute downloaded Python code. See the [H3 CLI setup commands](examples/headless/README.md#download-h3-components).

Choose a workflow first; install only its dependencies. The supported H3/LTX candidates are
alternatives, not a requirement to download every checkpoint. Optional control, preview,
upscaling, and refinement assets are needed only by workflows that use them.

Reuse compatible files from existing shared model roots instead of keeping separate copies for
ComfyUI and headless generation. The filenames below identify reference recipes, not a blanket
publisher restriction. Generic loaders still require a supported architecture, task, tensor
layout, and quantization format; some third-party variants need conversion. Exact checkpoint
pins on qualified FastH3 profiles apply to those measured profiles, not all model loading.

Share compatible encoders, VAEs, and upscalers by reference. Differently trained or quantized
variants are distinct assets. Do not rename an incompatible model to make it appear compatible,
and do not delete presumed duplicates based on filenames or size alone. The read-only
`scripts/inspect_model_library.py` inventory can identify physical aliases and optionally verify
duplicate content without moving or deleting weights.

### Shared-library headless recipes

The local asset registry can bind an existing, preflight-valid headless v2 recipe to persistent
asset IDs. Import only the recipes you want to use; this registers paths and does not copy,
convert, or download weights. These standalone CLI arguments accept local paths (unlike the
relative model-root selectors in ComfyUI nodes).

```bash
python scripts/import_model_recipe.py --recipe recipe.json \
  --model-library library.json --output library-recipe.json
python scripts/render_headless.py --recipe library-recipe.json \
  --model-library library.json --output-directory preflight-result --preflight-only
python scripts/render_headless.py --recipe library-recipe.json \
  --model-library library.json --output-directory render-result
```

Use new output names/directories. The runner saves the resolved recipe and asset provenance in
its result directory. Preflight-only validates existing engine contracts without a render;
it is not a quality certificate or complete third-party compatibility guarantee. LTX 2.3's
current preflight is primarily a bundle-presence check.

Headless recipes can add a versioned `conditioning` object. Every render now runs
preflight; unknown media fields and unsupported task combinations fail before weight
loading. The runner saves `effective-conditioning.json` alongside the original resolved
recipe. For example, add endpoint images to a compatible H3 FL2VA or LTX recipe:

```json
{
  "version": 1,
  "task": "fflf",
  "inputs": [
    {"id": "start", "kind": "image", "role": "keyframe", "path": "first.png", "frame_index": 0},
    {"id": "end", "kind": "image", "role": "keyframe", "path": "last.png", "frame_index": "last"}
  ]
}
```

Place this object under the recipe's `conditioning` key. Paths refer to existing local
files, relative to the runner's working directory or absolute. H3 FFLF requires
`components.task="fl2va"`; LTX 2.3 FFLF/A2V requires Dev `two_stage`. Resident and
low-RAM block-streamed transformers both accept normalized generic LoRA stacks. Keyframe indices are decoded
pixel-frame positions, not latent indices. `last` resolves after frame-count alignment.

Task transport currently covers H3 T2V, keyframes, image/video/audio Ref2VA, audio-driven
Ref2VA, and Fun ControlNet-Union; LTX 2.5 keyframes, one source-audio A2V input, compatible
preprocessed IC-LoRA guides, MSR, and extension; and LTX 2.3 keyframes, A2V, IC-LoRA control,
Ingredients references, and video extension. A2V uses one `kind="audio", role="audio_driver"` input.
LTX uses `audio_policy="source"` and freezes the waveform. H3 uses the Ref2VA partition with
`audio_policy="generated"`: the input drives motion/timing and a new soundtrack, rather than
copying the source samples. H3 video soundtracks require an explicit `soundtrack_path` on
the video input, or a separate audio reference. Embedded soundtracks are not inferred.
LTX controls use `role="control"` and an explicit `control_type` such as `canny_edges`,
`depth_map`, or `pose_skeleton`, with the matching IC-LoRA in the component specification.
H3 control uses `components.task="t2va"`, a local `components.fun_controlnet` SafeTensors path,
and exactly one preprocessed Canny, depth, HED, MLSD, or pose video. The five-block Union branch is
injected at base layers 0, 10, 20, 30, and 40; its residual is never applied to audio rows. Cached,
sparse-attention, VDN, and layer-thinned combinations fail closed pending separate qualification.
Inputs are not automatically preprocessed into control maps. The published checkpoint's license
excludes the U.S., EU, UK, and Republic of Korea. WeeTodd does not bundle the checkpoint, and it
must not be downloaded or executed in an excluded territory.

LTX 2.5 MSR uses `task="ref2va"` with one to five still-image `role="reference"` inputs.
Each input declares `reference_role`, `description`, conditioning/attention strengths,
reference frames, size policy, and density priority. The dedicated adapter must be both
`components.msr_lora_path` and the sole `components.ic_loras` entry, with distilled
full-resolution single-stage mode enabled. Headless rendering orders one optional background
last and automatically prepends the exact `Image 1...Image N` prompt guide once.

LTX 2.3 extension accepts resident distilled or Dev `one_stage`, an exact 8n+1 source
matching the config's width, height, fps, and frame count, and generates additions in groups
of eight. Distilled mode is the production speed path: eight positive-only evaluations with
a fused distilled checkpoint. Dev `one_stage` remains the slower 30-step CFG/STG quality
alternative. Final headless and Comfy publication preserves the supplied source segment and
appends only the new decoded segment; the contract reports
`audio_policy="source_reencoded_and_generated_extension"`. H3 external extension requires
the genuine Ref2VA checkpoint, a 2-15 second constant-24-fps source with embedded mono/stereo
audio at the configured output size, a 4-15 second generation window, and the released six-part
Ref2VA continuation prompt structure. The whole source video/audio is supplied as a reference
and its final frame is also placed at target frame zero as an explicit seam anchor. The old
FL2VA/T2VA latent-overlap route is rejected for external extension because its first render was
visually unusable; latent-overlap continuation remains an experimental internal node.
LTX 2.5 extension similarly accepts a matching constant-rate source with embedded audio,
an after-only 8n+1 context from 9 through 241 frames, and a multiple-of-eight addition. It
encodes low- and high-resolution video histories plus synchronized audio through the native
VAEs, uses the same 0.5 continuation strength as latent chaining, removes the repeated context,
and appends the new frames to the full source. Unqualified multi-control stacks and accelerated
H3 conditioning combinations remain gated.
Short transport diagnostics do not qualify visual quality.
Use `scripts/validate_task_conditioning.py` for an explicit candidate-by-task preflight
matrix and selected diagnostic renders. Do not combine `conditioning` with legacy
`reference_images`; legacy recipes remain supported without changing their sampling settings.

Symlink/hardlink aliases share an asset ID while retaining the selected path. After moving a
model on the same filesystem, re-import a direct-path recipe pointing to its new location to
refresh the registry; existing references can then follow it. Changed files invalidate old
references. Registry revisions use file identity, size, modification time, headers, and support
files—not full weight-payload hashes or proof that two separately copied checkpoints match.

Imported H3 recipes store explicit resolved LoRA profile/QKV settings to preserve existing math
if a path changes. New H3 `auto` selection reads declared profile, step, and QKV metadata; when
profile metadata is absent it defaults to standard independently of the filename. Select Turbo
explicitly for metadata-poor distilled adapters. The common adapter inspector validates A/B,
default/turbo A/B, down/up, and lowercase A/B pair structures; H3, resident LTX 2.3, and LTX 2.5
use its rank, schema, and canonical target descriptors. LoRA format conversion and automatic
missing-adapter downloads remain separate implementation work. Unsupported tensor formats are
rejected, never silently omitted.

### LTX 2.3 IC-LoRA controls

Attach one explicitly typed adapter with **LTX 2.3 IC-LoRA Loader (MLX)**. Supply a matching
preprocessed video through **LTX 2.3 Control Video**, or connect an `IMAGE` batch from the MLX
Canny, depth, DWPose, or Motion Track preprocessor through **LTX 2.3 Control Frames**. Guides must
match the requested frame count and output geometry. Temporary bridge media is removed after the
render.

The Generation Config's `ic_lora_topology=auto` is the production-safe choice. It keeps the
qualified clean two-stage path for Union Control, selects the explicit Dev-transformer plus
distilled-helper two-stage candidate for Ingredients, and uses full-resolution single-stage
generation for Motion Track so the adapter and trajectory reference remain active for every denoise
step. The explicit `two_stage_clean` mode preserves legacy parity, but Motion Track can lose its
control in the clean second stage. `control_refine` keeps control during a short full-resolution
refine; `upsample_only` and `single_stage` are also available for deliberate testing. Alternate
Ingredients topologies remain fail-closed.

For Ingredients, use the trained 768×448 bucket at 24 fps for at least 121 frames, set the IC-LoRA
loader strength to `1.4`, and keep the reference-sheet strength at `1.0`. These are different
controls: the former scales adapter weights; the latter controls how strongly the appended
reference latent is preserved.

### LTX 2.3 standard LoRAs

Chain **LTX 2.3 LoRA Loader (MLX)** nodes after the model loader, then run Preflight and Generate.
Use a local safetensors path or a path relative to a configured ComfyUI `loras` root. Generic
LoRAs remain active through T2V, FFLF, A2V, and extension generation, including
normalized per-block loading in low-RAM streaming mode. Specialized IC-LoRA/task-adapter topology
rules remain separate from ordinary style/character adapters. The headless
equivalent is a top-level recipe field:

```json
"loras": {"adapters": [{"path": "/local/models/style.safetensors", "strength": 0.75}]}
```

Up to eight standard adapters apply in order to all stages. Supported pair names include A/B,
default/turbo A/B, down/up, and lowercase A/B; native and supported Comfy projection names are
normalized without renaming or copying the adapter. Per-target alpha and explicit user alpha use
the actual pair rank. Common global alpha metadata uses a matching declared global rank when
present, or the pair rank otherwise; absent or dynamic/baked alpha means unit tensor scaling. The
node's `-1` means automatic. Unknown targets, shapes, non-finite values, and
unsupported tensor fields fail rather than being silently dropped.

This experimental path fuses only targeted projections in memory and retains float or affine
Q4/Q8 precision. Quantized fusion can introduce rounding; it is not exact full-precision adapter
math. It does not create another checkpoint file. With `low_ram_streaming=true`, block targets are
normalized and applied when each block is bound; small non-block targets remain resident.
`low_memory=true` staged unloading remains available. Adapter jobs always unload afterward so a refined model cannot
contaminate the next job. Quantized input dimensions receive their final check against the loaded
projection because packed tensor shapes alone do not uniquely determine group size and bit width.

Task/control adapters (IC-LoRA/MSR) use dedicated loaders and cannot currently be combined with
generic LoRAs. DoRA, LyCORIS, rsLoRA, and stage-specific adapter scheduling are not implemented by
this generic loader. Files can come from any source, but they must match supported LTX 2.3
projection and scaling contracts; this is not a promise that arbitrary LoRAs or arbitrary model
families are interchangeable.

### LTX 2.3 model bundle

The shipped two-stage workflow selects an existing MLX bundle at `ComfyUI/models/LTX-2.3/q8`
or the equivalent shared model root. Its current preflight requires these bundle entries:

```text
LTX-2.3/q8/
├── connector.safetensors
├── vae_encoder.safetensors
├── vae_decoder.safetensors
├── audio_vae.safetensors
├── vocoder.safetensors
├── transformer-dev.safetensors
├── ltx-2.3-22b-distilled-lora-384.safetensors
├── spatial_upscaler_x2_v1_1.safetensors
└── spatial_upscaler_x2_v1_1_config.json
```

The transformer, distilled LoRA, and spatial-upscaler entries also accept matching
`<stem>-*-of-*.safetensors` shards. Keep all shards together. Distilled mode instead requires
`transformer-distilled.safetensors` and does not require the development transformer or distilled
LoRA. Other modes have their own component checks. These names describe the supported MLX bundle;
renaming arbitrary source checkpoints does not convert their tensors into that layout.

Install the Gemma encoder separately before execution. The default
`mlx-community/gemma-3-12b-it-4bit` must resolve to a complete local Hugging Face cache snapshot,
or select an existing local encoder directory. Nodes do not download it during a graph. Run
**LTX 2.3 Preflight** after selecting the bundle and generation mode.

## H3 model layout

The component loader searches every ComfyUI model root, including shared roots from
`extra_model_paths.yaml`. Use relative loader values. Do not enter machine-specific absolute paths.

The profiled T2V loader uses these values:

```text
MiniMax-H3/FL2VA
MiniMax-H3/transformers/q8_extended_paged
MiniMax-H3/text_encoders/q8-paged
MiniMax-H3/FL2VA/processor
MiniMax-H3/FL2VA/tokenizer
MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors
MiniMax-H3/FL2VA/audio_vae
```

```text
ComfyUI/models/
├── MiniMax-H3/
│   ├── FL2VA/
│   │   ├── model_index.json
│   │   ├── text_encoder/
│   │   ├── processor/
│   │   ├── tokenizer/
│   │   └── audio_vae/
│   ├── Ref2VA/
│   │   ├── model_index.json
│   │   ├── transformer/
│   │   ├── text_encoder/
│   │   ├── processor/
│   │   ├── tokenizer/
│   │   ├── video_vae/
│   │   └── audio_vae/
│   ├── text_encoders/q8-paged/
│   ├── transformers/q8_extended_paged/
│   └── vae/q8/video_vae_affine_q8.safetensors
├── loras/
│   └── minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors
└── vae_approx/
    ├── taeh3.safetensors
    └── taeh3_coreml_256.mlpackage
```

Reference model sources (select only what your workflow needs):

- [Official MiniMax H3 components](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Q8-extended paged transformer](https://huggingface.co/Vayden/MiniMax-H3-MLX-q8-extended-paged)
- [Qwen3-VL Q8 paged conditioner](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-paged)
- [MiniMax H3 video VAE MLX Q8](https://huggingface.co/Vayden/MiniMax-H3-Video-VAE-MLX-Q8)
- [drbaph v4 step-600 Turbo LoRA](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI)
- [H3 tiny preview decoder](https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors)

Use the genuine Ref2VA partition for Ref2VA workflows. The profiled UI graphs do not enable the FL2VA
compatibility override.

## Experimental H3 reference paging

The [Q8 paged Ref2VA workflow](workflows/performance/ref2va/h3_ref2va_q8_paged_experimental.json)
and [matching API graph](examples/h3_ref2va_q8_paged_api.json) use genuine Ref2VA transformer
weights with the existing Q8-extended profile and four-block paging. They start at 640×384,
five seconds, one image reference and 19 dense evaluations. Chain another **H3 Reference Image**
for more images. More reference rows increase memory and sampling time.

Qwen paging v2 retains a separate vision page. Each reference's visual features are materialized
before the vision tower is released; language layers then load sequentially. Existing text-only
v1 exports remain supported for T2VA and are rejected for visual conditioning. The converter uses
bounded file copying for the Qwen pages and preserves packed Q8 storage. Hashes are verified by
default. Conversion does not download weights or change the originals.

**To skip encoder conversion**, download the
[preconverted Q8 vision encoder](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged)
through guided setup or the CLI:

```bash
python scripts/setup_models.py download h3-qwen-q8-vision-preconverted \
  --destination /path/to/shared-models
```

The encoder directory is `/path/to/shared-models/h3-qwen-q8-vision-preconverted`. Select it as the
text encoder in guided H3 reference setup alongside your genuine Ref2VA transformer and other
components. Download `h3-ref2va-q8-preconverted`, `h3-ref2va-support` and
`h3-video-vae-q8-preconverted` through the same command to complete the set. All conversion commands
below are optional when using these prepared downloads; retain them for users preparing their own files.

For manual conversion, the compact source is **`text_encoder.safetensors` plus `config.json`** from
[ddalcu’s 8-bit bundle](https://huggingface.co/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit/tree/64314cde0ac6d90f132bc94ae58e0c82f77396c6).
Its FL2VA bundle name does not make its transformer a Ref2VA transformer. The separate full architecture
configuration comes from [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/0cfaf48183f594c314753d30a4c4974bc75f3ccb/config.json).
Retain source LICENSE, NOTICE and modification records and review their applicable terms. The guided
**H3 Q8 vision encoder · Convert from source** preparation handles these exact files automatically;
it does not download the entire source bundle. Text-only v1 pages cannot be upgraded without vision weights.

Prepare the genuine native Ref2VA transformer, then the **compact Q8 Qwen encoder containing
vision weights**. Do not use the FL2VA Q8 transformer linked above as a Ref2VA substitute.
`--architecture-config` must identify the full Qwen3-VL architecture, including the original
64-layer text configuration and vision configuration. This tool repackages an existing compact
Q8 encoder; conversion/quantization of a full raw Qwen checkpoint is not implemented here.

```bash
python scripts/convert_mixed_checkpoint.py /path/to/Ref2VA/transformer \
  /path/to/ref2va-q8-extended --profile q8_extended --max-shard-mib 512
python scripts/convert_paged_checkpoint.py /path/to/ref2va-q8-extended \
  /path/to/ComfyUI/models/MiniMax-H3/transformers/ref2va-q8-extended-paged
python scripts/convert_paged_text_encoder.py /path/to/compact-qwen-q8 \
  /path/to/ComfyUI/models/MiniMax-H3/text_encoders/q8-vision-paged \
  --include-vision --architecture-config /path/to/full-qwen-config.json
```

The first two commands were exercised against all 13 genuine native Ref2VA shards, with source
SHA256 values matching the download records. Combined transformer preparation used a measured
5.86GB full-process peak on the validation Mac. Keep sufficient disk space for the original,
intermediate mixed checkpoint, and final pages; paging reduces active memory, not model storage.

A saved Comfy API render completed all 19 evaluations and published 124 frames at 24 fps with
stereo 32 kHz audio. At 640×384 with one image, the complete Comfy process peaked at **21.70GB**;
the largest instrumented MLX phase peaked at 8.30GB. Audio/video duration drift was 8.33 ms.
The matching headless recipe produced a byte-identical MP4 at a 21.26GB complete-process peak,
imported no ComfyUI modules and released every weighted runtime. Studio recipe composition,
clip-job export/preflight and the rebuilt application passed their checks.
This was measured on an M3 Ultra with 256 GiB of memory, not a physical 36GB Mac. The render
retains the reference subject and scene in the inspected frames; broad identity/quality testing
remains open. Do not substitute the smaller MLX-only figure for the full-process measurement.

The preflight's 26GB value is an estimate budget, not a hard memory cap. Its weight-stage estimate
includes vision paging but excludes media-dependent reference workspace. Suitability for a physical
36GB Mac and larger reference sets still needs qualification. Smaller hosts can reread more pages
from SSD instead of retaining filesystem cache; do not transfer the larger host's runtime to them.
Q8 changes numerical results relative to BF16. Small FP32/BF16 image/video tests establish paging
parity with the equivalent resident Q8 encoder, not full-model BF16 generation parity.

For Studio or a headless clip, prepare a recipe using the same model layout:

```bash
python scripts/prepare_h3_reference_recipe.py --models /path/to/ComfyUI/models \
  --reference /path/to/hero.png --prompt-file /path/to/prompt.txt \
  --output /path/to/H3_Reference_Q8_Paged.json
python scripts/render_headless.py --recipe /path/to/H3_Reference_Q8_Paged.json \
  --output-directory /path/to/new-render
```

Recipe preparation validates models and media before writing a new file. Import that recipe in
Studio Runtime Settings, select it for an H3 clip, attach the desired image with the **Reference**
role and write the native H3 prompt. Studio's clip/movie job export retains the same paged model
paths and conditioning. Model preparation remains an explicit setup step; guided setup and the CLI
can perform catalog preparations.

## LTX 2.5 model layout

For the distilled baseline, [download the complete preconverted Q8 package](#ready-to-use-q8-downloads)
and scan it in Studio. Its `transformer/` and `gemma/` directories are directly loadable paged
components. The BF16 layout below remains useful for source conversion and additional workflows.

Accept the [LTX 2.5 license](https://huggingface.co/Lightricks/LTX-2.5), then place the split files
in standard ComfyUI folders.
These are BF16 reference filenames; supported Q8-paged replacements can be used for the
corresponding transformer and text encoder. A workflow does not need both precisions installed.

| ComfyUI folder | Component / workflow dependency |
| --- | --- |
| `models/diffusion_models/` | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` |
| `models/diffusion_models/` | Guided modes / official DFR: `ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| `models/loras/` | Guided modes: `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` |
| `models/text_encoders/` | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| `models/vae/` | `ltx-2.5-video-vae-conv-bf16.safetensors` |
| `models/vae/` | Optional detail decoder: `ltx-2.5-video-vae-bf16.safetensors` |
| `models/vae/` | `ltx-2.5-audio-vae-bf16.safetensors` |
| `models/latent_upscale_models/` | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| `models/latent_upscale_models/` | Optional DFR temporal refinement: `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` |
| `models/model_patches/LTX-2.5/` | Optional automatic duration: `ltx-2.5-duration-head-bf16.safetensors` |
| `models/loras/LTX-2.5/` | Optional DFR: [`ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) |
| `models/loras/` | Optional IC control: [`ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) |
| `models/loras/` | Optional Motion Track: [`ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) |
| `models/loras/` | Optional Ingredients reference sheet: [`ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients) |
| `models/loras/LTX-2.5/` | Optional multi-subject reference: [`LTX-2.5-Licon-MSR-V1.safetensors`](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference) |
| `models/loras/LTX-2.5/` | Optional CrossView control: [`LTX2.3-22B_IC-LoRA-CrossView-Warp_v2_6000.safetensors`](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2) |
| `models/controlnet/` | Optional H3 Union branch: [`MiniMax-H3-Fun-Controlnet-Union.safetensors`](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union). Its MiniMax H3 Community License excludes the U.S., EU, UK, and Republic of Korea; do not download or run it in those territories. |
| `models/diffusion_models/` | Derived DFR stage one: `ltx-2.5-22b-dev-distilled450-q8-paged/` |
| `models/diffusion_models/` | Derived DFR stage two: `ltx-2.5-22b-dev-distilled450-detail2x-q8-paged/` |

The official 8+3 distilled workflow remains the recommended compatibility baseline. For eligible
long full-resolution clips, the experimental eight-forward single-stage Sol workflow can be faster,
but it changes the attention path and must report fused calls in generation metadata. For guided
generation, connect **LTX 2.5 Guided Model Loader** and **LTX 2.5 Quality Mode**. **Production guided** runs 30 guided Euler
iterations with CFG, STG, and audio-video modality guidance. **HQ guided** runs 15 second-order
`res_2s` iterations with CFG and modality guidance. Both reload the development transformer with
the official rank-450 distilled LoRA for the three full-resolution refinement iterations. Guided
iterations require several transformer predictions, so the displayed iteration count is not a
claim about total transformer forwards.

**LTX 2.5 LoRA Loader** accepts compatible local safetensors regardless of their download source or
filename. Resident generation normalizes A/B, PEFT default/turbo A/B, down/up, and lowercase A/B
pair names to the MLX loader layout without rewriting the file. Per-target alpha uses
`alpha / pair rank`; consistent `lora_*`, `ss_network_*`, or `network_*` global alpha/rank metadata
uses `alpha / declared rank`, or the pair rank when no global rank exists. Absent scaling metadata
means unit baked scaling, matching the official loader. Common PEFT base-model prefixes and Comfy/Diffusers target
names normalize to one target tree. Low-RAM streaming uses normalized block readers with the same
supported pair schemas and alpha scaling. Unsupported targets and adapter formats fail explicitly;
structural acceptance does not qualify every checkpoint combination visually.

Task-specific LTX adapters are classified from explicit metadata and complete structural
fingerprints rather than filenames. The released CrossView, Ingredients, Union, Motion Track,
Pixel-Spatial, and MSR layouts remain recognizable after arbitrary file renaming. A partial or
unknown reference adapter reports `unclassified_reference_conditioning` and cannot silently enter
a task-specific pipeline.

**LTX 2.5 Media Conditioning** provides one composable typed stack. Image keyframes execute through
the current Generate node. Audio-driven input freezes the encoded audio during both visual stages
and publishes the original source track. General video-reference input requires the dedicated
**LTX 2.5 IC-LoRA Loader**, which scopes the task adapter to stage one and reloads a clean stage-two
transformer. Lightricks' official LTX 2.5 workflows use selected LTX 2.3 22B IC-LoRAs. The loader
therefore accepts an older adapter only when every target and tensor shape matches the LTX 2.5 22B
block layout. The loader still rejects the specialized Pixel-Spatial upscaler from this path.

Use **LTX 2.5 IC-LoRA Control Guide** for Canny edges, depth maps, pose skeletons, Motion Track, or
another preprocessed control video. Connect the IMAGE batch from the matching preprocessor.
Motion Track does not accept raw source frames. Connect the colored trajectory video from
**Motion Track Guide (MLX)**, which supports multiple normalized or pixel-coordinate tracks,
spline control points, and explicit per-frame coordinates. Canny preserves edges and composition.
Depth preserves camera movement and scene geometry. Pose transfers human movement. Use one control
group at a time as the default memory policy. IC-LoRA requires the distilled model. Combined video
and audio reference input remains gated as an unvalidated LipDub topology.

For camera-view synthesis, connect **LTX 2.5 CrossView Camera Orbit** to **LTX 2.5 CrossView Warp**,
then use **LTX 2.5 CrossView Dual Reference Guide**. In **Mode: place camera**, drag the stock-Comfy
sphere to set azimuth and elevation and use the wheel for distance. Switch to **Mode: rotate view**,
or Shift/right-drag, to inspect the sphere from another axis without changing any camera pose.
Drag horizontally/vertically for yaw and pitch; Option-drag adds view roll.
The green path is sampled with the selected linear, ease, or smooth interpolation instead of
drawing straight endpoint chords. Use the visible numeric widgets for exact or headless workflows.
Green marks the best-tested adapter range. The warp node still works independently for
older workflows. Connect the source movie and its MLX Video Depth Anything output to the warp node.
For a moving camera path, choose a frame in the Orbit node, position the sphere, and click
**Add / Update**. Repeat for additional frames. The node draws the ordered keyframes and the backend
interpolates a camera pose for every source frame. The balanced workflow saves the complete moving
magenta-hole warp guide as a video before final LTX regeneration.
Connect **Get Video Components → audio** directly to **LTX 2.5 Generate → publication_audio**. This
preserves the source soundtrack during final muxing; it does not audio-condition sampling and avoids
replacing the soundtrack with generated audio. CrossView treats this connection as required and
fails before model loading when it is absent, preventing a long render with generated gibberish.
The guide preserves the checkpoint's required reference order: warped video first, source video
second. Keep both reference strengths at `1.0`. Start with azimuth within `±45°`, elevation from
`−20°` through `+30°`, and the exact prompt `crossview`. Use adapter strength `1.3` for people or
`1.0` through `1.15` for rigid objects. The v2 checkpoint has weak distance control. The balanced
workflow scales the source to 768×512 before preprocessing, streams normalized depth one frame at
a time, reuses at most two small projection grids, and streams the complete warp batch into the
native Comfy video writer. These policies bound memory without reducing the guide to a still image.

The balanced CrossView workflow also stacks the Ingredients reference IC-LoRA after CrossView.
**Get Image from Batch** selects one clear source frame, **Preview Image** shows the exact frame,
and **Ingredients Reference Sheet** appends character and scene identity conditioning. Change the
batch index when the default frame does not show every important subject clearly. The portable
workflow contains no image or video path. CrossView plus Ingredients is the validated two-adapter
limit; do not add a third task adapter.

For speaking sources, paste the dialogue verbatim into the Ingredients guide's generated-video
description. The workflow publishes the untouched source soundtrack, but publication audio does
not condition sampling. The current LTX 2.5 adapter path intentionally rejects combined video
reference plus audio-reference conditioning because a compatible LipDub topology has not been
validated. The transcript gives the model a mouth-motion cue and usually improves alignment, but
it is not sample-exact lip-sync.

For separate character, object, clothing, and background images, attach **LTX 2.5 MSR Loader**
and chain one to five **LTX 2.5 MSR Reference Stack** nodes. The stack preserves connection order,
moves the optional background to the final learned slot, and emits `Image 1` through `Image 5`
prompt guidance. MSR currently uses the full-resolution single-stage distilled path and cannot be
combined with another IC-LoRA, video-reference stack, or audio-reference stack. Each reference is
encoded independently; subject and object images are fitted without cropping, while the optional
background is center-cropped. `auto` selects 25 reference frames when Sol Attention is enabled and
33 for dense attention. `sol_auto` starts from the quality canvas and, only when needed, chooses the
largest no-upscale 32-pixel grid whose reference rows align to the fused 64-row key tile. Reports
record requested and resolved frame counts, dimensions, row counts, alignment, and any layout
adjustment. Automatic reference priority assigns full density to the first two non-background
references, supporting density to the next two, and background density to the fifth reference. At
1,152-by-640 with 25 frames, those tiers use 2,880, 1,152, and 576 rows. Explicit priority overrides
the assignment. A 33-frame full-quality reference produces 3,600 rows and safely falls back to
dense attention. Prefer one clean hero view per reference. Turnaround sheets containing repeated
figures, detached studies, or inset portraits can be copied as scene content; crop or prepare a
single-subject canvas before an expensive run. The loader validates the learned Fourier-slot
tensors and all 480 rank-128 adapter pairs before execution, and reads the five BF16 slot tensors
through MLX without evaluating the full adapter.

The Union Control checkpoint covers Canny, Depth, and Pose and declares `ref0.5`, so its reference
video is encoded at half of the active generation stage. The official two-stage workflow applies
the guide during the half-resolution first stage and therefore requires final width and height to
be divisible by 128. The shipped workflows use 768×512. **Canny Preprocessor (MLX)** executes the
complete Gaussian-blur, Sobel, non-maximum suppression, threshold, and hysteresis path on MLX.
**Video Depth Preprocessor (MLX)** creates the frame-aligned depth guide with a weighted MLX model.
**DWPose Preprocessor (MLX)** creates the frame-aligned whole-body pose guide with staged detector
and pose-model residency.

### MLX control preprocessors

Control preprocessors are independent from the H3 and LTX generation runtimes. A preprocessor
loads only when its node executes and returns a normal ComfyUI IMAGE batch. This boundary lets one
guide serve LTX 2.5 Union Control or H3 Fun ControlNet without keeping a generation model resident.
For H3, connect the guide to **H3 Encode Fun Control Video (MLX)** after selecting the checkpoint
with **H3 Fun ControlNet-Union Loader (MLX)**, then connect the encoded control to **H3 Sample**.
The encoder holds/trims time, cover-crops to the generation canvas, and stages the H3 video VAE
before the transformer branch is loaded.

Implementation order follows control usefulness and available checkpoint terms:

| Category | Options | Default direction | Status |
| --- | --- | --- | --- |
| Edges | Canny; TEED soft edge | Canny for exact structure; TEED for learned contours | Both available |
| Depth | Video Depth Anything Small; Depth Anything V2 Small | Video depth for consistency; frame depth for speed | Both available |
| Pose | whole-body pose; body-only pose | Whole-body for face and hand motion; body-only for speed | DWPose available |
| Structure | depth-derived normals; realistic line art; segmentation | Normals, line art, and Florence-2 text masks are available | Available |
| Motion | colored trajectory guides; automated optical-flow extraction | Use the optical-flow extractor or manual Motion Track guide with the dedicated adapter | Both available |

The MLX Canny defaults match current ComfyUI normalized thresholds: low `0.4`, high `0.8`, a 5×5
Gaussian kernel, sigma `1.0`, and hysteresis enabled. Lower thresholds retain more weak contours.
The output keeps the input frame count, width, and height.

**Motion Track Guide (MLX)** independently implements the dedicated adapter's colored-guide
contract. Enter one or more tracks as JSON, choose normalized or pixel coordinates, and select
spline control points or per-frame coordinates. The node interpolates sparse control points,
renders a 50-frame age trail on black using the training BGR color convention, reports its resolved
tracks, and checks ComfyUI cancellation while constructing the batch. On the development M4 Max,
two spline tracks over 121 frames at 768×512 rendered in 0.30 seconds (400.8 fps) at a 545 MiB MLX
peak. The full quality workflow then completed in 314.61 seconds with 15 real forwards, a 16.92 GB
MLX peak, and an 18.90 GB complete-Comfy lifetime peak. The `ref0.5` adapter encoded the guide at
384×256 while delivering the requested 768×512 output.

**Optical Flow Motion Tracks** detects strong points in the first source frame, tracks them with
pyramidal Lucas–Kanade flow, rejects inconsistent forward/backward observations, and renders the
same training-color guide. Its normalized per-frame JSON remains editable and portable. Reports
include held observations and the valid-observation ratio; flat clips use one explicit stationary
center fallback instead of silently returning no control.

**TEED Model Loader (MLX)** and **TEED Soft-Edge Preprocessor (MLX)** provide a learned contour
option using the MIT-licensed 58K-parameter Tiny and Efficient Edge Detector. The 233 KB converted
checkpoint processes a 121-frame 768×512 guide in 0.97 seconds (125.4 fps) on the development M4
Max, with a 1.86 GB MLX peak at the default eight-frame chunk. All four MLX output heads match the
official PyTorch model at relative L2 no worse than `1.86e-6`.

Download official `7_model.pth` from [TEED](https://github.com/xavysp/TEED), then convert it once:

```bash
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/convert_teed_mlx.py" \
  7_model.pth \
  "$COMFYUI_ROOT/models/annotators/teed/7_model_mlx.safetensors"
```

**Video Depth Model Loader (MLX)** selects a converted Video Depth Anything Small checkpoint
without loading it. **Video Depth Preprocessor (MLX)** processes 32-frame temporal windows with
the trained overlap-alignment policy. The default unloads the depth model before LTX sampling.
Use input size `518` for the quality default. Smaller input sizes reduce preprocessing cost. On the
development M4 Max, a 121-frame 768×512 guide took about 5.0 seconds at the default chunk sizes.
Chunking the high-resolution decoder reduced measured MLX peak allocation from 12.92 GB to 6.00 GB
without changing a single output value. These figures describe preprocessing only, not total
ComfyUI generation memory.

Download the Apache-2.0 Small checkpoint from
[Video Depth Anything Small](https://huggingface.co/depth-anything/Video-Depth-Anything-Small),
place it under `ComfyUI/models/annotators/video_depth_anything/`, and convert it once:

```bash
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/convert_video_depth_anything_mlx.py" \
  "$COMFYUI_ROOT/models/annotators/video_depth_anything/video_depth_anything_vits.pth" \
  "$COMFYUI_ROOT/models/annotators/video_depth_anything/video_depth_anything_vits_mlx.safetensors"
```

The converter requires PyTorch only to read the original checkpoint safely. Generation-time depth
inference and all weighted depth operations use MLX. Do not redistribute Base or Large Video Depth
Anything checkpoints with this project; their checkpoint terms are noncommercial.

**Fast Depth Model Loader (MLX)** loads the standard Apache-2.0 Depth Anything V2 Small Hugging
Face safetensors directly. No conversion is required. **Fast Depth Preprocessor (MLX)** processes
frames independently and is the speed-oriented alternative to Video Depth Anything. Use per-clip
normalization for a shared depth range. Independent frames can flicker when lighting or scene
content changes abruptly, so use Video Depth Anything for the quality default.

On the development M4 Max, the BF16 speed default processed 121 frames at 768×512 in 1.73 seconds
(70.1 fps) with a 1.21 GB MLX peak. The float32 MLX model matches the official Transformers output
at relative L2 `1.29e-6`. Place the standard checkpoint at:

```text
ComfyUI/models/annotators/depth_anything_v2/Depth-Anything-V2-Small-hf/model.safetensors
```

Download it from
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).

**Depth to Normal Map (MLX)** converts any grayscale depth IMAGE batch into an RGB surface-normal
guide with the standard +Z-blue channel convention. The node has no checkpoint. Sobel gradients
and strength `40` are the normalized-depth defaults. A discontinuity threshold can replace unstable
depth edges with a forward-facing normal. The 121-frame 768×512 probe took 0.43 seconds (284 fps)
with a 0.86 GB MLX peak. Match the normal-map convention to the target adapter before generation.

**Line Art Model Loader (MLX)** and **Realistic Line Art Preprocessor (MLX)** support the fine and
coarse realistic line-art checkpoints used by current ComfyUI auxiliary preprocessors. The node
defaults to ComfyUI-compatible white lines on black and two-frame chunks. The fine model processed
121 frames at 768×512 in 2.88 seconds (42.1 fps) with a 3.52 GB MLX peak. Its output matches the
Apache-2.0 PyTorch reference at relative L2 `1.24e-7`.

Download `sk_model.pth` or `sk_model2.pth` from
[lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators), then convert each selected
checkpoint once:

```bash
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/convert_lineart_mlx.py" \
  sk_model.pth \
  "$COMFYUI_ROOT/models/annotators/lineart/realistic_lineart_fine_mlx.safetensors"
```

The converter requires PyTorch. Runtime line-art inference uses only MLX. Use line art only with an
adapter that was trained for the same representation; Union Depth does not become Line Art by
changing the guide image.

**DWPose Model Loader (MLX)** selects converted YOLOX-L person-detection and DWPose-L whole-body
bundles without loading them. **DWPose Preprocessor (MLX)** renders body, face, and hand keypoints
by default; body-and-hands and body-only modes reduce guide detail. The models unload before LTX
sampling unless keep-warm is selected. On the development M4 Max, a 121-frame 768×512 guide took
5.78 seconds (20.9 fps), with a 2.11 GB MLX peak. The converted detector and pose logits match ONNX
Runtime at relative L2 `2.86e-6` and at most `2.43e-6`, respectively.

Download Apache-2.0 `yolox_l.onnx` and `dw-ll_ucoco_384.onnx` from the
[official DWPose model repository](https://huggingface.co/yzd-v/DWPose). Install the conversion
extra once, then create the local MLX bundles:

```bash
"$COMFYUI_ROOT/.venv/bin/python" -m pip install \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes[convert]"
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/convert_onnx_mlx.py" \
  yolox_l.onnx \
  "$COMFYUI_ROOT/models/annotators/dwpose_mlx/yolox_l"
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/convert_onnx_mlx.py" \
  dw-ll_ucoco_384.onnx \
  "$COMFYUI_ROOT/models/annotators/dwpose_mlx/dw-ll_ucoco_384"
```

ONNX is conversion-only. Runtime detection, pose estimation, and all weighted operations use MLX.

Use **LTX 2.5 Ingredients Reference Sheet** when one composite image defines characters, props,
and a location. The node creates the trained two-part `Reference sheet:` / `Generated video:`
prompt and repeats the still internally across the complete reference timeline. Connect **LTX 2.5
IC-LoRA Pipeline Mode** and select **CFG++ quality**. This uses Ingredients LoRA strength 1.2,
empty negative conditioning, and eight CFG++ sampler steps. It performs **15 useful real
full-resolution transformer forwards**: each nonterminal step uses conditional and unconditional
predictions, while the terminal update consumes only the conditional clean estimate. The optional
CFG++ schedule can reduce this to **12 forwards (balanced)** or **10 forwards (speed)** by applying
the correction selectively. On the matched 768×448, 121-frame validation, balanced reduced total
time from 378.9 to 359.5 seconds (5.1%), while speed reduced it to 304.1 seconds (19.7%); neither
schedule reduced peak memory. Select **Fast single stage** for the distinct eight-forward ancestral
shortcut. Automatic CFG++ execution selects serial. Experimental batched execution reduced peak
memory but was slower on the measured system. All single-stage modes keep the IC-LoRA active for
the complete generation and skip the spatial upscaler. The recommended training bucket is
768×448, 121 frames, and 24 fps. The reference sheet is context; it is not pasted into the first
output frame.

Reference sizing is independent from the output canvas. **Quality** retains the largest source and
target-compatible 32-pixel grid, **balanced** caps the sheet near 512×288, and **speed** caps it
near 384×224. The effective size, reference rows, target rows, and dense-attention multiplier are
recorded in generation metadata. In a matched 1344×768 Q8 Ingredients sweep, reducing the same
768×448 sheet to 512×288 cut sampling from 492.83 to 376.67 seconds (23.6%); 384×224 cut it to
323.46 seconds (34.4%). All three retained both test identities, while the smaller grids changed
framing and motion and can weaken tiny facial or accessory details. Peak MLX allocation remained
about 18.30 GiB, so this is a speed policy rather than a demonstrated memory reduction.

Full-resolution single-stage chaining can also use Sol. In a two-window 1344×768 validation, both
windows completed 384 fused calls with zero fallbacks. Window one used 16,128 target rows and
avoided 72.5% of dense key-row work; window two used 20,160 rows including 4,032 exact continuation
rows and avoided 47.3%. The nine-second workflow completed in 701.8 seconds at a 34.02 GB MLX peak.
Frame and audio-spectrum review found a continuous join. The unguided first window duplicated one
subject before the seam, so this validates execution and continuity—not strict character-count
adherence.

The video-refine workflow also requires the
[pixel-spatial upscaler IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)
under `models/loras/LTX-2.5/`.

The shipped balance video-refine workflow selects the Q8-paged transformer and Gemma pack with
`low_ram_streaming=true`. This avoids keeping both full BF16 weighted stages resident during the
2× refinement pass. A previous workflow revision selected resident BF16 components and could push
complete-process memory above 100 GB on a large input. The upscaler now also preserves Pixel
Spatial full-video conditioning when endpoint anchors are enabled; the two controls are additive.

The upscaler reuses exact Gemma prompt conditioning for repeated refinements with the same prompt,
context policy, and checkpoints. The cache holds two entries in the ComfyUI process. Use the
LTX 2.5 Unload node to clear it. Disable reuse when prompt outputs must not remain resident.

For dimensions outside the LTX grid, the default input policy selects a nearby 32-pixel canvas and
uses Lanczos resizing while keeping aspect error below 0.5 percent. The learned 2× output then lands
on the Pixel-Spatial 64-pixel grid. A centered-crop policy and a strict-grid policy remain available.
The node records output frame-megapixels and accepts an optional preflight limit. Treat that limit
as a workload guard, not a memory prediction.

Long Pixel-Spatial refinements can use the opt-in `auto scene-aware` temporal mode. The planner
prefers detected scene cuts, enforces exact frame coverage, keeps every weighted window at or above
49 frames, stores completed chunks for safe resume, and remuxes the untouched source audio once.
Use a measured frame-megapixel budget for the Mac. Keep chunking disabled when the complete clip
fits, because a forced boundary inside one continuous shot can remain visible. A matched 98-frame
672×384 to 1344×768 Q8-paged test used two 49-frame windows, completed in 148.57 seconds, and
reached a 10.90 GB MLX peak. The second window reused prompt conditioning and skipped 8.69 seconds
of Gemma work. An earlier two-by-nine-frame test reached 9.31 GB but produced unacceptable visual
quality; the node now rejects that configuration.

Resume state uses the source-and-settings fingerprint, not the final movie filename. A forced
republication under a new ComfyUI output name reused both weighted chunks and completed the node in
1.52 seconds instead of 148.57 seconds.

Pixel-Spatial refinement is a generative repaint. It can invent face details, logos, lettering, and
small objects. Compare identity-sensitive output with the source before publication.

Use `scripts/convert_ltx25_paged_q8.py` to create directly loadable Q8 pages. Keep the licensed BF16
source files and checkpoint terms. Generated page directories are model artifacts and must not be
committed.

The balance and speed Ingredients workflows use a self-describing Q8 transformer with the
Ingredients adapter baked at strength 1.2. Build the transformer pages and Gemma pages once, then
bake the adapter from the original transformer pages:

```bash
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/convert_ltx25_paged_q8.py" \
  transformer \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-distilled-transformer-q8-paged"

"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/convert_ltx25_paged_q8.py" \
  gemma \
  "$COMFYUI_ROOT/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  "$COMFYUI_ROOT/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-q8-paged"

"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/fuse_ltx25_paged_lora.py" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-distilled-transformer-q8-paged" \
  "$COMFYUI_ROOT/models/loras/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-distilled-ingredients1p2-q8-paged" \
  --strength 1.2
```

Build the reusable Union Control pages from the same original Q8 transformer pages:

```bash
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/fuse_ltx25_paged_lora.py" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-distilled-transformer-q8-paged" \
  "$COMFYUI_ROOT/models/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-distilled-union1p0-q8-paged" \
  --strength 1.0
```

One baked Union transformer serves Canny, Depth, and Pose. The matched Canny validation used
768×512, 121 frames, 24 fps, and the official eight-plus-three schedule. Internal generation took
87.79 seconds, complete Comfy wall time was 110.31 seconds, complete-process peak was 19.07 GB, and
the guide encoded to 192×128. The visual review preserved the controlled composition and motion
without a fade or scene reset. Matched Depth and Pose runs used the same dimensions, frame count,
seed, reference-FP32 backend, and eight-plus-three schedule. Depth generation took 92.86 seconds;
its complete graph took 99.58 seconds. Pose generation took 85.07 seconds and its complete graph
took 91.37 seconds. Both reported about 8.20 GiB MLX peak and a 21.74 GiB complete Comfy
process-lifetime peak. Each guide encoded all 121 frames at 192×128, produced synchronized stereo
audio, and preserved coherent subjects and movement without an obvious cut or scene reset.

Do not connect a separate IC-LoRA Loader when selecting the baked transformer. Preflight rejects
double application. In a matched 768×448, 121-frame, 10-forward test, baked Q8 produced the exact
same MP4 as live Q8 adapter fusion while reducing total time from 305.43 to 287.22 seconds and
complete Comfy peak from 18.65 to 18.18 GB. Against the matched BF16 speed run, total time improved
from 304.08 to 287.22 seconds and complete peak fell from 32.07 to 18.18 GB. Q8 remains an
approximation relative to BF16; use the performance workflow when BF16 quality is the priority.

For the optimized DFR workflow, build both adapter page sets from the same original development
transformer Q8 pages. The first command bakes the rank-450 adapter for stage one. The second command
bakes the rank-450 and Pixel-Spatial adapters together for stage two.

```bash
"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/fuse_ltx25_paged_lora.py" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-dev-transformer-q8-paged" \
  "$COMFYUI_ROOT/models/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-dev-distilled450-q8-paged"

"$COMFYUI_ROOT/.venv/bin/python" \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes/scripts/fuse_ltx25_paged_lora.py" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-dev-transformer-q8-paged" \
  "$COMFYUI_ROOT/models/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  "$COMFYUI_ROOT/models/diffusion_models/ltx-2.5-22b-dev-distilled450-detail2x-q8-paged" \
  --extra-lora \
  "$COMFYUI_ROOT/models/loras/LTX-2.5/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
```

Do not build stage two from the stage-one pages. Sequential requantization changes video output.
Preflight verifies the stage-one adapter, stage-two adapter, and selected Pixel-Spatial adapter.

## FastH3 production profile

Use the [FastH3 compact indexed Metal workflow](workflows/performance/t2v/h3_fasth3_compact_vsa_performance.json)
for the native FastH3 VSA student. FastH3's DiT is approximately 35.05B parameters; the 22B label
seen in LTX 2.5 material does not describe H3. The workflow applies five schedule points and four
real transformer evaluations, requires `weetodd-fasth3-vsa-datafree-q8-paged`, and rejects tasks
other than T2VA before sampling.

The **FastH3 Production Profile** node provides four explicit policies:

- **Balanced** uses the measured compact preordered indexed-Metal VSA path and is recommended.
- **Speed candidate** uses the same compact Metal path with 40 of 50 joint video/audio layers.
  It is generatively approximate and remains pending sound-effect listening acceptance.
- **Conservative** retains the grouped MLX VSA consumer as a compatibility fallback.
- **Experimental** adds fused QKV preparation; its small measured gain did not pass promotion.

**H3 Text Encode** can persist text-only conditioning in a bounded 1 GiB safetensors cache under
ComfyUI's user directory. Its `persistent_cache` switch defaults on; turning it off bypasses
disk reuse. Encoder/tokenizer/processor identity, prompt and task are part of the key, and media
conditioning bypasses this cache. Encoder unloading remains independent of feature reuse.
H3 sidecars now distinguish phase-local MLX allocation peaks and their aggregate from the
process counter; cached work from another Comfy prompt does not count toward the current job.

The existing **H3 Generation Config** appends `inference_optimization`: `off` (default),
`transient_q8`, `compiled_adaln`, or `combined`. These are opt-in experiments, not new model
precision settings or promoted speed presets. Transient Q8 uses temporary dense GEMM for eligible
wide projections; generic QMM/dense kernels may differ in rounding. Small MLP chunks retain QMM.
Compiled AdaLN targets only activation scale/add, not the full transformer.

The node also exposes the six measured canvases as a resolution selector. Each entry shows its
M3 Ultra complete time and MLX peak; **Keep Generation Config** preserves a custom canvas. Hardware
checks are advisory only: the profile reports the detected Apple chip and unified memory, warns
when the selected row lacks comfortable measured headroom, and never blocks a run solely because
the current hardware differs from the 256 GB M3 Ultra reference. The shipped graph connects
`profile_info` both to the sampler, so diagnostics persist in the publication sidecar, and to
**FastH3 Profile and Hardware Advisories**, so the same formatted report is visible in ComfyUI.
The ready-to-queue API graph is
[`examples/h3_fasth3_compact_vsa_768x448_api.json`](examples/h3_fasth3_compact_vsa_768x448_api.json).

The [40-layer candidate UI graph](workflows/speed/t2v/h3_fasth3_40layer_candidate.json) and
[matching API graph](examples/h3_fasth3_40layer_768x448_api.json) connect the profile's appended
`fastvideo` output to H3 Sample. Existing output indices are unchanged. The sampler checks the
profile contract before loading weights: native pinned Q8 VSA checkpoint, T2VA, five Euler schedule
points, matching attention/configuration, and no LoRA, cache, forecast, VDN, continuation, or token
pairing. Before publication it requires actual evidence of four evaluations, 40 kept/10 skipped
layers, and 160 compact-Metal calls without fallback. Layer indices and policy identity persist in
the final sidecar. The resolution dropdown's times remain **50-layer Balanced reference values**;
they are not 40-layer measurements. Keep Balanced for quality-sensitive sound synchronization until
the candidate's listening gate is accepted.

Fresh saved-workflow acceptance on M3 Ultra (768×448, 107 frames, 24 fps, staged unloading,
cache disabled) measured the following complete server execution times, including text encoding
and final audio/video publication:

| Scene | Balanced, 50 layers | Speed, 40 layers | Time reduction |
| --- | ---: | ---: | ---: |
| Robot, warm repeat | 152.35 s | 126.08 s | 17.2% |
| Dialogue and cup | 146.20 s | 125.05 s | 14.5% |
| Dog, disc, and surf | 146.72 s | 124.92 s | 14.9% |

All three candidates passed final-media technical checks and actual 160-call execution proof;
the robot's repeated outputs were byte-identical within each profile. Both dialogue versions
transcribed exactly as “Your coffee is ready.” This does **not** establish sound-effect quality
or lip synchronization. Composition and event timing change with layer thinning. The matched
dialogue/motion API fixtures live in `benchmarks/fasth3_acceptance/api/`; reproduce final-artifact
checks with `scripts/verify_fasth3_speed_artifact.py`. These timings are not a high-resolution,
long-duration, or lower-memory hardware qualification.

For actual FastH3 bottleneck attribution, `scripts/profile_fasth3_server.py` launches a dedicated
ComfyUI `--cache-none` server with `off`, `coarse`, or `detailed` normal-path tracing. Execute the
same saved API with `scripts/benchmark_saved_h3_workflow.py`, then join traces, final latent parity,
and MP4 parity with `scripts/summarize_fasth3_profile.py`. Detailed traces synchronize operations
and perturb scheduling; they are not uninstrumented speed measurements. The older dense-only
diagnostic path rejects VSA checkpoints rather than silently substituting different attention.
Profiling does not change generation defaults and records actual paged-block chunk sizes.

The standalone-engine milestone uses `scripts/export_h3_headless_recipe.py` for a one-time
Comfy-side export of the pinned benchmark API, then `scripts/render_h3_headless.py` to render
that resolved recipe without ComfyUI, its server, or the node catalog. The runner blocks Comfy
imports and retains the existing MLX sampler, preview decoder/safety guard, staged unloading,
and synchronized streaming A/V publication. It records raw latents, execution proof, phase
memory, import isolation, and render timings. This is a narrow 40-layer T2VA benchmark with a
warm conditioning cache, not a general graph converter, native rewrite, or packaged desktop app.

The broader `scripts/render_headless.py` runner accepts explicit H3, LTX 2.3, and LTX 2.5
component recipes and blocks ComfyUI imports. Candidate validation uses
`scripts/build_headless_validation_matrix.py`, `scripts/validate_headless_matrix.py`, and
`scripts/compare_headless_matrix.py`. Model files are reused by path; local bundle views link
shared components instead of copying weights. A preflight pass is not a completed render or
quality qualification. H3 recipes accept a separate generic LoRA stack; LTX 2.5 retains its
existing generic/task-specific stacks. LTX 2.3 exposes resident and low-RAM streamed
standard-LoRA stacks across T2V, FFLF, A2V, and extension; accepting a local path does not imply
every trainer format or checkpoint is visually qualified.

`scripts/inspect_model_library.py --root <existing-folder> --output <new-report.json>`
provides read-only model-library discovery without MLX or ComfyUI imports. It recognizes
symlink/hardlink aliases, avoids directory cycles, and reads bounded safetensors headers.
Optional `--hash-duplicates` verifies identical content across distinct same-size files;
equal sizes or filenames alone never establish duplication. It neither downloads nor deletes
weights. This inventory is a foundation for shared model management, not yet a complete
compatibility/conversion loader. Model variants and quantizations are not automatically merged.

The advanced **advisory memory budget** defaults to `0`, which uses detected physical memory. A
manual GiB value is useful for planning around other workloads or testing the warning policy. It
does not constrain MLX, simulate a smaller Mac, or validate that a render completes there. The
current policy marks headroom comfortable only when the budget is at least the larger of 1.35 times
the measured MLX peak or the measured peak plus 4.0 decimal GB. Use
`scripts/analyze_fasth3_memory_budgets.py` to reproduce the policy matrix.

The matched M3 Ultra scaling matrix uses one prompt, seed `20260829`, 24 fps, four transformer
evaluations, and direct synchronized publication. A 4.0-second request aligns to H3's `17n+5`
contract and therefore delivers 107 frames, or 4.458 seconds. Times include sampling, streamed
video/audio decode, and mux; shared text encoding is excluded. Peaks are MLX allocation peaks.

| Resolution | Megapixels | Complete time | MLX peak |
| --- | ---: | ---: | ---: |
| 512×256 | 0.13 | 1m 00s | 5.79 GB |
| 768×448 | 0.34 | 2m 22s | 7.18 GB |
| 1024×576 | 0.59 | 4m 14s | 9.29 GB |
| 1280×704 | 0.90 | 6m 43s | 11.96 GB |
| 1536×832 | 1.28 | 9m 28s | 15.23 GB |
| 1920×1088 | 2.09 | 17m 00s | 22.16 GB |

## VDN-H3 experimental integration

Use the [saved UI workflow](workflows/balance/t2v/h3_vdn_8_step_experimental.json) or its
[matching API prompt](examples/h3_vdn_8_step_api.json). The graph preserves the standard
component loader, preview override, preflight, text encoder, and joint sampler. **Direct Publish
Latents** performs staged VAE decoding and synchronized publication without a persistent
ComfyUI IMAGE tensor. **VDN Checkpoint** supplies the stage-specific config,
hybrid-attention branch, and required LoRA stack; connect all three to **H3 Sample**.

Download the chosen stage from [OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)
under `ComfyUI/models/OpenVDN/vdn-minimax-h3` or a registered shared model root. Retain
`model_spec.json`, `linear_branch/model.safetensors`, and `adapters/` together. The 8-step
`stage-dmd-step-250` uses both `default` and `turbo` adapters; `stage-b-step-2000` uses the
default adapter and 50 evaluations. These are nine and 51 schedule points in the MLX scheduler.
Existing compatible MLX H3 base components can be reused; the full Diffusers base directory
is not a drop-in MLX checkpoint and is not automatically downloaded by this node.

Select a resident or paged H3 base transformer, not a trained FastH3 VSA student.
The saved graph uses the same paged Q8 base as the standard H3 workflows.
For an AdaLN-pruned base, supply its original `h3_silu_temb_grid.safetensors` using the optional
`adaln_input_grid` field. The node also checks beside the Turbo adapter and transformer for
that filename. An ordinary unpruned base does not require the grid.

Status: experimental. Small-array numerical tests, resident/paged joint-DiT tests, and an
end-to-end 8-step checkpoint smoke render pass. The saved graph produced 124 frames at 672×384,
24 fps, with stereo 32 kHz audio and all 400 hybrid-attention calls. On an Apple M3 Ultra,
grouped windows and an FP32 Metal Cholesky solver reduced the matched workflow from
36.29 to 30.36 seconds per evaluation (16.3% less time), and from 5m33s to 4m45s end to end.
All 800 solver calls used Metal, with no fallback. Each new matrix-batch geometry is checked
against the CPU inverse on first use; a failed check or unsupported kernel latches a reported
CPU fallback. The solver uses FP32 throughout, but the final render is not pixel-identical to
the CPU reference. That solver-only measurement used the previous publication graph. No new
dependencies are required; restart ComfyUI after updating. This is an improvement over the
initial VDN port, not a demonstrated speed upgrade
over ordinary H3. Original-runtime full-checkpoint parity remains untested. A separate 50-step short render
completed and matched its saved ComfyUI control exactly; this does not qualify full-resolution quality. The 384p example is a wiring test, not a quality
benchmark. Do not combine VDN with cache, forecast, sparse-attention,
FastVideo, continuation, or Hi Res Fix controls. Base-only H3 memory estimates do not include the
additional VDN branch and adapters. Review the model's Community License before use.

The current paged graph uses `projection_backend=auto`, the fused five-tap temporal filter,
and previews every two evaluations (plus the first/final safety checks). The Q8-extended
checkpoint is mixed precision: 118 of its 200 block projections remain BF16. Verified MPP
can accelerate those on supported hardware; its 82 Q8 projections remain unchanged. Fused
temporal filtering preserves tap order and BF16 rounding, verifies the first geometry, and
latches a reported reference fallback on failure. LoRA batching candidates are available in
`scripts/benchmark_vdn_hotpaths.py` but are **not enabled**: they matched numerically yet
measured slightly slower on the tested M3 Ultra.

For repeated jobs on a high-memory Mac, use the explicit
[resident/warm UI workflow](workflows/performance/t2v/h3_vdn_8_step_resident_experimental.json)
or its [API prompt](examples/h3_vdn_8_step_resident_api.json).
H3 Sample's `block_residency=resident` retains the
same paged-checkpoint tensors in memory without conversion. `unload_after_sample=false`
keeps that transformer warm between compatible jobs; set it back to `true` on the last job,
or use **H3 Unload**. Text encoding and both VAEs still unload after use. The node defaults
remain checkpoint paging and staged unloading. Resident mode rejects low-memory mode, and
the base-only paged preflight estimate does **not** represent its larger memory footprint.

`scripts/benchmark_saved_h3_workflow.py` executes a saved API prompt unchanged and records
server-side timings and its SHA256. Use a dedicated ComfyUI server with `--cache-none` for
repeated same-seed runs: ComfyUI node caching would otherwise skip the work being measured.
Execution counters reset per sampling run; numerical verification verdicts remain warm.

Measured on M3 Ultra / 256 GB at the same seed, 672×384, 124 frames and eight evaluations:

| Execution | Seconds / evaluation | End-to-end | Complete process peak |
| --- | ---: | ---: | ---: |
| Previous optimized paged graph | 30.36 | 285.27 s | not recorded |
| Current paged graph | 29.96 | 287.64 s | 20.60 GB |
| Resident, initial run | 26.10 | 252.10 s | 52.17 GB |
| Resident, repeated warm run | 26.08 | 249.34 s | 52.67 GB lifetime peak |
| Optimized resident, initial run | 24.92 | 237.47 s | 60.75 GB MLX peak |
| Optimized resident, repeated warm run | 24.78 | 231.11 s | 60.75 GB MLX peak |

The current paged graph did not demonstrate an end-to-end gain; resident execution was
about 12–13% faster than that graph. Keeping the already-resident transformer warm saved
another 2.76 seconds. The two resident MP4s were byte-identical to the current paged MP4;
its decoded video frames also matched the previous optimized render exactly. Audio remained
32 kHz stereo with 8.3 ms A/V duration drift. These are individual controlled smoke runs,
not a multi-prompt quality or performance guarantee. The base-only memory estimate excludes
this measured resident footprint; leave ample room for macOS and other applications.

The optimized resident row uses verified VDN scan/state gathering, selective resident Q8
expansion, the native-layout BF16 video VAE, 272-pixel geometry-aware width tiles, and the compiled
Core ML preview sibling. Against the prior warm row, it reduced sampling from 208.62 to 198.24
seconds and video decoding from 31.24 to 22.83 seconds. Total wall time fell by 18.23 seconds
(7.3%). The optimized video measured 0.9818 full-video SSIM against the byte-identical control;
its extracted AAC stream was byte-identical. MLX peak allocation increased by about 17.5 GB.

### Additional opt-in H3 optimization path

The [optimized resident UI workflow](workflows/performance/t2v/h3_vdn_8_step_optimized_experimental.json)
and [matching API prompt](examples/h3_vdn_8_step_optimized_api.json) add selective resident Q8
expansion and geometry-aware final decoding. They require ample memory and a matching native-layout
BF16 video VAE at `models/MiniMax-H3/vae/bf16/video_vae_mlx_native.safetensors`, or an explicitly
selected existing copy. No weights are downloaded or rewritten automatically.

- VDN `inference_backend=verified` enables compiled scan/state gathering and eligible auxiliary
  MPP projections, with first-geometry numerical checks and reported reference fallback.
  `reference` disables these additional inference optimizations for comparisons.
- `mpp_resident_expanded_experimental` expands selected loaded Q8 projection values once in RAM;
  it does not substitute original BF16 transformer weights or merge adapters. Packed fallback
  weights remain available. It requires explicit resident blocks and normal memory mode.
- Video decode and Direct Publish Latents expose `video_tile_mode=geometry_experimental`.
  Fixed tiling remains the default. The bounded selector reduces redundant overlap but changes
  decoder attention context, so it is not pixel-exact and needs visual review. Encoding of
  references/keyframes is unchanged.
- VDN `indexed_experimental` reads exact static windows directly without K/V gather buffers.
  The attention mask is preserved, but floating-point accumulation differs. It remains opt-in
  and is not selected by the optimized saved workflow.
- Paged execution omits unused AdaLN tensors/adapters after modulation caching and shares adapter
  file mappings within each window only. It does not keep every adapter resident between windows.
- Preview path resolution accepts an existing `.mlmodelc` sibling of a missing `.mlpackage`
  (and vice versa), while preserving an explicitly existing requested model.

## Live H3 previews

Every shipped H3 UI workflow routes components through **H3 Model Preview Override** before sampling.
The node publishes a true-color contact sheet during sampling and releases the tiny decoder after
success, failure, or cancellation.

Historical API examples retain their individual wiring; some omit this optional preview node.
Inspect the saved API prompt and run its live preflight before an expensive render.

The default `auto` backend uses the optional Core ML package on macOS and otherwise falls back to
MLX. Core ML uses CPU and Neural Engine compute units, which avoids adding preview convolution to
the transformer's Metal GPU path. The conservative collapse guard stops only repeated featureless
previews after the schedule midpoint; non-finite latents stop immediately.

## Performance and memory optimizations

The tables below summarize measured production paths, approximate accelerators, and remaining
work. Results apply to the stated workflow and hardware conditions.

### Current high-impact options

| Optimization | Category | Impact | Measured result | Output effect |
| --- | --- | --- | --- | --- |
| H3 paged Qwen3-VL plus q8-extended transformer | Memory | Very high | Matched 384p complete-process peak fell from 28.823 GB to 14.951 GB. Four-block paging remains the default; a one-block test did not reduce complete-process peak and was 18.2% slower. | Qwen paging preserved the MP4 digest. Q8 transformer weights remain approximate relative to BF16. |
| H3 paged MPP projections | Sampling speed | Medium | On M3 Ultra, matched warm q8-paged 384p sampling fell from 113.35 to 108.49 seconds (4.3%) with no peak change. | The final MP4 was byte-identical. Unsupported and quantized projections retain MLX. |
| H3 four-evaluation Turbo | Speed | Very high | Uses four real evaluations instead of the 19-evaluation dense schedule. | Changes the sampling trajectory. |
| H3 staged Turbo | Speed/quality | Very high | Uses two base plus four Turbo evaluations. | Changes the sampling trajectory but retains base-model setup evaluations. |
| H3 direct latent publication | Memory | High | Streams decoded frames to FFmpeg and avoids a persistent complete `IMAGE` tensor. | Does not change sampling. |
| H3 adaptive Ref2VA video density | Ref2VA speed | High | In a matched 640×384 run, automatic half-density reduced sampling time by 32.35% and the observed process peak by 1.08 GB. Automatic mode retains full Qwen inspection and selects persistent VAE-row density from adjacent-frame activity. | Experimental; full density followed source motion more closely and remains the default. Automatic decisions are reported in conditioning metadata. |
| H3 head and FFN row chunking | Sampling memory | Unmeasured; bounded by construction | Independently limits SDPA head groups and packed-row SwiGLU intermediates through the Low-Memory Tuning node. | Same operations and ordering within each independent chunk; checkpoint parity tests are still required before claiming bit identity. |
| H3 token and workspace budget | Preflight | Diagnostic | Reports target, conditioning, packed-token, attention-score, and bounded-workspace estimates before allocation. | Reporting only. |
| LTX 2.5 Q8 paging | Memory and speed | Very high | 9.90 GB and 78.38 s versus 31.60 GB and 102.13 s for matched BF16 one-block streaming at 768×512. | Q8 changes the numerical trajectory. |
| LTX 2.5 fused Sol attention | Long-sequence sampling speed | Experimental | In the matched compact Ingredients test, paged-speed Sol reduced sampling from 358.94 to 323.46 seconds (9.9%) and total time from 379.15 to 343.74 seconds (9.3%). At 17,472 rows, mask storage fell from 610,541,568 to 69,888 BF16 bytes. A matched two-subject MSR run used two exact 2,880-row groups, executed all 384 fused calls, reduced sampling from 411.19 to 390.29 seconds, and reduced MLX peak from 16.05 to 14.58 GB. | Requires at least 16,000 video tokens. It casts eligible FP32 Q/K/V projections to BF16 and uses approximate routing only for target rows, so composition and trajectory can change while reference rows remain exact. Every grouped suffix must align to 64 rows; incompatible grids safely fall back. Use `paged_speed` only with low-RAM Q8 streaming. |
| LTX 2.5 MSR automatic layout | Multi-reference speed, memory, and fused-path reliability | High | Balanced automatic priority reduced a matched five-reference run from 777.90 to 533.98 seconds and complete ComfyUI peak from 21.12 to 17.37 GB. It retained all five identities and completed 384 fused calls without fallback. | `sol_auto` changes only reference density. It never changes target video size. The more aggressive one-primary layout reached 457.59 seconds but weakened one identity, so it remains opt-in. Use one clean hero view per reference. |
| LTX 2.5 Ingredients reference sizing | Reference-conditioning speed | High | On the matched 1344×768 run, balanced 512×288 and speed 384×224 reference grids reduced sampling by 23.6% and 34.4% versus the 768×448 quality grid. | Does not change the source file. It changes encoded reference density and may change fine identity, framing, and motion. Effective rows are recorded in metadata. |
| LTX 2.5 temporal VAE tiling | Decode memory | High for long clips | Activates from an explicit decode-memory budget and grows in value with duration. | Preserves the synchronized output contract. |
| LTX 2.5 generated-keyframe slots | Motion allocation | Experimental | Adds evenly distributed learned interior slots during stage one without changing existing workflow schemas. | Changes the latent token sequence and output. |
| LTX 2.5 Diffusion VAE | Decode quality | Experimental | On a matched fast-motion 512×512 latent, the reference decode took 66.41 s at an 8.41 GB MLX peak. | Runs the official one-step pixel-diffusion decoder and changes decoded pixels. Conv VAE remains the speed and low-memory default. |
| LTX 2.5 Diffusion VAE layouts | Decode performance and compatibility | Experimental | At 768×512 for five seconds, query-tiled Metal reduced decode from 966.45 to 74.72 s and complete-process peak from 164.58 to 14.01 GB. Output measured SSIM 0.98763 and PSNR 46.72 dB against exact decode. | Use the Diffusion VAE Optimization node. The Metal result is approximate. Keep combined mode as the exact reference. |
| LTX 2.5 query-tiled Metal Diffusion VAE | Decode memory | High | A 65,536-row tile reduced matched 512×512 Metal peak from 5.41 to 4.24 GB. Decode time changed from 15.83 to 16.77 s, and the MP4 digest remained identical. | Select `metal_na3d_query_tiled_experimental` only when Diffusion VAE memory has priority. Fine tiles are substantially slower. |
| LTX 2.5 automatic duration | Usability | Low runtime cost | A real Q8 prompt probe spent 0.027 s in the MLX duration head after prompt encoding. | Opt-in modifier; manual duration remains authoritative unless connected. Raw predicted seconds and resolved `8k+1` frames are recorded. |
| LTX 2.5 Diffusion VAE width tiling | Decode memory | Experimental | A 32-cell stage-four stripe reduced 512×512 peak from 8.27 GB to 7.62 GB. | Decode slowed from 61.99 s to 100.13 s and output was not pixel-identical. Select `stage4_width_tiles` only when memory is the priority. |
| LTX 2.5 DFR | Full-resolution detail | Experimental | Exact prebaked Q8 pages completed the matched 256×256 probe in 19.41 s versus 89.91 s with live fusion. At 768×512 for five seconds, sampling took 102.36 s at a 28.91 GB MLX peak. The exact Diffusion VAE then took 966.45 s and drove complete-process peak to 164.58 GB. | Decoded video and PCM audio hashes matched the live-Q8 control at 256×256. Use prebaked pages for DFR sampling. Do not treat the exact Diffusion VAE workflow as a low-memory default. DFR changes composition and motion but preserves stage-one audio. |
| LTX 2.5 DFR temporal refinement | Motion smoothness | Not production-ready | After correcting stage two to deterministic Euler, a 768×512 Q8-paged I2V probe produced 97 frames at 48 fps in 127.97 s and peaked at 9.65 GB. A matched control took 63.58 s and peaked at 9.46 GB. | Streams, audio preservation, first-frame landing, and evaluation counts pass. Both Q8 and BF16 temporal probes develop matching mid-clip color corruption, so quantization is not the cause. Keep this path diagnostic-only. |

### Remaining optimization priorities

| Rank | Candidate | Potential gain | Confidence |
| ---: | --- | --- | --- |
| 1 | Exact reusable LTX 2.5 reference K/V state | High IC-LoRA speed potential beyond density reduction | Medium-low; projections and attention depend on every denoise-step hidden state |
| 2 | H3-specific W4A8 projection kernel | Very high checkpoint and resident-memory reduction | Low |
| 3 | Complete-process BF16 one-block paging validation | Very high BF16 peak-memory reduction | Medium; q8 remains faster with four-block windows |
| 4 | Ref2VA automatic-density quality validation | High Ref2VA speed and memory reduction | Medium; full density remains the default |
| 5 | LTX 2.5 temporal image-conditioning and longer-chain parity | Medium feature completeness | Medium |

## Memory guidance

- Start with the H3 speed profile or LTX 2.5 Q8-paged workflow on a 32 GB system.
- Use staged unloading unless repeated generations justify keeping one component warm.
- Keep H3 caches disabled when memory has priority. Cache states can increase peak memory.
- Use full Ref2VA reference density for quality validation. Lower density is an explicit speed
  trade.
- Use one-block LTX 2.5 streaming. Two-block streaming increased peak memory without improving
  complete generation time in the matched test.
- Measure complete ComfyUI process physical footprint for user-facing memory claims.

## Output and interruption behavior

H3 publishes synchronized H.264 video and 32 kHz stereo AAC audio. LTX publishes 48 kHz stereo
audio. Publication uses a temporary file and atomically replaces the final output only after
validation succeeds.

Generation checks ComfyUI interruption between expensive stages and transformer evaluations.
Staged mode releases the active weighted component after success, failure, or cancellation.

## Node catalog

This table is generated from the registered node contracts. Run
`scripts/update_readme_node_catalog.py` after a node name, description, category, or maturity changes.

<!-- BEGIN GENERATED NODE CATALOG -->
| Node | Notes | Category | Status |
| --- | --- | --- | --- |
| H3 Component Loader | Describe native H3 components, including experimental DT-file T2V references. This node does not load tensor weights. | H3 — Loaders | Recommended |
| H3 Model Preview Override | Attach a true-color TAE preview and optional collapse guard to H3 sampling. Core ML can keep preview decoding on the Apple Neural Engine; MLX remains the fallback. Place this node between the component loader and sampler. | H3 — Sampling and acceleration | Experimental |
| H3 Quantized Transformer Loader | Select and validate a named mixed-precision H3 transformer without loading weights. Both q8 profiles are approximate and keep BlockCache disabled by default. | H3 — Loaders | Experimental |
| H3 Component Preflight | Validate MiniMax H3 components and estimate staged memory from file headers. Vision-capable paged Qwen is supported; reference workspace is not included. Set available memory to zero when unknown. | H3 — Loaders | Recommended |
| H3 Token + Memory Budget | Estimate H3 text, condition, audio, and video rows plus dense-attention scale without loading a checkpoint. Add encoded reference rows for Ref2VA planning. | H3 — Loaders | Supported |
| H3 First Frame | Use one image as the first-frame endpoint for an FL2VA generation. | H3 — Conditioning | Supported |
| H3 Last Frame | Use one image as the last-frame endpoint for an FL2VA generation. | H3 — Conditioning | Supported |
| H3 First + Last Frame | Use two images as the first-frame and last-frame endpoints for FL2VA. | H3 — Conditioning | Supported |
| H3 Chained Timeline | Map global timestamps onto equal-length H3 windows and define exact overlap trimming. | H3 — Continuation | Experimental |
| H3 Frames | Select first, last, and up to six numbered middle images in one visual frame strip. Frame numbers are one-based; the last frame follows the connected generation duration. | H3 — Conditioning | Supported |
| H3 Timed Keyframe | Append an FL2VA image at an exact 24 fps local timestamp, or map a global chained timeline timestamp into one window. | H3 — Conditioning | Experimental |
| H3 Reference Image | Append an image identity, subject, style, or scene reference. Reference order controls the prompt labels and packed rotary positions. A 100% pixel budget matches the output canvas area; lower values reduce persistent reference tokens and higher values retain more source detail. | H3 — Conditioning | Supported |
| H3 Reference Video | Append a video motion and camera reference, with an optional synchronized soundtrack. Supply the source frame rate explicitly. The recommended default matches the output pixel area; native reference resolution is available but can be dramatically slower. | H3 — Conditioning | Experimental |
| H3 Reference Audio | Append a standalone voice, sound, or music reference. Ref2VA also requires at least one image or video reference. | H3 — Conditioning | Supported |
| H3 Timeline Visual Guide | Place one image or an aligned H3 clip on the Ref2VA target timeline. A clip must contain 5, 22, 39, ... frames after 24 fps alignment; optional audio starts at the same frame. | H3 — Conditioning | Supported |
| H3 Timeline Audio Guide | Place an audio guide at an exact Ref2VA target frame. | H3 — Conditioning | Supported |
| H3 Encode First / Last Frames | Encode FL2VA prompt vision rows and first/last-frame VAE rows in separate staged phases. Each weighted component unloads before the next phase. | H3 — Conditioning | Supported |
| H3 Encode Timed Keyframes | Encode up to eight sparse FL2VA images at exact 24 fps timestamps, unloading Qwen3-VL before the video VAE stage. | H3 — Conditioning | Experimental |
| H3 Encode References | Prepare ordered Ref2VA media, then stage Qwen3-VL, the video VAE, and the audio VAE. Resident and vision-capable paged Qwen are supported. Each weighted component unloads before the next stage. | H3 — Conditioning | Experimental |
| H3 Reference Strength | Adjust how strongly FL2VA or Ref2VA trusts visual and audio condition rows. Defaults preserve the released H3 behavior. | H3 — Conditioning | Experimental |
| H3 Text Encode (Qwen3-VL) | Encode a text-only H3 prompt with Qwen3-VL. The vision tower stays unloaded. A bounded persistent feature cache can skip repeat encodes without keeping weights loaded. | H3 — Conditioning | Recommended |
| H3 Unload Qwen3-VL | Release the process-local Qwen3-VL conditioner and clear the MLX cache. | H3 — Conditioning | Supported |
| H3 Motion Continuation Context | Copy a synchronized tail from H3 video and audio latents for motion continuation. The recommended 22-frame overlap is about 0.92 seconds at 24 fps. | H3 — Continuation | Experimental |
| H3 Append Latent Chain Window | Append one synchronized latent window to a validated H3 chained timeline. | H3 — Continuation | Experimental |
| H3 Sample Video + Audio Latents | Sample synchronized MiniMax H3 video and audio latents with MLX. This node does not load or run either VAE. Optional resident block loading avoids repeated paging; staged unloading remains the default. | H3 — Sampling and acceleration | Recommended |
| H3 Learned Latent Upscaler Loader (MLX) | Select an MLX-native learned 3D latent upscaler for H3 Hi-Res Fix. The checkpoint is validated now and loaded only when the graph executes. | H3 — Loaders | Supported |
| H3 Latent Hi Res Fix | Enlarge an H3 video latent and run a second H3 visual refinement pass. The original synchronized audio latent is returned unchanged. | H3 — Sampling and acceleration | Experimental |
| H3 LoRA Loader (MLX) | Build a lazy, ordered MiniMax H3 LoRA stack. Validate safetensors headers now and load adapter tensors only when the H3 transformer executes. Reject malformed A/B pairs and unsupported tensor fields before loading weights. | H3 — Loaders | Supported |
| H3 VDN Checkpoint (MLX) | Select VDN stage, required adapters, and verified inference kernels; optional indexed attention is numerically approximate and experimental. | H3 — Sampling and acceleration | Experimental |
| H3 Validated Sampling Preset | Apply a measured dense, trajectory-replay, or Turbo sampling policy. Connect all three typed outputs to the H3 sampler. | H3 — Sampling and acceleration | Recommended |
| H3 FastH3 Production Profile | Native FastH3 VSA profile with fail-closed schedule, attention, and checkpoint wiring. Balanced is recommended; the explicit 40-layer Speed candidate requires listening acceptance and proves 160 compact-Metal calls before publication. | H3 — Sampling and acceleration | Recommended |
| H3 FastVideo Approximation (MLX Experimental) | Opt-in generative FastH3 approximations. Layer thinning is ranked once from the full AdaLN schedule; token pairing keeps a full-resolution residual bypass. | H3 — Sampling and acceleration | Supported |
| H3 Sparse Attention (MLX Experimental) | Experimental H3 sparse attention. Sol profiles use the fused MLX Metal backend; the FastH3 profiles preserve trained 64-token routing and compression gates with either grouped SDPA or an indexed Metal consumer. Both preserve the complete multimodal prefix. | H3 — Sampling and acceleration | Experimental |
| H3 EasyCache (MLX) | Configure joint MLX EasyCache residual reuse for H3 video and audio sampling. Choose quality-first, balanced, or speed-first bounded automatic reuse. | H3 — Sampling and acceleration | Experimental |
| H3 Trajectory Forecast (MLX) | Experimentally forecast compact post-transformer H3 video and audio features. Current timestep output heads still run on every step. Turbo LoRA is supported. | H3 — Sampling and acceleration | Experimental |
| H3 BlockCache (MLX) | Always run H3 block zero and the current output heads, then safely reuse the cached joint audio/video residual of later transformer blocks when both modality indicators agree. | H3 — Sampling and acceleration | Experimental |
| H3 Hierarchical BlockCache (MLX) | Split the 50 H3 blocks into three contiguous segments. Always evaluate each segment's anchor block, accept video and audio together, and reuse eligible segment tails independently. | H3 — Sampling and acceleration | Experimental |
| H3 Unload Transformer | Release the process-local H3 transformer and clear the MLX cache. | H3 — Sampling and acceleration | Supported |
| H3 Decode Video VAE | Decode final H3 video with fixed tiles or opt-in geometry-aware tiles; audio remains on the synchronized latent output. | H3 — Decoding | Supported |
| H3 Unload Video VAE | Release the process-local H3 video VAE and clear the MLX cache. | H3 — Decoding | Supported |
| H3 Decode Audio VAE | Decode the audio stream from synchronized H3 latents as 32 kHz stereo audio. The video latent stream remains available on the original latent output. | H3 — Decoding | Supported |
| H3 Unload Audio VAE | Release the process-local H3 audio VAE and clear the MLX cache. | H3 — Decoding | Supported |
| H3 Trim Continuation Overlap | Remove the repeated motion-continuation overlap from decoded video and audio, then normalize audio to the exact remaining video duration. | H3 — Continuation | Experimental |
| H3 Publish Video + Audio | Validate and publish synchronized H3 images and 32 kHz stereo audio as MP4. The node writes an atomic JSON metadata sidecar. | H3 — Output | Supported |
| H3 Direct Publish Latents (MLX) | Stream H3 video/audio to MP4 with staged VAE unloading; fixed decode tiles remain default, with experimental geometry-aware tiling available. | H3 — Output | Recommended |
| H3 Direct Publish Chained Timeline (MLX) | Decode an H3 latent chain by VAE stage, remove duplicated joins, force exact 24 fps / 32 kHz duration, and atomically publish one MP4. | H3 — Output | Experimental |
| H3 Model Loader (MLX) | Describe an MLX MiniMax H3 checkpoint. Weights load lazily at generation time. | H3 — Core and convenience | Legacy/convenience |
| H3 Generation Config | Choose a clearly labeled aspect ratio and move the short-edge size slider, or use exact dimensions. The canvas stays on H3's 32-pixel grid. Optional hot-path experiments default off. | H3 — Core and convenience | Recommended |
| H3 Paging Settings (Experimental) | Experimental bounded raw-page retention trades extra memory for fewer repeated H3 checkpoint loads. Disabled by default; original quantization is preserved. | H3 — Sampling and acceleration | Experimental |
| H3 Low-Memory Tuning (MLX) | Apply optional MLX attention-head and feed-forward row chunking without invalidating older Generation Config workflows. | H3 — Sampling and acceleration | Supported |
| H3 Generate Video + Audio | Generate synchronized video and audio with MiniMax H3 through MLX. | H3 — Core and convenience | Legacy/convenience |
| H3 Unload MLX Runtime | Release state held by the monolithic H3 runtime. | H3 — Core and convenience | Legacy/convenience |
| LTX 2.3 IC-LoRA Loader (MLX) | Experimental IC-LoRA with task-aware topology. Union and Motion use resident distilled mode; Ingredients uses Dev two_stage with a validated distilled helper. Declare the trained family; filenames are not used to infer it. | LTX 2.3 — Loaders | Experimental |
| LTX 2.3 Timed Keyframe | Chain timed keyframes for resident Dev two_stage generation. No post-decode frame insertion. | LTX 2.3 — Conditioning | Experimental |
| LTX 2.3 Control Video | Preprocessed local control video, matching output fps and covering every output frame. Does not extract edges/depth/pose/tracks. | LTX 2.3 — Conditioning | Experimental |
| LTX 2.3 Video Extension | Extend one exact 8n+1-frame source before or after. Distilled mode is the qualified eight-evaluation speed path; Dev one-stage remains available for quality. Final publication preserves the source AV prefix and appends groups of eight new frames. | LTX 2.3 — Conditioning | Supported |
| LTX 2.3 Control Frames | Bridge an IMAGE batch from the MLX Canny, depth, DWPose, or motion-track preprocessors into an exact LTX 2.3 IC-LoRA guide. | LTX 2.3 — Conditioning | Experimental |
| LTX 2.3 Ingredients Reference Sheet | Prepare one black-background Ingredients sheet and its trained two-part prompt. Use Dev two_stage mode, 768x448, 121+ frames, 24 fps, and adapter strength 1.4 in the loader. | LTX 2.3 — Conditioning | Experimental |
| LTX 2.3 Model Loader (MLX) | Select a local LTX 2.3 MLX bundle. No weights load in this node. | LTX 2.3 — Loaders | Supported |
| LTX 2.3 LoRA Loader (MLX) | Attach a local standard LTX 2.3 LoRA; chain nodes for ordered stacks. Alpha -1 uses file metadata or rank. Runs on all stages with resident float/Q4/Q8 or low-RAM block-streamed transformers. Task/control adapters use separate loaders and cannot currently be combined with generic LoRAs. | LTX 2.3 — Loaders | Experimental |
| LTX 2.3 Generation Config | Configure LTX 2.3 mode, canvas, duration, steps, guidance, and memory policy. | LTX 2.3 — Core | Supported |
| LTX 2.3 Preflight | Validate the selected LTX 2.3 bundle and mode-specific components before allocation. | LTX 2.3 — Loaders | Recommended |
| LTX 2.3 Generate Video + Audio | Generate synchronized LTX 2.3 video and 48 kHz stereo audio through MLX. | LTX 2.3 — Core | Experimental |
| LTX 2.3 Upscaler Loader | Select and preflight a learned LTX 2.3 spatial latent upscaler. | LTX 2.3 — Loaders | Experimental |
| LTX 2.3 Upscale + Publish | Upscale decoded H3 or other ComfyUI video frames with the LTX latent upscaler and preserve the supplied audio. | LTX 2.3 — Upscaling | Experimental |
| LTX 2.3 Unload MLX Runtime | Release the process-local LTX 2.3 pipeline. | LTX 2.3 — Core | Supported |
| LTX 2.5 Component Loader (MLX) | Select LTX 2.5 split components without loading weights or downloading files. Self-describing paged transformers may contain one prebaked IC-LoRA. | LTX 2.5 — Loaders | Experimental |
| LTX 2.5 LoRA Loader (MLX) | Attach a generic LTX 2.5 transformer LoRA, including attention gates, block and non-block targets. Multiple loader nodes may be chained. Use the dedicated loader for IC-LoRA task adapters. | LTX 2.5 — Loaders | Supported |
| LTX 2.5 IC-LoRA Loader (MLX) | Select and attach an installed LTX 2.5-compatible IC-LoRA for video/reference conditioning. The dropdown scans every ComfyUI loras model root. Up to two distinct task families may be stacked when their reference scale factors match; this supports combinations such as CrossView plus Ingredients character/scene reference. Official LTX 2.3 22B adapters pass an additional shape check. The selected IC-LoRA Pipeline Mode determines whether the adapter runs for stage one or the full generation. Do not use this node with a transformer that already bakes the same IC-LoRA. | LTX 2.5 — Loaders | Experimental |
| LTX 2.5 MSR Loader (MLX) | Attach one LTX 2.5 MSR adapter after validating all learned Fourier-slot tensors and 480 rank-128 transformer pairs. The slot tensors load only when references execute. | LTX 2.5 — Loaders | Supported |
| LTX 2.5 Guided Model Loader (MLX) | Select the LTX 2.5 development transformer for guided stage one and the official rank-450 distilled LoRA for stage two. No weights load in this node. | LTX 2.5 — Loaders | Experimental |
| LTX 2.5 Generation Config | Configure the official distilled 8+3-evaluation LTX 2.5 two-stage schedule. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Quality Mode | Choose fast distilled inference, production guided Euler, or the official HQ second-order res_2s recipe without changing the base Generation Config schema. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Automatic Duration | Predict one-shot duration from the prompt with the official LTX 2.5 duration head. The manual duration remains unchanged when this modifier is not connected. | LTX 2.5 — Core | Supported |
| LTX 2.5 Generated Keyframes | Apply LTX 2.5 generated interior keyframe slots as a composable config modifier. | LTX 2.5 — Conditioning | Supported |
| LTX 2.5 Full-Resolution Single Stage | Run the distilled transformer once at the final resolution without latent upscaling. This experimental path supports T2V and reference-conditioned generation. | LTX 2.5 — Optimization | Experimental |
| LTX 2.5 Sol Attention (MLX Experimental) | Experimental MLX Sol-style sparse video self-attention for long, full-resolution single-stage LTX 2.5 sequences, including Q8 paged mode, exact reference suffixes, compatible structured IC masks, and latent continuation. | LTX 2.5 — Optimization | Experimental |
| LTX 2.5 Diffusion VAE Optimization | Select an MLX Diffusion VAE execution layout. It does not affect the convolutional VAE. | LTX 2.5 — Optimization | Experimental |
| LTX 2.5 DFR Detail Refinement | Enable MLX Diffusion Fidelity Rendering: segment-grid generated keyframes, stage-one latent reference conditioning, stage-two-only Pixel-Spatial IC-LoRA, optional exact prebaked Q8 adapter pages, and untouched stage-one audio publication. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 DFR Temporal Refinement | Experimentally add one or two learned x2 temporal DFR rounds. Each round preserves stage-one audio, doubles playback frame rate, reapplies one-shot image anchors, and adds four transformer evaluations per temporal tile. Current MLX visual parity is not yet production-validated. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 Preflight | Validate LTX 2.5 component metadata and architecture requirements before allocation. | LTX 2.5 — Loaders | Experimental |
| LTX 2.5 Timed Keyframe | Append a first, middle, or last image at an exact zero-based pixel-frame index. The image is encoded as reference conditioning; generated keyframe slots are separate. | LTX 2.5 — Conditioning | Supported |
| LTX 2.5 Media Conditioning | Build a shared LTX 2.5 image, video, audio, or mask conditioning stack. Image keyframes, IC-LoRA video references, and one frozen audio-driven source execute. Standalone inpaint masks remain gated. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 IC-LoRA Control Guide | Add one preprocessed Canny, depth, pose, Motion Track, or custom IC-LoRA guide. Use the LTX 2.5 distilled model and the matching task adapter. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 CrossView Dual Reference Guide | Add the two CrossView IC-LoRA references in the trained order: warp first, source second. Use the v2 CrossView adapter with a reference downscale factor of one. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 IC-LoRA Pipeline Mode | Select full or hybrid CFG++, the eight-forward single-stage shortcut, or the existing two-stage stage-one-control pipeline. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 Ingredients Reference Sheet | Condition LTX 2.5 from one Ingredients reference sheet. The image is repeated internally across the full clip and encoded as IC-LoRA reference context. Quality, balanced, and speed policies control the encoded reference grid independently of the output canvas. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 MSR Reference Stack | Build an ordered one-to-five-image LTX 2.5 MSR stack. Subject and object references stay in connection order; one optional background is always assigned the final slot. Automatic priority gives the first two subjects full density and later references aligned supporting or background density. | LTX 2.5 — Conditioning | Supported |
| LTX 2.5 Generate Video + Audio | Generate synchronized LTX 2.5 video and audio through the MLX adapter. Connect publication_audio to preserve an original soundtrack without conditioning sampling. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Generate Chained Timeline | Generate two to four overlapping LTX 2.5 windows with timeline-aligned latent guides, causal-aware latent transitions, and one synchronized audio/video decode. Supports the two-stage path and full-resolution single-stage Sol configurations. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Video Upscale / Refine | Upscale decoded ComfyUI IMAGE+AUDIO from any movie through LTX 2.5 latent space, optionally adding generative video-only refinement while preserving the source audio. Refinement can invent identity details, logos, and text. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Unload MLX Runtime | Release process-local LTX 2.5 state. | LTX 2.5 — Core | Supported |
| Canny Preprocessor (MLX) | Create temporally aligned Canny control frames with MLX. The defaults match ComfyUI's current normalized-threshold Canny contract. | MLX preprocessors — Edges | Experimental |
| Video Depth Model Loader (MLX) | Select a converted Apache-2.0 Video Depth Anything Small checkpoint. This node does not load weights. | MLX preprocessors — Depth | Experimental |
| Video Depth Preprocessor (MLX) | Estimate temporally consistent relative depth with Video Depth Anything Small on MLX. The default unloads the model after preprocessing. | MLX preprocessors — Depth | Experimental |
| DWPose Model Loader (MLX) | Select converted YOLOX-L and DWPose whole-body MLX bundles without loading them. | MLX preprocessors — Pose | Experimental |
| DWPose Preprocessor (MLX) | Estimate whole-body pose with MLX YOLOX-L and DWPose. The default includes body, face, and hands, then unloads both models. | MLX preprocessors — Pose | Experimental |
| TEED Model Loader (MLX) | Select a converted MIT-licensed TEED checkpoint without loading it. | MLX preprocessors — Edges | Experimental |
| TEED Soft-Edge Preprocessor (MLX) | Create learned soft-edge guides with the tiny TEED model on MLX. The default unloads the model after preprocessing. | MLX preprocessors — Edges | Experimental |
| Fast Depth Model Loader (MLX) | Select the standard Apache-2.0 Depth Anything V2 Small safetensors checkpoint. Weights are loaded directly into MLX only when preprocessing runs. | MLX preprocessors — Depth | Experimental |
| Fast Depth Preprocessor (MLX) | Estimate fast per-frame relative depth with Depth Anything V2 Small on MLX. Use Video Depth Anything when maximum temporal consistency matters. | MLX preprocessors — Depth | Experimental |
| Depth to Normal Map (MLX) | Convert a relative-depth IMAGE batch into standard RGB +Z-blue surface normals on MLX. Strength compensates for the small slopes in normalized depth. The node is weightless and preserves the input frame count and dimensions. | MLX preprocessors — Normals | Experimental |
| Line Art Model Loader (MLX) | Select a converted realistic fine or coarse line-art checkpoint without loading it. | MLX preprocessors — Line art | Experimental |
| Realistic Line Art Preprocessor (MLX) | Extract realistic fine or coarse line art with a compact MLX residual generator. The default matches ComfyUI's conventional white-line guide on black. | MLX preprocessors — Line art | Experimental |
| Optical Flow Motion Tracks | Extract reliable sparse trajectories from an IMAGE batch with forward/backward optical flow, then render the LTX Motion Track training-color guide. | MLX preprocessors — Motion | Supported |
| Motion Track Guide (MLX) | Render sparse colored point trajectories into the guide-video representation expected by the LTX Motion Track IC-LoRA. This node uses MLX and does not require a checkpoint. | MLX preprocessors — Motion | Experimental |
| LTX 2.5 CrossView Camera Orbit | Build a multi-point CrossView camera path with a stock-Comfy visual sphere and frame timeline. The path preview follows the selected interpolation, and the sphere view can rotate independently without changing camera poses. Numeric widgets and camera_keyframes remain authoritative for API workflows. | LTX 2.5 — Camera preprocessing | Experimental |
| LTX 2.5 CrossView Warp | Build the full-resolution magenta-hole camera warp expected by the CrossView Warp IC-LoRA. Connect the result and the same source video to CrossView Dual Reference Guide. | LTX 2.5 — Camera preprocessing | Experimental |
| Unload MLX Preprocessors | Release weighted MLX preprocessor state without changing H3 or LTX residency. | MLX preprocessors — Lifecycle | Experimental |
| H3 Fun ControlNet-Union Loader (MLX) | Select the 5-block MiniMax-H3 Fun ControlNet-Union branch. Weights remain deferred until sampling. One checkpoint supports Canny, depth, HED, MLSD, and pose guides. | H3 — ControlNet | Experimental |
| H3 Encode Fun Control Video (MLX) | Fit a preprocessed control video to the H3 canvas and encode it with the selected H3 video VAE. Short guides hold their final frame; long guides are trimmed. | H3 — ControlNet | Experimental |
| CorridorKey Model Loader (MLX) | Select a separately installed CorridorKey MLX checkpoint and a measured speed, quality, or low-memory profile. Weights load only when the keyer executes. | CorridorKey — Keying | Experimental |
| CorridorKey Auto Chroma Hint | Create a coarse green-screen alpha hint from border chromaticity. The default erosion and blur match the hint style that CorridorKey expects. | CorridorKey — Keying | Experimental |
| CorridorKey Mask Refine | Shrink or grow, blur, fill, and clean any standard ComfyUI MASK before CorridorKey. Use this node with SAM, Florence-derived, Impact Pack, or manual masks. | CorridorKey — Keying | Experimental |
| CorridorKey Keyer (MLX) | Run the optional CorridorKey MLX engine on an IMAGE batch and coarse MASK. Return straight foreground, alpha, premultiplied color, preview, and provenance metadata. | CorridorKey — Keying | Experimental |
| CorridorKey Composite | Composite CorridorKey foreground and matte outputs over a matching ComfyUI IMAGE batch. | CorridorKey — Keying | Experimental |
| CorridorKey Unload | Release the process-local CorridorKey MLX engine and allocator cache. | CorridorKey — Keying | Experimental |
| Florence-2 Model Loader (MLX) | Select a local MLX Florence-2 bundle for text-guided auto masking. The node validates the bundle but does not load weights until detection executes. | MLX preprocessors — Segmentation | Experimental |
| Florence-2 Text Auto Mask (MLX) | Ground a text description with Florence-2 on sparse video frames, interpolate its location, and emit either a guided subject silhouette or a fast rectangular mask. | MLX preprocessors — Segmentation | Experimental |
| Unload Florence-2 (MLX) | Release Florence-2 MLX state without changing CorridorKey, H3, or LTX state. | MLX preprocessors — Segmentation | Experimental |
| H3 Motion Fidelity Settings (Experimental) | Experimental adaptive or uniform temporal expansion, partial-denoise strength, optional independent refinement evaluations, seed and frame budget. | H3 — Sampling and acceleration | Experimental |
| H3 Motion Fidelity Refine (Experimental) | Analyze or refine a native 24 fps H3 movie in an isolated process; retain original audio and recover original frame timing. Optional standard full-schedule repair LoRA. | H3 — Output | Experimental |
| Draw Things Connection | Configure route, endpoint, helper, and an environment-variable credential reference; no connection occurs here. | Draw Things — Remote generation | Experimental |
| Draw Things Discover | Explicitly refresh the server catalog without generating media; INPUT_TYPES never contacts the endpoint. | Draw Things — Remote generation | Experimental |
| Draw Things Request | Build a free-only canonical request; model IDs must be copied exactly from explicit discovery. | Draw Things — Remote generation | Experimental |
| Draw Things Estimate | Estimate CU and refresh eligibility only. Use the estimate-only workflow to avoid generation. | Draw Things — Remote generation | Experimental |
| Draw Things GenerateImage | Generate an image through the shared adapter and load only the returned image into an IMAGE tensor. | Draw Things — Remote generation | Experimental |
| Draw Things GenerateVideo | Generate video and finish it to a file; frames stay on disk for low-memory graphs. | Draw Things — Remote generation | Experimental |
<!-- END GENERATED NODE CATALOG -->

## Troubleshooting

### Studio Prepare Clip rejects the official LTX 2.5 distilled LoRA

Update the renderer if the error mentions `to_gate_logits` as an incompatible target. The
official rank-450 adapter includes attention-gate projections; older validation rejected these
valid weights. Keep the existing model files and prepare the clip again. For an app-managed
runtime, use the updated app's **Set Up Managed Renderer** to refresh its source snapshot.
Studio now shows the renderer's specific preparation or generation error; **Show Log** retains
the full traceback.

### A workflow opens with shifted widget values

Restart ComfyUI and reload the current workflow. Do not repair shifted fields manually. The saved
graph may use an older node contract.

### A component path exists but preflight cannot find it

Confirm that the value is relative to a configured ComfyUI model root. Confirm shared roots in
`extra_model_paths.yaml`. Do not paste an absolute path into a shipped workflow.

### FFmpeg is visible in a terminal but not in ComfyUI

Leave the node override empty to use the ComfyUI process environment or a compatible packaged
encoder. Set `WEETODD_FFMPEG` to an absolute executable path only in the local process environment.
Do not save that path in a workflow.

### Ref2VA is unexpectedly slow

Match reference media to the target output area. High-resolution reference video creates more
persistent tokens. Keep full density for the first quality test, then compare the experimental
half-density option.

### ComfyUI runs out of memory

Start a fresh process, choose the speed profile, use paged Qwen3-VL and q8-extended transformer
components, keep caches disabled, and retain staged unloading. Lower resolution before lowering
reference density.

## Development validation

Run these checks from the repository root using its compatible arm64 development environment:

```bash
python scripts/audit_workflow_catalog.py --project .
python scripts/update_readme_node_catalog.py --check
python scripts/preflight_python_environment.py --project . --python python --require-architecture arm64
python scripts/preflight_h3_workflow.py --project . --all-api
python scripts/lint_docs.py
python -m compileall -q src __init__.py
python -m pytest -q tests/test_nodes.py tests/test_runtime.py tests/test_readme.py tests/test_workflows.py
ruff check src/wee_todd_nodes tests
```

`tests/test_readme.py` and `tests/test_workflows.py` are required by the README/workflow commit gate.
For Studio changes, also run `swift test --package-path studio` and
`python -m pytest -q tests/test_studio_bridge.py tests/test_studio_packaging.py tests/test_studio_lora.py`, then build the app
using the [Studio instructions](studio/README.md). The build verifies its ad-hoc signature before
replacing the previous bundle.

Local `knowledge/`, research reports, and `.agents/` skills are intentionally untracked. If the local
knowledge bundle is installed, validate it separately with `python scripts/validate_okf.py knowledge`;
a fresh clone does not include it. See [AGENTS.md](AGENTS.md) for the local commit-gate procedure.
Full checkpoint parity and real generation tests are optional and expensive.

## License and status

WeeTodd source code is Apache-2.0. Model checkpoints and adapters keep their original licenses and
terms. The project is experimental and pre-release. Do not commit checkpoints, generated media,
caches, credentials, tokens, machine-specific paths, or private information.
