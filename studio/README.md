# WeeTodd Studio

A native Swift macOS editor around the shared WeeTodd MLX renderer. The app lives in this
repository; it does not need a running ComfyUI server. This is an initial development build,
not a notarized consumer release.

## Build and open

Run these commands from the repository root on Apple Silicon with Xcode 26 or newer.
The MetalFX frame interpolator uses the macOS 26 SDK:

```bash
python3 scripts/build_studio_app.py --configuration release
open "studio/.build/WeeTodd Studio.app"
```

The GUI supports macOS 14 and newer. Actual model/runtime requirements can require newer macOS.
MetalFX frame interpolation additionally requires macOS 26 and a supporting GPU. The helper checks
hardware support. The build script makes an ad-hoc signed application containing the Swift GUI,
`WeeToddCLI`, `StudioMetal`, the renderer source, and the runtime dependency lock. Building the
app does not install Python dependencies or copy models. All packaging/runtime preflight code is
tracked; no local agent skills are required. Packaging starts with a fresh bundle, verifies its
signature, and replaces the previous app only after success, removing obsolete bundled source files.
The app icon is bundled from `studio/Resources/AppIcon.icns`; `AppIcon.png` retains the supplied
artwork with only the outer white corners removed. The icon contains transparent standard and
Retina sizes.

In **Studio Settings**, choose **Set Up Managed Renderer** to download private Python 3.12.13 and
install pinned, hash-verified native dependencies. The installer verifies arm64, the project Python
requirement, package consistency, and Metal availability before activating the new runtime.
Each installation gets a new directory; prior runtimes remain available. The developer environment
and other applications' environments are preserved. There is no Linux virtual machine.
The bootstrap uses a pinned, checksum-verified uv release from Astral.

Advanced users can connect an existing compatible WeeTodd repository and Python environment.
Use guided model setup below, or import existing `weetodd-headless-v2` recipes to identify model
component sets. New clips select an engine, task and preset; automatic recipe selection inspects
compatible recipe contents. FFmpeg/FFprobe and
optional RIFE remain separately configured tools; a retail installer must package these tools
with their licenses and complete clean-Mac qualification.

Runtime settings, autosave, global assets, recipes, previews and jobs live under
`~/Library/Application Support/WeeTodd Studio`. User media and model weights stay in their existing
locations. `WEETODD_STUDIO_DATA` selects a separate data directory for isolated development tests.

## LTX 2.3 single-pass text-to-video

In Model setup, select **LTX 2.3 · Text to video · Single-pass distilled 1.1**. Import the existing
MLX distilled 1.1 bundle and a local Gemma 3 12B encoder, create the recipe, then choose **Use Recipe
for Selected Clip** on an LTX 2.3 T2V clip. The recipe uses eight evaluations, Shift 5 and staged
unloading with streamed weights. The inspector exposes editable Steps and Shift, fixed CFG 1,
and no refinement pass. Existing two-stage recipes keep their behavior.

This route creates synchronized video and audio directly at the clip dimensions. It currently
supports T2V; use the existing recipes for image, reference, audio conditioning and extension.
Exported clips and movie jobs preserve the same mode and settings. No model copies are needed.
See the [single-pass documentation](../README.md#ltx-23-single-pass-distilled-11) for model layout,
ComfyUI usage and measured validation. Physical 36 GB qualification remains open.

## Reuse local Draw Things H3 models

In **Studio Settings → Model setup**, select **MiniMax H3 · Draw Things models · Text to video**.
Scan your Draw Things model folder or import these existing files individually:

- H3 transformer: the supported `minimax_h3_i8x.ckpt` layout.
- Qwen encoder: the supported `qwen_3_vl_32b_50_i8x.ckpt` layout.
- Video/audio VAE: the supported `minimax_h3_vae_f16.ckpt` layout.
- H3 tokenizer folder: reuse one already installed, or use the small tokenizer download.

Names are examples; setup checks the tensor inventory, shapes and supported codecs. **Create Recipe**
creates metadata references next to the recipe. Choose an **H3** clip, **Text to video**, and this
recipe. This uses the native renderer and your local DT weight files; no DT connection or CU is
involved. A **Draw Things** clip still selects the separate gRPC/cloud route.

Original weights remain read-only and are decoded only in memory. No converted checkpoint or
persistent weight cache is created. Keep the generated `h3-dt-components-*` directory beside its
recipe. Model files and tokenizer remain in their original locations and must be available on the
machine executing an exported job. Existing H3 ComfyUI Component Loader fields can use the recipe's
component paths, including the original transformer file and generated metadata directories;
choose task `t2va`. The combined legacy pipeline loader is not this setup route.

This first release is experimental and limited to text-to-video with generated audio. Other DT
model families, first/last/reference frames and audio inputs need separate adapters/qualification.
LoRAs, resident execution and accelerators other than verified MPP projections have not been
qualified with DT weights. Use the checkpoint's default paging. Performance
and memory differ from the DT app; the VAEs currently execute in FP32. The transformer's packed
weights now decode and reorder on Metal automatically, retaining the previous native weight values.
Existing DT-weight recipes use this improvement without reimporting models. The matched 512×512,
124-frame, 19-evaluation run fell from 15:25 to 12:05 with a byte-identical movie, and process peak
fell from 18.07 to 16.35 GiB on M3 Ultra/256 GiB. Transformer MLX peak rose by 0.37 GiB; overall
MLX peak was unchanged. A physical 36 GB hardware test remains outstanding.

New DT recipes with normal working memory use **Automatic** projections. The renderer checks GPU
and macOS support, compares each new projection shape with standard MLX on first use and falls
back if verification fails. Lower-memory setup retains standard MLX. Existing recipes remain
unchanged; choose **Automatic** in the clip's projection control to opt in, or **MLX** to disable it.
For exported recipes, the equivalent field is `config.projection_backend` (`auto` or `mlx`);
ComfyUI offers it on **H3 Generation Config**. This does not enable resident loading or alter steps,
precision, memory chunk sizes or conditioning.

The matched normal-memory M3 Ultra run with Automatic projections completed in **10:51**, versus
**12:05** with standard MLX: **10.1% less total time**, with byte-identical video/audio. Process peak
remained **16.35 GiB** and MLX peaks were unchanged. This qualifies the measured configuration,
not lower-memory settings or other hardware.

Existing DT recipes also receive bounded read-only transformer payload mapping and compact video
decoder storage automatically. The video decoder keeps the original F16 weights, uses FP32
activations, and completes one block at a time to bound temporary memory. It does not load the
unused encoder. No new model download, conversion, permanent weight cache or settings change is
required. Other native model loaders and DT server/cloud clips use their existing paths.

The matched 512×512/124-frame/19-evaluation M3 Ultra (256 GiB) render now peaks at **9.48 GiB
process footprint**, down from 16.35 GiB, with a byte-identical video/audio MP4. Total time was
**10:50 versus 10:51**, effectively unchanged; this pass improves memory rather than overall
speed. Peak MLX allocation fell from 12.53 to 6.58 GiB. These measurements use fresh renderer
processes and fresh prompt caches, without clearing OS file caches, and do not qualify 36 GB Macs.

Cached-modulation DT blocks now prepare as one GPU batch, avoiding a wait after each tensor while
preserving the same weight values and sampler. Fixed weights and initial modulation preparation
remain eager. This is part of the shared renderer, requires no new model download or settings
change, and retains no weights after a block is released. Batch preparation timing is reported
separately from tensor decode submission timing.

The matched M3 Ultra run with batching completed in **10:13 versus 10:50**, with byte-identical
video/audio, unchanged MLX peaks, and effectively unchanged process footprint (**9.51 versus
9.48 GiB**). Block preparation fell from 109.4 to 91.7 seconds. The observed 5.7% total-time
reduction is from one desktop comparison; unchanged block computation also ran faster, so batching
does not account for every second saved. Physical 36 GB qualification remains open.

Normal-memory DT recipes with **Automatic** projections can now prepare one decoded transformer
block ahead while the current block computes. The renderer enables this only for its resolved
MPP path, one-block paging, cached modulation and no retained-page cache. It uses about 0.72 GiB
for the additional block plus temporary preparation buffers. **MLX** projections or **Lower memory**
disable it. It creates no converted model files and releases the worker and prepared weights on
unload, failure or cancellation. Exported headless jobs and composable ComfyUI nodes use the same
behavior. Render reports include lookahead hits, weight bytes and waiting time; overlapping
preparation/compute timings cannot be summed as separate elapsed stages.

The integrated 512×512/124-frame/19-evaluation M3 Ultra run completed in **9:07 versus 10:05**
(9.7% less time), with a byte-identical video/audio MP4 and all weighted stages unloaded. Process
footprint was **9.48 GiB** and overall MLX peak remained **6.58 GiB**; transformer MLX peak rose
from 5.31 to 5.91 GiB. The earlier same-workload prototype took 9:31, so elapsed gains vary.
This is desktop evidence for the tested recipe, not physical-36-GB or DT sampler parity.

The same setup is available from the CLI, without writing a recipe by hand:

```bash
python scripts/setup_models.py download h3-dt-tokenizer --destination /path/to/library
python scripts/setup_models.py create h3-draw-things-text \
  --component dt_transformer=/path/to/DrawThings/minimax_h3_i8x.ckpt \
  --component dt_qwen=/path/to/DrawThings/qwen_3_vl_32b_50_i8x.ckpt \
  --component dt_vae=/path/to/DrawThings/minimax_h3_vae_f16.ckpt \
  --component tokenizer=/path/to/tokenizer \
  --profiles-directory /path/to/recipes
python scripts/render_headless.py --recipe /path/to/recipes/created-recipe.json \
  --output-directory /path/to/output
```

Use the tokenizer folder returned by the download/scan and the recipe filename returned by setup.
The model files stay in the DT model store. The tokenizer download retains its pinned source terms.

## Guided model setup

For remote generation, see [Draw Things](#draw-things--experimental). Local model recipes below
configure the native MLX engines.

1. Open **Studio Settings → Model setup**. Built-in presets appear independently of installed recipe
   count once the renderer is configured. Choose H3 text/image/reference or LTX 2.3/2.5 text/image.
2. Choose **Set Up… → Use Existing Models**, select model folders (including an existing ComfyUI
   `models` folder), then scan. Inspection reads bounded headers and manifests, never model tensors.
   A single candidate is selected automatically; multiple candidates require your choice. Missing
   components have **Import…** and **Download…** controls beside their labels. Import links a local
   file/folder without copying it. Download selects a compatible package and opens its size, contents,
   source terms and destination for review; it does not immediately start the transfer. Components
   without a catalog download can still be imported. Validation explains incompatible architecture
   or task support.
3. Use **Automatic** to select a lower-memory policy on Macs with 64 GB or less, **Lower Memory** to
   request supported memory-saving settings, or **Custom** to retain the preset policy for later
   advanced adjustment. Memory information is advisory and does not promise fit, allocate RAM, or
   enforce a hard limit. Clip size/duration and other resident applications still matter.
4. **Create Recipe** runs the shared component/configuration preflight and writes a new recipe.
   Image/reference presets still need media attached to a clip before full render preflight can pass.
   Existing recipes and model files are preserved. Use the new recipe for a compatible selected clip,
   or choose it in the clip inspector. **Set Up Models…** is also available from missing-model actions.

**Download or prepare a model** shows compatible catalog items with source terms, download size and
required space before an explicit download. Prefer the
[H3 Q8 vision encoder](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged) or
[LTX 2.5 distilled Q8 package](https://huggingface.co/Vayden/LTX-2.5-MLX-Q8-Paged) marked
**Preconverted (Recommended)**. Source conversion remains an alternative. LTX 2.5 source preparation
downloads the five official distilled components and
converts transformer/Gemma to paged Q8. H3 encoder preparation downloads just the compact Q8 encoder,
its support files and full Qwen architecture config, then retains the vision page.

For H3, choose four downloads in the same library: the **text/image or reference Q8 transformer**,
its matching **support files**, the **Q8 vision encoder**, and the **Q8 video VAE**. The support files
include the official task manifest, audio VAE, tokenizer and processor. Downloads specific to another
task are hidden; the encoder and video VAE can be reused across tasks. After installation, scan the
downloaded folders, choose the components and create the recipe. No H3 weight conversion is required
for these prepared sets. Optional Turbo/control LoRAs remain separate.

Selected existing roots are checked for exact source checksums before downloading replacements.
Same-volume source files are reused through links; cross-volume source references remain linked,
with copies only where final component packaging needs them. Existing converted components can be
selected directly without conversion. Partial downloads resume after cancellation or network failure.
Sources are retained in the chosen library’s `.weetodd-downloads` directory for reuse. Only verified,
fully prepared output is installed; source terms/attribution remain with the output. Gated sources
require accepting their upstream access terms. In **Hugging Face access**, open the
token settings link, paste a read token and save it to macOS Keychain. The field clears after saving;
the token is passed only in the download process environment and never written to recipes or setup logs.
Remove it with the same controls. An existing CLI login (`hf auth login`) remains supported when no
Studio token is saved.
The setup log explains authentication, disk-space, checksum and conversion failures.

For CLI users, `python scripts/setup_models.py --help` exposes the same catalog, scans, recipe creation
and downloads. For example:

```bash
python scripts/setup_models.py download h3-qwen-q8-vision-preconverted \
  --destination /path/to/shared-models --existing-root /path/to/ComfyUI/models
python scripts/setup_models.py download ltx25-distilled-q8-preconverted \
  --destination /path/to/shared-models --existing-root /path/to/ComfyUI/models
```

Run the command for the model you need. These downloads skip local conversion. The catalog also
offers `h3-qwen-q8-vision` and `ltx25-distilled-q8` to convert from pinned source files.
After a download, scan the returned directory, choose the components, and create the recipe. ComfyUI
users can select those same paths in existing loaders; setup never downloads or converts during a graph.
The [headless example/schema](../examples/headless/README.md) remains available for direct JSON users.

### Model setup troubleshooting

| Symptom | Next action |
| --- | --- |
| LTX clip shows **0 recipes** | Choose its built-in preset in **Studio Settings → Model setup**, scan existing components or download the prepared package, then **Create Recipe** and **Use Recipe for Selected Clip**. Downloading weights alone does not create a recipe. |
| **Prepare Clip** rejects the official LTX 2.5 distilled LoRA with `to_gate_logits` targets | Update the renderer and prepare again. The validator now accepts the official attention-gate weights. Existing models and prompts can be reused. For an app-managed runtime, refresh it with **Set Up Managed Renderer** from the updated app. Preparation and generation alerts now show the renderer's specific error; **Show Log** keeps the traceback. |
| Built-in presets or preconverted downloads are missing | Use the updated app and renderer. Rebuilding the app updates its bundled source; an existing managed runtime keeps its old source snapshot. Choose **Set Up Managed Renderer** to install the updated snapshot, or connect an updated repository with a compatible Python environment. Existing media and models can be reused. |
| Download reports denied access / 401 / 403 | Open **Model source**, accept that repository's access terms with your Hugging Face account, and save a read token for the same account under **Hugging Face access**. Review the setup log for the exact failure. |
| Download was interrupted | Repeat the same download with the same destination. Verified files are reused and partial files resume. A completed package should instead be opened through **Use Existing Models**. |
| Setup reports insufficient space | Choose a library on a drive with enough free space. Preconverted downloads skip local quantization and its intermediate storage; the displayed download size is not a RAM estimate. |
| Scan finds several transformer candidates | Select the distilled transformer for the LTX 2.5 distilled preset. Common architecture headers alone cannot prove Dev/distilled training identity; the curated preconverted package removes that ambiguity. |
| H3 reference clip rejects a text-only encoder | Select the new **H3 Q8 vision encoder · Preconverted (Recommended)** package. The older v1 text-only export remains useful for T2VA but lacks the vision weights needed for image/reference conditioning. |

Setup creates component recipes, while **Prepare clip** validates the final clip's media and settings.
If a reference/image clip still needs attention, attach the required media and follow its Actions entry.
For a complete CLI walkthrough, see [download and create an LTX 2.5 recipe](../examples/headless/README.md#download-and-create-an-ltx-25-recipe).

## Clip generation controls

Choose **Engine → Task → Preset** in the clip inspector. Tasks come from the installed compatible
model recipes. Image to video requires a first image; First + last frame requires both endpoints.
Changing tasks preserves attachments and names any conflicting or missing input. A reference-only
H3 recipe cannot make a text-only clip appear ready. Model filenames remain available in Advanced
for deliberate custom selection.

The inspector displays sampling controls and compatible LoRAs/groups with editable strengths.
Ordinary native H3 Euler **Steps** means actual evaluations: 19 evaluations correspond to 20 stored
schedule points. Specialized or fixed schedules explain their restrictions. LTX stage-one and
refinement controls are separate; fixed distilled schedules remain fixed. Native H3 uses distilled
guidance and fixed video/audio shifts, so CFG and Shift are visibly unavailable. Supported native
LTX CFG and Draw Things configuration values remain editable. Unsupported submitted overrides fail
validation instead of being ignored.

Edited preset controls show **Modified**; **Reset** removes those overrides. Existing clips open as
**Custom**, preserving their imported recipe settings. Choosing a preset opts into the new explicit
selection. **Generate** prepares and validates the clip automatically. The separate preparation and
prompt/settings review remain available. Clip/movie headless export uses the same resolved recipe.

### Acceleration settings

App-level H3 acceleration preferences are separate from creative sampling settings. Automatic
projections use the existing hardware-qualified backend and numerical fallback checks; **MLX**
explicitly uses the standard backend. Automatic memory selection uses the lower-memory policy on
Macs with 64 GiB or less, and retains the recipe policy on larger Macs pending further qualification.
The detected RAM figure is advisory, not available RAM or a hard allocation limit.

Each explicit clip can override those preferences. **Paged** selects lower-memory execution with
the checkpoint's existing layout; it does not convert a resident checkpoint into pages.
**Paged · larger workspace** keeps that layout but uses normal working buffers, allowing a separate
comparison of working memory and weight residency. This option is not qualified for 36 GB hardware.
**Resident**
retains all transformer blocks during sampling, requires normal memory mode and a zero page-cache
budget, and is intended only for ample-memory systems. The transformer still unloads before VAE
decoding, including failure/cancellation cleanup. This is separate from keeping weights warm across
jobs. Existing Custom clips do not silently inherit newly changed app preferences.

Balanced and Speed currently preserve the recipe's sampling schedule. Speed is not a promise of
fewer evaluations or a measured speedup; experimental approximations are not silently enabled.
Low memory selects the supported lower-memory policy. Dimensions, duration and attached media
remain explicit clip choices. Hardware-specific speed recommendations require matched measurements.

### Matched H3 execution measurements

On an M3 Ultra with 256 GiB unified memory (2026-09-10), a saved 512×512, 124-frame,
24 FPS H3 T2V recipe used 19 Euler evaluations, seed 42, Q8 paged FL2VA transformer,
paged Qwen and Q8 video VAE. Prompt, models, quantization, dimensions and schedule were fixed.
Each run used a fresh renderer process; the conditioning cache could reuse encoded text, so compare
sampling separately from whole-job time. No other generation ran concurrently.

| Execution | Total | Sampling | Video decode | Process RSS peak | MLX stage peak |
| --- | --- | --- | --- | --- | --- |
| Paged · lower memory, MLX (saved baseline) | 1065.8 s | 1016.1 s | 35.2 s | 5.69 GiB | 6.64 GiB |
| Paged · lower memory, Automatic | 1046.4 s | 1010.0 s | 34.6 s | 5.62 GiB | 6.64 GiB |
| Paged · larger workspace, MLX | 574.7 s | 535.5 s | 37.9 s | 6.34 GiB | 6.98 GiB |
| Resident, MLX | 535.1 s | 493.8 s | 38.5 s | 31.18 GiB | 32.32 GiB |
| Resident, Automatic | 558.6 s | 514.9 s | 41.0 s | 31.18 GiB | 32.32 GiB |

All five movies were byte-identical, with 124 video frames, stereo 32 kHz audio, 8.3 ms
A/V drift and every weighted runtime released. These are individual runs, not a statistical
backend ranking. Automatic projections passed the hardware/numerical checks but showed no useful
speed gain here. The resident policy roughly halved sampling time, with a substantial memory cost.
The larger-workspace paged run achieved most of that gain with only a small measured peak increase,
showing that chunking/working-buffer policy accounts for much of this baseline's slowdown.
Full residency saved another 39.6 s overall while adding about 25 GiB to the MLX stage peak.
Try **Paged · larger workspace** first for this recipe before full residency. Normal memory mode
also selects a larger decode batch; decode did not improve in these runs.
Process RSS and the largest instrumented MLX stage are distinct counters, not additive RAM totals.
This is evidence for this recipe on this Mac, not qualification for a 36 GB Mac or every task.

## Progress, measurements, and H3 page retention

Native renders now send live stage/evaluation updates to the status bar. Sampling shows its own
progress, followed by decoding/publication; the bar is not an estimated whole-job percentage.
Elapsed time and the age of the last renderer output remain visible during long steps. A quiet
interval alone does not mean the renderer has stalled. Cancellation remains attached to the job.

New render versions retain measured render time and available memory/timing statistics. Select a
version to see its summary or expand **Versions** to compare takes. Older projects still open;
historical versions without saved statistics show no invented measurements. **Process peak** is
renderer RSS, while **Instrumented stages** and **MLX generation** describe different MLX counter
scopes. Do not add those peaks or treat them as total system RAM. LTX **Pre-decode pipeline** time
includes encoding, model loading, sampling and latent upscaling; H3 reports transformer sampling.

For H3, **Advanced generation → H3 page cache** offers Recipe default, Off, or 4/8/12/16 GB.
The recipe setting is `config.paging_cache_gb` (0–16 decimal GB; default 0). This is extra retained
raw transformer weight memory, not a limit on total generation memory or a promise of fit.
Start with Off versus 4 GB using identical prompt, seed, dimensions, schedule and storage. Compare
the measured process/MLX peaks as well as wall time; return to Off if memory pressure increases.
The cache pins a bounded subset of quantized pages between evaluations and releases it after each
sampling run, including cancellation/failure. It requires a paged transformer and cannot be
combined with full block residency. A page larger than the remaining budget is bypassed.

ComfyUI exposes the same setting through **H3 Paging Settings (Experimental)** between Generation
Config and the composable H3 Sampler. Headless movie/clip jobs carry the selected value. H3
`result.json` includes paging counters, retained peak/budget bytes, and loading/setup/compute timing.
Cache counters cover the configured run; other pager counters identify their executor-lifetime
scope. File-load calls are not measurements of physical disk reads. The cache avoids some reloads
but does not remove module construction or attention computation. Tiny FP32/Q8/LoRA tests establish
parity and cache behavior; a full-size speedup on a 36 GB Mac has not yet been measured.

Head and FFN chunk controls now reach newly loaded paged blocks. Previous automatic/lower-memory
comparisons could therefore exercise effectively identical transformer settings. Rebaseline after
updating: smaller chunks can reduce workspace but add dispatch overhead, so the fix alone is not
a speedup claim. Query chunking and the retained-page budget are separate controls.

## LTX 2.5 images, controls, and references

For ordinary image-to-video, select the **LTX 2.5 Image to video** recipe, import/select an image,
and choose **Use in clip → First frame · Image to video**. Prepare the clip after attaching it.
The Reference role selects MSR conditioning and requires its separate adapter recipe.

Guided setup now also offers **IC-LoRA control**, **Ingredients reference sheet**, and **MSR image
references**. Import the corresponding dedicated adapter in addition to the existing model
components. Setup checks its tensor headers and compatibility; a style LoRA cannot substitute.
These recipes use the existing distilled full-resolution single-stage renderer. They are separate
from the basic image preset and do not require a spatial upscaler for that single-stage route.

- **Control:** attach a preprocessed guide video as Control and choose its matching guide type
  (such as depth, pose, motion tracks or crossview). The recipe's adapter must support that type.
- **Ingredients:** attach one image using **Ingredients reference sheet · IC-LoRA**, use at least
  121 output frames, and describe the sheet and intended scene in the prompt.
- **MSR:** attach one to five images as **MSR reference · dedicated adapter**, describe each, and
  choose subject/object/clothing/background, priority, reference frames, sizing and attention
  strength in Conditioning. Only one background is allowed. Recipe default preserves matching-image
  options from an imported recipe; removed attachments never leave hidden recipe media active.

These routes reuse shared renderer validation and remain subject to adapter and memory constraints.
First-frame success on 36 GB does not establish MSR/control memory fit or reference quality.

## H3 reference clips with paged Q8 models

Import a recipe produced by `scripts/prepare_h3_reference_recipe.py`, select **H3 Reference Q8
Paged** for the clip, and attach images with the **Reference** role. The shared renderer uses genuine
Ref2VA Q8 transformer pages and vision-capable Qwen v2 pages. Start with one image, five seconds,
640×384 and the recipe's 19 dense evaluations; add a second reference only after checking memory.
The existing text-only Qwen page export cannot encode reference images.

See the [model preparation commands](../README.md#experimental-h3-reference-paging). Choose the
[preconverted Q8 vision encoder](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged)
in guided setup, together with **H3 reference transformer Q8**, **H3 reference support files** and
**H3 video VAE Q8**. Alternatively, prepare your own files with the bounded-memory conversion
commands. Select the genuine Ref2VA transformer; the text/image transformer is a different model.
Clip/movie headless
export preserves this recipe so Studio can be closed during generation. One-image 640×384 generation measured a 21.70GB complete Comfy process peak on an M3 Ultra
with 256 GiB. The headless output was byte-identical and peaked at 21.26GB.
A 36GB physical-device maximum is not yet established; the header-based estimate omits reference-dependent workspace.

## Editing

- Clip inspector at upper left, inherited movie settings below; central viewport and timeline;
  collapsible Global, Project, and selected Clip asset stores at right.
- System, Light, and Dark appearance. Section colors in the design wireframe are not used.
- One main video track, titles, and additional named audio tracks with mute, solo, and source-audio
  replacement. Set each region's start, source in, duration, volume and fades in its inspector.
- H3, LTX 2.3 and LTX 2.5 generated clips, imported movies/stills, and image sequences. Sequence import
  uses the movie frame rate, natural filename order, linked original frames and a ProRes editing proxy.
- Drop a movie asset on the timeline to create a clip and a source reference in its Clip Assets.
  Generated clips expose **FF** (First Frame / I2V) and **LF** (Last Frame) slots at their timeline
  endpoints when supported by the model and recipe. Drop one image from any asset store or Finder
  onto a slot, or click an empty slot to import. Filled slots show thumbnails; another drop replaces
  that endpoint, and the context menu removes it. Assignments are undoable, link the original into
  Clip Assets, update the inspector/task, and invalidate prepared generation. Draw Things H3 offers
  both slots (last requires first); Draw Things LTX currently offers first only. Movie/still clips
  have no generation slots. Drop into a native generated clip's body for an interior keyframe.
  Drag clip cards to reorder them.
- Split rendered/imported clips, duplicate, edit source in/duration, and create extension clips.
  Before-extension is offered for LTX 2.3. Extension uses the referenced source movie as model context;
  trimmed timeline boundaries are not currently extracted into a new extension context automatically.
- **Insert Bridge to Next Clip** extracts the selected clip's last visible frame and the next clip's
  first visible frame. It creates an LTX 2.5 shot with linked first/last anchors and opens its prompt.
- The prompt editor fills the window. **Prepare clip** validates model paths, task support and inputs,
  then displays the exact resolved prompt. **Generate clip** executes that prepared recipe.
  H3 reference/audio/extension tasks currently require their complete native six-section prompt.
- Automatic task selection and the selected recipe's compatible adapters preserve the existing
  backend contracts. Controls expect preprocessed guide media. Missing or incompatible inputs fail;
  the editor does not guess an arbitrary ControlNet, LoRA, pose extractor or reference description.
- Generated versions remain in Clip Assets. Undo/redo and autosave protect edits. Collect Media writes
  a separate project with relative media references, including used Global assets. It preserves shared
  model/LoRA paths; it does not package weights or rewrite model recipes for another machine.
- Seed typing is grouped into one Undo step per editing session. Command-Z and Shift-Command-Z
  restore the complete previous/next seed, including while the field retains focus. Unchanged field
  writes do not add Undo steps or clear Redo.

**Preview movie** builds a reduced-resolution movie containing the actual transitions, titles,
and active audio tracks. Clip preview is immediate; movie preview is rebuilt after edits. Missing
renders must be generated first. Movie preview omits interpolation/upscaling, which are applied
in final export. This first build does not provide live multitrack compositing or waveform editing.

## LoRAs and groups

Open **LoRAs & Groups…** in Assets, or **Add / Groups…** in the clip inspector. Import linked
SafeTensors adapters into the reusable Global library. Select the adapter's **Trained model** before
importing; recognized checkpoint metadata takes precedence. Older imports without provenance appear
under **Imports needing a trained model**, where you can classify them explicitly. Filenames are
never used to infer the training model.

| Selected clip / group model | LoRAs offered |
| --- | --- |
| MiniMax H3 | H3 |
| LTX 2.3 | LTX 2.3 |
| LTX 2.5 | LTX 2.3 and LTX 2.5, including mixed groups |
| Movie / Still | None |

Use **New group**, name it, add members from the filtered library, and set each strength. Groups can
be edited or deleted. **Apply to clip** adds an individual LoRA; **Apply group** adds the group's
ordered members. Each clip entry has a slider and exact numeric strength field from 0 to 2.
Remove individual entries or an entire applied group from the inspector. Duplicate file application
is rejected, including overlap between an individual entry and a group.

Groups are reusable templates stored beside Global assets. Application creates independent linked
Clip Assets and copies the strengths and group label into the clip. Editing/deleting the template
does not change existing clips, and clip strength edits do not change the template. Project saves,
autosave, undo/redo, duplication and splitting preserve applied settings. Movie and clip job exports
embed the flattened renderer stack, so headless execution does not need the group library.
Collect Media preserves shared LoRA file paths; it does not copy model weights.

Training versions identify candidates, not a guarantee of compatibility or quality. The shared
renderer still checks actual projection targets, dimensions, scaling and recipe restrictions before
weighted work. Specialized IC/control/reference and schedule adapters remain in their task recipes.
Switching a clip to an incompatible engine retains its settings and marks the clip as needing
attention until incompatible LoRAs are removed. Rebuild the app and use an updated managed renderer
source snapshot when upgrading; existing private runtimes retain their installed source.

Validation covers model filtering, mixed groups, independent clip strengths, serialization, duplicate
rejection, split asset ownership and movie/clip recipe export. The native app was exercised with
small synthetic header fixtures; these checks do not qualify LoRA visual quality or every adapter.

## Status and actions

| Clip color | Meaning |
| --- | --- |
| Green | Generated source matches the current generation settings and linked inputs. |
| Yellow | A generated clip's settings or inputs changed, or a different version was selected. |
| Orange | A new clip has the required basic setup and is ready for full preflight. |
| Red | A generated clip has missing configuration or its last preflight needs attention. |
| Blue | Imported movie/still/sequence clip. Missing files appear in Actions. |

The status area's **Actions** button orders blockers before generation work, then save reminders
and suggestions. Each item opens the relevant clip, prompt, runtime settings, save or job export.
Output settings do not unnecessarily invalidate the source generation; finishing happens during export.

## Movie settings and finishing

Movie dimensions, frame rate, interpolation/upscaling, fit and format are inherited by every clip,
with optional clip overrides. Each clip finishes separately, then assembly conforms it to the movie
canvas and frame rate. Supported outputs are H.264 MP4/MOV, ProRes 422 HQ, or a PNG sequence plus WAV.
Current intermediate clips use H.264; ProRes and PNG output are not end-to-end lossless masters.

Finishing order is resize/upscale, then interpolate, then transitions/titles/audio assembly.
RIFE supports integer 2×/3×/4× interpolation with its configured MLX executable and weights.
MetalFX spatial upscaling uses the native helper. Neither method downloads models silently.

Experimental MetalFX frame interpolation supports 2× and requires genuine per-frame guides at the
processed clip's resolution and timing. Set depth and motion folders in the clip's advanced settings:

- `000001.f32` through the last input-frame index: tightly packed little-endian float32.
- Depth is one channel per pixel; motion is two channels of backward displacement in pixels.
- The depth folder must contain `camera.json` with `nearPlane`, `farPlane`, `fieldOfView` in degrees,
  and Boolean `depthReversed`, describing the actual guide camera.
- Guides must already match trimming, fit, dimensions, and frame rate. The app does not synthesize
  missing depth/motion or imply that ordinary movie files contain these buffers.

## Headless movie and clip jobs

Use **Movie → Export Movie Headless Job** or **Export Clip Headless Job**. A single
`.weetodd-job.json` embeds the edit, resolved generation recipes, runtime paths and finishing settings.
A companion text file describes execution. Files reference local media/models; collect the project
first if media needs to travel. A job still needs its recorded runtime and model locations.

The app contains a CLI launcher that finds the job's native Python automatically:

```bash
"/Applications/WeeTodd Studio.app/Contents/MacOS/WeeToddCLI" \
  --job Movie.weetodd-job.json --output-directory Render --preflight-only
"/Applications/WeeTodd Studio.app/Contents/MacOS/WeeToddCLI" \
  --job Movie.weetodd-job.json --output-directory Render --resume
```

Developers can also use the existing Python client directly:

```bash
.venv/bin/python scripts/render_headless.py \
  --job Movie.weetodd-job.json --output-directory Render --resume
```

The editor can be closed. Jobs preflight all clips and finishing dependencies before generation,
run one generation process at a time, finish clips serially, and assemble the result with titles,
transitions and audio. Each model process exits before the next clip loads. The selected model's
minimum working memory still applies; headless execution cannot make an oversized model fit.
FFmpeg assembly can use additional memory for long edits with many transitions/audio inputs.

`--resume` verifies source observations and completed-artifact hashes, preserving completed renders
and finished clips. Changed jobs, renderer source or inputs require re-export/new output directories.
The output folder is locked against concurrent execution. Interruptions record job state; cancelling
passes through to the active child process. Existing unverified final exports are preserved.
A clip job retains only that clip's intersecting titles/audio, shifted into clip-local time.

## Validation and next release work

The initial validation includes Swift document tests; real FFmpeg movie/transition/title/audio and
sequence/anchor tests; job locking, integrity and resume tests; a real LTX 2.5 generated job and resume;
a fresh private-runtime installation; and real MetalFX spatial/RIFE and guided MetalFX interpolation.
The private-runtime test produced byte-identical generated and assembled MP4s to the development
runtime for the one-second LTX 2.5 fixture. This is a narrow integration check, not universal parity.

Seed Undo/Redo was also checked in the release app with a focused field, after Tab, through the Edit
menu, with multi-digit replacement, and across separate editing sessions. Restoring the generated
seed restores the clip's green status without rerendering.

Before a broad consumer release: expanded model download coverage and tool packaging, sign and
notarize the app, test clean Macs and lower-memory hardware, qualify more conditioning combinations,
and improve live timeline playback. Useful next features are audio waveforms, proxy/cache management,
crash-recovery history, and a render-cost/memory estimate before queuing large movies.

## Motion Fidelity (De-Roping) · experimental H3

Select an H3 clip and open **Motion Fidelity** in the clip inspector. Enable **Use enhanced
motion**, then choose **Analyze** to inspect the expansion plan or **Enhance** to render it.
The original movie remains in Versions and Clip Assets. The enhancement is a separate Clip Asset;
turn the option off to compare the original at the same playhead position. Changing enhancement
settings marks the clip yellow without invalidating its base generation. Actions links to pending
motion work. A stale or missing enhancement blocks direct movie export rather than silently using
the original. Export a headless job to process pending enhancements with Studio closed.

Choose **Edit Repair Prompt** to open the complete resolved repair prompt in a full-window editor.
Save keeps the nonblank text exactly as written, including surrounding whitespace; it does not
rewrite or reformat the prompt. **Use Recipe Prompt** clears the clip override and returns repair to
the selected recipe's original prompt. Changing or clearing the override makes only the existing
enhancement stale. It does not invalidate the base generation.

- **Adaptive** uses third temporal differences of H3 video latents, adjusted for the VAE's five-phase
  cadence, to allocate integer frame holds. It is a heuristic, not a reliable artifact detector.
  A quiet plan bypasses refinement. **Uniform** expands every source frame equally.
  Inspect the analysis before rendering: adaptive coverage can be sparse even during continuous
  action. Use Uniform when you want the entire clip treated.
- **Maximum hold** is 2–4×. Sensitivity affects adaptive coverage; refinement strength controls
  partial denoising. A value of 0.5 starts video at 50% noise; audio uses its corresponding
  shifted clock. The inspector displays strength numerically with 0.01 increments.
  By default, evaluation count is the ceiling of the recipe's full evaluations × strength.
  Enable **Set refinement evaluations** to choose 1–64 evaluations independently of strength;
  the initial suggested value is 14. This makes equal-budget strength comparisons possible
  without changing the base generation recipe. More evaluations cost time and do not guarantee
  better visual results. Existing projects retain their automatic evaluation counts.
  Long refinements report completed/total evaluations in the status area.
  The partial interval is resampled instead of taking the tail of a heavily shifted schedule.
  The seed belongs to enhancement independently of the base clip.
- Use an H3 T2VA repair recipe with at least 16 schedule points. The default uses the selected
  base render's recipe. Choose an explicit compatible repair recipe when the original used image,
  reference or audio conditioning. Its prompt and components govern refinement. A selected repair
  recipe may include standard LoRAs active for the full schedule; they apply only to refinement and
  their application is recorded in the result. Turbo/distilled or staged LoRAs, FastH3/VDN, cache
  accelerators and extra conditioning are rejected before weighted work.
  Check the adapter's fused attention layout separately from its tensor shapes. For a
  ComfyUI-exported adapter whose Q, K and V rows are contiguous, set the recipe adapter's
  `qkv_layout` to `contiguous_qkv`. The native engine uses per-head interleaved rows; choosing
  `native_interleaved` for contiguous weights applies the wrong attention deltas even when every
  target shape passes validation. `auto` uses declared layout metadata when present, otherwise
  it currently defaults standard adapters to the native layout. An undeclared export layout must
  be established before rendering. The result records `qkv_permuted_targets` alongside the
  applied target count. Adapter strength and refinement noise strength are separate controls.
- Input is constant 24 fps, with frame-aligned trims, 32-pixel-grid dimensions and 60–345 source
  frames. Expansion is padded to H3's `17k+5` geometry and must fit the configured budget, at most
  345 frames. The current RGB conversion also limits expanded width × height × frames to
  160 million pixels. Split longer clips or reduce the budget; no automatic windowing is claimed.
- Expanded conditioning audio is stretched with pitch-preserving FFmpeg filters. Final output
  uses the original trimmed soundtrack, re-encoded to AAC, at the original duration and frame
  rate. Silent sources receive silence. Model refinement can still alter mouth motion or identity.

Movie/clip jobs with Motion Fidelity use `weetodd-studio-job-v2` and embed their repair recipes,
including a clip's resolved prompt override. The bridge's read-only `motion-prepare` action returns
the resolved prompt, original recipe prompt and whether the clip uses an override; it validates the
recipe without writing preparation files or loading model weights. Whitespace-only or non-string
overrides are rejected before enhancement starts.
The current `WeeToddCLI` runs generation, enhancement, upscaling/interpolation and assembly serially.
Completed enhancements are hash-checked on resume; an interrupted enhancement restarts that clip's
refinement. It does not resume inside transformer sampling. The app-managed native runtime needs
this renderer revision; install a fresh runtime when adopting it and export a fresh job. Jobs with
the option off retain v1 behavior. Project files and Collect Media retain both source and enhancement,
including the enhancement's analysis and repair-recipe record. Model weights remain shared.

ComfyUI exposes **H3 Motion Fidelity Settings** and **H3 Motion Fidelity Refine**. Connect H3
Components, Generation Config and Motion Settings; provide a source movie path, native trim and
repair prompt. Analyze-only defaults on. The adapter runs the same isolated helper used by Studio
and jobs, returning a movie path and JSON analysis report. No paid workflow or external node pack
is required. The settings node's optional **evaluations** input uses 0 for the existing automatic
behavior, or 1–64 for a fixed count. Headless motion settings use `"evaluations": 14` for a fixed
count; omission or `null` keeps the automatic behavior. A change invalidates enhancement only,
while the original generation remains reusable. Existing workflow contracts remain compatible.

Validation includes a real native MLX partial-denoise render, exact 73-frame recovery at 24 fps,
32 kHz source-audio remuxing, mixed audio holds, optional bypass, old project decoding and separate
base/enhancement invalidation. Low-resolution fixtures establish execution and timing only;
they are unsuitable for judging motion fidelity. Refinement uses an explicit source-noise fraction
to avoid injecting near-pure noise from H3's heavily shifted generation schedule.
The motion-quality reference is a dense H3 boxing clip at 896×512, 124 frames and 24 fps, generated
with 20 schedule points (19 evaluations) and no FastVideo/VDN acceleration. Uniform 2× expansion
with strength 0.5 completed ten refinement evaluations over 260 padded frames, then recovered
exactly 124 frames with less than 1 ms AV drift. Matching-frame review retained the subject and
action, but changes were subtle and fast-glove blur remained. This does not establish a general
visual improvement or qualify dialogue/identity preservation across scenes.
The packaged CLI completed a v2 enhanced movie and reused the same final hash on resume. Native
Studio Enhance added a separate Clip Asset; toggling the source comparison reused it. The complete
suite passed 1,349 Python tests (one skip) and 13 Swift tests for this checkpoint.
LTX, imported-movie UI support, regional editing, overlapping long-clip windows, side-by-side viewing,
per-stage latent resume and broad dialogue/identity qualification remain future work.

The repair-prompt and standard-adapter checkpoint passed 1,470 Python tests (two optional skips)
and 14 Swift tests. The packaged app's full-window repair editor was checked interactively in
Light and Dark modes, including exact multiline Save, Escape cancellation, recipe-prompt reset
and missing-recipe recovery. These checks establish implementation behavior, not a quality preset.
Matched tests must also verify the adapter export layout; a shape-compatible wrong-layout run is
not valid evidence for or against that adapter.

A native LTX 2.5 implementation is a planned follow-on. It can reuse the clip editor, enhancement
versions and headless job behavior, but needs a separate engine plan for LTX's `8k+1` frame grid,
conditioning clock, refinement schedule and bounded overlapping windows. The H3 motion adapter
cannot be applied to LTX. Existing LTX source-latent refinement and frozen audio provide building
blocks; temporal DFR remains experimental and is not equivalent to this expansion/recovery method.
LTX support must preserve source timing and pass identity, action, audio and seam comparisons before
it becomes an available Motion Fidelity clip option.

## Development files and cleanup

- `studio/.build/` contains Swift build products and the local app. Rebuilding refreshes the bundled
  renderer; an already installed private runtime retains its own source snapshot. Set up a new runtime
  when adopting renderer changes, and export new headless jobs for that runtime.
- Studio projects, collected media folders, headless job JSON and companion instructions are ignored
  throughout the repository because they contain user content and local file paths.
- Application Support contains user work as well as caches: generated clip versions, autosave,
  global assets, imported recipes, jobs and runtime receipts. Back it up before manual maintenance.
  Collect Media is the supported way to preserve a movie's referenced media for portability.
- Prior managed runtimes are retained for existing jobs. There is no automatic cache cleanup or
  rollback selector yet; do not remove a runtime or render directory still referenced by a project/job.

Run `swift test --package-path studio` and
`python -m pytest -q tests/test_studio_bridge.py tests/test_studio_packaging.py tests/test_studio_lora.py` before packaging.
The packaging tests exercise a source tree without `.agents/`, stale-bundle replacement, and failure
preservation without downloading Python or installing models.

## Draw Things — experimental

Draw Things is an optional clip provider alongside native H3/LTX and imported movies. The shared
Python adapter invokes a separately built Swift gRPC helper; it does not import ComfyUI or load
native MLX generation weights. Native projects and v1/v2 headless jobs remain readable.

### Build the optional connection runtime

```bash
python3 scripts/build_drawthings_client.py
python3 scripts/build_studio_app.py --configuration release \
  --drawthings-distribution studio/.build/drawthings
open "studio/.build/WeeTodd Studio.app"
```

The helper uses Draw Things' official `_MediaGenerationKit` product, pinned to community revision
`08e798b5ad59c3db78b2be53f0ed60b071653302`, and requires Swift 6 on Apple Silicon.
This revision adds H3 transport/configuration support beyond the older public wrapper release.
Building it downloads software dependencies, not model weights. The distribution includes the helper,
its hash manifest, licenses within a complete dependency-source archive, and instructions for
rebuilding with modified libraries. Studio lets users import a replacement executable. The synthetic
fixture server is a development test target and is never bundled with Studio.

The current community revision is GPLv3. The older public `media-generation-kit` wrapper's
[LGPL grant](https://github.com/drawthingsai/media-generation-kit#license) applies to that package's
distribution; it does not establish a grant for this newer direct dependency. The build commands
above are for local development. Publishing a bundled Studio/helper app remains pending resolution
of the GPL distribution requirements or an applicable upstream alternative license. The helper's
notices include this distinction and its corresponding source; WeeTodd's own source remains
Apache-2.0.

When upgrading an existing installation, Studio automatically uses its bundled helper if the saved
helper setting is missing or empty. An explicitly imported helper path stays selected, and existing
Python, model-recipe, and finishing-tool settings are preserved.

A normal Studio build can omit this optional distribution. Existing native generation still works;
Draw Things Connections then requires importing a helper executable. Python and FFmpeg remain
necessary for the shared job/finishing bridge.

### Connections, allowance, and CU

Open **Movie → Draw Things Connections** and save a connection:

- **Self-hosted gRPC:** enter the Draw Things server host/port, TLS choice, and optional shared
  secret. Confirm that this server has cloud offload disabled. A local-looking address alone does
  not establish that generation is self-hosted. The server must stay running.
- **Draw Things Cloud API:** create an API key in the Draw Things dashboard and enter it in Studio.
  The helper uses fixed official HTTPS/gRPC endpoints with TLS verification. It first obtains an
  authentication session and reads billing/free-request status; it does not change billing settings.
  Preparation requires explicit PAYG-disabled status, remaining free requests, and a fresh monthly
  record. The saved API key is checked even when discovery omits CU thresholds. Missing or invalid
  billing/quota fields block generation.
- **DT+ App Bridge:** discovery can be configured, but free-only generation is unavailable. The
  current bridge protocol does not expose a verifiable account/allowance/no-paid-fallback policy.
  Draw Things being open or subscribed to Plus is not sufficient evidence.

**CU measures the estimated work of one generation.** It is separate from the remaining monthly
request count and is not a currency balance. Studio shows the estimate for the resolved settings.
When the service advertises CU thresholds, cloud preflight uses the lower threshold until generation
authorization confirms account class. A request equal to or above that limit is refused. If neither
Echo nor Hours publishes thresholds, Studio shows **CU limit checked by Draw Things on submission**;
it does not invent a numerical limit. A verified remaining free allowance with PAYG disabled permits
requesting server authorization when Generate is pressed. A fresh free-only authorization is still
required before the generation RPC; paid, Boost, unknown, and expired grants are rejected. Reduce
dimensions, duration, or steps if the server refuses the job. Allowance is rechecked at generation time.

This release supports `freeOnly`; it neither selects PAYG/Boost nor silently falls back to them.
Generation authorization is performed only after local files, model availability, connection, and
output creation pass. An interrupted authorization/submission may already have consumed a request;
there is no automatic retry. A live Studio LTX 2.3 Cloud API run verified the saved key and free
allowance, completed generation, and saved 81 frames at 768×448/25 FPS with 48 kHz stereo audio.
This validates that route for the tested request, not every cloud model or account configuration.

Credentials stay in Keychain, or in an explicitly selected runtime environment variable for CLI and
ComfyUI use. They are never embedded in project requests or job JSON. An exported credential reference
identifies how to supply a secret; it does not include that secret. The Python runner accepts
`WEETODD_DT_CREDENTIAL` or a profile `credentialRef` of `env:VARIABLE_NAME`. Treat exported project
prompts and media paths as private even though credentials are excluded.

### Images, clips, and LoRAs

Use the **+ → Generate Image…** action on Global, Project, or Clip Assets. The whole-window prompt
editor provides a central canvas, a separate ordered mood board (up to eight references), left-side
settings, CU preparation, result preview, and headless image-job export. Import files, drop image
assets, or choose **From Assets**. Canvas images support fit/fill placement and **Generation strength**
from 0–100%. Mood-board thumbnails have independent enable and strength controls; zero strength
omits a reference. FLUX.2/Klein currently treats positive reference weights as enabled references,
so intermediate weights are transmitted but are not a promise of proportionally reduced influence.
Multiple mood-board references are initially enabled for FLUX.2/Klein; other image families retain
canvas image-to-image support. Model choices reflect enabled input combinations. Steps, CFG, seed,
sampler, shift, and compatible LoRAs/groups are editable. **Use result as canvas** explicitly starts
another edit; generation never silently replaces the input. Control images and masks are not yet
enabled. Images are added to the captured destination store without changing the
timeline. A removed destination clip cannot silently redirect the completed image to another clip.

The initial canvas-plus-two-reference route was smoke-tested locally with FLUX.2 Klein 9B KV
at 512×512, four configured steps, and 65% generation strength. Ordered references and their
weights also have transport and headless-export tests. This does not qualify every image model
or DT Cloud; control adapters and model-specific reference behavior need separate validation.
The live Studio check also covered reference reordering, strength editing, enlarged preview,
explicit result-to-canvas reuse, and export. A 512×512, four-step Klein job with one canvas and
two references produced byte-identical PNGs in Studio and WeeToddCLI with Studio closed.

Image drafts now recover across restarts, separately for Global, Project and individual Clip
stores. Studio saves linked paths, prompts, settings, references, LoRA strengths and the latest
preview path in its application-support directory. It does not copy source media or save API
keys in drafts. Missing files remain linked for replacement. Reopening a draft requires fresh
CU/eligibility preparation; an old estimate is not treated as authorization.

**Import Config…** is available in both the image workspace and Draw Things clip inspector.
Open an exported JSON file or paste a configuration, then choose **Preview Import**. Named
`configuration` objects and arrays of presets are supported. The adjacent **Draw Things presets**
link opens the [official preset directory](https://github.com/drawthingsai/community-models/tree/main/configs);
each preset's `metadata.json` can be loaded here. Review the model, LoRAs, settings and omissions
before applying. If the preset names a model absent from the connection, explicitly choose an
installed model in the preview—for example, the same family in another precision. Studio never
silently substitutes model files. Prepare checks the resulting task, inputs and model together.

This initial importer supports model, prompt/negative prompt, dimensions, steps, CFG, seed,
sampler, generation strength, Shift, video FPS/frame count, Audio Shift, and whole-model LoRAs.
It accepts the `fpsId` and `shiftForAudio` aliases. Unspecified settings remain as they were;
an explicit empty LoRA list clears assignments. Prompt replacement is optional. Unsupported
settings, including controls, masks, High Res Fix and specialized adapter modes, are listed
and require explicit acknowledgment before omission. An imported preset is therefore not a
promise of complete Draw Things configuration parity.

Additional local M3 Ultra checks completed Krea 2 Turbo canvas I2I at 512×512/eight steps with
35% and 75% generation strength, and Klein 9B KV mood-board-only generation with two references
and two compatible LoRAs at different strengths. The Klein request completed in 11.5 seconds;
this is a functional smoke test, not a speed or broad image-quality benchmark. Cloud image-input
qualification remains pending; previous Cloud video validation does not establish image parity.
Live Studio testing also verified config preview/omission acknowledgment, an explicit Q6-to-Q8
model choice, LoRA-group saving, and quit/reopen recovery of both references, the prompt and
sampling/LoRA settings. Preparing the recovered request displayed 327 estimated CU for the
self-hosted route, and generation saved its result in Project Assets.

If macOS requests Keychain access when refreshing or generating, resolve its permission dialog.
Studio now performs credential reads off the UI thread and reports that wait in the status area.
Cancelling during that wait prevents job submission after the credential request returns.

Use the timeline **+ → Draw Things** to create a video clip. Refresh models, select an exact server
model, write its prompt, and prepare it. Native MLX recipe files are not needed for this provider.
Setup follows **Engine → Task → Connection → Model → LoRAs / Groups**. Tasks narrow verified
connections and models using their advertised input combinations. Unverified connections remain
available with **refresh to verify** until Studio has their catalog. Changing the task clears a
verified incompatible model selection; changing the connection clears the model selection.
Frame images and saved settings are retained. LoRAs and groups are filtered by the selected model.
Dimensions use a 64-pixel grid. Generation FPS must be an integer; LTX frame counts round upward to
`8n+1` to cover the requested duration. Movie finishing applies the project/clip output settings.
Generation adds an audiovisual movie to version history and Clip Assets. A changed clip is not
marked current by an older render finishing later.

**H3 first and last frames:** select **First and last frames** and drop images onto the timeline's
**FF** and **LF** slots, even before selecting a connection or model. Then choose a discovered
**MiniMax H3 FL2VA** model. You can also use **Use in clip → First frame** and **Use in clip → Last frame**
in Media & Assets. Selecting an incompatible model preserves your images and reports the mismatch;
Prepare Clip requires a compatible model before generation.
Studio sends the first image as Draw Things' canvas input and the last image as its first enabled
mood-board (`shuffle`) hint. Both images are center-cropped to the generation dimensions and hashed
before submission. The last endpoint follows the resolved frame count when duration changes.
There is no need to arrange the canvas or mood board in the Draw Things app.
After generation, endpoint clips adopt the resolved duration (124 / 24 = 5.167 seconds for a
five-second H3 request) so the timeline and headless movie finishing preserve the final frame.

H3 uses 24 FPS and `17n+5` frames (five seconds rounds up to 124 frames). Its defaults are 50 steps,
DDIM Trailing, CFG 1, Shift 12, and Audio Shift 3; steps and both shifts remain editable. H3 video
and 32 kHz stereo audio stay together. Only models actually advertised by the selected endpoint are
offered; a model installed in the local app is not necessarily available through the cloud API.
LTX still supports first-frame input only through this adapter. Last-only, arbitrary middle
keyframes, and H3 reference-model conditioning are not enabled. A conflicting attachment/task is
reported before generation rather than discarded. Headless exports use the same image contracts.
Clip Assets are storage; only items listed under **Conditioning** are generation inputs.
Keep previous renders in Clip Assets without attaching them as a Reference to a Draw Things clip.
Prepare Clip names unsupported attachments, and existing unsupported inputs show an inline warning.
Remove the attachment with **×** to retain its media in the asset store. Draw Things input menus
offer supported image endpoints; endpoint attachment strength is fixed at 1 (LoRA strength remains editable).

**H3 Turbo:** import a compatible Turbo LoRA into Draw Things, then click **Refresh** in Studio.
Enable it under **Server LoRAs**, set its strength, and edit **Steps**. A local FL2VA test used
`minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16` at **0.6 strength**, **4 steps**, DDIM Trailing,
CFG 1, Shift 12, and Audio Shift 3. It completed at 768×448 with 124 frames and 32 kHz stereo audio
in 226.8 seconds on an M3 Ultra. First/last images closely matched the supplied endpoints with
reconstruction differences. This is one local test, not a quality or speed guarantee for other
LoRAs, hardware, resolutions, or prompts. Studio references the server's installed LoRA and does
not copy its weights.

Start remote LTX clips at **24 or 25 FPS**. The pinned Draw Things decoder produces audio on a
fixed causal clock, independently of playback FPS. Its complete audio can end slightly before the
last picture: a 121-frame clip contains 4.81 seconds of audio, versus 5.042 seconds of video at
24 FPS. The helper and Python bridge independently verify the exact causal sample count, then
preserve the original frame rate and soundtrack without stretching either. The remaining picture
plays after the soundtrack ends. Other audio still uses the one-frame duration check; missing or
incomplete audio is rejected. This exception does not permit audio extending past the video.

Earlier helpers could report a receiving/finalization error after all frames and audio arrived.
Update both the helper and renderer (refresh an app-managed runtime if applicable). Preserve the
failed job's `render/media` folder before retrying: complete received files may be recoverable
locally without another cloud generation. A missing completion manifest is not proof that the
remote request failed or that another submission would be free.

The pinned helper recognizes selected FLUX, Qwen Image, and Z-Image model families for images, and
LTX 2/2.3 and H3 FL2VA for video. Only exact endpoint IDs advertised by both the helper and server
appear. Remote LTX 2.5 remains unsupported; use its native engine.

One **First Frame** image is supported for LTX video. Its full file hash participates in preparation
and request identity, and the helper rechecks it before submission. Orientation is respected and the
image is center-cropped to generation dimensions. H3 FL2VA also supports a First Frame / Last Frame
pair, as described above. LTX last frames, interior keyframes, generic references, audio drivers,
control hints, and native clip extensions are rejected explicitly in this remote adapter.

Server LoRAs are filtered by exact remote model compatibility. Strength ranges from 0 to 2. Named
remote groups copy their members/strengths to a clip and remain separate from the native LoRA library.
Before discovery, or after a failed refresh, saved models and LoRAs are marked unverified rather than
unavailable. Saved LoRA strengths remain editable. For local servers, enable the gRPC API and Model
Browsing in Draw Things, then Refresh in Studio. Only a successful catalog can mark an assignment
unavailable for that connection/model; generation always revalidates it.
Only ordinary LoRAs with matching SDK model family are advertised; specialized modifiers, alternate
decoders, and unverified variants are excluded. Importing local SafeTensors as a remote LoRA is not
supported: this adapter has no verified converter/upload workflow.

### Headless jobs and qualification

Movie/clip exports containing Draw Things work use `weetodd-studio-job-v3`. Image jobs can be exported
from the image prompt editor. Jobs remain sequential and refresh eligibility before each request.
Image-to-video dependencies bind a completed image as the first frame before the video is estimated.
See [headless examples and credential setup](../examples/headless/README.md).

Resume reuses only verified artifact hashes. A recorded submitted/completed remote request whose
artifact is unavailable is never automatically regenerated; create a deliberate new job/output after
checking the earlier attempt. Closing Studio is supported. Self-hosted jobs still require their server;
Cloud API jobs do not require the Draw Things app.

| Path | Qualification |
| --- | --- |
| gRPC discovery/image transfer | Synthetic server and Studio image UI tested |
| Video + separate audio | Synthetic gRPC, Studio clip UI, and real FFmpeg timing/publication tested |
| First/last frame / LoRA contracts | Automated wire mapping, compatibility, hash, duration/resume, status, and failure tests; real local H3 FL2VA four-step Turbo render verified from both the shared renderer and packaged Studio |
| CLI image-to-video and movie assembly | Synthetic server with Studio closed; real FFmpeg clip, dissolve, title, and supplementary-audio assembly tested |
| Completed-job resume | With fixture server stopped, reused both remote artifacts and the same final movie hash |
| ComfyUI image/video/estimate workflows | Saved, fixture-bound API graphs executed in isolated ComfyUI; repeated estimates refreshed and image output saved |
| Packaged Studio | Bundled helper discovery, connection test, prompt CU, project reload, and light/dark appearance checked; H3 endpoint render marked Generated after restart and exported with all 124 frames |
| Helper corresponding source | Distributed archive extracted and rebuilt against its supplied editable dependencies |
| Native project/job compatibility | Focused regression tests |
| Real Draw Things model generation | Local H3 FL2VA: 124 frames at 768×448/24 FPS plus 32 kHz stereo audio; LTX 2.3 Cloud: 81 frames at 768×448/25 FPS plus 48 kHz stereo audio |
| Direct Cloud free-tier generation | Saved-key verification, free-allowance check, Prepare and Generate completed in Studio with PAYG disabled; unknown allowance still fails closed |
| DT+ App Bridge generation | Unavailable pending verifiable billing policy |

Fixture tests establish software behavior, not output quality or a promise that a particular remote
model will fit a free-tier allowance. Retail signing/notarization and clean-Mac qualification remain
separate release work.
