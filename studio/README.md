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

In **Studio Settings**, choose **Set Up Managed Renderer** to download private Python 3.12.13 and
install pinned, hash-verified native dependencies. The installer verifies arm64, the project Python
requirement, package consistency, and Metal availability before activating the new runtime.
Each installation gets a new directory; prior runtimes remain available. The developer environment
and other applications' environments are preserved. There is no Linux virtual machine.
The bootstrap uses a pinned, checksum-verified uv release from Astral.

Advanced users can connect an existing compatible WeeTodd repository and Python environment.
Import existing `weetodd-headless-v2` recipes to identify model component sets. Automatic selection
matches the clip engine and media roles. Model downloads and conversions are not automated by
this first GUI build. FFmpeg/FFprobe and optional RIFE remain separately configured tools;
a retail installer must package these tools with their licenses and finish the model setup flow.

Runtime settings, autosave, global assets, recipes, previews and jobs live under
`~/Library/Application Support/WeeTodd Studio`. User media and model weights stay in their existing
locations. `WEETODD_STUDIO_DATA` selects a separate data directory for isolated development tests.

## Editing

- Clip inspector at upper left, inherited movie settings below; central viewport and timeline;
  collapsible Global, Project, and selected Clip asset stores at right.
- System, Light, and Dark appearance. Section colors in the design wireframe are not used.
- One main video track, titles, and additional named audio tracks with mute, solo, and source-audio
  replacement. Set each region's start, source in, duration, volume and fades in its inspector.
- H3, LTX 2.3 and LTX 2.5 generated clips, imported movies/stills, and image sequences. Sequence import
  uses the movie frame rate, natural filename order, linked original frames and a ProRes editing proxy.
- Drop a movie asset on the timeline to create a clip and a source reference in its Clip Assets.
  Drop images on a generated clip near its beginning/end for endpoint conditioning, or in its middle
  for a timed keyframe. Roles can also be set explicitly. Drag clip cards to reorder them.
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

**Preview movie** builds a reduced-resolution movie containing the actual transitions, titles,
and active audio tracks. Clip preview is immediate; movie preview is rebuilt after edits. Missing
renders must be generated first. Movie preview omits interpolation/upscaling, which are applied
in final export. This first build does not provide live multitrack compositing or waveform editing.

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

Before a broad consumer release: complete model discovery/download UI and tool packaging, sign and
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

- **Adaptive** uses third temporal differences of H3 video latents, adjusted for the VAE's five-phase
  cadence, to allocate integer frame holds. It is a heuristic, not a reliable artifact detector.
  A quiet plan bypasses refinement. **Uniform** expands every source frame equally.
  Inspect the analysis before rendering: adaptive coverage can be sparse even during continuous
  action. Use Uniform when you want the entire clip treated.
- **Maximum hold** is 2–4×. Sensitivity affects adaptive coverage; refinement strength controls
  partial denoising. A value of 0.5 starts video at 50% noise; audio uses its corresponding
  shifted clock. Evaluation count is the ceiling of the recipe's full evaluations × strength.
  The partial interval is resampled instead of taking the tail of a heavily shifted schedule.
  The seed belongs to enhancement independently of the base clip.
- Use a plain H3 T2VA repair recipe with at least 16 schedule points. The default uses the selected
  base render's recipe. Choose an explicit compatible repair recipe when the original used image,
  reference or audio conditioning. Its prompt and components govern refinement. FastH3/VDN,
  LoRAs, cache accelerators and extra conditioning are rejected, not silently stripped.
- Input is constant 24 fps, with frame-aligned trims, 32-pixel-grid dimensions and 60–345 source
  frames. Expansion is padded to H3's `17k+5` geometry and must fit the configured budget, at most
  345 frames. The current RGB conversion also limits expanded width × height × frames to
  160 million pixels. Split longer clips or reduce the budget; no automatic windowing is claimed.
- Expanded conditioning audio is stretched with pitch-preserving FFmpeg filters. Final output
  uses the original trimmed soundtrack, re-encoded to AAC, at the original duration and frame
  rate. Silent sources receive silence. Model refinement can still alter mouth motion or identity.

Movie/clip jobs with Motion Fidelity use `weetodd-studio-job-v2` and embed their repair recipes.
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
is required. The existing 42 shipped workflows are unchanged.

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
suite passed 1,345 Python tests (one skip) and 12 Swift tests for this checkpoint.
LTX, imported-movie UI support, regional editing, overlapping long-clip windows, side-by-side viewing,
per-stage latent resume and broad dialogue/identity qualification remain future work.

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
`python -m pytest -q tests/test_studio_bridge.py tests/test_studio_packaging.py` before packaging.
The packaging tests exercise a source tree without `.agents/`, stale-bundle replacement, and failure
preservation without downloading Python or installing models.
