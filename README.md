# WeeTodd Nodes

Experimental MLX-native MiniMax H3 nodes for ComfyUI on Apple Silicon.

WeeTodd Nodes keeps the MiniMax H3 engine separate from the ComfyUI adapter. The current release
focuses on synchronized text-to-video-plus-audio generation, staged model unloading, and a
low-memory path that can be tested before enabling optional acceleration.

![MiniMax H3 512p resident and low-memory benchmark matrix](assets/h3_512p_sampling_preset_comparison.png)

The matrix compares all six validated sampling policies with clean-speech and fast-motion prompts.
The measured low-memory profile reduced complete ComfyUI process peak by approximately 35 to 39 GB
while increasing workflow time by approximately 5 to 16 percent. See
[Validated sampling presets](#validated-sampling-presets) for the measurement boundary.

## Current scope

- Apple Silicon and macOS
- Python 3.11 or later in the ComfyUI environment
- MLX 0.32.0 or later
- Text-to-video-plus-audio (`t2va`)
- Experimental first-frame, last-frame, and combined first/last-frame (`fl2va`) conditioning
- Experimental ordered image, video, soundtrack, and audio reference (`ref2va`) conditioning
- Independent visual and audio reference-strength control
- Synchronized H.264 video and 32 kHz stereo AAC audio
- Thirty-two composable nodes under `WeeTodd/H3`

The FL2VA path stages Qwen3-VL vision and video-VAE encoding before transformer sampling. The
Ref2VA path stages media preparation, Qwen3-VL vision, video-VAE encoding, audio-VAE encoding, and
transformer sampling. Each weighted component unloads before the next low-memory stage. A genuine
Ref2VA checkpoint remains the intended strict path, but it is not generation-validated in this
release. One separately compressed Ref2VA checkpoint produced invalid low-variance video at both
eight and 20 requested schedule points and was rejected. The Component Loader also offers an
advanced, explicit FL2VA-to-Ref2VA compatibility switch for comparisons when only FL2VA weights
are installed. That switch is off by default: the official partitions share transformer
architecture and sampling shifts but contain different learned weights. The current sampler does
not provide a live latent preview.

Reference images use an output-relative pixel budget. The 100-percent default matches the output
canvas area while preserving the source aspect ratio. Values from 50 to 400 percent trade
reference-token cost against source detail; they do not change the output resolution.

Place **Reference Strength** after the FL2VA or Ref2VA encoder to adjust how strongly the sampler
trusts its prepared condition rows. Visual strength defaults to `0.999`; audio strength defaults to
`1.0`. Lower values add more seeded noise to that modality. Start with visual values from `0.7` to
`0.999`. Values below `0.7` can weaken the FL2VA last-frame anchor. Lowering either value can alter
generated audio because H3 evaluates audio and video jointly. Keep the seed, prompt, and all other
settings fixed when comparing strengths.

Start with [`fl2va_first_frame_workflow.json`](examples/fl2va_first_frame_workflow.json) for image-
to-video-plus-audio or [`ref2va_image_workflow.json`](examples/ref2va_image_workflow.json) for one
ordered image reference. Replace the placeholder input image and edit the prompt before queuing.

## Install

Clone the project into the ComfyUI custom-node directory. Install it with the same Python
interpreter that runs ComfyUI.

```bash
COMFYUI_ROOT=/path/to/ComfyUI

git clone https://github.com/wee-todd/WeeTodd-Nodes.git \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes"

"$COMFYUI_ROOT/.venv/bin/python" -m pip install -e \
  "$COMFYUI_ROOT/custom_nodes/WeeTodd-Nodes"
```

Install FFmpeg with H.264 encoding and Advanced Audio Coding (AAC) support. Restart ComfyUI and
confirm that the `WeeTodd/H3` category is present.

## Obtain the model components

The following public, gated MLX artifacts are provided for the low-memory workflow:

- [H3 q8-extended paged transformer](https://huggingface.co/Vayden/MiniMax-H3-MLX-q8-extended-paged)
- [Qwen3-VL Q8 paged conditioner](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-paged)
- [MiniMax H3 video VAE MLX Q8](https://huggingface.co/Vayden/MiniMax-H3-Video-VAE-MLX-Q8)

They are grouped in the
[WeeTodd MiniMax H3 MLX collection](https://huggingface.co/collections/Vayden/weetodd-minimax-h3-mlx-for-comfyui-6a7765772401cdd54a992af0).

Review and accept the license terms on each model page. Then download the artifacts into the
ComfyUI model directory.

```bash
COMFYUI_ROOT=/path/to/ComfyUI

"$COMFYUI_ROOT/.venv/bin/hf" auth login

"$COMFYUI_ROOT/.venv/bin/hf" download Vayden/MiniMax-H3-MLX-q8-extended-paged \
  --local-dir "$COMFYUI_ROOT/models/MiniMax-H3/transformers/q8_extended_paged"

"$COMFYUI_ROOT/.venv/bin/hf" download Vayden/Qwen3-VL-32B-H3-MLX-q8-paged \
  --local-dir "$COMFYUI_ROOT/models/MiniMax-H3/text_encoders/q8-paged"

"$COMFYUI_ROOT/.venv/bin/hf" download Vayden/MiniMax-H3-Video-VAE-MLX-Q8 \
  --local-dir "$COMFYUI_ROOT/models/MiniMax-H3/vae/q8"
```

These repositories do not form a complete H3 checkpoint. Obtain the remaining MiniMax H3
components under their applicable licenses. The Component Loader still requires a partition root
with `model_index.json`, plus the processor, tokenizer, and audio VAE.

Use this portable layout:

```text
ComfyUI/models/
└── MiniMax-H3/
    ├── FL2VA/
    │   ├── model_index.json
    │   └── <remaining licensed H3 components>
    ├── text_encoders/
    │   └── q8-paged/
    ├── transformers/
    │   └── q8_extended_paged/
    └── vae/
        └── q8/
            └── video_vae_affine_q8.safetensors
```

Do not add a second directory level inside any downloaded repository. The paging manifests must
be at the roots shown above.

The Component Loader resolves relative paths through all compatible model roots registered by
ComfyUI, including roots from `extra_model_paths.yaml`. It checks the component's normal ComfyUI
category first and then the remaining registered model categories. Absolute paths remain available
for diagnostics, but portable workflows should use relative paths.

### Choose the canvas size

Generation Config separates shape from size:

1. Select a labeled landscape, square, or portrait aspect ratio.
2. Select a named short-edge shortcut or choose `Use size slider — 32 px steps`.
3. Move the short-edge slider from 32 through 1088. Resolved width and height update in the node.

Every dimension stays on H3's 32-pixel grid. Neither resolved dimension can exceed 1920 pixels,
so very wide or tall ratios automatically use a lower slider maximum. For example, H3 widescreen
resolves 640 to 1120 by 640, 768 to 1344 by 768, and 1088 to 1920 by 1088. The named shortcuts are
384, 480, 512, 576, 640, 672, 704, 736, 768, 896, 1024, and 1088 pixels on the short edge.

Choose `exact dimensions` for a manually entered canvas. Width and height must each be 32 through
1920 in 32-pixel steps. The aspect-ratio display changes to `custom` when those dimensions do not
match a supported ratio. Legacy preset names remain readable through workflow migration.

## Run the low-memory ComfyUI smoke test

Load
[`examples/t2va_low_memory_paged_workflow.json`](examples/t2va_low_memory_paged_workflow.json) in
ComfyUI. The matching API prompt is
[`examples/t2va_low_memory_paged_api.json`](examples/t2va_low_memory_paged_api.json).

The graph contains six nodes:

| Order | Node | Purpose |
| ---: | --- | --- |
| 1 | Component Loader | Select the base partition and three paged or Q8 overrides. |
| 2 | Generation Config | Set the smoke-test geometry and low-memory policy. |
| 3 | Component Preflight | Validate every component before allocating model weights. |
| 4 | Text Encode | Run paged Qwen and unload it after encoding. |
| 5 | Sample Video + Audio Latents | Run the paged transformer and unload it after sampling. |
| 6 | Direct Publish Latents | Decode each VAE in sequence and publish synchronized media. |

Keep the supplied first-run settings:

| Setting | Value |
| --- | --- |
| Task | `t2va` |
| Resolution | `384 px short edge — fast smoke`, `16:9 — widescreen landscape` |
| Canvas | 640 by 384 |
| Duration | 5 seconds |
| Requested schedule points | 5 |
| Transformer evaluations | 4 |
| Memory mode | `low_memory_bf16` |
| Attention chunk size | `automatic` |
| Drop AdaLN after cache build | Enabled |
| Projection backend | `mlx` |
| Qwen unload after encode | Enabled |
| Transformer unload after sample | Enabled |
| Cache, forecast, and LoRA nodes | Disabled |

The workflow selects these portable overrides:

```text
transformer: MiniMax-H3/transformers/q8_extended_paged
text_encoder: MiniMax-H3/text_encoders/q8-paged
video_vae: MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors
```

Edit the prompt if desired, but keep the first test concrete and physically coherent. Describe the
shot, subject motion, camera motion, environmental motion, sound effects, ambience, and dialogue in
time order.

### Check preflight before queuing

Run **Component Preflight** before generation. Do not queue the sampler if preflight reports a
missing file, unsupported task, incompatible checkpoint, or negative memory headroom.

For the published artifacts, preflight should report:

- Transformer paging format: `weetodd-h3-paged-v1`
- Text-encoder paging format: `weetodd-h3-qwen-paged-v1`
- Video VAE precision: `mlx-affine-8bit-group-64`
- Partition: `fl2va`
- Output frames: 124 for the five-second test

The header-based staged estimate for the verified workflow was 5.696 GB. This estimate excludes
ComfyUI, Python, Metal workspace allocations, mapped pages, allocator retention, and operating
system pressure. It is not a complete-process requirement.

### Expected memory behavior

The low-memory path limits simultaneous residency across the pipeline:

1. Paged Qwen retains its fixed tensors and loads one language layer at a time.
2. Qwen unloads before transformer sampling begins.
3. The transformer retains its fixed tensors and loads four H3 blocks at a time.
4. AdaLN projection weights are released after the request-local cache is built.
5. The transformer unloads before VAE decoding begins.
6. Direct publication runs the video and audio VAEs in sequence and removes partial files after a
   failure or cancellation.

A clean 640 by 384 validation with dual paging, the Q8 video VAE, staged unloading, and Turbo LoRA
reached a 14.951 GB complete ComfyUI process peak. The supplied smoke workflow omits Turbo and all
caches. Treat 14.951 GB as one measured capacity point, not a guarantee for every Mac or prompt.
A 16 GB system has little safety margin after macOS and other applications are included.

## After the baseline succeeds

Change one variable at a time and retain the same seed for comparisons.

- Increase requested schedule points from 5 to 8 before testing higher values.
- Test a larger resolution only after the 640 by 384 workflow completes.
- Add one LoRA only after a no-LoRA control succeeds.
- Add one acceleration node only after an uncached control succeeds.
- Restart ComfyUI for clean complete-process memory comparisons.

EasyCache, BlockCache, Hierarchical BlockCache, and Trajectory Forecast are mutually exclusive.
They can change video and audio output. Turbo LoRA with a cache also requires the explicit
experimental opt-in. None of these options belongs in the first low-memory smoke test.

Trajectory Forecast provides an optional `offline_smoothing_replay` mode for audio-sensitive runs.
The first pass captures actual joint video and audio features. A transformer-free second pass
restarts from the original latents, reuses the actual anchors, and reconstructs skipped steps from
past and future anchors. The default replay blend is 0.5 for video and zero for audio. Replay can
prevent forecasted video state from entering a later joint transformer evaluation, but the anchor
archive increases memory use. Start with eight or more requested schedule points and compare the
same prompt and seed with replay disabled before adopting the mode.

The optional MPP projection backend targets speed for eligible BF16 projections. It does not lower
the measured MLX peak and is not part of the minimum-memory workflow. Q4 artifacts are not
published or supported.

### Validated sampling presets

Use **Validated Sampling Preset** after **Generation Config** to apply a complete measured sampling
policy. Connect its config, LoRA, and trajectory outputs to **Sample Video + Audio Latents**. The
node preserves the selected canvas, duration, seed, memory mode, and component paths. It corrects
the Euler schedule and loads a required Turbo adapter only when sampling starts.

| Preset | Requested points | Transformer evaluations | Additional policy |
| --- | ---: | ---: | --- |
| Dense baseline | 20 | 19 | None |
| Trajectory speed with offline replay | 20 | Up to 11 | Guarded trajectory capture and replay |
| Ref2VA four-reference BF16 with Forward Attention replay | 20 | Up to 11 | Four ordered image references, normal-memory BF16 sampling, guarded replay |
| Turbo, Larry EMA-850 | 5 | 4 | Matching Turbo LoRA |
| Turbo, Larry v4 step-600 | 5 | 4 | Matching Turbo LoRA |
| Turbo, LightX2V full rank | 5 | 4 | Matching Turbo LoRA |
| Turbo, LightX2V dynamic rank 21 | 5 | 4 | Matching Turbo LoRA |

The [`workflows/`](workflows/) directory contains one ready-to-load 896 by 512 ComfyUI workflow
for each preset. Install a workflow's named LoRA in a ComfyUI LoRA model folder before loading a
Turbo workflow. WeeTodd does not bundle or download adapters.

[`h3_512p_ref2va_four_reference_forward_attention.json`](workflows/h3_512p_ref2va_four_reference_forward_attention.json)
reproduces the measured four-image Ref2VA graph. Select Little Red, wolf, Granny, and woodsman
images in that order after loading it. The workflow uses the explicit FL2VA-to-Ref2VA compatibility
switch because the separately compressed Ref2VA checkpoint tested for this release produced
invalid low-variance video. Treat the compatibility path as experimental and review reference
fidelity before production use. The measured 896 by 512 run used normal-memory BF16 sampling,
20 requested points, 11 actual transformer evaluations, eight forecasts, no LoRA, and reached
47.32 GB of MLX peak allocation. This MLX allocator figure is not complete ComfyUI process memory.

The following matrix used one Apple Silicon system, seed 246813579, 896 by 512 output, 5.17
seconds, 124 frames, and Euler sampling. It compares a clean-speech prompt and a fast-motion stress
prompt with resident and low-memory component profiles. The low-memory profile uses a paged Q8
transformer, paged Q8 text encoder, and Q8 video VAE. It reduced measured complete-process peak by
about 35 to 39 GB, or 70 to 72 percent, while increasing workflow time by 5 to 16 percent.

Complete-process memory includes ComfyUI, Python, mapped pages, Metal allocations, and allocator
retention. The resident low-motion dense control did not record a fresh-process peak. Compare
memory profiles only within the same prompt class. Treat every result as a measured capacity point,
not a guarantee. Trajectory replay and Turbo adapters change the generated video and audio.

The featured benchmark matrix near the top of this README contains the complete results.

### Turbo LoRA compatibility

The LoRA Loader keeps adapter updates in activation space. Do not fuse a small Turbo update into a
BF16 base weight because BF16 rounding can remove most of the update.

Turbo adapters that use contiguous Q, K, and V output rows require conversion to the transformer's
per-head interleaved row layout. The `auto` QKV layout selects this conversion for the Turbo
profile. Generation metadata reports the resolved layout and the number of converted QKV targets.
Use `native_interleaved` only when the adapter publisher confirms that layout.

The four tested Turbo adapter variants contain 52 fused-QKV targets. The two full Turbo variants
also contain 51 AdaLN targets. The two lighter FL variants omit AdaLN.

A same-seed 640 by 480 validation loaded all 259 targets from a full Turbo adapter and converted all
52 QKV targets. The resident BF16 run sampled in 128.47 seconds and reached a 49.52 GB complete
ComfyUI process peak. A dual-paged Q8 run sampled in 157.19 seconds and reached 14.94 GB. Both runs
produced valid 124-frame H.264 video with stereo 32-kHz AAC audio and 8.3 milliseconds of drift.
The outputs are not a numerical parity comparison because the paged run also uses quantized
transformer, text-encoder, and video-VAE components.

## Troubleshooting

### A model page is visible but downloads return 401

Sign in to Hugging Face, accept the gate on that model page, and run `hf auth login` in the same
user account that performs the download.

### Preflight does not detect paging

Check that `paged_manifest.json` or `paged_text_encoder_manifest.json` is directly inside the
selected override directory. Remove accidental nested repository folders.

### Generation Config values shift after loading an older workflow

Restart ComfyUI and reload the workflow. The frontend migration restores legacy Generation Config
layouts and converts older `P` labels to the explicit ratio-and-short-edge controls. Save the
workflow again after it loads correctly.

### Preflight reports a missing processor, tokenizer, or audio VAE

The three WeeTodd artifacts intentionally omit those components. Complete the licensed base
`FL2VA` partition or set the corresponding Component Loader overrides.

### ComfyUI runs out of memory

Start a fresh ComfyUI process. Restore the supplied 640 by 384 settings. Disable every LoRA,
cache, forecast, and preview. Confirm that both unload controls remain enabled and use Direct
Publish Latents instead of retaining a complete ComfyUI image batch.

### Publication fails

Read the `publication` section of the Component Preflight report. It shows the output directory
registered by the running ComfyUI backend and the FFmpeg executable selected for publication.

If a Desktop or Finder launch does not inherit Homebrew's shell path, set the advanced
`ffmpeg_path` field on the preflight and publication nodes, or set `WEETODD_FFMPEG` before starting
ComfyUI. Discovery then checks the process path and an installed `imageio-ffmpeg` package. Do not
place a symlink inside ComfyUI's virtual environment; replacing that environment removes it.

The nodes always publish below `folder_paths.get_output_directory()` so ComfyUI previews remain
safe and addressable. Configure a shared output root in the ComfyUI backend, such as with
`--output-directory`; a frontend-only folder preference does not change the backend root. Confirm
that the reported directory is writable. Publication is atomic and removes incomplete media after
failure.

## Development validation

Run inexpensive validation with the project environment:

```bash
python -m compileall -q src __init__.py
python -m pytest -q tests/test_nodes.py tests/test_runtime.py tests/test_workflows.py
python -m ruff check src/wee_todd_nodes tests
python scripts/lint_docs.py
```

The published artifacts were also tested by a complete authenticated download into a clean
ComfyUI 0.30.0 installation. All remote inventories and weight hashes matched, every node present
in that release registered, the supplied graph contracts validated, and header-only component
preflight passed. No generation is started by preflight.

## License and status

The code in this repository is licensed under Apache-2.0. Model repositories carry their own
license files and access conditions. WeeTodd Nodes does not bundle or automatically download model
weights.

The project is experimental and pre-release. Node interfaces and workflow compatibility may
change.
