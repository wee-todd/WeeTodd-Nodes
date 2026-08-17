# WeeTodd Nodes

MLX-native MiniMax H3, LTX 2.3, and LTX 2.5 custom nodes for ComfyUI on Apple Silicon.

WeeTodd keeps each model engine behind a separate ComfyUI adapter. H3 generation preserves video
and audio as one synchronized latent contract. Weighted components load only when the graph runs
and can unload between Qwen3-VL, transformer, video VAE, and audio VAE stages.

- 47 composable nodes under `WeeTodd/H3`

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

Each H3 profile includes the same four task graphs. Media selectors are empty by design. Select
images, video, or audio after loading the workflow.

| Task | Speed | Balance | Performance |
| --- | --- | --- | --- |
| T2V + audio | [Open workflow](workflows/speed/t2v/h3_t2v_speed.json) | [Open workflow](workflows/balance/t2v/h3_t2v_balance.json) | [Open workflow](workflows/performance/t2v/h3_t2v_performance.json) |
| I2V + audio | [Open workflow](workflows/speed/i2v/h3_i2v_speed.json) | [Open workflow](workflows/balance/i2v/h3_i2v_balance.json) | [Open workflow](workflows/performance/i2v/h3_i2v_performance.json) |
| First/last-frame video + audio | [Open workflow](workflows/speed/fflf2va/h3_fflf2va_speed.json) | [Open workflow](workflows/balance/fflf2va/h3_fflf2va_balance.json) | [Open workflow](workflows/performance/fflf2va/h3_fflf2va_performance.json) |
| Ref2VA | [Open workflow](workflows/speed/ref2va/h3_ref2va_speed.json) | [Open workflow](workflows/balance/ref2va/h3_ref2va_balance.json) | [Open workflow](workflows/performance/ref2va/h3_ref2va_performance.json) |

`fflf2va` covers first frame, last frame, and combined first/last-frame conditioning. Use **H3
Frames** when the graph needs numbered middle frames.

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
| Balance | [LTX 2.5 Union Canny](workflows/balance/ref2va/ltx25_union_canny_balanced.json) | MLX-native Canny preprocessing with baked Q8 Union Control. |
| Balance | [LTX 2.5 Union Depth](workflows/balance/ref2va/ltx25_union_depth_balanced.json) | MLX Video Depth Anything preprocessing with baked Q8 Union Control. |
| Balance | [LTX 2.5 Union Pose](workflows/balance/ref2va/ltx25_union_pose_balanced.json) | Preprocessed pose-video control with baked Q8 Union Control. |
| Performance | [LTX 2.5 Motion Track quality](workflows/performance/i2v/ltx25_motion_track_quality.json) | MLX-generated colored trajectories with the dedicated Motion Track IC-LoRA and full 15-forward CFG++. |
| Performance | [LTX 2.5 768×512 guided HQ](workflows/performance/t2v/ltx25_768x512_guided_hq.json) | Development transformer with selectable 30-step guided Euler or 15-step guided `res_2s`, followed by the official distilled-LoRA refinement stage. |
| Balance | [LTX 2.5 768×512 practical DFR](workflows/balance/t2v/ltx25_768x512_dfr_conv_vae.json) | Exact prebaked Q8 DFR sampling with bounded convolutional-VAE publication. |
| Performance | [LTX 2.5 768×512 DFR + Diffusion VAE](workflows/performance/t2v/ltx25_768x512_dfr_diffusion_vae.json) | Experimental full-resolution detail generation with exact prebaked Q8 adapters and one-step pixel-diffusion decode. |
| Performance | [LTX 2.5 768×512 accelerated DFR + Diffusion VAE](workflows/performance/t2v/ltx25_768x512_dfr_diffusion_vae_metal_tiled.json) | Experimental query-tiled Metal decoder with substantially lower runtime and memory than the exact Diffusion VAE. |
| Performance | [LTX 2.5 768×512 DFR temporal 48 fps](workflows/performance/t2v/ltx25_768x512_dfr_temporal_48fps.json) | Diagnostic-only learned temporal refinement with untouched stage-one audio; current MLX visual parity is not production-ready. |
| Performance | [LTX 2.5 1920×1088](workflows/performance/t2v/ltx25_1920x1088_two_stage.json) | High-resolution quality-first generation. |
| Balance | [LTX 2.5 chained timeline](workflows/balance/continuation/ltx25_768p_15s_three_window_chain.json) | Experimental long-timeline continuation and selective regeneration. |
| Balance | [LTX 2.5 video refine](workflows/balance/video-upscale/ltx25_any_video_pixel_spatial_2x.json) | Refine and upscale any source movie while preserving its audio. |

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

The project requires MLX 0.32.0 or later. Confirm that ComfyUI uses arm64 Python before installing:

```bash
"$COMFYUI_ROOT/.venv/bin/python" -c \
  'import platform, sys; print(sys.version); print(platform.machine())'
```

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

Model sources:

- [Official MiniMax H3 components](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Q8-extended paged transformer](https://huggingface.co/Vayden/MiniMax-H3-MLX-q8-extended-paged)
- [Qwen3-VL Q8 paged conditioner](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-paged)
- [MiniMax H3 video VAE MLX Q8](https://huggingface.co/Vayden/MiniMax-H3-Video-VAE-MLX-Q8)
- [drbaph v4 step-600 Turbo LoRA](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI)
- [H3 tiny preview decoder](https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors)

Use the genuine Ref2VA partition for Ref2VA workflows. The shipped graphs do not enable the FL2VA
compatibility override.

## LTX 2.5 model layout

Accept the [LTX 2.5 license](https://huggingface.co/Lightricks/LTX-2.5), then place the split files
in standard ComfyUI folders.

| ComfyUI folder | Required file |
| --- | --- |
| `models/diffusion_models/` | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` |
| `models/diffusion_models/` | Official DFR: `ltx-2.5-22b-dev-transformer-bf16.safetensors` |
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

For separate character, object, clothing, and background images, attach **LTX 2.5 MSR Loader**
and chain one to five **LTX 2.5 MSR Reference Stack** nodes. The stack preserves connection order,
moves the optional background to the final learned slot, and emits `Image 1` through `Image 5`
prompt guidance. MSR currently uses the full-resolution single-stage distilled path and cannot be
combined with another IC-LoRA, video-reference stack, or audio-reference stack. Each reference is
encoded independently; subject and object images are fitted without cropping, while the optional
background is center-cropped. The default 33-frame quality policy matches the target canvas;
balanced and speed are explicit lower-density experiments. Use 25 reference frames with Sol
Attention when the resolved reference grid produces 64-row-aligned groups; the standard
1,152-by-640 quality grid then produces 2,880 rows per subject and stays on the fused path. A
33-frame reference at that grid produces 3,600 rows and safely falls back to dense attention. The
loader validates the learned Fourier-slot tensors and all 480 rank-128 adapter pairs before
execution, and reads the five BF16 slot tensors through MLX without evaluating the full adapter.

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
guide serve LTX 2.5 Union Control or another compatible graph without keeping a generation model
resident.

Implementation order follows control usefulness and available checkpoint terms:

| Category | Options | Default direction | Status |
| --- | --- | --- | --- |
| Edges | Canny; TEED soft edge | Canny for exact structure; TEED for learned contours | Both available |
| Depth | Video Depth Anything Small; Depth Anything V2 Small | Video depth for consistency; frame depth for speed | Both available |
| Pose | whole-body pose; body-only pose | Whole-body for face and hand motion; body-only for speed | DWPose available |
| Structure | depth-derived normals; realistic line art; segmentation | Normals and line art are available; segmentation remains gated | Partial |
| Motion | colored trajectory guides; automated optical-flow extraction | Use the Motion Track guide with the dedicated adapter; tracker extraction remains future work | Guide available |

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

## Live H3 previews

Every shipped H3 workflow routes components through **H3 Model Preview Override** before sampling.
The node publishes a true-color contact sheet during sampling and releases the tiny decoder after
success, failure, or cancellation.

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
| 4 | Extend real LTX 2.5 MSR validation from two references to three through five | High reference feature value | Medium; real two-subject identity, BF16 loading, dense masks, and 384-call grouped Metal execution pass |
| 5 | Ref2VA automatic-density quality validation | High Ref2VA speed and memory reduction | Medium; full density remains the default |
| 6 | LTX 2.5 temporal image-conditioning and longer-chain parity | Medium feature completeness | Medium |

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
| H3 Component Loader | Describe every MiniMax H3 component. This node does not load tensor weights. | H3 — Loaders | Recommended |
| H3 Model Preview Override | Attach a true-color TAE preview and optional collapse guard to H3 sampling. Core ML can keep preview decoding on the Apple Neural Engine; MLX remains the fallback. Place this node between the component loader and sampler. | H3 — Sampling and acceleration | Experimental |
| H3 Quantized Transformer Loader | Select and validate a named mixed-precision H3 transformer without loading weights. Both q8 profiles are approximate and keep BlockCache disabled by default. | H3 — Loaders | Experimental |
| H3 Component Preflight | Validate MiniMax H3 components and estimate staged memory from file headers. Set available memory to zero when unknown. | H3 — Loaders | Recommended |
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
| H3 Encode References | Prepare ordered Ref2VA media, then stage Qwen3-VL, the video VAE, and the audio VAE. Each weighted component unloads before the next stage. | H3 — Conditioning | Experimental |
| H3 Reference Strength | Adjust how strongly FL2VA or Ref2VA trusts visual and audio condition rows. Defaults preserve the released H3 behavior. | H3 — Conditioning | Experimental |
| H3 Text Encode (Qwen3-VL) | Encode a text-only H3 prompt with Qwen3-VL. The vision tower stays unloaded. The encoder can unload after it produces conditioning. | H3 — Conditioning | Recommended |
| H3 Unload Qwen3-VL | Release the process-local Qwen3-VL conditioner and clear the MLX cache. | H3 — Conditioning | Supported |
| H3 Motion Continuation Context | Copy a synchronized tail from H3 video and audio latents for motion continuation. The recommended 22-frame overlap is about 0.92 seconds at 24 fps. | H3 — Continuation | Experimental |
| H3 Append Latent Chain Window | Append one synchronized latent window to a validated H3 chained timeline. | H3 — Continuation | Experimental |
| H3 Sample Video + Audio Latents | Sample synchronized MiniMax H3 video and audio latents with MLX. This node does not load or run either VAE. | H3 — Sampling and acceleration | Recommended |
| H3 Latent Hi Res Fix | Enlarge an H3 video latent and run a second H3 visual refinement pass. The original synchronized audio latent is returned unchanged. | H3 — Sampling and acceleration | Experimental |
| H3 LoRA Loader (MLX) | Build a lazy, ordered MiniMax H3 LoRA stack. Validate safetensors headers now and load adapter tensors only when the H3 transformer executes. | H3 — Loaders | Supported |
| H3 Validated Sampling Preset | Apply a measured dense, trajectory-replay, or Turbo sampling policy. Connect all three typed outputs to the H3 sampler. | H3 — Sampling and acceleration | Recommended |
| H3 Sol Attention (MLX Experimental) | Experimental fused MLX Metal sparse attention for long H3 sequences. It preserves the complete multimodal prefix exactly and falls back to dense MLX for unsupported calls. | H3 — Sampling and acceleration | Experimental |
| H3 EasyCache (MLX) | Configure joint MLX EasyCache residual reuse for H3 video and audio sampling. Choose quality-first, balanced, or speed-first bounded automatic reuse. | H3 — Sampling and acceleration | Experimental |
| H3 Trajectory Forecast (MLX) | Experimentally forecast compact post-transformer H3 video and audio features. Current timestep output heads still run on every step. Turbo LoRA is supported. | H3 — Sampling and acceleration | Experimental |
| H3 BlockCache (MLX) | Always run H3 block zero and the current output heads, then safely reuse the cached joint audio/video residual of later transformer blocks when both modality indicators agree. | H3 — Sampling and acceleration | Experimental |
| H3 Hierarchical BlockCache (MLX) | Split the 50 H3 blocks into three contiguous segments. Always evaluate each segment's anchor block, accept video and audio together, and reuse eligible segment tails independently. | H3 — Sampling and acceleration | Experimental |
| H3 Unload Transformer | Release the process-local H3 transformer and clear the MLX cache. | H3 — Sampling and acceleration | Supported |
| H3 Decode Video VAE | Decode the video stream from synchronized H3 latents with the final video VAE. The audio latent stream remains available on the original latent output. | H3 — Decoding | Supported |
| H3 Unload Video VAE | Release the process-local H3 video VAE and clear the MLX cache. | H3 — Decoding | Supported |
| H3 Decode Audio VAE | Decode the audio stream from synchronized H3 latents as 32 kHz stereo audio. The video latent stream remains available on the original latent output. | H3 — Decoding | Supported |
| H3 Unload Audio VAE | Release the process-local H3 audio VAE and clear the MLX cache. | H3 — Decoding | Supported |
| H3 Trim Continuation Overlap | Remove the repeated motion-continuation overlap from decoded video and audio, then normalize audio to the exact remaining video duration. | H3 — Continuation | Experimental |
| H3 Publish Video + Audio | Validate and publish synchronized H3 images and 32 kHz stereo audio as MP4. The node writes an atomic JSON metadata sidecar. | H3 — Output | Supported |
| H3 Direct Publish Latents (MLX) | Decode synchronized H3 latents directly to MP4 through staged MLX VAEs. The node avoids a persistent ComfyUI IMAGE tensor and unloads each VAE after use. | H3 — Output | Recommended |
| H3 Direct Publish Chained Timeline (MLX) | Decode an H3 latent chain by VAE stage, remove duplicated joins, force exact 24 fps / 32 kHz duration, and atomically publish one MP4. | H3 — Output | Experimental |
| H3 Model Loader (MLX) | Describe an MLX MiniMax H3 checkpoint. Weights load lazily at generation time. | H3 — Core and convenience | Legacy/convenience |
| H3 Generation Config | Choose a clearly labeled aspect ratio and move the short-edge size slider, or use exact dimensions. The live canvas remains on H3's required 32-pixel grid. | H3 — Core and convenience | Recommended |
| H3 Low-Memory Tuning (MLX) | Apply optional MLX attention-head and feed-forward row chunking without invalidating older Generation Config workflows. | H3 — Sampling and acceleration | Supported |
| H3 Generate Video + Audio | Generate synchronized video and audio with MiniMax H3 through MLX. | H3 — Core and convenience | Legacy/convenience |
| H3 Unload MLX Runtime | Release state held by the monolithic H3 runtime. | H3 — Core and convenience | Legacy/convenience |
| LTX 2.3 Model Loader (MLX) | Select a local LTX 2.3 MLX bundle. No weights load in this node. | LTX 2.3 — Loaders | Supported |
| LTX 2.3 Generation Config | Configure LTX 2.3 mode, canvas, duration, steps, guidance, and memory policy. | LTX 2.3 — Core | Supported |
| LTX 2.3 Preflight | Validate the selected LTX 2.3 bundle and mode-specific components before allocation. | LTX 2.3 — Loaders | Recommended |
| LTX 2.3 Generate Video + Audio | Generate synchronized LTX 2.3 video and 48 kHz stereo audio through MLX. | LTX 2.3 — Core | Experimental |
| LTX 2.3 Upscaler Loader | Select and preflight a learned LTX 2.3 spatial latent upscaler. | LTX 2.3 — Loaders | Experimental |
| LTX 2.3 Upscale + Publish | Upscale decoded H3 or other ComfyUI video frames with the LTX latent upscaler and preserve the supplied audio. | LTX 2.3 — Upscaling | Experimental |
| LTX 2.3 Unload MLX Runtime | Release the process-local LTX 2.3 pipeline. | LTX 2.3 — Core | Supported |
| LTX 2.5 Component Loader (MLX) | Select LTX 2.5 split components without loading weights or downloading files. Self-describing paged transformers may contain one prebaked IC-LoRA. | LTX 2.5 — Loaders | Experimental |
| LTX 2.5 LoRA Loader (MLX) | Attach a generic LTX 2.5 transformer LoRA, including block and non-block targets. Multiple loader nodes may be chained. Use the dedicated loader for IC-LoRA task adapters. | LTX 2.5 — Loaders | Supported |
| LTX 2.5 IC-LoRA Loader (MLX) | Attach one LTX 2.5-compatible IC-LoRA for video/reference conditioning. Official LTX 2.3 22B adapters pass an additional shape check. The selected IC-LoRA Pipeline Mode determines whether the adapter runs for stage one or the full generation. Do not use this node with a transformer that already bakes the same IC-LoRA. | LTX 2.5 — Loaders | Experimental |
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
| LTX 2.5 IC-LoRA Pipeline Mode | Select full or hybrid CFG++, the eight-forward single-stage shortcut, or the existing two-stage stage-one-control pipeline. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 Ingredients Reference Sheet | Condition LTX 2.5 from one Ingredients reference sheet. The image is repeated internally across the full clip and encoded as IC-LoRA reference context. Quality, balanced, and speed policies control the encoded reference grid independently of the output canvas. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 MSR Reference Stack | Build an ordered one-to-five-image LTX 2.5 MSR stack. Subject and object references stay in connection order; one optional background is always assigned the final slot. | LTX 2.5 — Conditioning | Supported |
| LTX 2.5 Generate Video + Audio | Generate synchronized LTX 2.5 video and audio through the MLX adapter. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Generate Chained Timeline | Generate two to four overlapping LTX 2.5 windows with timeline-aligned latent guides, causal-aware latent transitions, and one synchronized audio/video decode. Supports the two-stage path and full-resolution single-stage Sol configurations. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Video Upscale / Refine | Upscale decoded ComfyUI IMAGE+AUDIO from any movie through LTX 2.5 latent space, optionally adding video-only refinement while preserving the source audio. | LTX 2.5 — Core | Experimental |
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
| Motion Track Guide (MLX) | Render sparse colored point trajectories into the guide-video representation expected by the LTX Motion Track IC-LoRA. This node uses MLX and does not require a checkpoint. | MLX preprocessors — Motion | Experimental |
| Unload MLX Preprocessors | Release weighted MLX preprocessor state without changing H3 or LTX residency. | MLX preprocessors — Lifecycle | Experimental |
<!-- END GENERATED NODE CATALOG -->

## Troubleshooting

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

Run inexpensive validation before each commit:

```bash
python scripts/audit_workflow_catalog.py --project .
python scripts/update_readme_node_catalog.py --check
python scripts/validate_okf.py knowledge
python scripts/lint_docs.py
python -m compileall -q src __init__.py
python -m pytest -q tests/test_nodes.py tests/test_runtime.py tests/test_readme.py tests/test_workflows.py
ruff check src/wee_todd_nodes tests
```

`tests/test_readme.py` and `tests/test_workflows.py` are required by the README/workflow commit gate.
Full checkpoint parity and real generation tests are optional and expensive.

## License and status

WeeTodd source code is Apache-2.0. Model checkpoints and adapters keep their original licenses and
terms. The project is experimental and pre-release. Do not commit checkpoints, generated media,
caches, credentials, tokens, machine-specific paths, or private information.
