# WeeTodd Nodes

MLX-native MiniMax H3, LTX 2.3, and LTX 2.5 custom nodes for ComfyUI on Apple Silicon.

WeeTodd keeps each model engine behind a separate ComfyUI adapter. H3 generation preserves video
and audio as one synchronized latent contract. Weighted components load only when the graph runs
and can unload between Qwen3-VL, transformer, video VAE, and audio VAE stages.

- 46 composable nodes under `WeeTodd/H3`

Recent experimental controls include target-frame H3 image, clip, and audio guides; an H3 token
and attention-workspace estimator; independent MLX attention-head and feed-forward row chunking;
LTX 2.5 timed input keyframes; learned generated-keyframe slots; and lazy ordered LTX 2.5 LoRA
stacks. An optional modifier predicts a one-shot duration from the prompt with the official LTX
2.5 duration head. A separate DFR modifier can add learned temporal refinement at 48 or 96 fps.
Existing modifier and conditioning nodes preserve older Generation Config socket order.
The H3 projection backend adds a backward-compatible `auto` choice; saved workflows that explicitly
select `mlx` retain that selection. Add a modifier or conditioning node only when the feature is
needed.

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
| Performance | [LTX 2.5 768×512 DFR + Diffusion VAE](workflows/performance/t2v/ltx25_768x512_dfr_diffusion_vae.json) | Experimental full-resolution detail generation and one-step pixel-diffusion decode. |
| Performance | [LTX 2.5 768×512 DFR temporal 48 fps](workflows/performance/t2v/ltx25_768x512_dfr_temporal_48fps.json) | Experimental learned temporal refinement with untouched stage-one audio. |
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
| `models/text_encoders/` | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| `models/vae/` | `ltx-2.5-video-vae-conv-bf16.safetensors` |
| `models/vae/` | Optional detail decoder: `ltx-2.5-video-vae-bf16.safetensors` |
| `models/vae/` | `ltx-2.5-audio-vae-bf16.safetensors` |
| `models/latent_upscale_models/` | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| `models/latent_upscale_models/` | Optional DFR temporal refinement: `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` |
| `models/model_patches/LTX-2.5/` | Optional automatic duration: `ltx-2.5-duration-head-bf16.safetensors` |
| `models/loras/LTX-2.5/` | Optional DFR: [`ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) |

The video-refine workflow also requires the
[pixel-spatial upscaler IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)
under `models/loras/LTX-2.5/`.

Use `scripts/convert_ltx25_paged_q8.py` to create directly loadable Q8 pages. Keep the licensed BF16
source files and checkpoint terms. Generated page directories are model artifacts and must not be
committed.

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
| H3 paged Qwen3-VL plus q8-extended transformer | Memory | Very high | Matched 384p complete-process peak fell from 28.823 GB to 14.951 GB. | Qwen paging preserved the MP4 digest. Q8 transformer weights remain approximate relative to BF16. |
| H3 four-evaluation Turbo | Speed | Very high | Uses four real evaluations instead of the 19-evaluation dense schedule. | Changes the sampling trajectory. |
| H3 staged Turbo | Speed/quality | Very high | Uses two base plus four Turbo evaluations. | Changes the sampling trajectory but retains base-model setup evaluations. |
| H3 direct latent publication | Memory | High | Streams decoded frames to FFmpeg and avoids a persistent complete `IMAGE` tensor. | Does not change sampling. |
| H3 adaptive Ref2VA video density | Ref2VA speed | High | In a matched 640×384 run, automatic half-density reduced sampling time by 32.35% and the observed process peak by 1.08 GB. Automatic mode retains full Qwen inspection and selects persistent VAE-row density from adjacent-frame activity. | Experimental; full density followed source motion more closely and remains the default. Automatic decisions are reported in conditioning metadata. |
| H3 head and FFN row chunking | Sampling memory | Unmeasured; bounded by construction | Independently limits SDPA head groups and packed-row SwiGLU intermediates through the Low-Memory Tuning node. | Same operations and ordering within each independent chunk; checkpoint parity tests are still required before claiming bit identity. |
| H3 token and workspace budget | Preflight | Diagnostic | Reports target, conditioning, packed-token, attention-score, and bounded-workspace estimates before allocation. | Reporting only. |
| LTX 2.5 Q8 paging | Memory and speed | Very high | 9.90 GB and 78.38 s versus 31.60 GB and 102.13 s for matched BF16 one-block streaming at 768×512. | Q8 changes the numerical trajectory. |
| LTX 2.5 temporal VAE tiling | Decode memory | High for long clips | Activates from an explicit decode-memory budget and grows in value with duration. | Preserves the synchronized output contract. |
| LTX 2.5 generated-keyframe slots | Motion allocation | Experimental | Adds evenly distributed learned interior slots during stage one without changing existing workflow schemas. | Changes the latent token sequence and output. |
| LTX 2.5 Diffusion VAE | Decode quality | Experimental | On a matched fast-motion 512×512 latent, the reference decode took 66.41 s at an 8.41 GB MLX peak. | Runs the official one-step pixel-diffusion decoder and changes decoded pixels. Conv VAE remains the speed and low-memory default. |
| LTX 2.5 Diffusion VAE layouts | Decode performance and compatibility | Experimental | `metal_na3d_experimental` reduced the matched decode to 14.79 s, or 4.49× faster, and reduced MLX peak from 8.41 to 5.41 GB. Output measured SSIM 0.98886 and PSNR 44.45 dB. | Use the Diffusion VAE Optimization node. The Metal result is approximate. Keep combined mode as the default. |
| LTX 2.5 query-tiled Metal Diffusion VAE | Decode memory | High | A 65,536-row tile reduced matched 512×512 Metal peak from 5.41 to 4.24 GB. Decode time changed from 15.83 to 16.77 s, and the MP4 digest remained identical. | Select `metal_na3d_query_tiled_experimental` only when Diffusion VAE memory has priority. Fine tiles are substantially slower. |
| LTX 2.5 automatic duration | Usability | Low runtime cost | A real Q8 prompt probe spent 0.027 s in the MLX duration head after prompt encoding. | Opt-in modifier; manual duration remains authoritative unless connected. Raw predicted seconds and resolved `8k+1` frames are recorded. |
| LTX 2.5 Diffusion VAE width tiling | Decode memory | Experimental | A 32-cell stage-four stripe reduced 512×512 peak from 8.27 GB to 7.62 GB. | Decode slowed from 61.99 s to 100.13 s and output was not pixel-identical. Select `stage4_width_tiles` only when memory is the priority. |
| LTX 2.5 DFR | Full-resolution detail | Experimental | Adds generated stage-one slots, stage-one latent reference conditioning, and a stage-two-only Pixel-Spatial IC-LoRA. | DFR generatively changes composition and motion. It preserves stage-one audio but is not a decoder-only enhancement. |
| LTX 2.5 DFR temporal refinement | Motion smoothness | Experimental | One preserved 256×256 T2V latent refined from 49 frames at 24 fps to 97 frames at 48 fps in 11.68 s. A complete matched I2V probe took 41.34 s and peaked at 9.35 GB for the process. | One round adds four evaluations per seam-aware tile. Stage-one audio is untouched. One-shot image and timed-keyframe anchors are reapplied; chained timelines remain gated. |

### Remaining optimization priorities

| Rank | Candidate | Potential gain | Confidence |
| ---: | --- | --- | --- |
| 1 | Fused H3 QKV, Q/K normalization, rotary, SDPA, and output projection | High speed and data-movement reduction | Medium |
| 2 | Ref2VA reference-feature reuse and adaptive temporal packing | High Ref2VA speed and memory reduction | Medium |
| 3 | H3-specific W4A8 projection kernel | Very high checkpoint and resident-memory reduction | Low |
| 4 | Adaptive full-block H3 paging | Very high BF16 peak-memory reduction | Medium |
| 5 | LTX 2.5 visual-token routing parity | High long-duration speed potential | Low until the routing contract is verified |
| 6 | H3-specific MLX attention tiles | High transformer speed potential | Medium; preserve multimodal prefix and exact output shape |
| 7 | LTX 2.5 temporal image-conditioning and chain parity | Medium feature completeness | Medium |

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
| LTX 2.5 Component Loader (MLX) | Select LTX 2.5 split components without loading weights or downloading files. | LTX 2.5 — Loaders | Experimental |
| LTX 2.5 LoRA Loader (MLX) | Attach a generic LTX 2.5 transformer LoRA. Multiple loader nodes may be chained; IC-LoRA control media still requires its matching conditioning workflow. | LTX 2.5 — Loaders | Supported |
| LTX 2.5 Generation Config | Configure the official distilled 8+3-evaluation LTX 2.5 two-stage schedule. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Automatic Duration | Predict one-shot duration from the prompt with the official LTX 2.5 duration head. The manual duration remains unchanged when this modifier is not connected. | LTX 2.5 — Core | Supported |
| LTX 2.5 Generated Keyframes | Apply LTX 2.5 generated interior keyframe slots as a composable config modifier. | LTX 2.5 — Conditioning | Supported |
| LTX 2.5 Diffusion VAE Optimization | Select an MLX Diffusion VAE execution layout. It does not affect the convolutional VAE. | LTX 2.5 — Optimization | Experimental |
| LTX 2.5 DFR Detail Refinement | Enable MLX Diffusion Fidelity Rendering: segment-grid generated keyframes, stage-one latent reference conditioning, stage-two-only Pixel-Spatial IC-LoRA, and untouched stage-one audio publication. | LTX 2.5 — Conditioning | Experimental |
| LTX 2.5 DFR Temporal Refinement | Add one or two learned x2 temporal DFR rounds. Each round preserves stage-one audio, doubles playback frame rate, reapplies one-shot image anchors, and adds four transformer evaluations per temporal tile. | LTX 2.5 — Conditioning | Supported |
| LTX 2.5 Preflight | Validate LTX 2.5 component metadata and architecture requirements before allocation. | LTX 2.5 — Loaders | Experimental |
| LTX 2.5 Timed Keyframe | Append a first, middle, or last image at an exact zero-based pixel-frame index. Nonzero frames use LTX 2.5's generated-keyframe token slots. | LTX 2.5 — Conditioning | Supported |
| LTX 2.5 Generate Video + Audio | Generate synchronized LTX 2.5 video and audio through the MLX adapter. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Generate Chained Timeline | Generate two to four overlapping LTX 2.5 windows with timeline-aligned latent guides, causal-aware latent transitions, and one synchronized audio/video decode. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Video Upscale / Refine | Upscale decoded ComfyUI IMAGE+AUDIO from any movie through LTX 2.5 latent space, optionally adding video-only refinement while preserving the source audio. | LTX 2.5 — Core | Experimental |
| LTX 2.5 Unload MLX Runtime | Release process-local LTX 2.5 state. | LTX 2.5 — Core | Supported |
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
