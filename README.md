# WeeTodd Nodes

Experimental MLX-native MiniMax H3 nodes for ComfyUI on Apple Silicon.

WeeTodd Nodes keeps the MiniMax H3 engine separate from the ComfyUI adapter. The current release
focuses on synchronized text-to-video-plus-audio generation, staged model unloading, and a
low-memory path that can be tested before enabling optional acceleration.

## Current scope

- Apple Silicon and macOS
- Python 3.11 or later in the ComfyUI environment
- MLX 0.32.0 or later
- Text-to-video-plus-audio (`t2va`)
- Synchronized H.264 video and 32 kHz stereo AAC audio
- Twenty-two composable nodes under `WeeTodd/H3`

First-frame, last-frame, combined first/last-frame, and reference-conditioning nodes are not yet
registered. The current sampler does not provide a live latent preview.

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
| Resolution | `384P (fast smoke)`, `16:9` |
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

The optional MPP projection backend targets speed for eligible BF16 projections. It does not lower
the measured MLX peak and is not part of the minimum-memory workflow. Q4 artifacts are not
published or supported.

## Troubleshooting

### A model page is visible but downloads return 401

Sign in to Hugging Face, accept the gate on that model page, and run `hf auth login` in the same
user account that performs the download.

### Preflight does not detect paging

Check that `paged_manifest.json` or `paged_text_encoder_manifest.json` is directly inside the
selected override directory. Remove accidental nested repository folders.

### Preflight reports a missing processor, tokenizer, or audio VAE

The three WeeTodd artifacts intentionally omit those components. Complete the licensed base
`FL2VA` partition or set the corresponding Component Loader overrides.

### ComfyUI runs out of memory

Start a fresh ComfyUI process. Restore the supplied 640 by 384 settings. Disable every LoRA,
cache, forecast, and preview. Confirm that both unload controls remain enabled and use Direct
Publish Latents instead of retaining a complete ComfyUI image batch.

### Publication fails

Confirm that FFmpeg can encode H.264 video and AAC audio. Check the ComfyUI output directory for
write permission. The publisher writes atomically and removes incomplete media after failure.

## Development validation

Run inexpensive validation with the project environment:

```bash
python -m compileall -q src __init__.py
python -m pytest -q tests/test_nodes.py tests/test_runtime.py tests/test_workflows.py
python -m ruff check src/wee_todd_nodes tests
python scripts/lint_docs.py
```

The published artifacts were also tested by a complete authenticated download into a clean
ComfyUI 0.30.0 installation. All remote inventories and weight hashes matched, all 22 nodes
registered, the supplied graph contracts validated, and header-only component preflight passed.
No generation is started by preflight.

## License and status

The code in this repository is licensed under Apache-2.0. Model repositories carry their own
license files and access conditions. WeeTodd Nodes does not bundle or automatically download model
weights.

The project is experimental and pre-release. Node interfaces and workflow compatibility may
change.
