---
type: Contract
title: Synchronized H3 output publication
description: Defines final media validation, atomic MP4 publication, metadata, and cleanup.
resource: ../docs/ARCHITECTURE.md
tags: [minimax-h3, comfyui, ffmpeg, audio-video, output]
status: draft
generated:
  by: process:codex-review
  at: 2026-08-05T23:00:00-07:00
sources:
  - id: smoke-test-plan
    resource: ../docs/SMOKE_TEST_PLAN.md
    title: First ComfyUI smoke-test plan
  - id: comfyui-output-host
    resource: https://github.com/Comfy-Org/ComfyUI/tree/15989f87ca89bfe2e7c47763252c559e96d97551
    title: ComfyUI output host
---

# Input

The publisher accepts decoded ComfyUI `IMAGE`, decoded ComfyUI `AUDIO`, the component
specification, the generation configuration, and optional JSON metadata.

# Validation

Video must use 24 fps RGB frames. Audio must use 32 kHz stereo samples. The publisher permits at
most 25 ms of duration drift because one H3 audio latent spans 25 ms.

The publisher rejects incorrect tensor shapes, non-finite audio, unsafe output prefixes, invalid
metadata, and configuration mismatches before encoding.

# Publication

The publisher converts images directly to contiguous byte RGB and audio to contiguous float audio.
FFmpeg writes H.264 video and AAC audio to a temporary MP4. The publisher atomically moves the MP4
and JSON sidecar to collision-safe names under the ComfyUI output directory.

The JSON sidecar records sanitized component names, generation values, software versions, measured
durations, drift, codecs, encode time, and the precision policy.

# Cleanup

Success, failure, and cancellation remove the temporary WAV file. Failure and cancellation also
remove partial video and metadata files.

# Limitation

Cancellation is checked before and after the FFmpeg call. Mid-encode FFmpeg process termination is
not implemented.
