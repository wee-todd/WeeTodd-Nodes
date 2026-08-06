---
type: Playbook
title: MiniMax H3 prompting contract
description: Required prompt structure and disclosure rule for synchronized MiniMax H3 generation.
resource: ../docs/reference/H3_PROMPTING.md
tags: [minimax-h3, prompting, audio, video]
status: stable
generated:
  by: process:codex-review
  at: 2026-08-05T19:45:00-07:00
sources:
  - id: minimax-h3-base-prompt-guide
    resource: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
    title: Video Prompt Writing Guide for T2VA, I2VA, FL2VA, and L2VA
  - id: minimax-h3-video-guide
    resource: https://platform.minimax.io/docs/guides/video-generation
    title: MiniMax H3 Video Generation
---

# Contract

Write the integrated timeline, overall soundscape, and non-diegetic music fields in that order.
Describe visible actions chronologically. Connect physical sounds to the actions that produce them.

# Disclosure rule

Show the complete prompt to the user before every generation. Preserve the exact prompt in
generation metadata.

# Short-duration rule

Prefer one continuous shot for a five-second smoke test. Use one clear action sequence, one camera
move, and one ending state.
