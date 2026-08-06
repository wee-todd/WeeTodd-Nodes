---
name: h3-video-prompting
description: Write, review, or optimize prompts for MiniMax H3 synchronized video-and-audio generation. Use for T2VA, I2VA, FL2VA, L2VA, reference conditioning, dialogue, sound design, camera direction, or any WeeTodd H3 generation prompt.
---

# H3 Video Prompting

Write the prompt before starting generation. Show the complete prompt to the user before the
request enters the queue.

## Workflow

1. Identify the task, duration, references, intended action, camera behavior, and audio intent.
2. Read `references/prompt-contract.md` when the request uses dialogue, multiple shots, keyframes,
   or reference media.
3. Fit the number of actions and cuts to the duration. Prefer one continuous shot for a five-second
   smoke test.
4. Write the exact H3 three-field structure.
5. Describe visible action in chronological order. Give each action a clear cause and result.
6. State camera motion naturally. Add amplitude and speed only when they improve control.
7. Put synchronized physical sounds and ambience in `overall_soundscape`.
8. Set `non_diegetic_music: N/A` unless the user requests background music.
9. Show the final prompt verbatim before generation. Do not replace it with a summary.
10. Preserve the exact prompt in generation metadata.

## T2VA template

```text
integrated_multimodal_description: [Shot 1] <style, composition, subject, chronological action,
camera motion, lighting, and ending state>.

overall_soundscape: <ambient sound and synchronized physical action sounds>.

non_diegetic_music: <score description or N/A>
```

## Guardrails

- Keep one stable subject description across the prompt.
- Use observable actions instead of abstract mood instructions.
- Avoid contradictory camera commands and simultaneous unrelated actions.
- Do not invent dialogue. Preserve user-provided dialogue exactly inside the official `<d>` syntax.
- Do not use image-alignment syntax for T2VA.
- Label an off-distribution smoke-test resolution separately from the creative prompt.
