# MiniMax H3 prompting research

The official MiniMax H3 guide defines a structured audiovisual prompt rather than an unstructured
scene caption. Use three ordered fields: an integrated timeline, an overall soundscape, and
non-diegetic music direction.

## T2VA guidance

- Start with `[Shot 1]` and describe style, composition, subject, action, camera, and ending state.
- Describe actions in chronological order and connect physical causes to visible results.
- Use one continuous shot when a short duration cannot support a meaningful cut.
- Express camera motion as motion type plus meaningful amplitude and speed.
- Put synchronized physical sounds and ambience in `overall_soundscape`.
- Use `non_diegetic_music: N/A` when no background score is required.

## Other tasks

I2VA, FL2VA, and L2VA add exact picture-alignment instructions before the same three core fields.
Reference tasks can assign images, videos, and audio to subject, motion, camera, style, voice, or
editing-rhythm roles. The prompt must describe a continuous path between frame anchors.

## Project decision

WeeTodd generation prompts must be shown verbatim before a generation starts. Generation metadata
must preserve the exact prompt. The project-local `h3-video-prompting` skill applies this rule.

## Sources

- [MiniMax H3 base prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [MiniMax H3 reference prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [MiniMax H3 video-generation guide](https://platform.minimax.io/docs/guides/video-generation)
