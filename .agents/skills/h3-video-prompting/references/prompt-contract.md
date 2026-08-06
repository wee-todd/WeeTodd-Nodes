# MiniMax H3 prompt contract

## Core fields

Use these fields in order:

1. `integrated_multimodal_description`
2. `overall_soundscape`
3. `non_diegetic_music`

Start the first visual section with `[Shot 1]`. Do not put a timestamp on the first shot. Start a
later shot with a strictly increasing cut time, such as `[Shot 2] At 00:03.500`.

## Camera direction

Write camera movement as a natural action. Supported concepts include push, pull, pan, truck, tilt,
pedestal, arc, tracking, static, shake, point of view, and roll. Add small or large amplitude and
slow or fast speed only when needed.

## Audio

Put dialogue and visible diegetic music inside the shot description. Assign stable speaker IDs.
Format spoken content as `<d>[Language] exact words</d>`. Summarize ambience, impacts, movement,
and non-verbal sounds in `overall_soundscape`. Use `non_diegetic_music: N/A` when no audience-only
score is requested.

## Keyframe alignment

- I2VA: State that Picture 1 is fully referenced at 0.00 seconds.
- FL2VA: State the exact first-frame and last-frame alignment times before the core fields.
- L2VA: State the final Picture 1 alignment time before the core fields.
- Prefer a continuous action path between keyframes. Do not repeat static image descriptions.

## Sources

- [Official MiniMax H3 base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [Official MiniMax H3 reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [Official MiniMax H3 video-generation guide](https://platform.minimax.io/docs/guides/video-generation)
