# Engine contract

- H3 jointly denoises video and audio rows in one packed sequence.
- Video runs at 24 fps; audio latent time runs at 40 Hz; decoded audio is 32 kHz stereo.
- Supported duration is currently 5–15 seconds.
- Canvas dimensions must be divisible by 32. Native geometry uses a 768-pixel short edge; smaller defaults are off-distribution wiring tests.
- CFG is distilled into the released model; there is no conventional negative prompt or guidance scale.
- AdaLN schedule precomputation can remove roughly 13B resident parameters after schedule construction.
- Keep the Qwen3-VL encoder truncated at the hidden state H3 consumes.
- Reuse compatible pipelines and provide explicit unload behavior.
- Dense attention dominates wall time. Quantization does not remove attention FLOPs.

Use detailed parity and optimization records under `docs/reference/` when modifying the engine.
