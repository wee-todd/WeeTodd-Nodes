# MiniMax H3 on ComfyUI: August 2026 snapshot

ComfyUI now contains native MiniMax H3 nodes for empty synchronized audio/video latents, first/last image conditioning, and multi-reference image/video/audio conditioning. That implementation is PyTorch-oriented and uses ComfyUI's nested tensors and model patching.

Spectrum adds transformer-feature forecasting around native PyTorch H3. Its conservative 20-step schedules report roughly 30–35% fewer native transformer evaluations, but quality, sampler, topology, and memory constraints remain material. It is GPL-3.0 and serves only as a research reference here.

This review found no mature public full MLX-native H3 ComfyUI suite. WeeTodd Nodes fills that gap: direct MLX execution on Apple Silicon, with ComfyUI as the workflow shell rather than the tensor runtime.

The Phosphene-derived engine has parity coverage for DiT, packing, scheduler, both VAEs, and text encoder. Dense attention cost is its main practical limitation. Initial node defaults intentionally favor wiring tests over native training resolution.
