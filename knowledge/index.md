---
okf_version: "0.2"
---

# WeeTodd H3 knowledge

## Contracts

- [Engine contract](engine-contract.md) - Stable invariants for synchronized MiniMax H3 inference.
- [H3 component stack](h3-component-stack.md) - Required weighted and unweighted pipeline components.
- [Transformer sampling contract](transformer-sampling-contract.md) - Synchronized latent sampling and lifecycle rules.
- [Video VAE decoding contract](video-vae-decoding-contract.md) - Final frame decoding, provenance, and lifecycle rules.
- [Audio VAE decoding contract](audio-vae-decoding-contract.md) - Stereo waveform decoding, timing, provenance, and lifecycle rules.
- [Synchronized output publication](synchronized-output-publication.md) - Final validation, atomic media output, metadata, and cleanup.
- [Documentation standard](documentation-standard.md) - Selective controlled-English and Markdown rules for project writing.
- [Python environment preflight](python-environment-preflight.md) - Required checks before Python, venv, dependency, or MLX changes.
- [H3 resolution selection](h3-resolution-selection.md) - Preset tiers, aspect ratios, custom dimensions, and the single-source canvas contract.

## Decisions and plans

- [Technical audit](technical-audit.md) - Ranked scaffold findings and foundational decisions.
- [Implementation plan](implementation-plan.md) - Prioritized path to composable H3 nodes.
- [First ComfyUI smoke test](first-comfy-smoke-test.md) - Gated path to the first MLX-backed generation.
- [BF16-class first T2VA baseline](bf16-baseline-policy.md) - Unquantized baseline with audited FP32 stability exceptions.

## Research

- [H3 EasyCache](h3-easycache.md) - Joint MLX audiovisual residual reuse and first calibration.
- [H3 BlockCache](h3-blockcache.md) - Block-level joint audiovisual residual reuse and calibration.
- [MiniMax H3 prompting contract](h3-prompting-contract.md) - Required audiovisual prompt structure and disclosure rule.
- [Apple Silicon optimization research](optimization-research.md) - Quantization and attention findings.
- [ComfyUI H3 ecosystem](comfyui-h3-ecosystem.md) - Native, Kijai, and Spectrum design references.
