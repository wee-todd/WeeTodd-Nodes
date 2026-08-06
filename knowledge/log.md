# Knowledge update log

## 2026-08-06

- **Implementation**: Added independent MLX H3 BlockCache with block-zero probing, separate audio
  and video scores, bounded automatic policies, current output heads, metadata, and request cleanup.
- **Experiment**: Started the 640 by 384 BlockCache policy and sampling-step matrix with the
  established prompt, seed, component identities, and staged unloading contract.
- **Experiment**: Completed the 12-point BlockCache matrix. Speed auto reduced 20-step sampling
  time by 45.9 percent and used a 95.3 MiB request-local cache.
- **Experiment**: Completed and recorded the 16-point 1344 by 768 EasyCache matrix. All runs
  produced synchronized MP4 and JSON artifacts without a generation failure.
- **Evidence**: Added native-resolution runtime data, scaling fits, memory measurements, endpoint
  inspection, canonical charts, and a local gitignored media bundle.
- **Documentation**: Added the EasyCache benchmark results and cross-resolution cost comparison to
  the project README.

## 2026-08-05

- **Knowledge**: Recorded the complete 16-point EasyCache matrix, shared benchmark conditions,
  memory measurements, artifact checks, and project-validation results in the H3 EasyCache concept.
- **Experiment**: Completed a 16-point EasyCache scaling matrix across no cache, conservative,
  balanced, and speed policies at 8, 12, 16, and 20 requested steps.
- **Experiment**: Measured balanced EasyCache at 8, 12, 16, and 20 requested steps. Sampling time
  increased approximately 16.20 seconds per requested step while 27 to 33 percent were cached.
- **Experiment**: Added and ran balanced automatic H3 EasyCache. The seed-matched run skipped two
  of seven evaluations and reduced sampling from 165.82 to 116.58 seconds.
- **Implementation**: Added preset-first H3 resolution tiers, eleven aspect ratios, advanced custom
  dimensions, resolved-canvas metadata, and one shared configuration for preflight and generation.
- **Experiment**: Added conservative and speed automatic H3 EasyCache policies. The seed-matched
  speed run skipped three of seven evaluations and reduced sampling from 165.82 to 100.15 seconds.
- **Experiment**: Added joint MLX H3 EasyCache, registered node 16, and recorded a zero-skip,
  byte-identical first calibration at the official threshold.
- **Creation**: Added the official MiniMax H3 audiovisual prompting contract and a reusable
  project-local prompting skill.
- **Validation**: Loaded the minimal eight-node T2VA workflow in clean ComfyUI 0.30.0, confirmed
  fifteen registered nodes, and passed live graph-schema validation without weights.
- **Validation**: Passed isolated strict loader probes for the selected transformer, Q8 text
  encoder, video VAE, and audio VAE. No inference forward or generation ran.
- **Implementation**: Scoped compact text-encoder memory accounting, resolved compact Qwen3-VL
  architecture provenance, and allowed tokenizer-only processor assets for text-only T2VA.
- **Policy**: Made staged weighted-component residency a full-pipeline invariant with unload-by-
  default behavior, explicit keep-warm controls, resident-state reporting, and failure cleanup.
- **Implementation**: Added synchronized ComfyUI image and audio validation, atomic MP4 and JSON
  publication, collision-safe naming, timing metadata, and partial-file cleanup.
- **Decision**: Selected the unquantized BF16-class reference policy for the first T2VA smoke test
  while preserving audited FP32 stability exceptions.
- **Creation**: Added output-publication and BF16-baseline concepts.
- **Implementation**: Added staged final audio VAE decoding, ComfyUI audio output, synchronized
  timing metadata, sample-rate and provenance checks, cancellation cleanup, and explicit unloading.
- **Creation**: Added the audio VAE decoding contract.
- **Policy**: Added deterministic Python environment preflight, incompatible-venv recovery rules,
  and CI enforcement before dependency installation.
- **Creation**: Added the Python environment preflight procedure and project-local skill.
- **Implementation**: Added staged final video VAE decoding, ComfyUI image output, component
  provenance checks, failure cleanup, and explicit unloading.
- **Creation**: Added the video VAE decoding contract.
- **Implementation**: Added transformer-only synchronized sampling, evaluation progress,
  cancellation callbacks, schedule-safe reuse, failure cleanup, latent output, and explicit unload.
- **Creation**: Added the transformer sampling contract.
- **Implementation**: Added text-only Qwen3-VL conditioning, independent processor and tokenizer
  paths, process-local reuse, unload-after-encode, failure cleanup, and explicit unloading.
- **Implementation**: Added lazy component specification and header-only preflight nodes with task,
  manifest, shard, configuration, quantization, and staged-memory validation.
- **Creation**: Added the ComfyUI H3 ecosystem, complete component stack, and first smoke-test
  concepts.
- **Update**: Defined abbreviations, separated contract and decision semantics, and marked evolving
  plan and optimization concepts as draft.
- **Update**: Applied the strict controlled-summary profile to OKF concepts and restructured the
  optimization research concept by claim, evidence, limitation, and required validation.
- **Update**: Adopted a selective controlled-English profile for procedures, user-facing text,
  research, and architecture documentation.
- **Creation**: Added the documentation-standard concept.
- **Initialization**: Created the OKF v0.2 knowledge bundle.
- **Creation**: Indexed the engine contract, technical audit, implementation plan, and optimization research.
