# H3 EasyCache design and first calibration

## Design

ComfyUI EasyCache estimates how much the diffusion-model output changes relative to its input. The
node reuses the previous output residual when the accumulated estimate stays below a threshold.

WeeTodd implements the method independently in MLX. H3 jointly predicts video and audio with
different sigma schedules. The H3 implementation therefore stores both residuals, computes one
joint change estimate, and advances both schedulers when it skips a transformer evaluation.

The composable `WeeTodd H3 EasyCache (MLX)` node supplies the threshold and sampling window to the
H3 sampler. Cached state exists for one sampling request and is released with the transformer.

## Automatic policies

All automatic policies use the requested sampling-step count to protect two calibration
evaluations and the final evaluation. They derive a bounded threshold from the live joint
video/audio change estimate after calibration. The metadata records the selected policy, resolved
threshold, and actual skipped count.

`automatic_conservative` bounds the threshold between 0.05 and 0.50, permits no consecutive skips,
and uses the configured skip cap. `automatic_balanced` bounds the threshold between 0.10 and 0.80,
uses at least a 1.40 calibration multiplier, permits no consecutive skips, and permits up to 35
percent of scheduled evaluations to be skipped. `automatic_speed` bounds the threshold between
0.25 and 1.25, uses at least a 1.75 calibration multiplier, permits at most two consecutive skips,
and permits up to 50 percent of scheduled evaluations to be skipped. The speed policy is
intentionally more likely to change motion, detail, and synchronized sound.

No automatic policy is a quality guarantee. Resolution and duration do not determine a safe
threshold by themselves because the prompt and latent trajectory also affect the change rate.

## First calibration

The first real test used five seconds, 640 by 384 pixels, eight sampling steps, seed zero, threshold
0.20, start 0.15, and end 0.95. The sampler executed all seven transformer evaluations and skipped
zero. Total workflow time was 194.21 seconds. The published MP4 was byte-identical to the uncached
baseline.

The official default is lossless for this short schedule because no reuse occurred. The result does
not establish a useful H3 threshold. Test higher thresholds and longer schedules against the exact
baseline before changing the default.

## Automatic policy calibration

The conservative policy used the same five-second, 640 by 384, eight-step, seed-zero request. The
policy skipped one of seven transformer evaluations. Sampling took 141.00 seconds, and the complete
workflow took 174.74 seconds. The resolved threshold reached the conservative ceiling of 0.50.

The balanced policy used the same request and skipped two of seven transformer evaluations.
Sampling took 116.58 seconds, and the complete workflow took 161.14 seconds. The resolved threshold
reached the balanced ceiling of 0.80. The cached evaluations were sampling steps 4 and 6.

The speed policy used the same request and skipped three of seven transformer evaluations. Sampling
took 100.15 seconds, and the complete workflow took 128.94 seconds. The resolved threshold reached
the speed ceiling of 1.25. The cached evaluations were sampling steps 3, 5, and 6.

The balanced and speed contact sheets showed the requested walk, turn, stop, and wave sequence.
Visual inspection does not prove motion or audio parity. Keep both policies opt-in until repeatable
audiovisual quality metrics and more prompts establish an acceptable quality range.

## Source boundary

ComfyUI's experimental native EasyCache node was an algorithm reference. WeeTodd does not copy the
ComfyUI implementation. WeeTodd preserves the MLX engine boundary and H3's joint audiovisual
contract.

## Source

- [ComfyUI EasyCache implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_easycache.py)
