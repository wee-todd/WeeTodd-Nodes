# H3 EasyCache policy and step scaling at 768P

## Conditions

The benchmark measured no EasyCache, conservative auto, balanced auto, and speed auto at 8, 12,
16, and 20 requested steps. Every generation used five seconds, 1344 by 768 pixels, seed zero, the
same H3 audiovisual prompt, the BF16 pruned transformer, the Q8 text encoder, and the same final
video and audio VAEs. Each weighted stage unloaded after use.

The sampling time comes from each generation sidecar. The complete workflow time comes from
ComfyUI execution timestamps. The AdaLN cache size comes from the execution log. The queue ran in
step-major order: no cache, conservative, balanced, then speed.

## Prompt

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide eye-level shot frames a small red wind-up robot standing on a worn wooden workbench in a warm, sunlit repair workshop. The robot begins walking from left to right with clear alternating steps as its silver key turns, its arms swing, and its head pivots toward the camera. The camera trucks right with small amplitude at slow speed to follow the robot. Near the edge of the workbench, the robot stops, plants both feet, raises its right hand, and gives one deliberate wave while holding a stable final pose.

overall_soundscape: Quiet workshop room tone continues beneath the scene. Each metal footstep makes a light tap on the wood, the winding mechanism clicks steadily in sync with the turning key, and the robot’s raised arm ends with a soft mechanical stop.

non_diegetic_music: N/A
```

## Runtime results

| Steps | Policy | Evaluations | Cached | Sampling | Workflow | Sampling saving |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | None | 7 | 0 | 1322.97 s | 1448.11 s | — |
| 8 | Conservative | 6 | 1 | 1001.85 s | 1111.53 s | 24.3% |
| 8 | Balanced | 5 | 2 | 810.47 s | 918.39 s | 38.7% |
| 8 | Speed | 4 | 3 | 666.57 s | 776.06 s | 49.6% |
| 12 | None | 11 | 0 | 1818.87 s | 1928.98 s | — |
| 12 | Conservative | 9 | 2 | 1491.06 s | 1598.02 s | 18.0% |
| 12 | Balanced | 8 | 3 | 1329.48 s | 1435.85 s | 26.9% |
| 12 | Speed | 6 | 5 | 984.29 s | 1093.20 s | 45.9% |
| 16 | None | 15 | 0 | 2480.15 s | 2590.66 s | — |
| 16 | Conservative | 12 | 3 | 1980.13 s | 2088.79 s | 20.2% |
| 16 | Balanced | 10 | 5 | 1643.93 s | 1753.07 s | 33.7% |
| 16 | Speed | 8 | 7 | 1317.57 s | 1434.05 s | 46.9% |
| 20 | None | 19 | 0 | 3139.98 s | 3256.57 s | — |
| 20 | Conservative | 15 | 4 | 2494.27 s | 2604.63 s | 20.6% |
| 20 | Balanced | 13 | 6 | 2175.15 s | 2281.12 s | 30.7% |
| 20 | Speed | 10 | 9 | 1668.15 s | 1776.75 s | 46.9% |

![768P EasyCache policy and step-scaling graph](../../benchmarks/artifacts/charts/h3_easycache_policy_step_scaling_768p.svg)

## Scaling fits

| Policy | Sampling seconds per requested step | Sampling R-squared | Workflow seconds per requested step | Workflow R-squared |
| --- | ---: | ---: | ---: | ---: |
| None | 152.81 | 0.996 | 152.18 | 0.995 |
| Conservative | 124.16 | 1.000 | 124.25 | 1.000 |
| Balanced | 110.21 | 0.991 | 110.14 | 0.991 |
| Speed | 83.45 | 1.000 | 83.57 | 1.000 |

## Findings

After the first cold run, one complete transformer evaluation took approximately 162 through 171
seconds. Transformer evaluation count explained most runtime scaling. At 20 requested steps,
conservative saved 20.6 percent of sampling time, balanced saved 30.7 percent, and speed saved 46.9
percent.

Complete-workflow savings at 20 steps were 20.0 percent for conservative, 30.0 percent for
balanced, and 45.4 percent for speed. Text encoding, component loading, VAE decoding, and output
publication do not benefit from EasyCache.

The measured AdaLN schedule cache depended on the schedule, not the canvas. It increased from 126
MB at 8 steps to 358 MB at 20 steps.

## Limitations

Each point is one run with one prompt, seed, and duration. Queue order and system temperature were
not randomized. The uncached eight-step run was the first native-resolution run and had a higher
per-evaluation time than later runs. The benchmark measures runtime, not perceptual quality. Do not
infer motion, detail, or audio parity from timing alone.
