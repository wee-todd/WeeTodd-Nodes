# H3 resolution selection

The `WeeTodd H3 Generation Config` node provides one authoritative canvas configuration. Select a
quality tier and an aspect ratio. Use custom mode only when exact dimensions are required.

## Quality tiers

| Tier | Intended use | Widescreen canvas |
| --- | --- | --- |
| `384P (fast smoke)` | Fast wiring and prompt checks | 672 by 384 |
| `512P (balanced)` | Intermediate quality and cost | 896 by 512 |
| `768P (native quality)` | Released H3 native-quality geometry | 1344 by 768 |
| `2K (experimental, very high memory)` | Explicit high-cost experiment | 2048 by 1152 |

The experimental 2K tier exceeds the initial H3 native short-edge geometry. Dense attention makes
the tier substantially more expensive. Run Preflight before generation.

## Aspect ratios

The node provides `21:9`, `16:9`, `5:3`, `3:2`, `4:3`, `1:1`, `3:4`, `2:3`, `3:5`, `9:16`, and
`9:21`. The resolver places the selected canvas on H3's required 32-pixel grid.

## Custom mode

Select `custom` to use the advanced width and height controls. Both dimensions must be divisible by
32. Preflight consumes the generation configuration directly, so memory estimates and generation
use the same resolved canvas.
