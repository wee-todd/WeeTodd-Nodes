"""H3-native spatial latent resizing for a second visual refinement pass."""

from __future__ import annotations

import math

import mlx.core as mx

LATENT_RESIZE_METHODS = ("nearest exact", "bilinear", "bicubic", "lanczos-3")


def resolve_hires_canvas(
    width: int,
    height: int,
    scale: float,
    *,
    multiple: int = 32,
    max_dimension: int = 1920,
    max_short_edge: int = 1088,
) -> tuple[int, int]:
    """Resolve the closest H3-compatible canvas to ``scale`` times the source."""
    if width < multiple or height < multiple or width % multiple or height % multiple:
        raise ValueError("H3 Hi Res Fix source dimensions must be positive multiples of 32.")
    if not math.isfinite(scale) or not 1.0 < scale <= 2.0:
        raise ValueError("H3 Hi Res Fix scale must be greater than 1 and no more than 2.")

    target_width = max(multiple, int(math.floor(width * scale / multiple + 0.5)) * multiple)
    target_height = max(multiple, int(math.floor(height * scale / multiple + 0.5)) * multiple)
    if max(target_width, target_height) > max_dimension:
        raise ValueError(
            f"H3 Hi Res Fix resolves to {target_width}x{target_height}, exceeding the "
            f"{max_dimension}-pixel axis limit."
        )
    if min(target_width, target_height) > max_short_edge:
        raise ValueError(
            f"H3 Hi Res Fix resolves to a {min(target_width, target_height)}-pixel short edge, "
            f"exceeding the {max_short_edge}-pixel limit."
        )
    return target_width, target_height


def _validate_resize_request(
    latents: mx.array,
    target_height: int,
    target_width: int,
) -> tuple[int, int]:
    if latents.ndim != 5:
        raise ValueError("H3 video latents must have shape (batch, channels, time, height, width).")
    source_height = int(latents.shape[3])
    source_width = int(latents.shape[4])
    if target_height < 1 or target_width < 1:
        raise ValueError("H3 target latent dimensions must be positive.")
    return source_height, source_width


def _half_pixel_positions(source: int, target: int) -> mx.array:
    return (mx.arange(target, dtype=mx.float32) + 0.5) * (source / target) - 0.5


def resize_video_latents_nearest_exact(
    latents: mx.array,
    target_height: int,
    target_width: int,
) -> mx.array:
    """Resize H3 video latents with center-aligned nearest-neighbor selection."""
    source_height, source_width = _validate_resize_request(latents, target_height, target_width)
    if target_height == source_height and target_width == source_width:
        return latents

    y = mx.floor(
        (mx.arange(target_height, dtype=mx.float32) + 0.5) * (source_height / target_height)
    ).astype(mx.int32)
    x = mx.floor(
        (mx.arange(target_width, dtype=mx.float32) + 0.5) * (source_width / target_width)
    ).astype(mx.int32)
    resized = mx.take(latents, mx.clip(y, 0, source_height - 1), axis=3)
    resized = mx.take(resized, mx.clip(x, 0, source_width - 1), axis=4)
    return resized.astype(latents.dtype)


def resize_video_latents_bilinear(
    latents: mx.array,
    target_height: int,
    target_width: int,
) -> mx.array:
    """Resize H3 video latents with half-pixel-centered bilinear interpolation."""
    source_height, source_width = _validate_resize_request(latents, target_height, target_width)
    if target_height == source_height and target_width == source_width:
        return latents

    def coordinates(source: int, target: int):
        position = _half_pixel_positions(source, target)
        lower = mx.floor(position).astype(mx.int32)
        upper = lower + 1
        weight = position - lower.astype(mx.float32)
        return mx.clip(lower, 0, source - 1), mx.clip(upper, 0, source - 1), weight

    y0, y1, yw = coordinates(source_height, target_height)
    top = mx.take(latents, y0, axis=3)
    bottom = mx.take(latents, y1, axis=3)
    resized_y = top + (bottom - top) * yw.reshape(1, 1, 1, target_height, 1)

    x0, x1, xw = coordinates(source_width, target_width)
    left = mx.take(resized_y, x0, axis=4)
    right = mx.take(resized_y, x1, axis=4)
    resized = left + (right - left) * xw.reshape(1, 1, 1, 1, target_width)
    return resized.astype(latents.dtype)


def _cubic_kernel(distance: mx.array, *, coefficient: float = -0.5) -> mx.array:
    absolute = mx.abs(distance)
    inside_one = (coefficient + 2.0) * absolute**3 - (coefficient + 3.0) * absolute**2 + 1.0
    inside_two = (
        coefficient * absolute**3
        - 5.0 * coefficient * absolute**2
        + 8.0 * coefficient * absolute
        - 4.0 * coefficient
    )
    return mx.where(
        absolute <= 1.0,
        inside_one,
        mx.where(absolute < 2.0, inside_two, mx.zeros_like(absolute)),
    )


def _resize_axis_bicubic(
    latents: mx.array,
    *,
    axis: int,
    source: int,
    target: int,
) -> mx.array:
    positions = _half_pixel_positions(source, target)
    base = mx.floor(positions).astype(mx.int32)
    result = None
    for offset in (-1, 0, 1, 2):
        indices = mx.clip(base + offset, 0, source - 1)
        distance = positions - (base + offset).astype(mx.float32)
        weights = _cubic_kernel(distance)
        shape = [1] * latents.ndim
        shape[axis] = target
        contribution = mx.take(latents, indices, axis=axis) * weights.reshape(*shape)
        result = contribution if result is None else result + contribution
    return result


def resize_video_latents_bicubic(
    latents: mx.array,
    target_height: int,
    target_width: int,
) -> mx.array:
    """Resize H3 video latents with separable Catmull-Rom bicubic interpolation."""
    source_height, source_width = _validate_resize_request(latents, target_height, target_width)
    if target_height == source_height and target_width == source_width:
        return latents

    resized = _resize_axis_bicubic(
        latents,
        axis=3,
        source=source_height,
        target=target_height,
    )
    resized = _resize_axis_bicubic(
        resized,
        axis=4,
        source=source_width,
        target=target_width,
    )
    return resized.astype(latents.dtype)


def _normalized_sinc(value: mx.array) -> mx.array:
    pi_value = math.pi * value
    near_zero = mx.abs(value) < 1e-7
    denominator = mx.where(near_zero, mx.ones_like(pi_value), pi_value)
    result = mx.sin(pi_value) / denominator
    return mx.where(near_zero, mx.ones_like(result), result)


def _lanczos_kernel(distance: mx.array, *, radius: int) -> mx.array:
    absolute = mx.abs(distance)
    windowed = _normalized_sinc(distance) * _normalized_sinc(distance / radius)
    return mx.where(absolute < radius, windowed, mx.zeros_like(windowed))


def _resize_axis_lanczos(
    latents: mx.array,
    *,
    axis: int,
    source: int,
    target: int,
    radius: int,
) -> mx.array:
    positions = _half_pixel_positions(source, target)
    base = mx.floor(positions).astype(mx.int32)
    offsets = tuple(range(-(radius - 1), radius + 1))
    raw_weights = tuple(
        _lanczos_kernel(
            positions - (base + offset).astype(mx.float32),
            radius=radius,
        )
        for offset in offsets
    )
    weight_sum = raw_weights[0]
    for weights in raw_weights[1:]:
        weight_sum = weight_sum + weights

    shape = [1] * latents.ndim
    shape[axis] = target
    result = None
    for offset, raw_weight in zip(offsets, raw_weights, strict=True):
        indices = mx.clip(base + offset, 0, source - 1)
        weights = raw_weight / weight_sum
        contribution = mx.take(latents, indices, axis=axis) * weights.reshape(*shape)
        result = contribution if result is None else result + contribution
    return result


def resize_video_latents_lanczos(
    latents: mx.array,
    target_height: int,
    target_width: int,
    *,
    radius: int = 3,
) -> mx.array:
    """Resize H3 video latents with normalized separable Lanczos interpolation."""
    source_height, source_width = _validate_resize_request(latents, target_height, target_width)
    if radius < 2:
        raise ValueError("H3 Lanczos resize radius must be at least 2.")
    if target_height == source_height and target_width == source_width:
        return latents

    resized = _resize_axis_lanczos(
        latents,
        axis=3,
        source=source_height,
        target=target_height,
        radius=radius,
    )
    # Materialize the smaller height pass before constructing the wider six-tap graph. This keeps
    # the one-time interpolation peak bounded without a host transfer.
    mx.eval(resized)
    resized = _resize_axis_lanczos(
        resized,
        axis=4,
        source=source_width,
        target=target_width,
        radius=radius,
    )
    return resized.astype(latents.dtype)


def resize_video_latents(
    latents: mx.array,
    target_height: int,
    target_width: int,
    *,
    method: str = "bilinear",
) -> mx.array:
    """Resize ``(B,C,T,H,W)`` H3 latents with an MLX-native spatial method."""
    methods = {
        "nearest exact": resize_video_latents_nearest_exact,
        "bilinear": resize_video_latents_bilinear,
        "bicubic": resize_video_latents_bicubic,
        "lanczos-3": resize_video_latents_lanczos,
    }
    try:
        resize = methods[method]
    except KeyError as error:
        supported = ", ".join(LATENT_RESIZE_METHODS)
        raise ValueError(
            f"Unsupported H3 latent resize method {method!r}. Select one of: {supported}."
        ) from error
    return resize(latents, target_height, target_width)
