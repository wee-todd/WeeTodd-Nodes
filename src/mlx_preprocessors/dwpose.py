"""MLX DWPose whole-body preprocessing from converted standard ONNX bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .onnx_graph import MLXONNXGraph

BODY_EDGES = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
HAND_EDGES = tuple(
    edge
    for finger in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                   (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for edge in zip(finger[:-1], finger[1:], strict=True)
)
COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)


@dataclass(frozen=True)
class DWPoseConfig:
    detection_threshold: float = 0.3
    keypoint_threshold: float = 0.3
    render_mode: str = "whole body"

    def validate(self):
        if not 0.01 <= self.detection_threshold <= 0.99:
            raise ValueError("DWPose detection threshold must be between 0.01 and 0.99.")
        if not 0.01 <= self.keypoint_threshold <= 0.99:
            raise ValueError("DWPose keypoint threshold must be between 0.01 and 0.99.")
        if self.render_mode not in {"whole body", "body and hands", "body only"}:
            raise ValueError("DWPose render mode is not supported.")


class DWPoseMLX:
    def __init__(self, detector_bundle, pose_bundle):
        self.detector_bundle = detector_bundle
        self.pose_bundle = pose_bundle
        self.detector = None
        self.pose = None

    def load_detector(self):
        if self.detector is None:
            self.detector = MLXONNXGraph(self.detector_bundle)
        return self.detector

    def load_pose(self):
        if self.pose is None:
            self.pose = MLXONNXGraph(self.pose_bundle)
        return self.pose

    def unload_detector(self):
        import gc

        import mlx.core as mx

        self.detector = None
        gc.collect()
        mx.clear_cache()

    def unload_pose(self):
        import gc

        import mlx.core as mx

        self.pose = None
        gc.collect()
        mx.clear_cache()


def _as_rgb_uint8(images: Any):
    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    value = np.asarray(images)
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError("DWPose requires a ComfyUI IMAGE frame batch.")
    value = value[..., :3]
    if value.dtype != np.uint8:
        value = np.clip(value.astype(np.float32), 0, 1) * 255
    return np.ascontiguousarray(value.astype(np.uint8))


def _letterbox(frame, size=640):
    ratio = min(size / frame.shape[0], size / frame.shape[1])
    resized = cv2.resize(
        frame,
        (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)[None], ratio


def _nms(boxes, scores, threshold):
    if not len(boxes):
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        current = order[0]
        keep.append(current)
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])
        intersection = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        union = areas[current] + areas[order[1:]] - intersection
        overlap = intersection / np.maximum(union, 1e-6)
        order = order[1:][overlap <= threshold]
    return np.asarray(keep, dtype=np.int64)


def _detect_people(frame, graph, score_threshold):
    prepared, ratio = _letterbox(frame)
    output = np.asarray(graph(images=prepared)[0])[0]
    strides = (8, 16, 32)
    grids = []
    expanded = []
    for stride in strides:
        height = width = 640 // stride
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((x, y), axis=-1).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), stride))
    grid = np.concatenate(grids)[None]
    stride = np.concatenate(expanded)[None]
    centers = (output[None, :, :2] + grid) * stride
    sizes = np.exp(output[None, :, 2:4]) * stride
    scores = output[:, 4] * output[:, 5]
    valid = scores >= score_threshold
    centers = centers[0, valid]
    sizes = sizes[0, valid]
    scores = scores[valid]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, frame.shape[1] - 1)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, frame.shape[0] - 1)
    return boxes[_nms(boxes, scores, 0.45)]


def _affine_crop(frame, box):
    x1, y1, x2, y2 = box
    center = np.array(((x1 + x2) / 2, (y1 + y2) / 2), dtype=np.float32)
    scale = np.array((x2 - x1, y2 - y1), dtype=np.float32) * 1.25
    aspect = 288 / 384
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    source = np.array(
        (
            center,
            center + (0, -scale[0] / 2),
            center + (-scale[0] / 2, 0),
        ),
        dtype=np.float32,
    )
    destination = np.array(((144, 192), (144, 48), (0, 192)), dtype=np.float32)
    matrix = cv2.getAffineTransform(source, destination)
    crop = cv2.warpAffine(frame, matrix, (288, 384), flags=cv2.INTER_LINEAR)
    mean = np.array((123.675, 116.28, 103.53), dtype=np.float32)
    deviation = np.array((58.395, 57.12, 57.375), dtype=np.float32)
    crop = (crop.astype(np.float32) - mean) / deviation
    return np.ascontiguousarray(crop.transpose(2, 0, 1))[None], center, scale


def _decode_pose(output_x, output_y, center, scale):
    x = np.asarray(output_x)[0]
    y = np.asarray(output_y)[0]
    x_index = np.argmax(x, axis=1).astype(np.float32)
    y_index = np.argmax(y, axis=1).astype(np.float32)
    score = np.minimum(np.max(x, axis=1), np.max(y, axis=1))
    points = np.stack((x_index, y_index), axis=1) / 2
    points = points / np.array((288, 384), dtype=np.float32) * scale + center - scale / 2
    return points, score


def _openpose_order(points, scores):
    neck = np.mean(points[[5, 6]], axis=0)
    neck_score = min(scores[5], scores[6])
    points = np.insert(points, 17, neck, axis=0)
    scores = np.insert(scores, 17, neck_score, axis=0)
    source = (17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3)
    target = (1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17)
    reordered_points = points.copy()
    reordered_scores = scores.copy()
    reordered_points[list(target)] = points[list(source)]
    reordered_scores[list(target)] = scores[list(source)]
    return reordered_points, reordered_scores


def _draw_pose(height, width, people, threshold, mode):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    thickness = max(2, round(min(height, width) / 160))
    radius = max(2, round(min(height, width) / 128))
    for points, scores in people:
        points, scores = _openpose_order(points, scores)
        for index, (start, end) in enumerate(BODY_EDGES):
            if scores[start] >= threshold and scores[end] >= threshold:
                cv2.line(
                    canvas,
                    tuple(np.rint(points[start]).astype(int)),
                    tuple(np.rint(points[end]).astype(int)),
                    COLORS[index],
                    thickness,
                    cv2.LINE_AA,
                )
        for index in range(18):
            if scores[index] >= threshold:
                cv2.circle(
                    canvas,
                    tuple(np.rint(points[index]).astype(int)),
                    radius,
                    COLORS[index],
                    -1,
                    cv2.LINE_AA,
                )
        if mode in {"whole body", "body and hands"}:
            for hand_number, hand in enumerate((points[92:113], points[113:134])):
                hand_scores = scores[92 + hand_number * 21 : 113 + hand_number * 21]
                for index, (start, end) in enumerate(HAND_EDGES):
                    if hand_scores[start] >= threshold and hand_scores[end] >= threshold:
                        color = COLORS[index % len(COLORS)]
                        cv2.line(
                            canvas,
                            tuple(np.rint(hand[start]).astype(int)),
                            tuple(np.rint(hand[end]).astype(int)),
                            color,
                            max(1, thickness - 1),
                            cv2.LINE_AA,
                        )
        if mode == "whole body":
            for point, score in zip(points[24:92], scores[24:92], strict=True):
                if score >= threshold:
                    cv2.circle(
                        canvas,
                        tuple(np.rint(point).astype(int)),
                        max(1, radius - 1),
                        (255, 255, 255),
                        -1,
                    )
    return canvas.astype(np.float32) / 255


def infer_dwpose(
    images,
    model: DWPoseMLX,
    config: DWPoseConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    config = config or DWPoseConfig()
    config.validate()
    frames = _as_rgb_uint8(images)
    detector = model.load_detector()
    boxes_per_frame = []
    try:
        for frame_index, frame in enumerate(frames):
            if interruption_callback is not None:
                interruption_callback()
            boxes_per_frame.append(
                _detect_people(frame, detector, config.detection_threshold)
            )
            if progress_callback is not None:
                progress_callback(frame_index + 1, len(frames) * 2)
    finally:
        model.unload_detector()

    pose_model = model.load_pose()
    output = []
    people_per_frame = []
    for frame_index, (frame, boxes) in enumerate(
        zip(frames, boxes_per_frame, strict=True)
    ):
        if interruption_callback is not None:
            interruption_callback()
        people = []
        for box in boxes:
            crop, center, scale = _affine_crop(frame, box)
            output_x, output_y = pose_model(input=crop)
            people.append(_decode_pose(output_x, output_y, center, scale))
        output.append(
            _draw_pose(
                frame.shape[0],
                frame.shape[1],
                people,
                config.keypoint_threshold,
                config.render_mode,
            )
        )
        people_per_frame.append(len(people))
        if progress_callback is not None:
            progress_callback(len(frames) + frame_index + 1, len(frames) * 2)
    return np.stack(output).astype(np.float32), {
        "backend": "mlx",
        "algorithm": "dwpose_wholebody",
        "frames": len(frames),
        "render_mode": config.render_mode,
        "detection_threshold": config.detection_threshold,
        "keypoint_threshold": config.keypoint_threshold,
        "people_min": min(people_per_frame, default=0),
        "people_max": max(people_per_frame, default=0),
        "people_mean": float(np.mean(people_per_frame)) if people_per_frame else 0.0,
    }
