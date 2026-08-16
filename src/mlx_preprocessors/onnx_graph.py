"""Small MLX executor for converted, standard-operator preprocessor graphs."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.numpy import load_file


def _pair(value, default):
    value = value if value is not None else default
    return tuple(int(item) for item in value)


def _scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


class MLXONNXGraph:
    """Execute the limited ONNX operator set used by supported preprocessors."""

    def __init__(self, bundle: str | Path):
        bundle = Path(bundle)
        self.graph = json.loads((bundle / "graph.json").read_text(encoding="utf-8"))
        if self.graph.get("format") != "weetodd-onnx-mlx-v1":
            raise ValueError(f"Unsupported converted ONNX bundle: {bundle}")
        self.weights = {
            name: mx.array(value)
            for name, value in load_file(bundle / "weights.safetensors").items()
        }

    def __call__(self, **inputs):
        missing = set(self.graph["inputs"]) - set(inputs)
        if missing:
            raise ValueError(f"Missing converted ONNX inputs: {sorted(missing)}")
        values = dict(self.weights)
        values.update({name: mx.array(value) for name, value in inputs.items()})
        for node in self.graph["nodes"]:
            args = [values[name] if name else None for name in node["inputs"]]
            outputs = self._execute(node["op"], args, node["attributes"])
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            if len(outputs) != len(node["outputs"]):
                raise RuntimeError(
                    f"{node['name']} returned {len(outputs)} values for "
                    f"{len(node['outputs'])} outputs."
                )
            values.update(zip(node["outputs"], outputs, strict=True))
        result = tuple(values[name] for name in self.graph["outputs"])
        mx.eval(*result)
        return result

    def _attribute(self, value):
        if isinstance(value, dict) and "tensor" in value:
            return self.weights[value["tensor"]]
        return value

    def _execute(self, operation, args, raw_attributes):
        attributes = {name: self._attribute(value) for name, value in raw_attributes.items()}
        if operation == "Identity":
            return args[0]
        if operation == "Constant":
            return attributes["value"]
        if operation == "Sigmoid":
            return mx.sigmoid(args[0])
        if operation == "Relu":
            return mx.maximum(args[0], 0)
        if operation == "Mul":
            return args[0] * args[1]
        if operation == "Add":
            return args[0] + args[1]
        if operation == "Div":
            return args[0] / args[1]
        if operation == "Sqrt":
            return mx.sqrt(args[0])
        if operation == "MatMul":
            return mx.matmul(args[0], args[1])
        if operation == "Concat":
            return mx.concatenate(args, axis=int(attributes.get("axis", 0)))
        if operation == "Transpose":
            permutation = attributes.get("perm", range(args[0].ndim - 1, -1, -1))
            return mx.transpose(args[0], tuple(permutation))
        if operation == "Shape":
            return mx.array(args[0].shape, dtype=mx.int64)
        if operation == "Reshape":
            shape = tuple(int(item) for item in np.asarray(args[1]).tolist())
            return mx.reshape(args[0], shape)
        if operation == "Squeeze":
            axes = attributes.get("axes")
            if axes is None:
                axes = np.asarray(args[1]).tolist()
            axes = tuple(int(item) for item in axes)
            return mx.squeeze(args[0], axis=axes)
        if operation == "Unsqueeze":
            axes = attributes.get("axes")
            if axes is None:
                axes = np.asarray(args[1]).tolist()
            axes = sorted(int(item) for item in axes)
            output = args[0]
            for axis in axes:
                output = mx.expand_dims(output, axis)
            return output
        if operation == "Split":
            axis = int(attributes.get("axis", 0))
            sections = attributes.get("split")
            if sections is None and len(args) > 1 and args[1] is not None:
                sections = np.asarray(args[1]).tolist()
            if sections is None:
                return tuple(mx.split(args[0], len(raw_attributes), axis=axis))
            indices = np.cumsum(sections)[:-1].tolist()
            return tuple(mx.split(args[0], indices, axis=axis))
        if operation == "ReduceSum":
            axes = attributes.get("axes")
            if axes is None and len(args) > 1 and args[1] is not None:
                axes = np.asarray(args[1]).tolist()
            return mx.sum(
                args[0],
                axis=None if axes is None else tuple(int(item) for item in axes),
                keepdims=bool(attributes.get("keepdims", 1)),
            )
        if operation == "GlobalAveragePool":
            return mx.mean(args[0], axis=tuple(range(2, args[0].ndim)), keepdims=True)
        if operation == "HardSigmoid":
            alpha = float(attributes.get("alpha", 0.2))
            beta = float(attributes.get("beta", 0.5))
            return mx.clip(args[0] * alpha + beta, 0, 1)
        if operation == "Clip":
            minimum = args[1] if len(args) > 1 and args[1] is not None else attributes.get("min")
            maximum = args[2] if len(args) > 2 and args[2] is not None else attributes.get("max")
            return mx.clip(args[0], _scalar(minimum), _scalar(maximum))
        if operation == "Conv":
            return self._conv(args, attributes)
        if operation == "MaxPool":
            return self._max_pool(args[0], attributes)
        if operation == "Resize":
            return self._resize(args, attributes)
        if operation == "Slice":
            return self._slice(args, attributes)
        raise NotImplementedError(f"Converted ONNX operator {operation!r} is not supported.")

    @staticmethod
    def _conv(args, attributes):
        value = mx.transpose(args[0], (0, 2, 3, 1))
        weight = mx.transpose(args[1], (0, 2, 3, 1))
        pads = _pair(attributes.get("pads"), (0, 0, 0, 0))
        if pads[:2] != pads[2:]:
            value = mx.pad(value, ((0, 0), (pads[0], pads[2]), (pads[1], pads[3]), (0, 0)))
            padding = (0, 0)
        else:
            padding = pads[:2]
        output = mx.conv2d(
            value,
            weight,
            stride=_pair(attributes.get("strides"), (1, 1)),
            padding=padding,
            dilation=_pair(attributes.get("dilations"), (1, 1)),
            groups=int(attributes.get("group", 1)),
        )
        if len(args) > 2 and args[2] is not None:
            output = output + args[2].reshape(1, 1, 1, -1)
        return mx.transpose(output, (0, 3, 1, 2))

    @staticmethod
    def _max_pool(value, attributes):
        pads = _pair(attributes.get("pads"), (0, 0, 0, 0))
        if pads[:2] != pads[2:]:
            raise ValueError("Asymmetric ONNX MaxPool padding is not supported.")
        output = nn.MaxPool2d(
            _pair(attributes["kernel_shape"], (1, 1)),
            stride=_pair(attributes.get("strides"), attributes["kernel_shape"]),
            padding=pads[:2],
        )(mx.transpose(value, (0, 2, 3, 1)))
        return mx.transpose(output, (0, 3, 1, 2))

    @staticmethod
    def _resize(args, attributes):
        value = args[0]
        scales = args[2] if len(args) > 2 else None
        sizes = args[3] if len(args) > 3 else None
        if sizes is not None:
            target = [int(item) for item in np.asarray(sizes).tolist()]
            scale = (target[2] / value.shape[2], target[3] / value.shape[3])
        elif scales is not None:
            scale_values = np.asarray(scales).tolist()
            scale = (float(scale_values[2]), float(scale_values[3]))
        else:
            raise ValueError("Converted ONNX Resize requires scales or sizes.")
        mode = attributes.get("mode", "nearest")
        output = nn.Upsample(scale, mode="nearest" if mode == "nearest" else "linear")(
            mx.transpose(value, (0, 2, 3, 1))
        )
        return mx.transpose(output, (0, 3, 1, 2))

    @staticmethod
    def _slice(args, attributes):
        starts = attributes.get("starts")
        ends = attributes.get("ends")
        axes = attributes.get("axes")
        steps = attributes.get("steps")
        if starts is None:
            starts = np.asarray(args[1]).tolist()
            ends = np.asarray(args[2]).tolist()
            axes = (
                np.asarray(args[3]).tolist()
                if len(args) > 3 and args[3] is not None
                else range(len(starts))
            )
            steps = (
                np.asarray(args[4]).tolist()
                if len(args) > 4 and args[4] is not None
                else [1] * len(starts)
            )
        selection = [slice(None)] * args[0].ndim
        for start, end, axis, step in zip(
            starts, ends, axes, steps or [1] * len(starts), strict=True
        ):
            axis = int(axis)
            step = int(step)
            extent = int(args[0].shape[axis])
            start = max(-extent, min(extent, int(start)))
            end = max(-extent - 1, min(extent, int(end)))
            selection[axis] = slice(start, end, step)
        return args[0][tuple(selection)]


def load_onnx_mlx_bundle(path: str | Path):
    return MLXONNXGraph(path)
