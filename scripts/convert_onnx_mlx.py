#!/usr/bin/env python3
"""Convert a standard-operator ONNX graph into a lightweight MLX bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def _attribute_value(attribute):
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, onnx.TensorProto):
        return np.ascontiguousarray(numpy_helper.to_array(value))
    if isinstance(value, tuple):
        return list(value)
    return value


def convert(source: Path, destination: Path):
    model = onnx.load(source, load_external_data=True)
    weights = {
        tensor.name: np.ascontiguousarray(numpy_helper.to_array(tensor))
        for tensor in model.graph.initializer
    }
    nodes = []
    for index, node in enumerate(model.graph.node):
        attributes = {}
        for attribute in node.attribute:
            value = _attribute_value(attribute)
            if isinstance(value, np.ndarray):
                key = f"__constant_{index}_{attribute.name}"
                weights[key] = value
                attributes[attribute.name] = {"tensor": key}
            else:
                attributes[attribute.name] = value
        nodes.append(
            {
                "name": node.name or f"{node.op_type}_{index}",
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": attributes,
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    save_file(weights, destination / "weights.safetensors")
    graph = {
        "format": "weetodd-onnx-mlx-v1",
        "source_name": source.name,
        "opsets": {item.domain: item.version for item in model.opset_import},
        "inputs": [item.name for item in model.graph.input if item.name not in weights],
        "outputs": [item.name for item in model.graph.output],
        "nodes": nodes,
    }
    (destination / "graph.json").write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source": str(source),
                "destination": str(destination),
                "nodes": len(nodes),
                "tensors": len(weights),
                "operators": sorted({node["op"] for node in nodes}),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source.expanduser().resolve(), args.destination.expanduser().resolve())


if __name__ == "__main__":
    main()
