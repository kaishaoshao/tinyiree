#!/usr/bin/env python3
"""Runs a single-input ONNX model with ONNX's built-in reference evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--input", required=True, help="comma-separated f32 values")
    args = parser.parse_args()

    model = onnx.load(args.model)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [value for value in model.graph.input if value.name not in initializer_names]
    if len(graph_inputs) != 1:
        parser.error("model must have exactly one non-initializer input")
    shape = [int(dimension.dim_value) for dimension in graph_inputs[0].type.tensor_type.shape.dim]
    values = np.asarray([float(value) for value in args.input.split(",")], dtype=np.float32)
    values = values.reshape(shape)
    output = ReferenceEvaluator(model).run(None, {graph_inputs[0].name: values})[0].reshape(-1)
    print("result: [" + ", ".join(f"{float(value):.6e}" for value in output) + "]")
    print(f"argmax: {int(np.argmax(output))}")


if __name__ == "__main__":
    main()
