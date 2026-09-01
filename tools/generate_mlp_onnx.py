#!/usr/bin/env python3
"""Generates the deterministic MLP used by the tiny-iree end-to-end test."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def generate(output_path: Path) -> None:
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [4, 3],
        [0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9, 1.0, -1.1, 1.2],
    )
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [3], [0.1, 0.2, -0.1])
    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["matmul"]),
        helper.make_node("Add", ["matmul", "bias"], ["biased"]),
        helper.make_node("Relu", ["biased"], ["activated"]),
        helper.make_node("Softmax", ["activated"], ["probabilities"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_iree_mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 3])],
        [weight, bias],
    )
    model = helper.make_model(
        graph,
        producer_name="tiny-iree",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
