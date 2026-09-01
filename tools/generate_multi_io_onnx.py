#!/usr/bin/env python3
"""Generates a deterministic two-input/two-output ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def generate(output: Path) -> None:
    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [1, 3])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [1, 3])
    summed = helper.make_tensor_value_info("summed", TensorProto.FLOAT, [1, 3])
    activated = helper.make_tensor_value_info(
        "activated", TensorProto.FLOAT, [1, 3]
    )
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["lhs", "rhs"], ["summed"]),
            helper.make_node("Relu", ["summed"], ["activated"]),
        ],
        "tiny_iree_multi_io",
        [lhs, rhs],
        [summed, activated],
    )
    model = helper.make_model(
        graph,
        producer_name="tiny-iree",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
