#!/usr/bin/env python3
"""Generates a three-dispatch chain that exercises transient buffer reuse."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["temporary0"]),
            helper.make_node("Relu", ["temporary0"], ["temporary1"]),
            helper.make_node("Relu", ["temporary1"], ["output"]),
        ],
        "tiny_iree_resource_reuse",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="tiny-iree",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
