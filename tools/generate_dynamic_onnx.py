#!/usr/bin/env python3
"""Generates a two-dynamic-dimension ONNX model for runtime shape tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def generate(output: Path) -> None:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["rows", "columns"]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["rows", "columns"]
    )
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])],
        "tiny_iree_dynamic_batch",
        [input_info],
        [output_info],
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
