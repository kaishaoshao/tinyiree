#!/usr/bin/env python3
"""Generates an ONNX Split model for multi-result dispatch tests."""

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
        [helper.make_node("Split", ["input"], ["left", "right"],
                          axis=1, split=[2, 2])],
        "tiny_iree_split",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info("left", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, [1, 2]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="tiny-iree",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
