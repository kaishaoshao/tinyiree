#!/usr/bin/env python3
"""Generates a square MatMul model that exercises CPU tiling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")

    size = args.size
    weight = numpy_helper.from_array(np.eye(size, dtype=np.float32), "weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "tiny_iree_tiled_matmul",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [size, size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [size, size])],
        [weight],
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
