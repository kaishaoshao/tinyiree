#!/usr/bin/env python3
"""Generates a 16x16 MatMul model that exercises CPU tiling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    weight = numpy_helper.from_array(np.eye(16, dtype=np.float32), "weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "tiny_iree_tiled_matmul",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [16, 16])],
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
