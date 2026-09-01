#!/usr/bin/env python3
"""Generates a per-tensor int8 Q/DQ ONNX model for tiny-iree tests."""

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

    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")
    zero_point = numpy_helper.from_array(np.array(0, dtype=np.int8), "zero_point")
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantizeLinear", ["input", "scale", "zero_point"], ["quantized"]
            ),
            helper.make_node(
                "DequantizeLinear", ["quantized", "scale", "zero_point"], ["dequantized"]
            ),
            helper.make_node("Relu", ["dequantized"], ["output"]),
        ],
        "tiny_iree_qdq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [scale, zero_point],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19)],
        producer_name="tiny-iree",
    )
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
