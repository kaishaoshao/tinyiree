#!/usr/bin/env python3
"""Generates a tiny deterministic CNN used to test the full operator path."""

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

    conv_weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    conv_bias = np.zeros((1,), dtype=np.float32)
    dense_weight = np.asarray([[0.01, 0.02]], dtype=np.float32)
    dense_bias = np.zeros((2,), dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv"]),
            helper.make_node("Relu", ["conv"], ["relu"]),
            helper.make_node(
                "MaxPool", ["relu"], ["pool"], kernel_shape=[2, 2],
                strides=[2, 2]
            ),
            helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
            helper.make_node(
                "Gemm", ["flat", "dense_w", "dense_b"], ["logits"]
            ),
            helper.make_node("Softmax", ["logits"], ["output"], axis=1),
        ],
        "tiny_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [
            numpy_helper.from_array(conv_weight, "conv_w"),
            numpy_helper.from_array(conv_bias, "conv_b"),
            numpy_helper.from_array(dense_weight, "dense_w"),
            numpy_helper.from_array(dense_bias, "dense_b"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)],
        producer_name="tiny-iree-learning"
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
