#!/usr/bin/env python3
"""Imports a small f32 ONNX subset into the tiny-iree input dialect."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


SUPPORTED_OPS = {
    "MatMul": "rank-2 f32 tensors with (MxK) x (KxN) -> (MxN)",
    "Gemm": "MatMul plus optional bias; no transpose; alpha=beta=1",
    "Add": "equal f32 shapes or final-dimension bias broadcast",
    "Relu": "ranked f32 tensor with unchanged shape",
    "Softmax": "ranked f32 tensor over the final dimension",
    "QuantizeLinear": "matched per-tensor signed-int8 Q/DQ pair",
    "DequantizeLinear": "matched per-tensor signed-int8 Q/DQ pair",
    "Split": "final dimension, exactly two equal-shaped outputs",
    "Conv": "static NCHW, bias required, valid stride-one, group=1",
    "MaxPool": "static NCHW, 2x2 kernel, stride two, no padding",
    "Flatten": "axis=1; lowered to a static reshape",
    "Reshape": "static input/output with equal element counts",
    "Transpose": "static rank-4 NCHW to NHWC only",
}

STANDARD_ONNX_DOMAINS = {"", "ai.onnx"}


def _tensor_type(shape: list[int]) -> str:
    dimensions = ("?" if dim < 0 else str(dim) for dim in shape)
    return "tensor<" + "x".join([*dimensions, "f32"]) + ">"


def _shape_from_value_info(value_info: Any, require_f32: bool = False) -> list[int]:
    tensor_type = value_info.type.tensor_type
    if require_f32 and tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError(f"{value_info.name}: only f32 tensors are supported")
    shape = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(-1)
        else:
            raise ValueError(f"{value_info.name}: dimension has no size or symbol")
    return shape


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite initializer values are not supported")
    text = repr(float(value))
    return text if "." in text or "e" in text.lower() else text + ".0"


def _format_dense(values: list[float], shape: list[int]) -> str:
    if not shape:
        if len(values) != 1:
            raise ValueError("scalar initializer must have exactly one value")
        return _format_float(values[0])
    if len(shape) == 1:
        return "[" + ", ".join(_format_float(value) for value in values) + "]"
    stride = math.prod(shape[1:])
    rows = [
        _format_dense(values[index : index + stride], shape[1:])
        for index in range(0, len(values), stride)
    ]
    return "[" + ", ".join(rows) + "]"


def _attributes(node: Any) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _symbol(name: str, fallback: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_$.-]", "_", name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = fallback + sanitized
    return sanitized


def _load_model(input_path: Path) -> Any:
    model = shape_inference.infer_shapes(onnx.load(input_path))
    onnx.checker.check_model(model)
    return model


def _node_identity(node: Any) -> str:
    return f"{node.domain or 'ai.onnx'}::{node.op_type}"


def _is_supported_node(node: Any) -> bool:
    return node.domain in STANDARD_ONNX_DOMAINS and node.op_type in SUPPORTED_OPS


def _shape_text(value_info: Any) -> str:
    return "x".join("?" if dim < 0 else str(dim)
                    for dim in _shape_from_value_info(value_info)) or "scalar"


def analyze_onnx(input_path: Path) -> None:
    model = _load_model(input_path)
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_inputs = [value for value in graph.input
                    if value.name not in initializer_names]

    print("format: onnx")
    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx"
        print(f"opset: {domain}={opset.version}")
    print("graph inputs:")
    for value in graph_inputs:
        print(f"  {value.name}: {_shape_text(value)}")
    print(f"initializers: {len(graph.initializer)}")
    for initializer in graph.initializer:
        shape = "x".join(str(dim) for dim in initializer.dims) or "scalar"
        print(f"  {initializer.name}: {shape}")
    print("nodes:")
    for index, node in enumerate(graph.node):
        supported = "yes" if _is_supported_node(node) else "no"
        inputs = ", ".join(name for name in node.input if name)
        outputs = ", ".join(node.output)
        print(
            f"  [{index}] {_node_identity(node)}: {inputs} -> {outputs} "
            f"(op-type-supported={supported})"
        )
    print("graph outputs:")
    for value in graph.output:
        print(f"  {value.name}: {_shape_text(value)}")


def import_onnx(input_path: Path, output_path: Path, function_name: str) -> None:
    model = _load_model(input_path)
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    graph_inputs = [value for value in graph.input if value.name not in initializer_names]
    if not graph_inputs or not graph.output:
        raise ValueError("model requires at least one graph input and output")

    shapes: dict[str, list[int]] = {}
    element_types: dict[str, int] = {}
    for value in [*graph.input, *graph.value_info, *graph.output]:
        shapes[value.name] = _shape_from_value_info(value)
        element_types[value.name] = value.type.tensor_type.elem_type
    for value in [*graph_inputs, *graph.output]:
        _shape_from_value_info(value, require_f32=True)
    for initializer in graph.initializer:
        if initializer.data_type not in (
            TensorProto.FLOAT, TensorProto.INT8, TensorProto.UINT8,
            TensorProto.INT64,
        ):
            raise ValueError(
                f"{initializer.name}: unsupported initializer element type"
            )
        shapes[initializer.name] = [int(dim) for dim in initializer.dims]
        element_types[initializer.name] = initializer.data_type

    input_names = [value.name for value in graph_inputs]
    output_names = [value.name for value in graph.output]
    ssa_values = {name: f"%arg{index}" for index, name in enumerate(input_names)}
    arguments = ", ".join(
        f"%arg{index}: {_tensor_type(shapes[name])}"
        for index, name in enumerate(input_names)
    )
    result_types = ", ".join(_tensor_type(shapes[name]) for name in output_names)
    function_results = result_types if len(output_names) == 1 else f"({result_types})"
    lines = [
        "module {",
        f"  func.func @{_symbol(function_name, 'predict')}({arguments}) "
        f"-> {function_results} {{",
    ]

    for index, initializer in enumerate(graph.initializer):
        if initializer.data_type == TensorProto.INT64:
            continue
        name = f"%c{index}"
        ssa_values[initializer.name] = name
        values = numpy_helper.to_array(initializer).reshape(-1).tolist()
        dense = _format_dense([float(value) for value in values], shapes[initializer.name])
        lines.append(
            f"    {name} = arith.constant dense<{dense}> : "
            f"{_tensor_type(shapes[initializer.name])}"
        )

    next_value = 0

    def new_value() -> str:
        nonlocal next_value
        value = f"%v{next_value}"
        next_value += 1
        return value

    pending_quantize: dict[str, tuple[str, str, str]] = {}
    for node_index, node in enumerate(graph.node):
        if not _is_supported_node(node):
            raise ValueError(
                f"node {node_index}: unsupported ONNX op {_node_identity(node)}"
            )
        if node.op_type == "Split":
            if len(node.input) != 1 or len(node.output) != 2:
                raise ValueError("Split requires one input and exactly two outputs")
            attributes = _attributes(node)
            axis = int(attributes.get("axis", 0))
            rank = len(shapes[node.input[0]])
            if (axis + rank if axis < 0 else axis) != rank - 1:
                raise ValueError("only Split over the final dimension is supported")
            if any(name not in shapes for name in node.output):
                raise ValueError("Split outputs require inferred shapes")
            results = [new_value(), new_value()]
            output_types = ", ".join(
                _tensor_type(shapes[name]) for name in node.output
            )
            lines.append(
                f'    {", ".join(results)} = "tiree_input.split"('
                f'{ssa_values[node.input[0]]}) {{axis = {axis} : i64}} : '
                f'({_tensor_type(shapes[node.input[0]])}) -> ({output_types})'
            )
            for name, result in zip(node.output, results):
                ssa_values[name] = result
            continue
        if len(node.output) != 1 or node.output[0] not in shapes:
            raise ValueError(f"node {node_index}: requires one shaped output")
        if node.op_type == "QuantizeLinear":
            if len(node.input) != 3:
                raise ValueError("QuantizeLinear requires input, scale, and zero point")
            if node.input[1] not in initializer_names or node.input[2] not in initializer_names:
                raise ValueError("quantization scale and zero point must be initializers")
            if element_types[node.input[2]] != TensorProto.INT8:
                raise ValueError("only signed int8 QuantizeLinear is supported")
            if (shapes[node.input[1]] not in ([], [1]) or
                    shapes[node.input[2]] not in ([], [1])):
                raise ValueError("only per-tensor QuantizeLinear is supported")
            scale_value = float(
                numpy_helper.to_array(initializers[node.input[1]]).reshape(-1)[0]
            )
            if not math.isfinite(scale_value) or scale_value <= 0.0:
                raise ValueError("QuantizeLinear scale must be finite and positive")
            pending_quantize[node.output[0]] = tuple(node.input)
            continue
        if node.op_type == "DequantizeLinear":
            if len(node.input) != 3 or node.input[0] not in pending_quantize:
                raise ValueError("DequantizeLinear must immediately consume a supported QuantizeLinear")
            source, scale, zero_point = pending_quantize.pop(node.input[0])
            if node.input[1:] != [scale, zero_point]:
                raise ValueError("QuantizeLinear and DequantizeLinear parameters must match")
            operands = [ssa_values[name] for name in (source, scale, zero_point)]
            result = new_value()
            output_type = _tensor_type(shapes[node.output[0]])
            operand_types = ", ".join(
                _tensor_type(shapes[name]) for name in (source, scale, zero_point)
            )
            lines.append(
                f'    {result} = "tiree_input.fake_quant"('
                + ", ".join(operands)
                + f") : ({operand_types}) -> {output_type}"
            )
            ssa_values[node.output[0]] = result
            continue
        if node.op_type == "Conv":
            attributes = _attributes(node)
            if len(node.input) != 3:
                raise ValueError("Conv requires input, weight, and bias")
            strides = [int(value) for value in attributes.get("strides", [1, 1])]
            pads = [int(value) for value in attributes.get("pads", [0, 0, 0, 0])]
            dilations = [int(value) for value in attributes.get("dilations", [1, 1])]
            group = int(attributes.get("group", 1))
            if strides != [1, 1] or pads != [0, 0, 0, 0] or dilations != [1, 1] or group != 1:
                raise ValueError("only valid stride-one non-grouped Conv is supported")
            operands = [ssa_values[name] for name in node.input]
            result = new_value()
            output_type = _tensor_type(shapes[node.output[0]])
            operand_types = ", ".join(_tensor_type(shapes[name]) for name in node.input)
            lines.append(
                f'    {result} = "tiree_input.conv2d"('
                + ", ".join(operands)
                + f") {{pads = array<i64: 0, 0, 0, 0>, strides = array<i64: 1, 1>}} "
                f": ({operand_types}) -> {output_type}"
            )
            ssa_values[node.output[0]] = result
            continue
        if node.op_type == "MaxPool":
            attributes = _attributes(node)
            kernel_shape = [int(value) for value in attributes.get("kernel_shape", [])]
            strides = [int(value) for value in attributes.get("strides", kernel_shape)]
            pads = [int(value) for value in attributes.get("pads", [0, 0, 0, 0])]
            if kernel_shape != [2, 2] or strides != [2, 2] or pads != [0, 0, 0, 0]:
                raise ValueError("only valid 2x2 stride-two MaxPool is supported")
            result = new_value()
            lines.append(
                f'    {result} = "tiree_input.max_pool2d"({ssa_values[node.input[0]]}) '
                f'{{kernel_shape = array<i64: 2, 2>, strides = array<i64: 2, 2>}} '
                f': ({_tensor_type(shapes[node.input[0]])}) -> '
                f'{_tensor_type(shapes[node.output[0]])}'
            )
            ssa_values[node.output[0]] = result
            continue
        if node.op_type in ("Flatten", "Reshape"):
            attributes = _attributes(node)
            if node.op_type == "Flatten" and int(attributes.get("axis", 1)) != 1:
                raise ValueError("only Flatten axis=1 is supported")
            result = new_value()
            lines.append(
                f'    {result} = "tiree_input.reshape"({ssa_values[node.input[0]]}) '
                f': ({_tensor_type(shapes[node.input[0]])}) -> '
                f'{_tensor_type(shapes[node.output[0]])}'
            )
            ssa_values[node.output[0]] = result
            continue
        if node.op_type == "Transpose":
            attributes = _attributes(node)
            permutation = [int(value) for value in attributes.get("perm", [])]
            if permutation != [0, 2, 3, 1]:
                raise ValueError("only NCHW to NHWC Transpose is supported")
            result = new_value()
            lines.append(
                f'    {result} = "tiree_input.transpose"({ssa_values[node.input[0]]}) '
                f'{{permutation = array<i64: 0, 2, 3, 1>}} : '
                f'({_tensor_type(shapes[node.input[0]])}) -> '
                f'{_tensor_type(shapes[node.output[0]])}'
            )
            ssa_values[node.output[0]] = result
            continue
        try:
            operands = [ssa_values[name] for name in node.input if name]
        except KeyError as error:
            raise ValueError(f"node {node_index}: value {error.args[0]} is not available") from error
        output_type = _tensor_type(shapes[node.output[0]])
        attributes = _attributes(node)

        if node.op_type == "Gemm":
            if attributes.get("transA", 0) or attributes.get("transB", 0):
                raise ValueError("Gemm transpose attributes are not supported")
            if float(attributes.get("alpha", 1.0)) != 1.0 or float(attributes.get("beta", 1.0)) != 1.0:
                raise ValueError("Gemm alpha/beta values other than 1 are not supported")
            if len(operands) not in (2, 3):
                raise ValueError("Gemm requires two or three inputs")
            matmul_value = new_value()
            lines.append(
                f'    {matmul_value} = "tiree_input.matmul"({operands[0]}, {operands[1]}) '
                f": ({_tensor_type(shapes[node.input[0]])}, "
                f"{_tensor_type(shapes[node.input[1]])}) -> {output_type}"
            )
            result = matmul_value
            if len(operands) == 3:
                result = new_value()
                lines.append(
                    f'    {result} = "tiree_input.add"({matmul_value}, {operands[2]}) '
                    f": ({output_type}, {_tensor_type(shapes[node.input[2]])}) -> {output_type}"
                )
        else:
            result = new_value()
            mnemonic = node.op_type.lower()
            attribute_text = ""
            if node.op_type == "Softmax":
                axis = int(attributes.get("axis", -1))
                rank = len(shapes[node.input[0]])
                normalized_axis = axis + rank if axis < 0 else axis
                if normalized_axis != rank - 1:
                    raise ValueError("only Softmax over the final dimension is supported")
                attribute_text = f" {{axis = {axis} : i64}}"
            operand_types = ", ".join(_tensor_type(shapes[name]) for name in node.input if name)
            lines.append(
                f'    {result} = "tiree_input.{mnemonic}"('
                + ", ".join(operands)
                + f"){attribute_text} : ({operand_types}) -> {output_type}"
            )

        ssa_values[node.output[0]] = result

    if pending_quantize:
        raise ValueError("QuantizeLinear without a matching DequantizeLinear is unsupported")

    missing_outputs = [name for name in output_names if name not in ssa_values]
    if missing_outputs:
        raise ValueError(f"graph outputs were not produced: {missing_outputs}")
    return_values = ", ".join(ssa_values[name] for name in output_names)
    lines.extend(
        [
            f"    return {return_values} : {result_types}",
            "  }",
            "}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--function", default="predict")
    parser.add_argument(
        "--analyze", action="store_true",
        help="print the inferred ONNX graph and op-type recognition results",
    )
    parser.add_argument(
        "--list-supported-ops", action="store_true",
        help="list supported ONNX op types and their restrictions",
    )
    args = parser.parse_args()
    if args.list_supported_ops:
        for name, restriction in SUPPORTED_OPS.items():
            print(f"{name}: {restriction}")
        return
    if args.input is None:
        parser.error("input is required unless --list-supported-ops is used")
    if args.analyze:
        analyze_onnx(args.input)
        return
    if args.output is None:
        parser.error("--output is required when importing a model")
    import_onnx(args.input, args.output, args.function)


if __name__ == "__main__":
    main()
