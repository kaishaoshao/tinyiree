#!/usr/bin/env python3
"""Generates structured CPU kernel MLIR from tiny HAL dispatch interfaces."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _shape(tensor_type: str) -> list[int]:
    match = re.fullmatch(r"tensor<(.+)>", tensor_type.strip())
    if not match:
        raise ValueError(f"only ranked f32 tensors are supported: {tensor_type}")
    parts = match.group(1).split("x")
    if not parts or parts[-1] != "f32":
        raise ValueError(f"only ranked f32 tensors are supported: {tensor_type}")
    try:
        return [-1 if value == "?" else int(value) for value in parts[:-1]]
    except ValueError as error:
        raise ValueError(f"invalid tensor shape: {tensor_type}") from error


def _compatible_shape(lhs: list[int], rhs: list[int]) -> bool:
    return len(lhs) == len(rhs) and all(
        left < 0 or right < 0 or left == right for left, right in zip(lhs, rhs)
    )


def _memref(tensor_type: str) -> str:
    _shape(tensor_type)
    return "memref<" + tensor_type.strip()[len("tensor<") :]


def _function_header(kernel: str, inputs: list[str], output: str) -> list[str]:
    arguments = [f"%arg{index}: {_memref(value)}" for index, value in enumerate(inputs)]
    arguments.append(f"%output: {_memref(output)}")
    return [
        f"  func.func @tiree_kernel_{kernel}(",
        "      " + ", ".join(arguments) + ")",
        "      attributes {llvm.emit_c_interface} {",
    ]


def _matmul(kernel: str, inputs: list[str], output: str, fused: bool) -> list[str]:
    if len(inputs) != (3 if fused else 2):
        raise ValueError(f"{kernel}: invalid input count")
    lhs_shape, rhs_shape, output_shape = _shape(inputs[0]), _shape(inputs[1]), _shape(output)
    if len(lhs_shape) != 2 or len(rhs_shape) != 2 or len(output_shape) != 2:
        raise ValueError(f"{kernel}: only rank-2 matmul is supported")
    lines = _function_header(kernel, inputs, output)
    lines.extend(
        [
            "    %zero = arith.constant 0.0 : f32",
            f"    linalg.fill ins(%zero : f32) outs(%output : {_memref(output)})",
            f"    linalg.matmul ins(%arg0, %arg1 : {_memref(inputs[0])}, {_memref(inputs[1])})",
            f"                  outs(%output : {_memref(output)})",
        ]
    )
    if fused:
        if _shape(inputs[2]) != [output_shape[-1]]:
            raise ValueError("matmul_add_relu: bias must match the final dimension")
        lines.extend(
            [
                "    linalg.generic {",
                "      indexing_maps = [",
                "        affine_map<(d0, d1) -> (d0, d1)>,",
                "        affine_map<(d0, d1) -> (d1)>,",
                "        affine_map<(d0, d1) -> (d0, d1)>],",
                '      iterator_types = ["parallel", "parallel"]',
                f"    }} ins(%output, %arg2 : {_memref(output)}, {_memref(inputs[2])})",
                f"      outs(%output : {_memref(output)}) {{",
                "    ^bb0(%value: f32, %bias: f32, %unused: f32):",
                "      %sum = arith.addf %value, %bias : f32",
                "      %activated = arith.maximumf %sum, %zero : f32",
                "      linalg.yield %activated : f32",
                "    }",
            ]
        )
    lines.extend(["    return", "  }"])
    return lines


def _elementwise(symbol: str, kernel: str, inputs: list[str], output: str) -> list[str]:
    output_shape = _shape(output)
    rank = len(output_shape)
    if rank not in (1, 2):
        raise ValueError(f"{kernel}: only rank-1/rank-2 elementwise kernels are supported")
    dimensions = ", ".join(f"d{index}" for index in range(rank))
    identity = f"affine_map<({dimensions}) -> ({dimensions})>"
    lines = _function_header(symbol, inputs, output)
    lines.append("    %zero = arith.constant 0.0 : f32")
    maps = []
    if kernel == "add":
        if len(inputs) != 2 or not _compatible_shape(_shape(inputs[0]), output_shape):
            raise ValueError("add: invalid interface")
        rhs_shape = _shape(inputs[1])
        if rhs_shape == output_shape:
            rhs_map = identity
        elif rhs_shape == [output_shape[-1]]:
            rhs_map = f"affine_map<({dimensions}) -> (d{rank - 1})>"
        else:
            raise ValueError("add: unsupported broadcast")
        maps = [identity, rhs_map, identity]
    elif kernel == "relu":
        if len(inputs) != 1 or not _compatible_shape(_shape(inputs[0]), output_shape):
            raise ValueError("relu: invalid interface")
        maps = [identity, identity]
    elif kernel == "fake_quant":
        if (len(inputs) != 3 or
                not _compatible_shape(_shape(inputs[0]), output_shape) or
                _shape(inputs[1]) not in ([], [1]) or
                _shape(inputs[2]) not in ([], [1])):
            raise ValueError("fake_quant: invalid per-tensor interface")
        parameter_map = lambda shape: (
            f"affine_map<({dimensions}) -> ()>" if not shape
            else f"affine_map<({dimensions}) -> (0)>"
        )
        maps = [identity, parameter_map(_shape(inputs[1])),
                parameter_map(_shape(inputs[2])), identity]
    else:
        raise ValueError(f"unsupported elementwise kernel: {kernel}")
    lines.extend(["    linalg.generic {", "      indexing_maps = ["])
    lines.extend(
        f"        {value}{',' if index + 1 < len(maps) else ''}"
        for index, value in enumerate(maps)
    )
    lines.extend(
        [
            "      ],",
            "      iterator_types = [" + ", ".join('"parallel"' for _ in range(rank)) + "]",
        ]
    )
    input_values = ", ".join(f"%arg{index}" for index in range(len(inputs)))
    input_types = ", ".join(_memref(value) for value in inputs)
    lines.extend(
        [
            f"    }} ins({input_values} : {input_types}) outs(%output : {_memref(output)}) {{",
        ]
    )
    if kernel == "add":
        lines.extend(
            [
                "    ^bb0(%lhs: f32, %rhs: f32, %unused: f32):",
                "      %value = arith.addf %lhs, %rhs : f32",
            ]
        )
    elif kernel == "relu":
        lines.extend(
            [
                "    ^bb0(%input: f32, %unused: f32):",
                "      %value = arith.maximumf %input, %zero : f32",
            ]
        )
    else:
        lines.extend(
            [
                "    ^bb0(%input: f32, %scale: f32, %zero_point: f32, %unused: f32):",
                "      %scaled = arith.divf %input, %scale : f32",
                "      %rounded = math.roundeven %scaled : f32",
                "      %shifted = arith.addf %rounded, %zero_point : f32",
                "      %low = arith.constant -128.0 : f32",
                "      %high = arith.constant 127.0 : f32",
                "      %bounded_low = arith.maximumf %shifted, %low : f32",
                "      %bounded = arith.minimumf %bounded_low, %high : f32",
                "      %centered = arith.subf %bounded, %zero_point : f32",
                "      %value = arith.mulf %centered, %scale : f32",
            ]
        )
    lines.extend(["      linalg.yield %value : f32", "    }", "    return", "  }"])
    return lines


def _softmax(kernel: str, inputs: list[str], output: str) -> list[str]:
    if len(inputs) != 1 or not _compatible_shape(_shape(inputs[0]), _shape(output)):
        raise ValueError("softmax: invalid interface")
    shape = _shape(output)
    if len(shape) != 2:
        raise ValueError("softmax: v1 codegen requires rank 2")
    rows, columns = shape
    lines = _function_header(kernel, inputs, output)
    lines.extend(
        [
            "    %c0 = arith.constant 0 : index",
            "    %c1 = arith.constant 1 : index",
            (f"    %rows = arith.constant {rows} : index" if rows >= 0 else
             f"    %rows = memref.dim %arg0, %c0 : {_memref(inputs[0])}"),
            (f"    %columns = arith.constant {columns} : index" if columns >= 0 else
             f"    %columns = memref.dim %arg0, %c1 : {_memref(inputs[0])}"),
            "    %negative_infinity = arith.constant 0xFF800000 : f32",
            "    %zero = arith.constant 0.0 : f32",
            "    scf.for %row = %c0 to %rows step %c1 {",
            "      %maximum = scf.for %column = %c0 to %columns step %c1",
            "          iter_args(%current = %negative_infinity) -> f32 {",
            f"        %value = memref.load %arg0[%row, %column] : {_memref(inputs[0])}",
            "        %next = arith.maximumf %current, %value : f32",
            "        scf.yield %next : f32",
            "      }",
            "      %sum = scf.for %column = %c0 to %columns step %c1",
            "          iter_args(%current = %zero) -> f32 {",
            f"        %value = memref.load %arg0[%row, %column] : {_memref(inputs[0])}",
            "        %shifted = arith.subf %value, %maximum : f32",
            "        %exponential = math.exp %shifted : f32",
            f"        memref.store %exponential, %output[%row, %column] : {_memref(output)}",
            "        %next = arith.addf %current, %exponential : f32",
            "        scf.yield %next : f32",
            "      }",
            "      scf.for %column = %c0 to %columns step %c1 {",
            f"        %value = memref.load %output[%row, %column] : {_memref(output)}",
            "        %normalized = arith.divf %value, %sum : f32",
            f"        memref.store %normalized, %output[%row, %column] : {_memref(output)}",
            "      }",
            "    }",
            "    return",
            "  }",
        ]
    )
    return lines


def _split(kernel: str, inputs: list[str], outputs: list[str]) -> list[str]:
    if len(inputs) != 1 or len(outputs) != 2 or outputs[0] != outputs[1]:
        raise ValueError("split: expected one input and two equal output types")
    input_shape = _shape(inputs[0])
    output_shape = _shape(outputs[0])
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise ValueError("split: codegen currently supports rank-2 tensors")
    arguments = [f"%arg0: {_memref(inputs[0])}"]
    arguments.extend(
        f"%output{index}: {_memref(output)}" for index, output in enumerate(outputs)
    )
    rows, columns = output_shape
    return [
        f"  func.func @tiree_kernel_{kernel}(",
        "      " + ", ".join(arguments) + ")",
        "      attributes {llvm.emit_c_interface} {",
        "    %c0 = arith.constant 0 : index",
        "    %c1 = arith.constant 1 : index",
        (f"    %rows = arith.constant {rows} : index" if rows >= 0 else
         f"    %rows = memref.dim %output0, %c0 : {_memref(outputs[0])}"),
        (f"    %columns = arith.constant {columns} : index" if columns >= 0 else
         f"    %columns = memref.dim %output0, %c1 : {_memref(outputs[0])}"),
        "    scf.for %row = %c0 to %rows step %c1 {",
        "      scf.for %column = %c0 to %columns step %c1 {",
        "        %right_column = arith.addi %column, %columns : index",
        f"        %left = memref.load %arg0[%row, %column] : {_memref(inputs[0])}",
        f"        %right = memref.load %arg0[%row, %right_column] : {_memref(inputs[0])}",
        f"        memref.store %left, %output0[%row, %column] : {_memref(outputs[0])}",
        f"        memref.store %right, %output1[%row, %column] : {_memref(outputs[1])}",
        "      }",
        "    }",
        "    return",
        "  }",
    ]


def _conv2d(kernel: str, inputs: list[str], output: str) -> list[str]:
    if len(inputs) != 3:
        raise ValueError("conv2d: expected input, weight, and bias")
    input_shape, weight_shape = _shape(inputs[0]), _shape(inputs[1])
    bias_shape, output_shape = _shape(inputs[2]), _shape(output)
    if (len(input_shape) != 4 or len(weight_shape) != 4 or
            len(output_shape) != 4 or bias_shape != [weight_shape[0]] or
            any(value < 0 for value in [*input_shape, *weight_shape, *output_shape])):
        raise ValueError("conv2d: requires static NCHW tensors")
    batches, input_channels, _, _ = input_shape
    output_channels, _, kernel_height, kernel_width = weight_shape
    _, _, output_height, output_width = output_shape
    lines = _function_header(kernel, inputs, output)
    lines.extend(
        [
            "    %c0 = arith.constant 0 : index",
            "    %c1 = arith.constant 1 : index",
            f"    %batches = arith.constant {batches} : index",
            f"    %input_channels = arith.constant {input_channels} : index",
            f"    %output_channels = arith.constant {output_channels} : index",
            f"    %output_height = arith.constant {output_height} : index",
            f"    %output_width = arith.constant {output_width} : index",
            f"    %kernel_height = arith.constant {kernel_height} : index",
            f"    %kernel_width = arith.constant {kernel_width} : index",
            "    scf.for %batch = %c0 to %batches step %c1 {",
            "      scf.for %oc = %c0 to %output_channels step %c1 {",
            "        %bias = memref.load %arg2[%oc] : " + _memref(inputs[2]),
            "        scf.for %oh = %c0 to %output_height step %c1 {",
            "          scf.for %ow = %c0 to %output_width step %c1 {",
            "            %sum = scf.for %ic = %c0 to %input_channels step %c1",
            "                iter_args(%channel_sum = %bias) -> f32 {",
            "              %height_sum = scf.for %kh = %c0 to %kernel_height step %c1",
            "                  iter_args(%current_height = %channel_sum) -> f32 {",
            "                %width_sum = scf.for %kw = %c0 to %kernel_width step %c1",
            "                    iter_args(%current_width = %current_height) -> f32 {",
            "                  %ih = arith.addi %oh, %kh : index",
            "                  %iw = arith.addi %ow, %kw : index",
            f"                  %input_value = memref.load %arg0[%batch, %ic, %ih, %iw] : {_memref(inputs[0])}",
            f"                  %weight_value = memref.load %arg1[%oc, %ic, %kh, %kw] : {_memref(inputs[1])}",
            "                  %product = arith.mulf %input_value, %weight_value : f32",
            "                  %next = arith.addf %current_width, %product : f32",
            "                  scf.yield %next : f32",
            "                }",
            "                scf.yield %width_sum : f32",
            "              }",
            "              scf.yield %height_sum : f32",
            "            }",
            f"            memref.store %sum, %output[%batch, %oc, %oh, %ow] : {_memref(output)}",
            "          }",
            "        }",
            "      }",
            "    }",
            "    return",
            "  }",
        ]
    )
    return lines


def _max_pool2d(kernel: str, inputs: list[str], output: str) -> list[str]:
    if len(inputs) != 1:
        raise ValueError("max_pool2d: expected one input")
    input_shape, output_shape = _shape(inputs[0]), _shape(output)
    if (len(input_shape) != 4 or len(output_shape) != 4 or
            any(value < 0 for value in [*input_shape, *output_shape])):
        raise ValueError("max_pool2d: requires static NCHW tensors")
    batches, channels, _, _ = input_shape
    _, _, output_height, output_width = output_shape
    lines = _function_header(kernel, inputs, output)
    lines.extend(
        [
            "    %c0 = arith.constant 0 : index",
            "    %c1 = arith.constant 1 : index",
            "    %c2 = arith.constant 2 : index",
            f"    %batches = arith.constant {batches} : index",
            f"    %channels = arith.constant {channels} : index",
            f"    %output_height = arith.constant {output_height} : index",
            f"    %output_width = arith.constant {output_width} : index",
            "    %negative_infinity = arith.constant 0xFF800000 : f32",
            "    scf.for %batch = %c0 to %batches step %c1 {",
            "      scf.for %channel = %c0 to %channels step %c1 {",
            "        scf.for %oh = %c0 to %output_height step %c1 {",
            "          scf.for %ow = %c0 to %output_width step %c1 {",
            "            %ih_base = arith.muli %oh, %c2 : index",
            "            %iw_base = arith.muli %ow, %c2 : index",
            "            %maximum = scf.for %kh = %c0 to %c2 step %c1",
            "                iter_args(%height_max = %negative_infinity) -> f32 {",
            "              %width_max = scf.for %kw = %c0 to %c2 step %c1",
            "                  iter_args(%current = %height_max) -> f32 {",
            "                %ih = arith.addi %ih_base, %kh : index",
            "                %iw = arith.addi %iw_base, %kw : index",
            f"                %value = memref.load %arg0[%batch, %channel, %ih, %iw] : {_memref(inputs[0])}",
            "                %next = arith.maximumf %current, %value : f32",
            "                scf.yield %next : f32",
            "              }",
            "              scf.yield %width_max : f32",
            "            }",
            f"            memref.store %maximum, %output[%batch, %channel, %oh, %ow] : {_memref(output)}",
            "          }",
            "        }",
            "      }",
            "    }",
            "    return",
            "  }",
        ]
    )
    return lines


def _reshape(kernel: str, inputs: list[str], output: str) -> list[str]:
    if len(inputs) != 1:
        raise ValueError("reshape: expected one input")
    input_shape, output_shape = _shape(inputs[0]), _shape(output)
    if len(input_shape) != 4 or len(output_shape) != 2 or input_shape[0] != output_shape[0]:
        raise ValueError("reshape: codegen supports NCHW flatten to rank 2")
    batches, channels, height, width = input_shape
    lines = _function_header(kernel, inputs, output)
    lines.extend(
        [
            "    %c0 = arith.constant 0 : index",
            "    %c1 = arith.constant 1 : index",
            f"    %batches = arith.constant {batches} : index",
            f"    %channels = arith.constant {channels} : index",
            f"    %height = arith.constant {height} : index",
            f"    %width = arith.constant {width} : index",
            "    scf.for %batch = %c0 to %batches step %c1 {",
            "      scf.for %channel = %c0 to %channels step %c1 {",
            "        scf.for %h = %c0 to %height step %c1 {",
            "          scf.for %w = %c0 to %width step %c1 {",
            "            %channel_offset = arith.muli %channel, %height : index",
            "            %height_offset = arith.addi %channel_offset, %h : index",
            "            %row_offset = arith.muli %height_offset, %width : index",
            "            %flat = arith.addi %row_offset, %w : index",
            f"            %value = memref.load %arg0[%batch, %channel, %h, %w] : {_memref(inputs[0])}",
            f"            memref.store %value, %output[%batch, %flat] : {_memref(output)}",
            "          }",
            "        }",
            "      }",
            "    }",
            "    return",
            "  }",
        ]
    )
    return lines


def _transpose(kernel: str, inputs: list[str], output: str) -> list[str]:
    if len(inputs) != 1:
        raise ValueError("transpose: expected one input")
    input_shape, output_shape = _shape(inputs[0]), _shape(output)
    if len(input_shape) != 4 or output_shape != [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]:
        raise ValueError("transpose: expected NCHW to NHWC interface")
    bounds = [f"%d{index}" for index in range(4)]
    lines = _function_header(kernel, inputs, output)
    lines.extend(["    %c0 = arith.constant 0 : index", "    %c1 = arith.constant 1 : index"])
    lines.extend(
        f"    {bound} = arith.constant {dimension} : index"
        for bound, dimension in zip(bounds, output_shape)
    )
    lines.extend(
        [
            "    scf.for %n = %c0 to %d0 step %c1 {",
            "      scf.for %h = %c0 to %d1 step %c1 {",
            "        scf.for %w = %c0 to %d2 step %c1 {",
            "          scf.for %c = %c0 to %d3 step %c1 {",
            f"            %value = memref.load %arg0[%n, %c, %h, %w] : {_memref(inputs[0])}",
            f"            memref.store %value, %output[%n, %h, %w, %c] : {_memref(output)}",
            "          }",
            "        }",
            "      }",
            "    }",
            "    return",
            "  }",
        ]
    )
    return lines


def generate(input_path: Path, output_path: Path, cpu_codegen: str) -> None:
    plan = json.loads(input_path.read_text(encoding="utf-8"))
    if plan.get("format") != "tiny-iree-codegen-plan-v1":
        raise ValueError("unsupported tiny-iree codegen plan")
    entry_points = plan.get("entry_points")
    if not isinstance(entry_points, list) or not entry_points:
        raise ValueError("codegen plan contains no entry points")

    module_header = (
        "module attributes {transform.with_named_sequence} {"
        if cpu_codegen == "vector"
        else "module {"
    )
    lines = [module_header]
    seen_names: set[str] = set()
    for entry_point in entry_points:
        name = entry_point["name"]
        kernel = entry_point["kernel"]
        workload = entry_point["workload"]
        inputs = entry_point["inputs"]
        outputs = entry_point["outputs"]
        if name in seen_names or not outputs:
            raise ValueError(f"invalid executable entry point: {name}")
        seen_names.add(name)
        expected_workloads = {
            "matmul_add_relu": ["tiree_input.fused_matmul_add_relu"],
            "matmul": ["tiree_input.matmul"],
            "add": ["tiree_input.add"],
            "relu": ["tiree_input.relu"],
            "fake_quant": ["tiree_input.fake_quant"],
            "split": ["tiree_input.split"],
            "conv2d": ["tiree_input.conv2d"],
            "max_pool2d": ["tiree_input.max_pool2d"],
            "reshape": ["tiree_input.reshape"],
            "transpose": ["tiree_input.transpose"],
            "softmax": ["tiree_input.softmax"],
        }
        if workload != expected_workloads.get(kernel):
            raise ValueError(f"{name}: workload does not match kernel {kernel}")
        output = outputs[0]
        if kernel == "matmul_add_relu":
            body = _matmul(name, inputs, output, fused=True)
        elif kernel == "matmul":
            body = _matmul(name, inputs, output, fused=False)
        elif kernel in ("add", "relu", "fake_quant"):
            body = _elementwise(name, kernel, inputs, output)
        elif kernel == "softmax":
            body = _softmax(name, inputs, output)
        elif kernel == "split":
            body = _split(name, inputs, outputs)
        elif kernel == "conv2d":
            body = _conv2d(name, inputs, output)
        elif kernel == "max_pool2d":
            body = _max_pool2d(name, inputs, output)
        elif kernel == "reshape":
            body = _reshape(name, inputs, output)
        elif kernel == "transpose":
            body = _transpose(name, inputs, output)
        else:
            raise ValueError(f"unsupported HAL kernel: {kernel}")
        lines.extend(body)
        lines.append("")
    if cpu_codegen == "vector":
        lines.extend(
            [
                "  transform.named_sequence private @__transform_main(",
                "      %root: !transform.any_op {transform.readonly}) {",
                '    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root',
                "      : (!transform.any_op) -> !transform.any_op",
                "    %tiled, %m, %n, %k = transform.structured.tile_using_for %matmuls",
                "      tile_sizes [8, 8, 4] : (!transform.any_op) ->",
                "      (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)",
                "    transform.structured.vectorize %tiled vector_sizes [8, 8, 4]",
                "      : !transform.any_op",
                "    transform.yield",
                "  }",
            ]
        )
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="tiny HAL codegen plan JSON")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--cpu-codegen", choices=("scalar", "vector"), default="vector"
    )
    args = parser.parse_args()
    generate(args.input, args.output, args.cpu_codegen)


if __name__ == "__main__":
    main()
