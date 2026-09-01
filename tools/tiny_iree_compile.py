#!/usr/bin/env python3
"""Compiles ONNX or input-dialect MLIR into a tiny-iree native bundle."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def compile_bundle(
    input_path: Path,
    output_path: Path,
    opt_path: Path,
    translate_path: Path,
    export_codegen_path: Path,
    clang_path: Path,
    mlir_opt_path: Path,
    mlir_translate_path: Path,
    llc_path: Path,
    function_name: str,
    cpu_codegen: str,
    target_backend: str,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    output_path.mkdir(parents=True, exist_ok=True)
    input_mlir = output_path / "module.input.mlir"

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        target_triple = "arm64-apple-macosx13.0.0"
    elif system == "Darwin" and machine == "x86_64":
        target_triple = "x86_64-apple-macosx13.0.0"
    elif system == "Linux" and machine in ("arm64", "aarch64"):
        target_triple = "aarch64-unknown-linux-gnu"
    elif system == "Linux" and machine == "x86_64":
        target_triple = "x86_64-unknown-linux-gnu"
    else:
        raise ValueError(f"unsupported native target: {system}-{machine}")
    if target_backend != "llvm-cpu":
        raise ValueError(f"unsupported HAL target backend: {target_backend}")

    if input_path.suffix.lower() == ".onnx":
        _run(
            [
                sys.executable,
                str(source_root / "tools" / "import_onnx.py"),
                str(input_path),
                "-o",
                str(input_mlir),
                "--function",
                function_name,
            ]
        )
    elif input_path.suffix.lower() == ".mlir":
        shutil.copyfile(input_path, input_mlir)
    else:
        raise ValueError("input must be an .onnx or .mlir file")

    optimized_input_mlir = output_path / "module.global-opt.mlir"
    flow_mlir = output_path / "module.flow.mlir"
    stream_mlir = output_path / "module.stream.mlir"
    hal_mlir = output_path / "module.hal.mlir"
    vm_mlir = output_path / "module.vm.mlir"
    vm_mlir_bytecode = output_path / "module.vm.mlirbc"
    vm_bytecode = output_path / "module.tvm"
    _run(
        [
            str(opt_path),
            str(input_mlir),
            "--tiree-global-optimize",
            "-o",
            str(optimized_input_mlir),
        ]
    )
    _run(
        [str(opt_path), str(optimized_input_mlir), "--tiree-input-to-flow",
         "-o", str(flow_mlir)]
    )
    _run(
        [str(opt_path), str(flow_mlir), "--tiree-flow-to-stream",
         "-o", str(stream_mlir)]
    )
    _run(
        [str(opt_path), str(stream_mlir), "--tiree-stream-to-hal",
         "-o", str(hal_mlir)]
    )
    _run(
        [str(opt_path), str(hal_mlir), "--tiree-hal-to-vm",
         "-o", str(vm_mlir)]
    )
    _run([str(opt_path), str(vm_mlir), "--emit-bytecode", "-o", str(vm_mlir_bytecode)])
    _run([str(translate_path), str(vm_mlir), "-o", str(vm_bytecode)])

    kernel_mlir = output_path / "module.executable.mlir"
    codegen_plan = output_path / "module.executable.json"
    optimized_mlir = output_path / "module.executable.optimized.mlir"
    llvm_mlir = output_path / "module.executable.llvm.mlir"
    llvm_ir = output_path / "module.executable.ll"
    object_path = output_path / "module.executable.o"
    _run([str(export_codegen_path), str(hal_mlir), "-o", str(codegen_plan)])
    codegen_metadata = json.loads(codegen_plan.read_text(encoding="utf-8"))
    codegen_metadata["target"] = {
        "device": "cpu-sync",
        "backend": target_backend,
        "triple": target_triple,
    }
    codegen_plan.write_text(
        json.dumps(codegen_metadata, indent=2) + "\n", encoding="utf-8"
    )
    _run(
        [
            sys.executable,
            str(source_root / "tools" / "generate_kernel_mlir.py"),
            str(codegen_plan),
            "-o",
            str(kernel_mlir),
            "--cpu-codegen",
            cpu_codegen,
        ]
    )
    if cpu_codegen == "vector":
        _run(
            [
                str(mlir_opt_path),
                str(kernel_mlir),
                "--pass-pipeline=builtin.module("
                "transform-interpreter,symbol-dce,canonicalize,cse)",
                "-o",
                str(optimized_mlir),
            ]
        )
    else:
        shutil.copyfile(kernel_mlir, optimized_mlir)
    lowering_pipeline = (
        "builtin.module(func.func(convert-linalg-to-loops,lower-vector-mask,"
        "lower-vector-multi-reduction,convert-vector-to-scf{target-rank=1},"
        "expand-strided-metadata,lower-affine),"
        "convert-scf-to-cf,"
        "convert-vector-to-llvm,convert-ub-to-llvm,"
        "convert-math-to-llvm,"
        "convert-arith-to-llvm,finalize-memref-to-llvm,"
        "convert-func-to-llvm,convert-cf-to-llvm,"
        "reconcile-unrealized-casts)"
    )
    _run(
        [
            str(mlir_opt_path),
            str(optimized_mlir),
            f"--pass-pipeline={lowering_pipeline}",
            "-o",
            str(llvm_mlir),
        ]
    )
    _run(
        [
            str(mlir_translate_path),
            str(llvm_mlir),
            "--mlir-to-llvmir",
            "-o",
            str(llvm_ir),
        ]
    )

    _run(
        [
            str(llc_path),
            "-filetype=obj",
            f"-mtriple={target_triple}",
            "-relocation-model=pic",
            str(llvm_ir),
            "-o",
            str(object_path),
        ]
    )

    library_name = "libtiny_iree_kernels.dylib" if system == "Darwin" else "libtiny_iree_kernels.so"
    library_path = output_path / library_name
    link_mode = "-dynamiclib" if system == "Darwin" else "-shared"
    _run(
        [
            str(clang_path),
            link_mode,
            str(object_path),
            "-o",
            str(library_path),
        ]
    )

    entry_points = sorted(entry["name"] for entry in codegen_metadata["entry_points"])
    manifest = {
        "format": "tiny-iree-bundle-v1",
        "source": str(input_path.resolve()),
        "function": function_name,
        "target": f"{platform.system().lower()}-{platform.machine()}",
        "vm_module": vm_bytecode.name,
        "vm_format": "tiny-vm-bytecode-v1",
        "hal_executable": library_path.name,
        "codegen": f"mlir-linalg-{cpu_codegen}-llvm",
        "hal": {
            "device": "cpu-sync",
            "target_backend": target_backend,
            "target_triple": target_triple,
        },
        "entry_points": entry_points,
        "pipeline": [
            "onnx-import",
            "global-optimization",
            "flow-dispatch-formation",
            "stream-resource-scheduling",
            "hal-executable-translation",
            "vm-bytecode-serialization",
            "cpu-executable-codegen",
        ],
        "artifacts": {
            "input": input_mlir.name,
            "global_optimization": optimized_input_mlir.name,
            "flow": flow_mlir.name,
            "stream": stream_mlir.name,
            "hal": hal_mlir.name,
            "vm": vm_mlir.name,
            "codegen_plan": codegen_plan.name,
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"compiled {input_path} -> {output_path}")
    print(f"target: {manifest['target']}")
    print("entry points: " + ", ".join(entry_points))


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    iree_root = Path(os.environ.get("IREE_DIR", repository_root.parent / "iree"))
    iree_build = Path(
        os.environ.get("IREE_BUILD_DIR", iree_root / "build_tools" / "build-host")
    )
    default_opt = repository_root / "build" / "bin" / "tiny-iree-opt"
    default_translate = repository_root / "build" / "bin" / "tiny-iree-translate"
    default_export_codegen = (
        repository_root / "build" / "bin" / "tiny-iree-export-codegen"
    )
    llvm_bin = iree_build / "llvm-project" / "bin"
    default_clang = shutil.which("clang") or "clang"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--opt", type=Path, default=default_opt)
    parser.add_argument("--translate", type=Path, default=default_translate)
    parser.add_argument("--export-codegen", type=Path, default=default_export_codegen)
    parser.add_argument("--clang", type=Path, default=Path(default_clang))
    parser.add_argument("--mlir-opt", type=Path, default=llvm_bin / "mlir-opt")
    parser.add_argument(
        "--mlir-translate", type=Path, default=llvm_bin / "mlir-translate"
    )
    parser.add_argument("--llc", type=Path, default=llvm_bin / "llc")
    parser.add_argument("--function", default="predict")
    parser.add_argument(
        "--cpu-codegen", choices=("scalar", "vector"), default="vector"
    )
    parser.add_argument(
        "--target-backend", choices=("llvm-cpu",), default="llvm-cpu"
    )
    args = parser.parse_args()
    for name, path in (
        ("tiny-iree-opt", args.opt),
        ("tiny-iree-translate", args.translate),
        ("tiny-iree-export-codegen", args.export_codegen),
    ):
        if not path.is_file():
            parser.error(f"{name} not found: {path}")
    for name, path in (
        ("mlir-opt", args.mlir_opt),
        ("mlir-translate", args.mlir_translate),
        ("llc", args.llc),
    ):
        if not path.is_file():
            parser.error(f"{name} not found: {path}")
    compile_bundle(
        args.input,
        args.output,
        args.opt,
        args.translate,
        args.export_codegen,
        args.clang,
        args.mlir_opt,
        args.mlir_translate,
        args.llc,
        args.function,
        args.cpu_codegen,
        args.target_backend,
    )


if __name__ == "__main__":
    main()
