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


def _host_configuration() -> tuple[str, str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return system, "darwin-arm64", "arm64-apple-macosx13.0.0"
    if system == "Darwin" and machine in ("x86_64", "amd64"):
        return system, "darwin-x86_64", "x86_64-apple-macosx13.0.0"
    if system == "Linux" and machine in ("arm64", "aarch64"):
        return system, "linux-aarch64", "aarch64-unknown-linux-gnu"
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return system, "linux-x86_64", "x86_64-unknown-linux-gnu"
    raise ValueError(f"unsupported native target: {system}-{machine}")


def _default_build_dir(repository_root: Path) -> Path:
    configured = os.environ.get("TINY_IREE_BUILD_DIR")
    if configured:
        return Path(configured)

    default = repository_root / "build"
    cache = default / "CMakeCache.txt"
    if cache.is_file():
        prefix = "CMAKE_HOME_DIRECTORY:INTERNAL="
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix):
                cached_source = Path(line[len(prefix) :])
                if cached_source != repository_root:
                    _, host_id, _ = _host_configuration()
                    return repository_root / f"build-{host_id}"
                break
    return default


def _has_cmake_packages(root: Path) -> bool:
    return (root / "lib/cmake/mlir/MLIRConfig.cmake").is_file() and (
        (root / "lib/cmake/llvm/LLVMConfig.cmake").is_file()
        or (root / "llvm-project/lib/cmake/llvm/LLVMConfig.cmake").is_file()
    )


def _find_toolchain_root(repository_root: Path) -> Path:
    candidates: list[Path] = []
    for variable in ("IREE_BUILD_DIR", "LLVM_BUILD_DIR"):
        if value := os.environ.get(variable):
            candidates.append(Path(value))
    if value := os.environ.get("IREE_DIR"):
        iree_root = Path(value)
        candidates.extend(
            [iree_root / "build_tools/build-host", iree_root / "build"]
        )
    candidates.extend(
        [
            repository_root.parent / "iree/build_tools/build-host",
            repository_root.parent / "iree/build",
            repository_root.parent / "llvm-project/build",
        ]
    )
    for candidate in candidates:
        if _has_cmake_packages(candidate):
            return candidate.resolve()
    raise ValueError(
        "unable to find LLVM/MLIR; set IREE_BUILD_DIR or LLVM_BUILD_DIR"
    )


def _find_llvm_tool(toolchain_root: Path, name: str) -> Path:
    candidates = [
        toolchain_root / "llvm-project/bin" / name,
        toolchain_root / "bin" / name,
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


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
    parallel_threads: int,
    l2_tile: str,
    l1_tile: str,
    register_tile: str,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    output_path.mkdir(parents=True, exist_ok=True)
    input_mlir = output_path / "module.input.mlir"

    system, host_id, target_triple = _host_configuration()
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
            "--l2-tile",
            l2_tile,
            "--l1-tile",
            l1_tile,
            "--register-tile",
            register_tile,
            "--parallel-threads",
            str(parallel_threads),
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
    function_lowering = (
        "func.func(convert-linalg-to-loops,lower-vector-mask,"
        "lower-vector-multi-reduction,convert-vector-to-scf{target-rank=1},"
        "expand-strided-metadata,lower-affine,convert-scf-to-cf)"
    )
    if cpu_codegen == "hierarchical":
        lowering_pipeline = (
            f"builtin.module({function_lowering},"
            "convert-openmp-to-llvm,convert-vector-to-llvm,convert-ub-to-llvm,"
            "convert-math-to-llvm,convert-arith-to-llvm,"
            "finalize-memref-to-llvm,convert-func-to-llvm,"
            "convert-cf-to-llvm,reconcile-unrealized-casts)"
        )
    else:
        lowering_pipeline = (
            f"builtin.module({function_lowering},convert-scf-to-cf,"
            "convert-vector-to-llvm,convert-ub-to-llvm,convert-math-to-llvm,"
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

    library_name = (
        "libtiny_iree_kernels.dylib"
        if system == "Darwin"
        else "libtiny_iree_kernels.so"
    )
    library_path = output_path / library_name
    link_mode = "-dynamiclib" if system == "Darwin" else "-shared"
    link_command = [str(clang_path), link_mode, str(object_path)]
    if cpu_codegen == "hierarchical":
        if system == "Darwin":
            openmp_candidates: list[Path] = []
            if configured_lib_dir := os.environ.get(
                "TINY_IREE_OPENMP_LIB_DIR"
            ):
                openmp_candidates.append(Path(configured_lib_dir))
            if brew := shutil.which("brew"):
                brew_prefix = subprocess.run(
                    [brew, "--prefix", "libomp"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if brew_prefix.returncode == 0 and brew_prefix.stdout.strip():
                    openmp_candidates.append(
                        Path(brew_prefix.stdout.strip()) / "lib"
                    )
            openmp_lib = next(
                (path for path in openmp_candidates if (path / "libomp.dylib").is_file()),
                None,
            )
            if openmp_lib is None:
                raise ValueError(
                    "hierarchical codegen requires libomp; install it with "
                    "`brew install libomp` or set TINY_IREE_OPENMP_LIB_DIR"
                )
            link_command.extend(
                [f"-L{openmp_lib}", "-lomp", f"-Wl,-rpath,{openmp_lib}"]
            )
        else:
            link_command.append("-fopenmp")
    link_command.extend(["-o", str(library_path)])
    _run(link_command)

    entry_points = sorted(entry["name"] for entry in codegen_metadata["entry_points"])
    manifest = {
        "format": "tiny-iree-bundle-v1",
        "source": str(input_path.resolve()),
        "function": function_name,
        "target": host_id,
        "vm_module": vm_bytecode.name,
        "vm_format": "tiny-vm-bytecode-v1",
        "hal_executable": library_path.name,
        "codegen": f"mlir-linalg-{cpu_codegen}-llvm",
        "codegen_config": {
            "parallel_threads": parallel_threads if cpu_codegen == "hierarchical" else 1,
            "l2_tile": [int(value) for value in l2_tile.split(",")],
            "l1_tile": [int(value) for value in l1_tile.split(",")],
            "register_tile": [int(value) for value in register_tile.split(",")],
        },
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
    toolchain_root = _find_toolchain_root(repository_root)
    build_dir = _default_build_dir(repository_root)
    default_opt = build_dir / "bin" / "tiny-iree-opt"
    default_translate = build_dir / "bin" / "tiny-iree-translate"
    default_export_codegen = (
        build_dir / "bin" / "tiny-iree-export-codegen"
    )
    default_clang = shutil.which("clang") or "clang"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--opt", type=Path, default=default_opt)
    parser.add_argument("--translate", type=Path, default=default_translate)
    parser.add_argument("--export-codegen", type=Path, default=default_export_codegen)
    parser.add_argument("--clang", type=Path, default=Path(default_clang))
    parser.add_argument(
        "--mlir-opt", type=Path,
        default=_find_llvm_tool(toolchain_root, "mlir-opt")
    )
    parser.add_argument(
        "--mlir-translate", type=Path,
        default=_find_llvm_tool(toolchain_root, "mlir-translate")
    )
    parser.add_argument(
        "--llc", type=Path, default=_find_llvm_tool(toolchain_root, "llc")
    )
    parser.add_argument("--function", default="predict")
    parser.add_argument(
        "--cpu-codegen",
        choices=("scalar", "vector", "hierarchical"),
        default="vector",
    )
    parser.add_argument("--parallel-threads", type=int, default=4)
    parser.add_argument("--l2-tile", default="64,64,32")
    parser.add_argument("--l1-tile", default="16,16,8")
    parser.add_argument("--register-tile", default="4,4")
    parser.add_argument(
        "--target-backend", choices=("llvm-cpu",), default="llvm-cpu"
    )
    args = parser.parse_args()
    if args.parallel_threads <= 0:
        parser.error("--parallel-threads must be positive")
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
        args.parallel_threads,
        args.l2_tile,
        args.l1_tile,
        args.register_tile,
    )


if __name__ == "__main__":
    main()
