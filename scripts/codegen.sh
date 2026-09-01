#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <HAL MLIR> <output directory>" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
iree_dir="${IREE_DIR:-$(cd "$repo_dir/../iree" && pwd)}"
iree_build="${IREE_BUILD_DIR:-$iree_dir/build_tools/build-host}"
hal_mlir="$1"
output_dir="$2"
mkdir -p "$output_dir"

exporter="$repo_dir/build/bin/tiny-iree-export-codegen"
mlir_opt="$iree_build/llvm-project/bin/mlir-opt"
mlir_translate="$iree_build/llvm-project/bin/mlir-translate"
llc="$iree_build/llvm-project/bin/llc"

"$exporter" "$hal_mlir" -o "$output_dir/module.executable.json"
python3 "$repo_dir/tools/generate_kernel_mlir.py" \
  "$output_dir/module.executable.json" \
  --cpu-codegen=scalar -o "$output_dir/module.executable.mlir"

pipeline='builtin.module(func.func(convert-linalg-to-loops,lower-affine),convert-scf-to-cf,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'
"$mlir_opt" "$output_dir/module.executable.mlir" \
  "--pass-pipeline=$pipeline" -o "$output_dir/module.executable.llvm.mlir"
"$mlir_translate" "$output_dir/module.executable.llvm.mlir" \
  --mlir-to-llvmir -o "$output_dir/module.executable.ll"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) triple=arm64-apple-macosx13.0.0; library=libtiny_iree_kernels.dylib; link=-dynamiclib ;;
  Darwin-x86_64) triple=x86_64-apple-macosx13.0.0; library=libtiny_iree_kernels.dylib; link=-dynamiclib ;;
  Linux-aarch64|Linux-arm64) triple=aarch64-unknown-linux-gnu; library=libtiny_iree_kernels.so; link=-shared ;;
  Linux-x86_64) triple=x86_64-unknown-linux-gnu; library=libtiny_iree_kernels.so; link=-shared ;;
  *) echo "unsupported host" >&2; exit 1 ;;
esac

"$llc" -filetype=obj -mtriple="$triple" -relocation-model=pic \
  "$output_dir/module.executable.ll" -o "$output_dir/module.executable.o"
clang "$link" "$output_dir/module.executable.o" -o "$output_dir/$library"
printf '%s\n' "$output_dir/$library"
