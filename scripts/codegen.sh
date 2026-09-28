#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <HAL MLIR> <output directory>" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo_dir/scripts/platform.sh"

toolchain_root="$(tiny_iree_find_toolchain_root "$repo_dir")"
build_dir="$(tiny_iree_build_dir "$repo_dir")"
hal_mlir="$1"
output_dir="$2"
mkdir -p "$output_dir"

exporter="$build_dir/bin/tiny-iree-export-codegen"
mlir_opt="$(tiny_iree_find_llvm_tool "$toolchain_root" mlir-opt)"
mlir_translate="$(tiny_iree_find_llvm_tool "$toolchain_root" mlir-translate)"
llc="$(tiny_iree_find_llvm_tool "$toolchain_root" llc)"
clang="${TINY_IREE_CLANG:-$(command -v clang)}"

if [[ ! -x "$exporter" ]]; then
  printf 'tiny-iree-export-codegen not found: %s\n' "$exporter" >&2
  printf 'build it with: cmake --build %q --target tiny-iree-export-codegen\n' \
    "$build_dir" >&2
  exit 1
fi

"$exporter" "$hal_mlir" -o "$output_dir/module.executable.json"
python3 "$repo_dir/tools/generate_kernel_mlir.py" \
  "$output_dir/module.executable.json" \
  --cpu-codegen=scalar -o "$output_dir/module.executable.mlir"

pipeline='builtin.module(func.func(convert-linalg-to-loops,lower-affine),convert-scf-to-cf,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'
"$mlir_opt" "$output_dir/module.executable.mlir" \
  "--pass-pipeline=$pipeline" -o "$output_dir/module.executable.llvm.mlir"
"$mlir_translate" "$output_dir/module.executable.llvm.mlir" \
  --mlir-to-llvmir -o "$output_dir/module.executable.ll"

triple="$(tiny_iree_target_triple)"
library="$(tiny_iree_library_name)"
link="$(tiny_iree_link_mode)"

"$llc" -filetype=obj -mtriple="$triple" -relocation-model=pic \
  "$output_dir/module.executable.ll" -o "$output_dir/module.executable.o"
"$clang" "$link" "$output_dir/module.executable.o" -o "$output_dir/$library"
printf '%s\n' "$output_dir/$library"
