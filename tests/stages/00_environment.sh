#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source "$repo_dir/scripts/platform.sh"
toolchain_root="$(tiny_iree_find_toolchain_root "$repo_dir")"
mlir_dir="$(tiny_iree_mlir_cmake_dir "$toolchain_root")"
llvm_dir="$(tiny_iree_llvm_cmake_dir "$toolchain_root")"

command -v cmake >/dev/null
command -v python3 >/dev/null
command -v clang++ >/dev/null
test -f "$mlir_dir/MLIRConfig.cmake"
test -f "$llvm_dir/LLVMConfig.cmake"
tiny_iree_find_llvm_tool "$toolchain_root" mlir-opt >/dev/null
tiny_iree_find_llvm_tool "$toolchain_root" mlir-translate >/dev/null
tiny_iree_find_llvm_tool "$toolchain_root" llc >/dev/null

printf 'environment: ok\n'
printf 'host: %s\n' "$(tiny_iree_host_id)"
printf 'LLVM/MLIR build: %s\n' "$toolchain_root"
