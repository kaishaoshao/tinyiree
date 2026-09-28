#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo_dir/scripts/platform.sh"

toolchain_root="$(tiny_iree_find_toolchain_root "$repo_dir")"
mlir_dir="$(tiny_iree_mlir_cmake_dir "$toolchain_root")"
llvm_dir="$(tiny_iree_llvm_cmake_dir "$toolchain_root")"
build_dir="$(tiny_iree_build_dir "$repo_dir")"

printf 'host: %s\n' "$(tiny_iree_host_id)"
printf 'toolchain: %s\n' "$toolchain_root"
printf 'build: %s\n' "$build_dir"

cmake -G Ninja -S "$repo_dir" -B "$build_dir" \
  -DMLIR_DIR="$mlir_dir" \
  -DLLVM_DIR="$llvm_dir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$build_dir" --target tiny-iree-aot-tools -j"${JOBS:-4}"
