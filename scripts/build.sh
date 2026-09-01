#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
iree_dir="${IREE_DIR:-$(cd "$repo_dir/../iree" && pwd)}"
iree_build="${IREE_BUILD_DIR:-$iree_dir/build_tools/build-host}"
build_dir="${TINY_IREE_BUILD_DIR:-$repo_dir/build}"

cmake -G Ninja -S "$repo_dir" -B "$build_dir" \
  -DMLIR_DIR="$iree_build/lib/cmake/mlir" \
  -DLLVM_DIR="$iree_build/llvm-project/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$build_dir" --target tiny-iree-opt -j"${JOBS:-4}"

