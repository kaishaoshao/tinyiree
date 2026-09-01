#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
iree_dir="${IREE_DIR:-$(cd "$repo_dir/../iree" && pwd)}"
build_dir="${IREE_BUILD_DIR:-$iree_dir/build_tools/build-host}"

command -v cmake >/dev/null
command -v python3 >/dev/null
command -v clang++ >/dev/null
test -f "$build_dir/lib/cmake/mlir/MLIRConfig.cmake"
test -x "$build_dir/llvm-project/bin/mlir-opt"
test -x "$build_dir/llvm-project/bin/mlir-translate"
test -x "$build_dir/llvm-project/bin/llc"

printf 'environment: ok\n'
printf 'iree build: %s\n' "$build_dir"

