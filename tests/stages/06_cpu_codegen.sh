#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source "$repo_dir/scripts/platform.sh"
build_dir="$(tiny_iree_build_dir "$repo_dir")"
bash "$repo_dir/scripts/build.sh"
cmake --build "$build_dir" --target tiny-iree-export-codegen \
  -j"${JOBS:-4}"
opt="$build_dir/bin/tiny-iree-opt"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-codegen.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$opt" "$repo_dir/examples/mlp.mlir" \
  --pass-pipeline='builtin.module(tiree-global-optimize,tiree-input-to-flow,tiree-flow-to-stream,tiree-stream-to-hal)' \
  -o "$tmp/module.hal.mlir"
bash "$repo_dir/scripts/codegen.sh" "$tmp/module.hal.mlir" "$tmp/aot" \
  >/dev/null

grep -q 'linalg.matmul' "$tmp/aot/module.executable.mlir"
grep -q 'llvm.func' "$tmp/aot/module.executable.llvm.mlir"
grep -q '_mlir_ciface_tiree_kernel_matmul_add_relu' \
  "$tmp/aot/module.executable.ll"
if [[ "$(uname -s)" == Darwin ]]; then
  test -f "$tmp/aot/libtiny_iree_kernels.dylib"
else
  test -f "$tmp/aot/libtiny_iree_kernels.so"
fi
printf 'stage-06 cpu codegen: ok\n'
