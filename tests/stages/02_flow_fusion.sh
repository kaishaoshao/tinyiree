#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
bash "$repo_dir/scripts/build.sh"
opt="$repo_dir/build/bin/tiny-iree-opt"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-flow.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$opt" "$repo_dir/examples/mlp.mlir" --tiree-global-optimize \
  -o "$tmp/fused.mlir"
grep -q 'tiree_input.fused_matmul_add_relu' "$tmp/fused.mlir"
! grep -q 'tiree_input.matmul' "$tmp/fused.mlir"

"$opt" "$repo_dir/examples/mlp.mlir" --tiree-compile-pipeline \
  -o "$tmp/flow.mlir"
grep -q 'tiree_flow.dispatch' "$tmp/flow.mlir"
grep -q 'kernel = "matmul_add_relu"' "$tmp/flow.mlir"

if "$opt" "$repo_dir/tests/invalid_flow_region.mlir" -o /dev/null 2>/dev/null; then
  echo "invalid flow region unexpectedly passed verification" >&2
  exit 1
fi
printf 'stage-02 fusion and flow: ok\n'
