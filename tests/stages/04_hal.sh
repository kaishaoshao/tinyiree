#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
bash "$repo_dir/scripts/build.sh"
opt="$repo_dir/build/bin/tiny-iree-opt"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-hal.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$opt" "$repo_dir/examples/mlp.mlir" --tiree-compile-pipeline \
  -o "$tmp/hal.mlir"
grep -q 'tiree_hal.executable' "$tmp/hal.mlir"
grep -q 'tiree_hal.dispatch' "$tmp/hal.mlir"
grep -q '!tiree_hal.buffer' "$tmp/hal.mlir"
grep -q 'device = "cpu-sync"' "$tmp/hal.mlir"
! grep -q 'tiree_flow.dispatch' "$tmp/hal.mlir"
! grep -q 'tiree_stream.dispatch' "$tmp/hal.mlir"
printf 'stage-04 hal: ok\n'
