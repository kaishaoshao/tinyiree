#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
bash "$repo_dir/scripts/build.sh"
cmake --build "$repo_dir/build" --target tiny-iree-translate -j"${JOBS:-4}"
opt="$repo_dir/build/bin/tiny-iree-opt"
translate="$repo_dir/build/bin/tiny-iree-translate"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-vm.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$opt" "$repo_dir/examples/mlp.mlir" --tiree-compile-pipeline \
  -o "$tmp/module.vm.mlir"
grep -q 'tiree_vm.alloc' "$tmp/module.vm.mlir"
grep -q 'tiree_vm.call' "$tmp/module.vm.mlir"
grep -q '!tiree_vm.ref' "$tmp/module.vm.mlir"
! grep -q 'tiree_hal.dispatch' "$tmp/module.vm.mlir"

"$translate" "$tmp/module.vm.mlir" -o "$tmp/module.tvm"
test "$(od -An -tx1 -N8 "$tmp/module.tvm" | tr -d ' \n')" = \
  "54495245564d3100"
printf 'stage-05 vm bytecode: ok\n'
