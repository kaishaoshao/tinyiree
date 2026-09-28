#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source "$repo_dir/scripts/platform.sh"
build_dir="$(tiny_iree_build_dir "$repo_dir")"
bash "$repo_dir/scripts/build.sh"
opt="$build_dir/bin/tiny-iree-opt"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-stream.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$opt" "$repo_dir/examples/mlp.mlir" --tiree-compile-pipeline \
  -o "$tmp/stream.mlir"
grep -q 'tiree_stream.alloc' "$tmp/stream.mlir"
grep -q 'tiree_stream.dispatch' "$tmp/stream.mlir"
grep -q '!tiree_stream.resource' "$tmp/stream.mlir"
! grep -q 'tiree_flow.dispatch' "$tmp/stream.mlir"

if "$opt" "$repo_dir/tests/invalid_stream_resources.mlir" \
    --tiree-verify-stream-resources -o /dev/null 2>/dev/null; then
  echo "double deallocation unexpectedly passed verification" >&2
  exit 1
fi
printf 'stage-03 stream resources: ok\n'
