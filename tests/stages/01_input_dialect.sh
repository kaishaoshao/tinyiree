#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
bash "$repo_dir/scripts/build.sh"
opt="$repo_dir/build/bin/tiny-iree-opt"

"$opt" "$repo_dir/examples/mlp.mlir" --verify-each -o /dev/null
if "$opt" "$repo_dir/tests/invalid_input.mlir" -o /dev/null 2>/dev/null; then
  echo "invalid MatMul shape unexpectedly passed verification" >&2
  exit 1
fi
printf 'stage-01 input dialect: ok\n'

