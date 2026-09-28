#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source "$repo_dir/scripts/platform.sh"
build_dir="$(tiny_iree_build_dir "$repo_dir")"
bash "$repo_dir/scripts/build.sh"
cmake --build "$build_dir" \
  --target tiny-iree-translate tiny-iree-export-codegen tiny-iree-run-module \
  -j"${JOBS:-4}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-runtime.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

python3 "$repo_dir/tools/tiny_iree_compile.py" \
  "$repo_dir/examples/runtime_inputs.mlir" \
  -o "$tmp/mlp.tiree" --cpu-codegen=scalar
actual="$($build_dir/bin/tiny-iree-run-module "$tmp/mlp.tiree" \
  --function=predict \
  --input='1,2,3,4;1,0,0,0,1,0,0,0,1,1,1,1;-1,-2,-3')"

printf '%s\n' "$actual"
grep -q 'backend: native-aot' <<<"$actual"
grep -q 'vm: tiny-bytecode-v1' <<<"$actual"
grep -q 'result: \[4.000000e+00, 4.000000e+00, 4.000000e+00\]' \
  <<<"$actual"
printf 'stage-07 runtime: ok\n'
