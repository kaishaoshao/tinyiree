#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <python-with-onnx> <tiny-iree-opt> <tiny-iree-run-module>" >&2
  exit 1
fi

python="$1"
opt="$2"
runtime="$3"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-reuse.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/reuse.onnx"
bundle="$temporary_dir/reuse.tiree"
"$python" "$project_dir/tools/generate_reuse_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"

test "$(grep -c 'tiree_stream.dealloc' "$bundle/module.stream.mlir")" -eq 2
output="$("$runtime" "$bundle" --function=predict --input='1,-2,3,-4')"
printf '%s\n' "$output"
grep -q 'resources: peak=32B live=16B' <<<"$output"
grep -q 'allocator: new=2 reused=1' <<<"$output"
grep -q 'result: \[1.000000e+00, 0.000000e+00, 3.000000e+00, 0.000000e+00\]' \
  <<<"$output"
