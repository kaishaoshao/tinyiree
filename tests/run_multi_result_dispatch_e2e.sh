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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-multi-result.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/split.onnx"
bundle="$temporary_dir/split.tiree"
"$python" "$project_dir/tools/generate_split_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"

grep -Eq '%[0-9]+:2 = "tiree_flow.dispatch"' "$bundle/module.flow.mlir"
grep -q 'result_bytes = array<i64: 8, 8>' "$bundle/module.stream.mlir"
grep -q 'func.func @tiree_kernel_split_0' "$bundle/module.executable.mlir"
output="$("$runtime" "$bundle" --function=predict --input='1,2,3,4')"
printf '%s\n' "$output"
grep -q 'result\[0\]: \[1.000000e+00, 2.000000e+00\]' <<<"$output"
grep -q 'result\[1\]: \[3.000000e+00, 4.000000e+00\]' <<<"$output"
