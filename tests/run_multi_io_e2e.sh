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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-multi.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/multi.onnx"
bundle="$temporary_dir/multi.tiree"
"$python" "$project_dir/tools/generate_multi_io_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"

grep -q 'func.func @predict(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>)' \
  "$bundle/module.input.mlir"
output="$($runtime "$bundle" --function=predict --input='1,-2,3;4,1,-6')"
printf '%s\n' "$output"
grep -q 'vm: tiny-bytecode-v1' <<<"$output"
grep -q 'result\[0\]: \[5.000000e+00, -1.000000e+00, -3.000000e+00\]' <<<"$output"
grep -q 'result\[1\]: \[5.000000e+00, 0.000000e+00, 0.000000e+00\]' <<<"$output"
