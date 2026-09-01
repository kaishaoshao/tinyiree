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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-dynamic.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/dynamic.onnx"
bundle="$temporary_dir/dynamic.tiree"
"$python" "$project_dir/tools/generate_dynamic_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"

grep -q 'tensor<?x?xf32>' "$bundle/module.input.mlir"
grep -q 'bytes = -1 : i64' "$bundle/module.vm.mlir"
grep -q 'memref<?x?xf32>' "$bundle/module.executable.mlir"
output="$($runtime "$bundle" --function=predict --input='2x3=1,-2,3,-4,5,-6')"
printf '%s\n' "$output"
grep -q 'vm: tiny-bytecode-v1' <<<"$output"
grep -q 'resources: peak=24B live=24B' <<<"$output"
grep -q 'result: \[1.000000e+00, 0.000000e+00, 3.000000e+00, 0.000000e+00, 5.000000e+00, 0.000000e+00\]' <<<"$output"
