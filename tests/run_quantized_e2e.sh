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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-quantized.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/quantized.onnx"
bundle="$temporary_dir/quantized.tiree"
input='-1.26,0.14,2.26,13.0'

"$python" "$project_dir/tools/generate_quantized_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"

grep -q 'tiree_input.fake_quant' "$bundle/module.input.mlir"
grep -q 'kernel = "fake_quant"' "$bundle/module.hal.mlir"
grep -q 'math.roundeven' "$bundle/module.executable.mlir"
actual="$("$runtime" "$bundle" --function=predict --input="$input")"
reference="$("$python" "$project_dir/tools/run_onnx_reference.py" \
  "$model" --input="$input")"
printf '%s\n' "$actual"
printf 'ONNX reference:\n%s\n' "$reference"

"$python" - "$actual" "$reference" <<'PY'
import math
import re
import sys

def values(text: str) -> list[float]:
    match = re.search(r"result: \[([^]]+)\]", text)
    if not match:
        raise SystemExit("result line not found")
    return [float(value) for value in match.group(1).split(",")]

actual = values(sys.argv[1])
reference = values(sys.argv[2])
if len(actual) != len(reference) or not all(
    math.isclose(lhs, rhs, rel_tol=1e-6, abs_tol=1e-6)
    for lhs, rhs in zip(actual, reference)
):
    raise SystemExit(f"native result {actual} != ONNX reference {reference}")
PY
