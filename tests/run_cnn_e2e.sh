#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <python-with-onnx> <tiny-opt> <tiny-runtime>" >&2
  exit 2
fi
python="$1"
opt="$2"
runtime="$3"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-cnn.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$python" "$repo_dir/tools/generate_cnn_onnx.py" -o "$tmp/cnn.onnx"
"$python" "$repo_dir/tools/tiny_iree_compile.py" "$tmp/cnn.onnx" \
  -o "$tmp/cnn.tiree" --opt="$opt" --cpu-codegen=scalar

grep -q 'tiree_input.conv2d' "$tmp/cnn.tiree/module.input.mlir"
grep -q 'tiree_input.max_pool2d' "$tmp/cnn.tiree/module.input.mlir"
grep -q 'tiree_input.reshape' "$tmp/cnn.tiree/module.input.mlir"

input='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16'
actual="$($runtime "$tmp/cnn.tiree" --function=predict --input="$input")"
reference="$("$python" "$repo_dir/tools/run_onnx_reference.py" \
  "$tmp/cnn.onnx" --input="$input")"
printf '%s\nCNN reference:\n%s\n' "$actual" "$reference"

"$python" - "$actual" "$reference" <<'PY'
import math
import re
import sys

def values(text):
    match = re.search(r"result: \[([^]]+)\]", text)
    if not match:
        raise SystemExit("missing result")
    return [float(value) for value in match.group(1).split(",")]

actual = values(sys.argv[1])
reference = values(sys.argv[2])
if not all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
           for a, b in zip(actual, reference)):
    raise SystemExit(f"CNN mismatch: {actual} != {reference}")
PY
grep -q 'argmax: 1' <<<"$actual"
