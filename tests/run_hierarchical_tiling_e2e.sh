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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-hierarchical.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/hierarchical.onnx"
bundle="$temporary_dir/hierarchical.tiree"
"$python" "$project_dir/tools/generate_tiled_onnx.py" -o "$model" --size=8
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt" --cpu-codegen=hierarchical \
  --parallel-threads=4 --l2-tile=4,4,8 --l1-tile=4,4,4 \
  --register-tile=2,2

grep -q 'l2_tile = array<i64: 4, 4, 8>' "$bundle/module.executable.mlir"
grep -q 'l1_tile = array<i64: 4, 4, 4>' "$bundle/module.executable.mlir"
grep -q 'register_tile = array<i64: 2, 2>' "$bundle/module.executable.mlir"
grep -q 'omp.parallel' "$bundle/module.executable.mlir"
grep -q 'vector.load' "$bundle/module.executable.mlir"
grep -q '__kmpc_fork_call' "$bundle/module.executable.ll"

input="$($python - <<'PY'
print(",".join(str((index % 17) - 8) for index in range(8 * 8)))
PY
)"
output="$($runtime "$bundle" --function=predict --input="$input")"
printf '%s\n' "$output"
"$python" - "$output" "$input" <<'PY'
import math
import re
import sys

match = re.search(r"result: \[([^]]+)\]", sys.argv[1])
if not match:
    raise SystemExit("result line not found")
actual = [float(value) for value in match.group(1).split(",")]
expected = [float(value) for value in sys.argv[2].split(",")]
if len(actual) != len(expected) or any(
    not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
    for left, right in zip(actual, expected)
):
    raise SystemExit("hierarchical identity MatMul produced an unexpected result")
PY
