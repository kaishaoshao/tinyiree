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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-tiling.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/tiled.onnx"
bundle="$temporary_dir/tiled.tiree"
"$python" "$project_dir/tools/generate_tiled_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt" --cpu-codegen=vector

grep -q 'tile_sizes \[8, 8, 4\]' "$bundle/module.executable.mlir"
grep -q 'scf.for' "$bundle/module.executable.optimized.mlir"
grep -q 'vector.transfer_read' "$bundle/module.executable.optimized.mlir"

input="$($python - <<'PY'
print(",".join("1" for _ in range(16 * 16)))
PY
)"
output="$("$runtime" "$bundle" --function=predict --input="$input")"
printf '%s\n' "$output"
"$python" - "$output" <<'PY'
import math
import re
import sys

match = re.search(r"result: \[([^]]+)\]", sys.argv[1])
if not match:
    raise SystemExit("result line not found")
values = [float(value) for value in match.group(1).split(",")]
if len(values) != 256 or not all(math.isclose(value, 1.0) for value in values):
    raise SystemExit("tiled identity MatMul produced an unexpected result")
PY
