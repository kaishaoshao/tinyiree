#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <python-with-onnx> <tiny-iree-opt> <tiny-runtime> <iree-compile> <iree-run-module>" >&2
  exit 1
fi

python="$1"
opt="$2"
tiny_runtime="$3"
iree_compile="$4"
iree_runtime="$5"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-official.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/mlp.onnx"
bundle="$temporary_dir/mlp.tiree"
vmfb="$temporary_dir/mlp.vmfb"

"$python" "$project_dir/tools/generate_mlp_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"
"$iree_compile" "$project_dir/examples/official_mlp_stablehlo.mlir" \
  --iree-input-type=stablehlo \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  -o "$vmfb"

tiny_output="$("$tiny_runtime" "$bundle" --function=predict --input=1,2,3,4)"
official_output="$("$iree_runtime" --module="$vmfb" --device=local-task \
  --function=predict --input='1x4xf32=1 2 3 4')"
printf 'Tiny-IREE:\n%s\n' "$tiny_output"
printf 'Official IREE VMFB:\n%s\n' "$official_output"

"$python" - "$tiny_output" "$official_output" <<'PY'
import math
import re
import sys

def tiny_values(text: str) -> list[float]:
    match = re.search(r"result: \[([^]]+)\]", text)
    if not match:
        raise SystemExit("Tiny-IREE result line not found")
    return [float(value) for value in match.group(1).split(",")]

def official_values(text: str) -> list[float]:
    match = re.search(r"\b[0-9?x]+xf32=\[([^]]+)\]", text)
    if not match:
        raise SystemExit("official IREE result line not found")
    return [float(value) for value in match.group(1).replace(",", " ").split()]

tiny = tiny_values(sys.argv[1])
official = official_values(sys.argv[2])
if len(tiny) != len(official) or not all(
    math.isclose(lhs, rhs, rel_tol=1e-5, abs_tol=1e-6)
    for lhs, rhs in zip(tiny, official)
):
    raise SystemExit(f"Tiny-IREE {tiny} != official IREE {official}")
PY
