#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
iree_dir="${IREE_DIR:-$(cd "$repo_dir/../iree" && pwd)}"
default_python="$iree_dir/build_tools/build-host/tiny-iree-venv/bin/python"
python="${TINY_IREE_PYTHON:-$default_python}"
if ! "$python" -c 'import onnx, numpy' 2>/dev/null; then
  bash "$repo_dir/scripts/setup_python.sh"
  python="$repo_dir/.venv/bin/python"
fi

bash "$repo_dir/scripts/build.sh"
cmake --build "$repo_dir/build" \
  --target tiny-iree-translate tiny-iree-export-codegen tiny-iree-run-module \
  -j"${JOBS:-4}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-onnx.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$python" "$repo_dir/tools/generate_mlp_onnx.py" -o "$tmp/mlp.onnx"
"$python" "$repo_dir/tools/tiny_iree_compile.py" "$tmp/mlp.onnx" \
  -o "$tmp/mlp.tiree" --cpu-codegen=scalar
actual="$($repo_dir/build/bin/tiny-iree-run-module "$tmp/mlp.tiree" \
  --function=predict --input=1,2,3,4)"
reference="$("$python" "$repo_dir/tools/run_onnx_reference.py" \
  "$tmp/mlp.onnx" --input=1,2,3,4)"

printf '%s\nONNX reference:\n%s\n' "$actual" "$reference"
"$python" - "$actual" "$reference" <<'PY'
import math
import re
import sys

def values(text):
    match = re.search(r"result: \[([^]]+)\]", text)
    if not match:
        raise SystemExit("result line not found")
    return [float(value) for value in match.group(1).split(",")]

actual = values(sys.argv[1])
reference = values(sys.argv[2])
if len(actual) != len(reference) or not all(
    math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
    for a, b in zip(actual, reference)
):
    raise SystemExit(f"mismatch: {actual} != {reference}")
PY
grep -q 'argmax: 2' <<<"$actual"
printf 'stage-08 onnx frontend: ok\n'
