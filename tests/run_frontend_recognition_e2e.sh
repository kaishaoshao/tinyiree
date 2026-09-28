#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <python-with-onnx> <tiny-iree-opt> <tiny-runtime>" >&2
  exit 2
fi

python="$1"
opt="$2"
runtime="$3"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-frontend.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

supported="$($python "$repo_dir/tools/import_onnx.py" --list-supported-ops)"
grep -q '^MatMul: rank-2 f32' <<<"$supported"
grep -q '^Conv: static NCHW' <<<"$supported"
grep -q '^QuantizeLinear: matched per-tensor signed-int8' <<<"$supported"

"$python" "$repo_dir/tools/generate_mlp_onnx.py" -o "$tmp/mlp.onnx"
analysis="$($python "$repo_dir/tools/import_onnx.py" --analyze "$tmp/mlp.onnx")"
grep -q '^format: onnx$' <<<"$analysis"
grep -q '^graph inputs:$' <<<"$analysis"
grep -q 'ai.onnx::MatMul: .*op-type-supported=yes' <<<"$analysis"
grep -q '^initializers: 2$' <<<"$analysis"

cp "$tmp/mlp.onnx" "$tmp/model.pb"
if "$python" "$repo_dir/tools/tiny_iree_compile.py" "$tmp/model.pb" \
    -o "$tmp/auto.tiree" --opt "$opt" \
    >"$tmp/auto.out" 2>"$tmp/auto.err"; then
  echo "extensionless ONNX unexpectedly passed auto frontend detection" >&2
  exit 1
fi
grep -q 'use --input-type=onnx or --input-type=mlir' "$tmp/auto.err"

"$python" "$repo_dir/tools/tiny_iree_compile.py" "$tmp/model.pb" \
  -o "$tmp/onnx.tiree" --input-type=onnx --opt "$opt" \
  --cpu-codegen=scalar
grep -q '"input_type": "onnx"' "$tmp/onnx.tiree/manifest.json"
output="$($runtime "$tmp/onnx.tiree" --function=predict --input=1,2,3,4)"
grep -q 'argmax: 2' <<<"$output"

cp "$repo_dir/examples/runtime_inputs.mlir" "$tmp/module.txt"
"$python" "$repo_dir/tools/tiny_iree_compile.py" "$tmp/module.txt" \
  -o "$tmp/mlir.tiree" --input-type=mlir --opt "$opt" \
  --cpu-codegen=scalar
grep -q '"input_type": "mlir"' "$tmp/mlir.tiree/manifest.json"

"$python" - "$tmp/unsupported.onnx" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper

path = sys.argv[1]
graph = helper.make_graph(
    [helper.make_node("Sin", ["input"], ["output"])],
    "unsupported_frontend_model",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 18)]
)
onnx.checker.check_model(model)
onnx.save(model, path)
PY

if "$python" "$repo_dir/tools/import_onnx.py" "$tmp/unsupported.onnx" \
    -o "$tmp/unsupported.mlir" \
    >"$tmp/unsupported.out" 2>"$tmp/unsupported.err"; then
  echo "unsupported ONNX op unexpectedly imported" >&2
  exit 1
fi
grep -q 'unsupported ONNX op ai.onnx::Sin' "$tmp/unsupported.err"

"$python" - "$tmp/custom-domain.onnx" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper

path = sys.argv[1]
graph = helper.make_graph(
    [helper.make_node(
        "MatMul", ["lhs", "rhs"], ["output"], domain="example.custom"
    )],
    "custom_domain_model",
    [
        helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [2, 1]),
    ],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
)
model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_opsetid("", 18),
        helper.make_opsetid("example.custom", 1),
    ],
)
onnx.checker.check_model(model)
onnx.save(model, path)
PY

custom_analysis="$($python "$repo_dir/tools/import_onnx.py" \
  --analyze "$tmp/custom-domain.onnx")"
grep -q 'example.custom::MatMul: .*op-type-supported=no' <<<"$custom_analysis"
if "$python" "$repo_dir/tools/import_onnx.py" "$tmp/custom-domain.onnx" \
    -o "$tmp/custom-domain.mlir" \
    >"$tmp/custom-domain.out" 2>"$tmp/custom-domain.err"; then
  echo "custom-domain MatMul unexpectedly imported as ai.onnx MatMul" >&2
  exit 1
fi
grep -q 'unsupported ONNX op example.custom::MatMul' \
  "$tmp/custom-domain.err"

printf 'frontend recognition: ok\n'
