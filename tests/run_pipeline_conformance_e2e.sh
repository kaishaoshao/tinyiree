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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-pipeline.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/mlp.onnx"
bundle="$temporary_dir/mlp.tiree"
"$python" "$project_dir/tools/generate_mlp_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"

grep -q 'tiree_input.matmul' "$bundle/module.input.mlir"
! grep -q 'tiree_flow' "$bundle/module.input.mlir"
grep -q 'tiree_input.fused_matmul_add_relu' "$bundle/module.global-opt.mlir"
! grep -q 'tiree_flow' "$bundle/module.global-opt.mlir"
grep -q 'tiree_flow.dispatch' "$bundle/module.flow.mlir"
! grep -q 'tiree_stream' "$bundle/module.flow.mlir"
grep -q 'tiree_stream.dispatch' "$bundle/module.stream.mlir"
! grep -q 'tiree_flow.dispatch' "$bundle/module.stream.mlir"
grep -q 'tiree_hal.executable' "$bundle/module.hal.mlir"
! grep -q 'tiree_stream.dispatch' "$bundle/module.hal.mlir"
grep -q 'tiree_vm.call' "$bundle/module.vm.mlir"
! grep -q 'tiree_hal.dispatch' "$bundle/module.vm.mlir"
grep -q '"backend": "llvm-cpu"' "$bundle/module.executable.json"
grep -q '"target_backend": "llvm-cpu"' "$bundle/manifest.json"
grep -q '"flow": "module.flow.mlir"' "$bundle/manifest.json"

output="$("$runtime" "$bundle" --function=predict --input=1,2,3,4)"
grep -q 'argmax: 2' <<<"$output"
