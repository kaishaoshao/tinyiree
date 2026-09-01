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
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-onnx.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

model="$temporary_dir/mlp.onnx"
bundle="$temporary_dir/mlp.tiree"

"$python" "$project_dir/tools/generate_mlp_onnx.py" -o "$model"
"$python" "$project_dir/tools/tiny_iree_compile.py" \
  "$model" -o "$bundle" --opt "$opt"

grep -q 'tiree_input.matmul' "$bundle/module.input.mlir"
grep -q 'tiree_input.fused_matmul_add_relu' "$bundle/module.global-opt.mlir"
grep -q 'tiree_flow.dispatch' "$bundle/module.flow.mlir"
grep -q 'tiree_stream.dispatch' "$bundle/module.stream.mlir"
grep -q 'tiree_flow.yield' "$bundle/module.hal.mlir" && {
  echo "flow region unexpectedly leaked into HAL IR" >&2
  exit 1
}
grep -q 'tiree_hal.dispatch' "$bundle/module.hal.mlir"
grep -q 'tiree_hal.executable' "$bundle/module.hal.mlir"
grep -q '!tiree_hal.buffer' "$bundle/module.hal.mlir"
grep -q 'tiree_vm.call' "$bundle/module.vm.mlir"
grep -q 'tiree_vm.alloc' "$bundle/module.vm.mlir"
grep -q 'tiree_vm.dealloc' "$bundle/module.vm.mlir"
grep -q '!tiree_vm.ref' "$bundle/module.vm.mlir"
grep -q '"format": "tiny-iree-codegen-plan-v1"' "$bundle/module.executable.json"
grep -q '"format": "tiny-iree-bundle-v1"' "$bundle/manifest.json"
grep -q '"vm_format": "tiny-vm-bytecode-v1"' "$bundle/manifest.json"
grep -q '"global-optimization"' "$bundle/manifest.json"
grep -q '"codegen": "mlir-linalg-vector-llvm"' "$bundle/manifest.json"
test "$(od -An -tx1 -N8 "$bundle/module.tvm" | tr -d ' \n')" = "54495245564d3100"
grep -q 'linalg.matmul' "$bundle/module.executable.mlir"
grep -q 'tile_sizes \[8, 8, 4\]' "$bundle/module.executable.mlir"
grep -q 'vector.mask' "$bundle/module.executable.optimized.mlir"
grep -q 'llvm.func' "$bundle/module.executable.llvm.mlir"
grep -q '_mlir_ciface_tiree_kernel_matmul_add_relu' \
  "$bundle/module.executable.ll"
if [[ "$(uname -s)" == "Darwin" ]]; then
  file "$bundle/libtiny_iree_kernels.dylib" | grep -q 'Mach-O 64-bit dynamically linked shared library arm64'
  nm -gU "$bundle/libtiny_iree_kernels.dylib" | \
    grep -q '__mlir_ciface_tiree_kernel_softmax'
fi

actual="$($runtime "$bundle" --function=predict --input=1,2,3,4)"
reference="$($python "$project_dir/tools/run_onnx_reference.py" \
  "$model" --input=1,2,3,4)"

printf '%s\n' "$actual"
printf 'ONNX reference:\n%s\n' "$reference"
grep -q 'backend: native-aot' <<<"$actual"
grep -q 'vm: tiny-bytecode-v1' <<<"$actual"
grep -q 'resources: peak=24B live=12B' <<<"$actual"
grep -q 'argmax: 2' <<<"$actual"

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
    math.isclose(lhs, rhs, rel_tol=1e-6, abs_tol=1e-7)
    for lhs, rhs in zip(actual, reference)
):
    raise SystemExit(f"native result {actual} != ONNX reference {reference}")
PY

bad_bundle="$temporary_dir/wrong-target.tiree"
cp -R "$bundle" "$bad_bundle"
"$python" - "$bad_bundle/manifest.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as file:
    manifest = json.load(file)
manifest["target"] = "wrong-architecture"
with open(path, "w", encoding="utf-8") as file:
    json.dump(manifest, file)
PY
if "$runtime" "$bad_bundle" --function=predict --input=1,2,3,4 \
    >"$temporary_dir/wrong-target.out" 2>"$temporary_dir/wrong-target.err"; then
  echo "wrong-target bundle unexpectedly executed" >&2
  exit 1
fi
grep -q 'does not match runtime target' "$temporary_dir/wrong-target.err"

bad_bytecode="$temporary_dir/bad.tvm"
cp "$bundle/module.tvm" "$bad_bytecode"
printf '\377' | dd of="$bad_bytecode" bs=1 seek=0 conv=notrunc status=none
if "$runtime" "$bad_bytecode" --function=predict --input=1,2,3,4 \
    >"$temporary_dir/bad-bytecode.out" 2>"$temporary_dir/bad-bytecode.err"; then
  echo "corrupt Tiny VM bytecode unexpectedly executed" >&2
  exit 1
fi
