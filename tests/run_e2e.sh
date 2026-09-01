#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <tiny-iree-opt> <tiny-iree-run-module>" >&2
  exit 1
fi

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

input_module="$project_dir/examples/mlp.mlir"

"$1" "$input_module" \
  --pass-pipeline='builtin.module(tiree-global-optimize,tiree-input-to-flow)' \
  -o "$temporary_dir/flow.mlir"
grep -q 'tiree_flow.dispatch' "$temporary_dir/flow.mlir"
grep -q 'tiree_flow.yield' "$temporary_dir/flow.mlir"
grep -q 'tiree_input.fused_matmul_add_relu' "$temporary_dir/flow.mlir"
grep -q 'tiree_input.softmax' "$temporary_dir/flow.mlir"

"$1" "$input_module" \
  --pass-pipeline='builtin.module(tiree-global-optimize,tiree-input-to-flow,tiree-flow-to-stream)' \
  -o "$temporary_dir/stream.mlir"
grep -q 'tiree_stream.dispatch' "$temporary_dir/stream.mlir"
grep -q 'tiree_stream.alloc' "$temporary_dir/stream.mlir"
grep -q 'tiree_stream.dealloc' "$temporary_dir/stream.mlir"
grep -q '!tiree_stream.resource' "$temporary_dir/stream.mlir"
if grep -Eq '"tiree_input\.[a-z_]+"\(' "$temporary_dir/stream.mlir"; then
  echo "input payload leaked past Flow -> Stream" >&2
  exit 1
fi

"$1" "$input_module" \
  --pass-pipeline='builtin.module(tiree-global-optimize,tiree-input-to-flow,tiree-flow-to-stream,tiree-stream-to-hal)' \
  -o "$temporary_dir/hal.mlir"
grep -q 'tiree_hal.dispatch' "$temporary_dir/hal.mlir"
grep -q 'tiree_hal.alloc' "$temporary_dir/hal.mlir"
grep -q 'tiree_hal.dealloc' "$temporary_dir/hal.mlir"
grep -q 'tiree_hal.executable' "$temporary_dir/hal.mlir"
grep -q '!tiree_hal.buffer' "$temporary_dir/hal.mlir"

"$1" "$input_module" \
  --tiree-compile-pipeline \
  -o "$temporary_dir/vm.mlir"
grep -q 'tiree_vm.call' "$temporary_dir/vm.mlir"
grep -q 'tiree_vm.alloc' "$temporary_dir/vm.mlir"
grep -q 'tiree_vm.dealloc' "$temporary_dir/vm.mlir"
grep -q '!tiree_vm.ref' "$temporary_dir/vm.mlir"

output="$("$2" "$temporary_dir/vm.mlir" \
  --function=predict \
  --input=1,2,3,4)"

printf '%s\n' "$output"
grep -q 'result: \[2.655812e-02, 1.461313e-03, 9.719805e-01\]' <<<"$output"
grep -q 'argmax: 2' <<<"$output"
grep -q 'resources: peak=24B live=12B' <<<"$output"

if "$1" "$project_dir/tests/invalid_flow_region.mlir" \
    -o /dev/null 2>"$temporary_dir/invalid.err"; then
  echo "invalid flow region unexpectedly passed verification" >&2
  exit 1
fi
grep -q 'kernel and workload payload operation order differ' \
  "$temporary_dir/invalid.err"

if "$1" "$project_dir/tests/invalid_stream_resources.mlir" \
    --tiree-verify-stream-resources -o /dev/null \
    2>"$temporary_dir/invalid-stream.err"; then
  echo "invalid stream resource plan unexpectedly passed verification" >&2
  exit 1
fi
grep -q 'resource is not live' "$temporary_dir/invalid-stream.err"

if "$2" "$project_dir/tests/invalid_vm_resources.mlir" \
    --function=predict --input=1,2,3 \
    >"$temporary_dir/invalid-vm.out" 2>"$temporary_dir/invalid-vm.err"; then
  echo "invalid VM resource plan unexpectedly executed" >&2
  exit 1
fi
grep -q 'invalid output resource' "$temporary_dir/invalid-vm.err"
