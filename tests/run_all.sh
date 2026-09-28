#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
source "$repo_dir/scripts/platform.sh"
toolchain_root="$(tiny_iree_find_toolchain_root "$repo_dir")"
build_dir="$(tiny_iree_build_dir "$repo_dir")"
if [[ -x "$repo_dir/.venv/bin/python" ]]; then
  default_python="$repo_dir/.venv/bin/python"
else
  default_python="$toolchain_root/tiny-iree-venv/bin/python"
fi
python="${TINY_IREE_PYTHON:-$default_python}"
if ! "$python" -c 'import onnx, numpy' 2>/dev/null; then
  bash "$repo_dir/scripts/setup_python.sh"
  python="$repo_dir/.venv/bin/python"
fi

bash "$repo_dir/scripts/build.sh"
cmake --build "$build_dir" --target tiny-iree-aot-tools \
  -j"${JOBS:-4}"
opt="$build_dir/bin/tiny-iree-opt"
runtime="$build_dir/bin/tiny-iree-run-module"

bash "$repo_dir/tests/run_e2e.sh" "$opt" "$runtime"
bash "$repo_dir/tests/run_benchmark_e2e.sh" \
  "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_frontend_recognition_e2e.sh" \
  "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_pipeline_conformance_e2e.sh" \
  "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_onnx_aot_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_multi_io_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_multi_result_dispatch_e2e.sh" \
  "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_dynamic_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_quantized_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_tiling_vector_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_hierarchical_tiling_e2e.sh" \
  "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_resource_reuse_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_cnn_e2e.sh" "$python" "$opt" "$runtime"

if [[ -x "$toolchain_root/tools/iree-compile" &&
      -x "$toolchain_root/tools/iree-run-module" ]]; then
  bash "$repo_dir/tests/run_official_iree_reference_e2e.sh" \
    "$python" "$opt" "$runtime" \
    "$toolchain_root/tools/iree-compile" \
    "$toolchain_root/tools/iree-run-module"
fi

printf 'all tiny-iree tests: ok\n'
