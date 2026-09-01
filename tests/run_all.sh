#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
iree_dir="${IREE_DIR:-$(cd "$repo_dir/../iree" && pwd)}"
iree_build="${IREE_BUILD_DIR:-$iree_dir/build_tools/build-host}"
default_python="$iree_build/tiny-iree-venv/bin/python"
python="${TINY_IREE_PYTHON:-$default_python}"
if ! "$python" -c 'import onnx, numpy' 2>/dev/null; then
  bash "$repo_dir/scripts/setup_python.sh"
  python="$repo_dir/.venv/bin/python"
fi

bash "$repo_dir/scripts/build.sh"
cmake --build "$repo_dir/build" --target tiny-iree-aot-tools \
  -j"${JOBS:-4}"
opt="$repo_dir/build/bin/tiny-iree-opt"
runtime="$repo_dir/build/bin/tiny-iree-run-module"

bash "$repo_dir/tests/run_e2e.sh" "$opt" "$runtime"
bash "$repo_dir/tests/run_pipeline_conformance_e2e.sh" \
  "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_onnx_aot_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_multi_io_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_multi_result_dispatch_e2e.sh" \
  "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_dynamic_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_quantized_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_tiling_vector_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_resource_reuse_e2e.sh" "$python" "$opt" "$runtime"
bash "$repo_dir/tests/run_cnn_e2e.sh" "$python" "$opt" "$runtime"

if [[ -x "$iree_build/tools/iree-compile" &&
      -x "$iree_build/tools/iree-run-module" ]]; then
  bash "$repo_dir/tests/run_official_iree_reference_e2e.sh" \
    "$python" "$opt" "$runtime" \
    "$iree_build/tools/iree-compile" "$iree_build/tools/iree-run-module"
fi

printf 'all tiny-iree tests: ok\n'
