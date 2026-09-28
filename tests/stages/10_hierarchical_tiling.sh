#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
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
bash "$repo_dir/tests/run_hierarchical_tiling_e2e.sh" \
  "$python" "$build_dir/bin/tiny-iree-opt" \
  "$build_dir/bin/tiny-iree-run-module"
printf 'stage-10 hierarchical tiling: ok\n'
