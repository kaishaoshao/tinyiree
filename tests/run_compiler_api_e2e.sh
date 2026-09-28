#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <build-dir>" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$1"
compiler_api="$build_dir/lib/libTinyIREECompiler.so"
if [[ "$(uname -s)" == "Darwin" ]]; then
  compiler_api="$build_dir/lib/libTinyIREECompiler.dylib"
fi
if [[ ! -f "$compiler_api" ]]; then
  echo "compiler shared library missing: $compiler_api" >&2
  exit 1
fi

scratch_dir="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-capi.XXXXXX")"
trap 'rm -rf "$scratch_dir"' EXIT
cc="${CC:-cc}"
"$cc" -std=c11 -I"$repo_dir/include" "$repo_dir/tests/compiler_api_smoke.c" \
  -L"$build_dir/lib" -lTinyIREECompiler -Wl,-rpath,"$build_dir/lib" \
  -o "$scratch_dir/compiler-api-smoke"
"$scratch_dir/compiler-api-smoke"
