#!/usr/bin/env bash

# Shared host/toolchain discovery for Tiny-IREE shell entry points.
# This file is sourced by other scripts and intentionally does not change
# the caller's shell options.

tiny_iree_host_id() {
  local system machine
  system="$(uname -s)"
  machine="$(uname -m)"
  case "$system-$machine" in
    Darwin-arm64|Darwin-aarch64) printf '%s\n' darwin-arm64 ;;
    Darwin-x86_64|Darwin-amd64) printf '%s\n' darwin-x86_64 ;;
    Linux-aarch64|Linux-arm64) printf '%s\n' linux-aarch64 ;;
    Linux-x86_64|Linux-amd64) printf '%s\n' linux-x86_64 ;;
    *)
      printf 'unsupported host: %s-%s\n' "$system" "$machine" >&2
      return 1
      ;;
  esac
}

tiny_iree_build_dir() {
  local repo_dir="$1"
  if [[ -n "${TINY_IREE_BUILD_DIR:-}" ]]; then
    printf '%s\n' "$TINY_IREE_BUILD_DIR"
    return 0
  fi

  local default_dir="$repo_dir/build"
  local cache="$default_dir/CMakeCache.txt"
  if [[ -f "$cache" ]]; then
    local cached_source
    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache" | head -n 1)"
    if [[ -n "$cached_source" && "$cached_source" != "$repo_dir" ]]; then
      printf '%s-%s\n' "$default_dir" "$(tiny_iree_host_id)"
      return 0
    fi
  fi
  printf '%s\n' "$default_dir"
}

tiny_iree_has_cmake_packages() {
  local root="$1"
  [[ -f "$root/lib/cmake/mlir/MLIRConfig.cmake" ]] &&
    { [[ -f "$root/lib/cmake/llvm/LLVMConfig.cmake" ]] ||
      [[ -f "$root/llvm-project/lib/cmake/llvm/LLVMConfig.cmake" ]]; }
}

tiny_iree_find_toolchain_root() {
  local repo_dir="$1"
  local -a candidates=()
  [[ -n "${IREE_BUILD_DIR:-}" ]] && candidates+=("$IREE_BUILD_DIR")
  [[ -n "${LLVM_BUILD_DIR:-}" ]] && candidates+=("$LLVM_BUILD_DIR")
  if [[ -n "${IREE_DIR:-}" ]]; then
    candidates+=("$IREE_DIR/build_tools/build-host" "$IREE_DIR/build")
  fi
  candidates+=(
    "$repo_dir/../iree/build_tools/build-host"
    "$repo_dir/../iree/build"
    "$repo_dir/../llvm-project/build"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if tiny_iree_has_cmake_packages "$candidate"; then
      (cd "$candidate" && pwd)
      return 0
    fi
  done

  cat >&2 <<EOF
unable to find a usable LLVM/MLIR build.
Set IREE_BUILD_DIR or LLVM_BUILD_DIR to a build containing:
  lib/cmake/mlir/MLIRConfig.cmake
  lib/cmake/llvm/LLVMConfig.cmake (or llvm-project/lib/cmake/llvm)
EOF
  return 1
}

tiny_iree_mlir_cmake_dir() {
  printf '%s/lib/cmake/mlir\n' "$1"
}

tiny_iree_llvm_cmake_dir() {
  local root="$1"
  if [[ -f "$root/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
    printf '%s/lib/cmake/llvm\n' "$root"
  else
    printf '%s/llvm-project/lib/cmake/llvm\n' "$root"
  fi
}

tiny_iree_find_llvm_tool() {
  local root="$1" tool="$2" candidate
  for candidate in "$root/llvm-project/bin/$tool" "$root/bin/$tool"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'LLVM tool not found: %s (toolchain: %s)\n' "$tool" "$root" >&2
  return 1
}

tiny_iree_target_triple() {
  case "$(tiny_iree_host_id)" in
    darwin-arm64) printf '%s\n' arm64-apple-macosx13.0.0 ;;
    darwin-x86_64) printf '%s\n' x86_64-apple-macosx13.0.0 ;;
    linux-aarch64) printf '%s\n' aarch64-unknown-linux-gnu ;;
    linux-x86_64) printf '%s\n' x86_64-unknown-linux-gnu ;;
  esac
}

tiny_iree_library_name() {
  case "$(tiny_iree_host_id)" in
    darwin-*) printf '%s\n' libtiny_iree_kernels.dylib ;;
    linux-*) printf '%s\n' libtiny_iree_kernels.so ;;
  esac
}

tiny_iree_link_mode() {
  case "$(tiny_iree_host_id)" in
    darwin-*) printf '%s\n' -dynamiclib ;;
    linux-*) printf '%s\n' -shared ;;
  esac
}

tiny_iree_print_platform() {
  printf 'backend=llvm-cpu\n'
  printf 'host_id=%s\n' "$(tiny_iree_host_id)"
  printf 'target_triple=%s\n' "$(tiny_iree_target_triple)"
  printf 'library=%s\n' "$(tiny_iree_library_name)"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  tiny_iree_print_platform
fi
