#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <00..09>" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
script="$repo_dir/tests/stages/${1}_"*
matches=($script)
if [[ ${#matches[@]} -ne 1 || ! -f "${matches[0]}" ]]; then
  echo "stage test not found or ambiguous: $1" >&2
  exit 2
fi
bash "${matches[0]}"

