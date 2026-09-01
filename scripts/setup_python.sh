#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python="${PYTHON:-python3}"
"$python" -m venv "$repo_dir/.venv"
"$repo_dir/.venv/bin/pip" install -r "$repo_dir/requirements.txt"
printf 'python environment: %s\n' "$repo_dir/.venv/bin/python"
