#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <python-with-onnx> <tiny-iree-opt> <tiny-runtime>" >&2
  exit 2
fi

python="$1"
opt="$2"
runtime="$3"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-benchmark.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$python" "$repo_dir/tools/generate_tiled_onnx.py" \
  -o "$tmp/model.onnx" --size=8
"$python" "$repo_dir/tools/tiny_iree_compile.py" "$tmp/model.onnx" \
  -o "$tmp/model.tiree" --opt "$opt" --cpu-codegen=scalar

input="$($python - <<'PY'
print(",".join("1" for _ in range(8 * 8)))
PY
)"
normal="$($runtime "$tmp/model.tiree" --function=predict --input="$input")"
grep -q '^backend: native-aot$' <<<"$normal"
grep -q '^result: ' <<<"$normal"

benchmark="$($runtime "$tmp/model.tiree" --function=predict --input="$input" \
  --benchmark-warmup=2 --benchmark-repetitions=7)"
grep -q '^benchmark: steady-state-invocation$' <<<"$benchmark"
grep -q '^warmup_iterations: 2$' <<<"$benchmark"
grep -q '^measured_iterations: 7$' <<<"$benchmark"
grep -Eq '^latency_us_min: [0-9]+([.][0-9]+)?$' <<<"$benchmark"
grep -Eq '^latency_us_median: [0-9]+([.][0-9]+)?$' <<<"$benchmark"
grep -Eq '^latency_us_p90: [0-9]+([.][0-9]+)?$' <<<"$benchmark"
grep -Eq '^latency_us_mean: [0-9]+([.][0-9]+)?$' <<<"$benchmark"
if grep -q '^result: ' <<<"$benchmark"; then
  echo "benchmark mode unexpectedly included tensor printing" >&2
  exit 1
fi

if "$runtime" "$tmp/model.tiree" --function=predict --input="$input" \
    --benchmark-warmup=1 --benchmark-repetitions=0 \
    >"$tmp/invalid.out" 2>"$tmp/invalid.err"; then
  echo "warmup without measured repetitions unexpectedly passed" >&2
  exit 1
fi
grep -q 'warmup requires measured repetitions' "$tmp/invalid.err"

printf 'runtime benchmark: ok\n'
