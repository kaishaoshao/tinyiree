# 真实性能基准方法

性能数字只有在测量边界、输入、编译参数和机器状态都明确时才可比较。Tiny-IREE
把性能问题分为编译耗时、冷启动、稳态 invocation 和单个 native kernel 四层；不要
把一次 shell 命令的总时间直接称为“模型推理时间”。

## 1. 先定义测量边界

| 指标 | 包含 | 不包含 | 适合回答 |
| --- | --- | --- | --- |
| 编译耗时 | ONNX import、passes、LLVM codegen、link | runtime | 部署前成本 |
| 冷启动延迟 | 新进程、manifest、bytecode、`dlopen`、一次 invocation | 编译 | CLI/短任务首帧 |
| 稳态 invocation | 输入解析、VM 调度、资源建立、native kernels | 进程启动、bundle/bytecode 解析、`dlopen`、结果打印 | 常驻进程吞吐基础 |
| kernel-only | 一个 native entry point | VM、allocator、输入解析 | codegen 本身 |

当前 runtime 原生提供稳态 invocation 基准。它在同一进程中先完成 bundle 校验、
动态库加载和 bytecode 解析，再执行预热与计时循环。每次 invocation 仍会重建 VM
value/resource 表和 allocator，因此它不是 kernel-only 计时。

## 2. 建立可比较的产物

使用 `RelWithDebInfo` 构建工具，并对同一个 ONNX、输入和 git revision 分别生成
scalar、vector、hierarchical bundle。以下 128x128 MatMul 足够演示流程；过小的
MLP 会被计时器、输入解析和调度噪声支配。

```bash
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
bash scripts/build.sh
bash scripts/setup_python.sh
PYTHON="$PWD/.venv/bin/python"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-bench.XXXXXX")"

"$PYTHON" tools/generate_tiled_onnx.py \
  -o "$WORK_DIR/matmul.onnx" --size=128
for MODE in scalar vector hierarchical; do
  "$PYTHON" tools/tiny_iree_compile.py "$WORK_DIR/matmul.onnx" \
    -o "$WORK_DIR/$MODE.tiree" --cpu-codegen="$MODE" \
    --parallel-threads=4
done
INPUT="$($PYTHON - <<'PY'
print(",".join("1" for _ in range(128 * 128)))
PY
)"
```

先运行一次非基准模式，确认结果和实际 backend。基准不能代替正确性测试：

```bash
"$BUILD_DIR/bin/tiny-iree-run-module" "$WORK_DIR/vector.tiree" \
  --function=predict --input="$INPUT"
```

输出必须包含 `backend: native-aot`。若出现 `interpreter`，测到的是解释器，不应与
native bundle 比较。

## 3. 测量稳态 invocation

```bash
for MODE in scalar vector hierarchical; do
  printf '\n== %s ==\n' "$MODE"
  "$BUILD_DIR/bin/tiny-iree-run-module" "$WORK_DIR/$MODE.tiree" \
    --function=predict --input="$INPUT" \
    --benchmark-warmup=10 --benchmark-repetitions=100
done
```

runtime 使用 `std::chrono::steady_clock`，以微秒输出 `min`、`median`、线性插值
`p90` 和 `mean`。基准模式不打印 tensor，避免 stdout 时间污染样本。主要比较
median，同时报告 p90；不要只挑最小值。若单次低于几十微秒，应增大问题规模，
而不是增加小数位。

预热用于触发页映射、指令/数据 cache 和 OpenMP worker 初始化。hierarchical
模式应固定线程条件，例如：

```bash
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
```

编译时的 `--parallel-threads`、运行时 OpenMP 环境和物理核心数都要记录。不要把
不同线程数的结果只标成“M4”和“x86”。

## 4. 测量冷启动

冷启动必须启动新进程。先避免系统 `time` 只跑一次的偶然值，可用标准 Python
收集多次 wall-clock；这包括进程创建、加载和结果打印：

```bash
"$PYTHON" - "$BUILD_DIR/bin/tiny-iree-run-module" \
  "$WORK_DIR/vector.tiree" "$INPUT" <<'PY'
import statistics
import subprocess
import sys
import time

runtime, bundle, input_values = sys.argv[1:]
samples = []
for _ in range(20):
    start = time.perf_counter_ns()
    subprocess.run(
        [runtime, bundle, "--function=predict", f"--input={input_values}"],
        check=True, stdout=subprocess.DEVNULL,
    )
    samples.append((time.perf_counter_ns() - start) / 1_000)
print(f"cold_start_us_median: {statistics.median(samples):.3f}")
print(f"cold_start_us_min: {min(samples):.3f}")
PY
```

冷启动和稳态 invocation 是不同产品指标，不应放在同一列直接求加速比。

## 5. 实验控制与记录模板

每组结果至少保存：

```bash
git rev-parse HEAD
uname -a
cmake -LA -N "$BUILD_DIR" | grep CMAKE_BUILD_TYPE
cat "$WORK_DIR/vector.tiree/manifest.json"
printf 'OMP_NUM_THREADS=%s\n' "${OMP_NUM_THREADS:-unset}"
```

另外记录 CPU 型号、物理/逻辑核数、内存、操作系统版本、电源模式、是否接电、
温度状态、后台负载、输入 shape/内容、预热次数和样本数。Linux 可补充 `lscpu`；
macOS 可补充 `sysctl -n machdep.cpu.brand_string` 与 `sysctl -n hw.physicalcpu`。

建议每个配置至少做 3 轮独立进程实验，交错运行待比较配置，避免温升或后台任务
总是偏向最后一个配置。Apple M4 与 x86 的结果只能说明各自在已记录配置下的表现；
它们不能单独证明 ISA、操作系统或某个 pass 是差异的唯一原因。

## 6. 解读结果

- 先检查数值一致，再比较延迟。
- 同时报告绝对延迟和相对加速比，注明基线。
- hierarchical 在小矩阵上变慢通常是线程调度成本，不一定是错误。
- median 稳定但 p90 很高，优先检查调频、温度、后台负载和 OpenMP 线程迁移。
- scalar/vector 几乎相同，检查最终 LLVM IR 是否真的包含 vector lowering。
- 结果跨机器不可复现时，先对比 manifest、commit、构建类型和输入，不要先改 tile。

本入口由 `tests/run_benchmark_e2e.sh` 回归，测试只验证计时模式和输出契约，不把
CI 的绝对耗时设为性能门槛。
