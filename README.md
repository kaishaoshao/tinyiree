# Tiny-IREE Learning Repository

这是一个按 IREE 核心分层逐步实现的小型 AI 编译器教学仓库。重点不是复制
IREE 的全部功能，而是让模型前端、MLIR Dialect、优化、dispatch、资源规划、
HAL、VM、CPU codegen 和 runtime 都具有可观察的简化实现。

## 代码布局

- `compiler/`：`TinyIREECompiler` 库、Dialect/type/op 实现和阶段 lowering passes。
- `runtime/`：`TinyIREERuntime` 库、bundle loader、VM bytecode interpreter、allocator
  和 native dispatch。
- `include/tiny_iree/`：compiler/runtime 的公共声明及 TableGen 定义。
- `tools/`：只构建/承载 `tiny-iree-opt`、translate、codegen export 和 runtime 的
  CLI 入口；Python compile/import/codegen 脚本是当前教学前端的直接调用接口。
- `tests/`：分阶段测试和端到端 native AOT 回归。

根目录只负责共享配置与 TableGen；三个子目录各有 `CMakeLists.txt`，分别拥有
compiler library、runtime library 与 executable wrappers。`compiler/README.md` 与
`runtime/README.md` 进一步说明两侧边界；目录划分表示源码职责，不会把 M4 和 x86
拆成两套 compiler/runtime。

## 如何学习

本仓库刻意保留线性 Git 历史。每个提交都是一个学习阶段，并配套：

- `docs/stages/`：阶段目标、概念、源码入口和实验。
- `tests/stages/`：只验证本阶段职责的测试脚本。
- `tests/run_all.sh`：最终端到端回归入口。
- Git tag `stage-00` 到 `stage-10`：快速切换学习快照。

查看历史：

```bash
git log --reverse --oneline --decorate
git show stage-02
git switch --detach stage-02
```

返回最终版本：

```bash
git switch main
```

## 本机依赖与支持平台

支持 Linux x86_64、Linux AArch64、macOS x86_64 和 Apple Silicon（包括 M4）。
生成的 bundle 是本机产物；请在目标机器上重新编译，不要在 x86 与 arm64 间
直接复制 `.tiree` bundle。

M4 与 x86 使用同一个 `llvm-cpu` 后端，脚本根据操作系统和 CPU 架构选择不同
的 target triple、动态库格式和 bundle target。可查看当前选择：

```bash
bash scripts/platform.sh
```

完整判定规则见 [平台与目标说明](docs/PLATFORMS.md)。

脚本会依次查找：

- `IREE_BUILD_DIR` 或 `LLVM_BUILD_DIR` 指定的 build-tree；
- 相邻 `../iree/build_tools/build-host` 或 `../iree/build`；
- 相邻 `../llvm-project/build`。

build-tree 需要包含 LLVM/MLIR 的 CMake package，以及 `mlir-opt`、
`mlir-translate` 和 `llc`。例如：

```bash
# WSL / Linux x86_64（本仓库当前目录布局可自动发现）
export LLVM_BUILD_DIR=../llvm-project/build

# Apple M4 + IREE host tools
export IREE_BUILD_DIR=../iree/build_tools/build-host
```

以上路径均为相对路径，也可以设置为本机任意位置；仓库内不依赖某个用户的
磁盘卷或 home 目录。

先执行环境检查：

```bash
bash tests/stages/00_environment.sh
```

完整阶段路线见 [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md)。
模型格式选择、ONNX 图识别、Q/DQ 模式匹配和算子限制见
[模型前端说明](docs/MODEL_FRONTEND.md)。
测量方法、二进制接口和运行故障定位分别见
[真实性能基准](docs/PERFORMANCE_BENCHMARKING.md)、
[VM 字节码与 Native ABI](docs/VM_BYTECODE_AND_NATIVE_ABI.md)和
[常见失败诊断](docs/TROUBLESHOOTING.md)。

## 构建与完整回归

```bash
bash tests/run_all.sh
```

也可以只构建全部编译器和 runtime 工具：

```bash
bash scripts/build.sh
```

若仓库携带了另一台机器生成的 `build/CMakeCache.txt`，脚本会自动使用
`build-<os>-<arch>`，避免覆盖原平台构建。可用 `TINY_IREE_BUILD_DIR` 显式指定。

编译并运行 ONNX MLP：

```bash
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
bash scripts/build.sh
bash scripts/setup_python.sh
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" tools/generate_mlp_onnx.py -o /tmp/mlp.onnx
"$PYTHON" tools/tiny_iree_compile.py /tmp/mlp.onnx -o /tmp/mlp.tiree
"$BUILD_DIR/bin/tiny-iree-run-module" /tmp/mlp.tiree \
  --function=predict --input=1,2,3,4
```

预期使用 `native-aot` backend，输出类别 `argmax: 2`。Git 学习操作见
[docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)。
