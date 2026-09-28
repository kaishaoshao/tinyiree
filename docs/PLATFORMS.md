# 平台与 CPU 目标

Tiny-IREE 的 M4 与 x86 路径共享同一个 `llvm-cpu` backend。这里所谓的
“M4 后端”和“x86 后端”，实际是同一后端的两个本机 CPU target，而不是两套
独立实现。

不要把硬件 target 与 `--cpu-codegen` 混为一谈：前者选择机器架构和对象文件
格式；后者选择 kernel lowering 策略。`scalar`、`vector`、`hierarchical` 三种
codegen 都可以在受支持的 M4 或 x86 主机上生成该主机的本机代码。

| 本机 | canonical host ID | LLVM target triple | AOT 动态库 | 链接方式 |
| --- | --- | --- | --- | --- |
| Apple Silicon（含 M4） | `darwin-arm64` | `arm64-apple-macosx13.0.0` | `.dylib` | `-dynamiclib` |
| macOS Intel | `darwin-x86_64` | `x86_64-apple-macosx13.0.0` | `.dylib` | `-dynamiclib` |
| Linux AArch64 | `linux-aarch64` | `aarch64-unknown-linux-gnu` | `.so` | `-shared` |
| Linux x86_64 / WSL2 | `linux-x86_64` | `x86_64-unknown-linux-gnu` | `.so` | `-shared` |

## 判定过程

1. Shell 入口使用 `uname -s` 和 `uname -m`，在 `scripts/platform.sh` 中归一化
   为 canonical host ID。
2. Python 编译入口使用 `platform.system()` 和 `platform.machine()`，选择相同的
   host ID 与 LLVM target triple。
3. 编译结果把 host ID 写入 `manifest.json` 的 `target` 字段。
4. runtime 使用编译期平台宏（例如 Apple arm64 的 `__arm64__`、Linux x86_64
   的 `__x86_64__`）计算自己的 target；两者不一致就拒绝加载 bundle。

因此，M4 产出的 `.tiree` bundle 不能直接复制到 x86 上运行，反之亦然。源码和
构建脚本是跨平台的，但 AOT 机器码必须在目标平台重新生成。

查看本机选择：

```bash
bash scripts/platform.sh
```

输出示例（WSL2 x86_64）：

```text
backend=llvm-cpu
host_id=linux-x86_64
target_triple=x86_64-unknown-linux-gnu
library=libtiny_iree_kernels.so
```

## 路径配置

仓库只自动尝试相邻的 IREE/LLVM build-tree，不包含 Apple 用户目录或磁盘卷的
绝对路径。工具链位于其他位置时使用：

```bash
export IREE_BUILD_DIR="..."
# 或
export LLVM_BUILD_DIR="..."
```

Apple hierarchical codegen 所需的 `libomp` 优先读取
`TINY_IREE_OPENMP_LIB_DIR`；未设置时通过 `brew --prefix libomp` 动态查询，
不会假定 Homebrew 安装在固定目录。
