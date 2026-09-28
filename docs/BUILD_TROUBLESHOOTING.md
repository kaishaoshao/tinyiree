# 构建排错

## 找不到 MLIRConfig.cmake

设置 IREE 或独立 LLVM build-tree 后重试：

```bash
# IREE build-tree（Apple M4 常用布局）
export IREE_DIR=../iree
export IREE_BUILD_DIR="$IREE_DIR/build_tools/build-host"

# 或独立 LLVM/MLIR build-tree（WSL/Linux x86_64 常用布局）
export LLVM_BUILD_DIR=../llvm-project/build
bash scripts/build.sh
```

不需要把某台 Mac 的磁盘卷路径写进仓库；工具链不在相邻目录时，只在
当前 shell 或个人 shell 配置中设置上述环境变量。

## 命中错误版本的 MLIR 头文件

本机可能同时存在包管理器安装版和 IREE build-tree 中的 MLIR。不要混用不同
版本的 `mlir-tblgen`、头文件和静态库。`scripts/build.sh` 会显式传入
`MLIR_DIR` 与 `LLVM_DIR`；`CMAKE_NO_SYSTEM_FROM_IMPORTED` 只控制 imported target
的头文件是否被视为 system include，并不负责选择 MLIR 版本。

检查实际编译命令：

```bash
source scripts/platform.sh
ninja -C "$(tiny_iree_build_dir "$PWD")" -v tiny-iree-opt
```

## build 缓存来自另一台机器

如果 `build/CMakeCache.txt` 记录的是另一条源码路径（例如从 Apple M4 拷到
WSL），脚本会自动改用 `build-<os>-<arch>`。也可以显式隔离构建目录：

```bash
export TINY_IREE_BUILD_DIR="$PWD/build-linux-x86_64"
bash scripts/build.sh
```

## IREE 源码目录被移动

本仓库不重新配置完整 IREE，只复用它已有的 LLVM/MLIR build-tree。只要
`MLIRConfig.cmake`、`mlir-opt`、`mlir-translate` 和 `llc` 仍在，Tiny-IREE 可以
独立构建。官方 IREE 对照测试还要求 `iree-compile` 和 `iree-run-module`。

## Python 缺少 ONNX

```bash
bash scripts/setup_python.sh
export TINY_IREE_PYTHON="$PWD/.venv/bin/python"
```

## Apple hierarchical codegen 找不到 libomp

先安装并让 `brew` 可从 `PATH` 找到：

```bash
brew install libomp
brew --prefix libomp
```

编译脚本会通过 `brew --prefix libomp` 查询安装位置。使用其他安装方式时设置：

```bash
export TINY_IREE_OPENMP_LIB_DIR=".../lib"
```
