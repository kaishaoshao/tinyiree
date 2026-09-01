# 构建排错

## 找不到 MLIRConfig.cmake

设置 IREE build-tree 后重试：

```bash
export IREE_BUILD_DIR=/Volumes/wsk/code/llvm-mlir/iree/build_tools/build-host
bash scripts/build.sh
```

## 命中错误版本的 MLIR 头文件

本机可能同时存在 Homebrew、`/usr/local` 和 IREE build-tree 的 MLIR。不要混用
不同版本的 `mlir-tblgen`、头文件和静态库。本仓库通过
`CMAKE_NO_SYSTEM_FROM_IMPORTED` 优先使用 `MLIR_DIR/LLVM_DIR` 对应版本。

检查实际编译命令：

```bash
ninja -C build -v tiny-iree-opt
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
