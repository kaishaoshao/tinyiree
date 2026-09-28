# 阶段 06：CPU Codegen

## 学习目标

`tiny-iree-export-codegen` 从 HAL executable 提取 entry point、workload 和 tensor
ABI。`generate_kernel_mlir.py` 将 plan 变为 Linalg on MemRef kernel，随后使用标准
MLIR pass 降到 LLVM dialect、LLVM IR、object 和动态库。

这一阶段与 VM 路径并行：VM 只保存调用名字，native library 导出同名 C interface。
二者由 runtime 在下一阶段绑定。

## 源码入口

- `tools/tiny-iree-export-codegen.cpp`：HAL executable 到 JSON codegen plan。
- `tools/generate_kernel_mlir.py`：plan 到 Linalg/Vector/OpenMP kernel MLIR。
- `scripts/codegen.sh`：MLIR -> LLVM dialect -> LLVM IR -> object -> 动态库。
- `include/tiny_iree/Runtime/NativeABI.h`：runtime 与 native kernel 共用的 ABI。

## 实验

```bash
bash tests/stages/06_cpu_codegen.sh

source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-stage06.XXXXXX")"
"$BUILD_DIR/bin/tiny-iree-opt" examples/mlp.mlir \
  --pass-pipeline='builtin.module(tiree-global-optimize,tiree-input-to-flow,tiree-flow-to-stream,tiree-stream-to-hal)' \
  -o "$WORK_DIR/module.hal.mlir"
bash scripts/codegen.sh "$WORK_DIR/module.hal.mlir" "$WORK_DIR/aot"
printf 'artifacts: %s\n' "$WORK_DIR/aot"
```

依次阅读生成的 `module.executable.mlir`、`.llvm.mlir` 和 `.ll`，找到同一个循环、
memref descriptor 和 `_mlir_ciface_` 符号在不同抽象层的表示。最后用 `file` 检查
动态库：M4 应为 arm64 Mach-O，WSL x86_64 应为 x86-64 ELF。
