# 阶段 06：CPU Codegen

## 学习目标

`tiny-iree-export-codegen` 从 HAL executable 提取 entry point、workload 和 tensor
ABI。`generate_kernel_mlir.py` 将 plan 变为 Linalg on MemRef kernel，随后使用标准
MLIR pass 降到 LLVM dialect、LLVM IR、object 和动态库。

这一阶段与 VM 路径并行：VM 只保存调用名字，native library 导出同名 C interface。
二者由 runtime 在下一阶段绑定。

## 实验

```bash
bash tests/stages/06_cpu_codegen.sh
```

依次阅读生成的 `module.executable.mlir`、`.llvm.mlir` 和 `.ll`，找到同一个循环、
memref descriptor 和 `_mlir_ciface_` 符号在不同抽象层的表示。
