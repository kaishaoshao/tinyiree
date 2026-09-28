# Compiler 实现

这个目录保存 Tiny-IREE 编译器库的主体实现，对应 IREE 的 compiler 职责：

- `IR/`：注册 Input、Flow、Stream、HAL、VM dialect，实现 type 和 operation
  verifier。
- `Transforms/`：实现 global optimization 以及 Input → Flow → Stream → HAL → VM
  的阶段 lowering。
- `include/tiny_iree/IR/`：公共 dialect、op、type 声明和 TableGen 定义。
- `include/tiny_iree/Transforms/`：公共 pass 声明。

`tools/tiny-iree-opt.cpp`、`tiny-iree-translate.cpp` 和
`tiny-iree-export-codegen.cpp` 是编译器库的 CLI 驱动。Python 的 ONNX importer、
pipeline orchestration 和 CPU kernel MLIR 生成器也保留在 `tools/`，因为它们是当前
教学实现的可直接运行入口，不是独立可链接的 C++ compiler library。

`runtime/` 不属于 compiler：它只消费 manifest、VM bytecode 和 native executable，
不会执行编译 pass。
