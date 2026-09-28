# Compiler 实现

这个目录保存 Tiny-IREE 编译器库的主体实现，对应 IREE 的 `compiler/` 职责。
`compiler/CMakeLists.txt` 唯一负责定义可复用的 `TinyIREECompiler` 库；上层
`CMakeLists.txt` 只发现 LLVM/MLIR、生成 TableGen 文件并装配各组件。

- `IR/`：注册 Input、Flow、Stream、HAL、VM dialect，实现 type 和 operation
  verifier。
- `Transforms/`：实现 global optimization 以及 Input → Flow → Stream → HAL → VM
  的阶段 lowering。
- `include/tiny_iree/IR/`：公共 dialect、op、type 声明和 TableGen 定义。
- `include/tiny_iree/Transforms/`：公共 pass 声明。

对应关系以职责为准，而非逐项复制 IREE 的规模：`IR/` 对应 Dialect，
`Transforms/` 合并了 GlobalOptimization、DispatchCreation 和
Input → Flow → Stream → HAL → VM pipelines；ONNX importer、pipeline orchestration
及 CPU kernel MLIR 生成器目前作为 `tools/` 中的可执行编译器前端保留。

`tools/tiny-iree-opt.cpp`、`tiny-iree-translate.cpp` 和
`tiny-iree-export-codegen.cpp` 是如 IREE 根目录 `tools/` 一样的薄 CLI 入口，
不会承载可复用 compiler implementation。

`runtime/` 不属于 compiler：它只消费 manifest、VM bytecode 和 native executable，
不会执行编译 pass。
