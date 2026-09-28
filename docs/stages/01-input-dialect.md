# 阶段 01：Input Dialect

## 学习目标

这一阶段把模型计算表示成 `tiree_input` MLIR。重点掌握 TableGen/ODS 如何定义
operation，以及 verifier 如何检查 tensor element type、rank、shape 和 broadcast。

## 源码入口

- `include/tiny_iree/IR/TinyInputOps.td`：声明 MatMul、Add、Relu、Softmax。
- `lib/IR/Dialects.cpp`：注册 Dialect 和 operation。
- `lib/IR/Ops.cpp`：实现 shape/type verifier。
- `tools/tiny-iree-opt.cpp`：最小 MLIR optimizer driver。

## 实验

```bash
bash tests/stages/01_input_dialect.sh
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
"$BUILD_DIR/bin/tiny-iree-opt" examples/mlp.mlir
```

然后修改 `tests/invalid_input.mlir` 的矩阵维度，观察 verifier 的错误信息。此时
编译器只能解析和验证高层 IR，还没有 dispatch、资源和执行能力。

观察时区分两层错误：ODS 生成的结构验证负责 operand/result 个数和基础类型约束，
`lib/IR/Ops.cpp` 中的自定义 verifier 负责矩阵维度、broadcast 和算子语义。
