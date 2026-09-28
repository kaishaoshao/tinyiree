# 阶段 03：Stream 资源规划

## 学习目标

Flow 只描述 workload；Stream 开始处理“结果放在哪里、占多少字节、活到什么时候”。
每个结果对应一个 SSA `!tiree_stream.resource`，静态 tensor 由 shape 和 element
bit width 计算字节数，动态 shape 暂记为 `-1`。

`scheduleStreamDeallocs` 在最后使用点后插入 dealloc，函数输出则逃逸给调用者。
`tiree-verify-stream-resources` 独立检查重复 ID、双重释放和无效引用。

## 源码入口

- `include/tiny_iree/IR/TinyStreamOps.td`：resource、alloc、dispatch、dealloc。
- `lib/Transforms/Passes.cpp`：`FlowToStreamPass` 和资源释放点安排。
- `lib/IR/Ops.cpp`：Stream resource verifier。
- `tests/invalid_stream_resources.mlir`：释放后使用的反例。

## 实验

```bash
bash tests/stages/03_stream_resources.sh
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
"$BUILD_DIR/bin/tiny-iree-opt" examples/mlp.mlir \
  --pass-pipeline='builtin.module(tiree-global-optimize,tiree-input-to-flow,tiree-flow-to-stream)'
```

重点沿着 tensor SSA value 和 resource SSA value 分别追踪，它们表达的是计算结果
和存储所有权两个不同概念。
