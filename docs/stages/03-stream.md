# 阶段 03：Stream 资源规划

## 学习目标

Flow 只描述 workload；Stream 开始处理“结果放在哪里、占多少字节、活到什么时候”。
每个结果对应一个 SSA `!tiree_stream.resource`，静态 tensor 由 shape 和 element
bit width 计算字节数，动态 shape 暂记为 `-1`。

`scheduleStreamDeallocs` 在最后使用点后插入 dealloc，函数输出则逃逸给调用者。
`tiree-verify-stream-resources` 独立检查重复 ID、双重释放和无效引用。

## 实验

```bash
bash tests/stages/03_stream_resources.sh
build/bin/tiny-iree-opt examples/mlp.mlir --tiree-compile-pipeline
```

重点沿着 tensor SSA value 和 resource SSA value 分别追踪，它们表达的是计算结果
和存储所有权两个不同概念。
