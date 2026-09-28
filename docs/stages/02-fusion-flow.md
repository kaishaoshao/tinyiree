# 阶段 02：融合与 Flow

## 学习目标

`tiree-global-optimize` 使用 `RewritePattern` 把只有单一消费者的
`MatMul -> Add(bias) -> Relu` 改写为一个 fused op。随后
`tiree-input-to-flow` 将 workload 克隆进 `IsolatedFromAbove` dispatch region。

Flow 的职责不是执行算子，而是确定一次独立派发的边界和 ABI。region 参数表示
显式 capture，`tiree_flow.yield` 表示 dispatch 结果。

## 源码入口

- `lib/Transforms/Passes.cpp`：`GlobalOptimizationPass` 和 `InputToFlowPass`。
- `include/tiny_iree/IR/TinyFlowOps.td`：带 `IsolatedFromAbove` trait 的 dispatch。
- `tests/invalid_flow_region.mlir`：kernel 与 workload 顺序不一致的失败路径。

## 实验

```bash
bash tests/stages/02_flow_fusion.sh
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
"$BUILD_DIR/bin/tiny-iree-opt" examples/mlp.mlir \
  --tiree-global-optimize --mlir-print-ir-after-all
```

尝试让 MatMul 结果同时被两个 op 使用，融合应该安全地拒绝匹配。然后比较
`--tiree-global-optimize` 与 `--tiree-input-to-flow` 后的 IR：前者改变 workload，
后者改变 dispatch 边界，两者职责不同。
