# 阶段 04：HAL

## 学习目标

HAL 把“执行一次 workload”拆为两部分：模块级 `tiree_hal.executable` 描述可编译
entry point 和函数 ABI；函数内 `tiree_hal.dispatch` 选择设备并绑定输入输出 buffer。

Tiny-IREE 只实现同步 CPU，因此 device 固定为 `cpu-sync`。这个限制让我们能看清
HAL 的抽象职责，而不必先实现异步 command buffer、queue 和多后端 registry。

## 源码入口

- `include/tiny_iree/IR/TinyHALOps.td`：buffer、executable、dispatch 和生命周期 op。
- `compiler/IR/HALTypes.cpp`：HAL buffer type 的 parse/print。
- `compiler/Transforms/Passes.cpp`：`StreamToHALPass`。
- `tools/tiny-iree-export-codegen.cpp`：下一阶段读取的 executable plan 导出器。

## 实验

```bash
bash tests/stages/04_hal.sh
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
"$BUILD_DIR/bin/tiny-iree-opt" examples/mlp.mlir \
  --pass-pipeline='builtin.module(tiree-global-optimize,tiree-input-to-flow,tiree-flow-to-stream,tiree-stream-to-hal)'
```

检查 executable 的 `function_type` 是否与 dispatch 输入输出完全一致。修改任一
类型后，HAL verifier 应该拒绝模块。
