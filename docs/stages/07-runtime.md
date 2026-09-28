# 阶段 07：Runtime

## 学习目标

runtime 首先验证 manifest 和 `TIREVM1` bytecode，再为 VM value 建立 tensor 表和
resource 表。遇到 call 时，它通过 `dlopen/dlsym` 找到 HAL entry point 对应的
`_mlir_ciface_` 函数，用 ranked memref descriptor 传递数据。

allocator 将已释放的相同大小 backing storage 放入池中。VM 控制生命周期，native
kernel 只读取输入 memref 并写入预先分配的输出，因此 runtime 不需要保存算子库。

## 源码入口

- `runtime/Runtime.cpp`：manifest 校验、bytecode interpreter、allocator、
  `dlopen`/`dlsym`、`_mlir_ciface_` 调用和 benchmark loop。
- `tools/tiny-iree-run-module.cpp`：保持命令行工具名称稳定的薄 `main` wrapper。
- `include/tiny_iree/Runtime/Runtime.h`：runtime CLI 入口声明。
- `include/tiny_iree/Runtime/NativeABI.h`：ranked memref descriptor ABI。
- `tests/invalid_vm_resources.mlir`：无效输出 resource 的失败路径。
- `examples/runtime_inputs.mlir`：多输入 runtime 示例。

## 实验

```bash
bash tests/stages/07_runtime.sh
```

把动态库临时改名再运行，观察 loader 层错误；再修改 bytecode magic，观察模块在
执行任何 kernel 前被拒绝。测试脚本只在 `mktemp` 目录中做这些破坏性实验，并在
退出时删除副本，不会修改仓库文件。这能区分 loader、VM 和 kernel 三类问题。
