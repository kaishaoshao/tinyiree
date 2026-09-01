# 阶段 07：Runtime

## 学习目标

runtime 首先验证 manifest 和 `TIREVM1` bytecode，再为 VM value 建立 tensor 表和
resource 表。遇到 call 时，它通过 `dlopen/dlsym` 找到 HAL entry point 对应的
`_mlir_ciface_` 函数，用 ranked memref descriptor 传递数据。

allocator 将已释放的相同大小 backing storage 放入池中。VM 控制生命周期，native
kernel 只读取输入 memref 并写入预先分配的输出，因此 runtime 不需要保存算子库。

## 实验

```bash
bash tests/stages/07_runtime.sh
```

把动态库临时改名再运行，观察 loader 层错误；再修改 bytecode magic，观察模块在
执行任何 kernel 前被拒绝。这能区分 loader、VM 和 kernel 三类问题。
