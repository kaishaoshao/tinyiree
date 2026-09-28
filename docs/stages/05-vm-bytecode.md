# 阶段 05：VM 与字节码

## 学习目标

VM IR 是 host 控制程序，不包含 MatMul 的循环实现。它只保留 ref allocation、
调用哪个 executable、何时释放以及返回哪些 tensor。`tiny-iree-translate` 将其
序列化为小端二进制格式，magic 为 `TIREVM1\0`。

字节码中的每条指令都有显式 opcode 和长度/数量字段。serializer 会拒绝未知 op、
先使用后定义和 ref 元数据不匹配，避免把任意 MLIR 当作可执行模块。

## 源码入口

- `include/tiny_iree/IR/TinyVMOps.td`：ref、alloc、call、dealloc。
- `lib/Transforms/Passes.cpp`：`HALToVMPass`。
- `include/tiny_iree/Runtime/VMBytecode.h`：opcode 和磁盘格式结构。
- `tools/tiny-iree-translate.cpp`：MLIR VM module 的验证与序列化。

## 实验

```bash
bash tests/stages/05_vm_bytecode.sh

source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-stage05.XXXXXX")"
"$BUILD_DIR/bin/tiny-iree-opt" examples/mlp.mlir \
  --tiree-compile-pipeline -o "$WORK_DIR/module.vm.mlir"
"$BUILD_DIR/bin/tiny-iree-translate" "$WORK_DIR/module.vm.mlir" \
  -o "$WORK_DIR/module.tvm"
od -Ax -tx1 "$WORK_DIR/module.tvm" | head
printf 'artifacts: %s\n' "$WORK_DIR"
```

前 8 字节应为 `54 49 52 45 56 4d 31 00`，即 `TIREVM1\0`。比较
`module.vm.mlir` 和二进制字段。此时已经有控制模块，但还没有 native kernel 和
runtime，所以仍不能得到数值结果。
