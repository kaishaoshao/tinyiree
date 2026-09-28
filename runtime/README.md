# Runtime 实现

这个目录保存可执行 runtime 的主体实现，与编译器工具分开：

- `Runtime.cpp`：校验 bundle、解析 Tiny VM bytecode、管理资源、dispatch 解释器或
  native kernel，并实现 benchmark loop。
- `include/tiny_iree/Runtime/Runtime.h`：公开 CLI 入口。
- `include/tiny_iree/Runtime/VMBytecode.h`：定义 bytecode 版本和 opcode。
- `include/tiny_iree/Runtime/NativeABI.h`：定义 generic native fallback ABI。
- `tools/tiny-iree-run-module.cpp`：只保留可执行程序的薄 wrapper。

runtime 目前仍保持为一个主体 translation unit，方便沿着 bundle loading、VM
dispatch、resource lifetime 和 native call 顺序学习。格式与 ABI 细节见
`docs/VM_BYTECODE_AND_NATIVE_ABI.md`。
