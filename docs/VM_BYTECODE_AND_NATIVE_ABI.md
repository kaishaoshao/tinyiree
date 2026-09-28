# VM 字节码与 Native ABI

Tiny-IREE bundle 把 host 控制程序和 CPU 机器码分开：`module.tvm` 描述 value、资源
生命周期和 dispatch；`libtiny_iree_kernels.so`（Linux）或 `.dylib`（macOS）保存
native kernels。`manifest.json` 将二者与目标平台绑定。

本文描述仓库当前的 `tiny-vm-bytecode-v1`，不是上游 IREE VM FlatBuffer 格式，也
不是稳定的跨版本公共 ABI。修改字段或调用约定时必须升级版本或保持兼容解析。

## 1. Bundle 装载顺序

runtime 对目录输入执行：

1. 读取并解析 `manifest.json`。
2. 验证 `format == tiny-iree-bundle-v1`。
3. 检查 `target` 与当前 runtime 的 OS/架构一致。
4. 拒绝绝对路径和包含 `..` 的 artifact 路径。
5. 读取 `vm_module`，并用 `dlopen(..., RTLD_NOW | RTLD_LOCAL)` 加载
   `hal_executable`。
6. 校验 bytecode header，解析全部函数和指令，然后执行指定 `--function`。

Apple M4 和 Linux x86_64 使用相同 VM 格式；不同的是 manifest target、native
动态库格式和库内机器码。不能把某台机器生成的 bundle 复制到另一架构直接运行。

## 2. 基本编码规则

- 所有整数使用 little-endian。
- `u8/u32/u64/i64` 分别是固定宽度 1/4/8/8 字节。
- string 编码为 `u32 byte_length`，随后是不带结尾 `NUL` 的原始字节。
- tensor type 编码为 `u32 element_type`、`u32 rank`、`rank * i64 dims`。
- 当前唯一 element type 是 `1 = f32`；动态维写作 `-1`。
- runtime 当前限制 `rank <= 4`，维度不能为 0 或小于 -1。

文件头：

```text
8 bytes  magic = 54 49 52 45 56 4d 31 00  ("TIREVM1\0")
u32      version = 1
u32      function_count
```

每个函数：

```text
string   function_name
u32      argument_count
type[]   argument tensor types
u32      instruction_count
inst[]   instructions
```

parser 还限制函数数、参数数和单条 call 的输入/输出数不超过 1024，单函数指令数
不超过 1,000,000，并拒绝文件末尾的额外数据。

## 3. 指令布局

| Opcode | 名称 | Payload |
| --- | --- | --- |
| 1 | Constant | `u32 value_id, type, u64 element_count, element_count*u32 f32_bits` |
| 2 | Alloc | `i64 resource_id, i64 bytes` |
| 3 | Call | 见下方 |
| 4 | Dealloc | `i64 resource_id` |
| 5 | Return | `u32 count, count*u32 value_id` |

Call 的 payload：

```text
string callee
string device                 # 当前必须为 cpu-sync
u32 input_count
u32 input_value_ids[input_count]
u32 output_count
repeat output_count times:
  u32 output_value_id
  type output_tensor_type
  i64 resource_id
  i64 bytes                   # 动态 allocation 可为 -1
```

`value_id` 表示 SSA 风格数据依赖，`resource_id` 表示 backing storage 生命周期，两者
不能混用。Alloc 建立资源，Call 把结果写入对应资源，Dealloc 释放资源，Return 只能
返回仍然 live 的资源。serializer 和 runtime 都会检查先定义后使用、重复定义、大小
匹配和释放后读取。

查看文件头：

```bash
od -Ax -tx1 -N 16 bundle/module.tvm
```

前 16 字节应为 magic、`01 00 00 00` 版本和 little-endian function count。

## 4. Native symbol 查找

每个 VM Call 的 `callee` 最终对应动态库符号。runtime 先查找 MLIR C interface：

```text
_mlir_ciface_tiree_kernel_<callee>
```

找不到时才尝试 Tiny-IREE generic ABI：

```text
tiree_kernel_<callee>
```

Linux 查看符号：

```bash
nm -D --defined-only bundle/libtiny_iree_kernels.so | \
  grep tiree_kernel
ldd bundle/libtiny_iree_kernels.so
```

macOS：

```bash
nm -gU bundle/libtiny_iree_kernels.dylib | grep tiree_kernel
otool -L bundle/libtiny_iree_kernels.dylib
```

`nm` 在 macOS 展示的 C 符号可能多一个平台前导 `_`；runtime 传给 `dlsym` 的名字
仍是上面列出的字符串。

## 5. MLIR C interface memref ABI

默认 codegen 为 kernel 加 `llvm.emit_c_interface`。runtime 为每个输入和输出构造
ranked memref descriptor，并把 descriptor 地址按“所有输入、所有输出”的顺序传入
`_mlir_ciface_...`：

```text
word 0                 allocated pointer
word 1                 aligned pointer
word 2                 offset = 0
word 3 .. 3+rank-1     sizes[rank]
word 3+rank .. end     row-major strides[rank]
```

当前实现以 `uint64_t` 保存 descriptor word，面向 64 位宿主；allocated/aligned 都
指向 runtime 管理的连续 f32 storage。stride 从最后一维的 1 反向累乘。wrapper
返回 `void`，当前 runtime 支持输入加输出 descriptor 总数 2 到 5。

## 6. Generic C ABI fallback

[NativeABI.h](../include/tiny_iree/Runtime/NativeABI.h) 定义：

```c
typedef struct tiree_tensor_view_t {
  float *data;
  int64_t rank;
  int64_t dims[4];
} tiree_tensor_view_t;

typedef int (*tiree_kernel_fn_t)(const tiree_tensor_view_t *inputs,
                                 int64_t input_count,
                                 tiree_tensor_view_t *output);
```

返回 0 表示成功；fallback 只支持一个输出，rank 不超过 4。该 ABI 不传 stride，
所以 tensor 必须是连续 row-major f32。

两种 ABI 都遵守相同所有权规则：runtime 分配并持有输入/输出 storage，kernel 只在
调用期间借用指针，不得释放、替换或保存指针供返回后使用。输出 shape 和字节数由
VM/resource metadata 决定，kernel 不负责扩容。

## 7. 版本与兼容性检查表

改变以下任一项时应同步 serializer、parser、测试和本文：magic/version、整数宽度、
tensor type、opcode payload、symbol 前缀、memref descriptor、最大 rank、输出数量或
所有权。若旧 runtime 不能安全读取新布局，应提升 `kVersion`，不要让同一版本号
表达两种格式。

相关实现入口：

- `include/tiny_iree/Runtime/VMBytecode.h`
- `tools/tiny-iree-translate.cpp`
- `include/tiny_iree/Runtime/NativeABI.h`
- `tools/tiny-iree-run-module.cpp`
- `tools/generate_kernel_mlir.py`
- `tests/run_onnx_aot_e2e.sh`
