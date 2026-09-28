# Tiny-IREE 与 IREE 阶段对照

| Tiny-IREE | 对应 IREE 职责 | 可观察产物 | 范围 |
| --- | --- | --- | --- |
| ONNX importer | Input conversion | `module.input.mlir` | 受限算子集 |
| Input verifier/fusion | Preprocessing/GlobalOptimization | `module.global-opt.mlir` | MatMul+Add+Relu |
| Flow | Dispatch formation | `module.flow.mlir` | isolated workload |
| Stream | Resource scheduling | `module.stream.mlir` | 同步 SSA lifetime |
| HAL | Device/executable abstraction | `module.hal.mlir` | cpu-sync/llvm-cpu |
| VM | Host control program | `module.vm.mlir` | alloc/call/dealloc |
| Tiny bytecode | VM target | `module.tvm` | 自有 `TIREVM1` 格式 |
| CPU codegen | Executable backend | Linalg/Vector/OpenMP/LLVM/dylib/so | 本机 CPU |
| Runtime | VM/HAL loader | 推理结果与 allocator 统计 | 同步执行 |

“缩小复刻”表示每个核心阶段都有独立 IR、真实 lowering、产物和测试，不表示
二进制兼容官方 VMFB。当前不包含完整 ONNX、真正 i8 MatMul、异步多线程调度、
GPU 后端、成本模型、自动调优和官方 VM ABI。

已通过的非占位能力包括：多结果 Split 贯穿 native ABI；动态维度在 runtime
绑定；vector codegen 生成真实 masked vector IR；三段 ReLU 展示 backing storage
复用；CNN 子集与 ONNX ReferenceEvaluator 逐元素一致；静态二维 MatMul 可生成
L2/L1/register 三级 blocking、OpenMP 外层并行和 vector register tile。

这里的 `llvm-cpu` 是统一 backend；Apple M4 与 Linux x86_64 的差异是 target
triple、动态库格式和 bundle target，而不是两套独立的编译器后端。

前端格式由扩展名或显式 `--input-type` 选择；ONNX 算子识别基于标准
`(NodeProto.domain, NodeProto.op_type)`，再由 importer 检查 attribute、shape、
element type 和受支持的局部模式。`--analyze` 只报告图结构和算子身份，完整语义
是否合法仍以实际导入和 Input dialect verifier 为准。
