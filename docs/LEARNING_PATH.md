# Tiny-IREE 分阶段学习路线

| Tag | 阶段 | 需要回答的问题 | 验收对象 |
| --- | --- | --- | --- |
| `stage-00` | 环境和全局架构 | 编译器与 runtime 如何分工？ | 环境测试 |
| `stage-01` | Input Dialect | 模型算子如何成为可验证的 MLIR op？ | `tiree_input` IR |
| `stage-02` | 融合与 Flow | 算子链如何形成 dispatch？ | `tiree_flow.dispatch` |
| `stage-03` | Stream | tensor 结果如何获得资源生命周期？ | alloc/dispatch/dealloc |
| `stage-04` | HAL | 编译器如何抽象设备与 executable？ | HAL buffer 和 entry point |
| `stage-05` | VM 与字节码 | host 控制程序如何序列化？ | `TIREVM1` bytecode |
| `stage-06` | CPU Codegen | tensor workload 如何变成本机机器码？ | Linalg、LLVM、dylib/so |
| `stage-07` | Runtime | VM、allocator 和 native kernel 如何协作？ | 本机推理结果 |
| `stage-08` | ONNX 前端 | protobuf 计算图如何导入高层 IR？ | ONNX 与 Tiny-IREE 数值对照 |
| `stage-09` | 完整能力 | 动态 shape、量化、向量化和复用如何贯通？ | 全量回归 |
| `stage-10` | 多级并行 Tiling | cache blocking 如何映射到 CPU 线程和向量？ | SCF/OpenMP/LLVM IR |

## 学习纪律

每次只检出一个阶段，先读对应文档，再运行测试，然后修改一个很小的能力。
推荐实验顺序是：修改 verifier、观察错误；修改 rewrite、比较前后 IR；最后修改
runtime 并做数值对照。不要同时调试 ONNX、MLIR lowering 和 native ABI。

## 专题附录

- `MODEL_FRONTEND.md`：模型格式、ONNX 图和算子模式如何识别。
- `VM_BYTECODE_AND_NATIVE_ABI.md`：VM v1 的逐字段布局及 native kernel 边界。
- `PERFORMANCE_BENCHMARKING.md`：区分冷启动、稳态调用和 kernel 性能的测量方法。
- `TROUBLESHOOTING.md`：从构建、前端、IR、bundle、VM 到 ABI 的故障定位顺序。
