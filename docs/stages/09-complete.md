# 阶段 09：扩展完整闭环

## 本阶段能力

- 动态 shape 和显式运行时 shape 绑定。
- 多模型输入、多个函数输出和单 dispatch 多结果 ABI。
- ONNX Q/DQ 的 per-tensor int8 fake-quant 数值语义。
- Transform Dialect 驱动的 tiling 和 masked vectorization。
- 按精确大小复用已释放的瞬态 buffer。
- Conv、MaxPool、Reshape、Transpose 的受限教学实现。
- 同一 MLP 与官方 IREE VMFB 的数值对照。

## 学习方法

不要一次阅读所有扩展。为每个能力沿同一纵向路径检查：ONNX importer、Input op
与 verifier、Flow workload、codegen、runtime ABI、数值测试。这样能清楚区分
“新增算子”和“修改编译器架构”。

## 纵向源码与测试索引

| 能力 | 主要实现 | 端到端测试 |
| --- | --- | --- |
| 动态 shape | importer、Stream/VM `bytes = -1`、runtime shape 绑定 | `run_dynamic_e2e.sh` |
| 多输入/多结果 | Flow/HAL/VM result arrays、native ABI | `run_multi_io_e2e.sh`、`run_multi_result_dispatch_e2e.sh` |
| Q/DQ fake quant | `TinyInput_FakeQuantOp`、kernel generator | `run_quantized_e2e.sh` |
| Transform tiling/vector | `generate_kernel_mlir.py` 的 named sequence | `run_tiling_vector_e2e.sh` |
| buffer 复用 | Stream dealloc 与 `RuntimeAllocator` free list | `run_resource_reuse_e2e.sh` |
| CNN 子集 | importer、Input verifier、kernel generator | `run_cnn_e2e.sh` |

这里的量化是 Q/DQ 的浮点 fake-quant 语义，不是 i8 MatMul；动态 shape 也只在
受支持算子和显式输入 shape 绑定范围内成立。

## 回归

```bash
bash tests/stages/09_complete_pipeline.sh
```

通过后，本仓库形成 ONNX -> 分层 MLIR -> Tiny VM bytecode + native executable
-> runtime 的 CPU 教学闭环。它仍不声称兼容官方 VMFB，也不包含 GPU、异步调度、
完整 ONNX 或真正 i8 MatMul。
