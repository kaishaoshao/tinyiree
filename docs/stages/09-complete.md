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

## 回归

```bash
bash tests/stages/09_complete_pipeline.sh
```

通过后，本仓库形成 ONNX -> 分层 MLIR -> Tiny VM bytecode + native executable
-> runtime 的 CPU 教学闭环。它仍不声称兼容官方 VMFB，也不包含 GPU、异步调度、
完整 ONNX 或真正 i8 MatMul。
