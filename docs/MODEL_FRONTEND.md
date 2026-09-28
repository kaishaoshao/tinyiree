# 模型前端如何识别

Tiny-IREE 把“识别模型”分成两层：先选择输入文件属于哪个 frontend，再在 ONNX
frontend 内识别 graph、value、initializer、node 和局部模式。两层不能混为一谈。

## 1. 选择输入 frontend

`tools/tiny_iree_compile.py` 支持 `onnx` 和 `mlir`：

```bash
# 自动模式：根据 .onnx 或 .mlir 扩展名选择
.venv/bin/python tools/tiny_iree_compile.py model.onnx -o model.tiree

# 无扩展名或非标准扩展名必须显式指定
.venv/bin/python tools/tiny_iree_compile.py model.pb \
  --input-type=onnx -o model.tiree
```

`--input-type=auto` 是默认值。ONNX protobuf 没有适合这里使用的稳定文件 magic，
所以 auto 模式不会猜测未知扩展名的二进制内容；它会要求用户显式指定。这避免
把损坏的 ONNX 文件误判成 MLIR，或把任意二进制交给错误的 parser。

选择 `mlir` 时不会经过 ONNX importer：文件被复制为 `module.input.mlir`，随后由
`tiny-iree-opt` 的 parser、dialect verifier 和 pass pipeline 验证。

## 2. ONNX 容器与图结构

ONNX 路径依次执行：

1. `onnx.load` 解析 protobuf。
2. `shape_inference.infer_shapes` 补全中间 value 的 shape。
3. `onnx.checker.check_model` 验证 ONNX 容器和 graph 基本合法性。
4. 从 `graph.input` 中排除同名 initializer，剩余项才是 runtime 用户输入。
5. 收集 graph input、`value_info`、graph output 和 initializer 的 shape/type。
6. 按 `graph.node` 的拓扑顺序转换；每个 ONNX value name 映射到一个 MLIR SSA
   value。引用尚未产生的 value 会立即报错。

模型输入和输出必须是 ranked f32 tensor。initializer 可接受 f32、int8、uint8 和
int64；普通权重会变成 `arith.constant dense<...>`，int64 shape initializer 只用于
ONNX shape/reshape 语义，不作为 runtime 输入。

可以在不编译的情况下观察识别结果：

```bash
.venv/bin/python tools/import_onnx.py --analyze model.onnx
```

输出包含 opset、真正的 graph input、initializer、按顺序排列的 node、graph output
以及每个 `(domain, op_type)` 是否在候选集合中。空 domain 会显示成标准域
`ai.onnx`。`op-type-supported=yes` 只代表域和名字已知；attribute、shape 和局部
模式仍要在实际 import 时继续验证。

## 3. 算子与模式识别

普通算子按 `(NodeProto.domain, NodeProto.op_type)` 分派。当前只接受空 domain 和
它的标准写法 `ai.onnx`；自定义 domain 即使复用了 `MatMul` 这个名字也会被拒绝。
`Gemm` 会展开为 MatMul 和可选 Add；`Flatten` 会归一化为 Reshape。Q/DQ 是局部
模式而不是两个独立 kernel：

```text
QuantizeLinear(x, scale, zero_point)
          -> DequantizeLinear(q, same_scale, same_zero_point)
          -> tiree_input.fake_quant
```

scale 和 zero point 必须是 scalar/单元素 initializer，zero point 必须是 signed int8，
两端参数名称必须匹配。没有配对的 QuantizeLinear 会被拒绝。

当前支持矩阵如下：

| ONNX op | 限制/归一化 |
| --- | --- |
| MatMul | rank-2 f32，形状为 `(MxK) x (KxN) -> (MxN)` |
| Gemm | 不支持 transpose，`alpha=beta=1`，可带 bias |
| Add | 等形状，或最后一维 bias broadcast |
| Relu | ranked f32，输入输出类型相同 |
| Softmax | 只支持最后一维 |
| QuantizeLinear + DequantizeLinear | 配对的 per-tensor signed-int8 fake quant |
| Split | 只沿最后一维，恰好两个等形状输出 |
| Conv | 静态 NCHW，必须有 bias，valid、stride 1、dilation 1、group 1 |
| MaxPool | 静态 NCHW，2x2、stride 2、无 padding |
| Flatten | 只支持 `axis=1`，归一化为 Reshape |
| Reshape | 静态 shape，总元素数不变 |
| Transpose | 只支持 rank-4 NCHW -> NHWC `[0,2,3,1]` |

机器可读的当前清单直接来自 importer 中同一份定义：

```bash
.venv/bin/python tools/import_onnx.py --list-supported-ops
```

## 4. 从 ONNX value 到 MLIR SSA

- graph input 依次变成 `%arg0`、`%arg1`。
- initializer 变成 `%c0`、`%c1` 等 `arith.constant`。
- node output 依次变成 `%v0`、`%v1`。
- graph output name 最后必须能在 SSA 映射中找到，否则 importer 拒绝模型。
- 多输出 Split 一次产生两个 SSA result；多 graph output 则生成多值 `return`。

生成的 `module.input.mlir` 还不是“因为文本成功生成就一定正确”。下一层
Input dialect verifier 会再次检查 MatMul 维度、Add broadcast、Conv layout、Reshape
元素数等约束。这形成 ONNX checker、importer 约束、MLIR verifier 三层防线。

## 5. 错误边界与可追溯性

以下情况会显式失败，而不是静默猜测：

- auto 模式遇到未知扩展名；
- ONNX checker 或 shape inference 失败；
- 未知 `(domain, op_type)`；
- 已知 op 使用不支持的 attribute/layout/type；
- node 使用尚不存在的 value；
- Q/DQ 参数不匹配或没有配对；
- graph output 没有对应的 SSA value。

成功编译后，`manifest.json` 的 `input_type` 记录 `onnx` 或 `mlir`，`source` 记录
源文件，便于确认 bundle 来自哪个 frontend。前端识别回归入口是：

```bash
bash tests/stages/08_onnx_frontend.sh
```

## 6. 增加一个新 ONNX op 时要改什么

至少沿下面路径补齐，不要只在字符串集合里加入名字：

1. 在 `import_onnx.py` 加入 attribute/type/shape 限制和 lowering。
2. 在 `TinyInputOps.td` 定义或复用 Input op，并在 `Ops.cpp` 写 verifier。
3. 在 Flow/Stream/HAL lowering 中建立 dispatch 与资源信息。
4. 在 `generate_kernel_mlir.py` 生成数值 kernel。
5. 必要时扩展 native ABI/runtime。
6. 同时加入正确模型、错误 attribute/shape 模型和 ONNX ReferenceEvaluator 数值测试。

如果某个算子只加入 `SUPPORTED_OPS` 而没有完成上述纵向路径，它不应被视为真正
支持。
