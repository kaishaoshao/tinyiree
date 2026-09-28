# 阶段 08：ONNX 前端

## 学习目标

importer 读取 ONNX protobuf，执行 shape inference，区分模型输入与 initializer，
再按拓扑顺序把受支持节点转换为 `tiree_input` op。权重变为
`arith.constant dense<...>`，所以 runtime 只需要接收真正的用户输入。

前端必须显式拒绝不支持的 op、attribute、layout 和动态情况。静默猜测语义会让
后端得到“合法但错误”的程序，这比直接报错更危险。

## 源码入口

- `tools/import_onnx.py`：shape inference、initializer 读取、拓扑转换和限制检查。
- `tools/run_onnx_reference.py`：ONNX `ReferenceEvaluator` 数值基线。
- `tools/generate_mlp_onnx.py`：可重复生成的最小测试模型。
- `tests/run_onnx_aot_e2e.sh`：逐阶段产物、AOT 架构、错误 target 和坏字节码测试。

## 实验

```bash
bash tests/stages/08_onnx_frontend.sh

bash scripts/setup_python.sh
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-stage08.XXXXXX")"
.venv/bin/python tools/generate_mlp_onnx.py -o "$WORK_DIR/mlp.onnx"
.venv/bin/python tools/import_onnx.py "$WORK_DIR/mlp.onnx" \
  -o "$WORK_DIR/module.input.mlir"
sed -n '1,120p' "$WORK_DIR/module.input.mlir"
```

先单独运行 `tools/import_onnx.py` 查看高层 IR，再用 ONNX ReferenceEvaluator 与
Tiny-IREE 逐元素比较。这建立了前端语义的可信基线。
