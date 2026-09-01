# 阶段 08：ONNX 前端

## 学习目标

importer 读取 ONNX protobuf，执行 shape inference，区分模型输入与 initializer，
再按拓扑顺序把受支持节点转换为 `tiree_input` op。权重变为
`arith.constant dense<...>`，所以 runtime 只需要接收真正的用户输入。

前端必须显式拒绝不支持的 op、attribute、layout 和动态情况。静默猜测语义会让
后端得到“合法但错误”的程序，这比直接报错更危险。

## 实验

```bash
bash tests/stages/08_onnx_frontend.sh
```

先单独运行 `tools/import_onnx.py` 查看高层 IR，再用 ONNX ReferenceEvaluator 与
Tiny-IREE 逐元素比较。这建立了前端语义的可信基线。
