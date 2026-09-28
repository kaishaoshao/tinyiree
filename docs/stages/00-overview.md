# 阶段 00：环境与全局架构

## 目标

先建立全局地图，不写算子。最终流水线分为两条相互配合的路径：

```text
ONNX -> Input -> Flow -> Stream -> HAL -> VM -> Tiny bytecode
                                HAL -> Linalg -> LLVM -> native library

Tiny bytecode + native library -> runtime -> result
```

VM 字节码负责资源和调用控制，native library 负责数值计算。把两者分开，是理解
IREE “host program + device executable”结构的关键。

## 实验

先查看平台选择，再检查构建环境：

```bash
bash scripts/platform.sh
bash tests/stages/00_environment.sh
```

环境测试确认 Python、CMake、C++ 编译器，以及 IREE 或独立 LLVM build-tree
中的 MLIR CMake package、`mlir-opt`、`mlir-translate` 和 `llc` 均可用。
