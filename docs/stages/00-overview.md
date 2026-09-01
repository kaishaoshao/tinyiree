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

运行 `bash tests/stages/00_environment.sh`，确认 Python、CMake、C++ 编译器和
相邻 IREE build-tree 中的 MLIR 工具存在。

