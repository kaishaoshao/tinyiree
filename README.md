# Tiny-IREE Learning Repository

这是一个按 IREE 核心分层逐步实现的小型 AI 编译器教学仓库。重点不是复制
IREE 的全部功能，而是让模型前端、MLIR Dialect、优化、dispatch、资源规划、
HAL、VM、CPU codegen 和 runtime 都具有可观察的简化实现。

## 如何学习

本仓库刻意保留线性 Git 历史。每个提交都是一个学习阶段，并配套：

- `docs/stages/`：阶段目标、概念、源码入口和实验。
- `tests/stages/`：只验证本阶段职责的测试脚本。
- `tests/run_all.sh`：最终端到端回归入口。
- Git tag `stage-00` 到 `stage-09`：快速切换学习快照。

查看历史：

```bash
git log --reverse --oneline --decorate
git show stage-02
git switch --detach stage-02
```

返回最终版本：

```bash
git switch main
```

## 本机依赖

默认复用相邻 IREE 仓库已经构建的 LLVM/MLIR：

```text
/Volumes/wsk/code/llvm-mlir/iree/build_tools/build-host
```

先执行环境检查：

```bash
bash tests/stages/00_environment.sh
```

完整阶段路线见 [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md)。

