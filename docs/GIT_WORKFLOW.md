# 使用 Git 提交历史学习

## 查看阶段

```bash
git log --reverse --oneline --decorate
git tag --list 'stage-*' --sort=version:refname
git show --stat stage-03
git diff stage-02..stage-03
```

`git diff stage-02..stage-03` 是最推荐的阅读方式：它只展示 Stream 阶段新增的
Dialect、Pass、测试和文档，不会被最终项目的其他功能干扰。

## 检出并实验

```bash
git switch --detach stage-03
bash tests/stages/03_stream_resources.sh
git switch main
```

detached HEAD 很适合只读学习。如果要保存自己的实验，从对应阶段创建分支：

```bash
git switch -c study/stream stage-03
```

## 每次实验的提交规范

建议继续使用仓库现有格式：`feat(stage-NN): ...`、`test(stage-NN): ...`、
`docs(stage-NN): ...`。一次提交只改变一个概念，并同时补充正向测试和至少一个
失败路径测试。
