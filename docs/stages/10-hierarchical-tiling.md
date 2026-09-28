# 阶段 10：CPU 多级并行 Tiling

## 本阶段能力

`--cpu-codegen=hierarchical` 为静态二维 MatMul 生成三级循环结构：

- L2 tile 默认是 `64x64x32`，负责工作集和外层缓存 blocking。
- L1 tile 默认是 `16x16x8`，负责缩小内层工作集。
- register tile 默认是 `4x4`，用 `vector<4xf32>` 保存局部累加值。
- M/N 的 L2 tile 由 `omp.parallel` 和 `omp.wsloop` 表示，并 lower 为 OpenMP
  runtime 调用。Tiny-IREE 直接产生显式 CPU 并行 IR，省略 IREE 中更完整的
  workgroup 分发选择过程。

在现代 CPU 上 L1/L2 是硬件管理的 cache，所以这里的 L2/L1 指 cache blocking，
而不是伪造两个可以直接寻址的内存空间。最终 LLVM IR 中的
`__kmpc_fork_call` 证明工作确实进入 OpenMP runtime，而不只是把 Linalg iterator
标成 `parallel`。

## 源码入口

- `tools/generate_kernel_mlir.py`：`_hierarchical_matmul` 生成三级循环和 vector tile。
- `tools/tiny_iree_compile.py`：OpenMP/Vector/LLVM lowering 与平台链接参数。
- `tools/generate_tiled_onnx.py`：静态二维 identity MatMul 测试模型。
- `tests/run_hierarchical_tiling_e2e.sh`：检查 IR 结构、OpenMP 调用和数值结果。

当前限制必须明确：hierarchical 模式只接受静态 rank-2 的裸 MatMul；每层外 tile
必须能被内 tile 整除，M/N 还必须能被 register tile 整除。它尚未覆盖动态 shape，
也不支持 `MatMul+Add+Relu` fused epilogue。因此本阶段使用 identity MatMul 单独
验证循环映射，不能把它当作完整 MLP 的默认 codegen。

## 观察与回归

```bash
bash tests/stages/10_hierarchical_tiling.sh

source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tiny-iree-stage10.XXXXXX")"
.venv/bin/python tools/generate_tiled_onnx.py -o "$WORK_DIR/model.onnx" --size=8
.venv/bin/python tools/tiny_iree_compile.py "$WORK_DIR/model.onnx" \
  -o "$WORK_DIR/model.tiree" --opt "$BUILD_DIR/bin/tiny-iree-opt" \
  --cpu-codegen=hierarchical --parallel-threads=4 \
  --l2-tile=4,4,8 --l1-tile=4,4,4 --register-tile=2,2
printf 'artifacts: %s\n' "$WORK_DIR/model.tiree"
```

重点比较 bundle 内的 `module.executable.mlir`、`module.executable.llvm.mlir` 和
`module.executable.ll`，观察 `omp.parallel -> __kmpc_fork_call`，并检查 manifest
中的 `parallel_threads`、`l2_tile`、`l1_tile` 和 `register_tile`。
