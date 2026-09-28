# 常见失败诊断手册

定位原则是从最早的失败边界开始，不要同时修改 ONNX、pass、codegen 和 runtime。
先保存完整命令、stdout/stderr、`git rev-parse HEAD` 和 bundle manifest，再按下列
顺序缩小范围。

## 1. 最小诊断信息

```bash
git status --short
git rev-parse HEAD
bash scripts/platform.sh
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
printf 'build=%s\n' "$BUILD_DIR"
cmake -LA -N "$BUILD_DIR" | grep CMAKE_BUILD_TYPE || true
"$BUILD_DIR/bin/tiny-iree-run-module" --version || true
```

bundle 问题还要保存：

```bash
python3 -m json.tool bundle/manifest.json
od -Ax -tx1 -N 16 bundle/module.tvm
file bundle/libtiny_iree_kernels.*
```

不要在原 bundle 上试验破坏性修改；先复制到 `mktemp -d`。

## 2. 快速分层

| 症状 | 首先检查 | 所属层 |
| --- | --- | --- |
| CMake 找不到 MLIR | `MLIR_DIR`、工具链 build-tree | 环境/构建 |
| `unsupported ONNX op` | `--analyze`、支持清单 | 前端 |
| MLIR verifier 报错 | 最早失败的阶段 IR | dialect/pass |
| bundle target mismatch | manifest target、`scripts/platform.sh` | 打包/平台 |
| `unsupported Tiny VM bytecode header` | magic、version、是否拿错文件 | VM |
| `unable to load native executable` | 文件、架构、依赖库 | loader |
| `native kernel failed`/找不到符号 | `nm`、callee 和 ABI arity | Native ABI |
| input count/element count mismatch | VM 函数签名和 `--input` | runtime 输入 |
| 数值错误但能运行 | reference、首个错误阶段、codegen 模式 | lowering/kernel |
| 性能波动 | 测量边界、线程、调频、温度 | benchmark |

## 3. 构建和工具链

### 找不到 `MLIRConfig.cmake`

```bash
export LLVM_BUILD_DIR=../llvm-project/build
# 或
export IREE_BUILD_DIR=../iree/build_tools/build-host
bash scripts/build.sh
```

路径可以在本机 shell 中配置，不要提交个人绝对路径。更详细的头文件版本、缓存和
Apple libomp 问题见 [BUILD_TROUBLESHOOTING.md](BUILD_TROUBLESHOOTING.md)。

### 构建目录来自另一台机器

症状通常是 CMake 记录旧源码路径或目标架构。使用平台隔离目录：

```bash
export TINY_IREE_BUILD_DIR="$PWD/build-$(bash scripts/platform.sh | \
  sed -n 's/^host_id=//p')"
bash scripts/build.sh
```

不要用 `git reset --hard` 或删除源码解决 CMake cache 问题；build 目录不是源码。

## 4. ONNX 前端

### auto 模式不认识 `.pb`

这是预期行为。显式选择：

```bash
.venv/bin/python tools/tiny_iree_compile.py model.pb \
  --input-type=onnx -o model.tiree
```

### `unsupported ONNX op domain::Op`

```bash
.venv/bin/python tools/import_onnx.py --analyze model.onnx
.venv/bin/python tools/import_onnx.py --list-supported-ops
```

先看 domain 是否为标准 `ai.onnx`，再看 op 名。显示
`op-type-supported=yes` 只代表候选身份已知，attribute/shape/pattern 仍可能在正式
导入时失败。完整限制见 [MODEL_FRONTEND.md](MODEL_FRONTEND.md)。

### shape inference/checker 失败

先用原始 ONNX 工具验证，不要跳过 checker：

```bash
.venv/bin/python - model.onnx <<'PY'
import sys, onnx
model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
print("onnx checker: ok")
PY
```

常见原因是 output shape 缺失、opset/attribute 不一致、initializer type 不受支持或
自定义 domain 没有对应 importer。

## 5. Pass 与阶段 IR

完整编译 bundle 会保留 `module.input.mlir`、`module.global-opt.mlir`、
`module.flow.mlir`、`module.stream.mlir`、`module.hal.mlir` 和 `module.vm.mlir`。
从前往后找第一个错误文件，而不是只看最终 LLVM 报错。

```bash
source scripts/platform.sh
BUILD_DIR="$(tiny_iree_build_dir "$PWD")"
"$BUILD_DIR/bin/tiny-iree-opt" bundle/module.input.mlir \
  --tiree-global-optimize -o /tmp/check.mlir
```

典型边界：Input verifier 检查算子 shape；Flow 检查 dispatch region；Stream 检查
resource id/lifetime；HAL 检查 executable 元数据；VM serializer 只接受 constant、
alloc、call、dealloc 和 return。

## 6. Bundle 与目标平台

### `bundle target ... does not match runtime target ...`

比较：

```bash
bash scripts/platform.sh
python3 -m json.tool bundle/manifest.json
file bundle/libtiny_iree_kernels.*
```

M4 bundle 不能在 x86 runtime 运行，反之亦然。应在目标机重新编译模型，不要手工
改 manifest 的 target；改字符串不能改变动态库里的机器码。

### manifest 缺失、JSON 无效或 unsafe path

bundle 必须是目录，包含 `manifest.json`、manifest 指向的 `module.tvm` 和动态库。
artifact 名必须是 bundle 内相对路径，不能是绝对路径或包含 `..`。

## 7. VM 字节码

### header/version 错误

```bash
od -Ax -tx1 -N 16 bundle/module.tvm
```

期望 magic 为 `54 49 52 45 56 4d 31 00`，随后版本
`01 00 00 00`。若看到 `ML\xefR` 等 MLIR bytecode magic，说明误把
`module.vm.mlirbc` 当成 `module.tvm`。两者不是同一种格式。

### undefined value、invalid resource、trailing data

这些错误表示 VM 控制程序损坏或 serializer/runtime 版本不匹配。对照
`module.vm.mlir`，重新运行 `tiny-iree-translate`；不要用十六进制编辑器“修复”
resource id。格式细节见 [VM_BYTECODE_AND_NATIVE_ABI.md](VM_BYTECODE_AND_NATIVE_ABI.md)。

## 8. 动态库与 Native ABI

### 动态库不能加载

Linux：

```bash
file bundle/libtiny_iree_kernels.so
ldd bundle/libtiny_iree_kernels.so
nm -D --defined-only bundle/libtiny_iree_kernels.so | grep tiree_kernel
```

macOS：

```bash
file bundle/libtiny_iree_kernels.dylib
otool -L bundle/libtiny_iree_kernels.dylib
nm -gU bundle/libtiny_iree_kernels.dylib | grep tiree_kernel
```

`wrong ELF class`、`bad CPU type` 通常是架构错误；`not found` 常是 OpenMP 或其他
动态依赖；符号为空则检查 codegen plan、`llvm.emit_c_interface` 和链接可见性。

### symbol/arity 错误

VM callee `foo` 优先要求 `_mlir_ciface_tiree_kernel_foo`。当前 MLIR C interface
只支持输入加输出共 2 到 5 个 descriptor；generic fallback 只支持一个输出。改变
函数参数后必须同步 VM call metadata、生成的 wrapper 和 runtime ABI。

## 9. Runtime 输入

- 多个 tensor 用 `;` 分隔。
- 静态 shape 可只给逗号分隔数值。
- 动态 shape 使用 `2x3=1,2,3,4,5,6`。

示例：

```bash
tiny-iree-run-module bundle --function=predict \
  '--input=2x3=1,2,3,4,5,6;3=1,1,1'
```

在 shell 中必须引用包含 `;` 的参数，否则分号会被解释为命令分隔符。element count
必须等于解析后的 shape 元素数。

## 10. 数值错误

1. 用 `tools/run_onnx_reference.py` 得到相同输入的 ONNX 基线。
2. 确认 runtime 输出 `native-aot`，不要把解释器和 native 路径混在一起。
3. 比较 scalar 与 vector；只有 vector 错通常指向 tiling/vector lowering。
4. 找第一个产生错误的 dispatch，不要只观察 Softmax 后的最终概率。
5. 使用绝对误差和相对误差，并单独处理接近 0、NaN、Inf。

不要通过放宽容差掩盖 shape、layout 或 ABI 错误。

## 11. 性能异常

使用 `--benchmark-warmup` 和 `--benchmark-repetitions`，不要把一次 shell wall-clock
当稳态 kernel 时间。若结果不稳定，依次检查后台负载、电源/温度、CPU 调频、
OpenMP 线程数、输入规模、构建类型和 bundle manifest。详细方法见
[PERFORMANCE_BENCHMARKING.md](PERFORMANCE_BENCHMARKING.md)。

## 12. 最小复现模板

报告问题时提供：平台、commit、构建类型、完整命令、最小模型/MLIR、输入、完整
stderr、manifest、最早失败的阶段 IR，以及是否能由 `tests/run_all.sh` 复现。不要
只提供最终一句错误，也不要附带含个人绝对路径的整份构建缓存。
