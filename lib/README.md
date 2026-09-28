# Compiler library distribution

This directory has the same *distribution* role as IREE's root `lib/`
directory. In upstream IREE it contains the Bazel packaging target for
`libIREECompiler.so`; compiler implementation itself lives under
`compiler/src/iree/compiler/API`.

Tiny-IREE is CMake-first, so the corresponding source lives in
`compiler/API/` and the public header is `include/tiny_iree/Compiler/API.h`.
Building `tiny-iree-aot-tools` produces the shared library here in the build
tree:

```text
build-<platform>/lib/libTinyIREECompiler.so      # Linux x86_64 / AArch64
build-<platform>/lib/libTinyIREECompiler.dylib   # macOS, including Apple M4
```

The current C ABI deliberately exposes parsing and verification only. It is
versioned, uses caller-owned diagnostic storage, and can be expanded without
making Python teaching scripts part of a binary compatibility promise.
