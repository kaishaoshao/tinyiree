# Command-line tools

This directory has the same top-level role as IREE's `tools/`: it owns command
entry points, while reusable implementation belongs to `compiler/` or
`runtime/`.

- The C++ files are thin front ends for `tiny-iree-opt`, `tiny-iree-translate`,
  `tiny-iree-export-codegen`, and `tiny-iree-run-module`.
- The Python scripts are the teaching project's directly invokable model
  frontend, compilation orchestrator, kernel generator, and model fixtures.
  They intentionally remain command-oriented scripts rather than a second C++
  compiler library.

The stable user-facing commands stay under `tools/`. If the Python frontend is
later promoted into a reusable package, its implementation should move under
`compiler/`, leaving a compatibility wrapper here.
