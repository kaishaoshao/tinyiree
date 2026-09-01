cmake_minimum_required(VERSION 3.20)
project(tiny_iree LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS include/tiny_iree/IR/TinyInputOps.td)
mlir_tablegen(include/tiny_iree/IR/TinyInputDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/tiny_iree/IR/TinyInputDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(include/tiny_iree/IR/TinyInputOps.h.inc -gen-op-decls)
mlir_tablegen(include/tiny_iree/IR/TinyInputOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TinyInputOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/tiny_iree/IR/TinyFlowOps.td)
mlir_tablegen(include/tiny_iree/IR/TinyFlowDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/tiny_iree/IR/TinyFlowDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(include/tiny_iree/IR/TinyFlowOps.h.inc -gen-op-decls)
mlir_tablegen(include/tiny_iree/IR/TinyFlowOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TinyFlowOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/tiny_iree/IR/TinyStreamOps.td)
mlir_tablegen(include/tiny_iree/IR/TinyStreamDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/tiny_iree/IR/TinyStreamDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(include/tiny_iree/IR/TinyStreamOps.h.inc -gen-op-decls)
mlir_tablegen(include/tiny_iree/IR/TinyStreamOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/tiny_iree/IR/TinyStreamTypes.h.inc
  -gen-typedef-decls -typedefs-dialect=tiree_stream)
mlir_tablegen(include/tiny_iree/IR/TinyStreamTypes.cpp.inc
  -gen-typedef-defs -typedefs-dialect=tiree_stream)
add_public_tablegen_target(TinyStreamOpsIncGen)

add_mlir_library(TinyIREECompiler
  lib/IR/Dialects.cpp
  lib/IR/Ops.cpp
  lib/IR/StreamTypes.cpp
  lib/Transforms/Passes.cpp
  DEPENDS TinyFlowOpsIncGen TinyInputOpsIncGen TinyStreamOpsIncGen
  LINK_LIBS PUBLIC
    MLIRBytecodeOpInterface
    MLIRIR
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTransformUtils
)

add_llvm_executable(tiny-iree-opt tools/tiny-iree-opt.cpp)
set_target_properties(tiny-iree-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
llvm_update_compile_flags(tiny-iree-opt)
target_link_libraries(tiny-iree-opt PRIVATE
  TinyIREECompiler
  MLIRArithDialect
  MLIRFuncDialect
  MLIROptLib
)
mlir_check_all_link_libraries(tiny-iree-opt)
