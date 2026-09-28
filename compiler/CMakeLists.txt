# IREE-style compiler component: owns the reusable compiler library.
add_mlir_library(TinyIREECompiler
  IR/Dialects.cpp
  IR/HALTypes.cpp
  IR/Ops.cpp
  IR/StreamTypes.cpp
  IR/VMTypes.cpp
  Transforms/Passes.cpp
  DEPENDS TinyFlowOpsIncGen TinyHALOpsIncGen TinyInputOpsIncGen
          TinyStreamOpsIncGen TinyVMOpsIncGen
  LINK_LIBS PUBLIC
    MLIRBytecodeOpInterface
    MLIRIR
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTransformUtils
)

# Distribution-facing C ABI, analogous to IREE's libIREECompiler packaging
# target. The implementation source belongs to compiler/API; lib/ is reserved
# for the resulting shared library and its distribution documentation.
add_library(TinyIREECompilerAPI SHARED API/CompilerAPI.cpp)
set_target_properties(TinyIREECompilerAPI PROPERTIES
  OUTPUT_NAME TinyIREECompiler
  VERSION 1.0.0
  SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  CXX_VISIBILITY_PRESET default
)
llvm_update_compile_flags(TinyIREECompilerAPI)
target_link_libraries(TinyIREECompilerAPI PRIVATE
  TinyIREECompiler
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
)
