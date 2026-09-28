# IREE-style runtime component: owns the reusable runtime library.
add_library(TinyIREERuntime STATIC Runtime.cpp)
llvm_update_compile_flags(TinyIREERuntime)
target_link_libraries(TinyIREERuntime PUBLIC
  TinyIREECompiler
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
  ${CMAKE_DL_LIBS}
)
