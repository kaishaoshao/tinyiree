# IREE-style command-line entry points. Shared implementation stays in the
# compiler and runtime libraries; these targets only assemble CLI behavior.
function(tiny_iree_configure_tool target)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  llvm_update_compile_flags(${target})
endfunction()

add_llvm_executable(tiny-iree-opt PARTIAL_SOURCES_INTENDED tiny-iree-opt.cpp)
tiny_iree_configure_tool(tiny-iree-opt)
target_link_libraries(tiny-iree-opt PRIVATE
  TinyIREECompiler
  MLIRArithDialect
  MLIRFuncDialect
  MLIROptLib
)
mlir_check_all_link_libraries(tiny-iree-opt)

add_llvm_executable(tiny-iree-translate PARTIAL_SOURCES_INTENDED
  tiny-iree-translate.cpp)
tiny_iree_configure_tool(tiny-iree-translate)
target_include_directories(tiny-iree-translate BEFORE PRIVATE
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(tiny-iree-translate PRIVATE
  TinyIREECompiler
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
)
mlir_check_all_link_libraries(tiny-iree-translate)

add_llvm_executable(tiny-iree-export-codegen PARTIAL_SOURCES_INTENDED
  tiny-iree-export-codegen.cpp)
tiny_iree_configure_tool(tiny-iree-export-codegen)
target_link_libraries(tiny-iree-export-codegen PRIVATE
  TinyIREECompiler
  MLIRArithDialect
  MLIRFuncDialect
  MLIRParser
)
mlir_check_all_link_libraries(tiny-iree-export-codegen)

add_llvm_executable(tiny-iree-run-module PARTIAL_SOURCES_INTENDED
  tiny-iree-run-module.cpp)
tiny_iree_configure_tool(tiny-iree-run-module)
target_link_libraries(tiny-iree-run-module PRIVATE TinyIREERuntime)
mlir_check_all_link_libraries(tiny-iree-run-module)
