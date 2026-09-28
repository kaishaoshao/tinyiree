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
