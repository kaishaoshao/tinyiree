#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"

namespace mlir::tiree::Flow {
void TinyFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny_iree/IR/TinyFlowOps.cpp.inc"
      >();
}
}  // namespace mlir::tiree::Flow

namespace mlir::tiree::Input {
void TinyInputDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny_iree/IR/TinyInputOps.cpp.inc"
      >();
}
}  // namespace mlir::tiree::Input

namespace mlir::tiree {
void registerTinyIREEDialects(DialectRegistry &registry) {
  registry.insert<Input::TinyInputDialect, Flow::TinyFlowDialect>();
}
}  // namespace mlir::tiree

#include "tiny_iree/IR/TinyFlowDialect.cpp.inc"
#include "tiny_iree/IR/TinyInputDialect.cpp.inc"
