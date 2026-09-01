#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"

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
  registry.insert<Input::TinyInputDialect>();
}
}  // namespace mlir::tiree

#include "tiny_iree/IR/TinyInputDialect.cpp.inc"

