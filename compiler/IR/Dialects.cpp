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

namespace mlir::tiree::HAL {
void TinyHALDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiny_iree/IR/TinyHALTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiny_iree/IR/TinyHALOps.cpp.inc"
      >();
}
}  // namespace mlir::tiree::HAL

namespace mlir::tiree::Stream {
void TinyStreamDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiny_iree/IR/TinyStreamTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiny_iree/IR/TinyStreamOps.cpp.inc"
      >();
}
}  // namespace mlir::tiree::Stream

namespace mlir::tiree::VM {
void TinyVMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiny_iree/IR/TinyVMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiny_iree/IR/TinyVMOps.cpp.inc"
      >();
}
}  // namespace mlir::tiree::VM

namespace mlir::tiree {
void registerTinyIREEDialects(DialectRegistry &registry) {
  registry.insert<Input::TinyInputDialect, Flow::TinyFlowDialect,
                  HAL::TinyHALDialect,
                  Stream::TinyStreamDialect, VM::TinyVMDialect>();
}
}  // namespace mlir::tiree

#include "tiny_iree/IR/TinyFlowDialect.cpp.inc"
#include "tiny_iree/IR/TinyHALDialect.cpp.inc"
#include "tiny_iree/IR/TinyInputDialect.cpp.inc"
#include "tiny_iree/IR/TinyStreamDialect.cpp.inc"
#include "tiny_iree/IR/TinyVMDialect.cpp.inc"
