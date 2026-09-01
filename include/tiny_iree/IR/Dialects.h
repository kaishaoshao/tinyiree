#ifndef TINY_IREE_IR_DIALECTS_H_
#define TINY_IREE_IR_DIALECTS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "tiny_iree/IR/TinyInputDialect.h.inc"

namespace mlir::tiree {
void registerTinyIREEDialects(DialectRegistry &registry);
}  // namespace mlir::tiree

#endif  // TINY_IREE_IR_DIALECTS_H_

