#ifndef TINY_IREE_IR_OPS_H_
#define TINY_IREE_IR_OPS_H_

#include "tiny_iree/IR/Dialects.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyInputOps.h.inc"

#endif  // TINY_IREE_IR_OPS_H_
