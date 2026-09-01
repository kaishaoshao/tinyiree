#include "tiny_iree/IR/Ops.h"

namespace mlir::tiree::Input {
namespace {

LogicalResult verifyF32Tensor(Operation *op, Type type, StringRef role) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isF32()) {
    return op->emitOpError() << role << " must be a ranked f32 tensor";
  }
  return success();
}

bool compatibleDimension(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

}  // namespace

LogicalResult MatMulOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getLhs().getType(), "lhs")) ||
      failed(verifyF32Tensor(getOperation(), getRhs().getType(), "rhs")) ||
      failed(verifyF32Tensor(getOperation(), getOutput().getType(), "output"))) {
    return failure();
  }
  auto lhs = cast<RankedTensorType>(getLhs().getType());
  auto rhs = cast<RankedTensorType>(getRhs().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || output.getRank() != 2 ||
      !compatibleDimension(lhs.getDimSize(1), rhs.getDimSize(0)) ||
      !compatibleDimension(output.getDimSize(0), lhs.getDimSize(0)) ||
      !compatibleDimension(output.getDimSize(1), rhs.getDimSize(1))) {
    return emitOpError("expects (MxK) x (KxN) -> (MxN)");
  }
  return success();
}

LogicalResult FusedMatMulAddReluOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getLhs().getType(), "lhs")) ||
      failed(verifyF32Tensor(getOperation(), getRhs().getType(), "rhs")) ||
      failed(verifyF32Tensor(getOperation(), getBias().getType(), "bias")) ||
      failed(verifyF32Tensor(getOperation(), getOutput().getType(), "output"))) {
    return failure();
  }
  auto lhs = cast<RankedTensorType>(getLhs().getType());
  auto rhs = cast<RankedTensorType>(getRhs().getType());
  auto bias = cast<RankedTensorType>(getBias().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || output.getRank() != 2 ||
      bias.getRank() != 1 ||
      !compatibleDimension(lhs.getDimSize(1), rhs.getDimSize(0)) ||
      !compatibleDimension(output.getDimSize(0), lhs.getDimSize(0)) ||
      !compatibleDimension(output.getDimSize(1), rhs.getDimSize(1)) ||
      !compatibleDimension(bias.getDimSize(0), output.getDimSize(1))) {
    return emitOpError("expects (MxK), (KxN), (N) -> (MxN)");
  }
  return success();
}

LogicalResult AddOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getLhs().getType(), "lhs")) ||
      failed(verifyF32Tensor(getOperation(), getRhs().getType(), "rhs")) ||
      failed(verifyF32Tensor(getOperation(), getOutput().getType(), "output"))) {
    return failure();
  }
  auto lhs = cast<RankedTensorType>(getLhs().getType());
  auto rhs = cast<RankedTensorType>(getRhs().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  bool sameShape = lhs == rhs;
  bool biasBroadcast = rhs.getRank() == 1 && lhs.getRank() >= 1 &&
                       compatibleDimension(rhs.getDimSize(0),
                                           lhs.getShape().back());
  if (output != lhs || (!sameShape && !biasBroadcast)) {
    return emitOpError(
        "only supports equal shapes or final-dimension bias broadcast");
  }
  return success();
}

LogicalResult ReluOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getInput().getType(), "input")) ||
      getInput().getType() != getOutput().getType()) {
    return emitOpError("expects identical ranked f32 input/output types");
  }
  return success();
}

LogicalResult SoftmaxOp::verify() {
  auto type = dyn_cast<RankedTensorType>(getInput().getType());
  if (!type || failed(verifyF32Tensor(getOperation(), type, "input")) ||
      getInput().getType() != getOutput().getType()) {
    return emitOpError("expects identical ranked f32 input/output types");
  }
  int64_t axis = getAxisAttr().getInt();
  int64_t normalizedAxis = axis < 0 ? axis + type.getRank() : axis;
  if (normalizedAxis != type.getRank() - 1) {
    return emitOpError("only supports softmax over the final dimension");
  }
  return success();
}

}  // namespace mlir::tiree::Input

#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyInputOps.cpp.inc"

