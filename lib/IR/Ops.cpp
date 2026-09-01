#include "tiny_iree/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"

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

namespace mlir::tiree::Flow {

LogicalResult DispatchOp::verify() {
  Region &body = getBody();
  if (!body.hasOneBlock()) {
    return emitOpError("requires exactly one workload block");
  }
  Block &block = body.front();
  if (block.getNumArguments() != getInputs().size()) {
    return emitOpError(
        "workload block argument count must match dispatch inputs");
  }
  for (auto [argument, input] :
       llvm::zip_equal(block.getArguments(), getInputs())) {
    if (argument.getType() != input.getType()) {
      return emitOpError(
          "workload block argument types must match dispatch inputs");
    }
  }
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield) {
    return emitOpError(
        "workload block must terminate with tiree_flow.yield");
  }
  if (yield.getValues().size() != getOutputs().size()) {
    return emitOpError("yield operand count must match dispatch results");
  }
  for (auto [value, output] : llvm::zip_equal(yield.getValues(), getOutputs())) {
    if (value.getType() != output.getType()) {
      return emitOpError("yield operand types must match dispatch results");
    }
  }
  SmallVector<StringRef> expectedPayload;
  StringRef kernel = getKernel();
  if (kernel == "matmul_add_relu") {
    expectedPayload = {Input::FusedMatMulAddReluOp::getOperationName()};
  } else if (kernel == "matmul") {
    expectedPayload = {Input::MatMulOp::getOperationName()};
  } else if (kernel == "add") {
    expectedPayload = {Input::AddOp::getOperationName()};
  } else if (kernel == "relu") {
    expectedPayload = {Input::ReluOp::getOperationName()};
  } else if (kernel == "softmax") {
    expectedPayload = {Input::SoftmaxOp::getOperationName()};
  } else {
    return emitOpError("references an unsupported kernel: ") << kernel;
  }
  SmallVector<Operation *> payload;
  for (Operation &operation : block.without_terminator()) {
    payload.push_back(&operation);
  }
  if (payload.size() != expectedPayload.size()) {
    return emitOpError("kernel and workload payload operation count differ");
  }
  for (auto [operation, expectedName] :
       llvm::zip_equal(payload, expectedPayload)) {
    if (operation->getName().getStringRef() != expectedName) {
      return emitOpError("kernel and workload payload operation order differ");
    }
  }
  if (payload.empty() || payload.back()->getResults() != yield.getValues()) {
    return emitOpError(
        "workload must yield the final payload operation results");
  }
  return success();
}

}  // namespace mlir::tiree::Flow

#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyFlowOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyInputOps.cpp.inc"
