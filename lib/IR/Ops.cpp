#include "tiny_iree/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::tiree {

LogicalResult verifyResourceId(Operation *operation, IntegerAttr resourceId) {
  if (resourceId.getInt() < 0) {
    return operation->emitOpError("resource_id must be non-negative");
  }
  return success();
}

LogicalResult verifyAllocation(Operation *operation, IntegerAttr resourceId,
                               IntegerAttr bytes) {
  if (failed(verifyResourceId(operation, resourceId))) return failure();
  if (bytes.getInt() == 0 || bytes.getInt() < -1) {
    return operation->emitOpError(
        "allocation bytes must be positive or -1 for dynamic");
  }
  return success();
}

LogicalResult verifyDispatchResources(Operation *operation,
                                      ResultRange outputs,
                                      DenseI64ArrayAttr resultBytes,
                                      DenseI64ArrayAttr resultResources) {
  if (outputs.size() != resultBytes.size() ||
      outputs.size() != resultResources.size()) {
    return operation->emitOpError(
        "result_bytes and result_resources must match result count");
  }
  for (int64_t bytes : resultBytes.asArrayRef()) {
    if (bytes == 0 || bytes < -1) {
      return operation->emitOpError(
          "result bytes must be positive or -1 for dynamic");
    }
  }
  for (int64_t resourceId : resultResources.asArrayRef()) {
    if (resourceId < 0) {
      return operation->emitOpError(
          "result resource ids must be non-negative");
    }
  }
  return success();
}

}  // namespace mlir::tiree

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

namespace mlir::tiree::HAL {

LogicalResult ExecutableOp::verify() {
  auto functionType = dyn_cast<FunctionType>(getFunctionType());
  if (!functionType || functionType.getNumResults() == 0) {
    return emitOpError("requires a function type with at least one result");
  }
  if (getKernel().empty() || getWorkload().empty()) {
    return emitOpError("requires a kernel and non-empty workload");
  }
  return success();
}

LogicalResult AllocOp::verify() {
  return verifyAllocation(getOperation(), getResourceIdAttr(), getBytesAttr());
}

LogicalResult DispatchOp::verify() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  auto executable = module ? module.lookupSymbol<ExecutableOp>(getEntryPoint())
                           : ExecutableOp();
  if (!executable) {
    return emitOpError("references a missing executable entry point: ")
           << getEntryPoint();
  }
  if (getOutputBuffers().size() != getOutputs().size()) {
    return emitOpError("output buffer count must match result count");
  }
  auto functionType = cast<FunctionType>(executable.getFunctionType());
  if (!llvm::equal(functionType.getInputs(), getInputs().getTypes()) ||
      !llvm::equal(functionType.getResults(), getOutputs().getTypes())) {
    return emitOpError("dispatch interface does not match executable");
  }
  return verifyDispatchResources(getOperation(), getOutputs(),
                                 getResultBytesAttr(),
                                 getResultResourcesAttr());
}

LogicalResult DeallocOp::verify() {
  return verifyResourceId(getOperation(), getResourceIdAttr());
}

}  // namespace mlir::tiree::HAL

namespace mlir::tiree::Stream {

LogicalResult AllocOp::verify() {
  return verifyAllocation(getOperation(), getResourceIdAttr(), getBytesAttr());
}

LogicalResult DispatchOp::verify() {
  if (getEntryPoint().empty() || getWorkload().empty()) {
    return emitOpError("requires an entry point and non-empty workload");
  }
  if (getOutputResources().size() != getOutputs().size()) {
    return emitOpError("output resource count must match result count");
  }
  return verifyDispatchResources(getOperation(), getOutputs(),
                                 getResultBytesAttr(),
                                 getResultResourcesAttr());
}

LogicalResult DeallocOp::verify() {
  return verifyResourceId(getOperation(), getResourceIdAttr());
}

}  // namespace mlir::tiree::Stream

#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyFlowOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyHALOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyInputOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyStreamOps.cpp.inc"
