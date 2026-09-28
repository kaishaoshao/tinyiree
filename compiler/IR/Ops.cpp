#include "tiny_iree/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::tiree {

static LogicalResult verifyResourceId(Operation *operation,
                                      IntegerAttr resourceId) {
  if (resourceId.getInt() < 0) {
    return operation->emitOpError("resource_id must be non-negative");
  }
  return success();
}

static LogicalResult verifyAllocation(Operation *operation,
                                      IntegerAttr resourceId,
                                      IntegerAttr bytes) {
  if (failed(verifyResourceId(operation, resourceId))) return failure();
  if (bytes.getInt() == 0 || bytes.getInt() < -1) {
    return operation->emitOpError(
        "allocation bytes must be positive or -1 for dynamic");
  }
  return success();
}

static LogicalResult verifyDispatchResources(
    Operation *operation, ResultRange outputs, DenseI64ArrayAttr resultBytes,
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
      return operation->emitOpError("result resource ids must be non-negative");
    }
  }
  return success();
}

}  // namespace mlir::tiree

namespace mlir::tiree::Input {

static LogicalResult verifyF32Tensor(Operation *op, Type type,
                                     StringRef role) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isF32()) {
    return op->emitOpError() << role << " must be a ranked f32 tensor";
  }
  return success();
}

static bool compatibleDimension(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

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
                       rhs.getDimSize(0) == lhs.getShape().back();
  if (output != lhs || (!sameShape && !biasBroadcast)) {
    return emitOpError("only supports equal shapes or final-dimension bias broadcast");
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

LogicalResult FakeQuantOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getInput().getType(), "input")) ||
      failed(verifyF32Tensor(getOperation(), getScale().getType(), "scale")) ||
      failed(verifyF32Tensor(getOperation(), getZeroPoint().getType(),
                             "zero_point")) ||
      getInput().getType() != getOutput().getType()) {
    return emitOpError(
        "expects identical ranked f32 input/output and f32 quantization parameters");
  }
  auto isScalarParameter = [](Type type) {
    auto tensorType = cast<RankedTensorType>(type);
    return tensorType.getRank() == 0 ||
           (tensorType.getRank() == 1 && tensorType.hasStaticShape() &&
            tensorType.getNumElements() == 1);
  };
  if (!isScalarParameter(getScale().getType()) ||
      !isScalarParameter(getZeroPoint().getType())) {
    return emitOpError("only supports scalar per-tensor scale and zero point");
  }
  return success();
}

LogicalResult SplitOp::verify() {
  auto input = dyn_cast<RankedTensorType>(getInput().getType());
  if (!input || failed(verifyF32Tensor(getOperation(), input, "input")) ||
      getOutputs().size() != 2) {
    return emitOpError("expects one ranked f32 input and exactly two outputs");
  }
  int64_t axis = getAxisAttr().getInt();
  int64_t normalizedAxis = axis < 0 ? axis + input.getRank() : axis;
  if (input.getRank() == 0 || normalizedAxis != input.getRank() - 1) {
    return emitOpError("only supports splitting the final dimension");
  }
  for (Value outputValue : getOutputs()) {
    auto output = dyn_cast<RankedTensorType>(outputValue.getType());
    if (!output || !output.getElementType().isF32() ||
        output.getRank() != input.getRank()) {
      return emitOpError("outputs must be ranked f32 tensors matching input rank");
    }
    for (int64_t dimension = 0; dimension < input.getRank() - 1; ++dimension) {
      if (!compatibleDimension(input.getDimSize(dimension),
                               output.getDimSize(dimension))) {
        return emitOpError("non-split output dimensions must match input");
      }
    }
    int64_t inputWidth = input.getShape().back();
    int64_t outputWidth = output.getShape().back();
    if (!ShapedType::isDynamic(inputWidth) &&
        !ShapedType::isDynamic(outputWidth) && inputWidth != outputWidth * 2) {
      return emitOpError("outputs must evenly halve the final dimension");
    }
  }
  if (getOutputs()[0].getType() != getOutputs()[1].getType()) {
    return emitOpError("two output types must be identical");
  }
  return success();
}

LogicalResult Conv2DOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getInput().getType(), "input")) ||
      failed(verifyF32Tensor(getOperation(), getWeight().getType(), "weight")) ||
      failed(verifyF32Tensor(getOperation(), getBias().getType(), "bias")) ||
      failed(verifyF32Tensor(getOperation(), getOutput().getType(), "output"))) {
    return failure();
  }
  auto input = cast<RankedTensorType>(getInput().getType());
  auto weight = cast<RankedTensorType>(getWeight().getType());
  auto bias = cast<RankedTensorType>(getBias().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> pads = getPads();
  if (input.getRank() != 4 || weight.getRank() != 4 || bias.getRank() != 1 ||
      output.getRank() != 4 || strides.size() != 2 || strides[0] != 1 ||
      strides[1] != 1 || pads.size() != 4 ||
      llvm::any_of(pads, [](int64_t value) { return value != 0; }) ||
      input.getDimSize(1) != weight.getDimSize(1) ||
      output.getDimSize(0) != input.getDimSize(0) ||
      output.getDimSize(1) != weight.getDimSize(0) ||
      bias.getDimSize(0) != weight.getDimSize(0) ||
      output.getDimSize(2) != input.getDimSize(2) - weight.getDimSize(2) + 1 ||
      output.getDimSize(3) != input.getDimSize(3) - weight.getDimSize(3) + 1) {
    return emitOpError("expects static NCHW valid stride-one convolution");
  }
  return success();
}

LogicalResult MaxPool2DOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getInput().getType(), "input")) ||
      failed(verifyF32Tensor(getOperation(), getOutput().getType(), "output"))) {
    return failure();
  }
  auto input = cast<RankedTensorType>(getInput().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  ArrayRef<int64_t> kernelShape = getKernelShape();
  ArrayRef<int64_t> strides = getStrides();
  if (input.getRank() != 4 || output.getRank() != 4 ||
      kernelShape.size() != 2 || kernelShape[0] != 2 || kernelShape[1] != 2 ||
      strides.size() != 2 || strides[0] != 2 || strides[1] != 2 ||
      output.getDimSize(0) != input.getDimSize(0) ||
      output.getDimSize(1) != input.getDimSize(1) ||
      output.getDimSize(2) != input.getDimSize(2) / 2 ||
      output.getDimSize(3) != input.getDimSize(3) / 2) {
    return emitOpError("expects static NCHW 2x2 stride-two pooling");
  }
  return success();
}

LogicalResult ReshapeOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getInput().getType(), "input")) ||
      failed(verifyF32Tensor(getOperation(), getOutput().getType(), "output"))) {
    return failure();
  }
  auto input = cast<RankedTensorType>(getInput().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  if (!input.hasStaticShape() || !output.hasStaticShape() ||
      input.getNumElements() != output.getNumElements()) {
    return emitOpError("requires static shapes with equal element counts");
  }
  return success();
}

LogicalResult TransposeOp::verify() {
  if (failed(verifyF32Tensor(getOperation(), getInput().getType(), "input")) ||
      failed(verifyF32Tensor(getOperation(), getOutput().getType(), "output"))) {
    return failure();
  }
  auto input = cast<RankedTensorType>(getInput().getType());
  auto output = cast<RankedTensorType>(getOutput().getType());
  ArrayRef<int64_t> permutation = getPermutation();
  if (input.getRank() != 4 || output.getRank() != 4 ||
      permutation.size() != 4 || permutation[0] != 0 || permutation[1] != 2 ||
      permutation[2] != 3 || permutation[3] != 1) {
    return emitOpError("only supports static NCHW to NHWC transpose");
  }
  for (int64_t index = 0; index < 4; ++index) {
    if (output.getDimSize(index) != input.getDimSize(permutation[index])) {
      return emitOpError("output shape does not match permutation");
    }
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
    return emitOpError("workload block argument count must match dispatch inputs");
  }
  for (auto [argument, input] : llvm::zip_equal(block.getArguments(), getInputs())) {
    if (argument.getType() != input.getType()) {
      return emitOpError("workload block argument types must match dispatch inputs");
    }
  }
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield) return emitOpError("workload block must terminate with tiree_flow.yield");
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
  } else if (kernel == "fake_quant") {
    expectedPayload = {Input::FakeQuantOp::getOperationName()};
  } else if (kernel == "split") {
    expectedPayload = {Input::SplitOp::getOperationName()};
  } else if (kernel == "conv2d") {
    expectedPayload = {Input::Conv2DOp::getOperationName()};
  } else if (kernel == "max_pool2d") {
    expectedPayload = {Input::MaxPool2DOp::getOperationName()};
  } else if (kernel == "reshape") {
    expectedPayload = {Input::ReshapeOp::getOperationName()};
  } else if (kernel == "transpose") {
    expectedPayload = {Input::TransposeOp::getOperationName()};
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
    return emitOpError("workload must yield the final payload operation results");
  }
  return success();
}

}  // namespace mlir::tiree::Flow

namespace mlir::tiree::Stream {

LogicalResult AllocOp::verify() {
  return verifyAllocation(getOperation(), getResourceIdAttr(), getBytesAttr());
}

LogicalResult DispatchOp::verify() {
  if (getEntryPoint().empty() || getWorkload().empty()) {
    return emitOpError("requires an entry point and non-empty workload");
  }
  for (Attribute attribute : getWorkload()) {
    if (!isa<StringAttr>(attribute)) {
      return emitOpError("workload entries must be operation name strings");
    }
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

namespace mlir::tiree::HAL {

LogicalResult ExecutableOp::verify() {
  auto functionType = dyn_cast<FunctionType>(getFunctionType());
  if (!functionType || functionType.getNumResults() == 0) {
    return emitOpError("requires a function type with at least one result");
  }
  if (getKernel().empty() || getWorkload().empty()) {
    return emitOpError("requires a kernel and non-empty workload");
  }
  for (Attribute attribute : getWorkload()) {
    if (!isa<StringAttr>(attribute)) {
      return emitOpError("workload entries must be operation name strings");
    }
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

namespace mlir::tiree::VM {

LogicalResult AllocOp::verify() {
  return verifyAllocation(getOperation(), getResourceIdAttr(), getBytesAttr());
}

LogicalResult CallOp::verify() {
  if (getOutputRefs().size() != getOutputs().size()) {
    return emitOpError("output ref count must match result count");
  }
  return verifyDispatchResources(getOperation(), getOutputs(),
                                 getResultBytesAttr(),
                                 getResultResourcesAttr());
}

LogicalResult DeallocOp::verify() {
  return verifyResourceId(getOperation(), getResourceIdAttr());
}

}  // namespace mlir::tiree::VM

#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyFlowOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyHALOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyInputOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyStreamOps.cpp.inc"
#define GET_OP_CLASSES
#include "tiny_iree/IR/TinyVMOps.cpp.inc"
