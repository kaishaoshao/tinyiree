#include "tiny_iree/Transforms/Passes.h"

#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tiree {
namespace {

static Operation *createOperation(OpBuilder &builder, Location location,
                                  StringRef name, TypeRange resultTypes,
                                  ValueRange operands,
                                  ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(attributes);
  return builder.create(state);
}

static std::pair<Operation *, Block *> createFlowDispatch(
    PatternRewriter &rewriter, Location location, TypeRange resultTypes,
    ValueRange inputs, StringRef kernel) {
  OperationState state(location, Flow::DispatchOp::getOperationName());
  state.addOperands(inputs);
  state.addTypes(resultTypes);
  state.addAttribute("kernel", rewriter.getStringAttr(kernel));
  state.addRegion();
  Operation *dispatch = rewriter.create(state);
  Region &body = dispatch->getRegion(0);
  body.push_back(new Block());
  Block &block = body.front();
  for (Value input : inputs) block.addArgument(input.getType(), location);
  return {dispatch, &block};
}

static void createFlowYield(PatternRewriter &rewriter, Location location,
                            ValueRange values) {
  OperationState state(location, Flow::YieldOp::getOperationName());
  state.addOperands(values);
  rewriter.create(state);
}

static SmallVector<int64_t> calculateResultBytes(TypeRange types) {
  SmallVector<int64_t> result;
  for (Type type : types) {
    auto tensorType = cast<RankedTensorType>(type);
    int64_t elementCount = 1;
    if (!tensorType.hasStaticShape()) {
      result.push_back(-1);
    } else {
      for (int64_t dimension : tensorType.getShape()) elementCount *= dimension;
      result.push_back(elementCount * tensorType.getElementTypeBitWidth() / 8);
    }
  }
  return result;
}

class FuseMatMulAddReluPattern final : public RewritePattern {
 public:
  explicit FuseMatMulAddReluPattern(MLIRContext *context)
      : RewritePattern(Input::MatMulOp::getOperationName(), 10, context) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {
    if (operation->getParentOfType<Flow::DispatchOp>()) return failure();
    auto matmul = cast<Input::MatMulOp>(operation);
    if (!matmul.getOutput().hasOneUse()) return failure();
    auto add = dyn_cast<Input::AddOp>(*matmul.getOutput().getUsers().begin());
    if (!add || add.getLhs() != matmul.getOutput() ||
        !add.getOutput().hasOneUse()) {
      return failure();
    }
    auto relu = dyn_cast<Input::ReluOp>(*add.getOutput().getUsers().begin());
    if (!relu || relu.getInput() != add.getOutput()) return failure();

    SmallVector<NamedAttribute> attributes;
    Operation *fused = createOperation(
        rewriter, matmul.getLoc(),
        Input::FusedMatMulAddReluOp::getOperationName(),
        relu.getOutput().getType(),
        {matmul.getLhs(), matmul.getRhs(), add.getRhs()}, attributes);
    rewriter.replaceOp(relu, fused->getResults());
    rewriter.eraseOp(add);
    rewriter.eraseOp(matmul);
    return success();
  }
};

class InputOpToFlowPattern final : public ConversionPattern {
 public:
  InputOpToFlowPattern(StringRef sourceName, StringRef kernel,
                       MLIRContext *context)
      : ConversionPattern(sourceName, 1, context), kernel(kernel) {}

  LogicalResult matchAndRewrite(
      Operation *operation, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto [dispatch, block] = createFlowDispatch(
        rewriter, operation->getLoc(), operation->getResultTypes(), operands,
        kernel);
    IRMapping mapping;
    for (auto [operand, argument] :
         llvm::zip_equal(operation->getOperands(), block->getArguments())) {
      mapping.map(operand, argument);
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(block);
      Operation *cloned = rewriter.clone(*operation, mapping);
      createFlowYield(rewriter, operation->getLoc(), cloned->getResults());
    }
    rewriter.replaceOp(operation, dispatch->getResults());
    return success();
  }

 private:
  std::string kernel;
};

class FlowToStreamPattern final : public OpConversionPattern<Flow::DispatchOp> {
 public:
  FlowToStreamPattern(MLIRContext *context, int64_t *nextResourceId,
                      int64_t *nextEntryPointId)
      : OpConversionPattern(context), nextResourceId(nextResourceId),
        nextEntryPointId(nextEntryPointId) {}

  LogicalResult matchAndRewrite(
      Flow::DispatchOp operation, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> resultBytes =
        calculateResultBytes(operation->getResultTypes());
    SmallVector<int64_t> resultResources;
    SmallVector<Value> outputResources;
    resultResources.reserve(resultBytes.size());
    for (int64_t bytes : resultBytes) {
      int64_t resourceId = (*nextResourceId)++;
      resultResources.push_back(resourceId);
      SmallVector<NamedAttribute> allocAttributes{
          rewriter.getNamedAttr("resource_id",
                                rewriter.getI64IntegerAttr(resourceId)),
          rewriter.getNamedAttr("bytes", rewriter.getI64IntegerAttr(bytes))};
      Operation *alloc = createOperation(
          rewriter, operation.getLoc(), Stream::AllocOp::getOperationName(),
          Stream::ResourceType::get(rewriter.getContext()), {},
          allocAttributes);
      outputResources.push_back(alloc->getResult(0));
    }
    SmallVector<Attribute> workload;
    for (Operation &payload : operation.getBody().front().without_terminator()) {
      workload.push_back(rewriter.getStringAttr(
          payload.getName().getStringRef()));
    }
    std::string entryPoint =
        (operation.getKernel() + "_" + Twine((*nextEntryPointId)++)).str();
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("kernel", operation.getKernelAttr()),
        rewriter.getNamedAttr("entry_point",
                              rewriter.getStringAttr(entryPoint)),
        rewriter.getNamedAttr("workload", rewriter.getArrayAttr(workload)),
        rewriter.getNamedAttr(
            "operandSegmentSizes",
            rewriter.getDenseI32ArrayAttr(
                {static_cast<int32_t>(adaptor.getOperands().size()),
                 static_cast<int32_t>(outputResources.size())})),
        rewriter.getNamedAttr(
            "result_bytes", rewriter.getDenseI64ArrayAttr(resultBytes)),
        rewriter.getNamedAttr(
            "result_resources",
            rewriter.getDenseI64ArrayAttr(resultResources))};
    SmallVector<Value> dispatchOperands(adaptor.getOperands());
    llvm::append_range(dispatchOperands, outputResources);
    Operation *dispatch = createOperation(
        rewriter, operation.getLoc(), Stream::DispatchOp::getOperationName(),
        operation->getResultTypes(), dispatchOperands, attributes);
    rewriter.replaceOp(operation, dispatch->getResults());
    return success();
  }

 private:
  int64_t *nextResourceId;
  int64_t *nextEntryPointId;
};

class StreamToHALPattern final
    : public OpConversionPattern<Stream::DispatchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Stream::DispatchOp operation, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto functionType = FunctionType::get(
        rewriter.getContext(), operation.getInputs().getTypes(),
        operation.getOutputs().getTypes());
    OperationState executableState(operation.getLoc(),
                                   HAL::ExecutableOp::getOperationName());
    executableState.addAttribute(
        SymbolTable::getSymbolAttrName(), operation.getEntryPointAttr());
    executableState.addAttribute("kernel", operation.getKernelAttr());
    executableState.addAttribute("function_type",
                                 TypeAttr::get(functionType));
    executableState.addAttribute("workload", operation.getWorkloadAttr());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      ModuleOp module = operation->getParentOfType<ModuleOp>();
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create(executableState);
    }
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("entry_point", operation.getEntryPointAttr()),
        rewriter.getNamedAttr("device", rewriter.getStringAttr("cpu-sync")),
        rewriter.getNamedAttr("result_bytes", operation.getResultBytesAttr()),
        rewriter.getNamedAttr("result_resources",
                              operation.getResultResourcesAttr())};
    attributes.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes",
        rewriter.getDenseI32ArrayAttr(
            {static_cast<int32_t>(adaptor.getInputs().size()),
             static_cast<int32_t>(adaptor.getOutputResources().size())})));
    SmallVector<Value> dispatchOperands(adaptor.getInputs());
    llvm::append_range(dispatchOperands, adaptor.getOutputResources());
    Operation *dispatch = createOperation(
        rewriter, operation.getLoc(), HAL::DispatchOp::getOperationName(),
        operation->getResultTypes(), dispatchOperands, attributes);
    rewriter.replaceOp(operation, dispatch->getResults());
    return success();
  }
};

class HALExecutableToVMPattern final
    : public OpConversionPattern<HAL::ExecutableOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      HAL::ExecutableOp operation, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(operation);
    return success();
  }
};

class StreamAllocToHALPattern final
    : public OpConversionPattern<Stream::AllocOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Stream::AllocOp operation, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("resource_id", operation.getResourceIdAttr()),
        rewriter.getNamedAttr("bytes", operation.getBytesAttr())};
    Operation *alloc = createOperation(
        rewriter, operation.getLoc(), HAL::AllocOp::getOperationName(),
        HAL::BufferType::get(rewriter.getContext()), {}, attributes);
    rewriter.replaceOp(operation, alloc->getResults());
    return success();
  }
};

class StreamDeallocToHALPattern final
    : public OpConversionPattern<Stream::DeallocOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Stream::DeallocOp operation, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("resource_id", operation.getResourceIdAttr())};
    createOperation(rewriter, operation.getLoc(),
                    HAL::DeallocOp::getOperationName(), {},
                    adaptor.getOperands(), attributes);
    rewriter.eraseOp(operation);
    return success();
  }
};

class HALToVMPattern final : public OpConversionPattern<HAL::DispatchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      HAL::DispatchOp operation, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("callee", operation.getEntryPointAttr()),
        rewriter.getNamedAttr("device", operation.getDeviceAttr()),
        rewriter.getNamedAttr("result_bytes", operation.getResultBytesAttr()),
        rewriter.getNamedAttr("result_resources",
                              operation.getResultResourcesAttr())};
    attributes.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes",
        rewriter.getDenseI32ArrayAttr(
            {static_cast<int32_t>(adaptor.getInputs().size()),
             static_cast<int32_t>(adaptor.getOutputBuffers().size())})));
    SmallVector<Value> callOperands(adaptor.getInputs());
    llvm::append_range(callOperands, adaptor.getOutputBuffers());
    Operation *call = createOperation(
        rewriter, operation.getLoc(), VM::CallOp::getOperationName(),
        operation->getResultTypes(), callOperands, attributes);
    rewriter.replaceOp(operation, call->getResults());
    return success();
  }
};

class HALAllocToVMPattern final : public OpConversionPattern<HAL::AllocOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      HAL::AllocOp operation, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("resource_id", operation.getResourceIdAttr()),
        rewriter.getNamedAttr("bytes", operation.getBytesAttr())};
    Operation *alloc = createOperation(
        rewriter, operation.getLoc(), VM::AllocOp::getOperationName(),
        VM::RefType::get(rewriter.getContext()), {}, attributes);
    rewriter.replaceOp(operation, alloc->getResults());
    return success();
  }
};

class HALDeallocToVMPattern final
    : public OpConversionPattern<HAL::DeallocOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      HAL::DeallocOp operation, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("resource_id", operation.getResourceIdAttr())};
    createOperation(rewriter, operation.getLoc(), VM::DeallocOp::getOperationName(),
                    {}, adaptor.getOperands(), attributes);
    rewriter.eraseOp(operation);
    return success();
  }
};

static void scheduleStreamDeallocs(ModuleOp module) {
  struct ResourceUse {
    Value result;
    Value resource;
    int64_t resourceId;
    Operation *producer;
  };
  SmallVector<ResourceUse> resources;
  module.walk([&](Stream::DispatchOp dispatch) {
    for (auto [result, resource, resourceId] : llvm::zip_equal(
             dispatch.getOutputs(), dispatch.getOutputResources(),
             dispatch.getResultResourcesAttr().asArrayRef())) {
      resources.push_back(
          {result, resource, resourceId, dispatch.getOperation()});
    }
  });

  for (const ResourceUse &resource : resources) {
    Operation *lastUse = resource.producer;
    bool escapes = false;
    for (Operation *user : resource.result.getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        escapes = true;
        break;
      }
      if (user->getBlock() == lastUse->getBlock() &&
          lastUse->isBeforeInBlock(user)) {
        lastUse = user;
      }
    }
    if (escapes) continue;
    OpBuilder builder(lastUse);
    builder.setInsertionPointAfter(lastUse);
    SmallVector<NamedAttribute> attributes{builder.getNamedAttr(
        "resource_id", builder.getI64IntegerAttr(resource.resourceId))};
    createOperation(builder, lastUse->getLoc(),
                    Stream::DeallocOp::getOperationName(), {},
                    resource.resource, attributes);
  }
}

static LogicalResult verifyStreamResourcePlan(ModuleOp module) {
  LogicalResult status = success();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.empty()) continue;
    DenseMap<int64_t, int64_t> liveResources;
    DenseMap<Value, int64_t> resourceValues;
    DenseMap<Value, int64_t> valueResources;
    DenseSet<int64_t> escapingResources;
    for (Operation &operation : function.front()) {
      if (auto alloc = dyn_cast<Stream::AllocOp>(operation)) {
        int64_t resourceId = alloc.getResourceIdAttr().getInt();
        if (liveResources.contains(resourceId)) {
          alloc.emitOpError("resource is already live");
          status = failure();
          continue;
        }
        liveResources[resourceId] = alloc.getBytesAttr().getInt();
        resourceValues[alloc.getResource()] = resourceId;
        continue;
      }
      if (auto dispatch = dyn_cast<Stream::DispatchOp>(operation)) {
        for (Value input : dispatch.getInputs()) {
          auto valueResource = valueResources.find(input);
          if (valueResource != valueResources.end() &&
              !liveResources.contains(valueResource->second)) {
            dispatch.emitOpError("reads a resource after it was released");
            status = failure();
          }
        }
        for (auto [resourceValue, resourceId] : llvm::zip_equal(
                 dispatch.getOutputResources(),
                 dispatch.getResultResourcesAttr().asArrayRef())) {
          auto mappedId = resourceValues.find(resourceValue);
          if (mappedId == resourceValues.end() || mappedId->second != resourceId) {
            dispatch.emitOpError(
                "output resource SSA value does not match resource id");
            status = failure();
          }
        }
        for (auto [result, resourceId, bytes] : llvm::zip_equal(
                 dispatch.getOutputs(),
                 dispatch.getResultResourcesAttr().asArrayRef(),
                 dispatch.getResultBytesAttr().asArrayRef())) {
          auto resource = liveResources.find(resourceId);
          if (resource == liveResources.end() || resource->second != bytes) {
            dispatch.emitOpError(
                "result references an unallocated or incorrectly sized resource");
            status = failure();
            continue;
          }
          valueResources[result] = resourceId;
          if (llvm::any_of(result.getUsers(),
                           [](Operation *user) {
                             return isa<func::ReturnOp>(user);
                           })) {
            escapingResources.insert(resourceId);
          }
        }
        continue;
      }
      if (auto dealloc = dyn_cast<Stream::DeallocOp>(operation)) {
        int64_t resourceId = dealloc.getResourceIdAttr().getInt();
        auto mappedId = resourceValues.find(dealloc.getResource());
        if (mappedId == resourceValues.end() || mappedId->second != resourceId) {
          dealloc.emitOpError(
              "resource SSA value does not match resource id");
          status = failure();
        }
        if (!liveResources.erase(resourceId)) {
          dealloc.emitOpError("resource is not live");
          status = failure();
        }
      }
    }
    for (auto [resourceId, bytes] : liveResources) {
      (void)bytes;
      if (!escapingResources.contains(resourceId)) {
        function.emitOpError("non-output resource remains live: ")
            << resourceId;
        status = failure();
      }
    }
  }
  return status;
}

template <typename SourceDialect, typename TargetDialect,
          typename... Patterns>
LogicalResult applyStageConversion(ModuleOp module) {
  MLIRContext *context = module.getContext();
  ConversionTarget target(*context);
  target.addIllegalDialect<SourceDialect>();
  target.addLegalDialect<TargetDialect>();
  target.addLegalOp<Flow::YieldOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  RewritePatternSet patterns(context);
  patterns.add<Patterns...>(context);
  return applyFullConversion(module, target, std::move(patterns));
}

class GlobalOptimizationPass final
    : public PassWrapper<GlobalOptimizationPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GlobalOptimizationPass)

  StringRef getArgument() const override { return "tiree-global-optimize"; }
  StringRef getDescription() const override {
    return "Run tiny whole-program tensor fusion and global optimization";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Input::TinyInputDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatMulAddReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class InputToFlowPass final
    : public PassWrapper<InputToFlowPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InputToFlowPass)

  StringRef getArgument() const override { return "tiree-input-to-flow"; }
  StringRef getDescription() const override {
    return "Legalize tiny input ops and form flow dispatches";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Input::TinyInputDialect, Flow::TinyFlowDialect>();
  }
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<Flow::TinyFlowDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    auto isFlowPayload = [](Operation *operation) {
      return operation->getParentOfType<Flow::DispatchOp>() != nullptr;
    };
    target.addDynamicallyLegalOp<Input::MatMulOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::FusedMatMulAddReluOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::AddOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::ReluOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::FakeQuantOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::SplitOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::Conv2DOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::MaxPool2DOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::ReshapeOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::TransposeOp>(isFlowPayload);
    target.addDynamicallyLegalOp<Input::SoftmaxOp>(isFlowPayload);
    RewritePatternSet patterns(&getContext());
    patterns.add<InputOpToFlowPattern>(Input::MatMulOp::getOperationName(),
                                      "matmul", &getContext());
    patterns.add<InputOpToFlowPattern>(
        Input::FusedMatMulAddReluOp::getOperationName(), "matmul_add_relu",
        &getContext());
    patterns.add<InputOpToFlowPattern>(Input::AddOp::getOperationName(), "add",
                                      &getContext());
    patterns.add<InputOpToFlowPattern>(Input::ReluOp::getOperationName(),
                                      "relu", &getContext());
    patterns.add<InputOpToFlowPattern>(Input::FakeQuantOp::getOperationName(),
                                      "fake_quant", &getContext());
    patterns.add<InputOpToFlowPattern>(Input::SplitOp::getOperationName(),
                                      "split", &getContext());
    patterns.add<InputOpToFlowPattern>(Input::Conv2DOp::getOperationName(),
                                      "conv2d", &getContext());
    patterns.add<InputOpToFlowPattern>(Input::MaxPool2DOp::getOperationName(),
                                      "max_pool2d", &getContext());
    patterns.add<InputOpToFlowPattern>(Input::ReshapeOp::getOperationName(),
                                      "reshape", &getContext());
    patterns.add<InputOpToFlowPattern>(Input::TransposeOp::getOperationName(),
                                      "transpose", &getContext());
    patterns.add<InputOpToFlowPattern>(Input::SoftmaxOp::getOperationName(),
                                      "softmax", &getContext());
    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class FlowToStreamPass final
    : public PassWrapper<FlowToStreamPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlowToStreamPass)
  StringRef getArgument() const override { return "tiree-flow-to-stream"; }
  StringRef getDescription() const override {
    return "Plan static resources for one synchronous stream";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Flow::TinyFlowDialect, Stream::TinyStreamDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addIllegalDialect<Flow::TinyFlowDialect>();
    target.addLegalDialect<Stream::TinyStreamDialect>();
    target.addLegalOp<Flow::YieldOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    int64_t nextResourceId = 0;
    int64_t nextEntryPointId = 0;
    RewritePatternSet patterns(context);
    patterns.add<FlowToStreamPattern>(context, &nextResourceId,
                                      &nextEntryPointId);
    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(patterns)))) {
      return signalPassFailure();
    }
    scheduleStreamDeallocs(getOperation());
    if (failed(getOperation().verify()) ||
        failed(verifyStreamResourcePlan(getOperation()))) {
      signalPassFailure();
    }
  }
};

class VerifyStreamResourcesPass final
    : public PassWrapper<VerifyStreamResourcesPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyStreamResourcesPass)
  StringRef getArgument() const override {
    return "tiree-verify-stream-resources";
  }
  StringRef getDescription() const override {
    return "Verify tiny stream allocation and lifetime planning";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Stream::TinyStreamDialect>();
  }
  void runOnOperation() override {
    if (failed(verifyStreamResourcePlan(getOperation()))) signalPassFailure();
  }
};

class StreamToHALPass final
    : public PassWrapper<StreamToHALPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StreamToHALPass)
  StringRef getArgument() const override { return "tiree-stream-to-hal"; }
  StringRef getDescription() const override {
    return "Select the synchronous CPU HAL device";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Stream::TinyStreamDialect, HAL::TinyHALDialect>();
  }
  void runOnOperation() override {
    if (failed(applyStageConversion<Stream::TinyStreamDialect,
                                    HAL::TinyHALDialect,
                                    StreamToHALPattern,
                                    StreamAllocToHALPattern,
                                    StreamDeallocToHALPattern>(
        getOperation()))) {
      signalPassFailure();
    }
  }
};

class HALToVMPass final
    : public PassWrapper<HALToVMPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HALToVMPass)
  StringRef getArgument() const override { return "tiree-hal-to-vm"; }
  StringRef getDescription() const override {
    return "Lower HAL dispatches to tiny VM calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HAL::TinyHALDialect, VM::TinyVMDialect>();
  }
  void runOnOperation() override {
    if (failed(applyStageConversion<HAL::TinyHALDialect, VM::TinyVMDialect,
                                    HALToVMPattern, HALAllocToVMPattern,
                                    HALDeallocToVMPattern,
                                    HALExecutableToVMPattern>(getOperation()))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createGlobalOptimizationPass() {
  return std::make_unique<GlobalOptimizationPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createInputToFlowPass() {
  return std::make_unique<InputToFlowPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createFlowToStreamPass() {
  return std::make_unique<FlowToStreamPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createVerifyStreamResourcesPass() {
  return std::make_unique<VerifyStreamResourcesPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createStreamToHALPass() {
  return std::make_unique<StreamToHALPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createHALToVMPass() {
  return std::make_unique<HALToVMPass>();
}

void registerTinyIREEPasses() {
  PassRegistration<GlobalOptimizationPass>();
  PassRegistration<InputToFlowPass>();
  PassRegistration<FlowToStreamPass>();
  PassRegistration<VerifyStreamResourcesPass>();
  PassRegistration<StreamToHALPass>();
  PassRegistration<HALToVMPass>();
  PassPipelineRegistration<>(
      "tiree-compile-pipeline",
      "Run the complete tiny-iree Input -> Flow -> Stream -> HAL -> VM pipeline",
      [](OpPassManager &passManager) {
        passManager.addPass(createGlobalOptimizationPass());
        passManager.addPass(createInputToFlowPass());
        passManager.addPass(createFlowToStreamPass());
        passManager.addPass(createStreamToHALPass());
        passManager.addPass(createHALToVMPass());
      });
}

}  // namespace mlir::tiree
