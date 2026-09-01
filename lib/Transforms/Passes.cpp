#include "tiny_iree/Transforms/Passes.h"

#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tiree {
namespace {

Operation *createOperation(OpBuilder &builder, Location location,
                           StringRef name, TypeRange resultTypes,
                           ValueRange operands) {
  OperationState state(location, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  return builder.create(state);
}

SmallVector<int64_t> calculateResultBytes(TypeRange types) {
  SmallVector<int64_t> result;
  for (Type type : types) {
    auto tensorType = cast<RankedTensorType>(type);
    int64_t elementCount = 1;
    if (!tensorType.hasStaticShape()) {
      result.push_back(-1);
    } else {
      for (int64_t dimension : tensorType.getShape()) {
        elementCount *= dimension;
      }
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
    auto matmul = cast<Input::MatMulOp>(operation);
    if (!matmul.getOutput().hasOneUse()) return failure();
    auto add = dyn_cast<Input::AddOp>(*matmul.getOutput().getUsers().begin());
    if (!add || add.getLhs() != matmul.getOutput() ||
        !add.getOutput().hasOneUse()) {
      return failure();
    }
    auto relu = dyn_cast<Input::ReluOp>(*add.getOutput().getUsers().begin());
    if (!relu || relu.getInput() != add.getOutput()) return failure();
    Operation *fused = createOperation(
        rewriter, matmul.getLoc(),
        Input::FusedMatMulAddReluOp::getOperationName(),
        relu.getOutput().getType(),
        {matmul.getLhs(), matmul.getRhs(), add.getRhs()});
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
    OperationState state(operation->getLoc(),
                         Flow::DispatchOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(operation->getResultTypes());
    state.addAttribute("kernel", rewriter.getStringAttr(kernel));
    state.addRegion();
    Operation *dispatch = rewriter.create(state);
    Region &body = dispatch->getRegion(0);
    body.push_back(new Block());
    Block &block = body.front();
    for (Value input : operands) {
      block.addArgument(input.getType(), operation->getLoc());
    }
    IRMapping mapping;
    for (auto [operand, argument] :
         llvm::zip_equal(operation->getOperands(), block.getArguments())) {
      mapping.map(operand, argument);
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&block);
      Operation *cloned = rewriter.clone(*operation, mapping);
      OperationState yieldState(operation->getLoc(),
                                Flow::YieldOp::getOperationName());
      yieldState.addOperands(cloned->getResults());
      rewriter.create(yieldState);
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
    for (int64_t bytes : resultBytes) {
      int64_t resourceId = (*nextResourceId)++;
      resultResources.push_back(resourceId);
      OperationState allocState(operation.getLoc(),
                                Stream::AllocOp::getOperationName());
      allocState.addTypes(Stream::ResourceType::get(rewriter.getContext()));
      allocState.addAttribute("resource_id",
                              rewriter.getI64IntegerAttr(resourceId));
      allocState.addAttribute("bytes", rewriter.getI64IntegerAttr(bytes));
      outputResources.push_back(rewriter.create(allocState)->getResult(0));
    }
    SmallVector<Attribute> workload;
    for (Operation &payload : operation.getBody().front().without_terminator()) {
      workload.push_back(
          rewriter.getStringAttr(payload.getName().getStringRef()));
    }
    std::string entryPoint =
        (operation.getKernel() + "_" + Twine((*nextEntryPointId)++)).str();
    OperationState dispatchState(operation.getLoc(),
                                 Stream::DispatchOp::getOperationName());
    dispatchState.addOperands(adaptor.getOperands());
    dispatchState.addOperands(outputResources);
    dispatchState.addTypes(operation->getResultTypes());
    dispatchState.addAttribute("kernel", operation.getKernelAttr());
    dispatchState.addAttribute("entry_point",
                               rewriter.getStringAttr(entryPoint));
    dispatchState.addAttribute("workload", rewriter.getArrayAttr(workload));
    dispatchState.addAttribute(
        "operandSegmentSizes",
        rewriter.getDenseI32ArrayAttr(
            {static_cast<int32_t>(adaptor.getOperands().size()),
             static_cast<int32_t>(outputResources.size())}));
    dispatchState.addAttribute("result_bytes",
                               rewriter.getDenseI64ArrayAttr(resultBytes));
    dispatchState.addAttribute(
        "result_resources", rewriter.getDenseI64ArrayAttr(resultResources));
    Operation *dispatch = rewriter.create(dispatchState);
    rewriter.replaceOp(operation, dispatch->getResults());
    return success();
  }

 private:
  int64_t *nextResourceId;
  int64_t *nextEntryPointId;
};

void scheduleStreamDeallocs(ModuleOp module) {
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
    OperationState deallocState(lastUse->getLoc(),
                                Stream::DeallocOp::getOperationName());
    deallocState.addOperands(resource.resource);
    deallocState.addAttribute("resource_id",
                              builder.getI64IntegerAttr(resource.resourceId));
    builder.create(deallocState);
  }
}

LogicalResult verifyStreamResourcePlan(ModuleOp module) {
  LogicalResult status = success();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.empty()) continue;
    DenseSet<int64_t> liveResources;
    DenseMap<Value, int64_t> resourceValues;
    for (Operation &operation : function.front()) {
      if (auto alloc = dyn_cast<Stream::AllocOp>(operation)) {
        int64_t id = alloc.getResourceIdAttr().getInt();
        if (!liveResources.insert(id).second) {
          alloc.emitOpError("resource is already live");
          status = failure();
        }
        resourceValues[alloc.getResource()] = id;
      } else if (auto dispatch = dyn_cast<Stream::DispatchOp>(operation)) {
        for (auto [resource, id] : llvm::zip_equal(
                 dispatch.getOutputResources(),
                 dispatch.getResultResourcesAttr().asArrayRef())) {
          if (!resourceValues.contains(resource) ||
              resourceValues.lookup(resource) != id ||
              !liveResources.contains(id)) {
            dispatch.emitOpError("references an invalid output resource");
            status = failure();
          }
        }
      } else if (auto dealloc = dyn_cast<Stream::DeallocOp>(operation)) {
        int64_t id = dealloc.getResourceIdAttr().getInt();
        if (!resourceValues.contains(dealloc.getResource()) ||
            resourceValues.lookup(dealloc.getResource()) != id ||
            !liveResources.erase(id)) {
          dealloc.emitOpError("resource is not live or id does not match");
          status = failure();
        }
      }
    }
  }
  return status;
}

class GlobalOptimizationPass final
    : public PassWrapper<GlobalOptimizationPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GlobalOptimizationPass)
  StringRef getArgument() const override { return "tiree-global-optimize"; }
  StringRef getDescription() const override {
    return "Fuse tiny whole-program tensor patterns";
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
    return "Form isolated flow dispatches";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Input::TinyInputDialect, Flow::TinyFlowDialect>();
  }
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<Flow::TinyFlowDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    auto isPayload = [](Operation *operation) {
      return operation->getParentOfType<Flow::DispatchOp>() != nullptr;
    };
    target.addDynamicallyLegalOp<Input::MatMulOp>(isPayload);
    target.addDynamicallyLegalOp<Input::FusedMatMulAddReluOp>(isPayload);
    target.addDynamicallyLegalOp<Input::AddOp>(isPayload);
    target.addDynamicallyLegalOp<Input::ReluOp>(isPayload);
    target.addDynamicallyLegalOp<Input::SoftmaxOp>(isPayload);
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
    return "Plan synchronous stream resources and lifetimes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Flow::TinyFlowDialect, Stream::TinyStreamDialect>();
  }
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<Flow::TinyFlowDialect>();
    target.addLegalDialect<Stream::TinyStreamDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    int64_t nextResourceId = 0;
    int64_t nextEntryPointId = 0;
    RewritePatternSet patterns(&getContext());
    patterns.add<FlowToStreamPattern>(&getContext(), &nextResourceId,
                                      &nextEntryPointId);
    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(patterns)))) {
      return signalPassFailure();
    }
    scheduleStreamDeallocs(getOperation());
    if (failed(verifyStreamResourcePlan(getOperation()))) {
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
  void runOnOperation() override {
    if (failed(verifyStreamResourcePlan(getOperation()))) signalPassFailure();
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
void registerTinyIREEPasses() {
  PassRegistration<GlobalOptimizationPass>();
  PassRegistration<InputToFlowPass>();
  PassRegistration<FlowToStreamPass>();
  PassRegistration<VerifyStreamResourcesPass>();
  PassPipelineRegistration<>(
      "tiree-compile-pipeline", "Run Input -> Flow",
      [](OpPassManager &manager) {
        manager.addPass(createGlobalOptimizationPass());
        manager.addPass(createInputToFlowPass());
        manager.addPass(createFlowToStreamPass());
      });
}

}  // namespace mlir::tiree
