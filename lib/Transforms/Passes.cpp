#include "tiny_iree/Transforms/Passes.h"

#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"
#include "llvm/ADT/STLExtras.h"
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

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createGlobalOptimizationPass() {
  return std::make_unique<GlobalOptimizationPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createInputToFlowPass() {
  return std::make_unique<InputToFlowPass>();
}
void registerTinyIREEPasses() {
  PassRegistration<GlobalOptimizationPass>();
  PassRegistration<InputToFlowPass>();
  PassPipelineRegistration<>(
      "tiree-compile-pipeline", "Run Input -> Flow",
      [](OpPassManager &manager) {
        manager.addPass(createGlobalOptimizationPass());
        manager.addPass(createInputToFlowPass());
      });
}

}  // namespace mlir::tiree
