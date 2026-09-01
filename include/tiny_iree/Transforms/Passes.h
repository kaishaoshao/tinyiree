#ifndef TINY_IREE_TRANSFORMS_PASSES_H_
#define TINY_IREE_TRANSFORMS_PASSES_H_

#include <memory>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tiree {
std::unique_ptr<OperationPass<ModuleOp>> createGlobalOptimizationPass();
std::unique_ptr<OperationPass<ModuleOp>> createInputToFlowPass();
std::unique_ptr<OperationPass<ModuleOp>> createFlowToStreamPass();
std::unique_ptr<OperationPass<ModuleOp>> createVerifyStreamResourcesPass();
void registerTinyIREEPasses();
}  // namespace mlir::tiree

#endif  // TINY_IREE_TRANSFORMS_PASSES_H_
