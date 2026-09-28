#ifndef TINY_IREE_RUNTIME_RUNTIME_H_
#define TINY_IREE_RUNTIME_RUNTIME_H_

namespace mlir::tiree::runtime {

// Runs the tiny-iree VM/HAL command-line runtime.
int runModuleMain(int argc, char **argv);

}  // namespace mlir::tiree::runtime

#endif  // TINY_IREE_RUNTIME_RUNTIME_H_
