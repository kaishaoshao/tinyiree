#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::tiree::registerTinyIREEDialects(registry);
  mlir::tiree::registerTinyIREEPasses();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "tiny-iree MLIR compiler driver\n", registry));
}
