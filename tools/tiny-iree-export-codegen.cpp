#include <string>

#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace {

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<tiny HAL MLIR module>"),
    llvm::cl::Required);
llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output codegen plan JSON"), llvm::cl::Required);

std::string printType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

llvm::json::Array printTypes(mlir::TypeRange types) {
  llvm::json::Array result;
  for (mlir::Type type : types) result.push_back(printType(type));
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Tiny IREE HAL codegen plan exporter\n");
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::tiree::registerTinyIREEDialects(registry);
  mlir::MLIRContext context(registry);

  auto file = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!file) {
    llvm::errs() << "unable to read HAL module: " << inputFilename << "\n";
    return EXIT_FAILURE;
  }
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module || mlir::failed(module->verify())) return EXIT_FAILURE;

  llvm::json::Array entryPoints;
  for (mlir::tiree::HAL::ExecutableOp executable :
       module->getOps<mlir::tiree::HAL::ExecutableOp>()) {
    auto functionType =
        mlir::cast<mlir::FunctionType>(executable.getFunctionType());
    llvm::json::Array workload;
    for (mlir::Attribute attribute : executable.getWorkload()) {
      workload.push_back(mlir::cast<mlir::StringAttr>(attribute).getValue());
    }
    llvm::json::Object entryPoint;
    entryPoint["name"] = executable.getSymName().str();
    entryPoint["kernel"] = executable.getKernel().str();
    entryPoint["workload"] = std::move(workload);
    entryPoint["inputs"] = printTypes(functionType.getInputs());
    entryPoint["outputs"] = printTypes(functionType.getResults());
    entryPoints.push_back(std::move(entryPoint));
  }
  if (entryPoints.empty()) {
    llvm::errs() << "HAL module contains no executable entry points\n";
    return EXIT_FAILURE;
  }
  llvm::json::Object root;
  root["format"] = "tiny-iree-codegen-plan-v1";
  root["entry_points"] = std::move(entryPoints);

  std::error_code error;
  llvm::raw_fd_ostream output(outputFilename, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "unable to open output: " << error.message() << "\n";
    return EXIT_FAILURE;
  }
  output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
  return EXIT_SUCCESS;
}
