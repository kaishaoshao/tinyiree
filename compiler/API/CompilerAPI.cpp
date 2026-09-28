#include "tiny_iree/Compiler/API.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "tiny_iree/IR/Dialects.h"

namespace {

void copyDiagnostic(const std::string &diagnostic, char *buffer,
                    size_t bufferSize) {
  if (!buffer || bufferSize == 0) return;
  size_t length = std::min(diagnostic.size(), bufferSize - 1);
  std::memcpy(buffer, diagnostic.data(), length);
  buffer[length] = '\0';
}

tiny_iree_compiler_status_t failWithDiagnostic(
    tiny_iree_compiler_status_t status, std::string diagnostic, char *buffer,
    size_t bufferSize) {
  if (diagnostic.empty()) diagnostic = "Tiny-IREE compiler operation failed";
  copyDiagnostic(diagnostic, buffer, bufferSize);
  return status;
}

}  // namespace

extern "C" const char *tiny_iree_compiler_api_version(void) {
  return "tiny-iree-compiler-api-v1";
}

extern "C" tiny_iree_compiler_status_t tiny_iree_compiler_validate_mlir(
    const char *mlirText, char *diagnosticBuffer, size_t diagnosticBufferSize) {
  if (!mlirText) {
    return failWithDiagnostic(TINY_IREE_COMPILER_STATUS_INVALID_ARGUMENT,
                              "mlir_text must not be null", diagnosticBuffer,
                              diagnosticBufferSize);
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::tiree::registerTinyIREEDialects(registry);
  mlir::MLIRContext context(registry);

  std::string diagnostic;
  llvm::raw_string_ostream diagnosticStream(diagnostic);
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [&diagnosticStream](mlir::Diagnostic &message) {
        message.print(diagnosticStream);
        diagnosticStream << '\n';
        return mlir::success();
      });

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(mlirText), &context);
  if (!module) {
    diagnosticStream.flush();
    return failWithDiagnostic(TINY_IREE_COMPILER_STATUS_PARSE_ERROR, diagnostic,
                              diagnosticBuffer, diagnosticBufferSize);
  }
  if (mlir::failed(module->verify())) {
    diagnosticStream.flush();
    return failWithDiagnostic(TINY_IREE_COMPILER_STATUS_VERIFICATION_ERROR,
                              diagnostic, diagnosticBuffer,
                              diagnosticBufferSize);
  }

  copyDiagnostic("", diagnosticBuffer, diagnosticBufferSize);
  return TINY_IREE_COMPILER_STATUS_OK;
}
