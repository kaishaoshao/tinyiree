#include <cstdint>
#include <string>

#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"
#include "tiny_iree/Runtime/VMBytecode.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace {

namespace bytecode = mlir::tiree::vm_bytecode;

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<tiny VM MLIR module>"),
    llvm::cl::Required);
llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output Tiny VM bytecode"), llvm::cl::Required);

class Writer {
 public:
  explicit Writer(llvm::raw_ostream &stream) : stream(stream) {}

  void writeU8(uint8_t value) { stream.write(reinterpret_cast<char *>(&value), 1); }

  void writeU32(uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
      writeU8(static_cast<uint8_t>(value >> shift));
    }
  }

  void writeU64(uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
      writeU8(static_cast<uint8_t>(value >> shift));
    }
  }

  void writeI64(int64_t value) { writeU64(static_cast<uint64_t>(value)); }

  void writeString(llvm::StringRef value) {
    writeU32(static_cast<uint32_t>(value.size()));
    stream.write(value.data(), value.size());
  }

  mlir::LogicalResult writeTensorType(mlir::Operation *operation,
                                      mlir::Type type) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType || !tensorType.getElementType().isF32()) {
      return operation->emitOpError(
          "Tiny VM bytecode requires ranked f32 tensors");
    }
    writeU32(bytecode::kF32ElementType);
    writeU32(static_cast<uint32_t>(tensorType.getRank()));
    for (int64_t dimension : tensorType.getShape()) {
      writeI64(mlir::ShapedType::isDynamic(dimension) ? -1 : dimension);
    }
    return mlir::success();
  }

 private:
  llvm::raw_ostream &stream;
};

mlir::LogicalResult serializeFunction(mlir::func::FuncOp function,
                                      Writer &writer) {
  if (function.empty() || !function.getBody().hasOneBlock()) {
    return function.emitOpError("Tiny VM bytecode requires one entry block");
  }

  writer.writeString(function.getName());
  writer.writeU32(function.getNumArguments());
  llvm::DenseMap<mlir::Value, uint32_t> valueIds;
  llvm::DenseMap<mlir::Value, std::pair<int64_t, int64_t>> resourceRefs;
  uint32_t nextValueId = 0;
  for (mlir::BlockArgument argument : function.getArguments()) {
    valueIds[argument] = nextValueId++;
    if (mlir::failed(writer.writeTensorType(function, argument.getType()))) {
      return mlir::failure();
    }
  }

  uint32_t instructionCount = 0;
  for (mlir::Operation &operation : function.front()) {
    if (mlir::isa<mlir::arith::ConstantOp, mlir::tiree::VM::AllocOp,
                  mlir::tiree::VM::CallOp, mlir::tiree::VM::DeallocOp,
                  mlir::func::ReturnOp>(operation)) {
      ++instructionCount;
    } else {
      return operation.emitOpError("cannot be serialized to Tiny VM bytecode");
    }
  }
  writer.writeU32(instructionCount);

  for (mlir::Operation &operation : function.front()) {
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation)) {
      auto elements = mlir::dyn_cast<mlir::DenseFPElementsAttr>(constant.getValue());
      auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
      if (!elements || !tensorType || !tensorType.getElementType().isF32()) {
        return constant.emitOpError("requires a dense f32 tensor constant");
      }
      writer.writeU8(static_cast<uint8_t>(bytecode::Opcode::Constant));
      writer.writeU32(nextValueId);
      valueIds[constant.getResult()] = nextValueId++;
      if (mlir::failed(writer.writeTensorType(constant, constant.getType()))) {
        return mlir::failure();
      }
      writer.writeU64(elements.getNumElements());
      for (llvm::APFloat element : elements.getValues<llvm::APFloat>()) {
        writer.writeU32(llvm::bit_cast<uint32_t>(element.convertToFloat()));
      }
      continue;
    }
    if (auto alloc = mlir::dyn_cast<mlir::tiree::VM::AllocOp>(operation)) {
      writer.writeU8(static_cast<uint8_t>(bytecode::Opcode::Alloc));
      writer.writeI64(alloc.getResourceId());
      writer.writeI64(alloc.getBytes());
      resourceRefs[alloc.getRef()] = {alloc.getResourceId(), alloc.getBytes()};
      continue;
    }
    if (auto call = mlir::dyn_cast<mlir::tiree::VM::CallOp>(operation)) {
      writer.writeU8(static_cast<uint8_t>(bytecode::Opcode::Call));
      writer.writeString(call.getCallee());
      writer.writeString(call.getDevice());
      writer.writeU32(call.getInputs().size());
      for (mlir::Value input : call.getInputs()) {
        auto valueId = valueIds.find(input);
        if (valueId == valueIds.end()) {
          return call.emitOpError("operand is not defined before use");
        }
        writer.writeU32(valueId->second);
      }
      writer.writeU32(call.getOutputs().size());
      for (auto [output, outputRef, resourceId, bytes] : llvm::zip_equal(
               call.getOutputs(), call.getOutputRefs(),
               call.getResultResourcesAttr().asArrayRef(),
               call.getResultBytesAttr().asArrayRef())) {
        auto resource = resourceRefs.find(outputRef);
        if (resource == resourceRefs.end() ||
            resource->second != std::make_pair(resourceId, bytes)) {
          return call.emitOpError(
              "output ref does not match its allocation metadata");
        }
        writer.writeU32(nextValueId);
        valueIds[output] = nextValueId++;
        if (mlir::failed(writer.writeTensorType(call, output.getType()))) {
          return mlir::failure();
        }
        writer.writeI64(resourceId);
        writer.writeI64(bytes);
      }
      continue;
    }
    if (auto dealloc = mlir::dyn_cast<mlir::tiree::VM::DeallocOp>(operation)) {
      auto resource = resourceRefs.find(dealloc.getRef());
      if (resource == resourceRefs.end() ||
          resource->second.first != dealloc.getResourceId()) {
        return dealloc.emitOpError(
            "deallocated ref does not match resource metadata");
      }
      writer.writeU8(static_cast<uint8_t>(bytecode::Opcode::Dealloc));
      writer.writeI64(dealloc.getResourceId());
      continue;
    }
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(operation);
    writer.writeU8(static_cast<uint8_t>(bytecode::Opcode::Return));
    writer.writeU32(returnOp.getNumOperands());
    for (mlir::Value value : returnOp.getOperands()) {
      auto valueId = valueIds.find(value);
      if (valueId == valueIds.end()) {
        return returnOp.emitOpError("return value is not materialized");
      }
      writer.writeU32(valueId->second);
    }
  }
  return mlir::success();
}

mlir::LogicalResult serializeModule(mlir::ModuleOp module,
                                    llvm::raw_ostream &stream) {
  Writer writer(stream);
  stream.write(bytecode::kMagic.data(), bytecode::kMagic.size());
  writer.writeU32(bytecode::kVersion);
  uint32_t functionCount =
      static_cast<uint32_t>(std::distance(module.getOps<mlir::func::FuncOp>().begin(),
                                          module.getOps<mlir::func::FuncOp>().end()));
  writer.writeU32(functionCount);
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (mlir::failed(serializeFunction(function, writer))) return mlir::failure();
  }
  return mlir::success();
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Tiny IREE VM bytecode serializer\n");
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::tiree::registerTinyIREEDialects(registry);
  mlir::MLIRContext context(registry);

  auto file = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!file) {
    llvm::errs() << "unable to read module: " << inputFilename << "\n";
    return EXIT_FAILURE;
  }
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module || mlir::failed(module->verify())) return EXIT_FAILURE;

  std::error_code error;
  llvm::raw_fd_ostream output(outputFilename, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "unable to open output: " << error.message() << "\n";
    return EXIT_FAILURE;
  }
  return mlir::succeeded(serializeModule(*module, output)) ? EXIT_SUCCESS
                                                           : EXIT_FAILURE;
}
