#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tiny_iree/IR/Dialects.h"
#include "tiny_iree/IR/Ops.h"
#include "tiny_iree/Runtime/NativeABI.h"
#include "tiny_iree/Runtime/VMBytecode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/CommandLine.h"
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

namespace bytecode = mlir::tiree::vm_bytecode;

class TensorValues {
 public:
  TensorValues() : storage(std::make_shared<std::vector<float>>()) {}
  TensorValues(std::vector<float> values)
      : storage(std::make_shared<std::vector<float>>(std::move(values))) {}
  explicit TensorValues(std::shared_ptr<std::vector<float>> storage)
      : storage(std::move(storage)) {}
  TensorValues(const TensorValues &other)
      : storage(std::make_shared<std::vector<float>>(*other.storage)) {}
  TensorValues &operator=(const TensorValues &other) {
    if (this != &other) {
      storage = std::make_shared<std::vector<float>>(*other.storage);
    }
    return *this;
  }
  TensorValues(TensorValues &&) = default;
  TensorValues &operator=(TensorValues &&) = default;

  float *data() { return storage->data(); }
  const float *data() const { return storage->data(); }
  size_t size() const { return storage->size(); }
  auto begin() { return storage->begin(); }
  auto end() { return storage->end(); }
  auto begin() const { return storage->begin(); }
  auto end() const { return storage->end(); }
  float &operator[](size_t index) { return (*storage)[index]; }
  const float &operator[](size_t index) const { return (*storage)[index]; }
  void reserve(size_t size) { storage->reserve(size); }
  void push_back(float value) { storage->push_back(value); }
  std::shared_ptr<std::vector<float>> getStorage() const { return storage; }

 private:
  std::shared_ptr<std::vector<float>> storage;
};

struct Tensor {
  std::vector<int64_t> shape;
  TensorValues values;
};

struct RuntimeResource {
  int64_t bytes;
  std::shared_ptr<std::vector<float>> storage;
};

class RuntimeAllocator {
 public:
  std::shared_ptr<std::vector<float>> acquire(int64_t bytes) {
    auto &bucket = freeStorage[bytes];
    if (!bucket.empty()) {
      std::shared_ptr<std::vector<float>> storage = std::move(bucket.back());
      bucket.pop_back();
      ++reusedAllocations;
      return storage;
    }
    ++newAllocations;
    return std::make_shared<std::vector<float>>(bytes / sizeof(float));
  }

  void release(const RuntimeResource &resource) {
    if (resource.bytes > 0 && resource.storage) {
      freeStorage[resource.bytes].push_back(resource.storage);
    }
  }

  int64_t getNewAllocations() const { return newAllocations; }
  int64_t getReusedAllocations() const { return reusedAllocations; }

 private:
  llvm::DenseMap<int64_t,
                 std::vector<std::shared_ptr<std::vector<float>>>>
      freeStorage;
  int64_t newAllocations = 0;
  int64_t reusedAllocations = 0;
};

struct BytecodeConstant {
  uint32_t resultId;
  Tensor value;
};
struct BytecodeAlloc {
  int64_t resourceId;
  int64_t bytes;
};
struct BytecodeCallOutput {
  uint32_t valueId;
  std::vector<int64_t> shape;
  int64_t resourceId;
  int64_t bytes;
};
struct BytecodeCall {
  std::string callee;
  std::string device;
  std::vector<uint32_t> inputs;
  std::vector<BytecodeCallOutput> outputs;
};
struct BytecodeDealloc {
  int64_t resourceId;
};
struct BytecodeReturn {
  std::vector<uint32_t> values;
};
using BytecodeInstruction =
    std::variant<BytecodeConstant, BytecodeAlloc, BytecodeCall,
                 BytecodeDealloc, BytecodeReturn>;
struct BytecodeFunction {
  std::string name;
  std::vector<std::vector<int64_t>> argumentShapes;
  std::vector<BytecodeInstruction> instructions;
};

class BytecodeReader {
 public:
  explicit BytecodeReader(llvm::ArrayRef<uint8_t> data) : data(data) {}

  bool readU8(uint8_t &value) {
    if (offset >= data.size()) return fail("unexpected end of bytecode");
    value = data[offset++];
    return true;
  }
  bool readU32(uint32_t &value) {
    value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
      uint8_t byte = 0;
      if (!readU8(byte)) return false;
      value |= static_cast<uint32_t>(byte) << shift;
    }
    return true;
  }
  bool readU64(uint64_t &value) {
    value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
      uint8_t byte = 0;
      if (!readU8(byte)) return false;
      value |= static_cast<uint64_t>(byte) << shift;
    }
    return true;
  }
  bool readI64(int64_t &value) {
    uint64_t bits = 0;
    if (!readU64(bits)) return false;
    value = static_cast<int64_t>(bits);
    return true;
  }
  bool readString(std::string &value) {
    uint32_t length = 0;
    if (!readU32(length) || length > data.size() - offset) {
      return fail("invalid bytecode string length");
    }
    value.assign(reinterpret_cast<const char *>(data.data() + offset), length);
    offset += length;
    return true;
  }
  bool readShape(std::vector<int64_t> &shape) {
    uint32_t elementType = 0;
    uint32_t rank = 0;
    if (!readU32(elementType) || !readU32(rank)) return false;
    if (elementType != bytecode::kF32ElementType ||
        rank > TINY_IREE_MAX_RANK) {
      return fail("unsupported tensor type in bytecode");
    }
    shape.resize(rank);
    for (int64_t &dimension : shape) {
      if (!readI64(dimension) || dimension == 0 || dimension < -1) {
        return fail("invalid tensor dimension in bytecode");
      }
    }
    return true;
  }
  bool atEnd() const { return offset == data.size(); }

 private:
  bool fail(llvm::StringRef message) {
    llvm::errs() << "invalid Tiny VM bytecode at offset " << offset << ": "
                 << message << "\n";
    return false;
  }

  llvm::ArrayRef<uint8_t> data;
  size_t offset = 0;
};

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<VM MLIR module or bundle>"),
    llvm::cl::Required);
llvm::cl::opt<std::string> functionName(
    "function", llvm::cl::desc("Function to invoke"),
    llvm::cl::init("predict"));
llvm::cl::opt<std::string> inputValues(
    "input", llvm::cl::desc(
                 "Comma-separated f32 values; separate tensors with ';'"),
    llvm::cl::Required);
llvm::cl::opt<std::string> executableFilename(
    "executable", llvm::cl::desc("Native HAL executable dynamic library"),
    llvm::cl::init(""));

class NativeExecutable {
 public:
  NativeExecutable() = default;
  NativeExecutable(const NativeExecutable &) = delete;
  NativeExecutable &operator=(const NativeExecutable &) = delete;
  ~NativeExecutable() {
    if (handle) dlclose(handle);
  }

  bool load(llvm::StringRef path) {
    handle = dlopen(path.str().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      llvm::errs() << "unable to load native executable: " << dlerror()
                   << "\n";
      return false;
    }
    return true;
  }

  void *lookup(llvm::StringRef symbol) const {
    dlerror();
    void *address = dlsym(handle, symbol.str().c_str());
    return dlerror() ? nullptr : address;
  }

  explicit operator bool() const { return handle != nullptr; }

 private:
  void *handle = nullptr;
};

bool resolveBundle(const std::filesystem::path &bundlePath,
                   std::filesystem::path &modulePath,
                   std::filesystem::path &executablePath) {
  auto manifestBuffer =
      llvm::MemoryBuffer::getFile((bundlePath / "manifest.json").string());
  if (!manifestBuffer) {
    llvm::errs() << "bundle manifest is missing\n";
    return false;
  }
  auto parsed = llvm::json::parse((*manifestBuffer)->getBuffer());
  if (!parsed) {
    llvm::errs() << "bundle manifest is not valid JSON\n";
    return false;
  }
  llvm::json::Object *manifest = parsed->getAsObject();
  if (!manifest || manifest->getString("format") != "tiny-iree-bundle-v1") {
    llvm::errs() << "unsupported bundle format\n";
    return false;
  }
  std::optional<llvm::StringRef> moduleName = manifest->getString("vm_module");
  std::optional<llvm::StringRef> executableName =
      manifest->getString("hal_executable");
  std::optional<llvm::StringRef> target = manifest->getString("target");
  if (!moduleName || !executableName || !target) {
    llvm::errs() << "bundle manifest is missing required fields\n";
    return false;
  }
#if defined(__APPLE__) && defined(__aarch64__)
  constexpr llvm::StringLiteral expectedTarget = "darwin-arm64";
#elif defined(__APPLE__) && defined(__x86_64__)
  constexpr llvm::StringLiteral expectedTarget = "darwin-x86_64";
#elif defined(__linux__) && defined(__aarch64__)
  constexpr llvm::StringLiteral expectedTarget = "linux-aarch64";
#elif defined(__linux__) && defined(__x86_64__)
  constexpr llvm::StringLiteral expectedTarget = "linux-x86_64";
#else
  constexpr llvm::StringLiteral expectedTarget = "unsupported";
#endif
  if (*target != expectedTarget) {
    llvm::errs() << "bundle target " << *target
                 << " does not match runtime target " << expectedTarget
                 << "\n";
    return false;
  }
  std::filesystem::path relativeModule(moduleName->str());
  std::filesystem::path relativeExecutable(executableName->str());
  if (relativeModule.is_absolute() || relativeExecutable.is_absolute() ||
      relativeModule.string().find("..") != std::string::npos ||
      relativeExecutable.string().find("..") != std::string::npos) {
    llvm::errs() << "bundle manifest contains an unsafe path\n";
    return false;
  }
  modulePath = bundlePath / relativeModule;
  executablePath = bundlePath / relativeExecutable;
  return true;
}

std::vector<float> parseInputValues(llvm::StringRef text) {
  std::vector<float> result;
  llvm::SmallVector<llvm::StringRef> pieces;
  text.split(pieces, ',', -1, false);
  for (llvm::StringRef piece : pieces) {
    double value = 0.0;
    if (piece.getAsDouble(value)) {
      llvm::errs() << "invalid input value: " << piece << "\n";
      std::exit(EXIT_FAILURE);
    }
    result.push_back(static_cast<float>(value));
  }
  return result;
}

struct ParsedInputTensor {
  std::vector<int64_t> shape;
  std::vector<float> values;
};

std::vector<ParsedInputTensor> parseInputTensors(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef> tensors;
  text.split(tensors, ';', -1, false);
  std::vector<ParsedInputTensor> result;
  result.reserve(tensors.size());
  for (llvm::StringRef tensor : tensors) {
    ParsedInputTensor parsed;
    auto [shapeText, valuesText] = tensor.split('=');
    if (valuesText.empty()) {
      valuesText = shapeText;
    } else {
      llvm::SmallVector<llvm::StringRef> dimensions;
      shapeText.split(dimensions, 'x', -1, false);
      for (llvm::StringRef dimensionText : dimensions) {
        int64_t dimension = 0;
        if (dimensionText.getAsInteger(10, dimension) || dimension <= 0) {
          llvm::errs() << "invalid explicit input shape: " << shapeText << "\n";
          std::exit(EXIT_FAILURE);
        }
        parsed.shape.push_back(dimension);
      }
    }
    parsed.values = parseInputValues(valuesText);
    result.push_back(std::move(parsed));
  }
  return result;
}

void printTensorResult(const Tensor &tensor, std::optional<size_t> index) {
  llvm::outs() << "result";
  if (index) llvm::outs() << "[" << *index << "]";
  llvm::outs() << ": [";
  for (size_t i = 0; i < tensor.values.size(); ++i) {
    if (i) llvm::outs() << ", ";
    llvm::outs() << tensor.values[i];
  }
  llvm::outs() << "]\n";
  auto maximum = std::max_element(tensor.values.begin(), tensor.values.end());
  llvm::outs() << "argmax";
  if (index) llvm::outs() << "[" << *index << "]";
  llvm::outs() << ": " << std::distance(tensor.values.begin(), maximum) << "\n";
}

std::vector<int64_t> getShape(mlir::Type type) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
  return {tensorType.getShape().begin(), tensorType.getShape().end()};
}

int64_t elementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dimension : shape) count *= dimension;
  return count;
}

std::vector<int64_t> resolveInputShape(llvm::ArrayRef<int64_t> declaredShape,
                                       int64_t valueCount,
                                       llvm::ArrayRef<int64_t> explicitShape) {
  if (!explicitShape.empty()) {
    if (explicitShape.size() != declaredShape.size()) {
      llvm::errs() << "explicit input shape rank does not match signature\n";
      std::exit(EXIT_FAILURE);
    }
    for (auto [declared, provided] :
         llvm::zip_equal(declaredShape, explicitShape)) {
      if (declared >= 0 && declared != provided) {
        llvm::errs() << "explicit input shape does not match static dimension\n";
        std::exit(EXIT_FAILURE);
      }
    }
    if (elementCount(explicitShape) != valueCount) {
      llvm::errs() << "explicit input shape does not match value count\n";
      std::exit(EXIT_FAILURE);
    }
    return {explicitShape.begin(), explicitShape.end()};
  }
  std::vector<int64_t> shape(declaredShape);
  int64_t staticElements = 1;
  int64_t dynamicIndex = -1;
  for (auto [index, dimension] : llvm::enumerate(shape)) {
    if (dimension < 0) {
      if (dynamicIndex >= 0) {
        llvm::errs() << "multiple dynamic dimensions require shape=values input syntax\n";
        std::exit(EXIT_FAILURE);
      }
      dynamicIndex = index;
    } else {
      staticElements *= dimension;
    }
  }
  if (dynamicIndex >= 0) {
    if (staticElements <= 0 || valueCount % staticElements != 0) {
      llvm::errs() << "input values cannot resolve dynamic shape\n";
      std::exit(EXIT_FAILURE);
    }
    shape[dynamicIndex] = valueCount / staticElements;
  }
  return shape;
}

llvm::StringRef baseKernelName(llvm::StringRef kernel) {
  size_t separator = kernel.rfind('_');
  if (separator != llvm::StringRef::npos && separator + 1 < kernel.size() &&
      std::all_of(kernel.begin() + separator + 1, kernel.end(),
                  [](char value) {
                    return std::isdigit(static_cast<unsigned char>(value));
                  })) {
    return kernel.take_front(separator);
  }
  return kernel;
}

std::vector<int64_t> resolveResultShape(
    llvm::StringRef kernel, llvm::ArrayRef<int64_t> declaredShape,
    llvm::ArrayRef<Tensor> inputs) {
  std::vector<int64_t> shape(declaredShape);
  kernel = baseKernelName(kernel);
  for (auto [index, dimension] : llvm::enumerate(shape)) {
    if (dimension >= 0) continue;
    if (kernel == "matmul" || kernel == "matmul_add_relu") {
      if (shape.size() != 2 || inputs.size() < 2) {
        llvm::errs() << "cannot resolve dynamic matmul result shape\n";
        std::exit(EXIT_FAILURE);
      }
      shape[index] = index == 0 ? inputs[0].shape[0] : inputs[1].shape[1];
    } else if (kernel == "split") {
      if (inputs.empty() || index >= inputs[0].shape.size()) {
        llvm::errs() << "cannot resolve dynamic split result shape\n";
        std::exit(EXIT_FAILURE);
      }
      shape[index] = inputs[0].shape[index];
      if (index + 1 == shape.size()) shape[index] /= 2;
    } else {
      if (inputs.empty() || index >= inputs[0].shape.size()) {
        llvm::errs() << "cannot resolve dynamic elementwise result shape\n";
        std::exit(EXIT_FAILURE);
      }
      shape[index] = inputs[0].shape[index];
    }
  }
  return shape;
}

bool parseBytecodeInstruction(BytecodeReader &reader,
                              BytecodeInstruction &instruction) {
  uint8_t rawOpcode = 0;
  if (!reader.readU8(rawOpcode)) return false;
  switch (static_cast<bytecode::Opcode>(rawOpcode)) {
    case bytecode::Opcode::Constant: {
      BytecodeConstant constant;
      uint64_t count = 0;
      if (!reader.readU32(constant.resultId) ||
          !reader.readShape(constant.value.shape) || !reader.readU64(count) ||
          count != static_cast<uint64_t>(elementCount(constant.value.shape))) {
        llvm::errs() << "invalid constant in Tiny VM bytecode\n";
        return false;
      }
      std::vector<float> values;
      values.reserve(count);
      for (uint64_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        if (!reader.readU32(bits)) return false;
        values.push_back(llvm::bit_cast<float>(bits));
      }
      constant.value.values = TensorValues(std::move(values));
      instruction = std::move(constant);
      return true;
    }
    case bytecode::Opcode::Alloc: {
      BytecodeAlloc alloc;
      if (!reader.readI64(alloc.resourceId) || !reader.readI64(alloc.bytes) ||
          alloc.resourceId < 0 || alloc.bytes == 0 || alloc.bytes < -1 ||
          (alloc.bytes > 0 && alloc.bytes % 4 != 0)) {
        llvm::errs() << "invalid allocation in Tiny VM bytecode\n";
        return false;
      }
      instruction = alloc;
      return true;
    }
    case bytecode::Opcode::Call: {
      BytecodeCall call;
      uint32_t inputCount = 0;
      uint32_t outputCount = 0;
      if (!reader.readString(call.callee) || !reader.readString(call.device) ||
          !reader.readU32(inputCount) || inputCount > 1024) {
        return false;
      }
      call.inputs.resize(inputCount);
      for (uint32_t &input : call.inputs) {
        if (!reader.readU32(input)) return false;
      }
      if (!reader.readU32(outputCount) || outputCount == 0 ||
          outputCount > 1024) {
        return false;
      }
      call.outputs.resize(outputCount);
      for (BytecodeCallOutput &output : call.outputs) {
        if (!reader.readU32(output.valueId) || !reader.readShape(output.shape) ||
            !reader.readI64(output.resourceId) || !reader.readI64(output.bytes) ||
            output.resourceId < 0 || output.bytes == 0 || output.bytes < -1 ||
            (output.bytes > 0 &&
             output.bytes != elementCount(output.shape) * 4)) {
          llvm::errs() << "invalid call result in Tiny VM bytecode\n";
          return false;
        }
      }
      instruction = std::move(call);
      return true;
    }
    case bytecode::Opcode::Dealloc: {
      BytecodeDealloc dealloc;
      if (!reader.readI64(dealloc.resourceId) || dealloc.resourceId < 0) {
        return false;
      }
      instruction = dealloc;
      return true;
    }
    case bytecode::Opcode::Return: {
      BytecodeReturn returnOp;
      uint32_t count = 0;
      if (!reader.readU32(count) || count == 0 || count > 1024) return false;
      returnOp.values.resize(count);
      for (uint32_t &value : returnOp.values) {
        if (!reader.readU32(value)) return false;
      }
      instruction = std::move(returnOp);
      return true;
    }
  }
  llvm::errs() << "unknown Tiny VM opcode: " << static_cast<int>(rawOpcode)
               << "\n";
  return false;
}

bool parseBytecode(llvm::ArrayRef<uint8_t> data,
                   std::vector<BytecodeFunction> &functions) {
  if (data.size() < bytecode::kMagic.size() ||
      !std::equal(bytecode::kMagic.begin(), bytecode::kMagic.end(),
                  reinterpret_cast<const char *>(data.data()))) {
    return false;
  }
  BytecodeReader reader(data.drop_front(bytecode::kMagic.size()));
  uint32_t version = 0;
  uint32_t functionCount = 0;
  if (!reader.readU32(version) || version != bytecode::kVersion ||
      !reader.readU32(functionCount) || functionCount == 0 ||
      functionCount > 1024) {
    llvm::errs() << "unsupported Tiny VM bytecode header\n";
    return false;
  }
  functions.resize(functionCount);
  for (BytecodeFunction &function : functions) {
    uint32_t argumentCount = 0;
    uint32_t instructionCount = 0;
    if (!reader.readString(function.name) || !reader.readU32(argumentCount) ||
        argumentCount > 1024) {
      return false;
    }
    function.argumentShapes.resize(argumentCount);
    for (auto &shape : function.argumentShapes) {
      if (!reader.readShape(shape)) return false;
    }
    if (!reader.readU32(instructionCount) || instructionCount > 1000000) {
      return false;
    }
    function.instructions.resize(instructionCount);
    for (BytecodeInstruction &instruction : function.instructions) {
      if (!parseBytecodeInstruction(reader, instruction)) return false;
    }
  }
  if (!reader.atEnd()) {
    llvm::errs() << "Tiny VM bytecode has trailing data\n";
    return false;
  }
  return true;
}

Tensor matmul(const Tensor &lhs, const Tensor &rhs) {
  int64_t m = lhs.shape[0];
  int64_t k = lhs.shape[1];
  int64_t n = rhs.shape[1];
  Tensor output{{m, n}, std::vector<float>(m * n, 0.0f)};
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      for (int64_t inner = 0; inner < k; ++inner) {
        output.values[row * n + column] +=
            lhs.values[row * k + inner] * rhs.values[inner * n + column];
      }
    }
  }
  return output;
}

Tensor add(const Tensor &lhs, const Tensor &rhs) {
  Tensor output{lhs.shape, lhs.values};
  if (lhs.values.size() == rhs.values.size()) {
    for (size_t i = 0; i < output.values.size(); ++i) {
      output.values[i] += rhs.values[i];
    }
  } else {
    size_t width = rhs.values.size();
    for (size_t i = 0; i < output.values.size(); ++i) {
      output.values[i] += rhs.values[i % width];
    }
  }
  return output;
}

Tensor relu(const Tensor &input) {
  Tensor output{input.shape, input.values};
  for (float &value : output.values) value = std::max(value, 0.0f);
  return output;
}

Tensor softmax(const Tensor &input) {
  Tensor output{input.shape, std::vector<float>(input.values.size())};
  int64_t width = input.shape.back();
  int64_t rows = input.values.size() / width;
  for (int64_t row = 0; row < rows; ++row) {
    auto begin = input.values.begin() + row * width;
    float maximum = *std::max_element(begin, begin + width);
    float total = 0.0f;
    for (int64_t i = 0; i < width; ++i) {
      float value = std::exp(input.values[row * width + i] - maximum);
      output.values[row * width + i] = value;
      total += value;
    }
    for (int64_t i = 0; i < width; ++i) {
      output.values[row * width + i] /= total;
    }
  }
  return output;
}

Tensor conv2d(const Tensor &input, const Tensor &weight, const Tensor &bias) {
  int64_t batches = input.shape[0];
  int64_t inputChannels = input.shape[1];
  int64_t inputHeight = input.shape[2];
  int64_t inputWidth = input.shape[3];
  int64_t outputChannels = weight.shape[0];
  int64_t kernelHeight = weight.shape[2];
  int64_t kernelWidth = weight.shape[3];
  int64_t outputHeight = inputHeight - kernelHeight + 1;
  int64_t outputWidth = inputWidth - kernelWidth + 1;
  Tensor output{{batches, outputChannels, outputHeight, outputWidth},
                std::vector<float>(static_cast<size_t>(
                    batches * outputChannels * outputHeight * outputWidth))};
  for (int64_t n = 0; n < batches; ++n) {
    for (int64_t oc = 0; oc < outputChannels; ++oc) {
      for (int64_t oh = 0; oh < outputHeight; ++oh) {
        for (int64_t ow = 0; ow < outputWidth; ++ow) {
          float sum = bias.values[oc];
          for (int64_t ic = 0; ic < inputChannels; ++ic) {
            for (int64_t kh = 0; kh < kernelHeight; ++kh) {
              for (int64_t kw = 0; kw < kernelWidth; ++kw) {
                size_t inputIndex = static_cast<size_t>(
                    ((n * inputChannels + ic) * inputHeight + oh + kh) *
                        inputWidth +
                    ow + kw);
                size_t weightIndex = static_cast<size_t>(
                    ((oc * inputChannels + ic) * kernelHeight + kh) *
                        kernelWidth +
                    kw);
                sum += input.values[inputIndex] * weight.values[weightIndex];
              }
            }
          }
          size_t outputIndex = static_cast<size_t>(
              ((n * outputChannels + oc) * outputHeight + oh) * outputWidth +
              ow);
          output.values[outputIndex] = sum;
        }
      }
    }
  }
  return output;
}

Tensor maxPool2d(const Tensor &input) {
  int64_t batches = input.shape[0];
  int64_t channels = input.shape[1];
  int64_t inputHeight = input.shape[2];
  int64_t inputWidth = input.shape[3];
  int64_t outputHeight = inputHeight / 2;
  int64_t outputWidth = inputWidth / 2;
  Tensor output{{batches, channels, outputHeight, outputWidth},
                std::vector<float>(static_cast<size_t>(
                    batches * channels * outputHeight * outputWidth))};
  for (int64_t n = 0; n < batches; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t oh = 0; oh < outputHeight; ++oh) {
        for (int64_t ow = 0; ow < outputWidth; ++ow) {
          float maximum = -std::numeric_limits<float>::infinity();
          for (int64_t kh = 0; kh < 2; ++kh) {
            for (int64_t kw = 0; kw < 2; ++kw) {
              size_t index = static_cast<size_t>(
                  ((n * channels + c) * inputHeight + oh * 2 + kh) *
                      inputWidth +
                  ow * 2 + kw);
              maximum = std::max(maximum, input.values[index]);
            }
          }
          output.values[static_cast<size_t>(
              ((n * channels + c) * outputHeight + oh) * outputWidth + ow)] =
              maximum;
        }
      }
    }
  }
  return output;
}

Tensor flatten(const Tensor &input) {
  return {{input.shape[0],
           static_cast<int64_t>(input.values.size()) / input.shape[0]},
          input.values};
}

Tensor transposeNCHWToNHWC(const Tensor &input) {
  int64_t nSize = input.shape[0];
  int64_t cSize = input.shape[1];
  int64_t hSize = input.shape[2];
  int64_t wSize = input.shape[3];
  Tensor output{{nSize, hSize, wSize, cSize},
                std::vector<float>(input.values.size())};
  for (int64_t n = 0; n < nSize; ++n) {
    for (int64_t c = 0; c < cSize; ++c) {
      for (int64_t h = 0; h < hSize; ++h) {
        for (int64_t w = 0; w < wSize; ++w) {
          size_t inputIndex =
              static_cast<size_t>(((n * cSize + c) * hSize + h) * wSize + w);
          size_t outputIndex =
              static_cast<size_t>(((n * hSize + h) * wSize + w) * cSize + c);
          output.values[outputIndex] = input.values[inputIndex];
        }
      }
    }
  }
  return output;
}

Tensor executeKernel(llvm::StringRef kernel, llvm::ArrayRef<Tensor> inputs) {
  kernel = baseKernelName(kernel);
  if (kernel == "matmul") return matmul(inputs[0], inputs[1]);
  if (kernel == "add") return add(inputs[0], inputs[1]);
  if (kernel == "relu") return relu(inputs[0]);
  if (kernel == "fake_quant") {
    if (inputs.size() != 3 || inputs[1].values.size() != 1 ||
        inputs[2].values.size() != 1 || inputs[1].values[0] <= 0.0f) {
      llvm::errs() << "invalid fake_quant parameters\n";
      std::exit(EXIT_FAILURE);
    }
    Tensor output{inputs[0].shape,
                  std::vector<float>(inputs[0].values.size())};
    float scale = inputs[1].values[0];
    float zeroPoint = inputs[2].values[0];
    for (auto [input, result] :
         llvm::zip_equal(inputs[0].values, output.values)) {
      float quantized = std::nearbyint(input / scale) + zeroPoint;
      quantized = std::clamp(quantized, -128.0f, 127.0f);
      result = (quantized - zeroPoint) * scale;
    }
    return output;
  }
  if (kernel == "softmax") return softmax(inputs[0]);
  if (kernel == "conv2d") return conv2d(inputs[0], inputs[1], inputs[2]);
  if (kernel == "max_pool2d") return maxPool2d(inputs[0]);
  if (kernel == "reshape") return flatten(inputs[0]);
  if (kernel == "transpose") return transposeNCHWToNHWC(inputs[0]);
  if (kernel == "matmul_add_relu") {
    return relu(add(matmul(inputs[0], inputs[1]), inputs[2]));
  }
  llvm::errs() << "unknown CPU kernel: " << kernel << "\n";
  std::exit(EXIT_FAILURE);
}

std::vector<Tensor> executeKernelResults(llvm::StringRef kernel,
                                         llvm::ArrayRef<Tensor> inputs) {
  if (baseKernelName(kernel) != "split") {
    return {executeKernel(kernel, inputs)};
  }
  if (inputs.size() != 1 || inputs[0].shape.size() != 2 ||
      inputs[0].shape.back() % 2 != 0) {
    llvm::errs() << "invalid split input\n";
    std::exit(EXIT_FAILURE);
  }
  int64_t rows = inputs[0].shape[0];
  int64_t inputWidth = inputs[0].shape[1];
  int64_t outputWidth = inputWidth / 2;
  std::vector<int64_t> outputShape{rows, outputWidth};
  Tensor left{outputShape,
              std::vector<float>(static_cast<size_t>(rows * outputWidth))};
  Tensor right{outputShape,
               std::vector<float>(static_cast<size_t>(rows * outputWidth))};
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < outputWidth; ++column) {
      left.values[row * outputWidth + column] =
          inputs[0].values[row * inputWidth + column];
      right.values[row * outputWidth + column] =
          inputs[0].values[row * inputWidth + outputWidth + column];
    }
  }
  return {std::move(left), std::move(right)};
}

tiree_tensor_view_t makeTensorView(Tensor &tensor) {
  if (tensor.shape.size() > TINY_IREE_MAX_RANK) {
    llvm::errs() << "native ABI only supports rank <= " << TINY_IREE_MAX_RANK
                 << "\n";
    std::exit(EXIT_FAILURE);
  }
  tiree_tensor_view_t view = {};
  view.data = tensor.values.data();
  view.rank = tensor.shape.size();
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    view.dims[i] = tensor.shape[i];
  }
  return view;
}

std::vector<uint64_t> makeMemRefDescriptor(Tensor &tensor) {
  size_t rank = tensor.shape.size();
  std::vector<uint64_t> descriptor(3 + 2 * rank, 0);
  uint64_t data = reinterpret_cast<uint64_t>(tensor.values.data());
  descriptor[0] = data;
  descriptor[1] = data;
  int64_t stride = 1;
  for (size_t reverseIndex = 0; reverseIndex < rank; ++reverseIndex) {
    size_t index = rank - reverseIndex - 1;
    descriptor[3 + index] = static_cast<uint64_t>(tensor.shape[index]);
    descriptor[3 + rank + index] = static_cast<uint64_t>(stride);
    stride *= tensor.shape[index];
  }
  return descriptor;
}

bool executeMLIRCInterface(void *address, llvm::MutableArrayRef<Tensor> inputs,
                           llvm::MutableArrayRef<Tensor> outputs) {
  std::vector<std::vector<uint64_t>> descriptors;
  descriptors.reserve(inputs.size() + outputs.size());
  for (Tensor &input : inputs) {
    descriptors.push_back(makeMemRefDescriptor(input));
  }
  for (Tensor &output : outputs) {
    descriptors.push_back(makeMemRefDescriptor(output));
  }
  switch (descriptors.size()) {
    case 2:
      reinterpret_cast<void (*)(void *, void *)>(address)(
          descriptors[0].data(), descriptors[1].data());
      return true;
    case 3:
      reinterpret_cast<void (*)(void *, void *, void *)>(address)(
          descriptors[0].data(), descriptors[1].data(),
          descriptors[2].data());
      return true;
    case 4:
      reinterpret_cast<void (*)(void *, void *, void *, void *)>(address)(
          descriptors[0].data(), descriptors[1].data(), descriptors[2].data(),
          descriptors[3].data());
      return true;
    case 5:
      reinterpret_cast<void (*)(void *, void *, void *, void *, void *)>(
          address)(descriptors[0].data(), descriptors[1].data(),
                   descriptors[2].data(), descriptors[3].data(),
                   descriptors[4].data());
      return true;
    default:
      return false;
  }
}

void executeNativeKernel(const NativeExecutable &executable,
                         llvm::StringRef kernel,
                         llvm::MutableArrayRef<Tensor> inputs,
                         llvm::MutableArrayRef<Tensor> outputs) {
  std::vector<tiree_tensor_view_t> inputViews;
  inputViews.reserve(inputs.size());
  for (Tensor &input : inputs) inputViews.push_back(makeTensorView(input));
  std::string cInterfaceSymbol = "_mlir_ciface_tiree_kernel_";
  cInterfaceSymbol.append(kernel.data(), kernel.size());
  if (void *cInterface = executable.lookup(cInterfaceSymbol)) {
    if (!executeMLIRCInterface(cInterface, inputs, outputs)) {
      llvm::errs() << "unsupported MLIR C interface arity: "
                   << inputs.size() + outputs.size() << "\n";
      std::exit(EXIT_FAILURE);
    }
    return;
  }

  std::string genericSymbol = "tiree_kernel_";
  genericSymbol.append(kernel.data(), kernel.size());
  auto function = reinterpret_cast<tiree_kernel_fn_t>(
      executable.lookup(genericSymbol));
  if (outputs.size() != 1) {
    llvm::errs() << "generic native ABI only supports one output\n";
    std::exit(EXIT_FAILURE);
  }
  tiree_tensor_view_t outputView = makeTensorView(outputs.front());
  if (!function || function(inputViews.data(),
                            static_cast<int64_t>(inputViews.size()),
                            &outputView)) {
    llvm::errs() << "native kernel failed: " << kernel << "\n";
    std::exit(EXIT_FAILURE);
  }
}

Tensor tensorFromConstant(mlir::arith::ConstantOp constant) {
  auto values = mlir::cast<mlir::DenseFPElementsAttr>(constant.getValue());
  Tensor tensor;
  tensor.shape = getShape(constant.getType());
  tensor.values.reserve(values.getNumElements());
  for (llvm::APFloat value : values.getValues<llvm::APFloat>()) {
    tensor.values.push_back(value.convertToFloat());
  }
  return tensor;
}

int run(mlir::ModuleOp module, const NativeExecutable *nativeExecutable) {
  auto function = module.lookupSymbol<mlir::func::FuncOp>(functionName);
  if (!function || function.empty()) {
    llvm::errs() << "function not found or has no body: " << functionName
                 << "\n";
    return EXIT_FAILURE;
  }
  std::vector<ParsedInputTensor> inputTensors = parseInputTensors(inputValues);
  if (function.getNumArguments() != inputTensors.size()) {
    llvm::errs() << "input tensor count does not match function signature\n";
    return EXIT_FAILURE;
  }

  llvm::DenseMap<mlir::Value, Tensor> values;
  llvm::DenseMap<int64_t, RuntimeResource> resources;
  llvm::DenseMap<mlir::Value, int64_t> valueResources;
  llvm::DenseMap<mlir::Value, int64_t> refResources;
  RuntimeAllocator allocator;
  int64_t liveResourceBytes = 0;
  int64_t peakResourceBytes = 0;
  mlir::Block &entry = function.front();
  for (auto [argument, inputData] :
       llvm::zip_equal(entry.getArguments(), inputTensors)) {
    Tensor input{resolveInputShape(getShape(argument.getType()),
                                   inputData.values.size(), inputData.shape),
                 std::move(inputData.values)};
    if (static_cast<int64_t>(input.values.size()) != elementCount(input.shape)) {
      llvm::errs() << "input element count does not match function argument\n";
      return EXIT_FAILURE;
    }
    values[argument] = std::move(input);
  }

  for (mlir::Operation &operation : entry) {
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation)) {
      values[constant.getResult()] = tensorFromConstant(constant);
      continue;
    }
    if (auto alloc = mlir::dyn_cast<mlir::tiree::VM::AllocOp>(operation)) {
      int64_t resourceId = alloc.getResourceIdAttr().getInt();
      int64_t bytes = alloc.getBytesAttr().getInt();
      if (resources.contains(resourceId)) {
        llvm::errs() << "resource allocated twice: " << resourceId << "\n";
        return EXIT_FAILURE;
      }
      resources[resourceId] = {
          bytes, bytes > 0 ? allocator.acquire(bytes) : nullptr};
      refResources[alloc.getRef()] = resourceId;
      if (bytes > 0) liveResourceBytes += bytes;
      peakResourceBytes = std::max(peakResourceBytes, liveResourceBytes);
      continue;
    }
    if (auto call = mlir::dyn_cast<mlir::tiree::VM::CallOp>(operation)) {
      std::vector<Tensor> arguments;
      arguments.reserve(call.getInputs().size());
      for (mlir::Value operand : call.getInputs()) {
        auto iterator = values.find(operand);
        if (iterator == values.end()) {
          llvm::errs() << "VM operand has no runtime value\n";
          return EXIT_FAILURE;
        }
        auto valueResource = valueResources.find(operand);
        if (valueResource != valueResources.end() &&
            !resources.contains(valueResource->second)) {
          llvm::errs() << "VM call reads a released resource\n";
          return EXIT_FAILURE;
        }
        arguments.push_back(iterator->second);
      }
      if (call.getOutputRefs().size() != call.getOutputs().size()) {
        llvm::errs() << "VM call output ref count mismatch\n";
        return EXIT_FAILURE;
      }
      std::vector<Tensor> results;
      results.reserve(call.getOutputs().size());
      auto resultResources = call.getResultResourcesAttr().asArrayRef();
      auto resultBytes = call.getResultBytesAttr().asArrayRef();
      for (size_t index = 0; index < call.getOutputs().size(); ++index) {
        int64_t resourceId = resultResources[index];
        int64_t expectedBytes = resultBytes[index];
        if (refResources.lookup(call.getOutputRefs()[index]) != resourceId) {
          llvm::errs() << "VM call output ref does not match resource id\n";
          return EXIT_FAILURE;
        }
        auto resource = resources.find(resourceId);
        if (resource == resources.end() ||
            (resource->second.bytes != expectedBytes && expectedBytes != -1)) {
          llvm::errs() << "VM call references an invalid output resource\n";
          return EXIT_FAILURE;
        }
        std::vector<int64_t> resultShape = resolveResultShape(
            call.getCallee(), getShape(call.getOutputs()[index].getType()),
            arguments);
        int64_t resolvedBytes = elementCount(resultShape) * sizeof(float);
        if (resource->second.bytes == -1) {
          resource->second.bytes = resolvedBytes;
          resource->second.storage = allocator.acquire(resolvedBytes);
          liveResourceBytes += resolvedBytes;
          peakResourceBytes = std::max(peakResourceBytes, liveResourceBytes);
        }
        if (resource->second.bytes != resolvedBytes) {
          llvm::errs() << "VM call result shape does not match allocation\n";
          return EXIT_FAILURE;
        }
        results.push_back(
            {std::move(resultShape), TensorValues(resource->second.storage)});
      }
      if (nativeExecutable) {
        executeNativeKernel(*nativeExecutable, call.getCallee(), arguments,
                            results);
      } else {
        std::vector<Tensor> interpreted =
            executeKernelResults(call.getCallee(), arguments);
        if (interpreted.size() != results.size()) {
          llvm::errs() << "interpreted kernel result count mismatch\n";
          return EXIT_FAILURE;
        }
        for (auto [source, destination] :
             llvm::zip_equal(interpreted, results)) {
          if (source.shape != destination.shape) {
            llvm::errs() << "interpreted kernel result shape mismatch\n";
            return EXIT_FAILURE;
          }
          std::copy(source.values.begin(), source.values.end(),
                    destination.values.begin());
        }
      }
      for (size_t index = 0; index < results.size(); ++index) {
        values[call.getOutputs()[index]] = std::move(results[index]);
        valueResources[call.getOutputs()[index]] = resultResources[index];
      }
      continue;
    }
    if (auto dealloc = mlir::dyn_cast<mlir::tiree::VM::DeallocOp>(operation)) {
      int64_t resourceId = dealloc.getResourceIdAttr().getInt();
      if (refResources.lookup(dealloc.getRef()) != resourceId) {
        llvm::errs() << "VM dealloc ref does not match resource id\n";
        return EXIT_FAILURE;
      }
      auto resource = resources.find(resourceId);
      if (resource == resources.end()) {
        llvm::errs() << "resource deallocated while not live: " << resourceId
                     << "\n";
        return EXIT_FAILURE;
      }
      liveResourceBytes -= resource->second.bytes;
      allocator.release(resource->second);
      resources.erase(resource);
      continue;
    }
    if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(operation)) {
      if (returnOp.getNumOperands() == 0) {
        llvm::errs() << "runtime expects at least one return value\n";
        return EXIT_FAILURE;
      }
      for (mlir::Value value : returnOp.getOperands()) {
        auto outputResource = valueResources.find(value);
        if (!values.contains(value) || outputResource == valueResources.end() ||
            !resources.contains(outputResource->second)) {
          llvm::errs() << "returned tensor resource is not live\n";
          return EXIT_FAILURE;
        }
      }
      llvm::outs() << "backend: "
                   << (nativeExecutable ? "native-aot" : "interpreter")
                   << "\n";
      llvm::outs() << "resources: peak=" << peakResourceBytes
                   << "B live=" << liveResourceBytes << "B\n";
      llvm::outs() << "allocator: new=" << allocator.getNewAllocations()
                   << " reused=" << allocator.getReusedAllocations() << "\n";
      for (auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
        printTensorResult(values.find(value)->second,
                          returnOp.getNumOperands() == 1
                              ? std::optional<size_t>()
                              : std::optional<size_t>(index));
      }
      return EXIT_SUCCESS;
    }
    llvm::errs() << "runtime cannot execute operation: "
                 << operation.getName() << "\n";
    return EXIT_FAILURE;
  }

  llvm::errs() << "function did not return\n";
  return EXIT_FAILURE;
}

int runBytecode(llvm::ArrayRef<BytecodeFunction> functions,
                const NativeExecutable *nativeExecutable) {
  auto function = std::find_if(
      functions.begin(), functions.end(),
      [](const BytecodeFunction &candidate) {
        return candidate.name == functionName;
      });
  if (function == functions.end()) {
    llvm::errs() << "function not found in Tiny VM bytecode: " << functionName
                 << "\n";
    return EXIT_FAILURE;
  }
  std::vector<ParsedInputTensor> parsedInputs = parseInputTensors(inputValues);
  if (function->argumentShapes.size() != parsedInputs.size()) {
    llvm::errs() << "input tensor count does not match bytecode signature\n";
    return EXIT_FAILURE;
  }

  llvm::DenseMap<uint32_t, Tensor> values;
  llvm::DenseMap<uint32_t, int64_t> valueResources;
  llvm::DenseMap<int64_t, RuntimeResource> resources;
  RuntimeAllocator allocator;
  for (auto [index, shape] : llvm::enumerate(function->argumentShapes)) {
    std::vector<int64_t> resolvedShape =
        resolveInputShape(shape, parsedInputs[index].values.size(),
                          parsedInputs[index].shape);
    if (static_cast<int64_t>(parsedInputs[index].values.size()) !=
        elementCount(resolvedShape)) {
      llvm::errs() << "input element count does not match bytecode signature\n";
      return EXIT_FAILURE;
    }
    values[index] = {std::move(resolvedShape),
                     std::move(parsedInputs[index].values)};
  }
  int64_t liveResourceBytes = 0;
  int64_t peakResourceBytes = 0;

  for (const BytecodeInstruction &instruction : function->instructions) {
    if (const auto *constant = std::get_if<BytecodeConstant>(&instruction)) {
      if (values.contains(constant->resultId)) {
        llvm::errs() << "bytecode value is defined twice\n";
        return EXIT_FAILURE;
      }
      values[constant->resultId] = constant->value;
      continue;
    }
    if (const auto *alloc = std::get_if<BytecodeAlloc>(&instruction)) {
      if (resources.contains(alloc->resourceId)) {
        llvm::errs() << "resource allocated twice: " << alloc->resourceId
                     << "\n";
        return EXIT_FAILURE;
      }
      resources[alloc->resourceId] = {
          alloc->bytes,
          alloc->bytes > 0 ? allocator.acquire(alloc->bytes) : nullptr};
      if (alloc->bytes > 0) liveResourceBytes += alloc->bytes;
      peakResourceBytes = std::max(peakResourceBytes, liveResourceBytes);
      continue;
    }
    if (const auto *call = std::get_if<BytecodeCall>(&instruction)) {
      if (call->device != "cpu-sync" || call->outputs.empty()) {
        llvm::errs() << "unsupported Tiny VM call ABI\n";
        return EXIT_FAILURE;
      }
      std::vector<Tensor> arguments;
      arguments.reserve(call->inputs.size());
      for (uint32_t inputId : call->inputs) {
        auto input = values.find(inputId);
        if (input == values.end()) {
          llvm::errs() << "bytecode call uses an undefined value\n";
          return EXIT_FAILURE;
        }
        auto valueResource = valueResources.find(inputId);
        if (valueResource != valueResources.end() &&
            !resources.contains(valueResource->second)) {
          llvm::errs() << "bytecode call reads a released resource\n";
          return EXIT_FAILURE;
        }
        arguments.push_back(input->second);
      }
      std::vector<Tensor> results;
      results.reserve(call->outputs.size());
      for (const BytecodeCallOutput &output : call->outputs) {
        if (values.contains(output.valueId)) {
          llvm::errs() << "bytecode value is defined twice\n";
          return EXIT_FAILURE;
        }
        auto resource = resources.find(output.resourceId);
        if (resource == resources.end() ||
            (resource->second.bytes != output.bytes && output.bytes != -1)) {
          llvm::errs() << "bytecode call references an invalid output resource\n";
          return EXIT_FAILURE;
        }
        std::vector<int64_t> resultShape =
            resolveResultShape(call->callee, output.shape, arguments);
        int64_t resolvedBytes = elementCount(resultShape) * sizeof(float);
        if (resource->second.bytes == -1) {
          resource->second.bytes = resolvedBytes;
          resource->second.storage = allocator.acquire(resolvedBytes);
          liveResourceBytes += resolvedBytes;
          peakResourceBytes = std::max(peakResourceBytes, liveResourceBytes);
        }
        if (resource->second.bytes != resolvedBytes) {
          llvm::errs() << "bytecode result shape does not match allocation\n";
          return EXIT_FAILURE;
        }
        results.push_back(
            {std::move(resultShape), TensorValues(resource->second.storage)});
      }
      if (nativeExecutable) {
        executeNativeKernel(*nativeExecutable, call->callee, arguments, results);
      } else {
        std::vector<Tensor> interpreted =
            executeKernelResults(call->callee, arguments);
        if (interpreted.size() != results.size()) {
          llvm::errs() << "interpreted kernel result count mismatch\n";
          return EXIT_FAILURE;
        }
        for (auto [source, destination] :
             llvm::zip_equal(interpreted, results)) {
          if (source.shape != destination.shape) {
            llvm::errs() << "interpreted kernel result shape mismatch\n";
            return EXIT_FAILURE;
          }
          std::copy(source.values.begin(), source.values.end(),
                    destination.values.begin());
        }
      }
      for (size_t index = 0; index < results.size(); ++index) {
        const BytecodeCallOutput &output = call->outputs[index];
        values[output.valueId] = std::move(results[index]);
        valueResources[output.valueId] = output.resourceId;
      }
      continue;
    }
    if (const auto *dealloc = std::get_if<BytecodeDealloc>(&instruction)) {
      auto resource = resources.find(dealloc->resourceId);
      if (resource == resources.end()) {
        llvm::errs() << "resource deallocated while not live: "
                     << dealloc->resourceId << "\n";
        return EXIT_FAILURE;
      }
      liveResourceBytes -= resource->second.bytes;
      allocator.release(resource->second);
      resources.erase(resource);
      continue;
    }
    const auto &returnOp = std::get<BytecodeReturn>(instruction);
    if (returnOp.values.empty()) {
      llvm::errs() << "bytecode requires at least one return value\n";
      return EXIT_FAILURE;
    }
    for (uint32_t outputId : returnOp.values) {
      auto outputResource = valueResources.find(outputId);
      if (!values.contains(outputId) || outputResource == valueResources.end() ||
          !resources.contains(outputResource->second)) {
        llvm::errs() << "returned tensor resource is not live\n";
        return EXIT_FAILURE;
      }
    }
    llvm::outs() << "backend: "
                 << (nativeExecutable ? "native-aot" : "interpreter") << "\n";
    llvm::outs() << "vm: tiny-bytecode-v1\n";
    llvm::outs() << "resources: peak=" << peakResourceBytes
                 << "B live=" << liveResourceBytes << "B\n";
    llvm::outs() << "allocator: new=" << allocator.getNewAllocations()
                 << " reused=" << allocator.getReusedAllocations() << "\n";
    for (auto [index, outputId] : llvm::enumerate(returnOp.values)) {
      printTensorResult(values.find(outputId)->second,
                        returnOp.values.size() == 1
                            ? std::optional<size_t>()
                            : std::optional<size_t>(index));
    }
    return EXIT_SUCCESS;
  }

  llvm::errs() << "bytecode function did not return\n";
  return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "tiny-iree VM runtime\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::tiree::registerTinyIREEDialects(registry);
  mlir::MLIRContext context(registry);

  std::filesystem::path modulePath(inputFilename.getValue());
  std::filesystem::path executablePath(executableFilename.getValue());
  if (std::filesystem::is_directory(modulePath)) {
    std::filesystem::path bundlePath = modulePath;
    if (!resolveBundle(bundlePath, modulePath, executablePath)) {
      return EXIT_FAILURE;
    }
  }

  NativeExecutable nativeExecutable;
  NativeExecutable *nativeExecutablePtr = nullptr;
  if (!executablePath.empty()) {
    if (!nativeExecutable.load(executablePath.string())) return EXIT_FAILURE;
    nativeExecutablePtr = &nativeExecutable;
  }

  auto file = llvm::MemoryBuffer::getFileOrSTDIN(modulePath.string());
  if (!file) {
    llvm::errs() << "unable to read module: " << modulePath.string() << "\n";
    return EXIT_FAILURE;
  }
  llvm::StringRef fileData = (*file)->getBuffer();
  llvm::ArrayRef<uint8_t> byteData(
      reinterpret_cast<const uint8_t *>(fileData.data()), fileData.size());
  if (byteData.size() >= bytecode::kMagic.size() &&
      std::equal(bytecode::kMagic.begin(), bytecode::kMagic.end(),
                 reinterpret_cast<const char *>(byteData.data()))) {
    std::vector<BytecodeFunction> functions;
    if (!parseBytecode(byteData, functions)) return EXIT_FAILURE;
    return runBytecode(functions, nativeExecutablePtr);
  }

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module) return EXIT_FAILURE;
  return run(*module, nativeExecutablePtr);
}
