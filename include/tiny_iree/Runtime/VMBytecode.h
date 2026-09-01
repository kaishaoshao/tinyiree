#ifndef TINY_IREE_RUNTIME_VM_BYTECODE_H_
#define TINY_IREE_RUNTIME_VM_BYTECODE_H_

#include <array>
#include <cstdint>

namespace mlir::tiree::vm_bytecode {

inline constexpr std::array<char, 8> kMagic = {'T', 'I', 'R', 'E',
                                               'V', 'M', '1', '\0'};
inline constexpr uint32_t kVersion = 1;
inline constexpr uint32_t kF32ElementType = 1;

enum class Opcode : uint8_t {
  Constant = 1,
  Alloc = 2,
  Call = 3,
  Dealloc = 4,
  Return = 5,
};

}  // namespace mlir::tiree::vm_bytecode

#endif  // TINY_IREE_RUNTIME_VM_BYTECODE_H_
