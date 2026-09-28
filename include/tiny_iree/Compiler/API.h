#ifndef TINY_IREE_COMPILER_API_H_
#define TINY_IREE_COMPILER_API_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TINY_IREE_COMPILER_API_VERSION 1

typedef enum tiny_iree_compiler_status_e {
  TINY_IREE_COMPILER_STATUS_OK = 0,
  TINY_IREE_COMPILER_STATUS_INVALID_ARGUMENT = 1,
  TINY_IREE_COMPILER_STATUS_PARSE_ERROR = 2,
  TINY_IREE_COMPILER_STATUS_VERIFICATION_ERROR = 3,
} tiny_iree_compiler_status_t;

// Returns a static, NUL-terminated identifier for this C ABI.
const char *tiny_iree_compiler_api_version(void);

// Parses and verifies an MLIR module using the Tiny-IREE dialect registry.
// |mlir_text| must be NUL-terminated. |diagnostic_buffer| is optional and,
// when supplied, receives a NUL-terminated diagnostic truncated to its size.
tiny_iree_compiler_status_t tiny_iree_compiler_validate_mlir(
    const char *mlir_text, char *diagnostic_buffer,
    size_t diagnostic_buffer_size);

#ifdef __cplusplus
}
#endif

#endif  // TINY_IREE_COMPILER_API_H_
