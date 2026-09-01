#ifndef TINY_IREE_RUNTIME_NATIVE_ABI_H_
#define TINY_IREE_RUNTIME_NATIVE_ABI_H_

#include <stdint.h>

#if defined(_WIN32)
#define TINY_IREE_EXPORT __declspec(dllexport)
#else
#define TINY_IREE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { TINY_IREE_MAX_RANK = 4 };

typedef struct tiree_tensor_view_t {
  float *data;
  int64_t rank;
  int64_t dims[TINY_IREE_MAX_RANK];
} tiree_tensor_view_t;

typedef int (*tiree_kernel_fn_t)(const tiree_tensor_view_t *inputs,
                                 int64_t input_count,
                                 tiree_tensor_view_t *output);

#ifdef __cplusplus
}
#endif

#endif  // TINY_IREE_RUNTIME_NATIVE_ABI_H_
