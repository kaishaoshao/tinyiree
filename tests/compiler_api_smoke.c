#include <stdio.h>
#include <string.h>

#include "tiny_iree/Compiler/API.h"

int main(void) {
  static const char kValidModule[] =
      "module {\n"
      "  func.func @identity(%input: tensor<1xf32>) -> tensor<1xf32> {\n"
      "    %0 = \"tiree_input.relu\"(%input) "
      ": (tensor<1xf32>) -> tensor<1xf32>\n"
      "    return %0 : tensor<1xf32>\n"
      "  }\n"
      "}\n";
  char diagnostic[512] = {0};

  if (strlen(tiny_iree_compiler_api_version()) == 0) {
    fprintf(stderr, "compiler API version is empty\n");
    return 1;
  }
  if (tiny_iree_compiler_validate_mlir(kValidModule, diagnostic,
                                       sizeof(diagnostic)) !=
      TINY_IREE_COMPILER_STATUS_OK) {
    fprintf(stderr, "valid module rejected: %s\n", diagnostic);
    return 1;
  }
  if (tiny_iree_compiler_validate_mlir("not MLIR", diagnostic,
                                       sizeof(diagnostic)) ==
      TINY_IREE_COMPILER_STATUS_OK ||
      diagnostic[0] == '\0') {
    fprintf(stderr, "invalid module was not diagnosed\n");
    return 1;
  }
  puts("compiler C API: ok");
  return 0;
}
