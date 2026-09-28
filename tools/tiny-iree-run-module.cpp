#include "tiny_iree/Runtime/Runtime.h"

int main(int argc, char **argv) {
  return mlir::tiree::runtime::runModuleMain(argc, argv);
}
