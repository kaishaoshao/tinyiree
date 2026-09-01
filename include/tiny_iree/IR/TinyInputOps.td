include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TinyInput_Dialect : Dialect {
  let name = "tiree_input";
  let cppNamespace = "::mlir::tiree::Input";
  let summary = "Tiny IREE model input dialect";
}

class TinyInput_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyInput_Dialect, mnemonic, traits>;

def TinyInput_MatMulOp : TinyInput_Op<"matmul", [Pure]> {
  let summary = "Rank-2 floating point matrix multiplication";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_FusedMatMulAddReluOp
    : TinyInput_Op<"fused_matmul_add_relu", [Pure]> {
  let summary = "Fused matrix multiplication, bias add, and ReLU";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_AddOp : TinyInput_Op<"add", [Pure]> {
  let summary = "Elementwise add with optional rank-1 bias broadcast";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_ReluOp : TinyInput_Op<"relu", [Pure]> {
  let summary = "Elementwise rectified linear unit";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_SoftmaxOp : TinyInput_Op<"softmax", [Pure]> {
  let summary = "Softmax over the final tensor dimension";
  let arguments = (ins AnyTensor:$input, I64Attr:$axis);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

