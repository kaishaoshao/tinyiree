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
  let summary = "Globally fused matrix multiplication, bias add, and ReLU";
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

def TinyInput_FakeQuantOp : TinyInput_Op<"fake_quant", [Pure]> {
  let summary = "Per-tensor int8 quantize/dequantize simulation";
  let arguments = (ins AnyTensor:$input, AnyTensor:$scale,
                       AnyTensor:$zero_point);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_SplitOp : TinyInput_Op<"split", [Pure]> {
  let summary = "Split a tensor into two equal tensors along its final axis";
  let arguments = (ins AnyTensor:$input, I64Attr:$axis);
  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasVerifier = 1;
}

def TinyInput_Conv2DOp : TinyInput_Op<"conv2d", [Pure]> {
  let summary = "Static NCHW valid stride-one convolution with bias";
  let arguments = (ins AnyTensor:$input, AnyTensor:$weight, AnyTensor:$bias,
                       DenseI64ArrayAttr:$strides, DenseI64ArrayAttr:$pads);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_MaxPool2DOp : TinyInput_Op<"max_pool2d", [Pure]> {
  let summary = "Static NCHW two-by-two stride-two maximum pooling";
  let arguments = (ins AnyTensor:$input,
                       DenseI64ArrayAttr:$kernel_shape,
                       DenseI64ArrayAttr:$strides);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_ReshapeOp : TinyInput_Op<"reshape", [Pure]> {
  let summary = "Static element-count-preserving reshape";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_TransposeOp : TinyInput_Op<"transpose", [Pure]> {
  let summary = "Static rank-four NCHW to NHWC transpose";
  let arguments = (ins AnyTensor:$input, DenseI64ArrayAttr:$permutation);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}

def TinyInput_SoftmaxOp : TinyInput_Op<"softmax", [Pure]> {
  let summary = "Softmax over the final tensor dimension";
  let arguments = (ins AnyTensor:$input, I64Attr:$axis);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
}
