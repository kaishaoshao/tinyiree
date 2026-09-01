include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TinyFlow_Dialect : Dialect {
  let name = "tiree_flow";
  let cppNamespace = "::mlir::tiree::Flow";
  let summary = "Tiny IREE dispatch formation dialect";
}

class TinyFlow_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyFlow_Dialect, mnemonic, traits>;

def TinyFlow_DispatchOp : TinyFlow_Op<"dispatch", [Pure, IsolatedFromAbove]> {
  let summary = "A fused tensor workload that can be independently dispatched";
  let arguments = (ins Variadic<AnyTensor>:$inputs, StrAttr:$kernel);
  let results = (outs Variadic<AnyTensor>:$outputs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def TinyFlow_YieldOp : TinyFlow_Op<"yield", [Pure, Terminator]> {
  let summary = "Return dispatch results from a flow workload region";
  let arguments = (ins Variadic<AnyTensor>:$values);
  let results = (outs);
}
