include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TinyStream_Dialect : Dialect {
  let name = "tiree_stream";
  let cppNamespace = "::mlir::tiree::Stream";
  let summary = "Tiny IREE resource scheduling dialect";
  let useDefaultTypePrinterParser = 1;
}

class TinyStream_TypeDef<string name> : TypeDef<TinyStream_Dialect, name>;
def TinyStream_ResourceType : TinyStream_TypeDef<"Resource"> {
  let mnemonic = "resource";
  let summary = "An SSA-managed stream resource";
}

class TinyStream_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyStream_Dialect, mnemonic, traits>;

def TinyStream_AllocOp : TinyStream_Op<"alloc"> {
  let arguments = (ins I64Attr:$resource_id, I64Attr:$bytes);
  let results = (outs TinyStream_ResourceType:$resource);
  let hasVerifier = 1;
}

def TinyStream_DispatchOp
    : TinyStream_Op<"dispatch", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyTensor>:$inputs,
                       Variadic<TinyStream_ResourceType>:$output_resources,
                       StrAttr:$kernel, StrAttr:$entry_point,
                       ArrayAttr:$workload,
                       DenseI64ArrayAttr:$result_bytes,
                       DenseI64ArrayAttr:$result_resources);
  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasVerifier = 1;
}

def TinyStream_DeallocOp : TinyStream_Op<"dealloc"> {
  let arguments = (ins TinyStream_ResourceType:$resource,
                       I64Attr:$resource_id);
  let results = (outs);
  let hasVerifier = 1;
}
