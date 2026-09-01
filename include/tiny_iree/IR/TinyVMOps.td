include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TinyVM_Dialect : Dialect {
  let name = "tiree_vm";
  let cppNamespace = "::mlir::tiree::VM";
  let summary = "Tiny IREE runtime control dialect";
  let useDefaultTypePrinterParser = 1;
}

class TinyVM_TypeDef<string name> : TypeDef<TinyVM_Dialect, name>;
def TinyVM_RefType : TinyVM_TypeDef<"Ref"> {
  let mnemonic = "ref";
  let summary = "A reference-counted runtime resource";
}

class TinyVM_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyVM_Dialect, mnemonic, traits>;

def TinyVM_AllocOp : TinyVM_Op<"alloc"> {
  let arguments = (ins I64Attr:$resource_id, I64Attr:$bytes);
  let results = (outs TinyVM_RefType:$ref);
  let hasVerifier = 1;
}

def TinyVM_CallOp : TinyVM_Op<"call", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyTensor>:$inputs,
                       Variadic<TinyVM_RefType>:$output_refs,
                       StrAttr:$callee, StrAttr:$device,
                       DenseI64ArrayAttr:$result_bytes,
                       DenseI64ArrayAttr:$result_resources);
  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasVerifier = 1;
}

def TinyVM_DeallocOp : TinyVM_Op<"dealloc"> {
  let arguments = (ins TinyVM_RefType:$ref, I64Attr:$resource_id);
  let results = (outs);
  let hasVerifier = 1;
}
