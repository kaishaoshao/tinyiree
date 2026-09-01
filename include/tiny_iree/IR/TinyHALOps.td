include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def TinyHAL_Dialect : Dialect {
  let name = "tiree_hal";
  let cppNamespace = "::mlir::tiree::HAL";
  let summary = "Tiny IREE hardware abstraction dialect";
  let useDefaultTypePrinterParser = 1;
}

class TinyHAL_TypeDef<string name> : TypeDef<TinyHAL_Dialect, name>;
def TinyHAL_BufferType : TinyHAL_TypeDef<"Buffer"> {
  let mnemonic = "buffer";
  let summary = "A device-visible HAL buffer";
}

class TinyHAL_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyHAL_Dialect, mnemonic, traits>;

def TinyHAL_ExecutableOp : TinyHAL_Op<"executable", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kernel,
                       TypeAttr:$function_type, ArrayAttr:$workload);
  let results = (outs);
  let hasVerifier = 1;
}

def TinyHAL_AllocOp : TinyHAL_Op<"alloc"> {
  let arguments = (ins I64Attr:$resource_id, I64Attr:$bytes);
  let results = (outs TinyHAL_BufferType:$buffer);
  let hasVerifier = 1;
}

def TinyHAL_DispatchOp
    : TinyHAL_Op<"dispatch", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyTensor>:$inputs,
                       Variadic<TinyHAL_BufferType>:$output_buffers,
                       StrAttr:$entry_point, StrAttr:$device,
                       DenseI64ArrayAttr:$result_bytes,
                       DenseI64ArrayAttr:$result_resources);
  let results = (outs Variadic<AnyTensor>:$outputs);
  let hasVerifier = 1;
}

def TinyHAL_DeallocOp : TinyHAL_Op<"dealloc"> {
  let arguments = (ins TinyHAL_BufferType:$buffer, I64Attr:$resource_id);
  let results = (outs);
  let hasVerifier = 1;
}
