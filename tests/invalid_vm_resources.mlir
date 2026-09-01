module {
  func.func @predict(%input: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %ref = "tiree_vm.alloc"() <{bytes = 8 : i64, resource_id = 7 : i64}> : () -> !tiree_vm.ref
    %0 = "tiree_vm.call"(%input, %ref) <{
      callee = "relu",
      device = "cpu-sync",
      operandSegmentSizes = array<i32: 1, 1>,
      result_bytes = array<i64: 12>,
      result_resources = array<i64: 7>
    }> : (tensor<1x3xf32>, !tiree_vm.ref) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
