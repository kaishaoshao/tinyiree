module {
  func.func @bad(%lhs: tensor<2x3xf32>, %rhs: tensor<4x2xf32>)
      -> tensor<2x2xf32> {
    %0 = "tiree_input.matmul"(%lhs, %rhs)
        : (tensor<2x3xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

