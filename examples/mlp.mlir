module {
  func.func @predict(%input: tensor<1x4xf32>,
                     %weight: tensor<4x3xf32>,
                     %bias: tensor<3xf32>) -> tensor<1x3xf32> {
    %0 = "tiree_input.matmul"(%input, %weight)
        : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = "tiree_input.add"(%0, %bias)
        : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    %2 = "tiree_input.relu"(%1)
        : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
}

