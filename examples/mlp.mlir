module {
  func.func @predict(%input: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %weight = arith.constant dense<[
      [0.1, -0.2, 0.3],
      [0.4, 0.5, -0.6],
      [-0.7, 0.8, 0.9],
      [1.0, -1.1, 1.2]
    ]> : tensor<4x3xf32>
    %bias = arith.constant dense<[0.1, 0.2, -0.1]> : tensor<3xf32>
    %0 = "tiree_input.matmul"(%input, %weight)
        : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = "tiree_input.add"(%0, %bias)
        : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    %2 = "tiree_input.relu"(%1)
        : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %3 = "tiree_input.softmax"(%2) {axis = 1 : i64}
        : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }
}
