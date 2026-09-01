// Official IREE reference for the deterministic tiny-iree MLP.
module {
  func.func @predict(%input: tensor<1x4xf32>) -> tensor<1x3xf32>
      attributes {iree.module.export} {
    %weight = stablehlo.constant dense<[
      [0.1, -0.2, 0.3],
      [0.4, 0.5, -0.6],
      [-0.7, 0.8, 0.9],
      [1.0, -1.1, 1.2]
    ]> : tensor<4x3xf32>
    %bias = stablehlo.constant dense<[0.1, 0.2, -0.1]> : tensor<3xf32>
    %zero_matrix = stablehlo.constant dense<0.0> : tensor<1x3xf32>
    %negative_infinity = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %matmul = stablehlo.dot %input, %weight
        : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %broadcast_bias = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<3xf32>) -> tensor<1x3xf32>
    %biased = stablehlo.add %matmul, %broadcast_bias : tensor<1x3xf32>
    %activated = stablehlo.maximum %biased, %zero_matrix : tensor<1x3xf32>
    %maximum = stablehlo.reduce(%activated init: %negative_infinity)
        across dimensions = [1]
        : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
        %value = stablehlo.maximum %lhs, %rhs : tensor<f32>
        stablehlo.return %value : tensor<f32>
      }
    %broadcast_maximum = stablehlo.broadcast_in_dim %maximum, dims = [0]
        : (tensor<1xf32>) -> tensor<1x3xf32>
    %shifted = stablehlo.subtract %activated, %broadcast_maximum
        : tensor<1x3xf32>
    %exponential = stablehlo.exponential %shifted : tensor<1x3xf32>
    %sum = stablehlo.reduce(%exponential init: %zero) across dimensions = [1]
        : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
        %value = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %value : tensor<f32>
      }
    %broadcast_sum = stablehlo.broadcast_in_dim %sum, dims = [0]
        : (tensor<1xf32>) -> tensor<1x3xf32>
    %probabilities = stablehlo.divide %exponential, %broadcast_sum
        : tensor<1x3xf32>
    return %probabilities : tensor<1x3xf32>
  }
}
