module {
  func.func @bad(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = "tiree_flow.dispatch"(%arg0) <{kernel = "relu"}> ({
    ^bb0(%input: tensor<1x4xf32>):
      "tiree_flow.yield"() : () -> ()
    }) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}
