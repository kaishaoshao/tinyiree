module {
  func.func @invalid(%input: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %0 = "tiree_flow.dispatch"(%input) <{kernel = "relu"}> ({
    ^bb0(%arg0: tensor<1x3xf32>):
      %1 = "tiree_input.softmax"(%arg0) {axis = -1 : i64}
          : (tensor<1x3xf32>) -> tensor<1x3xf32>
      "tiree_flow.yield"(%1) : (tensor<1x3xf32>) -> ()
    }) : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
