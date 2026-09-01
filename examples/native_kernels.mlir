module {
  func.func @tiree_kernel_matmul_add_relu(
      %lhs: memref<1x4xf32>, %rhs: memref<4x3xf32>,
      %bias: memref<3xf32>, %output: memref<1x3xf32>)
      attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%output : memref<1x3xf32>)
    linalg.matmul ins(%lhs, %rhs : memref<1x4xf32>, memref<4x3xf32>)
                  outs(%output : memref<1x3xf32>)
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%output, %bias : memref<1x3xf32>, memref<3xf32>)
      outs(%output : memref<1x3xf32>) {
    ^bb0(%value: f32, %bias_value: f32, %unused: f32):
      %sum = arith.addf %value, %bias_value : f32
      %activated = arith.maximumf %sum, %zero : f32
      linalg.yield %activated : f32
    }
    return
  }

  func.func @tiree_kernel_softmax(
      %input: memref<1x3xf32>, %output: memref<1x3xf32>)
      attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %negative_infinity = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    %maximum = scf.for %column = %c0 to %c3 step %c1
        iter_args(%current = %negative_infinity) -> f32 {
      %value = memref.load %input[%c0, %column] : memref<1x3xf32>
      %next = arith.maximumf %current, %value : f32
      scf.yield %next : f32
    }
    %sum = scf.for %column = %c0 to %c3 step %c1
        iter_args(%current = %zero) -> f32 {
      %value = memref.load %input[%c0, %column] : memref<1x3xf32>
      %shifted = arith.subf %value, %maximum : f32
      %exponential = math.exp %shifted : f32
      memref.store %exponential, %output[%c0, %column] : memref<1x3xf32>
      %next = arith.addf %current, %exponential : f32
      scf.yield %next : f32
    }
    scf.for %column = %c0 to %c3 step %c1 {
      %value = memref.load %output[%c0, %column] : memref<1x3xf32>
      %normalized = arith.divf %value, %sum : f32
      memref.store %normalized, %output[%c0, %column] : memref<1x3xf32>
    }
    return
  }
}
