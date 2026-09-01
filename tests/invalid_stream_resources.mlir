module {
  func.func @invalid() {
    %resource = "tiree_stream.alloc"()
        <{bytes = 4 : i64, resource_id = 7 : i64}>
        : () -> !tiree_stream.resource
    "tiree_stream.dealloc"(%resource) <{resource_id = 7 : i64}>
        : (!tiree_stream.resource) -> ()
    "tiree_stream.dealloc"(%resource) <{resource_id = 7 : i64}>
        : (!tiree_stream.resource) -> ()
    return
  }
}
