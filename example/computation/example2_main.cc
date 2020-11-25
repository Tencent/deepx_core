// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <iostream>
#include <vector>

namespace deepx_core {
namespace {

class Main : public DataType {
 public:
  static int main() {
    Graph graph;
    TensorMap param;

    // Initialize graph: Z = X * W + B.
    InstanceNode X("X", Shape(1), TENSOR_TYPE_TSR);
    VariableNode W("W", Shape(1), TENSOR_TYPE_TSR);
    VariableNode B("B", Shape(1), TENSOR_TYPE_TSR);
    MulNode XW("XW", &X, &W);
    AddNode Z("Z", &XW, &B);
    DXCHECK_THROW(graph.Compile({&Z}, 0));

    // Initialize param.
    auto& _W = param.insert<tsr_t>(W.name());
    _W.resize(W.shape());
    _W.data(0) = 2;
    auto& _B = param.insert<tsr_t>(B.name());
    _B.resize(B.shape());
    _B.data(0) = 3;

    // Initialize op context.
    OpContext op_context;
    op_context.Init(&graph, &param);
    DXCHECK_THROW(op_context.InitOp(std::vector<int>{0}, -1));
    auto& _X = op_context.mutable_inst()->insert<tsr_t>(X.name());
    _X.resize(X.shape());
    op_context.InitForward();

    // Input, forward, output.
    auto compute = [&op_context, &_X, &Z](float_t x) {
      _X.data(0) = x;
      op_context.Forward();
      const auto& _Z = op_context.hidden().get<tsr_t>(Z.name());
      float_t z = _Z.data(0);
      std::cout << "Z=" << z << std::endl;
    };
    compute(1);
    compute(2);
    compute(3);
    compute(10);
    return 0;
  }
};

}  // namespace
}  // namespace deepx_core

int main() { return deepx_core::Main::main(); }
