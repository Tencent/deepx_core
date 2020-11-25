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
#include <random>
#include <vector>

namespace deepx_core {
namespace {

class Main : public DataType {
 public:
  static int main() {
    std::default_random_engine engine;

    Graph graph;
    TensorMap param;

    // Initialize graph: Z = X * W + B.
    InstanceNode X("X", Shape(-1, 10), TENSOR_TYPE_TSR);
    VariableNode W("W", Shape(10, 1), TENSOR_TYPE_TSR);
    VariableNode B("B", Shape(1), TENSOR_TYPE_TSR);
    MatmulNode XW("XW", &X, &W);
    BroadcastAddNode Z("Z", &XW, &B);
    DXCHECK_THROW(graph.Compile({&XW, &Z}, 0));

    // Initialize param.
    auto& _W = param.insert<tsr_t>(W.name());
    _W.resize(W.shape());
    _W.randn(engine);
    auto& _B = param.insert<tsr_t>(B.name());
    _B.resize(B.shape());
    _B.randn(engine);

    // Initialize op context.
    OpContext op_context;
    op_context.Init(&graph, &param);
    DXCHECK_THROW(op_context.InitOp(std::vector<int>{0, 1}, -1));

    // Input, forward, output.
    for (int i = 0; i < 3; ++i) {
      auto& _X = op_context.mutable_inst()->insert<tsr_t>(X.name());
      _X.resize(2 + i, 10);
      _X.randn(engine);
      op_context.InitForward();
      op_context.Forward();
      const auto& _XW = op_context.hidden().get<tsr_t>(XW.name());
      const auto& _Z = op_context.hidden().get<tsr_t>(Z.name());
      std::cout << "XW=" << _XW << std::endl;
      std::cout << "Z=" << _Z << std::endl;
    }
    return 0;
  }
};

}  // namespace
}  // namespace deepx_core

int main() { return deepx_core::Main::main(); }
