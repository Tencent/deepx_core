// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>
#include <cmath>
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

    // Initialize graph: self attention.
    int m = 2, k = 3, n = 4;
    InstanceNode X("X", Shape(-1, m, k), TENSOR_TYPE_TSR);
    VariableNode Wq("Wq", Shape(k, n), TENSOR_TYPE_TSR);
    VariableNode Wk("Wk", Shape(k, n), TENSOR_TYPE_TSR);
    VariableNode Wv("Wv", Shape(k, n), TENSOR_TYPE_TSR);
    ConstantNode C("C", Shape(1), 1 / std::sqrt(1.0 * n));
    MatmulNode Q("Q", &X, &Wq);
    MatmulNode K("K", &X, &Wk);
    MatmulNode V("V", &X, &Wv);
    BatchGEMMNode Z1("Z1", &Q, &K, 0, 1);
    BroadcastMulNode Z2("Z2", &Z1, &C);
    SoftmaxNode Z3("Z3", &Z2, -1);
    BatchGEMMNode Z4("Z4", &Z3, &V, 0, 0);
    DXCHECK_THROW(graph.Compile({&Z4}, 0));

    // Initialize param.
    auto& _Wq = param.insert<tsr_t>(Wq.name());
    _Wq.resize(Wq.shape());
    _Wq.randn(engine);
    auto& _Wk = param.insert<tsr_t>(Wk.name());
    _Wk.resize(Wk.shape());
    _Wk.randn(engine);
    auto& _Wv = param.insert<tsr_t>(Wv.name());
    _Wv.resize(Wv.shape());
    _Wv.randn(engine);

    // Initialize op context.
    OpContext op_context;
    op_context.Init(&graph, &param);
    DXCHECK_THROW(op_context.InitOp(std::vector<int>{0}, -1));

    // Input, forward, output.
    for (int i = 0; i < 3; ++i) {
      auto& _X = op_context.mutable_inst()->insert<tsr_t>(X.name());
      _X.resize(2 + i, m, k);
      _X.randn(engine);
      op_context.InitForward();
      op_context.Forward();
      const auto& _Z4 = op_context.hidden().get<tsr_t>(Z4.name());
      std::cout << "Z4=" << _Z4 << std::endl;
    }
    return 0;
  }
};

}  // namespace
}  // namespace deepx_core

int main() { return deepx_core::Main::main(); }
