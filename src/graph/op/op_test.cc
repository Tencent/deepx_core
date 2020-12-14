// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "op_test.h"
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/op_context.h>
#include <memory>
#include <string>
#include <type_traits>  // std::is_same

namespace deepx_core {

/************************************************************************/
/* OpTestDataType */
/************************************************************************/
class OpTestDataType : public DataType {
 public:
  static constexpr float_t CHECK_FORWARD_EPS =
      std::is_same<float_t, double>::value ? (float_t)1e-6 : (float_t)1e-3;
  static constexpr float_t NUMERICAL_GRAD_EPS =
      std::is_same<float_t, double>::value ? (float_t)1e-6 : (float_t)1e-3;
  static constexpr float_t CHECK_GRAD_EPS =
      std::is_same<float_t, double>::value ? (float_t)1e-4 : (float_t)5e-2;
  static constexpr int FORWARD_SEED = 9527;  // magic number
  static const std::string LOSS_NAME;
};

const std::string OpTestDataType::LOSS_NAME = "op_test_loss";

/************************************************************************/
/* OpTestContext */
/************************************************************************/
class OpTestContext : public OpTestDataType {
 private:
  std::default_random_engine engine_;
  std::unique_ptr<GraphNode> loss_node_;
  Graph graph_;
  TensorMap param_;
  OpContext closed_;
  OpContext numerical_;

 private:
  void InitGraph(GraphNode* node, int on_heap, int expected_forward);
  void InitParam(const param_initializer_t& pre_param_initializer,
                 const param_initializer_t& post_param_initializer);
  void InitOpContext(const inst_initializer_t& inst_initializer,
                     OpContext* op_context);
  void InitClosedOpContext(const inst_initializer_t& inst_initializer);
  void InitNumericalOpContext(const inst_initializer_t& inst_initializer);
  void ClosedForward();
  void ClosedBackward();
  void ComputeNumericalGrad();
  void ComputeNumericalGrad(tsr_t* W, tsr_t* G);
  void ComputeNumericalGrad(tsr_t* W, srm_t* G);
  void ComputeNumericalGrad(srm_t* W, srm_t* G);
  void CompareGrad();

 public:
  void CheckForward(GraphNode* node, int on_heap,
                    const DataType::tsr_t& expected_forward,
                    const param_initializer_t& pre_param_initializer,
                    const param_initializer_t& post_param_initializer,
                    const inst_initializer_t& inst_initializer);
  void CheckBackward(GraphNode* node, int on_heap,
                     const param_initializer_t& pre_param_initializer,
                     const param_initializer_t& post_param_initializer,
                     const inst_initializer_t& inst_initializer);
};

void OpTestContext::InitGraph(GraphNode* node, int on_heap,
                              int expected_forward) {
  std::vector<GraphNode*> target_nodes;
  loss_node_.reset(new ReduceMeanNode(LOSS_NAME, node));
  if (on_heap) {
    target_nodes.emplace_back(loss_node_.release());
  } else {
    target_nodes.emplace_back(loss_node_.get());
  }
  if (expected_forward) {
    target_nodes.emplace_back(node);
  }
  ASSERT_TRUE(graph_.Compile(target_nodes, on_heap));
}

void OpTestContext::InitParam(
    const param_initializer_t& pre_param_initializer,
    const param_initializer_t& post_param_initializer) {
  if (pre_param_initializer) {
    pre_param_initializer(engine_, &param_);
  }

  for (const auto& entry : graph_.name_2_node()) {
    const GraphNode* node = entry.second;
    if (node->node_type() != GRAPH_NODE_TYPE_PARAM) {
      continue;
    }

    auto it = param_.find(node->name());
    if (it != param_.end()) {
      continue;
    }

    switch (node->tensor_type()) {
      case TENSOR_TYPE_TSR: {
        auto& W = param_.insert<tsr_t>(node->name());
        W.resize(node->shape());
        W.rand_init(engine_, node->initializer_type(),
                    (float_t)node->initializer_param1(),
                    (float_t)node->initializer_param2());
      } break;
      case TENSOR_TYPE_SRM: {
        auto& W = param_.insert<srm_t>(node->name());
        W.set_col(node->shape()[1]);
        W.set_initializer(node->initializer_type(),
                          (float_t)node->initializer_param1(),
                          (float_t)node->initializer_param2());
      } break;
    }
  }

  if (post_param_initializer) {
    post_param_initializer(engine_, &param_);
  }
}

void OpTestContext::InitOpContext(const inst_initializer_t& inst_initializer,
                                  OpContext* op_context) {
  if (inst_initializer) {
    inst_initializer(op_context->mutable_hidden()->mutable_inst());
  }
  op_context->Init(&graph_, &param_);
  ASSERT_TRUE(op_context->InitOp(std::vector<int>{0}, 0));
  op_context->InitForward();
  op_context->InitBackward();
}

void OpTestContext::InitClosedOpContext(
    const inst_initializer_t& inst_initializer) {
  InitOpContext(inst_initializer, &closed_);
}

void OpTestContext::InitNumericalOpContext(
    const inst_initializer_t& inst_initializer) {
  InitOpContext(inst_initializer, &numerical_);
}

void OpTestContext::ClosedForward() {
  closed_.mutable_hidden()->seed(FORWARD_SEED);
  closed_.Forward();
}

void OpTestContext::ClosedBackward() { closed_.Backward(); }

void OpTestContext::ComputeNumericalGrad() {
  TensorMap* grad = numerical_.mutable_grad();
  for (auto& entry : param_) {
    const std::string& name = entry.first;
    auto it = grad->find(name);
    if (it == grad->end()) {
      continue;
    }

    Any& Wany = entry.second;
    Any& Gany = it->second;
    if (Wany.is<tsr_t>()) {
      auto& W = Wany.unsafe_to_ref<tsr_t>();
      if (Gany.is<tsr_t>()) {
        auto& G = Gany.unsafe_to_ref<tsr_t>();
        ComputeNumericalGrad(&W, &G);
      } else if (Gany.is<srm_t>()) {
        auto& G = Gany.unsafe_to_ref<srm_t>();
        ComputeNumericalGrad(&W, &G);
      }
    } else if (Wany.is<srm_t>()) {
      auto& W = Wany.unsafe_to_ref<srm_t>();
      if (Gany.is<srm_t>()) {
        auto& G = Gany.unsafe_to_ref<srm_t>();
        ComputeNumericalGrad(&W, &G);
      }
    }
  }
}

void OpTestContext::ComputeNumericalGrad(tsr_t* W, tsr_t* G) {
  float_t w, loss1, loss2;
  float_t* _W = W->data();

  for (int i = 0; i < W->total_dim(); ++i) {
    w = _W[i];

    _W[i] = w + NUMERICAL_GRAD_EPS;
    numerical_.mutable_hidden()->seed(FORWARD_SEED);
    numerical_.Forward();
    loss1 = numerical_.loss();

    _W[i] = w - NUMERICAL_GRAD_EPS;
    numerical_.mutable_hidden()->seed(FORWARD_SEED);
    numerical_.Forward();
    loss2 = numerical_.loss();

    _W[i] = w;

    G->data(i) = (loss1 - loss2) / NUMERICAL_GRAD_EPS / 2;
  }
}

void OpTestContext::ComputeNumericalGrad(tsr_t* W, srm_t* G) {
  ASSERT_TRUE(W->is_rank(2));
  ASSERT_TRUE(W->dim(1) == G->col());
  float_t w, loss1, loss2, g;
  float_t* Wi;

  for (int i = 0; i < W->dim(0); ++i) {
    Wi = W->data() + W->dim(1) * i;
    for (int j = 0; j < W->dim(1); ++j) {
      w = Wi[j];

      Wi[j] = w + NUMERICAL_GRAD_EPS;
      numerical_.mutable_hidden()->seed(FORWARD_SEED);
      numerical_.Forward();
      loss1 = numerical_.loss();

      Wi[j] = w - NUMERICAL_GRAD_EPS;
      numerical_.mutable_hidden()->seed(FORWARD_SEED);
      numerical_.Forward();
      loss2 = numerical_.loss();

      Wi[j] = w;

      g = (loss1 - loss2) / NUMERICAL_GRAD_EPS / 2;
      if (g > NUMERICAL_GRAD_EPS || g < -NUMERICAL_GRAD_EPS) {
        G->get_row_no_init(i)[j] = g;
      }
    }
  }
}

void OpTestContext::ComputeNumericalGrad(srm_t* W, srm_t* G) {
  ASSERT_TRUE(W->col() == G->col());
  float_t w, loss1, loss2, g;

  for (auto& entry : *W) {
    int_t k = entry.first;
    float_t* v = entry.second;
    for (int j = 0; j < W->col(); ++j) {
      w = v[j];

      v[j] = w + NUMERICAL_GRAD_EPS;
      numerical_.mutable_hidden()->seed(FORWARD_SEED);
      numerical_.Forward();
      loss1 = numerical_.loss();

      v[j] = w - NUMERICAL_GRAD_EPS;
      numerical_.mutable_hidden()->seed(FORWARD_SEED);
      numerical_.Forward();
      loss2 = numerical_.loss();

      v[j] = w;

      g = (loss1 - loss2) / NUMERICAL_GRAD_EPS / 2;
      if (g > NUMERICAL_GRAD_EPS || g < -NUMERICAL_GRAD_EPS) {
        G->get_row_no_init(k)[j] = g;
      }
    }
  }
}

void OpTestContext::CompareGrad() {
  const TensorMap& closed_grad = closed_.grad();
  const TensorMap& numerical_grad = numerical_.grad();

  for (const auto& entry : param_) {
    const std::string& name = entry.first;
    auto it1 = closed_grad.find(name);
    auto it2 = numerical_grad.find(name);
    if (it1 == closed_grad.end() && it2 == numerical_grad.end()) {
      continue;
    }

    ASSERT_TRUE(it1 != closed_grad.end() && it2 != numerical_grad.end());
    const Any& Gany1 = it1->second;
    const Any& Gany2 = it2->second;
    if (Gany1.is<tsr_t>()) {
      const auto& G1 = Gany1.unsafe_to_ref<tsr_t>();
      const auto& G2 = Gany2.to_ref<tsr_t>();
      EXPECT_TSR_NEAR_EPS(G1, G2, CHECK_GRAD_EPS);
    } else if (Gany1.is<srm_t>()) {
      const auto& G1 = Gany1.unsafe_to_ref<srm_t>();
      const auto& G2 = Gany2.to_ref<srm_t>();
      EXPECT_SRM_NEAR_EPS(G1, G2, CHECK_GRAD_EPS);
    }
  }
}

void OpTestContext::CheckForward(
    GraphNode* node, int on_heap, const DataType::tsr_t& expected_forward,
    const param_initializer_t& pre_param_initializer,
    const param_initializer_t& post_param_initializer,
    const inst_initializer_t& inst_initializer) {
  InitGraph(node, on_heap, 1);
  InitParam(pre_param_initializer, post_param_initializer);
  InitClosedOpContext(inst_initializer);

  auto check = [this, node, &expected_forward]() {
    ClosedForward();
    auto* forward = closed_.ptr().get<DataType::tsr_t*>(node->name());
    EXPECT_TSR_NEAR_EPS(*forward, expected_forward, CHECK_FORWARD_EPS);
  };

  // check twice
  check();
  check();
}

void OpTestContext::CheckBackward(
    GraphNode* node, int on_heap,
    const param_initializer_t& pre_param_initializer,
    const param_initializer_t& post_param_initializer,
    const inst_initializer_t& inst_initializer) {
  InitGraph(node, on_heap, 0);
  InitParam(pre_param_initializer, post_param_initializer);
  InitClosedOpContext(inst_initializer);
  InitNumericalOpContext(inst_initializer);

  auto check = [this]() {
    ClosedForward();
    ClosedBackward();
    ComputeNumericalGrad();
    CompareGrad();
  };

  // check twice
  check();
  check();
}

void CheckOpForward(GraphNode* node, int on_heap,
                    const DataType::tsr_t& expected_forward,
                    const param_initializer_t& pre_param_initializer,
                    const param_initializer_t& post_param_initializer,
                    const inst_initializer_t& inst_initializer) {
  OpTestContext op_test_context;
  op_test_context.CheckForward(node, on_heap, expected_forward,
                               pre_param_initializer, post_param_initializer,
                               inst_initializer);
}

void CheckOpBackward(GraphNode* node, int on_heap,
                     const param_initializer_t& pre_param_initializer,
                     const param_initializer_t& post_param_initializer,
                     const inst_initializer_t& inst_initializer) {
  OpTestContext op_test_context;
  op_test_context.CheckBackward(node, on_heap, pre_param_initializer,
                                post_param_initializer, inst_initializer);
}

}  // namespace deepx_core
