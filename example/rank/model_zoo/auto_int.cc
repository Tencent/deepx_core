// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class AutoIntModel : public ModelZooImpl {
 private:
  int att_t_ = 8;
  int att_h_ = 2;
  int att_s_ = 2;

 public:
  DEFINE_MODEL_ZOO_LIKE(AutoIntModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "att_t") {
      att_t_ = std::stoi(v);
      if (att_t_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "att_h") {
      att_h_ = std::stoi(v);
      if (att_h_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "att_s") {
      att_s_ = std::stoi(v);
      if (att_s_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else {
      DXERROR("Unexpected config: %s=%s.", k.c_str(), v.c_str());
      return false;
    }
    return true;
  }

  bool PostInitConfig() override {
    if (items_.empty()) {
      DXERROR("Please specify group_config.");
      return false;
    }
    DXCHECK_THROW(item_is_fm_);
    return true;
  }

 private:
  static std::vector<GraphNode*> MHSAR(const std::string& prefix, GraphNode* X,
                                       int att_t, int att_h) {
    DXCHECK_THROW(X->shape().is_rank(3));
    int m = X->shape()[1];
    int k = X->shape()[2];
    int t = att_t;
    int h = att_h;
    std::vector<GraphNode*> Z(2);
    std::vector<GraphNode*> Z6in(h);
    for (int i = 0; i < h; ++i) {
      auto ii = std::to_string(i);
      auto get_name = [&prefix, &ii](const std::string& name) {
        return prefix + "_" + name + "_" + ii;
      };
      auto* Wq = GetVariableRandXavier(get_name("Wq"), Shape(k, t));
      auto* Wk = GetVariableRandXavier(get_name("Wk"), Shape(k, t));
      auto* Wv = GetVariableRandXavier(get_name("Wv"), Shape(k, t));
      auto* C = ConstantScalar("", 1 / std::sqrt(1.0 * k));
      auto* Q = Matmul("", X, Wq);            // (batch, m, t)
      auto* K = Matmul("", X, Wk);            // (batch, m, t)
      auto* V = Matmul("", X, Wv);            // (batch, m, t)
      auto* Z1 = BatchGEMM("", Q, K, 0, 1);   // (batch, m, m)
      auto* Z2 = BroadcastMul("", Z1, C);     // (batch, m, m)
      auto* Z3 = Softmax("", Z2, -1);         // (batch, m, m)
      auto* Z4 = BatchGEMM("", Z3, V, 0, 0);  // (batch, m, t)
      auto* Z5 = ReshapeFast("", Z4, Shape(-1, m * t));
      Z6in[i] = Z5;
    }
    auto get_name = [&prefix](const std::string& name) {
      return prefix + "_" + name;
    };
    auto* Z6 = Concat("", Z6in);                  // (batch, m * t * h)
    auto* Z7 = Reshape("", X, Shape(-1, m * k));  // (batch, m * k)
    auto* Wr = GetVariableRandXavier(get_name("Wr"), Shape(m * k, m * t * h));
    auto* Z8 = Matmul("", Z7, Wr);  // (batch, m * t * h)
    auto* Z9 = Add("", Z6, Z8);     // (batch, m * t * h)
    auto* Z10 = Relu("", Z9);       // (batch, m * t * h)
    auto* Z11 = Reshape("", Z10, Shape(-1, m, t * h));
    Z[0] = Z10;
    Z[1] = Z11;
    return Z;
  }

 public:
  bool InitGraph(Graph* graph) const override {
    auto* X = GetX();
    auto* quad1 = DeepGroupEmbeddingLookup("quad", X, items_, sparse_);
    auto* quad2 = ReshapeFast("", quad1, Shape(-1, item_m_, item_k_));
    auto* Xi = quad2;
    std::vector<GraphNode*> Z1in(att_s_);
    for (int i = 0; i < att_s_; ++i) {
      auto ii = std::to_string(i);
      auto mhsar = MHSAR("mhsar" + ii, Xi, att_t_, att_h_);
      Z1in[i] = mhsar[0];
      Xi = mhsar[1];
    }
    delete Xi;
    auto* Z1 = Concat("", Z1in);
    auto* Z2 = FullyConnect("Z2", Z1, 1);
    auto Z3 = BinaryClassificationTarget(Z2, has_w_);
    ReleaseVariable();
    return graph->Compile(Z3, 1);
  }
};

MODEL_ZOO_REGISTER(AutoIntModel, "AutoIntModel");
MODEL_ZOO_REGISTER(AutoIntModel, "auto_int");
MODEL_ZOO_REGISTER(AutoIntModel, "autoint");

}  // namespace deepx_core
