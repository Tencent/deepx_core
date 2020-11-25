// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class PairwiseDeepFMModel : public ModelZooImpl {
 private:
  std::vector<int> deep_dims_{64, 32};
  int hinge_loss_ = 1;
  double hinge_loss_margin_ = 0.1;

 public:
  DEFINE_MODEL_ZOO_LIKE(PairwiseDeepFMModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "deep_dims") {
      if (!ParseDeepDims(v, &deep_dims_, k.c_str())) {
        return false;
      }
    } else if (k == "hinge_loss") {
      hinge_loss_ = std::stoi(v);
    } else if (k == "hinge_loss_margin") {
      hinge_loss_margin_ = std::stod(v);
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

 protected:
  GraphNode* DeepFM(const std::string& prefix, GraphNode* X) const {
    auto* lin = WideGroupEmbeddingLookup(prefix + "wide", X, items_, sparse_);
    auto* quad1 = DeepGroupEmbeddingLookup(prefix + "deep", X, items_, sparse_);
    auto* quad2 = Reshape("", quad1, Shape(-1, item_m_, item_k_));
    auto* quad3 = BatchGroupFMQuadratic2("", quad2);
    auto* deep = StackedFullyConnect(prefix + "sfc", quad1, deep_dims_, "relu");
    auto* Z1 = Concat("", {lin, quad3, deep});
    auto* Z2 = FullyConnect(prefix + "fc", Z1, 1);
    return Z2;
  }

  GraphNode* Train() const {
    auto* X1 = GetX(0);
    auto* X2 = GetX(1);
    auto* Z1 = DeepFM("deepfm", X1);
    auto* Z2 = DeepFM("deepfm", X2);
    auto* Z3 = Sub("", Z1, Z2);

    GraphNode* L = nullptr;
    if (hinge_loss_) {
      auto* C = ConstantLike("", Z3, hinge_loss_margin_);
      auto* S = Sub("", C, Z3);
      L = Relu("", S);
    } else {
      auto* Y = GetY(1);
      L = SigmoidBCELoss("", Z3, Y);
    }

    GraphNode* M = nullptr;
    if (has_w_) {
      auto* W = GetW(1);
      auto* WL = Mul("", L, W);
      M = ReduceMean("", WL);
    } else {
      M = ReduceMean("", L);
    }
    return M;
  }

  GraphNode* Predict() const {
    auto* X = GetX();
    auto* Z = DeepFM("deepfm", X);
    auto* P = Sigmoid("", Z);
    return P;
  }

 public:
  bool InitGraph(Graph* graph) const override {
    auto targets = {Train(), Predict()};
    ReleaseVariable();
    return graph->Compile(targets, 1);
  }
};

MODEL_ZOO_REGISTER(PairwiseDeepFMModel, "PairwiseDeepFMModel");
MODEL_ZOO_REGISTER(PairwiseDeepFMModel, "pairwise_deep_fm");
MODEL_ZOO_REGISTER(PairwiseDeepFMModel, "pairwise_deepfm");

}  // namespace deepx_core
