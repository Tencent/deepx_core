// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class FMModel : public ModelZooImpl {
 private:
  int dim_ = 1000000;
  int k_ = 8;

 public:
  DEFINE_MODEL_ZOO_LIKE(FMModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "dim") {
      dim_ = std::stoi(v);
      if (dim_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "k") {
      k_ = std::stoi(v);
      if (k_ <= 1) {
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
    if (!items_.empty()) {
      DXCHECK_THROW(item_is_fm_);
    }
    return true;
  }

 private:
  bool InitGraph_NoGroupConfig(Graph* graph) const {
    DXINFO("sparse will be ignored.");
    auto* X = GetX();
    auto* W = GetVariableZeros("W", Shape(dim_, 1));
    auto* V = GetVariableRandnXavier("V", Shape(dim_, k_));
    auto* lin = EmbeddingLookup("", X, W);
    auto* quad = BatchFMQuadratic("", X, V);
    auto* Z1 = Add("", lin, quad);
    auto* Z2 = AddBias("Z2", Z1);
    auto Z3 = BinaryClassificationTarget(Z2, has_w_);
    ReleaseVariable();
    return graph->Compile(Z3, 1);
  }

  bool InitGraph_GroupConfig(Graph* graph) const {
    DXINFO("dim will be ignored.");
    DXINFO("k will be ignored.");
    auto* X = GetX();
    auto* lin1 = WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* lin2 = ReduceSum("", lin1, 1, 1);
    auto* quad1 = DeepGroupEmbeddingLookup("quad", X, items_, sparse_);
    auto* quad2 = ReshapeFast("", quad1, Shape(-1, item_m_, item_k_));
    auto* quad3 = BatchGroupFMQuadratic("", quad2);
    auto* Z1 = Add("", lin2, quad3);
    auto* Z2 = AddBias("Z2", Z1);
    auto Z3 = BinaryClassificationTarget(Z2, has_w_);
    ReleaseVariable();
    return graph->Compile(Z3, 1);
  }

 public:
  bool InitGraph(Graph* graph) const override {
    if (items_.empty()) {
      return InitGraph_NoGroupConfig(graph);
    } else {
      return InitGraph_GroupConfig(graph);
    }
  }
};

MODEL_ZOO_REGISTER(FMModel, "FMModel");
MODEL_ZOO_REGISTER(FMModel, "fm");

}  // namespace deepx_core
