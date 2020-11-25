// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class LRModel : public ModelZooImpl {
 private:
  int dim_ = 1000000;

 public:
  DEFINE_MODEL_ZOO_LIKE(LRModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "dim") {
      dim_ = std::stoi(v);
      if (dim_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else {
      DXERROR("Unexpected config: %s=%s.", k.c_str(), v.c_str());
      return false;
    }
    return true;
  }

 private:
  bool InitGraph_NoGroupConfig(Graph* graph) const {
    DXINFO("sparse will be ignored.");
    auto* X = GetX();
    auto* Wlin = GetVariableZeros("Wlin", Shape(dim_, 1));
    auto* lin = EmbeddingLookup("", X, Wlin);
    auto* Z1 = AddBias("Z1", lin);
    auto Z2 = BinaryClassificationTarget(Z1, has_w_);
    ReleaseVariable();
    return graph->Compile(Z2, 1);
  }

  bool InitGraph_GroupConfig(Graph* graph) const {
    DXINFO("dim will be ignored.");
    auto* X = GetX();
    auto* lin1 = WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* lin2 = ReduceSum("", lin1, 1, 1);
    auto* Z1 = AddBias("Z1", lin2);
    auto Z2 = BinaryClassificationTarget(Z1, has_w_);
    ReleaseVariable();
    return graph->Compile(Z2, 1);
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

MODEL_ZOO_REGISTER(LRModel, "LRModel");
MODEL_ZOO_REGISTER(LRModel, "lr");

}  // namespace deepx_core
