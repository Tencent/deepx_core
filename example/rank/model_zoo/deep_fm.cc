// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class DeepFMModel : public ModelZooImpl {
 private:
  std::vector<int> deep_dims_{64, 32, 1};

 public:
  DEFINE_MODEL_ZOO_LIKE(DeepFMModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "deep_dims") {
      if (!ParseDeepDimsAppendOne(v, &deep_dims_, k.c_str())) {
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

 public:
  bool InitGraph(Graph* graph) const override {
    auto* X = GetX();
    auto* lin1 = WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* lin2 = ReduceSum("", lin1, 1, 1);
    auto* quad1 = DeepGroupEmbeddingLookup("quad", X, items_, sparse_);
    auto* quad2 = Reshape("", quad1, Shape(-1, item_m_, item_k_));
    auto* quad3 = BatchGroupFMQuadratic("", quad2);
    auto* deep = StackedFullyConnect("deep", quad1, deep_dims_, "relu");
    auto* Z1 = AddN("", {lin2, quad3, deep});
    auto Z2 = BinaryClassificationTarget(Z1, has_w_);
    ReleaseVariable();
    return graph->Compile(Z2, 1);
  }
};

MODEL_ZOO_REGISTER(DeepFMModel, "DeepFMModel");
MODEL_ZOO_REGISTER(DeepFMModel, "deep_fm");
MODEL_ZOO_REGISTER(DeepFMModel, "deepfm");

}  // namespace deepx_core
