// Copyright 2020 the deepx authors.
// Author: Shuting Guo (tinkleguo@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class DeepFM3Model : public ModelZooImpl {
 private:
  // 'cont' in this file means continuous.
  std::vector<GroupConfigItem3> cont_items_;
  std::vector<int> deep_dims_{64, 32};

 public:
  DEFINE_MODEL_ZOO_LIKE(DeepFM3Model);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "cont_group_config") {
      if (!GuessGroupConfig(v, &cont_items_, nullptr, k.c_str())) {
        return false;
      }
    } else if (k == "deep_dims") {
      if (!ParseDeepDims(v, &deep_dims_, k.c_str())) {
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
    if (cont_items_.empty()) {
      DXERROR("Please specify cont_group_config.");
      return false;
    }
    return true;
  }

 private:
  GraphNode* GetCont(GraphNode* X) const {
    DXCHECK_THROW(!cont_items_.empty());
    std::vector<GraphNode*> W(cont_items_.size());
    std::vector<uint16_t> group_ids(cont_items_.size());
    W[0] = Constant("", Shape(1, 1), 1.0);
    for (size_t i = 0; i < cont_items_.size(); ++i) {
      group_ids[i] = cont_items_[i].group_id;
      W[i] = W[0];
    }
    return GroupEmbeddingLookup("", X, W, group_ids);
  }

 public:
  bool InitGraph(Graph* graph) const override {
    auto* X = GetX();
    auto* lin = WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* quad1 = DeepGroupEmbeddingLookup("quad", X, items_, sparse_);
    auto* quad2 = Reshape("", quad1, Shape(-1, item_m_, item_k_));
    auto* quad3 = BatchGroupFMQuadratic2("", quad2);
    auto* quad4 = Concat("", {quad1, GetCont(X)});
    auto* deep = StackedFullyConnect("deep", quad4, deep_dims_, "relu");
    auto* Z1 = Concat("", {lin, quad3, deep});
    auto* Z2 = FullyConnect("Z2", Z1, 1);
    auto Z3 = BinaryClassificationTarget(Z2, has_w_);
    ReleaseVariable();
    return graph->Compile(Z3, 1);
  }
};

MODEL_ZOO_REGISTER(DeepFM3Model, "DeepFM3Model");
MODEL_ZOO_REGISTER(DeepFM3Model, "deep_fm3");
MODEL_ZOO_REGISTER(DeepFM3Model, "deepfm3");

}  // namespace deepx_core
