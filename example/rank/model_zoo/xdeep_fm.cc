// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class XDeepFMModel : public ModelZooImpl {
 private:
  std::vector<int> deep_dims_{64, 32};
  std::vector<int> cin_dims_{32, 32, 32};

 public:
  DEFINE_MODEL_ZOO_LIKE(XDeepFMModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "deep_dims") {
      if (!ParseDeepDims(v, &deep_dims_, k.c_str())) {
        return false;
      }
    } else if (k == "cin_dims") {
      if (!ParseDeepDims(v, &cin_dims_, k.c_str())) {
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
    auto* lin = WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* quad1 = DeepGroupEmbeddingLookup("quad", X, items_, sparse_);
    auto* quad2 = Reshape("", quad1, Shape(-1, item_m_, item_k_));
    auto* cin = CIN("cin", quad2, cin_dims_);
    auto* deep = StackedFullyConnect("deep", quad1, deep_dims_, "relu");
    auto* Z1 = Concat("", {lin, cin, deep});
    auto* Z2 = FullyConnect("Z2", Z1, 1);
    auto Z3 = BinaryClassificationTarget(Z2, has_w_);
    ReleaseVariable();
    return graph->Compile(Z3, 1);
  }
};

MODEL_ZOO_REGISTER(XDeepFMModel, "XDeepFMModel");
MODEL_ZOO_REGISTER(XDeepFMModel, "xdeep_fm");
MODEL_ZOO_REGISTER(XDeepFMModel, "xdeepfm");

}  // namespace deepx_core
