// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "../model_zoo_impl.h"

namespace deepx_core {

class DCNModel : public ModelZooImpl {
 private:
  std::vector<int> deep_dims_{64, 32};
  int cross_ = 3;

 public:
  DEFINE_MODEL_ZOO_LIKE(DCNModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "deep_dims") {
      if (!ParseDeepDims(v, &deep_dims_, k.c_str())) {
        return false;
      }
    } else if (k == "cross") {
      cross_ = std::stoi(v);
      if (cross_ <= 0) {
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
    return true;
  }

 public:
  bool InitGraph(Graph* graph) const override {
    auto* X = GetX();
    auto* lin = WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* quad = DeepGroupEmbeddingLookup("quad", X, items_, sparse_);
    auto* cross = CrossNet("cross", quad, cross_);
    auto* deep = StackedFullyConnect("deep", quad, deep_dims_, "relu");
    auto* Z1 = Concat("", {lin, cross, deep});
    auto* Z2 = FullyConnect("Z2", Z1, 1);
    auto Z3 = BinaryClassificationTarget(Z2, has_w_);
    ReleaseVariable();
    return graph->Compile(Z3, 1);
  }
};

MODEL_ZOO_REGISTER(DCNModel, "DCNModel");
MODEL_ZOO_REGISTER(DCNModel, "dcn");

}  // namespace deepx_core
