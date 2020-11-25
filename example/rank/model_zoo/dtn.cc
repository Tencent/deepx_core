// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "dtn.h"
#include "../model_zoo_impl.h"

namespace deepx_core {

class DTNModel : public ModelZooImpl {
 private:
  std::vector<GroupConfigItem3> user_items_;
  std::vector<GroupConfigItem3> item_items_;
  std::vector<int> user_deep_dims_{64};
  std::vector<int> item_deep_dims_{64};
  std::vector<int> deep_dims_{64, 32, 1};

 public:
  DEFINE_MODEL_ZOO_LIKE(DTNModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "user_group_config") {
      if (!GuessGroupConfig(v, &user_items_, nullptr, k.c_str())) {
        return false;
      }
    } else if (k == "item_group_config") {
      if (!GuessGroupConfig(v, &item_items_, nullptr, k.c_str())) {
        return false;
      }
    } else if (k == "user_deep_dims") {
      if (!ParseDeepDims(v, &user_deep_dims_, k.c_str())) {
        return false;
      }
    } else if (k == "item_deep_dims") {
      if (!ParseDeepDims(v, &item_deep_dims_, k.c_str())) {
        return false;
      }
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
    DXCHECK_THROW(items_.empty());
    if (user_items_.empty()) {
      DXERROR("Please specify user_group_config.");
      return false;
    }
    if (item_items_.empty()) {
      DXERROR("Please specify item_group_config.");
      return false;
    }
    return true;
  }

 private:
  static GraphNode* GetXUser() {
    // Don't use BATCH_PLACEHOLDER.
    return new InstanceNode(DTN_X_USER_NAME, Shape(1, 0), TENSOR_TYPE_CSR);
  }

  static GraphNode* GetXItem() {
    return new InstanceNode(DTN_X_ITEM_NAME, Shape(BATCH_PLACEHOLDER, 0),
                            TENSOR_TYPE_CSR);
  }

  void TrainPredict(GraphNode** train_loss, GraphNode** predict_prob) const {
    auto* X = GetX();
    auto* UE = DeepGroupEmbeddingLookup("UE", X, user_items_, sparse_);
    auto* USFC = StackedFullyConnect("USFC", UE, user_deep_dims_);
    auto* IE = DeepGroupEmbeddingLookup("IE", X, item_items_, sparse_);
    auto* ISFC = StackedFullyConnect("ISFC", IE, item_deep_dims_);
    auto* C = Concat("", {USFC, ISFC});
    auto* SFC = StackedFullyConnect("SFC", C, deep_dims_);
    auto target = BinaryClassificationTarget(SFC, has_w_);
    *train_loss = target[0];
    *predict_prob = target[1];
  }

  void Infer(GraphNode** infer_prob) const {
    auto* Xuser = GetXUser();
    auto* Xitem = GetXItem();
    auto* UE = DeepGroupEmbeddingLookup("UE", Xuser, user_items_, sparse_);
    auto* USFC = StackedFullyConnect("USFC", UE, user_deep_dims_);
    auto* IE = DeepGroupEmbeddingLookup("IE", Xitem, item_items_, sparse_);
    auto* ISFC = StackedFullyConnect("ISFC", IE, item_deep_dims_);
    if (USFC->shape()[1] == ISFC->shape()[1]) {
      USFC = BroadcastToLike("", USFC, ISFC);
    } else {
      auto* ZERO = ConstantLike("", ISFC, 0);
      ZERO = SubscriptRange("", ZERO, 1, 0, 1);
      USFC = BroadcastAdd("", USFC, ZERO);
    }
    auto* C = Concat("", {USFC, ISFC});
    auto* SFC = StackedFullyConnect("SFC", C, deep_dims_);
    *infer_prob = Sigmoid("", SFC);
  }

 public:
  bool InitGraph(Graph* graph) const override {
    std::vector<GraphNode*> target(3);
    TrainPredict(&target[0], &target[1]);
    Infer(&target[2]);
    ReleaseVariable();
    return graph->Compile(target, 1);
  }
};

MODEL_ZOO_REGISTER(DTNModel, "DTNModel");
MODEL_ZOO_REGISTER(DTNModel, "dtn");

}  // namespace deepx_core
