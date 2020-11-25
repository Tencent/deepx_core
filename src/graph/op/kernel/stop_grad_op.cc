// Copyright 2019 the deepx authors.
// Author: Yalong Wang (vinceywang@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

StopGradNode::StopGradNode(std::string name, GraphNode* X)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {X};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (!X->shape().empty()) {
    shape_ = X->shape();
  }
}

class StopGradOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  const tsr_t* X_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gX_ = nullptr;
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(StopGradOp);

  void InitForward() override {
    Xnode_ = node_->input(0);
    X_ = GetPtrTSR(Xnode_);
    Z_ = InitHiddenTSR(node_, X_->shape());
  }

  void InitBackward() override { gX_ = InitGradTSR(Xnode_, X_->shape()); }

  void Forward() override { Z_->set_data(*X_); }
};

GRAPH_NODE_OP_REGISTER(StopGrad);

}  // namespace deepx_core
