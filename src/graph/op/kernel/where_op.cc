// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

bool WhereInferShape(const Shape& C, const Shape& X, const Shape& Y,
                     Shape* Z) noexcept {
  if (C != X) {
    DXERROR("Invalid C and X: inconsistent shape %s vs %s.",
            to_string(C).c_str(), to_string(X).c_str());
  }

  if (C != Y) {
    DXERROR("Invalid C and Y: inconsistent shape %s vs %s.",
            to_string(C).c_str(), to_string(Y).c_str());
  }

  *Z = C;
  return true;
}

template <typename T>
void Where(const Tensor<T>& C, const Tensor<T>& X, const Tensor<T>& Y,
           Tensor<T>* Z) noexcept {
  for (int i = 0; i < C.total_dim(); ++i) {
    if (C.data(i) != 0) {
      Z->data(i) = X.data(i);
    } else {
      Z->data(i) = Y.data(i);
    }
  }
}

template <typename T>
void WhereBackward(const Tensor<T>& C, const Tensor<T>& /*X*/,
                   const Tensor<T>& /*Y*/, const Tensor<T>& /*Z*/,
                   const Tensor<T>& gZ, Tensor<T>* gX, Tensor<T>* gY) noexcept {
  for (int i = 0; i < gZ.total_dim(); ++i) {
    if (C.data(i) != 0) {
      if (gX) {
        gX->data(i) += gZ.data(i);
      }
    } else {
      if (gY) {
        gY->data(i) += gZ.data(i);
      }
    }
  }
}

}  // namespace

WhereNode::WhereNode(std::string name, GraphNode* C, GraphNode* X, GraphNode* Y)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(C->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(Y->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {C, X, Y};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (!C->shape().empty() && !X->shape().empty() && !Y->shape().empty()) {
    (void)WhereInferShape(C->shape(), X->shape(), Y->shape(), &shape_);
  }
}

class WhereOp : public OpImpl {
 private:
  const GraphNode* Cnode_ = nullptr;
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Ynode_ = nullptr;
  const tsr_t* C_ = nullptr;
  const tsr_t* X_ = nullptr;
  const tsr_t* Y_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gC_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* gY_ = nullptr;

 public:
  DEFINE_OP_LIKE(WhereOp);

  void InitForward() override {
    Cnode_ = node_->input(0);
    Xnode_ = node_->input(1);
    Ynode_ = node_->input(2);
    C_ = GetPtrTSR(Cnode_);
    X_ = GetPtrTSR(Xnode_);
    Y_ = GetPtrTSR(Ynode_);
    DXCHECK_THROW(
        WhereInferShape(C_->shape(), X_->shape(), Y_->shape(), &Zshape_));
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gC_ = InitGradTSR(Cnode_, C_->shape());
    gX_ = InitGradTSR(Xnode_, X_->shape());
    gY_ = InitGradTSR(Ynode_, Y_->shape());
  }

  void Forward() override { Where(*C_, *X_, *Y_, Z_); }

  void Backward() override {
    if (gZ_) {
      WhereBackward(*C_, *X_, *Y_, *Z_, *gZ_, gX_, gY_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Where);

}  // namespace deepx_core
