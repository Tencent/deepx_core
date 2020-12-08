// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

bool AddNInferShape(const std::vector<const Shape*>& X, Shape* Z) {
  if (X.empty()) {
    DXERROR("Invalid X: X is empty.");
    return false;
  }

  const Shape& X0 = *X[0];
  int rank = X0.rank();
  if (rank == 0) {
    DXERROR("Invalid X: rank of X[0] is zero.");
    return false;
  }

  for (size_t i = 0; i < X.size(); ++i) {  // NOLINT
    const Shape& Xi = *X[i];
    if (Xi != X0) {
      DXERROR("Invalid X: inconsistent shape %s vs %s.", to_string(Xi).c_str(),
              to_string(X0).c_str());
      return false;
    }
  }

  *Z = X0;
  return true;
}

template <typename T>
void AddN(const std::vector<const Tensor<T>*>& X, Tensor<T>* Z) noexcept {
  Z->zeros();
  for (size_t i = 0; i < X.size(); ++i) {
    LLTensor<T>::add(*X[i], *Z, Z);
  }
}

template <typename T>
void AddNBackward(const std::vector<const Tensor<T>*>& /*X*/,
                  const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                  std::vector<Tensor<T>*>* gX) noexcept {
  for (size_t i = 0; i < gX->size(); ++i) {
    if ((*gX)[i]) {
      LLTensor<T>::add(gZ, *(*gX)[i], (*gX)[i]);
    }
  }
}

}  // namespace

AddNNode::AddNNode(std::string name, std::vector<GraphNode*> X)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(!X.empty());
  for (const GraphNode* _X : X) {
    DXCHECK_THROW(_X->tensor_type() == TENSOR_TYPE_TSR);
  }

  input_ = std::move(X);
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (HasShape(input_)) {
    std::vector<const Shape*> Xshape(input_size());
    for (int i = 0; i < input_size(); ++i) {
      Xshape[i] = &input_[i]->shape();
    }
    (void)AddNInferShape(Xshape, &shape_);
  }
}

class AddNOp : public OpImpl {
 private:
  std::vector<const Shape*> Xshape_;
  std::vector<const tsr_t*> X_;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  std::vector<tsr_t*> gX_;

 public:
  DEFINE_OP_LIKE(AddNOp);

  void InitForward() override {
    int input_size = node_->input_size();
    Xshape_.resize(input_size);
    X_.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
      const GraphNode* Xnode = node_->input(i);
      const tsr_t* X = GetPtrTSR(Xnode);
      Xshape_[i] = &X->shape();
      X_[i] = X;
    }
    DXCHECK_THROW(AddNInferShape(Xshape_, &Zshape_));
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    int input_size = node_->input_size();
    gZ_ = GetGradPtrTSR(node_);
    gX_.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
      const GraphNode* Xnode = node_->input(i);
      gX_[i] = InitGradTSR(Xnode, X_[i]->shape());
    }
  }

  void Forward() override { AddN(X_, Z_); }

  void Backward() override {
    if (gZ_) {
      AddNBackward(X_, *Z_, *gZ_, &gX_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(AddN);

}  // namespace deepx_core
