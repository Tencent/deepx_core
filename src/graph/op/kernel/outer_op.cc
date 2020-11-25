// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct OuterAux {
  Shape Z;
  int m = 0;  // total dim of X
  int n = 0;  // total dim of Y
};

bool OuterPrepare(const Shape& X, const Shape& Y, OuterAux* aux) noexcept {
  int Xrank = X.rank();
  if (Xrank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  int Yrank = Y.rank();
  if (Yrank == 0) {
    DXERROR("Invalid Y: rank of Y is zero.");
    return false;
  }

  aux->Z.resize(X.total_dim(), Y.total_dim());
  aux->m = X.total_dim();
  aux->n = Y.total_dim();
  return true;
}

bool OuterInferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
  OuterAux aux;
  if (!OuterPrepare(X, Y, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void Outer(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
           const OuterAux& aux) noexcept {
  int m = aux.m, n = aux.n;
  Z->zeros();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Z->data(i * n + j) = X.data(i) * Y.data(j);
    }
  }
}

template <typename T>
void OuterBackward(const Tensor<T>& X, const Tensor<T>& Y,
                   const Tensor<T>& /*Y*/, const Tensor<T>& gZ, Tensor<T>* gX,
                   Tensor<T>* gY, const OuterAux& aux) noexcept {
  int m = aux.m, n = aux.n;
  if (gX) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        gX->data(i) += Y.data(j) * gZ.data(i * n + j);
      }
    }
  }
  if (gY) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        gY->data(j) += X.data(i) * gZ.data(i * n + j);
      }
    }
  }
}

}  // namespace

OuterNode::OuterNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)OuterInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class OuterOp : public OpBinaryBase {
 private:
  OuterAux aux_;

 public:
  DEFINE_OP_LIKE(OuterOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(OuterPrepare(X_->shape(), Y_->shape(), &aux_));
    return aux_.Z;
  }

  void Forward() override { Outer(*X_, *Y_, Z_, aux_); }

  void Backward() override {
    if (gZ_) {
      OuterBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Outer);

}  // namespace deepx_core
