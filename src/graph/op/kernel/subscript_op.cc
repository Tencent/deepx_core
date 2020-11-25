// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct SubscriptAux {
  Shape Z;
  int axis = 0;
  int m = 0;   // pre axis dim
  int n = 0;   // post axis dim
  int k = 0;   // axis dim
  int nk = 0;  // axis & post axis dim
  int Xoffset = 0;
};

bool SubscriptPrepare(const Shape& X, int axis, int index,
                      SubscriptAux* aux) noexcept {
  int rank = X.rank();
  if (rank <= 1) {
    DXERROR("Invalid X: rank of X %d must be greater than 1.", rank);
    return false;
  }

  if (!X.real_axis(&axis)) {
    DXERROR("Invalid axis: %d.", axis);
    return false;
  }

  if (index < 0 || index >= X[axis]) {
    DXERROR("Invalid index: index %d must be in range [0, %d).", index,
            X[axis]);
    return false;
  }

  int Zdims[SHAPE_MAX_RANK];
  for (int i = 0; i < axis; ++i) {
    Zdims[i] = X[i];
  }
  for (int i = axis; i < rank - 1; ++i) {
    Zdims[i] = X[i + 1];
  }

  int m = 1, n = 1, k;
  for (int i = 0; i < axis; ++i) {
    m *= X[i];
  }
  k = X[axis];
  for (int i = axis + 1; i < rank; ++i) {
    n *= X[i];
  }

  aux->Z.assign(&Zdims[0], &Zdims[rank - 1]);
  aux->axis = axis;
  aux->m = m;
  aux->n = n;
  aux->k = k;
  aux->nk = n * k;
  aux->Xoffset = n * index;
  return true;
}

bool SubscriptInferShape(const Shape& X, int axis, int index,
                         Shape* Z) noexcept {
  SubscriptAux aux;
  if (!SubscriptPrepare(X, axis, index, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void Subscript(const Tensor<T>& X, Tensor<T>* Z, const SubscriptAux& aux) {
  int m = aux.m, n = aux.n, nk = aux.nk;
  const T* _X = X.data() + aux.Xoffset;
  T* _Z = Z->data();
  for (int i = 0; i < m; ++i) {
    LLMath<T>::copy(n, _X, _Z);
    _X += nk;
    _Z += n;
  }
}

template <typename T>
void SubscriptBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const SubscriptAux& aux) {
  int m = aux.m, n = aux.n, nk = aux.nk;
  const T* _gZ = gZ.data();
  T* _gX = gX->data() + aux.Xoffset;
  for (int i = 0; i < m; ++i) {
    LLMath<T>::add(n, _gX, _gZ, _gX);
    _gZ += n;
    _gX += nk;
  }
}

}  // namespace

SubscriptNode::SubscriptNode(std::string name, GraphNode* X, int axis,
                             int index)
    : GraphNodeUnaryBase(std::move(name), X), axis_(axis), index_(index) {
  if (!X->shape().empty()) {
    (void)SubscriptInferShape(X->shape(), axis_, index_, &shape_);
  }
}

class SubscriptOp : public OpUnaryBase {
 private:
  SubscriptAux aux_;

 public:
  DEFINE_OP_LIKE(SubscriptOp);

  const Shape& InferShape() override {
    const SubscriptNode* node = (const SubscriptNode*)node_;  // NOLINT
    DXCHECK_THROW(
        SubscriptPrepare(X_->shape(), node->axis(), node->index(), &aux_));
    return aux_.Z;
  }

  void Forward() override { Subscript(*X_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      SubscriptBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Subscript);

}  // namespace deepx_core
