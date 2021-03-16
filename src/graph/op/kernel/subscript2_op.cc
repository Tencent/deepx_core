// Copyright 2021 the deepx authors.
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct Subscript2Aux {
  Shape Z;
  int m = 0;   // pre axis dim
  int n = 0;   // post axis dim
  int k = 0;   // axis dim
  int nk = 0;  // axis & post axis dim
};

bool Subscript2Prepare(const Shape& X, const Shape& Y, int axis,
                       Subscript2Aux* aux) noexcept {
  int Xrank = X.rank();
  if (Xrank <= 1) {
    DXERROR("Invalid X: rank of X %d must be greater than 1.", Xrank);
    return false;
  }

  if (!X.real_axis(&axis)) {
    DXERROR("Invalid axis: %d.", axis);
    return false;
  }

  int Yrank = Y.rank();
  if (Yrank != Xrank - 1) {
    DXERROR("Invalid Y: rank of Y %d must be equal to %d.", Yrank, Xrank - 1);
    return false;
  }
  for (int i = 0; i < axis; ++i) {
    if (Y[i] != X[i]) {
      DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", X[i], Y[i]);
      return false;
    }
  }
  for (int i = axis; i < Xrank - 1; ++i) {
    if (Y[i] != X[i + 1]) {
      DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", X[i + 1], Y[i]);
      return false;
    }
  }

  int Zdims[SHAPE_MAX_RANK];
  for (int i = 0; i < axis; ++i) {
    Zdims[i] = X[i];
  }
  for (int i = axis; i < Xrank - 1; ++i) {
    Zdims[i] = X[i + 1];
  }

  int m = 1, n = 1, k;
  for (int i = 0; i < axis; ++i) {
    m *= X[i];
  }
  k = X[axis];
  for (int i = axis + 1; i < Xrank; ++i) {
    n *= X[i];
  }

  aux->Z.assign(&Zdims[0], &Zdims[Xrank - 1]);
  aux->m = m;
  aux->n = n;
  aux->k = k;
  aux->nk = n * k;
  return true;
}

bool Subscript2InferShape(const Shape& X, const Shape& Y, int axis,
                          Shape* Z) noexcept {
  Subscript2Aux aux;
  if (!Subscript2Prepare(X, Y, axis, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void Subscript2(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                const Subscript2Aux& aux) {
  int m = aux.m, n = aux.n, nk = aux.nk;
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  int index;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      index = (int)*_Y;
      DXASSERT(0 <= index && index < aux.k);
      *_Z = _X[n * index + j];
      ++_Y;
      ++_Z;
    }
    _X += nk;
  }
}

template <typename T>
void Subscript2Backward(const Tensor<T>& /*X*/, const Tensor<T>& Y,
                        const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                        Tensor<T>* gX, const Subscript2Aux& aux) {
  int m = aux.m, n = aux.n, nk = aux.nk;
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  int index;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      index = (int)*_Y;
      DXASSERT(0 <= index && index < aux.k);
      _gX[n * index + j] += *_gZ;
      ++_Y;
      ++_gZ;
    }
    _gX += nk;
  }
}

}  // namespace

Subscript2Node::Subscript2Node(std::string name, GraphNode* X, GraphNode* Y,
                               int axis)
    : GraphNodeBinaryBase(std::move(name), X, Y), axis_(axis) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)Subscript2InferShape(X->shape(), Y->shape(), axis_, &shape_);
  }
}

class Subscript2Op : public OpBinaryBase {
 private:
  Subscript2Aux aux_;

 public:
  DEFINE_OP_LIKE(Subscript2Op);

  const Shape& InferShape() override {
    const Subscript2Node* node = (const Subscript2Node*)node_;  // NOLINT
    DXCHECK_THROW(
        Subscript2Prepare(X_->shape(), Y_->shape(), node->axis(), &aux_));
    return aux_.Z;
  }

  void Forward() override { Subscript2(*X_, *Y_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      Subscript2Backward(*X_, *Y_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Subscript2);

}  // namespace deepx_core
