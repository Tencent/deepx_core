// Copyright 2019 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct SubscriptRangeAux {
  Shape Z;
  int axis = 0;
  int m = 0;    // pre axis dim
  int Znk = 0;  // axis & post axis dim of Z
  int Xnk = 0;  // axis & post axis dim of X
  int Xoffset = 0;
};

bool SubscriptRangePrepare(const Shape& X, int axis, int begin_index,
                           int end_index, SubscriptRangeAux* aux) noexcept {
  int rank = X.rank();
  if (rank == 0) {
    DXERROR("Invalid X: rank of X is zero.");
    return false;
  }

  if (!X.real_axis(&axis)) {
    DXERROR("Invalid axis: %d.", axis);
    return false;
  }

  int X_axis_dim = X[axis], Z_axis_dim = end_index - begin_index;
  if (!(Z_axis_dim > 0 && begin_index >= 0 && end_index <= X_axis_dim)) {
    DXERROR("Invalid index: [%d, %d) must be a subrange of [0, %d).",
            begin_index, end_index, X_axis_dim);
    return false;
  }

  int m = 1, n = 1;
  int Zdims[SHAPE_MAX_RANK];
  for (int i = 0; i < axis; ++i) {
    Zdims[i] = X[i];
    m *= X[i];
  }
  Zdims[axis] = Z_axis_dim;
  for (int i = axis + 1; i < rank; ++i) {
    Zdims[i] = X[i];
    n *= X[i];
  }

  aux->Z.assign(&Zdims[0], &Zdims[rank]);
  aux->axis = axis;
  aux->m = m;
  aux->Znk = n * Z_axis_dim;
  aux->Xnk = n * X_axis_dim;
  aux->Xoffset = n * begin_index;
  return true;
}

bool SubscriptRangeInferShape(const Shape& X, int axis, int begin_index,
                              int end_index, Shape* Z) noexcept {
  SubscriptRangeAux aux;
  if (!SubscriptRangePrepare(X, axis, begin_index, end_index, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void SubscriptRange(const Tensor<T>& X, Tensor<T>* Z,
                    const SubscriptRangeAux& aux) {
  int m = aux.m;
  const T* _X = X.data() + aux.Xoffset;
  T* _Z = Z->data();
  for (int i = 0; i < m; ++i) {
    LLMath<T>::copy(aux.Znk, _X, _Z);
    _X += aux.Xnk;
    _Z += aux.Znk;
  }
}

template <typename T>
void SubscriptRangeBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                            const Tensor<T>& gZ, Tensor<T>* gX,
                            const SubscriptRangeAux& aux) {
  int m = aux.m;
  const T* _gZ = gZ.data();
  T* _gX = gX->data() + aux.Xoffset;
  for (int i = 0; i < m; ++i) {
    LLMath<T>::add(aux.Znk, _gX, _gZ, _gX);
    _gZ += aux.Znk;
    _gX += aux.Xnk;
  }
}

}  // namespace

SubscriptRangeNode::SubscriptRangeNode(std::string name, GraphNode* X, int axis,
                                       int begin_index, int end_index)
    : GraphNodeUnaryBase(std::move(name), X),
      axis_(axis),
      begin_index_(begin_index),
      end_index_(end_index) {
  if (!X->shape().empty()) {
    (void)SubscriptRangeInferShape(X->shape(), axis_, begin_index_, end_index_,
                                   &shape_);
  }
}

class SubscriptRangeOp : public OpUnaryBase {
 private:
  SubscriptRangeAux aux_;

 public:
  DEFINE_OP_LIKE(SubscriptRangeOp);

  const Shape& InferShape() override {
    const SubscriptRangeNode* node =  // NOLINT
        (const SubscriptRangeNode*)node_;
    DXCHECK_THROW(SubscriptRangePrepare(X_->shape(), node->axis(),
                                        node->begin_index(), node->end_index(),
                                        &aux_));
    return aux_.Z;
  }

  void Forward() override { SubscriptRange(*X_, Z_, aux_); }

  void Backward() override {
    if (gX_) {
      SubscriptRangeBackward(*X_, *Z_, *gZ_, gX_, aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(SubscriptRange);

}  // namespace deepx_core
