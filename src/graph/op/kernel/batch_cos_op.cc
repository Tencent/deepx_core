// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct BatchCosAux {
  Shape Z;
  int batch = 0;
  int m = 0;
};

bool BatchCosPrepare(const Shape& X, const Shape& Y,
                     BatchCosAux* aux) noexcept {
  if (!X.is_rank(2)) {
    DXERROR("Invalid X: rank of X %d must be 2.", X.rank());
    return false;
  }

  if (X != Y) {
    DXERROR("Invalid X and Y: inconsistent shape %s vs %s.",
            to_string(X).c_str(), to_string(Y).c_str());
    return false;
  }

  int batch = X[0];
  int m = X[1];
  aux->Z.resize(batch, 1);
  aux->batch = batch;
  aux->m = m;
  return true;
}

bool BatchCosInferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
  BatchCosAux aux;
  if (!BatchCosPrepare(X, Y, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
struct BatchCosMutableAux {
  Tensor<T> XX;      // dot(X[i, :], X[i, :])
  Tensor<T> YY;      // dot(Y[i, :], Y[i, :])
  Tensor<T> XYnorm;  // sqrt(XX * YY);
};

template <typename T>
void BatchCosPrepare(const BatchCosAux& aux, BatchCosMutableAux<T>* maux) {
  maux->XX.resize(aux.batch);
  maux->YY.resize(aux.batch);
  maux->XYnorm.resize(aux.batch);
}

template <typename T>
void BatchCos(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
              const BatchCosAux& aux, BatchCosMutableAux<T>* maux) noexcept {
  int batch = aux.batch;
  int m = aux.m;
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _XX = maux->XX.data();
  T* _YY = maux->YY.data();
  T* _XYnorm = maux->XYnorm.data();
  for (int i = 0; i < batch; ++i) {
    _XX[i] = LLMath<T>::dot(m, _X, _X);
    _XX[i] = _XX[i] < (T)1e-12 ? (T)1e-12 : _XX[i];
    _YY[i] = LLMath<T>::dot(m, _Y, _Y);
    _YY[i] = _YY[i] < (T)1e-12 ? (T)1e-12 : _YY[i];
    _XYnorm[i] = std::sqrt(_XX[i] * _YY[i]);
    _Z[i] = LLMath<T>::dot(m, _X, _Y) / _XYnorm[i];
    _X += m;
    _Y += m;
  }
}

template <typename T>
void BatchCosBackward(const Tensor<T>& X, const Tensor<T>& Y,
                      const Tensor<T>& Z, const Tensor<T>& gZ, Tensor<T>* gX,
                      Tensor<T>* gY, const BatchCosAux& aux,
                      BatchCosMutableAux<T>* maux) noexcept {
  int batch = aux.batch;
  int m = aux.m;
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  const T* _XX = maux->XX.data();
  const T* _YY = maux->YY.data();
  const T* _XYnorm = maux->XYnorm.data();

  auto update_x = [ batch, m, _Z, _gZ, _XYnorm ](
      const T* _X, const T* _Y, const T* _XX, T* _gX) noexcept {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < m; ++j) {
        _gX[j] += _gZ[i] * (_Y[j] / _XYnorm[i] - _Z[i] * _X[j] / _XX[i]);
      }
      _X += m;
      _Y += m;
      _gX += m;
    }
  };

  if (gX) {
    T* _gX = gX->data();
    update_x(_X, _Y, _XX, _gX);
  }

  if (gY) {
    T* _gY = gY->data();
    update_x(_Y, _X, _YY, _gY);
  }
}

}  // namespace

BatchCosNode::BatchCosNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchCosInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BatchCosOp : public OpBinaryBase {
 private:
  BatchCosAux aux_;
  BatchCosMutableAux<float_t> maux_;

 public:
  DEFINE_OP_LIKE(BatchCosOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(BatchCosPrepare(X_->shape(), Y_->shape(), &aux_));
    BatchCosPrepare(aux_, &maux_);
    return aux_.Z;
  }

  void Forward() override { BatchCos(*X_, *Y_, Z_, aux_, &maux_); }

  void Backward() override {
    if (gZ_) {
      BatchCosBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, &maux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchCos);

}  // namespace deepx_core
