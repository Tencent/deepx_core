// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

/************************************************************************/
/* AbsoluteError */
/************************************************************************/
namespace {

template <typename T>
void AbsoluteError(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                   Tensor<T>* aux) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z, *aux);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  T r;
  for (int i = 0; i < X.total_dim(); ++i) {
    r = _X[i] - _Y[i];
    if (r > 0) {
      _Z[i] = r;
    } else {
      _Z[i] = -r;
    }
    _aux[i] = r;
  }
}

template <typename T>
void AbsoluteErrorBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Y*/,
                           const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                           Tensor<T>* gX, const Tensor<T>& aux) noexcept {
  DXASSERT_SAME_SHAPE(gZ, *gX, aux);
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux = aux.data();
  for (int i = 0; i < gZ.total_dim(); ++i) {
    if (_aux[i] > 0) {
      _gX[i] += _gZ[i];
    } else {
      _gX[i] -= _gZ[i];
    }
  }
}

}  // namespace

AbsoluteErrorNode::AbsoluteErrorNode(std::string name, GraphNode* X,
                                     GraphNode* Y)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}

class AbsoluteErrorOp : public OpBinaryElementWiseBase {
 private:
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(AbsoluteErrorOp);

  void InitForward() override {
    OpBinaryElementWiseBase::InitForward();
    aux_.resize(X_->shape());
  }

  void Forward() override { AbsoluteError(*X_, *Y_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      AbsoluteErrorBackward(*X_, *Y_, *Z_, *gZ_, gX_, aux_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(AbsoluteError);

/************************************************************************/
/* SquareError */
/************************************************************************/
namespace {

template <typename T>
void SquareError(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                 Tensor<T>* aux) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z, *aux);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  T r;
  for (int i = 0; i < X.total_dim(); ++i) {
    r = _X[i] - _Y[i];
    _Z[i] = r * r / 2;
    _aux[i] = r;
  }
}

template <typename T>
void SquareErrorBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Y*/,
                         const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                         Tensor<T>* gX, const Tensor<T>& aux) noexcept {
  DXASSERT_SAME_SHAPE(gZ, *gX, aux);
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux = aux.data();
  for (int i = 0; i < gZ.total_dim(); ++i) {
    _gX[i] += _gZ[i] * _aux[i];
  }
}

}  // namespace

SquareErrorNode::SquareErrorNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}

class SquareErrorOp : public OpBinaryElementWiseBase {
 private:
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(SquareErrorOp);

  void InitForward() override {
    OpBinaryElementWiseBase::InitForward();
    aux_.resize(X_->shape());
  }

  void Forward() override { SquareError(*X_, *Y_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      SquareErrorBackward(*X_, *Y_, *Z_, *gZ_, gX_, aux_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(SquareError);

/************************************************************************/
/* BCELoss */
/************************************************************************/
namespace {

template <typename T>
void BCELoss(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_Y[i] > 0) {
      _Z[i] = -LLMath<T>::safe_log(_X[i]);
    } else {
      _Z[i] = -LLMath<T>::safe_log(1 - _X[i]);
    }
  }
}

template <typename T>
void BCELossBackward(const Tensor<T>& X, const Tensor<T>& Y,
                     const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                     Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, gZ, *gX);
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  T x;
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_Y[i] > 0) {
      x = _X[i];
      x = x < (T)1e-6 ? (T)1e-6 : x;
      _gX[i] -= _gZ[i] / x;
    } else {
      x = 1 - _X[i];
      x = x < (T)1e-6 ? (T)1e-6 : x;
      _gX[i] += _gZ[i] / x;
    }
  }
}

}  // namespace

BCELossNode::BCELossNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}

class BCELossOp : public OpBinaryElementWiseBase {
 public:
  DEFINE_OP_LIKE(BCELossOp);

  void Forward() override { BCELoss(*X_, *Y_, Z_); }

  void Backward() override {
    if (gX_) {
      BCELossBackward(*X_, *Y_, *Z_, *gZ_, gX_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(BCELoss);

/************************************************************************/
/* BCELoss2 */
/************************************************************************/
namespace {

template <typename T>
void BCELoss2(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    _Z[i] = -_Y[i] * LLMath<T>::safe_log(_X[i]) -
            (1 - _Y[i]) * LLMath<T>::safe_log(1 - _X[i]);
  }
}

template <typename T>
void BCELoss2Backward(const Tensor<T>& X, const Tensor<T>& Y,
                      const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                      Tensor<T>* gX) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, gZ, *gX);
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  T x1, x2;
  for (int i = 0; i < X.total_dim(); ++i) {
    x1 = _X[i];
    x1 = x1 < (T)1e-6 ? (T)1e-6 : x1;
    x2 = 1 - _X[i];
    x2 = x2 < (T)1e-6 ? (T)1e-6 : x2;
    _gX[i] -= _gZ[i] * (_Y[i] / x1 + (_Y[i] - 1) / x2);
  }
}

}  // namespace

BCELoss2Node::BCELoss2Node(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}

class BCELoss2Op : public OpBinaryElementWiseBase {
 public:
  DEFINE_OP_LIKE(BCELoss2Op);

  void Forward() override { BCELoss2(*X_, *Y_, Z_); }

  void Backward() override {
    if (gX_) {
      BCELoss2Backward(*X_, *Y_, *Z_, *gZ_, gX_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(BCELoss2);

/************************************************************************/
/* SigmoidBCELoss */
/************************************************************************/
namespace {

template <typename T>
void SigmoidBCELoss(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                    Tensor<T>* aux) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z, *aux);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  T p;
  for (int i = 0; i < X.total_dim(); ++i) {
    p = LLMath<T>::sigmoid(_X[i]);
    if (_Y[i] > 0) {
      _Z[i] = -LLMath<T>::safe_log(p);
    } else {
      _Z[i] = -LLMath<T>::safe_log(1 - p);
    }
    _aux[i] = p;
  }
}

template <typename T>
void SigmoidBCELossBackward(const Tensor<T>& /*X*/, const Tensor<T>& Y,
                            const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                            Tensor<T>* gX, const Tensor<T>& aux) noexcept {
  DXASSERT_SAME_SHAPE(Y, gZ, *gX, aux);
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux = aux.data();
  for (int i = 0; i < Y.total_dim(); ++i) {
    if (_Y[i] > 0) {
      _gX[i] += _gZ[i] * (_aux[i] - 1);
    } else {
      _gX[i] += _gZ[i] * _aux[i];
    }
  }
}

}  // namespace

SigmoidBCELossNode::SigmoidBCELossNode(std::string name, GraphNode* X,
                                       GraphNode* Y)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}

class SigmoidBCELossOp : public OpBinaryElementWiseBase {
 private:
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(SigmoidBCELossOp);

  void InitForward() override {
    OpBinaryElementWiseBase::InitForward();
    aux_.resize(X_->shape());
  }

  void Forward() override { SigmoidBCELoss(*X_, *Y_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      SigmoidBCELossBackward(*X_, *Y_, *Z_, *gZ_, gX_, aux_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(SigmoidBCELoss);

/************************************************************************/
/* SigmoidBCELoss2 */
/************************************************************************/
namespace {

template <typename T>
void SigmoidBCELoss2(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                     Tensor<T>* aux) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z, *aux);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  T p;
  for (int i = 0; i < X.total_dim(); ++i) {
    p = LLMath<T>::sigmoid(_X[i]);
    _Z[i] = -_Y[i] * LLMath<T>::safe_log(p) -
            (1 - _Y[i]) * LLMath<T>::safe_log(1 - p);
    _aux[i] = p;
  }
}

template <typename T>
void SigmoidBCELoss2Backward(const Tensor<T>& /*X*/, const Tensor<T>& Y,
                             const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                             Tensor<T>* gX, const Tensor<T>& aux) noexcept {
  DXASSERT_SAME_SHAPE(Y, gZ, *gX, aux);
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux = aux.data();
  for (int i = 0; i < Y.total_dim(); ++i) {
    _gX[i] += _gZ[i] * (_aux[i] - _Y[i]);
  }
}

}  // namespace

SigmoidBCELoss2Node::SigmoidBCELoss2Node(std::string name, GraphNode* X,
                                         GraphNode* Y)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y) {}

class SigmoidBCELoss2Op : public OpBinaryElementWiseBase {
 private:
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(SigmoidBCELoss2Op);

  void InitForward() override {
    OpBinaryElementWiseBase::InitForward();
    aux_.resize(X_->shape());
  }

  void Forward() override { SigmoidBCELoss2(*X_, *Y_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      SigmoidBCELoss2Backward(*X_, *Y_, *Z_, *gZ_, gX_, aux_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(SigmoidBCELoss2);

/************************************************************************/
/* BatchCELoss */
/************************************************************************/
namespace {

bool BatchCELossInferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
  if (!X.is_rank(2)) {
    DXERROR("Invalid X: rank of X %d must be 2.", X.rank());
    return false;
  }

  int batch = X[0];
  if (!Y.same_shape(batch, 1)) {
    Shape expected(batch, 1);
    DXERROR("Invalid Y: inconsistent shape %s vs %s.", to_string(Y).c_str(),
            to_string(expected).c_str());
    return false;
  }

  Z->resize(batch, 1);
  return true;
}

template <typename T>
void BatchCELoss(const Tensor<T>& X, const Tensor<T>& Y,
                 Tensor<T>* Z) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  int label;
  for (int i = 0; i < batch; ++i) {
    // '_X' is not checked to be a probability distribution.
    label = (int)(*_Y);
    DXASSERT(0 <= label && label < m);
    *_Z = -LLMath<T>::safe_log(_X[label]);
    _X += m;
    _Y += 1;
    _Z += 1;
  }
}

template <typename T>
void BatchCELossBackward(const Tensor<T>& X, const Tensor<T>& Y,
                         const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                         Tensor<T>* gX) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  int label;
  T x;
  for (int i = 0; i < batch; ++i) {
    // '_X' is not checked to be a probability distribution.
    label = (int)(*_Y);
    DXASSERT(0 <= label && label < m);
    x = _X[label];
    x = x < (T)1e-6 ? (T)1e-6 : x;
    _gX[label] -= *_gZ / x;
    _X += m;
    _Y += 1;
    _gZ += 1;
    _gX += m;
  }
}

}  // namespace

BatchCELossNode::BatchCELossNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchCELossInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BatchCELossOp : public OpBinaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(BatchCELossOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(BatchCELossInferShape(X_->shape(), Y_->shape(), &Zshape_));
    return Zshape_;
  }

  void Forward() override { BatchCELoss(*X_, *Y_, Z_); }

  void Backward() override {
    if (gX_) {
      BatchCELossBackward(*X_, *Y_, *Z_, *gZ_, gX_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(BatchCELoss);

/************************************************************************/
/* BatchCELoss2 */
/************************************************************************/
namespace {

bool BatchCELoss2InferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
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
  Z->resize(batch, 1);
  return true;
}

template <typename T>
void BatchCELoss2(const Tensor<T>& X, const Tensor<T>& Y,
                  Tensor<T>* Z) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T z;
  for (int i = 0; i < batch; ++i) {
    // '_X' is not checked to be a probability distribution.
    // '_Y' is not checked to be a probability distribution.
    z = 0;
    for (int j = 0; j < m; ++j) {
      z -= std::log(_X[j]) * _Y[j];
    }
    *_Z = z;
    _X += m;
    _Y += m;
    _Z += 1;
  }
}

template <typename T>
void BatchCELoss2Backward(const Tensor<T>& X, const Tensor<T>& Y,
                          const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                          Tensor<T>* gX) noexcept {
  int batch = gX->dim(0);
  int m = gX->dim(1);
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  T x;
  for (int i = 0; i < batch; ++i) {
    // '_X' is not checked to be a probability distribution.
    // '_Y' is not checked to be a probability distribution.
    for (int j = 0; j < m; ++j) {
      x = _X[j];
      x = x < (T)1e-6 ? (T)1e-6 : x;
      _gX[j] -= *_gZ * _Y[j] / x;
    }
    _X += m;
    _Y += m;
    _gZ += 1;
    _gX += m;
  }
}

}  // namespace

BatchCELoss2Node::BatchCELoss2Node(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchCELoss2InferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BatchCELoss2Op : public OpBinaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(BatchCELoss2Op);

  const Shape& InferShape() override {
    DXCHECK_THROW(BatchCELoss2InferShape(X_->shape(), Y_->shape(), &Zshape_));
    return Zshape_;
  }

  void Forward() override { BatchCELoss2(*X_, *Y_, Z_); }

  void Backward() override {
    if (gX_) {
      BatchCELoss2Backward(*X_, *Y_, *Z_, *gZ_, gX_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(BatchCELoss2);

/************************************************************************/
/* BatchSoftmaxCELoss */
/************************************************************************/
namespace {

bool BatchSoftmaxCELossInferShape(const Shape& X, const Shape& Y,
                                  Shape* Z) noexcept {
  return BatchCELossInferShape(X, Y, Z);
}

template <typename T>
void BatchSoftmaxCELoss(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                        Tensor<T>* aux) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  DXASSERT(aux->same_shape(batch, m));
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  int label;
  for (int i = 0; i < batch; ++i) {
    LLMath<T>::softmax(m, _X, _aux);
    label = (int)(*_Y);
    DXASSERT(0 <= label && label < m);
    *_Z = -LLMath<T>::safe_log(_aux[label]);
    _X += m;
    _Y += 1;
    _Z += 1;
    _aux += m;
  }
}

template <typename T>
void BatchSoftmaxCELossBackward(const Tensor<T>& /*X*/, const Tensor<T>& Y,
                                const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                                Tensor<T>* gX, const Tensor<T>& aux) noexcept {
  int batch = gX->dim(0);
  int m = gX->dim(1);
  DXASSERT(aux.same_shape(batch, m));
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux = aux.data();
  int label;
  for (int i = 0; i < batch; ++i) {
    LLMath<T>::axpy(m, *_gZ, _aux, _gX);
    label = (int)(*_Y);
    DXASSERT(0 <= label && label < m);
    _gX[label] -= *_gZ;
    _Y += 1;
    _gZ += 1;
    _gX += m;
    _aux += m;
  }
}

}  // namespace

BatchSoftmaxCELossNode::BatchSoftmaxCELossNode(std::string name, GraphNode* X,
                                               GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchSoftmaxCELossInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BatchSoftmaxCELossOp : public OpBinaryBase {
 private:
  Shape Zshape_;
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(BatchSoftmaxCELossOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(
        BatchSoftmaxCELossInferShape(X_->shape(), Y_->shape(), &Zshape_));
    return Zshape_;
  }

  void InitForward() override {
    OpBinaryBase::InitForward();
    aux_.resize(X_->shape());
  }

  void Forward() override { BatchSoftmaxCELoss(*X_, *Y_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      BatchSoftmaxCELossBackward(*X_, *Y_, *Z_, *gZ_, gX_, aux_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(BatchSoftmaxCELoss);

/************************************************************************/
/* BatchSoftmaxCELoss2 */
/************************************************************************/
namespace {

bool BatchSoftmaxCELoss2InferShape(const Shape& X, const Shape& Y,
                                   Shape* Z) noexcept {
  return BatchCELoss2InferShape(X, Y, Z);
}

template <typename T>
void BatchSoftmaxCELoss2(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                         Tensor<T>* aux) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  DXASSERT(aux->same_shape(batch, m));
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  T z;
  for (int i = 0; i < batch; ++i) {
    // '_Y' is not checked to be a probability distribution.
    LLMath<T>::softmax(m, _X, _aux);
    z = 0;
    for (int j = 0; j < m; ++j) {
      z -= std::log(_aux[j]) * _Y[j];
    }
    *_Z = z;
    _X += m;
    _Y += m;
    _Z += 1;
    _aux += m;
  }
}

template <typename T>
void BatchSoftmaxCELoss2Backward(const Tensor<T>& /*X*/, const Tensor<T>& Y,
                                 const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                                 Tensor<T>* gX, const Tensor<T>& aux) noexcept {
  int batch = gX->dim(0);
  int m = gX->dim(1);
  DXASSERT(aux.same_shape(batch, m));
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux = aux.data();
  for (int i = 0; i < batch; ++i) {
    // '_Y' is not checked to be a probability distribution.
    for (int j = 0; j < m; ++j) {
      _gX[j] += *_gZ * (_aux[j] - _Y[j]);
    }
    _Y += m;
    _gZ += 1;
    _gX += m;
    _aux += m;
  }
}

}  // namespace

BatchSoftmaxCELoss2Node::BatchSoftmaxCELoss2Node(std::string name, GraphNode* X,
                                                 GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchSoftmaxCELoss2InferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BatchSoftmaxCELoss2Op : public OpBinaryBase {
 private:
  Shape Zshape_;
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(BatchSoftmaxCELoss2Op);

  const Shape& InferShape() override {
    DXCHECK_THROW(
        BatchSoftmaxCELoss2InferShape(X_->shape(), Y_->shape(), &Zshape_));
    return Zshape_;
  }

  void InitForward() override {
    OpBinaryBase::InitForward();
    aux_.resize(X_->shape());
  }

  void Forward() override { BatchSoftmaxCELoss2(*X_, *Y_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      BatchSoftmaxCELoss2Backward(*X_, *Y_, *Z_, *gZ_, gX_, aux_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(BatchSoftmaxCELoss2);

/************************************************************************/
/* FocalLoss */
/************************************************************************/
namespace {

template <typename T>
void FocalLoss(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z, T alpha,
               T gamma, Tensor<T>* aux) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z, *aux);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  for (int i = 0; i < X.total_dim(); ++i) {
    if (_Y[i] > 0) {
      _aux[i] = -alpha * std::pow((1 - _X[i]), gamma);
      _Z[i] = _aux[i] * LLMath<T>::safe_log(_X[i]);
    } else {
      _aux[i] = -(1 - alpha) * std::pow(_X[i], gamma);
      _Z[i] = _aux[i] * LLMath<T>::safe_log(1 - _X[i]);
    }
  }
}

template <typename T>
void FocalLossBackward(const Tensor<T>& X, const Tensor<T>& Y,
                       const Tensor<T>& Z, const Tensor<T>& gZ, Tensor<T>* gX,
                       T /*alpha*/, T gamma, const Tensor<T>& aux) noexcept {
  DXASSERT_SAME_SHAPE(Y, gZ, *gX, aux);
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux = aux.data();
  T x1, x2;
  for (int i = 0; i < Y.total_dim(); ++i) {
    x1 = _X[i];
    x1 = x1 < (T)1e-6 ? (T)1e-6 : x1;
    x2 = 1 - _X[i];
    x2 = x2 < (T)1e-6 ? (T)1e-6 : x2;
    if (_Y[i] > 0) {
      _gX[i] += _gZ[i] * (_aux[i] / x1 - gamma * _Z[i] / x2);
    } else {
      _gX[i] += _gZ[i] * (gamma * _Z[i] / x1 - _aux[i] / x2);
    }
  }
}

}  // namespace

FocalLossNode::FocalLossNode(std::string name, GraphNode* X, GraphNode* Y,
                             double alpha, double gamma)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y),
      alpha_(alpha),
      gamma_(gamma) {
  DXCHECK_THROW(gamma_ >= 0);
  DXCHECK_THROW(0 <= alpha_ && alpha_ <= 1);
}

class FocalLossOp : public OpBinaryElementWiseBase {
 private:
  tsr_t aux_;
  float_t alpha_ = 0;
  float_t gamma_ = 0;

 public:
  DEFINE_OP_LIKE(FocalLossOp);

  void InitForward() override {
    OpBinaryElementWiseBase::InitForward();
    aux_.resize(X_->shape());
    alpha_ = (float_t)((const FocalLossNode*)node_)->alpha();
    gamma_ = (float_t)((const FocalLossNode*)node_)->gamma();
  }

  void Forward() override { FocalLoss(*X_, *Y_, Z_, alpha_, gamma_, &aux_); }

  void Backward() override {
    if (gX_) {
      FocalLossBackward(*X_, *Y_, *Z_, *gZ_, gX_, alpha_, gamma_, aux_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(FocalLoss);

/************************************************************************/
/* SigmoidFocalLoss */
/************************************************************************/
namespace {

template <typename T>
void SigmoidFocalLoss(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                      T alpha, T gamma, Tensor<T>* aux1,
                      Tensor<T>* aux2) noexcept {
  DXASSERT_SAME_SHAPE(X, Y, *Z, *aux1, *aux2);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  T* _aux1 = aux1->data();
  T* _aux2 = aux2->data();
  T p;
  for (int i = 0; i < X.total_dim(); ++i) {
    p = LLMath<T>::sigmoid(_X[i]);
    if (_Y[i] > 0) {
      _aux1[i] = -alpha * std::pow(1 - p, gamma);
      _Z[i] = _aux1[i] * LLMath<T>::safe_log(p);
    } else {
      _aux1[i] = -(1 - alpha) * std::pow(p, gamma);
      _Z[i] = _aux1[i] * LLMath<T>::safe_log(1 - p);
    }
    _aux2[i] = p;
  }
}

template <typename T>
void SigmoidFocalLossBackward(const Tensor<T>& /*X*/, const Tensor<T>& Y,
                              const Tensor<T>& Z, const Tensor<T>& gZ,
                              Tensor<T>* gX, T /*alpha*/, T gamma,
                              const Tensor<T>& aux1, const Tensor<T>& aux2) {
  DXASSERT_SAME_SHAPE(Y, gZ, *gX, aux1, aux2);
  const T* _Y = Y.data();
  const T* _Z = Z.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  const T* _aux1 = aux1.data();
  const T* _aux2 = aux2.data();
  for (int i = 0; i < Y.total_dim(); ++i) {
    if (_Y[i] > 0) {
      _gX[i] +=
          _gZ[i] * (gamma * _Z[i] * (-_aux2[i]) + _aux1[i] * (1 - _aux2[i]));
    } else {
      _gX[i] +=
          _gZ[i] * (gamma * _Z[i] * (1 - _aux2[i]) + _aux1[i] * (-_aux2[i]));
    }
  }
}

}  // namespace

SigmoidFocalLossNode::SigmoidFocalLossNode(std::string name, GraphNode* X,
                                           GraphNode* Y, double alpha,
                                           double gamma)
    : GraphNodeBinaryElementWiseBase(std::move(name), X, Y),
      alpha_(alpha),
      gamma_(gamma) {
  DXCHECK_THROW(gamma_ >= 0);
  DXCHECK_THROW(0 <= alpha_ && alpha_ <= 1);
}

class SigmoidFocalLossOp : public OpBinaryElementWiseBase {
 private:
  tsr_t aux1_;
  tsr_t aux2_;

  float_t alpha_ = 0;
  float_t gamma_ = 0;

 public:
  DEFINE_OP_LIKE(SigmoidFocalLossOp);

  void InitForward() override {
    OpBinaryElementWiseBase::InitForward();
    aux1_.resize(X_->shape());
    aux2_.resize(X_->shape());
    alpha_ = (float_t)((const SigmoidFocalLossNode*)node_)->alpha();
    gamma_ = (float_t)((const SigmoidFocalLossNode*)node_)->gamma();
  }

  void Forward() override {
    SigmoidFocalLoss(*X_, *Y_, Z_, alpha_, gamma_, &aux1_, &aux2_);
  }

  void Backward() override {
    if (gX_) {
      SigmoidFocalLossBackward(*X_, *Y_, *Z_, *gZ_, gX_, alpha_, gamma_, aux1_,
                               aux2_);
    }
    // gY is not computed.
  }
};

GRAPH_NODE_OP_REGISTER(SigmoidFocalLoss);

}  // namespace deepx_core
