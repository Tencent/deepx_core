// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

/************************************************************************/
/* BatchFMInteraction */
/************************************************************************/
namespace {

bool BatchFMInteractionInferShape(const Shape& X, Shape* Z) noexcept {
  if (!X.is_rank(3)) {
    DXERROR("Invalid X: rank of X %d must be 3.", X.rank());
    return false;
  }

  int batch = X[0];
  int m = X[1];
  int n = X[2];
  Z->resize(batch, m * (m - 1) / 2, n);
  return true;
}

template <typename T>
void BatchFMInteraction(const Tensor<T>& X, Tensor<T>* Z) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  int n = X.dim(2);
  const T* _X = X.data();
  T* _Z = Z->data();
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = j + 1; k < m; ++k) {
        LLMath<T>::mul(n, _X + j * n, _X + k * n, _Z);
        _Z += n;
      }
    }
    _X += m * n;
  }
}

template <typename T>
void BatchFMInteractionBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                                const Tensor<T>& gZ, Tensor<T>* gX) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  int n = X.dim(2);
  const T* _X = X.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = j + 1; k < m; ++k) {
        LLMath<T>::xypz(n, _gZ, _X + k * n, _gX + j * n);
        LLMath<T>::xypz(n, _gZ, _X + j * n, _gX + k * n);
        _gZ += n;
      }
    }
    _X += m * n;
    _gX += m * n;
  }
}

}  // namespace

BatchFMInteractionNode::BatchFMInteractionNode(std::string name, GraphNode* X)
    : GraphNodeUnaryBase(std::move(name), X) {
  if (!X->shape().empty()) {
    (void)BatchFMInteractionInferShape(X->shape(), &shape_);
  }
}

class BatchFMInteractionOp : public OpUnaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(BatchFMInteractionOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(BatchFMInteractionInferShape(X_->shape(), &Zshape_));
    return Zshape_;
  }

  void Forward() override { BatchFMInteraction(*X_, Z_); }

  void Backward() override {
    if (gX_) {
      BatchFMInteractionBackward(*X_, *Z_, *gZ_, gX_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchFMInteraction);
// backward compatibility
CLASS_FACTORY_REGISTER(GraphNode, BatchFMInteractionNode,
                       "BatchPairWiseInteractionNode");

/************************************************************************/
/* BatchFMInteraction2 */
/************************************************************************/
namespace {

bool BatchFMInteraction2InferShape(const Shape& X, const Shape& Y,
                                   Shape* Z) noexcept {
  if (!X.is_rank(3)) {
    DXERROR("Invalid X: rank of X %d must be 3.", X.rank());
    return false;
  }

  if (!Y.is_rank(3)) {
    DXERROR("Invalid Y: rank of Y %d must be 3.", Y.rank());
    return false;
  }

  int batch = X[0];
  int m1 = X[1];
  int n = X[2];
  int m2 = Y[1];

  if (Y[0] != batch) {
    DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", batch, Y[0]);
    return false;
  }

  if (Y[2] != n) {
    DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", n, Y[2]);
    return false;
  }

  Z->resize(batch, m1 * m2, n);
  return true;
}

template <typename T>
void BatchFMInteraction2(const Tensor<T>& X, const Tensor<T>& Y,
                         Tensor<T>* Z) noexcept {
  int batch = X.dim(0);
  int m1 = X.dim(1);
  int n = X.dim(2);
  int m2 = Y.dim(1);
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < m1; ++j) {
      for (int k = 0; k < m2; ++k) {
        LLMath<T>::mul(n, _X + j * n, _Y + k * n, _Z);
        _Z += n;
      }
    }
    _X += m1 * n;
    _Y += m2 * n;
  }
}

template <typename T>
void BatchFMInteraction2Backward(const Tensor<T>& X, const Tensor<T>& Y,
                                 const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                                 Tensor<T>* gX, Tensor<T>* gY) noexcept {
  int batch = X.dim(0);
  int m1 = X.dim(1);
  int n = X.dim(2);
  int m2 = Y.dim(1);
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  if (gX && gY) {
    T* _gX = gX->data();
    T* _gY = gY->data();
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < m1; ++j) {
        for (int k = 0; k < m2; ++k) {
          LLMath<T>::xypz(n, _gZ, _Y + k * n, _gX + j * n);
          LLMath<T>::xypz(n, _gZ, _X + j * n, _gY + k * n);
          _gZ += n;
        }
      }
      _X += m1 * n;
      _gX += m1 * n;
      _Y += m2 * n;
      _gY += m2 * n;
    }
  } else if (gX) {
    T* _gX = gX->data();
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < m1; ++j) {
        for (int k = 0; k < m2; ++k) {
          LLMath<T>::xypz(n, _gZ, _Y + k * n, _gX + j * n);
          _gZ += n;
        }
      }
      _X += m1 * n;
      _gX += m1 * n;
      _Y += m2 * n;
    }
  } else if (gY) {
    T* _gY = gY->data();
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < m1; ++j) {
        for (int k = 0; k < m2; ++k) {
          LLMath<T>::xypz(n, _gZ, _X + j * n, _gY + k * n);
          _gZ += n;
        }
      }
      _X += m1 * n;
      _Y += m2 * n;
      _gY += m2 * n;
    }
  }
}

}  // namespace

BatchFMInteraction2Node::BatchFMInteraction2Node(std::string name, GraphNode* X,
                                                 GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchFMInteraction2InferShape(X->shape(), Y->shape(), &shape_);
  }
}

class BatchFMInteraction2Op : public OpBinaryBase {
 private:
  Shape Zshape_;

 public:
  DEFINE_OP_LIKE(BatchFMInteraction2Op);

  const Shape& InferShape() override {
    DXCHECK_THROW(
        BatchFMInteraction2InferShape(X_->shape(), Y_->shape(), &Zshape_));
    return Zshape_;
  }

  void Forward() override { BatchFMInteraction2(*X_, *Y_, Z_); }

  void Backward() override {
    if (gZ_) {
      BatchFMInteraction2Backward(*X_, *Y_, *Z_, *gZ_, gX_, gY_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchFMInteraction2);
// backward compatibility
CLASS_FACTORY_REGISTER(GraphNode, BatchFMInteraction2Node,
                       "BatchPairWiseInteraction2Node");

/************************************************************************/
/* BatchFMQuadratic */
/************************************************************************/
namespace {

bool BatchFMQuadraticInferShape(int Xrow, const Shape& V, Shape* Z) noexcept {
  if (!V.is_rank(2)) {
    DXERROR("Invalid V: rank of V %d must be 2.", V.rank());
    return false;
  }

  Z->resize(Xrow, 1);
  return true;
}

template <typename T, typename I>
void BatchFMQuadratic(const CSRMatrix<T, I>& X, const Tensor<T>& V,
                      Tensor<T>* Z) noexcept {
  int m = V.dim(0);
  int n = V.dim(1);
  T* _Z = Z->data();
  Z->zeros();
  CSR_FOR_EACH_ROW(X, i) {
    for (int k = 0; k < n; ++k) {
      T sum1 = 0, sum2 = 0, tmp;
      CSR_FOR_EACH_COL(X, i) {
        const T* _V = V.data() + (CSR_COL(X) % m) * n;
        tmp = CSR_VALUE(X) * _V[k];
        sum1 += tmp;
        sum2 += tmp * tmp;
      }
      _Z[i] += (sum1 * sum1 - sum2) * (T)0.5;
    }
  }
}

template <typename T, typename I>
void BatchFMQuadraticBackward(const CSRMatrix<T, I>& X, const Tensor<T>& V,
                              const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                              SparseRowMatrix<T, I>* gV,
                              Tensor<T>* aux) noexcept {
  int m = V.dim(0);
  int n = V.dim(1);
  DXASSERT_TOTAL_DIM(*aux, n);
  T* _aux = aux->data();
  CSR_FOR_EACH_ROW(X, i) {
    T _gZ = gZ.data(i);
    aux->zeros();
    CSR_FOR_EACH_COL(X, i) {
      T Xij = CSR_VALUE(X);
      I j = CSR_COL(X) % m;
      const T* _V = V.data() + j * n;
      LLMath<T>::axpy(n, Xij, _V, _aux);
    }
    CSR_FOR_EACH_COL(X, i) {
      T Xij = CSR_VALUE(X);
      I j = CSR_COL(X) % m;
      const T* _V = V.data() + j * n;
      T* _gV = gV->get_row_no_init(j);
      for (int k = 0; k < n; ++k) {
        _gV[k] += _gZ * Xij * (_aux[k] - Xij * _V[k]);
      }
    }
  }
}

}  // namespace

BatchFMQuadraticNode::BatchFMQuadraticNode(std::string name, GraphNode* X,
                                           GraphNode* V)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->node_type() == GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_CSR);
  DXCHECK_THROW(V->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {X, V};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (X->shape().is_rank(2) && !V->shape().empty()) {
    (void)BatchFMQuadraticInferShape(X->shape()[0], V->shape(), &shape_);
  }
}

class BatchFMQuadraticOp : public OpImpl {
 private:
  const csr_t* X_ = nullptr;
  const tsr_t* V_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  srm_t* gV_ = nullptr;
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(BatchFMQuadraticOp);

  void InitForward() override {
    DXCHECK_THROW(!node_->input(0)->need_grad());
    X_ = GetPtrCSR(node_->input(0));
    V_ = GetPtrTSR(node_->input(1));
    DXCHECK_THROW(BatchFMQuadraticInferShape(X_->row(), V_->shape(), &Zshape_));
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gV_ = InitGradSRM(node_->input(1), V_->shape());
    if (gV_) {
      aux_.resize(gV_->col());
    }
  }

  void Forward() override { BatchFMQuadratic(*X_, *V_, Z_); }

  void Backward() override {
    if (gV_) {
      BatchFMQuadraticBackward(*X_, *V_, *Z_, *gZ_, gV_, &aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchFMQuadratic);

/************************************************************************/
/* BatchGroupFMQuadratic */
/************************************************************************/
namespace {

bool BatchGroupFMQuadraticInferShape(const Shape& X, Shape* Z) noexcept {
  if (!X.is_rank(3)) {
    DXERROR("Invalid X: rank of X %d must be 3.", X.rank());
    return false;
  }

  int batch = X[0];
  Z->resize(batch, 1);
  return true;
}

template <typename T>
void BatchGroupFMQuadratic(const Tensor<T>& X, Tensor<T>* Z,
                           Tensor<T>* aux) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  int n = X.dim(2);
  DXASSERT(aux->same_shape(batch, n));
  const T* _X = X.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  Z->zeros();
  for (int i = 0; i < batch; ++i) {
    for (int k = 0; k < n; ++k) {
      T sum1 = 0, sum2 = 0;
      for (int j = 0; j < m; ++j) {
        T Xjk = _X[j * n + k];
        sum1 += Xjk;
        sum2 += Xjk * Xjk;
      }
      _aux[k] = sum1;
      *_Z += (sum1 * sum1 - sum2) * (T)0.5;
    }
    _X += m * n;
    _Z += 1;
    _aux += n;
  }
}

template <typename T>
void BatchGroupFMQuadraticBackward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                                   const Tensor<T>& gZ, Tensor<T>* gX,
                                   Tensor<T>* aux) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  int n = X.dim(2);
  DXASSERT(aux->same_shape(batch, n));
  const T* _X = X.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  T* _aux = aux->data();
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < n; ++k) {
        _gX[k] += *_gZ * (_aux[k] - _X[k]);
      }
      _X += n;
      _gX += n;
    }
    _gZ += 1;
    _aux += n;
  }
}

}  // namespace

BatchGroupFMQuadraticNode::BatchGroupFMQuadraticNode(std::string name,
                                                     GraphNode* X)
    : GraphNodeUnaryBase(std::move(name), X) {
  if (!X->shape().empty()) {
    (void)BatchGroupFMQuadraticInferShape(X->shape(), &shape_);
  }
}

class BatchGroupFMQuadraticOp : public OpUnaryBase {
 private:
  Shape Zshape_;
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(BatchGroupFMQuadraticOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(BatchGroupFMQuadraticInferShape(X_->shape(), &Zshape_));
    return Zshape_;
  }

  void InitForward() override {
    OpUnaryBase::InitForward();
    aux_.resize(X_->dim(0), X_->dim(2));
  }

  void Forward() override { BatchGroupFMQuadratic(*X_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      BatchGroupFMQuadraticBackward(*X_, *Z_, *gZ_, gX_, &aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchGroupFMQuadratic);

/************************************************************************/
/* BatchGroupFMQuadratic2 */
/************************************************************************/
namespace {

bool BatchGroupFMQuadratic2InferShape(const Shape& X, Shape* Z) noexcept {
  if (!X.is_rank(3)) {
    DXERROR("Invalid X: rank of X %d must be 3.", X.rank());
    return false;
  }

  int batch = X[0];
  int n = X[2];
  Z->resize(batch, n);
  return true;
}

template <typename T>
void BatchGroupFMQuadratic2(const Tensor<T>& X, Tensor<T>* Z,
                            Tensor<T>* aux) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  int n = X.dim(2);
  DXASSERT(aux->same_shape(batch, n));
  const T* _X = X.data();
  T* _Z = Z->data();
  T* _aux = aux->data();
  Z->zeros();
  for (int i = 0; i < batch; ++i) {
    for (int k = 0; k < n; ++k) {
      T sum1 = 0, sum2 = 0;
      for (int j = 0; j < m; ++j) {
        T Xjk = _X[j * n + k];
        sum1 += Xjk;
        sum2 += Xjk * Xjk;
      }
      _aux[k] = sum1;
      _Z[k] += (sum1 * sum1 - sum2) * (T)0.5;
    }
    _X += m * n;
    _Z += n;
    _aux += n;
  }
}

template <typename T>
void BatchGroupFMQuadratic2Backward(const Tensor<T>& X, const Tensor<T>& /*Z*/,
                                    const Tensor<T>& gZ, Tensor<T>* gX,
                                    Tensor<T>* aux) noexcept {
  int batch = X.dim(0);
  int m = X.dim(1);
  int n = X.dim(2);
  DXASSERT(aux->same_shape(batch, n));
  const T* _X = X.data();
  const T* _gZ = gZ.data();
  T* _gX = gX->data();
  T* _aux = aux->data();
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < n; ++k) {
        _gX[k] += _gZ[k] * (_aux[k] - _X[k]);
      }
      _X += n;
      _gX += n;
    }
    _gZ += n;
    _aux += n;
  }
}

}  // namespace

BatchGroupFMQuadratic2Node::BatchGroupFMQuadratic2Node(std::string name,
                                                       GraphNode* X)
    : GraphNodeUnaryBase(std::move(name), X) {
  if (!X->shape().empty()) {
    (void)BatchGroupFMQuadratic2InferShape(X->shape(), &shape_);
  }
}

class BatchGroupFMQuadratic2Op : public OpUnaryBase {
 private:
  Shape Zshape_;
  tsr_t aux_;

 public:
  DEFINE_OP_LIKE(BatchGroupFMQuadratic2Op);

  const Shape& InferShape() override {
    DXCHECK_THROW(BatchGroupFMQuadratic2InferShape(X_->shape(), &Zshape_));
    return Zshape_;
  }

  void InitForward() override {
    OpUnaryBase::InitForward();
    aux_.resize(X_->dim(0), X_->dim(2));
  }

  void Forward() override { BatchGroupFMQuadratic2(*X_, Z_, &aux_); }

  void Backward() override {
    if (gX_) {
      BatchGroupFMQuadratic2Backward(*X_, *Z_, *gZ_, gX_, &aux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchGroupFMQuadratic2);

}  // namespace deepx_core
