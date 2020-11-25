// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct LayerNormAux {
  Shape Z;
  int batch = 0;
  int m = 0;
};

bool LayerNormPrepare(const Shape& X, const Shape& gamma, const Shape& beta,
                      LayerNormAux* aux) {
  if (X.rank() <= 1) {
    DXERROR("Invalid X: rank of X %d must be greater than 1.", X.rank());
    return false;
  }

  int m = X.total_dim() / X[0];
  if (!gamma.same_shape(m)) {
    Shape expected(m);
    DXERROR("Invalid gamma: inconsistent shape %s vs %s.",
            to_string(gamma).c_str(), to_string(expected).c_str());
    return false;
  }

  if (!beta.same_shape(m)) {
    Shape expected(m);
    DXERROR("Invalid beta: inconsistent shape %s vs %s.",
            to_string(beta).c_str(), to_string(expected).c_str());
    return false;
  }

  aux->Z = X;
  aux->batch = X[0];
  aux->m = m;
  return true;
}

bool LayerNormInferShape(const Shape& X, const Shape& gamma, const Shape& beta,
                         Shape* Z) {
  LayerNormAux aux;
  if (!LayerNormPrepare(X, gamma, beta, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
struct LayerNormMutableAux {
  Tensor<T> mean;     // mean(X, axis=1)
  Tensor<T> var;      // var(X, axis=1)
  Tensor<T> inv_std;  // 1 / sqrt(var + eps)
  Tensor<T> Z1;       // X - mean
  Tensor<T> Z2;       // (X - mean) * (X - mean)
  Tensor<T> Z3;       // (X - mean) / sqrt(var + eps)

  Tensor<T> gvar;
  Tensor<T> gZ3;
  Tensor<T> I;
  Tensor<T> buf1;
  Tensor<T> buf2;
};

template <typename T>
void LayerNormPrepare(const LayerNormAux& aux, LayerNormMutableAux<T>* maux) {
  maux->mean.resize(aux.batch);
  maux->var.resize(aux.batch);
  maux->inv_std.resize(aux.batch);
  maux->Z1.resize(aux.Z);
  maux->Z2.resize(aux.Z);
  maux->Z3.resize(aux.Z);
}

template <typename T>
void LayerNormPrepareBackward(const LayerNormAux& aux,
                              LayerNormMutableAux<T>* maux) {
  maux->gvar.resize(aux.batch);
  maux->gZ3.resize(aux.Z);
  maux->I.resize(aux.Z);
  maux->buf1.resize(aux.Z);
  maux->buf2.resize(aux.batch);
}

template <typename T>
void LayerNorm(const Tensor<T>& X, const Tensor<T>& gamma,
               const Tensor<T>& beta, Tensor<T>* Z, const LayerNormAux& aux,
               LayerNormMutableAux<T>* maux) {
  int batch = aux.batch;
  int m = aux.m;
  int total_dim = batch * m;
  const T* _X = X.data();
  const T* _gamma = gamma.data();
  const T* _beta = beta.data();
  T* _Z = Z->data();
  T* _mean = maux->mean.data();
  T* _var = maux->var.data();
  T* _inv_std = maux->inv_std.data();
  T* _Z1 = maux->Z1.data();
  T* _Z2 = maux->Z2.data();
  T* _Z3 = maux->Z3.data();
  LLMath<T>::sum_col(batch, m, (T)1 / m, _X, 0, _mean);
  LLMath<T>::sub_col(batch, m, 1, _X, 1, _mean, _Z1);
  LLMath<T>::square(total_dim, _Z1, _Z2);
  LLMath<T>::sum_col(batch, m, (T)1 / m, _Z2, 0, _var);
  LLMath<T>::add_scalar(batch, _var, (T)1e-6, _inv_std);
  LLMath<T>::sqrt(batch, _inv_std, _inv_std);
  LLMath<T>::inv(batch, _inv_std, _inv_std);
  LLMath<T>::mul_col(batch, m, _Z1, _inv_std, _Z3);
  LLMath<T>::mul_row(batch, m, _Z3, _gamma, _Z);
  LLMath<T>::add_row(batch, m, 1, _Z, 1, _beta, _Z);
}

template <typename T>
void LayerNormBackward(const Tensor<T>& /*X*/, const Tensor<T>& gamma,
                       const Tensor<T>& /*beta*/, const Tensor<T>& /*Z*/,
                       const Tensor<T>& gZ, Tensor<T>* gX, Tensor<T>* ggamma,
                       Tensor<T>* gbeta, const LayerNormAux& aux,
                       LayerNormMutableAux<T>* maux) {
  int batch = aux.batch;
  int m = aux.m;
  int total_dim = batch * m;
  const T* _gamma = gamma.data();
  const T* _gZ = gZ.data();
  T* _inv_std = maux->inv_std.data();
  T* _Z1 = maux->Z1.data();
  T* _Z3 = maux->Z3.data();
  T* _gvar = maux->gvar.data();
  T* _gZ3 = maux->gZ3.data();
  T* _I = maux->I.data();
  T* _buf1 = maux->buf1.data();
  T* _buf2 = maux->buf2.data();

  if (gX) {
    T* _gX = gX->data();
    // gZ3
    LLMath<T>::mul_row(batch, m, _gZ, _gamma, _gZ3);
    // gvar
    LLMath<T>::mul(total_dim, _gZ3, _Z1, _buf1);
    LLMath<T>::cubic(batch, _inv_std, _buf2);
    LLMath<T>::mul_col(batch, m, _buf1, _buf2, _buf1);
    LLMath<T>::sum_col(batch, m, (T)-0.5, _buf1, 0, _gvar);
    // I
    LLMath<T>::mul_col(batch, m, _gZ3, _inv_std, _I);
    LLMath<T>::mul_col(batch, m, _Z1, _gvar, _buf1);
    LLMath<T>::mul_scalar(total_dim, _buf1, (T)2 / m, _buf1);
    LLMath<T>::add(total_dim, _I, _buf1, _I);
    // gX
    LLMath<T>::sum_col(batch, m, (T)-1 / m, _I, 0, _buf2);
    LLMath<T>::add_col(batch, m, 1, _I, 1, _buf2, _buf1);
    LLMath<T>::add(total_dim, _gX, _buf1, _gX);
  }

  if (ggamma) {
    T* _ggamma = ggamma->data();
    LLMath<T>::mul(total_dim, _gZ, _Z3, _buf1);
    LLMath<T>::sum_row(batch, m, 1, _buf1, 1, _ggamma);
  }

  if (gbeta) {
    T* _gbeta = gbeta->data();
    LLMath<T>::sum_row(batch, m, 1, _gZ, 1, _gbeta);
  }
}

}  // namespace

LayerNormNode::LayerNormNode(std::string name, GraphNode* X, GraphNode* gamma,
                             GraphNode* beta)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(gamma->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(beta->tensor_type() == TENSOR_TYPE_TSR);
  input_ = {X, gamma, beta};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (!X->shape().empty() && !gamma->shape().empty() &&
      !beta->shape().empty()) {
    (void)LayerNormInferShape(X->shape(), gamma->shape(), beta->shape(),
                              &shape_);
  }
}

class LayerNormOp : public OpImpl {
 private:
  const tsr_t* X_ = nullptr;
  const tsr_t* gamma_ = nullptr;
  const tsr_t* beta_ = nullptr;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* ggamma_ = nullptr;
  tsr_t* gbeta_ = nullptr;
  LayerNormAux aux_;
  LayerNormMutableAux<float_t> maux_;

 public:
  DEFINE_OP_LIKE(LayerNormOp);

  void InitForward() override {
    X_ = GetPtrTSR(node_->input(0));
    gamma_ = GetPtrTSR(node_->input(1));
    beta_ = GetPtrTSR(node_->input(2));
    DXCHECK_THROW(
        LayerNormPrepare(X_->shape(), gamma_->shape(), beta_->shape(), &aux_));
    Z_ = InitHiddenTSR(node_, aux_.Z);
    LayerNormPrepare(aux_, &maux_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(node_->input(0), X_->shape());
    ggamma_ = InitGradTSR(node_->input(1), gamma_->shape());
    gbeta_ = InitGradTSR(node_->input(2), beta_->shape());
    if (gX_ || ggamma_ || gbeta_) {
      LayerNormPrepareBackward(aux_, &maux_);
    }
  }

  void Forward() override { LayerNorm(*X_, *gamma_, *beta_, Z_, aux_, &maux_); }

  void Backward() override {
    if (gZ_) {
      LayerNormBackward(*X_, *gamma_, *beta_, *Z_, *gZ_, gX_, ggamma_, gbeta_,
                        aux_, &maux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(LayerNorm);

}  // namespace deepx_core
