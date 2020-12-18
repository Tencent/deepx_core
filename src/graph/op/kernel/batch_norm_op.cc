// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
// Author: Chunchen Su (hillsu@tencent.com)
//

#include <deepx_core/graph/op_impl.h>

namespace deepx_core {
namespace {

struct BatchNormAux {
  Shape Z;
  int batch = 0;
  int m = 0;
};

bool BatchNormPrepare(const Shape& X, const Shape& gamma, const Shape& beta,
                      BatchNormAux* aux) {
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

bool BatchNormInferShape(const Shape& X, const Shape& gamma, const Shape& beta,
                         Shape* Z) {
  BatchNormAux aux;
  if (!BatchNormPrepare(X, gamma, beta, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
struct BatchNormMutableAux {
  Tensor<T> mean;     // mean(X, axis=0)
  Tensor<T> var;      // var(X, axis=0)
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
void BatchNormPrepare(const BatchNormAux& aux, BatchNormMutableAux<T>* maux) {
  maux->mean.resize(aux.m);
  maux->var.resize(aux.m);
  maux->inv_std.resize(aux.m);
  maux->Z1.resize(aux.Z);
  maux->Z2.resize(aux.Z);
  maux->Z3.resize(aux.Z);
}

template <typename T>
void BatchNormPrepareBackward(const BatchNormAux& aux,
                              BatchNormMutableAux<T>* maux) {
  maux->gvar.resize(aux.m);
  maux->gZ3.resize(aux.Z);
  maux->I.resize(aux.Z);
  maux->buf1.resize(aux.Z);
  maux->buf2.resize(aux.m);
}

template <typename T>
void BatchNorm(const Tensor<T>& X, const Tensor<T>& gamma,
               const Tensor<T>& beta, Tensor<T>* Z, const BatchNormAux& aux,
               BatchNormMutableAux<T>* maux) {
  int batch = aux.batch;
  int m = aux.m;
  int total_dim = batch * m;
  const auto* _X = X.data();
  const auto* _gamma = gamma.data();
  const auto* _beta = beta.data();
  auto* _Z = Z->data();
  auto* _mean = maux->mean.data();
  auto* _var = maux->var.data();
  auto* _inv_std = maux->inv_std.data();
  auto* _Z1 = maux->Z1.data();
  auto* _Z2 = maux->Z2.data();
  auto* _Z3 = maux->Z3.data();
  LLMath<T>::sum_row(batch, m, (T)1 / batch, _X, 0, _mean);
  LLMath<T>::sub_row(batch, m, 1, _X, 1, _mean, _Z1);
  LLMath<T>::square(total_dim, _Z1, _Z2);
  LLMath<T>::sum_row(batch, m, (T)1 / batch, _Z2, 0, _var);
  LLMath<T>::add_scalar(m, _var, (T)1e-6, _inv_std);
  LLMath<T>::sqrt(m, _inv_std, _inv_std);
  LLMath<T>::inv(m, _inv_std, _inv_std);
  LLMath<T>::mul_row(batch, m, _Z1, _inv_std, _Z3);
  LLMath<T>::mul_row(batch, m, _Z3, _gamma, _Z);
  LLMath<T>::add_row(batch, m, 1, _Z, 1, _beta, _Z);
}

// for forward
template <typename T>
void BatchNorm(const Tensor<T>& X, const Tensor<T>& gamma,
               const Tensor<T>& beta, Tensor<T>* Z, const BatchNormAux& aux,
               BatchNormMutableAux<T>* maux, T moving_decay,
               Tensor<T>* moving_mean, Tensor<T>* moving_var) {
  BatchNorm(X, gamma, beta, Z, aux, maux);
  int m = aux.m;
  auto* _mean = maux->mean.data();
  auto* _var = maux->var.data();
  auto* _moving_mean = moving_mean->data();
  auto* _moving_var = moving_var->data();
  T a = moving_decay;
  T b = 1 - moving_decay;
  T new_mean, new_var;
  for (int i = 0; i < m; ++i) {
    new_mean = a * _moving_mean[i] + b * _mean[i];
    new_var = a * _moving_var[i] + b * _var[i];
    _moving_mean[i] = new_mean;
    _moving_var[i] = new_var;
  }
}

// for predict
template <typename T>
void BatchNorm(const Tensor<T>& X, const Tensor<T>& mean, const Tensor<T>& var,
               const Tensor<T>& gamma, const Tensor<T>& beta, Tensor<T>* Z,
               const BatchNormAux& aux, BatchNormMutableAux<T>* maux) {
  int batch = aux.batch;
  int m = aux.m;
  const auto* _X = X.data();
  const auto* _mean = mean.data();
  const auto* _var = var.data();
  const auto* _gamma = gamma.data();
  const auto* _beta = beta.data();
  auto* _Z = Z->data();
  auto* _inv_std = maux->inv_std.data();
  auto* _Z1 = maux->Z1.data();
  auto* _Z3 = maux->Z3.data();
  LLMath<T>::sub_row(batch, m, 1, _X, 1, _mean, _Z1);
  LLMath<T>::add_scalar(m, _var, (T)1e-6, _inv_std);
  LLMath<T>::sqrt(m, _inv_std, _inv_std);
  LLMath<T>::inv(m, _inv_std, _inv_std);
  LLMath<T>::mul_row(batch, m, _Z1, _inv_std, _Z3);
  LLMath<T>::mul_row(batch, m, _Z3, _gamma, _Z);
  LLMath<T>::add_row(batch, m, 1, _Z, 1, _beta, _Z);
}

template <typename T>
void BatchNormBackward(const Tensor<T>& /*X*/, const Tensor<T>& gamma,
                       const Tensor<T>& /*beta*/, const Tensor<T>& /*Z*/,
                       const Tensor<T>& gZ, Tensor<T>* gX, Tensor<T>* ggamma,
                       Tensor<T>* gbeta, const BatchNormAux& aux,
                       BatchNormMutableAux<T>* maux) {
  int batch = aux.batch;
  int m = aux.m;
  int total_dim = batch * m;
  const auto* _gamma = gamma.data();
  const auto* _gZ = gZ.data();
  auto* _inv_std = maux->inv_std.data();
  auto* _Z1 = maux->Z1.data();
  auto* _Z3 = maux->Z3.data();
  auto* _gvar = maux->gvar.data();
  auto* _gZ3 = maux->gZ3.data();
  auto* _I = maux->I.data();
  auto* _buf1 = maux->buf1.data();
  auto* _buf2 = maux->buf2.data();

  if (gX) {
    auto* _gX = gX->data();
    // gZ3
    LLMath<T>::mul_row(batch, m, _gZ, _gamma, _gZ3);
    // gvar
    LLMath<T>::mul(total_dim, _gZ3, _Z1, _buf1);
    LLMath<T>::cubic(m, _inv_std, _buf2);
    LLMath<T>::mul_row(batch, m, _buf1, _buf2, _buf1);
    LLMath<T>::sum_row(batch, m, (T)-0.5, _buf1, 0, _gvar);
    // I
    LLMath<T>::mul_row(batch, m, _gZ3, _inv_std, _I);
    LLMath<T>::mul_row(batch, m, _Z1, _gvar, _buf1);
    LLMath<T>::mul_scalar(total_dim, _buf1, (T)2 / batch, _buf1);
    LLMath<T>::add(total_dim, _I, _buf1, _I);
    // gX
    LLMath<T>::sum_row(batch, m, (T)-1 / batch, _I, 0, _buf2);
    LLMath<T>::add_row(batch, m, 1, _I, 1, _buf2, _buf1);
    LLMath<T>::add(total_dim, _gX, _buf1, _gX);
  }

  if (ggamma) {
    auto* _ggamma = ggamma->data();
    LLMath<T>::mul(total_dim, _gZ, _Z3, _buf1);
    LLMath<T>::sum_row(batch, m, 1, _buf1, 1, _ggamma);
  }

  if (gbeta) {
    auto* _gbeta = gbeta->data();
    LLMath<T>::sum_row(batch, m, 1, _gZ, 1, _gbeta);
  }
}

}  // namespace

BatchNormNode::BatchNormNode(std::string name, GraphNode* X, GraphNode* gamma,
                             GraphNode* beta, GraphNode* mean, GraphNode* var,
                             double moving_decay)
    : GraphNode(std::move(name)), moving_decay_(moving_decay) {
  DXCHECK_THROW(X->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(gamma->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(beta->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(mean->tensor_type() == TENSOR_TYPE_TSR);
  DXCHECK_THROW(var->tensor_type() == TENSOR_TYPE_TSR);
  // Shape of 'gamma' must be valid.
  DXCHECK_THROW(gamma->shape().rank() > 0);
  DXCHECK_THROW(gamma->shape().total_dim() > 0);
  // Shape of 'beta' must be valid.
  DXCHECK_THROW(gamma->shape() == beta->shape());
  // Shape of 'mean' must be valid.
  DXCHECK_THROW(gamma->shape() == mean->shape());
  // Shape of 'var' must be valid.
  DXCHECK_THROW(gamma->shape() == var->shape());
  DXCHECK_THROW(0 < moving_decay_ && moving_decay_ <= 1);

  input_ = {X, gamma, beta, mean, var};
  node_type_ = GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = TENSOR_TYPE_TSR;

  if (!X->shape().empty()) {
    (void)BatchNormInferShape(X->shape(), gamma->shape(), beta->shape(),
                              &shape_);
  }
}

class BatchNormOp : public OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* gamma_node_ = nullptr;
  const GraphNode* beta_node_ = nullptr;
  const GraphNode* mean_node_ = nullptr;
  const GraphNode* var_node_ = nullptr;
  const tsr_t* X_ = nullptr;
  const tsr_t* gamma_ = nullptr;
  const tsr_t* beta_ = nullptr;
  tsr_t* mean_ = nullptr;
  tsr_t* var_ = nullptr;
  float_t moving_decay_ = 0;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* ggamma_ = nullptr;
  tsr_t* gbeta_ = nullptr;
  BatchNormAux aux_;
  BatchNormMutableAux<float_t> maux_;
  mutable int pulled_ = 0;
  int push_count_ = 0;
  static constexpr int PUSH_INTERVAL = 128;  // magic number

 public:
  DEFINE_OP_LIKE(BatchNormOp);

  void InitForward() override {
    Xnode_ = node_->input(0);
    gamma_node_ = node_->input(1);
    beta_node_ = node_->input(2);
    mean_node_ = node_->input(3);
    var_node_ = node_->input(4);
    for (const GraphNode* output : mean_node_->output()) {
      DXCHECK_THROW(output->type_index() == typeid(BatchNormNode));
    }
    for (const GraphNode* output : var_node_->output()) {
      DXCHECK_THROW(output->type_index() == typeid(BatchNormNode));
    }
    DXCHECK_THROW(!mean_node_->need_grad());
    DXCHECK_THROW(!var_node_->need_grad());
    X_ = GetPtrTSR(Xnode_);
    gamma_ = GetPtrTSR(gamma_node_);
    beta_ = GetPtrTSR(beta_node_);
    mean_ = GetPtrTSR(mean_node_);
    var_ = GetPtrTSR(var_node_);
    moving_decay_ = (float_t)((const BatchNormNode*)node_)->moving_decay();
    DXCHECK_THROW(
        BatchNormPrepare(X_->shape(), gamma_->shape(), beta_->shape(), &aux_));
    BatchNormPrepare(aux_, &maux_);
    Z_ = InitHiddenTSR(node_, aux_.Z);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(Xnode_, X_->shape());
    ggamma_ = InitGradTSR(gamma_node_, gamma_->shape());
    gbeta_ = InitGradTSR(beta_node_, beta_->shape());
    if (gX_ || ggamma_ || gbeta_) {
      BatchNormPrepareBackward(aux_, &maux_);
    }
  }

  void Forward() override {
    BatchNorm(*X_, *gamma_, *beta_, Z_, aux_, &maux_, moving_decay_, mean_,
              var_);
  }

  void Predict() override {
    BatchNorm(*X_, *mean_, *var_, *gamma_, *beta_, Z_, aux_, &maux_);
  }

  void Backward() override {
    if (gZ_) {
      BatchNormBackward(*X_, *gamma_, *beta_, *Z_, *gZ_, gX_, ggamma_, gbeta_,
                        aux_, &maux_);
    }

    if (push_count_ == 0) {
      overwritten_param_->erase(mean_node_->name());
      overwritten_param_->erase(var_node_->name());
    }

    // NOTE: push mean and var every 'PUSH_INTERVAL' batch.
    if (++push_count_ == PUSH_INTERVAL) {
      if (mean_node_->node_type() == GRAPH_NODE_TYPE_PARAM) {
        (*overwritten_param_)[mean_node_->name()] = *mean_;
      }

      if (var_node_->node_type() == GRAPH_NODE_TYPE_PARAM) {
        (*overwritten_param_)[var_node_->name()] = *var_;
      }

      push_count_ = 0;
    }
  }

  void GetPullRequest(PullRequest* pull_request) const override {
    if (Xnode_->node_type() == GRAPH_NODE_TYPE_PARAM) {
      pull_request->tsr_set.emplace(Xnode_->name());
    }

    if (gamma_node_->node_type() == GRAPH_NODE_TYPE_PARAM) {
      pull_request->tsr_set.emplace(gamma_node_->name());
    }

    if (beta_node_->node_type() == GRAPH_NODE_TYPE_PARAM) {
      pull_request->tsr_set.emplace(beta_node_->name());
    }

    // NOTE: pull mean and var only once.
    if (!pulled_) {
      if (mean_node_->node_type() == GRAPH_NODE_TYPE_PARAM) {
        pull_request->tsr_set.emplace(mean_node_->name());
      }

      if (var_node_->node_type() == GRAPH_NODE_TYPE_PARAM) {
        pull_request->tsr_set.emplace(var_node_->name());
      }

      pulled_ = 1;
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchNorm);

}  // namespace deepx_core
