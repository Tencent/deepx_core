// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
#include <sage2/sgemm.h>
#endif

namespace deepx_core {
namespace {

struct InnerAux {
  Shape Z;
  int m = 0;  // total dim of X without the last axis
  int n = 0;  // total dim of Y without the last axis
  int k = 0;  // the last axis dim
};

bool InnerPrepare(const Shape& X, const Shape& Y, InnerAux* aux) noexcept {
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

  if (X.back() != Y.back()) {
    DXERROR("Invalid X and Y: inconsistent last dim %d vs %d.", X.back(),
            Y.back());
    return false;
  }

  int Zrank = Xrank + Yrank - 2;
  if (Zrank > SHAPE_MAX_RANK) {
    DXERROR("Rank of output %d is too large.", Zrank);
    return false;
  }

  int Zdims[SHAPE_MAX_RANK];
  int j = 0;
  for (int i = 0; i < X.rank() - 1; ++i) {
    Zdims[j++] = X[i];
  }
  for (int i = 0; i < Y.rank() - 1; ++i) {
    Zdims[j++] = Y[i];
  }
  if (Zrank == 0) {
    aux->Z.resize(1);
  } else {
    aux->Z.assign(&Zdims[0], &Zdims[Zrank]);
  }
  aux->m = X.total_dim() / X.back();
  aux->n = Y.total_dim() / X.back();
  aux->k = X.back();
  return true;
}

bool InnerInferShape(const Shape& X, const Shape& Y, Shape* Z) noexcept {
  InnerAux aux;
  if (!InnerPrepare(X, Y, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void Inner(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
           const InnerAux& aux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  LLMath<T>::gemm(0, 1, m, n, k, X.data(), Y.data(), Z->data());
}

template <typename T>
void InnerBackward(const Tensor<T>& X, const Tensor<T>& Y,
                   const Tensor<T>& /*Z*/, const Tensor<T>& gZ, Tensor<T>* gX,
                   Tensor<T>* gY, const InnerAux& aux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  if (gX) {
    LLMath<T>::gemm(0, 0, m, k, n, 1, gZ.data(), Y.data(), 1, gX->data());
  }
  if (gY) {
    LLMath<T>::gemm(1, 0, n, k, m, 1, gZ.data(), X.data(), 1, gY->data());
  }
}

template <typename T>
struct InnerJitAux {};

template <typename T>
void InnerJitPrepare(const InnerAux& /*aux*/,
                     InnerJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void InnerJitPrepareBackward(const InnerAux& /*aux*/,
                             InnerJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void InnerJit(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
              const InnerAux& aux, const InnerJitAux<T>& /*jaux*/) noexcept {
  Inner(X, Y, Z, aux);
}

template <typename T>
void InnerJitBackward(const Tensor<T>& X, const Tensor<T>& Y,
                      const Tensor<T>& Z, const Tensor<T>& gZ, Tensor<T>* gX,
                      Tensor<T>* gY, const InnerAux& aux,
                      const InnerJitAux<T>& /*jaux*/) noexcept {
  InnerBackward(X, Y, Z, gZ, gX, gY, aux);
}

#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
template <>
struct InnerJitAux<float> {
  void* forward_jit = nullptr;
  void* backward_gX_jit = nullptr;
  void* backward_gY_jit = nullptr;
  sage2_sgemm_t forward = nullptr;
  sage2_sgemm_t backward_gX = nullptr;
  sage2_sgemm_t backward_gY = nullptr;

  ~InnerJitAux() {
    if (forward_jit) {
      sage2_sgemm_jit_uninit(forward_jit);
    }
    if (backward_gX_jit) {
      sage2_sgemm_jit_uninit(backward_gX_jit);
    }
    if (backward_gY_jit) {
      sage2_sgemm_jit_uninit(backward_gY_jit);
    }
  }
};

template <>
void InnerJitPrepare<float>(const InnerAux& aux,
                            InnerJitAux<float>* jaux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->forward_jit) {
    sage2_sgemm_jit_uninit(jaux->forward_jit);
  }
  jaux->forward_jit =
      sage2_sgemm_jit_init(101, 111, 112, m, n, k, 1, k, k, 0, n);
  DXASSERT(jaux->forward_jit);
  jaux->forward = sage2_sgemm_jit_get(jaux->forward_jit);
  DXASSERT(jaux->forward);
}

template <>
void InnerJitPrepareBackward<float>(const InnerAux& aux,
                                    InnerJitAux<float>* jaux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->backward_gX_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gX_jit);
  }
  jaux->backward_gX_jit =
      sage2_sgemm_jit_init(101, 111, 111, m, k, n, 1, n, k, 1, k);
  DXASSERT(jaux->backward_gX_jit);
  jaux->backward_gX = sage2_sgemm_jit_get(jaux->backward_gX_jit);
  DXASSERT(jaux->backward_gX);

  if (jaux->backward_gY_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gY_jit);
  }
  jaux->backward_gY_jit =
      sage2_sgemm_jit_init(101, 112, 111, n, k, m, 1, n, k, 1, k);
  DXASSERT(jaux->backward_gY_jit);
  jaux->backward_gY = sage2_sgemm_jit_get(jaux->backward_gY_jit);
  DXASSERT(jaux->backward_gY);
}

template <>
void InnerJit<float>(const Tensor<float>& X, const Tensor<float>& Y,
                     Tensor<float>* Z, const InnerAux& /*aux*/,
                     const InnerJitAux<float>& jaux) noexcept {
  jaux.forward(jaux.forward_jit, X.data(), Y.data(), Z->data());
}

template <>
void InnerJitBackward<float>(const Tensor<float>& X, const Tensor<float>& Y,
                             const Tensor<float>& /*Z*/,
                             const Tensor<float>& gZ, Tensor<float>* gX,
                             Tensor<float>* gY, const InnerAux& /*aux*/,
                             const InnerJitAux<float>& jaux) noexcept {
  if (gX) {
    jaux.backward_gX(jaux.backward_gX_jit, gZ.data(), Y.data(), gX->data());
  }
  if (gY) {
    jaux.backward_gY(jaux.backward_gY_jit, gZ.data(), X.data(), gY->data());
  }
}
#endif

}  // namespace

InnerNode::InnerNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)InnerInferShape(X->shape(), Y->shape(), &shape_);
  }
}

class InnerOp : public OpBinaryBase {
 private:
  InnerAux aux_;
  InnerJitAux<float_t> jaux_;

 public:
  DEFINE_OP_LIKE(InnerOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(InnerPrepare(X_->shape(), Y_->shape(), &aux_));
    InnerJitPrepare(aux_, &jaux_);
    return aux_.Z;
  }

  void InitBackward() override {
    OpBinaryBase::InitBackward();
    InnerJitPrepareBackward(aux_, &jaux_);
  }

  void Forward() override { InnerJit(*X_, *Y_, Z_, aux_, jaux_); }

  void Backward() override {
    if (gZ_) {
      InnerJitBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, jaux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Inner);

}  // namespace deepx_core
