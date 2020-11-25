// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#include "transpose.h"
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
#include <sage2/sgemm.h>
#endif

namespace deepx_core {
namespace {

struct TensorDotAux {
  Shape Z;
  int m = 0;
  int n = 0;
  int k = 0;
  // 't' in this file means transpose.
  int tX = 0;
  int tY = 0;
  TransposeAux tXaux;
  TransposeAux tYaux;
};

bool TensorDotPrepare(const Shape& X, const Shape& Y, int axes_n,
                      TensorDotAux* aux) noexcept {
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

  if (axes_n > Xrank) {
    DXERROR(
        "Invalid Xaxes and Yaxes: axes_n %d must be less than or equal to %d.",
        axes_n, Xrank);
    return false;
  }

  if (axes_n > Yrank) {
    DXERROR(
        "Invalid Xaxes and Yaxes: axes_n %d must be less than or equal to %d.",
        axes_n, Yrank);
    return false;
  }

  int Zrank = Xrank + Yrank - axes_n - axes_n;
  if (Zrank > SHAPE_MAX_RANK) {
    DXERROR("Rank of output %d is too large.", Zrank);
    return false;
  }

  int Zdims[SHAPE_MAX_RANK];
  int Xaxis;
  int j;
  int m = 1, n = 1, k = 1;

  for (int i = 0; i < axes_n; ++i) {
    Xaxis = Xrank - axes_n + i;
    if (X[Xaxis] != Y[i]) {
      DXERROR("Invalid axes_n: inconsistent dim %d vs %d.", X[Xaxis], Y[i]);
      return false;
    }
  }

  for (int i = 0; i < axes_n; ++i) {
    Xaxis = Xrank - axes_n + i;
    k *= X[Xaxis];
  }
  j = 0;
  for (int i = 0; i < Xrank - axes_n; ++i) {
    Zdims[j++] = X[i];
    m *= X[i];
  }
  for (int i = axes_n; i < Yrank; ++i) {
    Zdims[j++] = Y[i];
    n *= Y[i];
  }

  if (Zrank == 0) {
    aux->Z.resize(1);
  } else {
    aux->Z.assign(&Zdims[0], &Zdims[Zrank]);
  }

  aux->m = m;
  aux->n = n;
  aux->k = k;
  aux->tX = 0;
  aux->tY = 0;
  return true;
}

bool TensorDotInferShape(const Shape& X, const Shape& Y, int axes_n,
                         Shape* Z) noexcept {
  TensorDotAux aux;
  if (!TensorDotPrepare(X, Y, axes_n, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

bool TensorDotPrepare(const Shape& X, const Shape& Y, const Shape& Xaxes,
                      const Shape& Yaxes, TensorDotAux* aux) noexcept {
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

  int axes_n = Xaxes.rank();
  if (axes_n != Yaxes.rank()) {
    DXERROR("Invalid Xaxes and Yaxes: inconsistent axes_n %d vs %d.", axes_n,
            Yaxes.rank());
    return false;
  }

  if (axes_n == 0) {
    return TensorDotPrepare(X, Y, 0, aux);
  }

  if (axes_n > Xrank) {
    DXERROR(
        "Invalid Xaxes and Yaxes: axes_n %d must be less than or equal to %d.",
        axes_n, Xrank);
    return false;
  }

  if (axes_n > Yrank) {
    DXERROR(
        "Invalid Xaxes and Yaxes: axes_n %d must be less than or equal to %d.",
        axes_n, Yrank);
    return false;
  }

  int Zrank = Xrank + Yrank - axes_n - axes_n;
  if (Zrank > SHAPE_MAX_RANK) {
    DXERROR("Rank of output %d is too large.", Zrank);
    return false;
  }

  using ai_t = std::array<int, SHAPE_MAX_RANK>;
  ai_t referred;
  int Zdims[SHAPE_MAX_RANK] = {0};
  int _Xaxes[SHAPE_MAX_RANK] = {0}, _Yaxes[SHAPE_MAX_RANK] = {0};
  int tXaxes[SHAPE_MAX_RANK] = {0}, tYaxes[SHAPE_MAX_RANK] = {0};
  int j;
  int tX = 0, tY = 0;
  int m = 1, n = 1, k = 1;

  for (int i = 0; i < axes_n; ++i) {
    _Xaxes[i] = Xaxes[i];
    _Yaxes[i] = Yaxes[i];
    if (!X.real_axis(&_Xaxes[i])) {
      DXERROR("Invalid Xaxes: %s.", to_string(Xaxes).c_str());
      return false;
    }
    if (!Y.real_axis(&_Yaxes[i])) {
      DXERROR("Invalid Yaxes: %s.", to_string(Yaxes).c_str());
      return false;
    }
    if (X[_Xaxes[i]] != Y[_Yaxes[i]]) {
      DXERROR("Invalid Xaxes and Yaxes: inconsistent dim %d vs %d.",
              X[_Xaxes[i]], Y[_Yaxes[i]]);
      return false;
    }
  }

  referred.fill(0);
  j = 0;
  for (int i = 0; i < axes_n; ++i) {
    referred[_Xaxes[i]] = 1;
  }
  for (int i = 0; i < Xrank; ++i) {
    if (referred[i] == 0) {
      tXaxes[j++] = i;
    }
  }
  for (int i = 0; i < axes_n; ++i) {
    tXaxes[j + i] = _Xaxes[i];
  }

  referred.fill(0);
  for (int i = 0; i < axes_n; ++i) {
    tYaxes[i] = _Yaxes[i];
  }
  j = axes_n;
  for (int i = 0; i < axes_n; ++i) {
    referred[_Yaxes[i]] = 1;
  }
  for (int i = 0; i < Yrank; ++i) {
    if (referred[i] == 0) {
      tYaxes[j++] = i;
    }
  }

  for (int i = Xrank - axes_n; i < Xrank; ++i) {
    if (tXaxes[i] != i) {
      tX = 1;
      break;
    }
  }
  for (int i = 0; i < axes_n; ++i) {
    if (tYaxes[i] != i) {
      tY = 1;
      break;
    }
  }

  for (int i = 0; i < axes_n; ++i) {
    k *= X[_Xaxes[i]];
  }
  j = 0;
  for (int i = 0; i < Xrank - axes_n; ++i) {
    Zdims[j++] = X[tXaxes[i]];
    m *= X[tXaxes[i]];
  }
  for (int i = axes_n; i < Yrank; ++i) {
    Zdims[j++] = Y[tYaxes[i]];
    n *= Y[tYaxes[i]];
  }

  if (Zrank == 0) {
    aux->Z.resize(1);
  } else {
    aux->Z.assign(&Zdims[0], &Zdims[Zrank]);
  }
  aux->m = m;
  aux->n = n;
  aux->k = k;
  aux->tX = tX;
  aux->tY = tY;
  if (tX) {
    (void)TransposePrepare(X, Shape(&tXaxes[0], &tXaxes[Xrank]), &aux->tXaux);
  }
  if (tY) {
    (void)TransposePrepare(Y, Shape(&tYaxes[0], &tYaxes[Yrank]), &aux->tYaux);
  }
  return true;
}

bool TensorDotInferShape(const Shape& X, const Shape& Y, const Shape& Xaxes,
                         const Shape& Yaxes, Shape* Z) noexcept {
  TensorDotAux aux;
  if (!TensorDotPrepare(X, Y, Xaxes, Yaxes, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
struct TensorDotMutableAux {
  Tensor<T> tX;
  Tensor<T> tY;
  Tensor<T> tgX;
  Tensor<T> tgY;
};

template <typename T>
void TensorDotPrepare(const TensorDotAux& aux, TensorDotMutableAux<T>* maux) {
  if (aux.tX) {
    maux->tX.resize(aux.tXaux.Z);
  }
  if (aux.tY) {
    maux->tY.resize(aux.tYaux.Z);
  }
}

template <typename T>
void TensorDotPrepareBackward(const TensorDotAux& aux,
                              TensorDotMutableAux<T>* maux) {
  if (aux.tX) {
    maux->tgX.resize(aux.tXaux.Z);
  }
  if (aux.tY) {
    maux->tgY.resize(aux.tYaux.Z);
  }
}

template <typename T>
void TensorDot(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
               const TensorDotAux& aux, TensorDotMutableAux<T>* maux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X;
  const T* _Y;
  T* _Z = Z->data();

  if (aux.tX) {
    Transpose(X, &maux->tX, aux.tXaux);
    _X = maux->tX.data();
  } else {
    _X = X.data();
  }

  if (aux.tY) {
    Transpose(Y, &maux->tY, aux.tYaux);
    _Y = maux->tY.data();
  } else {
    _Y = Y.data();
  }

  LLMath<T>::gemm(0, 0, m, n, k, _X, _Y, _Z);
}

template <typename T>
void TensorDotBackward(const Tensor<T>& X, const Tensor<T>& Y,
                       const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                       Tensor<T>* gX, Tensor<T>* gY, const TensorDotAux& aux,
                       TensorDotMutableAux<T>* maux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _gZ = gZ.data();

  if (gX) {
    const T* _Y = aux.tY ? maux->tY.data() : Y.data();
    if (aux.tX) {
      LLMath<T>::gemm(0, 1, m, k, n, 1, _gZ, _Y, 0, maux->tgX.data());
      TransposeBackward(X, maux->tX, maux->tgX, gX, aux.tXaux);
    } else {
      LLMath<T>::gemm(0, 1, m, k, n, 1, _gZ, _Y, 1, gX->data());
    }
  }

  if (gY) {
    const T* _X = aux.tX ? maux->tX.data() : X.data();
    if (aux.tY) {
      LLMath<T>::gemm(1, 0, k, n, m, 1, _X, _gZ, 0, maux->tgY.data());
      TransposeBackward(Y, maux->tY, maux->tgY, gY, aux.tYaux);
    } else {
      LLMath<T>::gemm(1, 0, k, n, m, 1, _X, _gZ, 1, gY->data());
    }
  }
}

template <typename T>
struct TensorDotJitAux {};

template <typename T>
void TensorDotJitPrepare(const TensorDotAux& /*aux*/,
                         TensorDotJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void TensorDotJitPrepareBackward(const TensorDotAux& /*aux*/,
                                 TensorDotJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void TensorDotJit(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                  const TensorDotAux& aux, const TensorDotJitAux<T>& /*jaux*/,
                  TensorDotMutableAux<T>* maux) noexcept {
  TensorDot(X, Y, Z, aux, maux);
}

template <typename T>
void TensorDotJitBackward(const Tensor<T>& X, const Tensor<T>& Y,
                          const Tensor<T>& Z, const Tensor<T>& gZ,
                          Tensor<T>* gX, Tensor<T>* gY, const TensorDotAux& aux,
                          const TensorDotJitAux<T>& /*jaux*/,
                          TensorDotMutableAux<T>* maux) noexcept {
  TensorDotBackward(X, Y, Z, gZ, gX, gY, aux, maux);
}

#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
template <>
struct TensorDotJitAux<float> {
  void* forward_jit = nullptr;
  void* backward_gX_jit = nullptr;
  void* backward_gY_jit = nullptr;
  sage2_sgemm_t forward = nullptr;
  sage2_sgemm_t backward_gX = nullptr;
  sage2_sgemm_t backward_gY = nullptr;

  ~TensorDotJitAux() {
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
void TensorDotJitPrepare<float>(const TensorDotAux& aux,
                                TensorDotJitAux<float>* jaux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->forward_jit) {
    sage2_sgemm_jit_uninit(jaux->forward_jit);
  }
  jaux->forward_jit =
      sage2_sgemm_jit_init(101, 111, 111, m, n, k, 1, k, n, 0, n);
  DXASSERT(jaux->forward_jit);
  jaux->forward = sage2_sgemm_jit_get(jaux->forward_jit);
  DXASSERT(jaux->forward);
}

template <>
void TensorDotJitPrepareBackward<float>(const TensorDotAux& aux,
                                        TensorDotJitAux<float>* jaux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->backward_gX_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gX_jit);
  }
  if (aux.tX) {
    jaux->backward_gX_jit =
        sage2_sgemm_jit_init(101, 111, 112, m, k, n, 1, n, n, 0, k);
  } else {
    jaux->backward_gX_jit =
        sage2_sgemm_jit_init(101, 111, 112, m, k, n, 1, n, n, 1, k);
  }
  DXASSERT(jaux->backward_gX_jit);
  jaux->backward_gX = sage2_sgemm_jit_get(jaux->backward_gX_jit);
  DXASSERT(jaux->backward_gX);

  if (jaux->backward_gY_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gY_jit);
  }
  if (aux.tY) {
    jaux->backward_gY_jit =
        sage2_sgemm_jit_init(101, 112, 111, k, n, m, 1, k, n, 0, n);
  } else {
    jaux->backward_gY_jit =
        sage2_sgemm_jit_init(101, 112, 111, k, n, m, 1, k, n, 1, n);
  }
  DXASSERT(jaux->backward_gY_jit);
  jaux->backward_gY = sage2_sgemm_jit_get(jaux->backward_gY_jit);
  DXASSERT(jaux->backward_gY);
}

template <>
void TensorDotJit<float>(const Tensor<float>& X, const Tensor<float>& Y,
                         Tensor<float>* Z, const TensorDotAux& aux,
                         const TensorDotJitAux<float>& jaux,
                         TensorDotMutableAux<float>* maux) noexcept {
  const float* _X;
  const float* _Y;
  float* _Z = Z->data();

  if (aux.tX) {
    Transpose(X, &maux->tX, aux.tXaux);
    _X = maux->tX.data();
  } else {
    _X = X.data();
  }

  if (aux.tY) {
    Transpose(Y, &maux->tY, aux.tYaux);
    _Y = maux->tY.data();
  } else {
    _Y = Y.data();
  }

  jaux.forward(jaux.forward_jit, _X, _Y, _Z);
}

template <>
void TensorDotJitBackward<float>(const Tensor<float>& X, const Tensor<float>& Y,
                                 const Tensor<float>& /*Z*/,
                                 const Tensor<float>& gZ, Tensor<float>* gX,
                                 Tensor<float>* gY, const TensorDotAux& aux,
                                 const TensorDotJitAux<float>& jaux,
                                 TensorDotMutableAux<float>* maux) noexcept {
  const float* _gZ = gZ.data();

  if (gX) {
    const float* _Y = aux.tY ? maux->tY.data() : Y.data();
    if (aux.tX) {
      jaux.backward_gX(jaux.backward_gX_jit, _gZ, _Y, maux->tgX.data());
      TransposeBackward(X, maux->tX, maux->tgX, gX, aux.tXaux);
    } else {
      jaux.backward_gX(jaux.backward_gX_jit, _gZ, _Y, gX->data());
    }
  }

  if (gY) {
    const float* _X = aux.tX ? maux->tX.data() : X.data();
    if (aux.tY) {
      jaux.backward_gY(jaux.backward_gY_jit, _X, _gZ, maux->tgY.data());
      TransposeBackward(Y, maux->tY, maux->tgY, gY, aux.tYaux);
    } else {
      jaux.backward_gY(jaux.backward_gY_jit, _X, _gZ, gY->data());
    }
  }
}
#endif

}  // namespace

TensorDotNode::TensorDotNode(std::string name, GraphNode* X, GraphNode* Y,
                             int axes_n)
    : GraphNodeBinaryBase(std::move(name), X, Y),
      use_axes_n_(1),
      axes_n_(axes_n) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)TensorDotInferShape(X->shape(), Y->shape(), axes_n_, &shape_);
  }
}

TensorDotNode::TensorDotNode(std::string name, GraphNode* X, GraphNode* Y,
                             const Shape& Xaxes, const Shape& Yaxes)
    : GraphNodeBinaryBase(std::move(name), X, Y), Xaxes_(Xaxes), Yaxes_(Yaxes) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)TensorDotInferShape(X->shape(), Y->shape(), Xaxes_, Yaxes_, &shape_);
  }
}

class TensorDotOp : public OpBinaryBase {
 private:
  TensorDotAux aux_;
  TensorDotJitAux<float_t> jaux_;
  TensorDotMutableAux<float_t> maux_;

 public:
  DEFINE_OP_LIKE(TensorDotOp);

  const Shape& InferShape() override {
    const TensorDotNode* node = (const TensorDotNode*)node_;  // NOLINT
    int use_axes_n = node->use_axes_n();
    if (use_axes_n) {
      int axes_n = node->axes_n();
      DXCHECK_THROW(TensorDotPrepare(X_->shape(), Y_->shape(), axes_n, &aux_));
    } else {
      const Shape& Xaxes = node->Xaxes();
      const Shape& Yaxes = node->Yaxes();
      DXCHECK_THROW(
          TensorDotPrepare(X_->shape(), Y_->shape(), Xaxes, Yaxes, &aux_));
    }
    TensorDotJitPrepare(aux_, &jaux_);
    TensorDotPrepare(aux_, &maux_);
    return aux_.Z;
  }

  void InitBackward() override {
    OpBinaryBase::InitBackward();
    if (gX_ || gY_) {
      TensorDotJitPrepareBackward(aux_, &jaux_);
      TensorDotPrepareBackward(aux_, &maux_);
    }
  }

  void Forward() override { TensorDotJit(*X_, *Y_, Z_, aux_, jaux_, &maux_); }

  void Backward() override {
    if (gZ_) {
      TensorDotJitBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, jaux_, &maux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(TensorDot);

}  // namespace deepx_core
