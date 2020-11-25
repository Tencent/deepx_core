// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
#include <sage2/sgemm.h>
#endif

namespace deepx_core {
namespace {

struct GEMMAux {
  Shape Z;
  int transX = 0;
  int transY = 0;
  int mode = 0;
  int m = 0;
  int n = 0;
  int k = 0;
};

bool GEMMPrepare(const Shape& X, const Shape& Y, int transX, int transY,
                 GEMMAux* aux) noexcept {
  if (!X.is_rank(2)) {
    DXERROR("Invalid X: rank of X %d must be 2.", X.rank());
    return false;
  }

  if (!Y.is_rank(2)) {
    DXERROR("Invalid Y: rank of Y %d must be 2.", Y.rank());
    return false;
  }

  int k1 = transX ? X[0] : X[1];
  int k2 = transY ? Y[1] : Y[0];
  if (k1 != k2) {
    DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", k1, k2);
    return false;
  }

  int m = transX ? X[1] : X[0];
  int n = transY ? Y[0] : Y[1];
  aux->Z.resize(m, n);
  aux->transX = transX;
  aux->transY = transY;
  aux->mode = (transX != 0) << 1 | (transY != 0);
  aux->m = m;
  aux->n = n;
  aux->k = k1;
  return true;
}

bool GEMMInferShape(const Shape& X, const Shape& Y, int transX, int transY,
                    Shape* Z) noexcept {
  GEMMAux aux;
  if (!GEMMPrepare(X, Y, transX, transY, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void GEMM(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
          const GEMMAux& aux) noexcept {
  int transX = aux.transX, transY = aux.transY;
  LLTensor<T>::gemm(transX, transY, 1, X, Y, 0, Z);
}

template <typename T>
void GEMMBackward(const Tensor<T>& X, const Tensor<T>& Y,
                  const Tensor<T>& /*Z*/, const Tensor<T>& gZ, Tensor<T>* gX,
                  Tensor<T>* gY, const GEMMAux& aux) noexcept {
  switch (aux.mode) {
    case 0:
      if (gX) {
        LLTensor<T>::gemm(0, 1, 1, gZ, Y, 1, gX);
      }
      if (gY) {
        LLTensor<T>::gemm(1, 0, 1, X, gZ, 1, gY);
      }
      break;
    case 1:
      if (gX) {
        LLTensor<T>::gemm(0, 0, 1, gZ, Y, 1, gX);
      }
      if (gY) {
        LLTensor<T>::gemm(1, 0, 1, gZ, X, 1, gY);
      }
      break;
    case 2:
      if (gX) {
        LLTensor<T>::gemm(0, 1, 1, Y, gZ, 1, gX);
      }
      if (gY) {
        LLTensor<T>::gemm(0, 0, 1, X, gZ, 1, gY);
      }
      break;
    case 3:
      if (gX) {
        LLTensor<T>::gemm(1, 1, 1, Y, gZ, 1, gX);
      }
      if (gY) {
        LLTensor<T>::gemm(1, 1, 1, gZ, X, 1, gY);
      }
      break;
  }
}

template <typename T>
struct GEMMJitAux {};

template <typename T>
void GEMMJitPrepare(const GEMMAux& /*aux*/, GEMMJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void GEMMJitPrepareBackward(const GEMMAux& /*aux*/,
                            GEMMJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void GEMMJit(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
             const GEMMAux& aux, const GEMMJitAux<T>& /*jaux*/) noexcept {
  GEMM(X, Y, Z, aux);
}

template <typename T>
void GEMMJitBackward(const Tensor<T>& X, const Tensor<T>& Y, const Tensor<T>& Z,
                     const Tensor<T>& gZ, Tensor<T>* gX, Tensor<T>* gY,
                     const GEMMAux& aux,
                     const GEMMJitAux<T>& /*jaux*/) noexcept {
  GEMMBackward(X, Y, Z, gZ, gX, gY, aux);
}

#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
template <>
struct GEMMJitAux<float> {
  void* forward_jit = nullptr;
  void* backward_gX_jit = nullptr;
  void* backward_gY_jit = nullptr;
  sage2_sgemm_t forward = nullptr;
  sage2_sgemm_t backward_gX = nullptr;
  sage2_sgemm_t backward_gY = nullptr;

  ~GEMMJitAux() {
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
void GEMMJitPrepare<float>(const GEMMAux& aux,
                           GEMMJitAux<float>* jaux) noexcept {
  int transX = aux.transX, transY = aux.transY;
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->forward_jit) {
    sage2_sgemm_jit_uninit(jaux->forward_jit);
  }
  jaux->forward_jit =
      sage2_sgemm_jit_init(101, transX ? 112 : 111, transY ? 112 : 111, m, n, k,
                           1, transX ? m : k, transY ? k : n, 0, n);
  DXASSERT(jaux->forward_jit);
  jaux->forward = sage2_sgemm_jit_get(jaux->forward_jit);
  DXASSERT(jaux->forward);
}

template <>
void GEMMJitPrepareBackward<float>(const GEMMAux& aux,
                                   GEMMJitAux<float>* jaux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->backward_gX_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gX_jit);
  }
  if (jaux->backward_gY_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gY_jit);
  }
  switch (aux.mode) {
    case 0:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 111, 112, m, k, n, 1, n, n, 1, k);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 112, 111, k, n, m, 1, k, n, 1, n);
      break;
    case 1:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 111, 111, m, k, n, 1, n, k, 1, k);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 112, 111, n, k, m, 1, n, k, 1, k);
      break;
    case 2:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 111, 112, k, m, n, 1, n, n, 1, m);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 111, 111, k, n, m, 1, m, n, 1, n);
      break;
    case 3:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 112, 112, k, m, n, 1, k, n, 1, m);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 112, 112, n, k, m, 1, n, m, 1, k);
      break;
  }
  DXASSERT(jaux->backward_gX_jit);
  jaux->backward_gX = sage2_sgemm_jit_get(jaux->backward_gX_jit);
  DXASSERT(jaux->backward_gX);
  DXASSERT(jaux->backward_gY_jit);
  jaux->backward_gY = sage2_sgemm_jit_get(jaux->backward_gY_jit);
  DXASSERT(jaux->backward_gY);
}

template <>
void GEMMJit<float>(const Tensor<float>& X, const Tensor<float>& Y,
                    Tensor<float>* Z, const GEMMAux& /*aux*/,
                    const GEMMJitAux<float>& jaux) noexcept {
  jaux.forward(jaux.forward_jit, X.data(), Y.data(), Z->data());
}

template <>
void GEMMJitBackward<float>(const Tensor<float>& X, const Tensor<float>& Y,
                            const Tensor<float>& /*Z*/, const Tensor<float>& gZ,
                            Tensor<float>* gX, Tensor<float>* gY,
                            const GEMMAux& aux,
                            const GEMMJitAux<float>& jaux) noexcept {
  switch (aux.mode) {
    case 0:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, gZ.data(), Y.data(), gX->data());
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, X.data(), gZ.data(), gY->data());
      }
      break;
    case 1:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, gZ.data(), Y.data(), gX->data());
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, gZ.data(), X.data(), gY->data());
      }
      break;
    case 2:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, Y.data(), gZ.data(), gX->data());
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, X.data(), gZ.data(), gY->data());
      }
      break;
    case 3:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, Y.data(), gZ.data(), gX->data());
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, gZ.data(), X.data(), gY->data());
      }
      break;
  }
}
#endif

}  // namespace

/************************************************************************/
/* GEMM */
/************************************************************************/
GEMMNode::GEMMNode(std::string name, GraphNode* X, GraphNode* Y, int transX,
                   int transY)
    : GraphNodeBinaryBase(std::move(name), X, Y),
      transX_(transX),
      transY_(transY) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)GEMMInferShape(X->shape(), Y->shape(), transX_, transY_, &shape_);
  }
}

class GEMMOp : public OpBinaryBase {
 private:
  GEMMAux aux_;
  GEMMJitAux<float_t> jaux_;

 public:
  DEFINE_OP_LIKE(GEMMOp);

  const Shape& InferShape() override {
    int transX = ((const GEMMNode*)node_)->transX();
    int transY = ((const GEMMNode*)node_)->transY();
    DXCHECK_THROW(GEMMPrepare(X_->shape(), Y_->shape(), transX, transY, &aux_));
    GEMMJitPrepare(aux_, &jaux_);
    return aux_.Z;
  }

  void InitBackward() override {
    OpBinaryBase::InitBackward();
    GEMMJitPrepareBackward(aux_, &jaux_);
  }

  void Forward() override { GEMMJit(*X_, *Y_, Z_, aux_, jaux_); }

  void Backward() override {
    if (gZ_) {
      GEMMJitBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, jaux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(GEMM);

/************************************************************************/
/* BatchGEMM */
/************************************************************************/
namespace {

struct BatchGEMMAux {
  Shape Z;
  int transX = 0;
  int transY = 0;
  int mode = 0;
  int batch = 0;
  int m = 0;
  int n = 0;
  int k = 0;
};

bool BatchGEMMPrepare(const Shape& X, const Shape& Y, int transX, int transY,
                      BatchGEMMAux* aux) noexcept {
  if (!X.is_rank(3)) {
    DXERROR("Invalid X: rank of X %d must be 3.", X.rank());
    return false;
  }

  if (!Y.is_rank(3)) {
    DXERROR("Invalid Y: rank of Y %d must be 3.", Y.rank());
    return false;
  }

  if (X[0] != Y[0]) {
    DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", X[0], Y[0]);
    return false;
  }

  int k1 = transX ? X[1] : X[2];
  int k2 = transY ? Y[2] : Y[1];
  if (k1 != k2) {
    DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", k1, k2);
    return false;
  }

  int batch = X[0];
  int m = transX ? X[2] : X[1];
  int n = transY ? Y[1] : Y[2];
  aux->Z.resize(batch, m, n);
  aux->transX = transX;
  aux->transY = transY;
  aux->mode = (transX != 0) << 1 | (transY != 0);
  aux->batch = batch;
  aux->m = m;
  aux->n = n;
  aux->k = k1;
  return true;
}

bool BatchGEMMInferShape(const Shape& X, const Shape& Y, int transX, int transY,
                         Shape* Z) noexcept {
  BatchGEMMAux aux;
  if (!BatchGEMMPrepare(X, Y, transX, transY, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void BatchGEMM(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
               const BatchGEMMAux& aux) noexcept {
  int transX = aux.transX, transY = aux.transY;
  int batch = aux.batch;
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  for (int i = 0; i < batch; ++i) {
    LLMath<T>::gemm(transX, transY, m, n, k, 1, _X, _Y, 0, _Z);
    _X += m * k;
    _Y += k * n;
    _Z += m * n;
  }
}

template <typename T>
void BatchGEMMBackward(const Tensor<T>& X, const Tensor<T>& Y,
                       const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                       Tensor<T>* gX, Tensor<T>* gY,
                       const BatchGEMMAux& aux) noexcept {
  int batch = aux.batch;
  int m = aux.m, n = aux.n, k = aux.k;
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _gX = gX ? gX->data() : nullptr;
  T* _gY = gY ? gY->data() : nullptr;
  switch (aux.mode) {
    case 0:
      if (_gX) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(0, 1, m, k, n, 1, _gZ, _Y, 1, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(1, 0, k, n, m, 1, _X, _gZ, 1, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
    case 1:
      if (_gX) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(0, 0, m, k, n, 1, _gZ, _Y, 1, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(1, 0, n, k, m, 1, _gZ, _X, 1, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
    case 2:
      if (_gX) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(0, 1, k, m, n, 1, _Y, _gZ, 1, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(0, 0, k, n, m, 1, _X, _gZ, 1, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
    case 3:
      if (_gX) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(1, 1, k, m, n, 1, _Y, _gZ, 1, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const T* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          LLMath<T>::gemm(1, 1, n, k, m, 1, _gZ, _X, 1, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
  }
}

template <typename T>
struct BatchGEMMJitAux {};

template <typename T>
void BatchGEMMJitPrepare(const BatchGEMMAux& /*aux*/,
                         BatchGEMMJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void BatchGEMMJitPrepareBackward(const BatchGEMMAux& /*aux*/,
                                 BatchGEMMJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void BatchGEMMJit(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
                  const BatchGEMMAux& aux,
                  const BatchGEMMJitAux<T>& /*jaux*/) noexcept {
  BatchGEMM(X, Y, Z, aux);
}

template <typename T>
void BatchGEMMJitBackward(const Tensor<T>& X, const Tensor<T>& Y,
                          const Tensor<T>& Z, const Tensor<T>& gZ,
                          Tensor<T>* gX, Tensor<T>* gY, const BatchGEMMAux& aux,
                          const BatchGEMMJitAux<T>& /*jaux*/) noexcept {
  BatchGEMMBackward(X, Y, Z, gZ, gX, gY, aux);
}

#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
template <>
struct BatchGEMMJitAux<float> {
  void* forward_jit = nullptr;
  void* backward_gX_jit = nullptr;
  void* backward_gY_jit = nullptr;
  sage2_sgemm_t forward = nullptr;
  sage2_sgemm_t backward_gX = nullptr;
  sage2_sgemm_t backward_gY = nullptr;

  ~BatchGEMMJitAux() {
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
void BatchGEMMJitPrepare<float>(const BatchGEMMAux& aux,
                                BatchGEMMJitAux<float>* jaux) noexcept {
  int transX = aux.transX, transY = aux.transY;
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->forward_jit) {
    sage2_sgemm_jit_uninit(jaux->forward_jit);
  }
  jaux->forward_jit =
      sage2_sgemm_jit_init(101, transX ? 112 : 111, transY ? 112 : 111, m, n, k,
                           1, transX ? m : k, transY ? k : n, 0, n);
  DXASSERT(jaux->forward_jit);
  jaux->forward = sage2_sgemm_jit_get(jaux->forward_jit);
  DXASSERT(jaux->forward);
}

template <>
void BatchGEMMJitPrepareBackward<float>(const BatchGEMMAux& aux,
                                        BatchGEMMJitAux<float>* jaux) noexcept {
  int m = aux.m, n = aux.n, k = aux.k;
  if (jaux->backward_gX_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gX_jit);
  }
  if (jaux->backward_gY_jit) {
    sage2_sgemm_jit_uninit(jaux->backward_gY_jit);
  }
  switch (aux.mode) {
    case 0:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 111, 112, m, k, n, 1, n, n, 1, k);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 112, 111, k, n, m, 1, k, n, 1, n);
      break;
    case 1:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 111, 111, m, k, n, 1, n, k, 1, k);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 112, 111, n, k, m, 1, n, k, 1, k);
      break;
    case 2:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 111, 112, k, m, n, 1, n, n, 1, m);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 111, 111, k, n, m, 1, m, n, 1, n);
      break;
    case 3:
      jaux->backward_gX_jit =
          sage2_sgemm_jit_init(101, 112, 112, k, m, n, 1, k, n, 1, m);
      jaux->backward_gY_jit =
          sage2_sgemm_jit_init(101, 112, 112, n, k, m, 1, n, m, 1, k);
      break;
  }
  DXASSERT(jaux->backward_gX_jit);
  jaux->backward_gX = sage2_sgemm_jit_get(jaux->backward_gX_jit);
  DXASSERT(jaux->backward_gX);
  DXASSERT(jaux->backward_gY_jit);
  jaux->backward_gY = sage2_sgemm_jit_get(jaux->backward_gY_jit);
  DXASSERT(jaux->backward_gY);
}

template <>
void BatchGEMMJit<float>(const Tensor<float>& X, const Tensor<float>& Y,
                         Tensor<float>* Z, const BatchGEMMAux& aux,
                         const BatchGEMMJitAux<float>& jaux) noexcept {
  int batch = aux.batch;
  int m = aux.m, n = aux.n, k = aux.k;
  const float* _X = X.data();
  const float* _Y = Y.data();
  float* _Z = Z->data();
  for (int i = 0; i < batch; ++i) {
    jaux.forward(jaux.forward_jit, _X, _Y, _Z);
    _X += m * k;
    _Y += k * n;
    _Z += m * n;
  }
}

template <>
void BatchGEMMJitBackward<float>(const Tensor<float>& X, const Tensor<float>& Y,
                                 const Tensor<float>& /*Z*/,
                                 const Tensor<float>& gZ, Tensor<float>* gX,
                                 Tensor<float>* gY, const BatchGEMMAux& aux,
                                 const BatchGEMMJitAux<float>& jaux) noexcept {
  int batch = aux.batch;
  int m = aux.m, n = aux.n, k = aux.k;
  const float* _X = X.data();
  const float* _Y = Y.data();
  float* _gX = gX ? gX->data() : nullptr;
  float* _gY = gY ? gY->data() : nullptr;
  switch (aux.mode) {
    case 0:
      if (_gX) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gX(jaux.backward_gX_jit, _gZ, _Y, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gY(jaux.backward_gY_jit, _X, _gZ, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
    case 1:
      if (_gX) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gX(jaux.backward_gX_jit, _gZ, _Y, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gY(jaux.backward_gY_jit, _gZ, _X, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
    case 2:
      if (_gX) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gX(jaux.backward_gX_jit, _Y, _gZ, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gY(jaux.backward_gY_jit, _X, _gZ, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
    case 3:
      if (_gX) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gX(jaux.backward_gX_jit, _Y, _gZ, _gX);
          _gX += m * k;
          _Y += k * n;
          _gZ += m * n;
        }
      }
      if (_gY) {
        const float* _gZ = gZ.data();
        for (int i = 0; i < batch; ++i) {
          jaux.backward_gY(jaux.backward_gY_jit, _gZ, _X, _gY);
          _X += m * k;
          _gY += k * n;
          _gZ += m * n;
        }
      }
      break;
  }
}
#endif

}  // namespace

BatchGEMMNode::BatchGEMMNode(std::string name, GraphNode* X, GraphNode* Y,
                             int transX, int transY)
    : GraphNodeBinaryBase(std::move(name), X, Y),
      transX_(transX),
      transY_(transY) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)BatchGEMMInferShape(X->shape(), Y->shape(), transX_, transY_,
                              &shape_);
  }
}

class BatchGEMMOp : public OpBinaryBase {
 private:
  BatchGEMMAux aux_;
  BatchGEMMJitAux<float_t> jaux_;

 public:
  DEFINE_OP_LIKE(BatchGEMMOp);

  const Shape& InferShape() override {
    int transX = ((const BatchGEMMNode*)node_)->transX();
    int transY = ((const BatchGEMMNode*)node_)->transY();
    DXCHECK_THROW(
        BatchGEMMPrepare(X_->shape(), Y_->shape(), transX, transY, &aux_));
    BatchGEMMJitPrepare(aux_, &jaux_);
    return aux_.Z;
  }

  void InitBackward() override {
    OpBinaryBase::InitBackward();
    BatchGEMMJitPrepareBackward(aux_, &jaux_);
  }

  void Forward() override { BatchGEMMJit(*X_, *Y_, Z_, aux_, jaux_); }

  void Backward() override {
    if (gZ_) {
      BatchGEMMJitBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, jaux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(BatchGEMM);

}  // namespace deepx_core
