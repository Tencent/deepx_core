// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <deepx_core/graph/op_impl.h>
#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
#include <sage2/sgemm.h>
#endif

namespace deepx_core {
namespace {

struct MatmulAux {
  Shape Z;
  int transX = 0;
  int transY = 0;
  int mode = 0;
  int no_broadcast = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  Shape Zpad;      // only for broadcast
  Shape Xstrides;  // only for broadcast
  Shape Ystrides;  // only for broadcast
  Shape Zstrides;  // only for broadcast
};

bool MatmulPrepare(const Shape& X, const Shape& Y, int transX, int transY,
                   MatmulAux* aux) noexcept {
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

  Shape Xcanonical;
  Shape Ycanonical;
  int prepend, append;
  if (Xrank == 1) {
    Xcanonical.resize(1, X[0]);
    prepend = 1;
    Xrank = 2;
  } else {
    Xcanonical = X;
    prepend = 0;
  }

  if (Yrank == 1) {
    Ycanonical.resize(Y[0], 1);
    append = 1;
    Yrank = 2;
  } else {
    Ycanonical = Y;
    append = 0;
  }

  int no_broadcast = (Xrank == 2) && (Yrank == 2);
  int k1 = transX ? Xcanonical[Xrank - 2] : Xcanonical[Xrank - 1];
  int k2 = transY ? Ycanonical[Yrank - 1] : Ycanonical[Yrank - 2];
  int m = transX ? Xcanonical[Xrank - 1] : Xcanonical[Xrank - 2];
  int n = transY ? Ycanonical[Yrank - 2] : Ycanonical[Yrank - 1];
  if (k1 != k2) {
    DXERROR("Invalid X and Y: inconsistent dim %d vs %d.", k1, k2);
    return false;
  }

  aux->transX = transX;
  aux->transY = transY;
  aux->mode = (transX != 0) << 1 | (transY != 0);
  aux->no_broadcast = no_broadcast;
  aux->m = m;
  aux->n = n;
  aux->k = k1;

  if (no_broadcast) {
    if (prepend && append) {
      aux->Z.resize(1);
    } else if (prepend) {
      aux->Z.resize(n);
    } else if (append) {
      aux->Z.resize(m);
    } else {
      aux->Z.resize(m, n);
    }
    return true;
  }

  int Zrank = (Xrank < Yrank) ? Yrank : Xrank;
  int Zrank_remain = SHAPE_MAX_RANK - Zrank;
  int Zrank_m2 = Zrank - 2;
  int Zrank_m2_remain = SHAPE_MAX_RANK - Zrank_m2;
  using ai_t = std::array<int, SHAPE_MAX_RANK>;
  ai_t X_reverse_dims, Y_reverse_dims, Z_reverse_dims;
  ai_t X_reverse_strides, Y_reverse_strides, Z_reverse_strides;
  int Xstride = m * k1, Ystride = k1 * n, Zstride = m * n;

  std::copy(Xcanonical.rbegin() + 2, Xcanonical.rend(), X_reverse_dims.begin());
  for (int i = Xrank - 2; i < Zrank_m2; ++i) {
    X_reverse_dims[i] = 1;
  }

  std::copy(Ycanonical.rbegin() + 2, Ycanonical.rend(), Y_reverse_dims.begin());
  for (int i = Yrank - 2; i < Zrank_m2; ++i) {
    Y_reverse_dims[i] = 1;
  }

  Z_reverse_dims[0] = n;
  Z_reverse_dims[1] = m;
  for (int i = 0; i < Zrank_m2; ++i) {
    if (X_reverse_dims[i] == Y_reverse_dims[i]) {
      Z_reverse_dims[i + 2] = X_reverse_dims[i];
    } else if (X_reverse_dims[i] == 1) {
      Z_reverse_dims[i + 2] = Y_reverse_dims[i];
    } else if (Y_reverse_dims[i] == 1) {
      Z_reverse_dims[i + 2] = X_reverse_dims[i];
    } else {
      DXERROR("Couldn't matmul broadcast %s to %s.", to_string(X).c_str(),
              to_string(Y).c_str());
      return false;
    }
  }

  aux->Zpad.assign(Z_reverse_dims.rbegin() + Zrank_remain,
                   Z_reverse_dims.rend());
  if (prepend) {
    Z_reverse_dims[1] = n;
    aux->Z.assign(Z_reverse_dims.rbegin() + Zrank_remain,
                  Z_reverse_dims.rend() - 1);
  } else if (append) {
    aux->Z.assign(Z_reverse_dims.rbegin() + Zrank_remain,
                  Z_reverse_dims.rend() - 1);
  } else {
    aux->Z = aux->Zpad;
  }

  for (int i = 0; i < Zrank_m2; ++i) {
    if (X_reverse_dims[i] == 1) {
      X_reverse_strides[i] = 0;
    } else {
      X_reverse_strides[i] = Xstride;
      Xstride *= X_reverse_dims[i];
    }

    if (Y_reverse_dims[i] == 1) {
      Y_reverse_strides[i] = 0;
    } else {
      Y_reverse_strides[i] = Ystride;
      Ystride *= Y_reverse_dims[i];
    }

    Z_reverse_strides[i] = Zstride;
    Zstride *= Z_reverse_dims[i + 2];
  }
  aux->Xstrides.assign(X_reverse_strides.rbegin() + Zrank_m2_remain,
                       X_reverse_strides.rend());
  aux->Ystrides.assign(Y_reverse_strides.rbegin() + Zrank_m2_remain,
                       Y_reverse_strides.rend());
  aux->Zstrides.assign(Z_reverse_strides.rbegin() + Zrank_m2_remain,
                       Z_reverse_strides.rend());
  return true;
}

bool MatmulInferShape(const Shape& X, const Shape& Y, int transX, int transY,
                      Shape* Z) noexcept {
  MatmulAux aux;
  if (!MatmulPrepare(X, Y, transX, transY, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

template <typename T>
void _GEMM(const T* X, const T* Y, T* Z, const MatmulAux& aux) noexcept {
  int transX = aux.transX, transY = aux.transY;
  int m = aux.m, n = aux.n, k = aux.k;
  LLMath<T>::gemm(transX, transY, m, n, k, 1, X, Y, 0, Z);
}

template <typename T>
void _Matmul(const T* X, const T* Y, T* Z, const MatmulAux& aux) noexcept {
  _GEMM(X, Y, Z, aux);
}

// 'BC' means broadcast.
template <typename T>
void _BCMatmulRank1(const T* X, const T* Y, T* Z,
                    const MatmulAux& aux) noexcept {
  for (int i = 0; i < aux.Zpad[0]; ++i) {
    _GEMM(X, Y, Z, aux);
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    Z += aux.Zstrides[0];
  }
}

template <typename T>
void _BCMatmulFallback(int level, const T* X, const T* Y, T* Z,
                       const MatmulAux& aux) noexcept {
  if (level < aux.Xstrides.rank()) {
    for (int i = 0; i < aux.Zpad[level]; ++i) {
      _BCMatmulFallback(level + 1, X, Y, Z, aux);
      X += aux.Xstrides[level];
      Y += aux.Ystrides[level];
      Z += aux.Zstrides[level];
    }
  } else {
    _GEMM(X, Y, Z, aux);
  }
}

template <typename T>
void _GEMMBackward(const T* X, const T* Y, const T* gZ, T* gX, T* gY,
                   const MatmulAux& aux) {
  int m = aux.m, n = aux.n, k = aux.k;
  switch (aux.mode) {
    case 0:
      if (gX) {
        LLMath<T>::gemm(0, 1, m, k, n, 1, gZ, Y, 1, gX);
      }
      if (gY) {
        LLMath<T>::gemm(1, 0, k, n, m, 1, X, gZ, 1, gY);
      }
      break;
    case 1:
      if (gX) {
        LLMath<T>::gemm(0, 0, m, k, n, 1, gZ, Y, 1, gX);
      }
      if (gY) {
        LLMath<T>::gemm(1, 0, n, k, m, 1, gZ, X, 1, gY);
      }
      break;
    case 2:
      if (gX) {
        LLMath<T>::gemm(0, 1, k, m, n, 1, Y, gZ, 1, gX);
      }
      if (gY) {
        LLMath<T>::gemm(0, 0, k, n, m, 1, X, gZ, 1, gY);
      }
      break;
    case 3:
      if (gX) {
        LLMath<T>::gemm(1, 1, k, m, n, 1, Y, gZ, 1, gX);
      }
      if (gY) {
        LLMath<T>::gemm(1, 1, n, k, m, 1, gZ, X, 1, gY);
      }
      break;
  }
}

template <typename T>
void _MatmulBackward(const T* X, const T* Y, const T* gZ, T* gX, T* gY,
                     const MatmulAux& aux) noexcept {
  _GEMMBackward(X, Y, gZ, gX, gY, aux);
}

template <typename T>
void _BCMatmulBackwardRank1(const T* X, const T* Y, const T* gZ, T* gX, T* gY,
                            const MatmulAux& aux) noexcept {
  for (int i = 0; i < aux.Zpad[0]; ++i) {
    _GEMMBackward(X, Y, gZ, gX, gY, aux);
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    gZ += aux.Zstrides[0];
    if (gX) {
      gX += aux.Xstrides[0];
    }
    if (gY) {
      gY += aux.Ystrides[0];
    }
  }
}

template <typename T>
void _BCMatmulBackwardFallback(int level, const T* X, const T* Y, const T* gZ,
                               T* gX, T* gY, const MatmulAux& aux) noexcept {
  if (level < aux.Xstrides.rank()) {
    for (int i = 0; i < aux.Zpad[level]; ++i) {
      _BCMatmulBackwardFallback(level + 1, X, Y, gZ, gX, gY, aux);
      X += aux.Xstrides[level];
      Y += aux.Ystrides[level];
      gZ += aux.Zstrides[level];
      if (gX) {
        gX += aux.Xstrides[level];
      }
      if (gY) {
        gY += aux.Ystrides[level];
      }
    }
  } else {
    _GEMMBackward(X, Y, gZ, gX, gY, aux);
  }
}

template <typename T>
void Matmul(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
            const MatmulAux& aux) noexcept {
  const T* _X = X.data();
  const T* _Y = Y.data();
  T* _Z = Z->data();
  if (aux.no_broadcast) {
    _Matmul(_X, _Y, _Z, aux);
  } else if (aux.Xstrides.is_rank(1)) {
    _BCMatmulRank1(_X, _Y, _Z, aux);
  } else {
    _BCMatmulFallback(0, _X, _Y, _Z, aux);
  }
}

template <typename T>
void MatmulBackward(const Tensor<T>& X, const Tensor<T>& Y,
                    const Tensor<T>& /*Z*/, const Tensor<T>& gZ, Tensor<T>* gX,
                    Tensor<T>* gY, const MatmulAux& aux) noexcept {
  const T* _X = X.data();
  const T* _Y = Y.data();
  const T* _gZ = gZ.data();
  T* _gX = gX ? gX->data() : nullptr;
  T* _gY = gY ? gY->data() : nullptr;
  if (_gX || _gY) {
    if (aux.no_broadcast) {
      _MatmulBackward(_X, _Y, _gZ, _gX, _gY, aux);
    } else if (aux.Xstrides.is_rank(1)) {
      _BCMatmulBackwardRank1(_X, _Y, _gZ, _gX, _gY, aux);
    } else {
      _BCMatmulBackwardFallback(0, _X, _Y, _gZ, _gX, _gY, aux);
    }
  }
}

template <typename T>
struct MatmulJitAux {};

template <typename T>
void MatmulJitPrepare(const MatmulAux& /*aux*/,
                      MatmulJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void MatmulJitPrepareBackward(const MatmulAux& /*aux*/,
                              MatmulJitAux<T>* /*jaux*/) noexcept {}

template <typename T>
void MatmulJit(const Tensor<T>& X, const Tensor<T>& Y, Tensor<T>* Z,
               const MatmulAux& aux, const MatmulJitAux<T>& /*jaux*/) noexcept {
  Matmul(X, Y, Z, aux);
}

template <typename T>
void MatmulJitBackward(const Tensor<T>& X, const Tensor<T>& Y,
                       const Tensor<T>& Z, const Tensor<T>& gZ, Tensor<T>* gX,
                       Tensor<T>* gY, const MatmulAux& aux,
                       const MatmulJitAux<T>& /*jaux*/) noexcept {
  MatmulBackward(X, Y, Z, gZ, gX, gY, aux);
}

#if HAVE_SAGE2_SGEMM_JIT == 1 && HAVE_FLOAT64 == 0
template <>
struct MatmulJitAux<float> {
  void* forward_jit = nullptr;
  void* backward_gX_jit = nullptr;
  void* backward_gY_jit = nullptr;
  sage2_sgemm_t forward = nullptr;
  sage2_sgemm_t backward_gX = nullptr;
  sage2_sgemm_t backward_gY = nullptr;

  ~MatmulJitAux() {
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
void MatmulJitPrepare<float>(const MatmulAux& aux,
                             MatmulJitAux<float>* jaux) noexcept {
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
void MatmulJitPrepareBackward<float>(const MatmulAux& aux,
                                     MatmulJitAux<float>* jaux) noexcept {
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

void _GEMMJit(const float* X, const float* Y, float* Z,
              const MatmulAux& /*aux*/,
              const MatmulJitAux<float>& jaux) noexcept {
  jaux.forward(jaux.forward_jit, X, Y, Z);
}

void _MatmulJit(const float* X, const float* Y, float* Z, const MatmulAux& aux,
                const MatmulJitAux<float>& jaux) noexcept {
  _GEMMJit(X, Y, Z, aux, jaux);
}

void _BCMatmulJitRank1(const float* X, const float* Y, float* Z,
                       const MatmulAux& aux,
                       const MatmulJitAux<float>& jaux) noexcept {
  for (int i = 0; i < aux.Zpad[0]; ++i) {
    _GEMMJit(X, Y, Z, aux, jaux);
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    Z += aux.Zstrides[0];
  }
}

void _BCMatmulJitFallback(int level, const float* X, const float* Y, float* Z,
                          const MatmulAux& aux,
                          const MatmulJitAux<float>& jaux) noexcept {
  if (level < aux.Xstrides.rank()) {
    for (int i = 0; i < aux.Zpad[level]; ++i) {
      _BCMatmulJitFallback(level + 1, X, Y, Z, aux, jaux);
      X += aux.Xstrides[level];
      Y += aux.Ystrides[level];
      Z += aux.Zstrides[level];
    }
  } else {
    _GEMMJit(X, Y, Z, aux, jaux);
  }
}

void _GEMMJitBackward(const float* X, const float* Y, const float* gZ,
                      float* gX, float* gY, const MatmulAux& aux,
                      const MatmulJitAux<float>& jaux) {
  switch (aux.mode) {
    case 0:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, gZ, Y, gX);
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, X, gZ, gY);
      }
      break;
    case 1:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, gZ, Y, gX);
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, gZ, X, gY);
      }
      break;
    case 2:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, Y, gZ, gX);
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, X, gZ, gY);
      }
      break;
    case 3:
      if (gX) {
        jaux.backward_gX(jaux.backward_gX_jit, Y, gZ, gX);
      }
      if (gY) {
        jaux.backward_gY(jaux.backward_gY_jit, gZ, X, gY);
      }
      break;
  }
}

void _MatmulJitBackward(const float* X, const float* Y, const float* gZ,
                        float* gX, float* gY, const MatmulAux& aux,
                        const MatmulJitAux<float>& jaux) noexcept {
  _GEMMJitBackward(X, Y, gZ, gX, gY, aux, jaux);
}

void _BCMatmulJitBackwardRank1(const float* X, const float* Y, const float* gZ,
                               float* gX, float* gY, const MatmulAux& aux,
                               const MatmulJitAux<float>& jaux) noexcept {
  for (int i = 0; i < aux.Zpad[0]; ++i) {
    _GEMMJitBackward(X, Y, gZ, gX, gY, aux, jaux);
    X += aux.Xstrides[0];
    Y += aux.Ystrides[0];
    gZ += aux.Zstrides[0];
    if (gX) {
      gX += aux.Xstrides[0];
    }
    if (gY) {
      gY += aux.Ystrides[0];
    }
  }
}

void _BCMatmulJitBackwardFallback(int level, const float* X, const float* Y,
                                  const float* gZ, float* gX, float* gY,
                                  const MatmulAux& aux,
                                  const MatmulJitAux<float>& jaux) noexcept {
  if (level < aux.Xstrides.rank()) {
    for (int i = 0; i < aux.Zpad[level]; ++i) {
      _BCMatmulJitBackwardFallback(level + 1, X, Y, gZ, gX, gY, aux, jaux);
      X += aux.Xstrides[level];
      Y += aux.Ystrides[level];
      gZ += aux.Zstrides[level];
      if (gX) {
        gX += aux.Xstrides[level];
      }
      if (gY) {
        gY += aux.Ystrides[level];
      }
    }
  } else {
    _GEMMJitBackward(X, Y, gZ, gX, gY, aux, jaux);
  }
}

template <>
void MatmulJit<float>(const Tensor<float>& X, const Tensor<float>& Y,
                      Tensor<float>* Z, const MatmulAux& aux,
                      const MatmulJitAux<float>& jaux) noexcept {
  const float* _X = X.data();
  const float* _Y = Y.data();
  float* _Z = Z->data();
  if (aux.no_broadcast) {
    _MatmulJit(_X, _Y, _Z, aux, jaux);
  } else if (aux.Xstrides.is_rank(1)) {
    _BCMatmulJitRank1(_X, _Y, _Z, aux, jaux);
  } else {
    _BCMatmulJitFallback(0, _X, _Y, _Z, aux, jaux);
  }
}

template <>
void MatmulJitBackward<float>(const Tensor<float>& X, const Tensor<float>& Y,
                              const Tensor<float>& /*Z*/,
                              const Tensor<float>& gZ, Tensor<float>* gX,
                              Tensor<float>* gY, const MatmulAux& aux,
                              const MatmulJitAux<float>& jaux) noexcept {
  const float* _X = X.data();
  const float* _Y = Y.data();
  const float* _gZ = gZ.data();
  float* _gX = gX ? gX->data() : nullptr;
  float* _gY = gY ? gY->data() : nullptr;
  if (_gX || _gY) {
    if (aux.no_broadcast) {
      _MatmulJitBackward(_X, _Y, _gZ, _gX, _gY, aux, jaux);
    } else if (aux.Xstrides.is_rank(1)) {
      _BCMatmulJitBackwardRank1(_X, _Y, _gZ, _gX, _gY, aux, jaux);
    } else {
      _BCMatmulJitBackwardFallback(0, _X, _Y, _gZ, _gX, _gY, aux, jaux);
    }
  }
}
#endif

}  // namespace

/************************************************************************/
/* Matmul */
/************************************************************************/
MatmulNode::MatmulNode(std::string name, GraphNode* X, GraphNode* Y)
    : GraphNodeBinaryBase(std::move(name), X, Y) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)MatmulInferShape(X->shape(), Y->shape(), 0, 0, &shape_);
  }
}

class MatmulOp : public OpBinaryBase {
 private:
  MatmulAux aux_;
  MatmulJitAux<float_t> jaux_;

 public:
  DEFINE_OP_LIKE(MatmulOp);

  const Shape& InferShape() override {
    DXCHECK_THROW(MatmulPrepare(X_->shape(), Y_->shape(), 0, 0, &aux_));
    MatmulJitPrepare(aux_, &jaux_);
    return aux_.Z;
  }

  void InitBackward() override {
    OpBinaryBase::InitBackward();
    MatmulJitPrepareBackward(aux_, &jaux_);
  }

  void Forward() override { MatmulJit(*X_, *Y_, Z_, aux_, jaux_); }

  void Backward() override {
    if (gZ_) {
      MatmulJitBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, jaux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Matmul);

/************************************************************************/
/* Matmul2 */
/************************************************************************/
Matmul2Node::Matmul2Node(std::string name, GraphNode* X, GraphNode* Y,
                         int transX, int transY)
    : GraphNodeBinaryBase(std::move(name), X, Y),
      transX_(transX),
      transY_(transY) {
  if (!X->shape().empty() && !Y->shape().empty()) {
    (void)MatmulInferShape(X->shape(), Y->shape(), transX, transY, &shape_);
  }
}

class Matmul2Op : public OpBinaryBase {
 private:
  MatmulAux aux_;
  MatmulJitAux<float_t> jaux_;

 public:
  DEFINE_OP_LIKE(Matmul2Op);

  const Shape& InferShape() override {
    int transX = ((const Matmul2Node*)node_)->transX();
    int transY = ((const Matmul2Node*)node_)->transY();
    DXCHECK_THROW(
        MatmulPrepare(X_->shape(), Y_->shape(), transX, transY, &aux_));
    MatmulJitPrepare(aux_, &jaux_);
    return aux_.Z;
  }

  void InitBackward() override {
    OpBinaryBase::InitBackward();
    MatmulJitPrepareBackward(aux_, &jaux_);
  }

  void Forward() override { MatmulJit(*X_, *Y_, Z_, aux_, jaux_); }

  void Backward() override {
    if (gZ_) {
      MatmulJitBackward(*X_, *Y_, *Z_, *gZ_, gX_, gY_, aux_, jaux_);
    }
  }
};

GRAPH_NODE_OP_REGISTER(Matmul2);

}  // namespace deepx_core
