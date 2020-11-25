// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

struct TransposeAux {
  Shape Z;
  Shape Xstrides;
  Shape Zstrides;
};

inline bool TransposePrepare(const Shape& X, const Shape& axes,
                             TransposeAux* aux) noexcept {
  int rank = axes.rank();
  if (rank == 0) {
    DXERROR("Invalid axes: size of axes is zero.");
    return false;
  }

  if (X.rank() != rank) {
    DXERROR("Invalid X: rank of X %d must be %d.", X.rank(), rank);
    return false;
  }

  int referred[SHAPE_MAX_RANK] = {0};
  int Zdims[SHAPE_MAX_RANK];
  int Xstrides[SHAPE_MAX_RANK], Zstrides[SHAPE_MAX_RANK];

  for (int i = 0; i < rank; ++i) {
    if (axes[i] >= rank) {
      DXERROR("Invalid axes: axis %d must be less than %d.", axes[i], rank);
      return false;
    }
    referred[axes[i]] = 1;
  }
  for (int i = 0; i < rank; ++i) {
    if (referred[i] == 0) {
      DXERROR("Invalid axes: axis %d must be referred.", i);
      return false;
    }
  }

  for (int i = 0; i < rank; ++i) {
    Zdims[i] = X[axes[i]];
  }
  aux->Z.assign(&Zdims[0], &Zdims[rank]);

  Xstrides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    Xstrides[i] = X[i + 1] * Xstrides[i + 1];
  }
  aux->Xstrides.assign(&Xstrides[0], &Xstrides[rank]);

  for (int i = 0; i < rank; ++i) {
    Zstrides[i] = aux->Xstrides[axes[i]];
  }
  aux->Zstrides.assign(&Zstrides[0], &Zstrides[rank]);
  return true;
}

inline bool TransposeInferShape(const Shape& X, const Shape& axes,
                                Shape* Z) noexcept {
  TransposeAux aux;
  if (!TransposePrepare(X, axes, &aux)) {
    return false;
  }
  *Z = aux.Z;
  return true;
}

namespace detail {

template <typename T>
void Transpose(const T* X, T* Z, const TransposeAux& aux) {
  int rank = aux.Z.rank();
  if (rank == 1) {
    LLMath<T>::copy(aux.Z.total_dim(), X, Z);
  } else if (rank == 2) {
    int i = 0;
    int k = 0;
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        Z[i++] = X[k];
        k += aux.Zstrides[1];
      }
      k -= aux.Z[1] * aux.Zstrides[1];
      k += aux.Zstrides[0];
    }
  } else if (rank == 3) {
    int i = 0;
    int k = 0;
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        for (int i2 = 0; i2 < aux.Z[2]; ++i2) {
          Z[i++] = X[k];
          k += aux.Zstrides[2];
        }
        k -= aux.Z[2] * aux.Zstrides[2];
        k += aux.Zstrides[1];
      }
      k -= aux.Z[1] * aux.Zstrides[1];
      k += aux.Zstrides[0];
    }
  } else {
    int index[SHAPE_MAX_RANK] = {0};
    int k = 0;
    int carry;
    Z[0] = X[k];
    for (int i = 1; i < aux.Z.total_dim(); ++i) {
      ++index[rank - 1];
      k += aux.Zstrides[rank - 1];
      carry = 0;
      for (int j = rank - 1; j >= 0; --j) {
        index[j] += carry;
        k += carry * aux.Zstrides[j];
        carry = index[j] == aux.Z[j];
        if (carry) {
          index[j] = 0;
          k -= aux.Z[j] * aux.Zstrides[j];
        } else {
          break;
        }
      }
      Z[i] = X[k];
    }
  }
}

template <typename T>
void TransposeBackward(const T* gZ, T* gX, const TransposeAux& aux) {
  int rank = aux.Z.rank();
  if (rank == 1) {
    LLMath<T>::add(aux.Z.total_dim(), gZ, gX, gX);
  } else if (rank == 2) {
    int i = 0;
    int k = 0;
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        gX[k] += gZ[i++];
        k += aux.Zstrides[1];
      }
      k -= aux.Z[1] * aux.Zstrides[1];
      k += aux.Zstrides[0];
    }
  } else if (rank == 3) {
    int i = 0;
    int k = 0;
    for (int i0 = 0; i0 < aux.Z[0]; ++i0) {
      for (int i1 = 0; i1 < aux.Z[1]; ++i1) {
        for (int i2 = 0; i2 < aux.Z[2]; ++i2) {
          gX[k] += gZ[i++];
          k += aux.Zstrides[2];
        }
        k -= aux.Z[2] * aux.Zstrides[2];
        k += aux.Zstrides[1];
      }
      k -= aux.Z[1] * aux.Zstrides[1];
      k += aux.Zstrides[0];
    }
  } else {
    int index[SHAPE_MAX_RANK] = {0};
    int k = 0;
    int carry;
    gX[0] = gZ[0];
    for (int i = 1; i < aux.Z.total_dim(); ++i) {
      ++index[rank - 1];
      k += aux.Zstrides[rank - 1];
      carry = 0;
      for (int j = rank - 1; j >= 0; --j) {
        index[j] += carry;
        k += carry * aux.Zstrides[j];
        carry = index[j] == aux.Z[j];
        if (carry) {
          index[j] = 0;
          k -= aux.Z[j] * aux.Zstrides[j];
        } else {
          break;
        }
      }
      gX[k] += gZ[i];
    }
  }
}

}  // namespace detail

template <typename T>
void Transpose(const Tensor<T>& X, Tensor<T>* Z,
               const TransposeAux& aux) noexcept {
  detail::Transpose(X.data(), Z->data(), aux);
}

template <typename T>
void TransposeBackward(const Tensor<T>& /*X*/, const Tensor<T>& /*Z*/,
                       const Tensor<T>& gZ, Tensor<T>* gX,
                       const TransposeAux& aux) noexcept {
  detail::TransposeBackward(gZ.data(), gX->data(), aux);
}

}  // namespace deepx_core
