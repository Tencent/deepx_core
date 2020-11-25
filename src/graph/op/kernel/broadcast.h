// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/graph/op_impl.h>
#include "binary_meta.h"

namespace deepx_core {

// Bidirectional broadcast rule.
// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

struct BroadcastAux {
  Shape Z;
  Shape Xstrides;
  Shape Ystrides;
  Shape Zstrides;
  int Z_total_dim = 0;
  int XY_same_shape = 0;
  int vectorization = 0;  // last axis can be vectorized.
};

inline bool BroadcastPrepare(const Shape& X, const Shape& Y,
                             BroadcastAux* aux) noexcept {
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

  int Zrank = (Xrank < Yrank) ? Yrank : Xrank;
  int Zrank_remain = SHAPE_MAX_RANK - Zrank;
  using ai_t = std::array<int, SHAPE_MAX_RANK>;
  ai_t X_reverse_dims, Y_reverse_dims, Z_reverse_dims;
  ai_t X_reverse_strides, Y_reverse_strides, Z_reverse_strides;
  int Xstride = 1, Ystride = 1, Zstride = 1;

  std::copy(X.rbegin(), X.rend(), X_reverse_dims.begin());
  for (int i = Xrank; i < Zrank; ++i) {
    X_reverse_dims[i] = 1;
  }

  std::copy(Y.rbegin(), Y.rend(), Y_reverse_dims.begin());
  for (int i = Yrank; i < Zrank; ++i) {
    Y_reverse_dims[i] = 1;
  }

  for (int i = 0; i < Zrank; ++i) {
    if (X_reverse_dims[i] == Y_reverse_dims[i]) {
      Z_reverse_dims[i] = X_reverse_dims[i];
    } else if (X_reverse_dims[i] == 1) {
      Z_reverse_dims[i] = Y_reverse_dims[i];
    } else if (Y_reverse_dims[i] == 1) {
      Z_reverse_dims[i] = X_reverse_dims[i];
    } else {
      DXERROR("Couldn't bidirectional broadcast %s to %s.",
              to_string(X).c_str(), to_string(Y).c_str());
      return false;
    }
  }
  aux->Z.assign(Z_reverse_dims.rbegin() + Zrank_remain, Z_reverse_dims.rend());

  for (int i = 0; i < Zrank; ++i) {
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

    if (Z_reverse_dims[i] == 1) {
      Z_reverse_strides[i] = 0;
    } else {
      Z_reverse_strides[i] = Zstride;
      Zstride *= Z_reverse_dims[i];
    }
  }
  aux->Xstrides.assign(X_reverse_strides.rbegin() + Zrank_remain,
                       X_reverse_strides.rend());
  aux->Ystrides.assign(Y_reverse_strides.rbegin() + Zrank_remain,
                       Y_reverse_strides.rend());
  aux->Zstrides.assign(Z_reverse_strides.rbegin() + Zrank_remain,
                       Z_reverse_strides.rend());
  aux->Z_total_dim = Zstride;
  aux->XY_same_shape = X == Y;
  aux->vectorization = (X_reverse_strides[0] == 1) &&
                       (Y_reverse_strides[0] == 1) &&
                       (Z_reverse_strides[0] == 1);
  return true;
}

}  // namespace deepx_core
