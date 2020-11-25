// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/graph/op_impl.h>

namespace deepx_core {

template <typename T>
struct BinaryAddMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X + *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    LLMath<T>::add(n, X, Y, Z);
  }

  static void Backward(const T* /*X*/, const T* /*Y*/, const T* /*Z*/,
                       const T* gZ, T* gX, T* gY) noexcept {
    if (gX) {
      *gX += *gZ;
    }
    if (gY) {
      *gY += *gZ;
    }
  }

  static void Backward(int n, const T* /*X*/, const T* /*Y*/, const T* /*Z*/,
                       const T* gZ, T* gX, T* gY) noexcept {
    if (gX) {
      LLMath<T>::add(n, gX, gZ, gX);
    }
    if (gY) {
      LLMath<T>::add(n, gY, gZ, gY);
    }
  }
};

template <typename T>
struct BinarySubMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X - *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    LLMath<T>::sub(n, X, Y, Z);
  }

  static void Backward(const T* /*X*/, const T* /*Y*/, const T* /*Z*/,
                       const T* gZ, T* gX, T* gY) noexcept {
    if (gX) {
      *gX += *gZ;
    }
    if (gY) {
      *gY -= *gZ;
    }
  }

  static void Backward(int n, const T* /*X*/, const T* /*Y*/, const T* /*Z*/,
                       const T* gZ, T* gX, T* gY) noexcept {
    if (gX) {
      LLMath<T>::add(n, gX, gZ, gX);
    }
    if (gY) {
      LLMath<T>::sub(n, gY, gZ, gY);
    }
  }
};

template <typename T>
struct BinaryMulMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X * *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    LLMath<T>::mul(n, X, Y, Z);
  }

  static void Backward(const T* X, const T* Y, const T* /*Z*/, const T* gZ,
                       T* gX, T* gY) noexcept {
    if (gX) {
      *gX += *gZ * *Y;
    }
    if (gY) {
      *gY += *gZ * *X;
    }
  }

  static void Backward(int n, const T* X, const T* Y, const T* /*Z*/,
                       const T* gZ, T* gX, T* gY) noexcept {
    if (gX) {
      LLMath<T>::xypz(n, gZ, Y, gX);
    }
    if (gY) {
      LLMath<T>::xypz(n, gZ, X, gY);
    }
  }
};

template <typename T>
struct BinaryDivMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept {
    DXASSERT(*Y != 0);
    *Z = *X / *Y;
  }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
#if !defined NDEBUG
    for (int i = 0; i < n; ++i) {
      DXASSERT(Y[i] != 0);
    }
#endif
    LLMath<T>::div(n, X, Y, Z);
  }

  static void Backward(const T* /*X*/, const T* Y, const T* Z, const T* gZ,
                       T* gX, T* gY) noexcept {
    T tmp = *gZ / *Y;
    if (gX) {
      *gX += tmp;
    }
    if (gY) {
      *gY -= tmp * *Z;
    }
  }

  static void Backward(int n, const T* /*X*/, const T* Y, const T* Z,
                       const T* gZ, T* gX, T* gY) noexcept {
    T tmp;
    for (int i = 0; i < n; ++i) {
      tmp = gZ[i] / Y[i];
      if (gX) {
        gX[i] += tmp;
      }
      if (gY) {
        gY[i] -= tmp * Z[i];
      }
    }
  }
};

template <typename T>
struct BinaryPowMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept {
    *Z = std::pow(*X, *Y);
  }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    LLMath<T>::pow(n, X, Y, Z);
  }

  static void Backward(const T* X, const T* Y, const T* Z, const T* gZ, T* gX,
                       T* gY) noexcept {
    if (gX) {
      if (*X != 0) {
        *gX += *Y * *Z / *X * *gZ;
      }
    }
    if (gY) {
      if (*X > 0) {
        *gY += *Z * std::log(*X) * *gZ;
      }
    }
  }

  static void Backward(int n, const T* X, const T* Y, const T* Z, const T* gZ,
                       T* gX, T* gY) noexcept {
    for (int i = 0; i < n; ++i) {
      if (gX) {
        if (X[i] != 0) {
          gX[i] += Y[i] * Z[i] / X[i] * gZ[i];
        }
      }
      if (gY) {
        if (X[i] > 0) {
          gY[i] += Z[i] * std::log(X[i]) * gZ[i];
        }
      }
    }
  }
};

template <typename T>
struct BinaryMaxMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept {
    *Z = (*X > *Y) ? *X : *Y;
  }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    for (int i = 0; i < n; ++i) {
      Forward(X + i, Y + i, Z + i);
    }
  }

  static void Backward(const T* X, const T* Y, const T* /*Z*/, const T* gZ,
                       T* gX, T* gY) noexcept {
    if (*X > *Y) {
      if (gX) {
        *gX += *gZ;
      }
    } else {
      if (gY) {
        *gY += *gZ;
      }
    }
  }

  static void Backward(int n, const T* X, const T* Y, const T* /*Z*/,
                       const T* gZ, T* gX, T* gY) noexcept {
    for (int i = 0; i < n; ++i) {
      if (X[i] > Y[i]) {
        if (gX) {
          gX[i] += gZ[i];
        }
      } else {
        if (gY) {
          gY[i] += gZ[i];
        }
      }
    }
  }
};

template <typename T>
struct BinaryMinMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept {
    *Z = (*X < *Y) ? *X : *Y;
  }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    for (int i = 0; i < n; ++i) {
      Forward(X + i, Y + i, Z + i);
    }
  }

  static void Backward(const T* X, const T* Y, const T* /*Z*/, const T* gZ,
                       T* gX, T* gY) noexcept {
    if (*X < *Y) {
      if (gX) {
        *gX += *gZ;
      }
    } else {
      if (gY) {
        *gY += *gZ;
      }
    }
  }

  static void Backward(int n, const T* X, const T* Y, const T* /*Z*/,
                       const T* gZ, T* gX, T* gY) noexcept {
    for (int i = 0; i < n; ++i) {
      if (X[i] < Y[i]) {
        if (gX) {
          gX[i] += gZ[i];
        }
      } else {
        if (gY) {
          gY[i] += gZ[i];
        }
      }
    }
  }
};

template <typename T>
struct BinaryEqualMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X == *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    for (int i = 0; i < n; ++i) {
      Forward(X + i, Y + i, Z + i);
    }
  }
};

template <typename T>
struct BinaryGreaterMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X > *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    for (int i = 0; i < n; ++i) {
      Forward(X + i, Y + i, Z + i);
    }
  }
};

template <typename T>
struct BinaryGreaterEqualMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X >= *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    for (int i = 0; i < n; ++i) {
      Forward(X + i, Y + i, Z + i);
    }
  }
};

template <typename T>
struct BinaryLessMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X < *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    for (int i = 0; i < n; ++i) {
      Forward(X + i, Y + i, Z + i);
    }
  }
};

template <typename T>
struct BinaryLessEqualMeta {
  static void Forward(const T* X, const T* Y, T* Z) noexcept { *Z = *X <= *Y; }

  static void Forward(int n, const T* X, const T* Y, T* Z) noexcept {
    for (int i = 0; i < n; ++i) {
      Forward(X + i, Y + i, Z + i);
    }
  }
};

}  // namespace deepx_core
