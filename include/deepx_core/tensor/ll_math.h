// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <cmath>
#include <cstring>  // memset, memcpy

namespace deepx_core {

/************************************************************************/
/* LLMath */
/************************************************************************/
template <typename T>
class LLMath {
 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;

 public:
  // Naming conventions.
  // 1. alpha, beta are scalars.
  // 2. x, y, z are vectors.
  // 3. X, Y, Z are matrices or tensors.
  // 4, m, n, k are dims.

  // Set y zero.
  static void zero(int n, ptr_t y) noexcept {
    memset(y, 0, n * sizeof(float_t));
  }

  // Set y = x.
  static void copy(int n, cptr_t x, ptr_t y) noexcept {
    if (x != y) {
      memcpy(y, x, n * sizeof(float_t));
    }
  }

  /************************************************************************/
  /* fused mul/div add */
  /************************************************************************/
  // Compute y = alpha * x + beta.
  static void axpb(int n, float_t alpha, cptr_t x, float_t beta,
                   ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = alpha * x[i] + beta;
    }
  }

  // Compute y = alpha * x + y.
  static void axpy(int n, float_t alpha, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] += alpha * x[i];
    }
  }

  // Compute y = alpha * x + beta * y.
  static void axpby(int n, float_t alpha, cptr_t x, float_t beta,
                    ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = alpha * x[i] + beta * y[i];
    }
  }

  // Compute z = x * y + z.
  static void xypz(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] += x[i] * y[i];
    }
  }

  // Compute z = x * y + beta * z.
  static void xypbz(int n, cptr_t x, cptr_t y, float_t beta, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] * y[i] + beta * z[i];
    }
  }

  // Compute z = x / y + z.
  static void xdypz(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] += x[i] / y[i];
    }
  }

  // Compute z = x / y + beta * z.
  static void xdypbz(int n, cptr_t x, cptr_t y, float_t beta,
                     ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] / y[i] + beta * z[i];
    }
  }

  /************************************************************************/
  /* add/sub/mul/div */
  /************************************************************************/
  // Element-wise compute z = x + y.
  static void add(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] + y[i];
    }
  }

  // Broadcast compute y = x + alpha.
  static void add_scalar(int n, cptr_t x, float_t alpha, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = x[i] + alpha;
    }
  }

  // Broadcast compute Z = alpha * X + beta * y.
  //
  // X: m * n buffer
  // y: 1 * n buffer
  // Z: m * n buffer
  static void add_row(int m, int n, float_t alpha, cptr_t X, float_t beta,
                      cptr_t y, ptr_t Z) noexcept {
    if (X != Z) {
      copy(m * n, X, Z);
    }
    for (int i = 0; i < m; ++i) {
      axpby(n, beta, y, alpha, Z);
      Z += n;
    }
  }

  // Broadcast compute Z = alpha * X + beta * y.
  //
  // X: m * n buffer
  // y: m * 1 buffer
  // Z: m * n buffer
  static void add_col(int m, int n, float_t alpha, cptr_t X, float_t beta,
                      cptr_t y, ptr_t Z) noexcept {
    for (int i = 0; i < m; ++i) {
      axpb(n, alpha, X, beta * y[i], Z);
      X += n;
      Z += n;
    }
  }

  // Element-wise compute z = x - y.
  static void sub(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] - y[i];
    }
  }

  // Broadcast compute y = x - alpha.
  static void sub_scalar(int n, cptr_t x, float_t alpha, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = x[i] - alpha;
    }
  }

  // Broadcast compute Z = alpha * X - beta * y.
  //
  // X: m * n buffer
  // y: 1 * n buffer
  // Z: m * n buffer
  static void sub_row(int m, int n, float_t alpha, cptr_t X, float_t beta,
                      cptr_t y, ptr_t Z) noexcept {
    add_row(m, n, alpha, X, -beta, y, Z);
  }

  // Broadcast compute Z = alpha * X - beta * y.
  //
  // X: m * n buffer
  // y: m * 1 buffer
  // Z: m * n buffer
  static void sub_col(int m, int n, float_t alpha, cptr_t X, float_t beta,
                      cptr_t y, ptr_t Z) noexcept {
    add_col(m, n, alpha, X, -beta, y, Z);
  }

  // Element-wise compute z = x * y.
  static void mul(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] * y[i];
    }
  }

  // Broadcast compute y = x * alpha.
  static void mul_scalar(int n, cptr_t x, float_t alpha, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = x[i] * alpha;
    }
  }

  // Broadcast compute Z = X * y.
  //
  // X: m * n buffer
  // y: 1 * n buffer
  // Z: m * n buffer
  static void mul_row(int m, int n, cptr_t X, cptr_t y, ptr_t Z) noexcept {
    for (int i = 0; i < m; ++i) {
      mul(n, X, y, Z);
      X += n;
      Z += n;
    }
  }

  // Broadcast compute Z = X * y.
  //
  // X: m * n buffer
  // y: m * 1 buffer
  // Z: m * n buffer
  static void mul_col(int m, int n, cptr_t X, cptr_t y, ptr_t Z) noexcept {
    for (int i = 0; i < m; ++i) {
      mul_scalar(n, X, y[i], Z);
      X += n;
      Z += n;
    }
  }

  // Element-wise compute z = x / y.
  static void div(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] / y[i];
    }
  }

  // Broadcast compute y = x / alpha.
  static void div_scalar(int n, cptr_t x, float_t alpha, ptr_t y) noexcept {
    float_t inv_alpha = 1 / alpha;
    for (int i = 0; i < n; ++i) {
      y[i] = x[i] * inv_alpha;
    }
  }

  // Broadcast compute Z = X / y.
  //
  // X: m * n buffer
  // y: 1 * n buffer
  // Z: m * n buffer
  static void div_row(int m, int n, cptr_t X, cptr_t y, ptr_t Z) noexcept {
    for (int i = 0; i < m; ++i) {
      div(n, X, y, Z);
      X += n;
      Z += n;
    }
  }

  // Broadcast compute Z = X / y.
  //
  // X: m * n buffer
  // y: m * 1 buffer
  // Z: m * n buffer
  static void div_col(int m, int n, cptr_t X, cptr_t y, ptr_t Z) noexcept {
    for (int i = 0; i < m; ++i) {
      div_scalar(n, X, y[i], Z);
      X += n;
      Z += n;
    }
  }

  /************************************************************************/
  /* element-wise */
  /************************************************************************/
  // Element-wise compute y = 1 / x.
  static void inv(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = 1 / x[i];
    }
  }

  // Element-wise compute y = sqrt(x).
  static void sqrt(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::sqrt(x[i]);
    }
  }

  // Element-wise compute y = cbrt(x).
  static void cbrt(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::cbrt(x[i]);
    }
  }

  // Element-wise compute y = square(x).
  static void square(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = x[i] * x[i];
    }
  }

  // Element-wise compute y = cubic(x).
  static void cubic(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = x[i] * x[i] * x[i];
    }
  }

  // Element-wise compute z = pow(x, y).
  static void pow(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = std::pow(x[i], y[i]);
    }
  }

  // Broadcast compute y = pow(x, alpha).
  static void pow_scalar(int n, cptr_t x, float_t alpha, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::pow(x[i], alpha);
    }
  }

  // Element-wise compute y = exp(x).
  static void exp(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::exp(x[i]);
    }
  }

  // Element-wise compute y = exp(x) - 1.
  static void expm1(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::expm1(x[i]);
    }
  }

  // Element-wise compute y = ln(x).
  static void log(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::log(x[i]);
    }
  }

  // Compute log(x).
  //
  // safe_log never returns -inf, so it is numerically safe.
  static float_t safe_log(float_t x) noexcept {
    return (x <= (float_t)1e-6) ? (float_t)-13.815510557964274 : std::log(x);
  }

  // Element-wise compute y = safe_log(x).
  static void safe_log(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = safe_log(x[i]);
    }
  }

  // Compute sigmoid(x).
  static float_t sigmoid(float_t x) noexcept { return 1 / (std::exp(-x) + 1); }

  // Element-wise compute y = sigmoid(x).
  static void sigmoid(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = 1 / (std::exp(-x[i]) + 1);
    }
  }

  // Element-wise compute y = sin(x).
  static void sin(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::sin(x[i]);
    }
  }

  // Element-wise compute y = asin(x).
  static void asin(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::asin(x[i]);
    }
  }

  // Element-wise compute y = sinh(x).
  static void sinh(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::sinh(x[i]);
    }
  }

  // Element-wise compute y = asinh(x).
  static void asinh(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::asinh(x[i]);
    }
  }

  // Element-wise compute y = cos(x).
  static void cos(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::cos(x[i]);
    }
  }

  // Element-wise compute y = acos(x).
  static void acos(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::acos(x[i]);
    }
  }

  // Element-wise compute y = cosh(x).
  static void cosh(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::cosh(x[i]);
    }
  }

  // Element-wise compute y = acosh(x).
  static void acosh(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::acosh(x[i]);
    }
  }

  // Element-wise compute y = tan(x).
  static void tan(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::tan(x[i]);
    }
  }

  // Element-wise compute y = atan(x).
  static void atan(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::atan(x[i]);
    }
  }

  // Element-wise compute y = tanh(x).
  static void tanh(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::tanh(x[i]);
    }
  }

  // Element-wise compute y = atanh(x).
  static void atanh(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::atanh(x[i]);
    }
  }

  // Element-wise compute y = abs(x).
  static void abs(int n, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = std::fabs(x[i]);
    }
  }

  /************************************************************************/
  /* max/min/sum */
  /************************************************************************/
  // Broadcast compute y = max(alpha, x).
  static void max_scalar(int n, float_t alpha, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = (x[i] < alpha) ? alpha : x[i];
    }
  }

  // Compute max(x).
  static float_t max(int n, cptr_t x) noexcept {
    float_t m = x[0];
    for (int i = 1; i < n; ++i) {
      if (m < x[i]) {
        m = x[i];
      }
    }
    return m;
  }

  // Element-wise compute z = max(x, y).
  static void max(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = (x[i] < y[i]) ? y[i] : x[i];
    }
  }

  // Broadcast compute y = min(alpha, x).
  static void min_scalar(int n, float_t alpha, cptr_t x, ptr_t y) noexcept {
    for (int i = 0; i < n; ++i) {
      y[i] = (x[i] > alpha) ? alpha : x[i];
    }
  }

  // Compute min(x).
  static float_t min(int n, cptr_t x) noexcept {
    float_t m = x[0];
    for (int i = 1; i < n; ++i) {
      if (m > x[i]) {
        m = x[i];
      }
    }
    return m;
  }

  // Element-wise compute z = min(x, y).
  static void min(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
    for (int i = 0; i < n; ++i) {
      z[i] = (x[i] > y[i]) ? y[i] : x[i];
    }
  }

  // Compute sum(x).
  static float_t sum(int n, cptr_t x) noexcept {
    float_t s = 0;
    for (int i = 0; i < n; ++i) {
      s += x[i];
    }
    return s;
  }

  // Compute y = alpha * sum(X, axis=0) + beta * y.
  //
  // X: m * n buffer
  // y: 1 * n buffer
  static void sum_row(int m, int n, float_t alpha, cptr_t X, float_t beta,
                      ptr_t y) noexcept {
    if (beta == 0) {
      zero(n, y);
    } else if (beta != 1) {
      mul_scalar(n, y, beta, y);
    }

    if (alpha == 0) {
      return;
    }

    for (int i = 0; i < m; ++i) {
      axpy(n, alpha, X, y);
      X += n;
    }
  }

  // Compute y = alpha * sum(X, axis=1) + beta * y.
  //
  // X: m * n buffer
  // y: m * 1 buffer
  static void sum_col(int m, int n, float_t alpha, cptr_t X, float_t beta,
                      ptr_t y) noexcept {
    if (beta == 0) {
      zero(m, y);
    } else if (beta != 1) {
      mul_scalar(m, y, beta, y);
    }

    if (alpha == 0) {
      return;
    }

    for (int i = 0; i < m; ++i) {
      y[i] += alpha * sum(n, X);
      X += n;
    }
  }

  // Compute dot product of x and y.
  static float_t dot(int n, cptr_t x, cptr_t y) noexcept {
    float_t s = 0;
    for (int i = 0; i < n; ++i) {
      s += x[i] * y[i];
    }
    return s;
  }

  // Compute L1 norm of x.
  static float_t norm1(int n, cptr_t x) noexcept {
    float_t s = 0;
    for (int i = 0; i < n; ++i) {
      s += x[i] >= 0 ? x[i] : -x[i];
    }
    return s;
  }

  // Compute L2 norm of x.
  static float_t norm2(int n, cptr_t x) noexcept {
    return std::sqrt(dot(n, x, x));
  }

  // Compute euclidean distance of x and y.
  static float_t euclidean_distance(int n, cptr_t x, cptr_t y) noexcept {
    float_t s = 0;
    for (int i = 0; i < n; ++i) {
      s += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return std::sqrt(s);
  }

  // Compute y = softmax(x).
  static void softmax(int n, cptr_t x, ptr_t y) noexcept {
    sub_scalar(n, x, max(n, x), y);
    exp(n, y, y);
    mul_scalar(n, y, 1 / sum(n, y), y);
  }

  // Compute y = softmax(x) * n.
  static void softmax2(int n, cptr_t x, ptr_t y) noexcept {
    sub_scalar(n, x, max(n, x), y);
    exp(n, y, y);
    mul_scalar(n, y, n / sum(n, y), y);
  }

  /************************************************************************/
  /* GEMV: General Matrix to Vector Multiplication */
  /************************************************************************/
  // Compute y = alpha * op(A) * x + beta * y.
  //
  // transA is 0.
  //   op(A) = A
  //   A: m * n buffer
  //   x: n * 1 buffer
  //   y: m * 1 buffer
  // transA is 1.
  //   op(A) = A.T
  //   A: m * n buffer
  //   x: m * 1 buffer
  //   y: n * 1 buffer
  // ldA: leading dim of A
  static void gemv(int transA, int m, int n, float_t alpha, cptr_t A, int ldA,
                   cptr_t x, float_t beta, ptr_t y) noexcept;

  // Compute y = alpha * op(A) * x + beta * y.
  static void gemv(int transA, int m, int n, float_t alpha, cptr_t A, cptr_t x,
                   float_t beta, ptr_t y) noexcept {
    gemv(transA, m, n, alpha, A, n, x, beta, y);
  }

  // Compute y = op(A) * x.
  static void gemv(int transA, int m, int n, cptr_t A, cptr_t x,
                   ptr_t y) noexcept {
    gemv(transA, m, n, 1, A, n, x, 0, y);
  }

  /************************************************************************/
  /* GEMM: General Matrix to Matrix Multiplication */
  /************************************************************************/
  // Compute Z = alpha * op(X) * op(Y) + beta * Z.
  //
  // transX is 0: op(X) = X
  // transX is 1: op(X) = X.T
  // transY is 0: op(Y) = Y
  // transY is 1: op(Y) = Y.T
  // op(X):       m * k buffer
  // op(Y):       k * n buffer
  // Z:           m * n buffer
  // ldX:         leading dim of X
  // ldY:         leading dim of Y
  // ldZ:         leading dim of Z
  static void gemm(int transX, int transY, int m, int n, int k, float_t alpha,
                   cptr_t X, int ldX, cptr_t Y, int ldY, float_t beta, ptr_t Z,
                   int ldZ) noexcept;

  // Compute Z = alpha * op(X) * op(Y) + beta * Z.
  static void gemm(int transX, int transY, int m, int n, int k, float_t alpha,
                   cptr_t X, cptr_t Y, float_t beta, ptr_t Z) noexcept {
    int ldX = transX ? m : k;
    int ldY = transY ? k : n;
    int ldZ = n;
    gemm(transX, transY, m, n, k, alpha, X, ldX, Y, ldY, beta, Z, ldZ);
  }

  // Compute Z = op(X) * op(Y).
  static void gemm(int transX, int transY, int m, int n, int k, cptr_t X,
                   cptr_t Y, ptr_t Z) noexcept {
    gemm(transX, transY, m, n, k, 1, X, Y, 0, Z);
  }
};

/************************************************************************/
/* LLMath */
/************************************************************************/
template <typename T>
void LLMath<T>::gemv(int transA, int m, int n, float_t alpha, cptr_t A, int ldA,
                     cptr_t x, float_t beta, ptr_t y) noexcept {
  cptr_t pA;
  float_t tmp;
  int ly;

  if (alpha == 0 && beta == 1) {
    return;
  }

  // y = beta * y
  if (!transA) {
    ly = m;
  } else {
    ly = n;
  }
  if (beta == 0) {
    zero(ly, y);
  } else if (beta != 1) {
    mul_scalar(ly, y, beta, y);
  }

  if (alpha == 0) {
    return;
  }

  // y += alpha * op(A) * x
  if (!transA) {
    pA = A;
    for (int i = 0; i < m; ++i) {
      y[i] += alpha * dot(n, pA, x);
      pA += ldA;
    }
  } else {
    pA = A;
    for (int i = 0; i < m; ++i) {
      tmp = alpha * x[i];
      if (tmp != 0) {
        axpy(n, tmp, pA, y);
      }
      pA += ldA;
    }
  }
}

template <typename T>
void LLMath<T>::gemm(int transX, int transY, int m, int n, int k, float_t alpha,
                     cptr_t X, int ldX, cptr_t Y, int ldY, float_t beta,
                     ptr_t Z, int ldZ) noexcept {
  cptr_t pX;
  cptr_t pY;
  ptr_t pZ;
  float_t tmp;

  if (alpha == 0 && beta == 1) {
    return;
  }

  // Z = beta * Z
  if (n == ldZ) {
    if (beta == 0) {
      zero(m * n, Z);
    } else if (beta != 1) {
      mul_scalar(m * n, Z, beta, Z);
    }
  } else {
    // n < ldZ
    if (beta == 0) {
      pZ = Z;
      for (int i = 0; i < m; ++i) {
        zero(n, pZ);
        pZ += ldZ;
      }
    } else if (beta != 1) {
      pZ = Z;
      for (int i = 0; i < m; ++i) {
        mul_scalar(n, pZ, beta, pZ);
        pZ += ldZ;
      }
    }
  }

  // Z += alpha * op(X) * op(Y)
  if (alpha == 0) {
    return;
  }

  if (!transX && !transY) {
    pY = Y;
    for (int kk = 0; kk < k; ++kk) {
      pX = X;
      pZ = Z;
      for (int i = 0; i < m; ++i) {
        tmp = alpha * pX[kk];
        if (tmp != 0) {
          axpy(n, tmp, pY, pZ);
        }
        pX += ldX;
        pZ += ldZ;
      }
      pY += ldY;
    }
  } else if (!transX && transY) {
    pX = X;
    pZ = Z;
    for (int i = 0; i < m; ++i) {
      pY = Y;
      for (int j = 0; j < n; ++j) {
        pZ[j] += alpha * dot(k, pX, pY);
        pY += ldY;
      }
      pX += ldX;
      pZ += ldZ;
    }
  } else if (transX && !transY) {
    pX = X;
    pY = Y;
    for (int kk = 0; kk < k; ++kk) {
      pZ = Z;
      for (int i = 0; i < m; ++i) {
        tmp = alpha * pX[i];
        if (tmp != 0) {
          axpy(n, tmp, pY, pZ);
        }
        pZ += ldZ;
      }
      pX += ldX;
      pY += ldY;
    }
  } else {
    pZ = Z;
    for (int i = 0; i < m; ++i) {
      pY = Y;
      for (int j = 0; j < n; ++j) {
        pX = X;
        tmp = 0;
        for (int kk = 0; kk < k; ++kk) {
          tmp += pX[i] * pY[kk];
          pX += ldX;
        }
        pY += ldY;
        pZ[j] += alpha * tmp;
      }
      pZ += ldZ;
    }
  }
}

}  // namespace deepx_core

#if HAVE_SAGE2 == 1
#include <deepx_core/tensor/ll_math_sage2.h>
#endif
#if HAVE_SAGE2_SGEMM == 1
#include <deepx_core/tensor/ll_math_sage2_sgemm.h>
#endif
