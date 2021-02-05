// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
#include <deepx_core/common/read_write_lock.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/tensor/csr_matrix.h>
#include <deepx_core/tensor/ll_math.h>
#include <deepx_core/tensor/sparse_row_matrix.h>
#include <deepx_core/tensor/tensor.h>
#include <deepx_core/tensor/tensor_type.h>
#include <cstdint>
#include <string>

namespace deepx_core {

/************************************************************************/
/* LLTensor */
/************************************************************************/
template <typename T>
class LLTensor {
 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using tsr_t = Tensor<float_t>;
  using ll_math_t = LLMath<float_t>;

 public:
  static int get_total_dim(const tsr_t& X) noexcept { return X.total_dim(); }
  static int get_total_dim(const tsr_t* X) noexcept { return X->total_dim(); }
  static int get_total_dim(tsr_t* X) noexcept { return X->total_dim(); }
  static cptr_t get_data(const tsr_t& X) noexcept { return X.data(); }
  static cptr_t get_data(const tsr_t* X) noexcept { return X->data(); }
  static ptr_t get_data(tsr_t* X) noexcept { return X->data(); }

 public:
  static void axpb(float_t alpha, const tsr_t& x, float_t beta,
                   tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::axpb(get_total_dim(x), alpha, get_data(x), beta, get_data(y));
  }

  static void axpy(float_t alpha, const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::axpy(get_total_dim(x), alpha, get_data(x), get_data(y));
  }

  static void axpby(float_t alpha, const tsr_t& x, float_t beta,
                    tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::axpby(get_total_dim(x), alpha, get_data(x), beta, get_data(y));
  }

  static void xypz(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::xypz(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void xypbz(const tsr_t& x, const tsr_t& y, float_t beta,
                    tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::xypbz(get_total_dim(x), get_data(x), get_data(y), beta,
                     get_data(z));
  }

  static void xdypz(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::xdypz(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void xdypbz(const tsr_t& x, const tsr_t& y, float_t beta,
                     tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::xdypbz(get_total_dim(x), get_data(x), get_data(y), beta,
                      get_data(z));
  }

  static void add(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::add(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void add_scalar(const tsr_t& x, float_t alpha, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::add_scalar(get_total_dim(x), get_data(x), alpha, get_data(y));
  }

  static void add_row(float_t alpha, const tsr_t& X, float_t beta,
                      const tsr_t& y, tsr_t* Z) noexcept {
    DXASSERT_RANK2(X);
    DXASSERT_SAME_SHAPE(X, *Z);
    int m = Z->dim(0);
    int n = Z->dim(1);
    DXASSERT_TOTAL_DIM(y, n);
    ll_math_t::add_row(m, n, alpha, get_data(X), beta, get_data(y),
                       get_data(Z));
  }

  static void add_col(float_t alpha, const tsr_t& X, float_t beta,
                      const tsr_t& y, tsr_t* Z) noexcept {
    DXASSERT_RANK2(X);
    DXASSERT_SAME_SHAPE(X, *Z);
    int m = Z->dim(0);
    int n = Z->dim(1);
    DXASSERT_TOTAL_DIM(y, m);
    ll_math_t::add_col(m, n, alpha, get_data(X), beta, get_data(y),
                       get_data(Z));
  }

  static void sub(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::sub(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void sub_scalar(const tsr_t& x, float_t alpha, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::sub_scalar(get_total_dim(x), get_data(x), alpha, get_data(y));
  }

  static void sub_row(float_t alpha, const tsr_t& X, float_t beta,
                      const tsr_t& y, tsr_t* Z) noexcept {
    add_row(alpha, X, -beta, y, Z);
  }

  static void sub_col(float_t alpha, const tsr_t& X, float_t beta,
                      const tsr_t& y, tsr_t* Z) noexcept {
    add_col(alpha, X, -beta, y, Z);
  }

  static void mul(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::mul(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void mul_scalar(const tsr_t& x, float_t alpha, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::mul_scalar(get_total_dim(x), get_data(x), alpha, get_data(y));
  }

  static void mul_row(const tsr_t& X, const tsr_t& y, tsr_t* Z) noexcept {
    DXASSERT_RANK2(X);
    DXASSERT_SAME_SHAPE(X, *Z);
    int m = Z->dim(0);
    int n = Z->dim(1);
    DXASSERT_TOTAL_DIM(y, n);
    ll_math_t::mul_row(m, n, get_data(X), get_data(y), get_data(Z));
  }

  static void mul_col(const tsr_t& X, const tsr_t& y, tsr_t* Z) noexcept {
    DXASSERT_RANK2(X);
    DXASSERT_SAME_SHAPE(X, *Z);
    int m = Z->dim(0);
    int n = Z->dim(1);
    DXASSERT_TOTAL_DIM(y, m);
    ll_math_t::mul_col(m, n, get_data(X), get_data(y), get_data(Z));
  }

  static void div(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::div(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void div_scalar(const tsr_t& x, float_t alpha, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::div_scalar(get_total_dim(x), get_data(x), alpha, get_data(y));
  }

  static void div_row(const tsr_t& X, const tsr_t& y, tsr_t* Z) noexcept {
    DXASSERT_RANK2(X);
    DXASSERT_SAME_SHAPE(X, *Z);
    int m = Z->dim(0);
    int n = Z->dim(1);
    DXASSERT_TOTAL_DIM(y, n);
    ll_math_t::div_row(m, n, get_data(X), get_data(y), get_data(Z));
  }

  static void div_col(const tsr_t& X, const tsr_t& y, tsr_t* Z) noexcept {
    DXASSERT_RANK2(X);
    DXASSERT_SAME_SHAPE(X, *Z);
    int m = Z->dim(0);
    int n = Z->dim(1);
    DXASSERT_TOTAL_DIM(y, m);
    ll_math_t::div_col(m, n, get_data(X), get_data(y), get_data(Z));
  }

  static void inv(const tsr_t& x, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, *z);
    ll_math_t::inv(get_total_dim(x), get_data(x), get_data(z));
  }

  static void sqrt(const tsr_t& x, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, *z);
    ll_math_t::sqrt(get_total_dim(x), get_data(x), get_data(z));
  }

  static void cbrt(const tsr_t& x, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, *z);
    ll_math_t::cbrt(get_total_dim(x), get_data(x), get_data(z));
  }

  static void square(const tsr_t& x, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, *z);
    ll_math_t::square(get_total_dim(x), get_data(x), get_data(z));
  }

  static void cubic(const tsr_t& x, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, *z);
    ll_math_t::cubic(get_total_dim(x), get_data(x), get_data(z));
  }

  static void pow(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::pow(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void pow_scalar(const tsr_t& x, float_t alpha, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, *z);
    ll_math_t::pow_scalar(get_total_dim(x), get_data(x), alpha, get_data(z));
  }

  static void exp(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::exp(get_total_dim(x), get_data(x), get_data(y));
  }

  static void expm1(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::expm1(get_total_dim(x), get_data(x), get_data(y));
  }

  static void log(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::log(get_total_dim(x), get_data(x), get_data(y));
  }

  static void safe_log(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::safe_log(get_total_dim(x), get_data(x), get_data(y));
  }

  static void sigmoid(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::sigmoid(get_total_dim(x), get_data(x), get_data(y));
  }

  static void sin(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::sin(get_total_dim(x), get_data(x), get_data(y));
  }

  static void asin(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::asin(get_total_dim(x), get_data(x), get_data(y));
  }

  static void sinh(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::sinh(get_total_dim(x), get_data(x), get_data(y));
  }

  static void asinh(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::asinh(get_total_dim(x), get_data(x), get_data(y));
  }

  static void cos(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::cos(get_total_dim(x), get_data(x), get_data(y));
  }

  static void acos(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::acos(get_total_dim(x), get_data(x), get_data(y));
  }

  static void cosh(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::cosh(get_total_dim(x), get_data(x), get_data(y));
  }

  static void acosh(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::acosh(get_total_dim(x), get_data(x), get_data(y));
  }

  static void tan(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::tan(get_total_dim(x), get_data(x), get_data(y));
  }

  static void atan(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::atan(get_total_dim(x), get_data(x), get_data(y));
  }

  static void tanh(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::tanh(get_total_dim(x), get_data(x), get_data(y));
  }

  static void atanh(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::atanh(get_total_dim(x), get_data(x), get_data(y));
  }

  static void abs(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::abs(get_total_dim(x), get_data(x), get_data(y));
  }

  static void max_scalar(float_t alpha, const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::max_scalar(get_total_dim(x), alpha, get_data(x), get_data(y));
  }

  static float_t max(const tsr_t& x) noexcept {
    return ll_math_t::max(get_total_dim(x), get_data(x));
  }

  static void max(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::max(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static void min_scalar(float_t alpha, const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::min_scalar(get_total_dim(x), alpha, get_data(x), get_data(y));
  }

  static float_t min(const tsr_t& x) noexcept {
    return ll_math_t::min(get_total_dim(x), get_data(x));
  }

  static void min(const tsr_t& x, const tsr_t& y, tsr_t* z) noexcept {
    DXASSERT_SAME_SHAPE(x, y, *z);
    ll_math_t::min(get_total_dim(x), get_data(x), get_data(y), get_data(z));
  }

  static float_t sum(const tsr_t& x) noexcept {
    return ll_math_t::sum(get_total_dim(x), get_data(x));
  }

  static void sum_row(float_t alpha, const tsr_t& X, float_t beta,
                      tsr_t* y) noexcept {
    DXASSERT_RANK2(X);
    int m = X.dim(0);
    int n = X.dim(1);
    DXASSERT_TOTAL_DIM(*y, n);
    ll_math_t::sum_row(m, n, alpha, get_data(X), beta, get_data(y));
  }

  static void sum_row(const tsr_t& X, tsr_t* y) noexcept {
    sum_row(1, X, 0, y);
  }

  static void sum_col(float_t alpha, const tsr_t& X, float_t beta,
                      tsr_t* y) noexcept {
    DXASSERT_RANK2(X);
    int m = X.dim(0);
    int n = X.dim(1);
    DXASSERT_TOTAL_DIM(*y, m);
    ll_math_t::sum_col(m, n, alpha, get_data(X), beta, get_data(y));
  }

  static void sum_col(const tsr_t& X, tsr_t* y) noexcept {
    sum_col(1, X, 0, y);
  }

  static float_t dot(const tsr_t& x, const tsr_t& y) noexcept {
    DXASSERT_SAME_SHAPE(x, y);
    return ll_math_t::dot(get_total_dim(x), get_data(x), get_data(y));
  }

  static float_t norm1(const tsr_t& x) noexcept {
    return ll_math_t::norm1(get_total_dim(x), get_data(x));
  }

  static float_t norm2(const tsr_t& x) noexcept {
    return ll_math_t::norm2(get_total_dim(x), get_data(x));
  }

  static float_t euclidean_distance(const tsr_t& x, const tsr_t& y) noexcept {
    DXASSERT_SAME_SHAPE(x, y);
    return ll_math_t::euclidean_distance(get_total_dim(x), get_data(x),
                                         get_data(y));
  }

  static void softmax(const tsr_t& x, tsr_t* y) noexcept {
    DXASSERT_SAME_SHAPE(x, *y);
    ll_math_t::softmax(get_total_dim(x), get_data(x), get_data(y));
  }

  static void gemv(int transA, float_t alpha, const tsr_t& A, const tsr_t& x,
                   float_t beta, tsr_t* y) noexcept {
    DXASSERT_RANK2(A);
    int m = A.dim(0);
    int n = A.dim(1);
    DXASSERT_TOTAL_DIM(x, transA ? m : n);
    DXASSERT_TOTAL_DIM(*y, transA ? n : m);
    ll_math_t::gemv(transA, m, n, alpha, get_data(A), n, get_data(x), beta,
                    get_data(y));
  }

  static void gemv(int transA, const tsr_t& A, const tsr_t& x,
                   tsr_t* y) noexcept {
    gemv(transA, 1, A, x, 0, y);
  }

  static void gemm(int transX, int transY, float_t alpha, const tsr_t& X,
                   const tsr_t& Y, float_t beta, tsr_t* Z) noexcept {
    DXASSERT_RANK2(X);
    DXASSERT_RANK2(Y);
    DXASSERT_RANK2(*Z);
    int m = Z->dim(0);
    int n = Z->dim(1);
    int k = transX ? X.dim(0) : X.dim(1);
    DXASSERT(X.dim(0) == (transX ? k : m));
    DXASSERT(X.dim(1) == (transX ? m : k));
    DXASSERT(Y.dim(0) == (transY ? n : k));
    DXASSERT(Y.dim(1) == (transY ? k : n));
    ll_math_t::gemm(transX, transY, m, n, k, alpha, get_data(X), get_data(Y),
                    beta, get_data(Z));
  }

  static void gemm(int transX, int transY, const tsr_t& X, const tsr_t& Y,
                   tsr_t* Z) noexcept {
    gemm(transX, transY, 1, X, Y, 0, Z);
  }
};

/************************************************************************/
/* LLSparseTensor */
/************************************************************************/
template <typename T, typename I>
class LLSparseTensor : public LLTensor<T> {
 private:
  using base_t = LLTensor<T>;

 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using tsr_t = Tensor<float_t>;
  using srm_t = SparseRowMatrix<float_t, int_t>;
  using csr_t = CSRMatrix<float_t, int_t>;
  using tsri_t = Tensor<int_t>;
  using tsrs_t = Tensor<std::string>;
  using ll_math_t = LLMath<float_t>;

  using base_t::get_data;
  using base_t::get_total_dim;

  using base_t::axpb;
  using base_t::axpby;
  using base_t::axpy;
  using base_t::xdypbz;
  using base_t::xdypz;
  using base_t::xypbz;
  using base_t::xypz;

  using base_t::add;
  using base_t::add_col;
  using base_t::add_row;
  using base_t::add_scalar;
  using base_t::div;
  using base_t::div_col;
  using base_t::div_row;
  using base_t::div_scalar;
  using base_t::mul;
  using base_t::mul_col;
  using base_t::mul_row;
  using base_t::mul_scalar;
  using base_t::sub;
  using base_t::sub_col;
  using base_t::sub_row;
  using base_t::sub_scalar;

  using base_t::cbrt;
  using base_t::cubic;
  using base_t::exp;
  using base_t::expm1;
  using base_t::inv;
  using base_t::log;
  using base_t::pow;
  using base_t::pow_scalar;
  using base_t::safe_log;
  using base_t::sigmoid;
  using base_t::sqrt;
  using base_t::square;

  using base_t::acos;
  using base_t::acosh;
  using base_t::asin;
  using base_t::asinh;
  using base_t::atan;
  using base_t::atanh;
  using base_t::cos;
  using base_t::cosh;
  using base_t::sin;
  using base_t::sinh;
  using base_t::tan;
  using base_t::tanh;

  using base_t::abs;
  using base_t::dot;
  using base_t::euclidean_distance;
  using base_t::max;
  using base_t::max_scalar;
  using base_t::min;
  using base_t::min_scalar;
  using base_t::norm1;
  using base_t::norm2;
  using base_t::softmax;
  using base_t::sum;
  using base_t::sum_col;
  using base_t::sum_row;

  using base_t::gemm;
  using base_t::gemv;

 public:
  // the same logic as libfeature
  static uint16_t get_group_id(int_t feature_id) noexcept {
    return (uint16_t)((feature_id & UINT64_C(0xffff000000000000)) >> 48);
  }

  static int_t get_sub_feature_id(int_t feature_id) noexcept {
    return feature_id & UINT64_C(0x0000ffffffffffff);
  }

  static int_t make_feature_id(uint16_t group_id,
                               int_t sub_feature_id) noexcept {
    return ((int_t)group_id << 48) |
           (sub_feature_id & UINT64_C(0x0000ffffffffffff));
  }

 public:
  // Compute Y = X + beta * Y.
  //
  // beta must be 0 or 1.
  // X: m * n srm
  // Y: m * n tsr
  static void add(const srm_t& X, int beta, tsr_t* Y) noexcept;

  // GESMM: General Sparse Matrix to Matrix Multiplication.
  // Compute Z = X * Y + beta * Z.
  //
  // beta must be 0 or 1.
  // X: m * ? csr
  // Y: k * n tsr
  // Z: m * n tsr
  //
  // A modulo-by-k operation will be performed to cols of X.
  static void gesmm_mod(const csr_t& X, const tsr_t& Y, int beta,
                        tsr_t* Z) noexcept;

  // GESMSM: General Sparse Matrix to Sparse Matrix Multiplication.
  // Compute Z = X * Y + beta * Z.
  //
  // beta must be 0 or 1.
  // X: m * ? csr
  // Y: ? * n srm
  // Z: m * n tsr
  static void gesmsm(const csr_t& X, const srm_t& Y, int beta,
                     tsr_t* Z) noexcept;

  // GESTMM: General Sparse Transposed Matrix to Matrix Multiplication.
  // Compute Z = X.T * Y + beta * Z.
  //
  // beta must be 0 or 1.
  // X:   m * ? csr
  // X.T: ? * m csr
  // Y:   m * n tsr
  // Z:   k * n srm
  //
  // A modulo-by-k operation will be performed to cols of X.
  static void gestmm_mod(int_t k, const csr_t& X, const tsr_t& Y, int beta,
                         srm_t* Z);

  // Compute Z = X.T * Y + beta * Z.
  //
  // beta must be 0 or 1.
  // X:   m * ? csr
  // X.T: ? * m csr
  // Y:   m * n tsr
  // Z:   ? * n srm
  static void gestmm(const csr_t& X, const tsr_t& Y, int beta, srm_t* Z);

  // Compute Z = X + Z.
  static void add_to(const tsr_t& X, tsr_t* Z) noexcept;
  static void add_to(const srm_t& X, srm_t* Z);

  // Compute Z = beta * Z.
  static void scale(float_t beta, tsr_t* Z) noexcept;
  static void scale(float_t beta, srm_t* Z) noexcept;
};

/************************************************************************/
/* LLSparseTensor */
/************************************************************************/
template <typename T, typename I>
void LLSparseTensor<T, I>::add(const srm_t& X, int beta, tsr_t* Y) noexcept {
  DXASSERT_RANK2(*Y);
  int n = Y->dim(1);
  DXASSERT(X.col() == n);
  int i;
  ptr_t _Y = get_data(Y);

  if (beta == 0) {
    Y->zeros();
  }

  if (n == 1) {
    for (const auto& entry : X) {
      i = (int)entry.first;
      DXASSERT(i < Y->dim(0));
      _Y[i] += *entry.second;
    }
  } else {
    cptr_t Xi;
    ptr_t Yi;
    for (const auto& entry : X) {
      i = (int)entry.first;
      DXASSERT(i < Y->dim(0));
      Xi = entry.second;
      Yi = _Y + i * n;
      ll_math_t::add(n, Xi, Yi, Yi);
    }
  }
}

template <typename T, typename I>
void LLSparseTensor<T, I>::gesmm_mod(const csr_t& X, const tsr_t& Y, int beta,
                                     tsr_t* Z) noexcept {
  DXASSERT_RANK2(Y);
  int k = Y.dim(0);
  int n = Y.dim(1);
  DXASSERT(Z->same_shape(X.row(), n));
  cptr_t _Y = get_data(Y);
  ptr_t _Z = get_data(Z);

  if (beta == 0) {
    Z->zeros();
  }

  if (n == 1) {
    float_t Yj;
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        Yj = _Y[CSR_COL(X) % k];
        *_Z += CSR_VALUE(X) * Yj;
      }
      _Z += 1;
    }
  } else {
    cptr_t Yj;
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        Yj = _Y + (CSR_COL(X) % k) * n;
        ll_math_t::axpy(n, CSR_VALUE(X), Yj, _Z);
      }
      _Z += n;
    }
  }
}

template <typename T, typename I>
void LLSparseTensor<T, I>::gesmsm(const csr_t& X, const srm_t& Y, int beta,
                                  tsr_t* Z) noexcept {
  int n = Y.col();
  DXASSERT(Z->same_shape(X.row(), n));
  ptr_t _Z = get_data(Z);

  if (beta == 0) {
    Z->zeros();
  }

  if (n == 1) {
    float_t Yj;
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        Yj = Y.get_scalar_no_init(CSR_COL(X));
        *_Z += CSR_VALUE(X) * Yj;
      }
      _Z += 1;
    }
  } else {
    cptr_t Yj;
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        Yj = Y.get_row_no_init(CSR_COL(X));
        if (Yj) {
          ll_math_t::axpy(n, CSR_VALUE(X), Yj, _Z);
        }
      }
      _Z += n;
    }
  }
}

template <typename T, typename I>
void LLSparseTensor<T, I>::gestmm_mod(int_t k, const csr_t& X, const tsr_t& Y,
                                      int beta, srm_t* Z) {
  int n = Z->col();
  DXASSERT(Y.same_shape(X.row(), n));
  cptr_t _Y = get_data(Y);

  if (beta == 0) {
    Z->zeros();
  }

  if (n == 1) {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        float_t& Zj = Z->get_scalar_no_init(CSR_COL(X) % k);
        Zj += CSR_VALUE(X) * *_Y;
      }
      _Y += 1;
    }
  } else {
    ptr_t Zj;
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        Zj = Z->get_row_no_init(CSR_COL(X) % k);
        ll_math_t::axpy(n, CSR_VALUE(X), _Y, Zj);
      }
      _Y += n;
    }
  }
}

template <typename T, typename I>
void LLSparseTensor<T, I>::gestmm(const csr_t& X, const tsr_t& Y, int beta,
                                  srm_t* Z) {
  int n = Z->col();
  DXASSERT(Y.same_shape(X.row(), n));
  cptr_t _Y = get_data(Y);

  if (beta == 0) {
    Z->zeros();
  }

  if (n == 1) {
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        float_t& Zj = Z->get_scalar_no_init(CSR_COL(X));
        Zj += CSR_VALUE(X) * *_Y;
      }
      _Y += 1;
    }
  } else {
    ptr_t Zj;
    CSR_FOR_EACH_ROW(X, i) {
      CSR_FOR_EACH_COL(X, i) {
        Zj = Z->get_row_no_init(CSR_COL(X));
        ll_math_t::axpy(n, CSR_VALUE(X), _Y, Zj);
      }
      _Y += n;
    }
  }
}

template <typename T, typename I>
void LLSparseTensor<T, I>::add_to(const tsr_t& X, tsr_t* Z) noexcept {
  add(X, *Z, Z);
}

template <typename T, typename I>
void LLSparseTensor<T, I>::add_to(const srm_t& X, srm_t* Z) {
  DXASSERT(X.col() == Z->col());
  for (const auto& entry : X) {
    ptr_t _Z = Z->get_row_no_init(entry.first);
    cptr_t _X = entry.second;
    ll_math_t::add(X.col(), _X, _Z, _Z);
  }
}

template <typename T, typename I>
void LLSparseTensor<T, I>::scale(float_t beta, tsr_t* Z) noexcept {
  mul_scalar(*Z, beta, Z);
}

template <typename T, typename I>
void LLSparseTensor<T, I>::scale(float_t beta, srm_t* Z) noexcept {
  for (auto& entry : *Z) {
    ll_math_t::mul_scalar(Z->col(), entry.second, beta, entry.second);
  }
}

/************************************************************************/
/* LLOptimizer */
/************************************************************************/
template <typename T, typename I>
class LLOptimizer : protected LLSparseTensor<T, I> {
 private:
  using base_t = LLSparseTensor<T, I>;

 public:
  using float_t = T;
  using ptr_t = float_t*;
  using cptr_t = const float_t*;
  using int_t = I;
  using tsr_t = Tensor<float_t>;
  using srm_t = SparseRowMatrix<float_t, int_t>;
  using csr_t = CSRMatrix<float_t, int_t>;
  using tsri_t = Tensor<int_t>;
  using tsrs_t = Tensor<std::string>;
  using ll_math_t = LLMath<float_t>;

  using base_t::get_data;
  using base_t::get_total_dim;

 public:
  static constexpr float_t SMOOTH = (float_t)1e-5;
  static constexpr float_t GRAD_CLIP_THRESHOLD = 20;

 public:
  static void ClipScalar(ptr_t g) noexcept {
    if (*g > GRAD_CLIP_THRESHOLD) {
      *g = GRAD_CLIP_THRESHOLD;
    } else if (*g < -GRAD_CLIP_THRESHOLD) {
      *g = -GRAD_CLIP_THRESHOLD;
    }
  }

  static void ClipArray(int n, ptr_t g) noexcept {
    for (int i = 0; i < n; ++i) {
      ClipScalar(&g[i]);
    }
  }

  static void Clip(tsr_t* G) noexcept {
    ClipArray(get_total_dim(G), get_data(G));
  }

  static void Clip(srm_t* G) noexcept {
    int n = G->col();
    if (n == 1) {
      for (auto& entry : *G) {
        ptr_t g = entry.second;
        ClipScalar(g);
      }
    } else {
      for (auto& entry : *G) {
        ptr_t g = entry.second;
        ClipArray(n, g);
      }
    }
  }

 public:
  template <class Config>
  static void Init(Config*) noexcept {}

  template <class Config>
  static void PreBatch(Config*) noexcept {}

  template <class Config>
  static void PostBatch(Config*) noexcept {}

 public:
  /************************************************************************/
  /* scalar */
  /************************************************************************/
  template <class Config>
  static void UpdateScalar(const Config& config, float_t g, ptr_t w) noexcept;

  template <class Config>
  static void UpdateScalar(const Config& config, float_t g, ptr_t w,
                           ptr_t a) noexcept;

  template <class Config>
  static void UpdateScalar(const Config& config, float_t g, ptr_t w, ptr_t a,
                           ptr_t b) noexcept;

 public:
  /************************************************************************/
  /* array */
  /************************************************************************/
  template <class Config>
  static void UpdateArray(const Config& config, int n, cptr_t g,
                          ptr_t w) noexcept {
    for (int i = 0; i < n; ++i) {
      UpdateScalar(config, g[i], &w[i]);
    }
  }

  template <class Config>
  static void UpdateArray(const Config& config, int n, cptr_t g, ptr_t w,
                          ptr_t a) noexcept {
    for (int i = 0; i < n; ++i) {
      UpdateScalar(config, g[i], &w[i], &a[i]);
    }
  }

  template <class Config>
  static void UpdateArray(const Config& config, int n, cptr_t g, ptr_t w,
                          ptr_t a, ptr_t b) noexcept {
    for (int i = 0; i < n; ++i) {
      UpdateScalar(config, g[i], &w[i], &a[i], &b[i]);
    }
  }

 public:
  /************************************************************************/
  /* grad tsr, param tsr */
  /************************************************************************/
  template <class Config>
  static void UpdateTSR2TSR(const Config& config, const tsr_t& G,
                            tsr_t* W) noexcept {
    UpdateArray(config, get_total_dim(G), get_data(G), get_data(W));
  }

  template <class Config>
  static void UpdateTSR2TSR(const Config& config, const tsr_t& G, tsr_t* W,
                            tsr_t* A) noexcept {
    UpdateArray(config, get_total_dim(G), get_data(G), get_data(W),
                get_data(A));
  }

  template <class Config>
  static void UpdateTSR2TSR(const Config& config, const tsr_t& G, tsr_t* W,
                            tsr_t* A, tsr_t* B) noexcept {
    UpdateArray(config, get_total_dim(G), get_data(G), get_data(W), get_data(A),
                get_data(B));
  }

 public:
  /************************************************************************/
  /* grad srm, param tsr */
  /************************************************************************/
  template <class Config>
  static void UpdateSRM2TSR(const Config& config, const srm_t& G,
                            tsr_t* W) noexcept {
    DXASSERT_RANK2(*W);
    int n = W->dim(1);
    DXASSERT(G.col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, get_data(W) + i);
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, get_data(W) + i * n);
      }
    }
  }

  template <class Config>
  static void UpdateSRM2TSR(const Config& config, const srm_t& G, tsr_t* W,
                            tsr_t* A) noexcept {
    DXASSERT_RANK2(*W);
    int n = W->dim(1);
    DXASSERT(G.col() == n);
    DXASSERT_SAME_SHAPE(*W, *A);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, get_data(W) + i, get_data(A) + i);
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, get_data(W) + i * n, get_data(A) + i * n);
      }
    }
  }

  template <class Config>
  static void UpdateSRM2TSR(const Config& config, const srm_t& G, tsr_t* W,
                            tsr_t* A, tsr_t* B) noexcept {
    DXASSERT_RANK2(*W);
    int n = W->dim(1);
    DXASSERT(G.col() == n);
    DXASSERT_SAME_SHAPE(*W, *A, *B);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, get_data(W) + i, get_data(A) + i,
                     get_data(B) + i);
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, get_data(W) + i * n, get_data(A) + i * n,
                    get_data(B) + i * n);
      }
    }
  }

 public:
  /************************************************************************/
  /* grad srm, param srm */
  /************************************************************************/
  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G, srm_t* W) {
    int n = G.col();
    DXASSERT(W->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, W->get_row_no_init(i));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W->get_row_no_init(i));
      }
    }
  }

  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G, srm_t* W,
                            srm_t* A) {
    int n = G.col();
    DXASSERT(W->col() == n);
    DXASSERT(A->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, W->get_row_no_init(i), A->get_row_no_init(i));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W->get_row_no_init(i), A->get_row_no_init(i));
      }
    }
  }

  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G, srm_t* W,
                            srm_t* A, srm_t* B) {
    int n = G.col();
    DXASSERT(W->col() == n);
    DXASSERT(A->col() == n);
    DXASSERT(B->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, W->get_row_no_init(i), A->get_row_no_init(i),
                     B->get_row_no_init(i));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W->get_row_no_init(i), A->get_row_no_init(i),
                    B->get_row_no_init(i));
      }
    }
  }

  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G, srm_t* W,
                            ReadWriteLock* Wlock) {
    int n = G.col();
    DXASSERT(W->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, W->get_row_no_init(i, Wlock));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W->get_row_no_init(i, Wlock));
      }
    }
  }

  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G, srm_t* W,
                            srm_t* A, ReadWriteLock* Wlock,
                            ReadWriteLock* Alock) {
    int n = G.col();
    DXASSERT(W->col() == n);
    DXASSERT(A->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, W->get_row_no_init(i, Wlock),
                     A->get_row_no_init(i, Alock));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W->get_row_no_init(i, Wlock),
                    A->get_row_no_init(i, Alock));
      }
    }
  }

  template <class Config>
  static void UpdateSRM2SRM(const Config& config, const srm_t& G, srm_t* W,
                            srm_t* A, srm_t* B, ReadWriteLock* Wlock,
                            ReadWriteLock* Alock, ReadWriteLock* Block) {
    int n = G.col();
    DXASSERT(W->col() == n);
    DXASSERT(A->col() == n);
    DXASSERT(B->col() == n);
    if (n == 1) {
      for (const auto& entry : G) {
        int_t i = entry.first;
        float_t g = *entry.second;
        UpdateScalar(config, g, W->get_row_no_init(i, Wlock),
                     A->get_row_no_init(i, Alock),
                     B->get_row_no_init(i, Block));
      }
    } else {
      for (const auto& entry : G) {
        int_t i = entry.first;
        cptr_t g = entry.second;
        UpdateArray(config, n, g, W->get_row_no_init(i, Wlock),
                    A->get_row_no_init(i, Alock), B->get_row_no_init(i, Block));
      }
    }
  }

 public:
  /************************************************************************/
  /* sgd */
  /************************************************************************/
  struct SGDConfig {
    float_t alpha = (float_t)0.01;
    float_t min_alpha = (float_t)1e-6;
    int batch_decay = 128;
    float_t batch_decay_rate = (float_t)0.95;
    int real_batch = 0;
    float_t real_alpha = 0;
  };

  static void Init(SGDConfig* config) noexcept {
    config->real_batch = 0;
    config->real_alpha = config->alpha;
  }

  static void PostBatch(SGDConfig* config) noexcept {
    if (config->batch_decay) {
      if (++config->real_batch >= config->batch_decay) {
        config->real_batch = 0;
        config->real_alpha *= config->batch_decay_rate;
        if (config->real_alpha < config->min_alpha) {
          config->real_alpha = config->min_alpha;
        }
      }
    }
  }

  static void UpdateScalar(const SGDConfig& config, float_t g,
                           ptr_t w) noexcept {
    *w -= config.real_alpha * g;
  }

  static void UpdateArray(const SGDConfig& config, int n, cptr_t G,
                          ptr_t W) noexcept {
    ll_math_t::axpy(n, -config.real_alpha, G, W);
  }

 public:
  /************************************************************************/
  /* ada delta */
  /************************************************************************/
  struct AdaDeltaConfig {
    float_t rho = (float_t)0.95;
    float_t alpha = 1;
    float_t beta = SMOOTH;
    float_t one_sub_rho = 0;
  };

  static void Init(AdaDeltaConfig* config) noexcept {
    config->one_sub_rho = 1 - config->rho;
  }

  static void UpdateScalar(const AdaDeltaConfig& config, float_t g, ptr_t w,
                           ptr_t n, ptr_t deltaw) noexcept {
    float_t new_n = config.rho * *n + config.one_sub_rho * g * g;
    float_t a =
        std::sqrt(*deltaw + config.beta) / std::sqrt(new_n + config.beta) * g;
    float_t new_deltaw = config.rho * *deltaw + config.one_sub_rho * a * a;
    float_t new_w = *w - config.alpha * a;
    *w = new_w;
    *n = new_n;
    *deltaw = new_deltaw;
  }

 public:
  /************************************************************************/
  /* ada grad */
  /************************************************************************/
  struct AdaGradConfig {
    float_t alpha = (float_t)0.01;
    float_t beta = SMOOTH;
  };

  static void UpdateScalar(const AdaGradConfig& config, float_t g, ptr_t w,
                           ptr_t n) noexcept {
    float_t new_n = *n + g * g;
    float_t new_w = *w - g / std::sqrt(new_n + config.beta) * config.alpha;
    *w = new_w;
    *n = new_n;
  }

 public:
  /************************************************************************/
  /* adam */
  /************************************************************************/
  struct AdamConfig {
    float_t rho1 = (float_t)0.9;
    float_t rho2 = (float_t)0.999;
    float_t alpha = (float_t)0.001;
    float_t beta = SMOOTH;
    float_t rho1t = 1;
    float_t rho2t = 1;
    float_t one_sub_rho1 = 0;
    float_t one_sub_rho2 = 0;
    float_t rho_aux = 0;
  };

  static void Init(AdamConfig* config) noexcept {
    config->rho1t = 1;
    config->rho2t = 1;
    config->one_sub_rho1 = 1 - config->rho1;
    config->one_sub_rho2 = 1 - config->rho2;
    config->rho_aux = 0;
  }

  static void PreBatch(AdamConfig* config) noexcept {
    config->rho1t *= config->rho1;
    config->rho2t *= config->rho2;
    config->rho_aux =
        std::sqrt(1 - config->rho2t) / (1 - config->rho1t) * config->alpha;
  }

  static void UpdateScalar(const AdamConfig& config, float_t g, ptr_t w,
                           ptr_t m, ptr_t v) noexcept {
    float_t new_m = config.rho1 * *m + config.one_sub_rho1 * g;
    float_t new_v = config.rho2 * *v + config.one_sub_rho2 * g * g;
    float_t new_w =
        *w - config.rho_aux * new_m / (std::sqrt(new_v) + config.beta);
    *w = new_w;
    *m = new_m;
    *v = new_v;
  }

 public:
  /************************************************************************/
  /* ftrl */
  /************************************************************************/
  struct FTRLConfig {
    float_t alpha = (float_t)0.01;
    float_t beta = 1;
    float_t l1 = 1;
    float_t l2 = 0;
    float_t inv_alpha = 0;
  };

  static void Init(FTRLConfig* config) noexcept {
    config->inv_alpha = 1 / config->alpha;
  }

  static void UpdateScalar(const FTRLConfig& config, float_t g, ptr_t w,
                           ptr_t n, ptr_t z) noexcept {
    float_t new_n = *n + g * g;
    float_t sqrt_n = std::sqrt(*n);
    float_t sqrt_new_n = std::sqrt(new_n);
    float_t sigma = (sqrt_n - sqrt_new_n) * config.inv_alpha;
    float_t new_z = *z + g + sigma * *w;
    float_t z_sign = (new_z < 0) ? (float_t)-1 : (float_t)1;
    float_t z_abs = z_sign * new_z;
    if (z_abs < config.l1) {
      *w = 0;
    } else {
      *w = (z_sign * config.l1 - new_z) /
           ((config.beta + sqrt_new_n) * config.inv_alpha + config.l2);
    }
    *z = new_z;
    *n = new_n;
  }

 public:
  /************************************************************************/
  /* gftrl */
  /************************************************************************/
  struct GFTRLConfig {
    float_t alpha = (float_t)0.1;
    float_t beta = (float_t)0.01;
    float_t lambda = (float_t)1e-4;
    float_t inv_alpha = 0;
  };

  static void Init(GFTRLConfig* config) noexcept {
    config->inv_alpha = 1 / config->alpha;
  }

  static void UpdateScalar(const GFTRLConfig& config, float_t g, ptr_t w,
                           ptr_t n, ptr_t z) noexcept {
    float_t new_n = *n + g * g;
    float_t sqrt_n = std::sqrt(*n);
    float_t sqrt_new_n = std::sqrt(new_n);
    float_t sigma = (sqrt_n - sqrt_new_n) * config.inv_alpha;
    float_t new_z = *z + g + sigma * *w;
    *z = new_z;
    *n = new_n;

    float_t norm2_z = std::fabs(new_z);
    float_t threshold = config.lambda;
    if (norm2_z < threshold) {
      *w = 0;
    } else {
      float_t tmp = config.alpha * (threshold / norm2_z - 1);
      *w = tmp * new_z / (config.beta + sqrt_new_n);
    }
  }

  static void UpdateArray(const GFTRLConfig& config, int _n, cptr_t g, ptr_t w,
                          ptr_t n, ptr_t z) noexcept {
    for (int i = 0; i < _n; ++i) {
      float_t new_n = n[i] + g[i] * g[i];
      float_t sqrt_n = std::sqrt(n[i]);
      float_t sqrt_new_n = std::sqrt(new_n);
      float_t sigma = (sqrt_n - sqrt_new_n) * config.inv_alpha;
      float_t new_z = z[i] + g[i] + sigma * w[i];
      z[i] = new_z;
      n[i] = new_n;
    }

    float_t norm2_z = ll_math_t::norm2(_n, z);
    float_t threshold = config.lambda * std::sqrt((float_t)_n);
    if (norm2_z < threshold) {
      for (int i = 0; i < _n; ++i) {
        w[i] = 0;
      }
    } else {
      float_t tmp = config.alpha * (threshold / norm2_z - 1);
      for (int i = 0; i < _n; ++i) {
        w[i] = tmp * z[i] / (config.beta + std::sqrt(n[i]));
      }
    }
  }

  static void UpdateTSR2TSR(const GFTRLConfig& config, const tsr_t& G, tsr_t* W,
                            tsr_t* A, tsr_t* B) noexcept {
    for (int i = 0; i < get_total_dim(G); ++i) {
      UpdateScalar(config, *(get_data(G) + i), get_data(W) + i, get_data(A) + i,
                   get_data(B) + i);
    }
  }

 public:
  /************************************************************************/
  /* momentum */
  /************************************************************************/
  struct MomentumConfig {
    float_t rho = (float_t)0.5;
    float_t alpha = (float_t)0.1;
  };

  static void UpdateScalar(const MomentumConfig& config, float_t g, ptr_t w,
                           ptr_t v) noexcept {
    float_t new_v = config.rho * *v + g;
    float_t new_w = *w - config.alpha * new_v;
    *w = new_w;
    *v = new_v;
  }

 public:
  /************************************************************************/
  /* rmsprop */
  /************************************************************************/
  struct RMSPropConfig {
    float_t rho = (float_t)0.5;
    float_t alpha = (float_t)0.1;
    float_t beta = SMOOTH;
    float_t one_sub_rho = 0;
  };

  static void Init(RMSPropConfig* config) noexcept {
    config->one_sub_rho = 1 - config->rho;
  }

  static void UpdateScalar(const RMSPropConfig& config, float_t g, ptr_t w,
                           ptr_t v) noexcept {
    float_t new_v = config.rho * *v + config.one_sub_rho * g * g;
    float_t new_w = *w - g / std::sqrt(new_v + config.beta) * config.alpha;
    *w = new_w;
    *v = new_v;
  }
};

}  // namespace deepx_core

#if HAVE_SAGE2 == 1
#include <deepx_core/tensor/ll_tensor_sage2.h>
#endif
