// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>

namespace deepx_core {

/************************************************************************/
/* sage2 implementations */
/************************************************************************/
template <>
inline void LLMath<float>::axpy(int n, float_t alpha, cptr_t x,
                                ptr_t y) noexcept {
  sage2_axpy_ps(n, alpha, x, y);
}

template <>
inline void LLMath<float>::axpby(int n, float_t alpha, cptr_t x, float_t beta,
                                 ptr_t y) noexcept {
  sage2_axpby_ps(n, alpha, x, beta, y);
}

template <>
inline void LLMath<float>::add(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
  sage2_add_ps(n, x, y, z);
}

template <>
inline void LLMath<float>::sub(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
  sage2_sub_ps(n, x, y, z);
}

template <>
inline void LLMath<float>::mul(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
  sage2_mul_ps(n, x, y, z);
}

template <>
inline void LLMath<float>::div(int n, cptr_t x, cptr_t y, ptr_t z) noexcept {
  sage2_div_ps(n, x, y, z);
}

template <>
inline void LLMath<float>::add_scalar(int n, cptr_t x, float_t alpha,
                                      ptr_t y) noexcept {
  sage2_add_scalar_ps(n, x, alpha, y);
}

template <>
inline void LLMath<float>::sub_scalar(int n, cptr_t x, float_t alpha,
                                      ptr_t y) noexcept {
  sage2_sub_scalar_ps(n, x, alpha, y);
}

template <>
inline void LLMath<float>::mul_scalar(int n, cptr_t x, float_t alpha,
                                      ptr_t y) noexcept {
  sage2_mul_scalar_ps(n, x, alpha, y);
}

template <>
inline void LLMath<float>::div_scalar(int n, cptr_t x, float_t alpha,
                                      ptr_t y) noexcept {
  sage2_div_scalar_ps(n, x, alpha, y);
}

template <>
inline void LLMath<float>::sqrt(int n, cptr_t x, ptr_t y) noexcept {
  sage2_sqrt_ps(n, x, y);
}

template <>
inline void LLMath<float>::exp(int n, cptr_t x, ptr_t y) noexcept {
  sage2_exp_ps(n, x, y);
}

template <>
inline void LLMath<float>::log(int n, cptr_t x, ptr_t y) noexcept {
  sage2_log_ps(n, x, y);
}

template <>
inline float LLMath<float>::safe_log(float_t x) noexcept {
  return sage2_log_ss(x);
}

template <>
inline void LLMath<float>::safe_log(int n, cptr_t x, ptr_t y) noexcept {
  sage2_log_ps(n, x, y);
}

template <>
inline float LLMath<float>::sigmoid(float_t x) noexcept {
  return sage2_sigmoid_ss(x);
}

template <>
inline void LLMath<float>::sigmoid(int n, cptr_t x, ptr_t y) noexcept {
  sage2_sigmoid_ps(n, x, y);
}

template <>
inline void LLMath<float>::tanh(int n, cptr_t x, ptr_t y) noexcept {
  sage2_tanh_ps(n, x, y);
}

template <>
inline float LLMath<float>::max(int n, cptr_t x) noexcept {
  return sage2_max_ps(n, x);
}

template <>
inline float LLMath<float>::min(int n, cptr_t x) noexcept {
  return sage2_min_ps(n, x);
}

template <>
inline float LLMath<float>::sum(int n, cptr_t x) noexcept {
  return sage2_sum_ps(n, x);
}

template <>
inline float LLMath<float>::dot(int n, cptr_t x, cptr_t y) noexcept {
  return sage2_dot_ps(n, x, y);
}

template <>
inline float LLMath<float>::norm1(int n, cptr_t x) noexcept {
  return sage2_nrm1_ps(n, x);
}

template <>
inline float LLMath<float>::norm2(int n, cptr_t x) noexcept {
  return sage2_nrm2_ps(n, x);
}

template <>
inline float LLMath<float>::euclidean_distance(int n, cptr_t x,
                                               cptr_t y) noexcept {
  return sage2_euclidean_distance_ps(n, x, y);
}

}  // namespace deepx_core
