// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_VMF_H_
#define SAGE2_VMF_H_

#include <sage2/macro.h>
#include <stddef.h>  // NOLINT
#include <stdint.h>  // NOLINT
#include <string.h>  // NOLINT

/************************************************************************/
/* sage2_zero_ps */
/* Set y zero. */
/************************************************************************/
#define sage2_zero_ps(n, y)                  \
  do {                                       \
    memset(y, 0, (size_t)n * sizeof(float)); \
  } while (0)

/************************************************************************/
/* sage2_axpy_ps */
/* Compute y = alpha * x + y. */
/************************************************************************/
SAGE2_C_API void sage2_axpy_ps(uint64_t n, float alpha, const float* x,
                               float* y);

/************************************************************************/
/* sage2_axpby_ps */
/* Compute y = alpha * x + beta * y. */
/************************************************************************/
SAGE2_C_API void sage2_axpby_ps(uint64_t n, float alpha, const float* x,
                                float beta, float* y);

/************************************************************************/
/* sage2_add_ps */
/* Compute z = x + y. */
/************************************************************************/
SAGE2_C_API void sage2_add_ps(uint64_t n, const float* x, const float* y,
                              float* z);

/************************************************************************/
/* sage2_sub_ps */
/* Compute z = x - y. */
/************************************************************************/
SAGE2_C_API void sage2_sub_ps(uint64_t n, const float* x, const float* y,
                              float* z);

/************************************************************************/
/* sage2_mul_ps */
/* Compute z = x * y. */
/************************************************************************/
SAGE2_C_API void sage2_mul_ps(uint64_t n, const float* x, const float* y,
                              float* z);

/************************************************************************/
/* sage2_div_ps */
/* Compute z = x / y. */
/************************************************************************/
SAGE2_C_API void sage2_div_ps(uint64_t n, const float* x, const float* y,
                              float* z);

/************************************************************************/
/* sage2_add_scalar_ps */
/* Compute y = x + alpha. */
/************************************************************************/
SAGE2_C_API void sage2_add_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);

/************************************************************************/
/* sage2_sub_scalar_ps */
/* Compute y = x - alpha. */
/************************************************************************/
SAGE2_C_API void sage2_sub_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);

/************************************************************************/
/* sage2_mul_scalar_ps */
/* Compute y = x * alpha. */
/************************************************************************/
SAGE2_C_API void sage2_mul_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);

/************************************************************************/
/* sage2_div_scalar_ps */
/* Compute y = x / alpha. */
/************************************************************************/
SAGE2_C_API void sage2_div_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);

/************************************************************************/
/* sage2_sqrt_ps */
/* Compute y = sqrt(x). */
/************************************************************************/
SAGE2_C_API void sage2_sqrt_ps(uint64_t n, const float* x, float* y);

/************************************************************************/
/* sage2_max_ps */
/* Compute max of x. */
/************************************************************************/
SAGE2_C_API float sage2_max_ps(uint64_t n, const float* x);

/************************************************************************/
/* sage2_min_ps */
/* Compute min of x. */
/************************************************************************/
SAGE2_C_API float sage2_min_ps(uint64_t n, const float* x);

/************************************************************************/
/* sage2_sum_ps */
/* Compute sum of x. */
/************************************************************************/
SAGE2_C_API float sage2_sum_ps(uint64_t n, const float* x);

/************************************************************************/
/* sage2_dot_ps */
/* Compute dot product of x and y. */
/************************************************************************/
SAGE2_C_API float sage2_dot_ps(uint64_t n, const float* x, const float* y);

/************************************************************************/
/* sage2_nrm1_ps */
/* Compute l1 norm of x. */
/************************************************************************/
SAGE2_C_API float sage2_nrm1_ps(uint64_t n, const float* x);

/************************************************************************/
/* sage2_nrm2_ps */
/* Compute l2 norm of x. */
/************************************************************************/
SAGE2_C_API float sage2_nrm2_ps(uint64_t n, const float* x);

/************************************************************************/
/* sage2_euclidean_distance_ps */
/* Compute euclidean distance of x and y. */
/************************************************************************/
SAGE2_C_API float sage2_euclidean_distance_ps(uint64_t n, const float* x,
                                              const float* y);

/************************************************************************/
/* sage2_exp_ss */
/* sage2_exp_ss1 */
/* sage2_exp_ss2 */
/* Compute exp(x). */
/* sage2_exp_ps */
/* sage2_exp_ps1 */
/* sage2_exp_ps2 */
/* Compute y = exp(x). */
/************************************************************************/
/* Version 1's relative error is less than 1.23e-10. */
/* Version 2's relative error is less than 1.88e-04. */
/* Version 2 is much faster than version 1. */
/* Functions without suffix are the same as version 2. */
/************************************************************************/
SAGE2_C_API float sage2_exp_ss(float x);
SAGE2_C_API void sage2_exp_ps(uint64_t n, const float* x, float* y);
SAGE2_C_API float sage2_exp_ss1(float x);
SAGE2_C_API void sage2_exp_ps1(uint64_t n, const float* x, float* y);
SAGE2_C_API float sage2_exp_ss2(float x);
SAGE2_C_API void sage2_exp_ps2(uint64_t n, const float* x, float* y);

/************************************************************************/
/* sage2_log_ss */
/* Compute log(x). */
/* sage2_log_ps */
/* Compute y = log(x). */
/************************************************************************/
SAGE2_C_API float sage2_log_ss(float x);
SAGE2_C_API void sage2_log_ps(uint64_t n, const float* x, float* y);

/************************************************************************/
/* sage2_sigmoid_ss */
/* Compute sigmoid(x). */
/* sage2_sigmoid_ps */
/* Compute y = sigmoid(x). */
/************************************************************************/
SAGE2_C_API float sage2_sigmoid_ss(float x);
SAGE2_C_API void sage2_sigmoid_ps(uint64_t n, const float* x, float* y);

/************************************************************************/
/* sage2_tanh_ss */
/* Compute tanh(x). */
/* sage2_tanh_ps */
/* Compute y = tanh(x). */
/************************************************************************/
SAGE2_C_API float sage2_tanh_ss(float x);
SAGE2_C_API void sage2_tanh_ps(uint64_t n, const float* x, float* y);

/************************************************************************/
/* sage2_relu_ps */
/* Compute y = relu(x). */
/************************************************************************/
SAGE2_C_API void sage2_relu_ps(uint64_t n, const float* x, float* y);

#endif  // SAGE2_VMF_H_
