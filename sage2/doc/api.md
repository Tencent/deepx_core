# API

[TOC]

本文档介绍sage2的主要头文件.

全部头文件参考["include/sage2"](../include/sage2).

## 半精度浮点数转换

头文件["sage2/half.h"](../include/sage2/half.h).

半精度浮点数和双精度/单精度浮点数的双向转换.

### 标量转换

```c
/************************************************************************/
/* double scalar <-> half scalar conversion */
/* single scalar <-> half scalar conversion */
/************************************************************************/
SAGE2_C_API sage2_half_t sage2_d2h(double d);
SAGE2_C_API double sage2_h2d(sage2_half_t h);
SAGE2_C_API sage2_half_t sage2_s2h(float s);
SAGE2_C_API float sage2_h2s(sage2_half_t h);
```

### 向量转换

```c
/************************************************************************/
/* double vector <-> half vector conversion */
/* single vector <-> half vector conversion */
/************************************************************************/
SAGE2_C_API void sage2_pd2ph(uint64_t n, const double* pd, sage2_half_t* ph);
SAGE2_C_API void sage2_ph2pd(uint64_t n, const sage2_half_t* ph, double* pd);
SAGE2_C_API void sage2_ps2ph(uint64_t n, const float* ps, sage2_half_t* ph);
SAGE2_C_API void sage2_ph2ps(uint64_t n, const sage2_half_t* ph, float* ps);
```

### 向量转换 for c++

```c++
#if defined __cplusplus
/************************************************************************/
/* double vector <-> half vector conversion for c++ */
/* single vector <-> half vector conversion for c++ */
/************************************************************************/
inline void sage2_half_convert(uint64_t n, const float* ps,
                               sage2_half_t* ph) SAGE2_NOEXCEPT {
  sage2_ps2ph(n, ps, ph);
}
inline void sage2_half_convert(uint64_t n, const sage2_half_t* ph,
                               float* ps) SAGE2_NOEXCEPT {
  sage2_ph2ps(n, ph, ps);
}
inline void sage2_half_convert(uint64_t n, const double* pd,
                               sage2_half_t* ph) SAGE2_NOEXCEPT {
  sage2_pd2ph(n, pd, ph);
}
inline void sage2_half_convert(uint64_t n, const sage2_half_t* ph,
                               double* pd) SAGE2_NOEXCEPT {
  sage2_ph2pd(n, ph, pd);
}
#endif
```

## memcpy

头文件["sage2/memcpy.h"](../include/sage2/memcpy.h).

```c
/************************************************************************/
/* sage2_memcpy */
/* They are the same as memcpy. */
/************************************************************************/
SAGE2_C_API void* sage2_memcpy(void* dst, const void* src, size_t n);
```

## 深度学习优化器

头文件["sage2/optimizer.h"](../include/sage2/optimizer.h).

头文件["sage2/tf\_optimizer.h"](../include/sage2/tf_optimizer.h).

### ada delta

```c
/************************************************************************/
/* ada delta */
/************************************************************************/
typedef struct _sage2_ada_delta_config_s {
  float rho;          // constant
  float alpha;        // constant
  float beta;         // constant
  float one_sub_rho;  // internal
} sage2_ada_delta_config_s;
// Get a default ada delta config.
SAGE2_C_API void sage2_ada_delta_config_s_default(
    sage2_ada_delta_config_s* config);
// Initialize the ada delta config.
SAGE2_C_API void sage2_ada_delta_config_s_init(
    sage2_ada_delta_config_s* config);
// Perform an ada delta update on scalar.
SAGE2_C_API void sage2_ada_delta_update_ss(
    const sage2_ada_delta_config_s* config, float g, float* w, float* n,
    float* deltaw);
// Perform an ada delta update on vector.
SAGE2_C_API void sage2_ada_delta_update_ps(
    const sage2_ada_delta_config_s* config, uint64_t _n, const float* g,
    float* w, float* n, float* deltaw);
```

### ada grad

```c
/************************************************************************/
/* ada grad */
/************************************************************************/
typedef struct _sage2_ada_grad_config_s {
  float alpha;  // constant
  float beta;   // constant
} sage2_ada_grad_config_s;
// Get a default ada grad config.
SAGE2_C_API void sage2_ada_grad_config_s_default(
    sage2_ada_grad_config_s* config);
// Perform an ada grad update on scalar.
SAGE2_C_API void sage2_ada_grad_update_ss(const sage2_ada_grad_config_s* config,
                                          float g, float* w, float* n);
// Perform an ada grad update on vector.
SAGE2_C_API void sage2_ada_grad_update_ps(const sage2_ada_grad_config_s* config,
                                          uint64_t _n, const float* g, float* w,
                                          float* n);
```

### adam

```c
/************************************************************************/
/* adam */
/************************************************************************/
typedef struct _sage2_adam_config_s {
  float rho1;          // constant
  float rho2;          // constant
  float alpha;         // constant
  float beta;          // constant
  float rho1t;         // mutable
  float rho2t;         // mutable
  float one_sub_rho1;  // internal
  float one_sub_rho2;  // internal
  float rho_aux;       // internal
} sage2_adam_config_s;
// Get a default adam config.
SAGE2_C_API void sage2_adam_config_s_default(sage2_adam_config_s* config);
// Initialize the adam config.
SAGE2_C_API void sage2_adam_config_s_init(sage2_adam_config_s* config);
// Pre-batch adjust the adam config.
SAGE2_C_API void sage2_adam_config_s_prebatch(sage2_adam_config_s* config);
// Perform an adam update on scalar.
SAGE2_C_API void sage2_adam_update_ss(const sage2_adam_config_s* config,
                                      float g, float* w, float* m, float* v);
// Perform an adam update on vector.
SAGE2_C_API void sage2_adam_update_ps(const sage2_adam_config_s* config,
                                      uint64_t n, const float* g, float* w,
                                      float* m, float* v);
```

### ftrl

```c
/************************************************************************/
/* ftrl */
/************************************************************************/
typedef struct _sage2_ftrl_config_s {
  float alpha;      // constant
  float beta;       // constant
  float l1;         // constant
  float l2;         // constant
  float inv_alpha;  // internal
} sage2_ftrl_config_s;
// Get a default ftrl config.
SAGE2_C_API void sage2_ftrl_config_s_default(sage2_ftrl_config_s* config);
// Initialize the ftrl config.
SAGE2_C_API void sage2_ftrl_config_s_init(sage2_ftrl_config_s* config);
// Perform a ftrl update on scalar.
SAGE2_C_API void sage2_ftrl_update_ss(const sage2_ftrl_config_s* config,
                                      float g, float* w, float* n, float* z);
// Perform a ftrl update on vector.
SAGE2_C_API void sage2_ftrl_update_ps(const sage2_ftrl_config_s* config,
                                      uint64_t _n, const float* g, float* w,
                                      float* n, float* z);
```

### gftrl

```c
/************************************************************************/
/* gftrl */
/************************************************************************/
typedef struct _sage2_gftrl_config_s {
  float alpha;      // constant
  float beta;       // constant
  float lambda;     // constant
  float inv_alpha;  // internal
} sage2_gftrl_config_s;
// Get a default gftrl config.
SAGE2_C_API void sage2_gftrl_config_s_default(sage2_gftrl_config_s* config);
// Initialize the gftrl config.
SAGE2_C_API void sage2_gftrl_config_s_init(sage2_gftrl_config_s* config);
// Perform a gftrl update on scalar.
SAGE2_C_API void sage2_gftrl_update_ss(const sage2_gftrl_config_s* config,
                                       float g, float* w, float* n, float* z);
// Perform a gftrl update on vector.
SAGE2_C_API void sage2_gftrl_update_ps(const sage2_gftrl_config_s* config,
                                       uint64_t _n, const float* g, float* w,
                                       float* n, float* z);
```

### momentum

```c
/************************************************************************/
/* momentum */
/************************************************************************/
typedef struct _sage2_momentum_config_s {
  float rho;    // constant
  float alpha;  // constant
} sage2_momentum_config_s;
// Get a default momentum config.
SAGE2_C_API void sage2_momentum_config_s_default(
    sage2_momentum_config_s* config);
// Perform a momentum update on scalar.
SAGE2_C_API void sage2_momentum_update_ss(const sage2_momentum_config_s* config,
                                          float g, float* w, float* v);
// Perform a momentum update on vector.
SAGE2_C_API void sage2_momentum_update_ps(const sage2_momentum_config_s* config,
                                          uint64_t _n, const float* g, float* w,
                                          float* v);
```

### rms prop

```c
/************************************************************************/
/* rms prop */
/************************************************************************/
typedef struct _sage2_rms_prop_config_s {
  float rho;          // constant
  float alpha;        // constant
  float beta;         // constant
  float one_sub_rho;  // internal
} sage2_rms_prop_config_s;
// Get a default rms prop config.
SAGE2_C_API void sage2_rms_prop_config_s_default(
    sage2_rms_prop_config_s* config);
// Initialize the rms prop config.
SAGE2_C_API void sage2_rms_prop_config_s_init(sage2_rms_prop_config_s* config);
// Perform a rms prop update on scalar.
SAGE2_C_API void sage2_rms_prop_update_ss(const sage2_rms_prop_config_s* config,
                                          float g, float* w, float* v);
// Perform a rms prop update on vector.
SAGE2_C_API void sage2_rms_prop_update_ps(const sage2_rms_prop_config_s* config,
                                          uint64_t _n, const float* g, float* w,
                                          float* v);
```

### 梯度裁剪

```c
/************************************************************************/
/* grad_clip */
/************************************************************************/
// Clip the grad into range [-20, 20].
SAGE2_C_API void sage2_grad_clip_20_ss(float* g);
SAGE2_C_API void sage2_grad_clip_20_ps(uint64_t n, float* g);
```

### TensorFlow ada grad v1

```c
/************************************************************************/
/* TensorFlow ada grad v1 */
/* It can accelerate tensorflow::functor::ApplyAdagrad */
/* when update_slots is true. */
/************************************************************************/
typedef struct _sage2_tf_ada_grad_v1_config_s {
  float alpha;  // constant
} sage2_tf_ada_grad_v1_config_s;
SAGE2_C_API void sage2_tf_ada_grad_v1_update_ps(
    const sage2_tf_ada_grad_v1_config_s* config, uint64_t _n, const float* g,
    float* w, float* n);
```

### TensorFlow ada grad v2

```c
/************************************************************************/
/* TensorFlow ada grad v2 */
/* It can accelerate tensorflow::functor::ApplyAdagradV2 */
/* when update_slots is true. */
/************************************************************************/
#define sage2_tf_ada_grad_v2_config_s sage2_ada_grad_config_s
#define sage2_tf_ada_grad_v2_update_ps sage2_ada_grad_update_ps
```

### TensorFlow adam

```c
/************************************************************************/
/* TensorFlow adam */
/* It can accelerate tensorflow::functor::ApplyAdam */
/************************************************************************/
#define sage2_tf_adam_config_s sage2_adam_config_s
#define sage2_tf_adam_update_ps sage2_adam_update_ps
```

### TensorFlow ftrl v1

```c
/************************************************************************/
/* TensorFlow ftrl v1 */
/* It can accelerate tensorflow::functor::ApplyFtrl */
/* when learning_rate_power is -0.5. */
/************************************************************************/
typedef struct _sage2_tf_ftrl_v1_config_s {
  float alpha;
  float l1;
  float l2;
  float inv_alpha;
} sage2_tf_ftrl_v1_config_s;
SAGE2_C_API void sage2_tf_ftrl_v1_update_ps(
    const sage2_tf_ftrl_v1_config_s* config, uint64_t _n, const float* g,
    float* w, float* n, float* z);
```

## 单精度浮点数矩阵乘法

头文件["sage2/sgemm.h"](../include/sage2/sgemm.h).

```c
/************************************************************************/
/* sage2_sgemm */
/* sage2_sgemm_jit_init */
/* They are the same as cblas_sgemm. */
/************************************************************************/
SAGE2_C_API void sage2_sgemm(int layout, int transX, int transY, int m, int n,
                             int k, float alpha, const float* X, int ldX,
                             const float* Y, int ldY, float beta, float* Z,
                             int ldZ);
typedef void (*sage2_sgemm_t)(void* jit, const float* X, const float* Y,
                              float* Z);
SAGE2_C_API void* sage2_sgemm_jit_init(int layout, int transX, int transY,
                                       int m, int n, int k, float alpha,
                                       int ldX, int ldY, float beta, int ldZ);
SAGE2_C_API sage2_sgemm_t sage2_sgemm_jit_get(void* jit);
SAGE2_C_API void sage2_sgemm_jit_uninit(void* jit);
```

## 向量数学函数

头文件["sage2/vmf.h"](../include/sage2/vmf.h).

### y = alpha(标量) * x + y

```c
/************************************************************************/
/* sage2_axpy_ps */
/* Compute y = alpha * x + y. */
/************************************************************************/
SAGE2_C_API void sage2_axpy_ps(uint64_t n, float alpha, const float* x,
                               float* y);
```

### y = alpha(标量) * x + beta(标量) * y

```c
/************************************************************************/
/* sage2_axpby_ps */
/* Compute y = alpha * x + beta * y. */
/************************************************************************/
SAGE2_C_API void sage2_axpby_ps(uint64_t n, float alpha, const float* x,
                                float beta, float* y);
```

### 向量加向量

```c
/************************************************************************/
/* sage2_add_ps */
/* Compute z = x + y. */
/************************************************************************/
SAGE2_C_API void sage2_add_ps(uint64_t n, const float* x, const float* y,
                              float* z);
```

### 向量减向量

```c
/************************************************************************/
/* sage2_sub_ps */
/* Compute z = x - y. */
/************************************************************************/
SAGE2_C_API void sage2_sub_ps(uint64_t n, const float* x, const float* y,
                              float* z);
```

### 向量乘向量

```c
/************************************************************************/
/* sage2_mul_ps */
/* Compute z = x * y. */
/************************************************************************/
SAGE2_C_API void sage2_mul_ps(uint64_t n, const float* x, const float* y,
                              float* z);
```

### 向量除向量

```c
/************************************************************************/
/* sage2_div_ps */
/* Compute z = x / y. */
/************************************************************************/
SAGE2_C_API void sage2_div_ps(uint64_t n, const float* x, const float* y,
                              float* z);
```

### 向量加标量

```c
/************************************************************************/
/* sage2_add_scalar_ps */
/* Compute y = x + alpha. */
/************************************************************************/
SAGE2_C_API void sage2_add_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);
```

### 向量减标量

```c
/************************************************************************/
/* sage2_sub_scalar_ps */
/* Compute y = x - alpha. */
/************************************************************************/
SAGE2_C_API void sage2_sub_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);
```

### 向量乘标量

```c
/************************************************************************/
/* sage2_mul_scalar_ps */
/* Compute y = x * alpha. */
/************************************************************************/
SAGE2_C_API void sage2_mul_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);
```

### 向量除标量

```c
/************************************************************************/
/* sage2_div_scalar_ps */
/* Compute y = x / alpha. */
/************************************************************************/
SAGE2_C_API void sage2_div_scalar_ps(uint64_t n, const float* x, float alpha,
                                     float* y);
```

### sqrt

```c
/************************************************************************/
/* sage2_sqrt_ps */
/* Compute y = sqrt(x). */
/************************************************************************/
SAGE2_C_API void sage2_sqrt_ps(uint64_t n, const float* x, float* y);
```

### 最大值

```c
/************************************************************************/
/* sage2_max_ps */
/* Compute max of x. */
/************************************************************************/
SAGE2_C_API float sage2_max_ps(uint64_t n, const float* x);
```

### 最小值

```c
/************************************************************************/
/* sage2_min_ps */
/* Compute min of x. */
/************************************************************************/
SAGE2_C_API float sage2_min_ps(uint64_t n, const float* x);
```

### 和

```c
/************************************************************************/
/* sage2_sum_ps */
/* Compute sum of x. */
/************************************************************************/
SAGE2_C_API float sage2_sum_ps(uint64_t n, const float* x);
```

### 点积

```c
/************************************************************************/
/* sage2_dot_ps */
/* Compute dot product of x and y. */
/************************************************************************/
SAGE2_C_API float sage2_dot_ps(uint64_t n, const float* x, const float* y);
```

### l1范数

```c
/************************************************************************/
/* sage2_nrm1_ps */
/* Compute l1 norm of x. */
/************************************************************************/
SAGE2_C_API float sage2_nrm1_ps(uint64_t n, const float* x);
```

### l2范数

```c
/************************************************************************/
/* sage2_nrm2_ps */
/* Compute l2 norm of x. */
/************************************************************************/
SAGE2_C_API float sage2_nrm2_ps(uint64_t n, const float* x);
```

### 欧式距离

```c
/************************************************************************/
/* sage2_euclidean_distance_ps */
/* Compute euclidean distance of x and y. */
/************************************************************************/
SAGE2_C_API float sage2_euclidean_distance_ps(uint64_t n, const float* x,
                                              const float* y);
```

### exp

```c
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
```

### log

```c
/************************************************************************/
/* sage2_log_ss */
/* Compute log(x). */
/* sage2_log_ps */
/* Compute y = log(x). */
/************************************************************************/
SAGE2_C_API float sage2_log_ss(float x);
SAGE2_C_API void sage2_log_ps(uint64_t n, const float* x, float* y);
```

### sigmoid

```c
/************************************************************************/
/* sage2_sigmoid_ss */
/* Compute sigmoid(x). */
/* sage2_sigmoid_ps */
/* Compute y = sigmoid(x). */
/************************************************************************/
SAGE2_C_API float sage2_sigmoid_ss(float x);
SAGE2_C_API void sage2_sigmoid_ps(uint64_t n, const float* x, float* y);
```

### tanh

```c
/************************************************************************/
/* sage2_tanh_ss */
/* Compute tanh(x). */
/* sage2_tanh_ps */
/* Compute y = tanh(x). */
/************************************************************************/
SAGE2_C_API float sage2_tanh_ss(float x);
SAGE2_C_API void sage2_tanh_ps(uint64_t n, const float* x, float* y);
```

### relu

```c
/************************************************************************/
/* sage2_relu_ps */
/* Compute y = relu(x). */
/************************************************************************/
SAGE2_C_API void sage2_relu_ps(uint64_t n, const float* x, float* y);
```
