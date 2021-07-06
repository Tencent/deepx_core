// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_SGEMM_H_
#define SAGE2_SGEMM_H_

#include <sage2/macro.h>

/************************************************************************/
/* sage2_sgemm */
/* sage2_sgemm_jit_init */
/* They are the same as cblas_sgemm. */
/************************************************************************/
SAGE2_C_API void sage2_sgemm(int layout, int transX, int transY, int m, int n,
                             int k, float alpha, const float* X, int ldX,
                             const float* Y, int ldY, float beta, float* Z,
                             int ldZ);
// NOLINTNEXTLINE
typedef void (*sage2_sgemm_t)(void* jit, const float* X, const float* Y,
                              float* Z);
SAGE2_C_API void* sage2_sgemm_jit_init(int layout, int transX, int transY,
                                       int m, int n, int k, float alpha,
                                       int ldX, int ldY, float beta, int ldZ);
SAGE2_C_API sage2_sgemm_t sage2_sgemm_jit_get(void* jit);
SAGE2_C_API void sage2_sgemm_jit_uninit(void* jit);

#endif  // SAGE2_SGEMM_H_
