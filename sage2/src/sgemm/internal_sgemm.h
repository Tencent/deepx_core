// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#ifndef SAGE2_SGEMM_INTERNAL_SGEMM_H_
#define SAGE2_SGEMM_INTERNAL_SGEMM_H_

#include <sage2/cblas_adapter.h>
#include <sage2/cpuid.h>
#include <sage2/sgemm.h>
#include <stdio.h>   // NOLINT
#include <stdlib.h>  // NOLINT
#include "internal_macro.h"
#include "sgemm/offset.h"
#if defined __cplusplus
#include <memory>
#include <new>
#include "xbyak_wrapper.h"
#endif

enum {
  FLAGS_NONE = 0,
  FLAGS_JIT = 2,
  FLAGS_USE_TLS = 4,
};

enum {
  METHOD_NONE = 0,
  METHOD_A0_B0,
  METHOD_A0_B0_ZCONT,
  METHOD_A0_B0_ZCONT_JIT,
  METHOD_A0_B1,
  METHOD_A0,
  METHOD_A0_ZCONT,
  METHOD_A0_ZCONT_JIT,
  METHOD_SCALAR,
  METHOD_SCALAR_JIT,
  METHOD_IC,
  METHOD_IC_JIT,
  METHOD_OC_ZRM,
  METHOD_OC_ZRM_JIT,
  METHOD_OC_ZCM,
  METHOD_OC_ZCM_JIT,
  METHOD_ULM,
};

struct _sage2_sgemm_ctx;
// NOLINTNEXTLINE
typedef struct _sage2_sgemm_ctx sage2_sgemm_ctx;
// NOLINTNEXTLINE
typedef void (*_sage2_sgemm_t)(sage2_sgemm_ctx* ctx, const float* X,
                               const float* Y, float* Z);
struct _sage2_sgemm_ctx {
  int flags;
  int layout;
  int transX;
  int transY;
  int m;
  int n;
  int k;
  int ldX;
  int ldY;
  int ldZ;
  float alpha;
  float beta;
  int X_inc_row;
  int X_inc_col;
  int Y_inc_row;
  int Y_inc_col;
  int Z_inc_row;
  int Z_inc_col;
  int Z_row_major;
  int Z_cont;
  _sage2_sgemm_t func;
  int method;
  void* method_data;

  // only for METHOD_ULM
  struct _ulm {
    int mb;
    int nb;
    int kb;
    int mc;
    int nc;
    int kc;
    float* packed_X;
    float* packed_Y;
    float* packed_Z;
    void (*pack_X)(int mc, int kc, const float* X, float* packed_X,
                   int X_inc_row, int X_inc_col);
    void (*pack_Y)(int kc, int nc, const float* Y, float* packed_X,
                   int Y_inc_row, int Y_inc_col);
    void (*micro_kernel)(int kc, float alpha, const float* X, const float* Y,
                         float beta, float* Z, int Z_inc_row, int Z_inc_col);
    int method;
  } ulm;
};

PRIVATE_C_FUNC void sage2_sgemm_ref(int layout, int transX, int transY, int m,
                                    int n, int k, float alpha, const float* X,
                                    int ldX, const float* Y, int ldY,
                                    float beta, float* Z, int ldZ);
PRIVATE_C_FUNC void sage2_sgemm_eigen(int layout, int transX, int transY, int m,
                                      int n, int k, float alpha, const float* X,
                                      int ldX, const float* Y, int ldY,
                                      float beta, float* Z, int ldZ);

PRIVATE_C_FUNC int sage2_sgemm_init_ctx(sage2_sgemm_ctx* ctx, int flags,
                                        int layout, int transX, int transY,
                                        int m, int n, int k, float alpha,
                                        int ldX, int ldY, float beta, int ldZ);

PRIVATE_C_FUNC void sage2_sgemm_a0_b0(sage2_sgemm_ctx* ctx, const float* X,
                                      const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_a0_b0_Zc(sage2_sgemm_ctx* ctx, const float* X,
                                         const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_a0_b0_Zc_ref(sage2_sgemm_ctx* ctx,
                                             const float* X, const float* Y,
                                             float* Z);
PRIVATE_C_FUNC int sage2_sgemm_a0_b0_Zc_jit_init(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_a0_b0_Zc_jit_uninit(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_a0_b1(sage2_sgemm_ctx* ctx, const float* X,
                                      const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_a0(sage2_sgemm_ctx* ctx, const float* X,
                                   const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_a0_Zc(sage2_sgemm_ctx* ctx, const float* X,
                                      const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_a0_Zc_ref(sage2_sgemm_ctx* ctx, const float* X,
                                          const float* Y, float* Z);
PRIVATE_C_FUNC int sage2_sgemm_a0_Zc_jit_init(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_a0_Zc_jit_uninit(sage2_sgemm_ctx* ctx);

PRIVATE_C_FUNC void sage2_sgemm_scalar(sage2_sgemm_ctx* ctx, const float* X,
                                       const float* Y, float* Z);
PRIVATE_C_FUNC int sage2_sgemm_scalar_jit_init(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_scalar_jit_uninit(sage2_sgemm_ctx* ctx);

PRIVATE_C_FUNC void sage2_sgemm_ic(sage2_sgemm_ctx* ctx, const float* X,
                                   const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_ic_ref(sage2_sgemm_ctx* ctx, const float* X,
                                       const float* Y, float* Z);
PRIVATE_C_FUNC int sage2_sgemm_ic_jit_init(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_ic_jit_uninit(sage2_sgemm_ctx* ctx);

PRIVATE_C_FUNC void sage2_sgemm_oc_Zrm(sage2_sgemm_ctx* ctx, const float* X,
                                       const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_oc_Zrm_ref(sage2_sgemm_ctx* ctx, const float* X,
                                           const float* Y, float* Z);
PRIVATE_C_FUNC int sage2_sgemm_oc_Zrm_jit_init(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_oc_Zrm_jit_uninit(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_oc_Zcm(sage2_sgemm_ctx* ctx, const float* X,
                                       const float* Y, float* Z);
PRIVATE_C_FUNC void sage2_sgemm_oc_Zcm_ref(sage2_sgemm_ctx* ctx, const float* X,
                                           const float* Y, float* Z);
PRIVATE_C_FUNC int sage2_sgemm_oc_Zcm_jit_init(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_oc_Zcm_jit_uninit(sage2_sgemm_ctx* ctx);

PRIVATE_C_FUNC int sage2_sgemm_ulm_init(sage2_sgemm_ctx* ctx);
PRIVATE_C_FUNC void sage2_sgemm_ulm_uninit(sage2_sgemm_ctx* ctx);

#endif  // SAGE2_SGEMM_INTERNAL_SGEMM_H_
