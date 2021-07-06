// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

void sage2_sgemm_a0_b0(sage2_sgemm_ctx* ctx, const float* X, const float* Y,
                       float* Z) {
  int m = ctx->m;
  int n = ctx->n;
  int Z_inc_row = ctx->Z_inc_row;
  int Z_inc_col = ctx->Z_inc_col;
  int i, j;
  (void)X;
  (void)Y;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < m; ++i) {
      Z[i * Z_inc_row + j * Z_inc_col] = 0;
    }
  }
}

void sage2_sgemm_a0_b0_Zc_ref(sage2_sgemm_ctx* ctx, const float* X,
                              const float* Y, float* Z) {
  int n = ctx->m * ctx->n;
  int i;
  (void)X;
  (void)Y;
  for (i = 0; i < n; ++i) {
    Z[i] = 0;
  }
}

void sage2_sgemm_a0_b1(sage2_sgemm_ctx* ctx, const float* X, const float* Y,
                       float* Z) {
  (void)ctx;
  (void)X;
  (void)Y;
  (void)Z;
}

void sage2_sgemm_a0(sage2_sgemm_ctx* ctx, const float* X, const float* Y,
                    float* Z) {
  int m = ctx->m;
  int n = ctx->n;
  float beta = ctx->beta;
  int Z_inc_row = ctx->Z_inc_row;
  int Z_inc_col = ctx->Z_inc_col;
  int i, j;
  (void)X;
  (void)Y;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < m; ++i) {
      Z[i * Z_inc_row + j * Z_inc_col] *= beta;
    }
  }
}

void sage2_sgemm_a0_Zc_ref(sage2_sgemm_ctx* ctx, const float* X, const float* Y,
                           float* Z) {
  int n = ctx->m * ctx->n;
  float beta = ctx->beta;
  int i;
  (void)X;
  (void)Y;
  for (i = 0; i < n; ++i) {
    Z[i] *= beta;
  }
}
