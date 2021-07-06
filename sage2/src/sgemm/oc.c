// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

void sage2_sgemm_oc_Zrm_ref(sage2_sgemm_ctx* ctx, const float* X,
                            const float* Y, float* Z) {
  int m = ctx->m;
  int n = ctx->n;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  int Z_inc_row = ctx->Z_inc_row;
  float x;
  int i, j;
  for (i = 0; i < m; ++i) {
    x = alpha * X[i];
    for (j = 0; j < n; ++j) {
      Z[j] = x * Y[j] + beta * Z[j];
    }
    Z += Z_inc_row;
  }
}

void sage2_sgemm_oc_Zcm_ref(sage2_sgemm_ctx* ctx, const float* X,
                            const float* Y, float* Z) {
  int m = ctx->m;
  int n = ctx->n;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  int Z_inc_col = ctx->Z_inc_col;
  float y;
  int i, j;
  for (j = 0; j < n; ++j) {
    y = alpha * Y[j];
    for (i = 0; i < m; ++i) {
      Z[i] = X[i] * y + beta * Z[i];
    }
    Z += Z_inc_col;
  }
}
