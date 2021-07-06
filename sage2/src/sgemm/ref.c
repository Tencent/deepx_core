// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

static void _sage2_sgemm_ref(sage2_sgemm_ctx* ctx, const float* X,
                             const float* Y, float* Z) {
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  int X_inc_row = ctx->X_inc_row;
  int X_inc_col = ctx->X_inc_col;
  int Y_inc_row = ctx->Y_inc_row;
  int Y_inc_col = ctx->Y_inc_col;
  int Z_inc_row = ctx->Z_inc_row;
  int Z_inc_col = ctx->Z_inc_col;
  float z;
  int i, j, l;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < m; ++i) {
      z = 0;
      for (l = 0; l < k; ++l) {
        z +=
            X[i * X_inc_row + l * X_inc_col] * Y[l * Y_inc_row + j * Y_inc_col];
      }
      Z[i * Z_inc_row + j * Z_inc_col] =
          alpha * z + beta * Z[i * Z_inc_row + j * Z_inc_col];
    }
  }
}

void sage2_sgemm_ref(int layout, int transX, int transY, int m, int n, int k,
                     float alpha, const float* X, int ldX, const float* Y,
                     int ldY, float beta, float* Z, int ldZ) {
  sage2_sgemm_ctx ctx;
  if (sage2_sgemm_init_ctx(&ctx, FLAGS_NONE, layout, transX, transY, m, n, k,
                           alpha, ldX, ldY, beta, ldZ) == 0) {
    _sage2_sgemm_ref(&ctx, X, Y, Z);
  }
}
