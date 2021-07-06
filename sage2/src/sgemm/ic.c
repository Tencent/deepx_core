// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

void sage2_sgemm_ic_ref(sage2_sgemm_ctx* ctx, const float* X, const float* Y,
                        float* Z) {
  int k = ctx->k;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  float sum = 0;
  int l;
  for (l = 0; l < k; ++l) {
    sum += X[l] * Y[l];
  }
  *Z = alpha * sum + beta * *Z;
}
