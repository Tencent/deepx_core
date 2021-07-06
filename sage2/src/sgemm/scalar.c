// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

void sage2_sgemm_scalar(sage2_sgemm_ctx* ctx, const float* X, const float* Y,
                        float* Z) {
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  *Z = alpha * *X * *Y + beta * *Z;
}
