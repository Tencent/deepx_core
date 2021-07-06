// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "eigen_wrapper.h"
#include "sgemm/internal_sgemm.h"

namespace sage2 {
namespace {

void RNN(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_rm_t rm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, m, k, ldX, rm);
  auto Yv = make_eigen_matrixxf_view(Y, k, n, ldY, rm);
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, rm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

void RNT(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_rm_t rm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, m, k, ldX, rm);
  auto Yv = make_eigen_matrixxf_view(Y, n, k, ldY, rm).transpose();
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, rm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

void RTN(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_rm_t rm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, k, m, ldX, rm).transpose();
  auto Yv = make_eigen_matrixxf_view(Y, k, n, ldY, rm);
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, rm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

void RTT(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_rm_t rm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, k, m, ldX, rm).transpose();
  auto Yv = make_eigen_matrixxf_view(Y, n, k, ldY, rm).transpose();
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, rm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

void CNN(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_cm_t cm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, m, k, ldX, cm);
  auto Yv = make_eigen_matrixxf_view(Y, k, n, ldY, cm);
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, cm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

void CNT(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_cm_t cm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, m, k, ldX, cm);
  auto Yv = make_eigen_matrixxf_view(Y, n, k, ldY, cm).transpose();
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, cm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

void CTN(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_cm_t cm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, k, m, ldX, cm).transpose();
  auto Yv = make_eigen_matrixxf_view(Y, k, n, ldY, cm);
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, cm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

void CTT(sage2_sgemm_ctx* ctx, const float* X, const float* Y, float* Z) {
  eigen_matrix_cm_t cm;
  int m = ctx->m;
  int n = ctx->n;
  int k = ctx->k;
  int ldX = ctx->ldX;
  int ldY = ctx->ldY;
  int ldZ = ctx->ldZ;
  float alpha = ctx->alpha;
  float beta = ctx->beta;
  auto Xv = make_eigen_matrixxf_view(X, k, m, ldX, cm).transpose();
  auto Yv = make_eigen_matrixxf_view(Y, n, k, ldY, cm).transpose();
  auto Zv = make_eigen_matrixxf_view(Z, m, n, ldZ, cm);
  if (beta == 0) {
    if (alpha != 1) {
      Zv = alpha * Xv * Yv;
    } else {
      Zv = Xv * Yv;
    }
  } else if (beta == 1) {
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  } else {
    Zv *= beta;
    if (alpha != 1) {
      Zv += alpha * Xv * Yv;
    } else {
      Zv += Xv * Yv;
    }
  }
}

_sage2_sgemm_t ptr[8] = {&RNN, &RNT, &RTN, &RTT, &CNN, &CNT, &CTN, &CTT};

}  // namespace
}  // namespace sage2

void sage2_sgemm_eigen(int layout, int transX, int transY, int m, int n, int k,
                       float alpha, const float* X, int ldX, const float* Y,
                       int ldY, float beta, float* Z, int ldZ) {
  sage2_sgemm_ctx ctx;
  if (sage2_sgemm_init_ctx(&ctx, FLAGS_NONE, layout, transX, transY, m, n, k,
                           alpha, ldX, ldY, beta, ldZ) == 0) {
    int index = (ctx.layout << 2) | (ctx.transX << 1) | ctx.transY;
    sage2::ptr[index](&ctx, X, Y, Z);
  }
}
