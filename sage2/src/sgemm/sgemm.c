// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

int sage2_sgemm_init_ctx(sage2_sgemm_ctx* ctx, int flags, int layout,
                         int transX, int transY, int m, int n, int k,
                         float alpha, int ldX, int ldY, float beta, int ldZ) {
  ctx->method = METHOD_NONE;

  ctx->flags = flags;

  if (!(layout == 101 || layout == 102)) {
    fprintf(stderr, "Invalid layout: %d.\n", layout);
    return -1;
  }
  layout -= 101;
  ctx->layout = layout;

  if (!(transX == 111 || transX == 112)) {
    fprintf(stderr, "Invalid transX: %d.\n", transX);
    return -1;
  }
  transX -= 111;
  ctx->transX = transX;

  if (!(transY == 111 || transY == 112)) {
    fprintf(stderr, "Invalid transY: %d.\n", transY);
    return -1;
  }
  transY -= 111;
  ctx->transY = transY;

  if (m <= 0) {
    fprintf(stderr, "Invalid m: %d.\n", m);
    return -1;
  }
  ctx->m = m;

  if (n <= 0) {
    fprintf(stderr, "Invalid n: %d.\n", n);
    return -1;
  }
  ctx->n = n;

  if (k <= 0) {
    fprintf(stderr, "Invalid k: %d.\n", k);
    return -1;
  }
  ctx->k = k;

  ctx->ldX = ldX;
  ctx->ldY = ldY;
  ctx->ldZ = ldZ;

  ctx->alpha = alpha;
  ctx->beta = beta;

  if ((!layout && !transX) || (layout && transX)) {
    if (ldX < k) {
      fprintf(stderr, "Invalid ldX: %d.\n", ldX);
      return -1;
    }
    ctx->X_inc_row = ldX;
    ctx->X_inc_col = 1;
  } else {
    if (ldX < m) {
      fprintf(stderr, "Invalid ldX: %d.\n", ldX);
      return -1;
    }
    ctx->X_inc_row = 1;
    ctx->X_inc_col = ldX;
  }

  if ((!layout && !transY) || (layout && transY)) {
    if (ldY < n) {
      fprintf(stderr, "Invalid ldY: %d.\n", ldY);
      return -1;
    }
    ctx->Y_inc_row = ldY;
    ctx->Y_inc_col = 1;
  } else {
    if (ldY < k) {
      fprintf(stderr, "Invalid ldY: %d.\n", ldY);
      return -1;
    }
    ctx->Y_inc_row = 1;
    ctx->Y_inc_col = ldY;
  }

  if (!layout) {
    if (ldZ < n) {
      fprintf(stderr, "Invalid ldZ: %d.\n", ldZ);
      return -1;
    }
    ctx->Z_inc_row = ldZ;
    ctx->Z_inc_col = 1;
    ctx->Z_row_major = 1;
    ctx->Z_cont = (ldZ == n) ? 1 : 0;
  } else {
    if (ldZ < m) {
      fprintf(stderr, "Invalid ldZ: %d.\n", ldZ);
      return -1;
    }
    ctx->Z_inc_row = 1;
    ctx->Z_inc_col = ldZ;
    ctx->Z_row_major = 0;
    ctx->Z_cont = (ldZ == m) ? 1 : 0;
  }
  return 0;
}

static int sage2_sgemm_init(sage2_sgemm_ctx* ctx, int flags, int layout,
                            int transX, int transY, int m, int n, int k,
                            float alpha, int ldX, int ldY, float beta,
                            int ldZ) {
  if (sage2_sgemm_init_ctx(ctx, flags, layout, transX, transY, m, n, k, alpha,
                           ldX, ldY, beta, ldZ) != 0) {
    return -1;
  }

  if (alpha == 0) {
    if (beta == 0) {
      if (ctx->Z_cont) {
        if (flags & FLAGS_JIT) {
          return sage2_sgemm_a0_b0_Zc_jit_init(ctx);
        } else {
          ctx->func = &sage2_sgemm_a0_b0_Zc;
          ctx->method = METHOD_A0_B0_ZCONT;
          return 0;
        }
      } else {
        ctx->func = &sage2_sgemm_a0_b0;
        ctx->method = METHOD_A0_B0;
        return 0;
      }
    } else if (beta == 1) {
      ctx->func = &sage2_sgemm_a0_b1;
      ctx->method = METHOD_A0_B1;
      return 0;
    } else {
      if (ctx->Z_cont) {
        if (flags & FLAGS_JIT) {
          return sage2_sgemm_a0_Zc_jit_init(ctx);
        } else {
          ctx->func = &sage2_sgemm_a0_Zc;
          ctx->method = METHOD_A0_ZCONT;
          return 0;
        }
      } else {
        ctx->func = &sage2_sgemm_a0;
        ctx->method = METHOD_A0;
        return 0;
      }
    }
  }

  if (m == 1 && n == 1) {
    // inner product
    if (k == 1) {
      // X, Y and Z are all scalars.
      if (flags & FLAGS_JIT) {
        return sage2_sgemm_scalar_jit_init(ctx);
      } else {
        ctx->func = &sage2_sgemm_scalar;
        ctx->method = METHOD_SCALAR;
        return 0;
      }
    }

    if (ctx->X_inc_col == 1 && ctx->Y_inc_row == 1) {
      // X and Y are all continuous.
      if (flags & FLAGS_JIT) {
        return sage2_sgemm_ic_jit_init(ctx);
      } else {
        ctx->func = &sage2_sgemm_ic;
        ctx->method = METHOD_IC;
        return 0;
      }
    }
  }

  if (k == 1 && ctx->X_inc_row == 1 && ctx->Y_inc_col == 1) {
    // outer product
    // X and Y are all continuous.
    if (ctx->Z_row_major) {
      if (flags & FLAGS_JIT) {
        return sage2_sgemm_oc_Zrm_jit_init(ctx);
      } else {
        ctx->func = &sage2_sgemm_oc_Zrm;
        ctx->method = METHOD_OC_ZRM;
        return 0;
      }
    } else {
      if (flags & FLAGS_JIT) {
        return sage2_sgemm_oc_Zcm_jit_init(ctx);
      } else {
        ctx->func = &sage2_sgemm_oc_Zcm;
        ctx->method = METHOD_OC_ZCM;
        return 0;
      }
    }
  }

  return sage2_sgemm_ulm_init(ctx);
}

static void sage2_sgemm_uninit(sage2_sgemm_ctx* ctx) {
  switch (ctx->method) {
    case METHOD_A0_B0_ZCONT_JIT:
      sage2_sgemm_a0_b0_Zc_jit_uninit(ctx);
      break;
    case METHOD_A0_ZCONT_JIT:
      sage2_sgemm_a0_Zc_jit_uninit(ctx);
      break;
    case METHOD_SCALAR_JIT:
      sage2_sgemm_scalar_jit_uninit(ctx);
      break;
    case METHOD_IC_JIT:
      sage2_sgemm_ic_jit_uninit(ctx);
      break;
    case METHOD_OC_ZRM_JIT:
      sage2_sgemm_oc_Zrm_jit_uninit(ctx);
      break;
    case METHOD_OC_ZCM_JIT:
      sage2_sgemm_oc_Zcm_jit_uninit(ctx);
      break;
    case METHOD_ULM:
      sage2_sgemm_ulm_uninit(ctx);
      break;
  }
}

static const sage2_cblas* mkl = NULL;
static void (*sgemm)(int layout, int transX, int transY, int m, int n, int k,
                     float alpha, const float* X, int ldX, const float* Y,
                     int ldY, float beta, float* Z, int ldZ);
static void* (*sgemm_jit_init)(int layout, int transX, int transY, int m, int n,
                               int k, float alpha, int ldX, int ldY, float beta,
                               int ldZ);
static sage2_sgemm_t (*sgemm_jit_get)(void* jit);
static void (*sgemm_jit_uninit)(void* jit);

static void* sage2_sgemm_jit_init_mkl(int layout, int transX, int transY, int m,
                                      int n, int k, float alpha, int ldX,
                                      int ldY, float beta, int ldZ) {
  void* jit = NULL;
  if (mkl->mkl_cblas_jit_create_sgemm(&jit, layout, transX, transY, m, n, k,
                                      alpha, ldX, ldY, beta, ldZ) == 2) {
    fprintf(stderr, "Failed to mkl_cblas_jit_create_sgemm.\n");
    return NULL;
  }
  return jit;
}

static sage2_sgemm_t sage2_sgemm_jit_get_mkl(void* jit) {
  return (sage2_sgemm_t)mkl->mkl_jit_get_sgemm_ptr(jit);
}

static void sage2_sgemm_jit_uninit_mkl(void* jit) {
  (void)mkl->mkl_jit_destroy(jit);
}

static void sage2_sgemm_default(int layout, int transX, int transY, int m,
                                int n, int k, float alpha, const float* X,
                                int ldX, const float* Y, int ldY, float beta,
                                float* Z, int ldZ) {
  sage2_sgemm_ctx ctx;
  if (sage2_sgemm_init(&ctx, FLAGS_USE_TLS, layout, transX, transY, m, n, k,
                       alpha, ldX, ldY, beta, ldZ) == 0) {
    ctx.func(&ctx, X, Y, Z);
    sage2_sgemm_uninit(&ctx);
  }
}

static void* sage2_sgemm_jit_init_default(int layout, int transX, int transY,
                                          int m, int n, int k, float alpha,
                                          int ldX, int ldY, float beta,
                                          int ldZ) {
  sage2_sgemm_ctx* ctx = malloc(sizeof(sage2_sgemm_ctx));
  if (ctx == NULL) {
    fprintf(stderr, "Failed to alloc memory.\n");
    return NULL;
  }
  if (sage2_sgemm_init(ctx, FLAGS_JIT, layout, transX, transY, m, n, k, alpha,
                       ldX, ldY, beta, ldZ) != 0) {
    free(ctx);
    return NULL;
  }
  return ctx;
}

static sage2_sgemm_t sage2_sgemm_jit_get_default(void* jit) {
  sage2_sgemm_ctx* ctx = jit;
  return (sage2_sgemm_t)ctx->func;
}

static void sage2_sgemm_jit_uninit_default(void* jit) {
  sage2_sgemm_ctx* ctx = jit;
  sage2_sgemm_uninit(ctx);
  free(ctx);
}

ATTR_CTOR(110) static void init() {
  // 'MKL_DEBUG_CPU_TYPE' is an undocumented environment variable.
  //
  // Reference:
  // https://en.wikipedia.org/wiki/Math_Kernel_Library
  const char* MKL_DEBUG_CPU_TYPE = getenv("MKL_DEBUG_CPU_TYPE");
  if ((sage2_cpu_vendor_type == SAGE2_CPU_VENDOR_TYPE_INTEL) ||
      (sage2_cpu_vendor_type != SAGE2_CPU_VENDOR_TYPE_INTEL &&
       MKL_DEBUG_CPU_TYPE)) {
    mkl = sage2_cblas_mkl();
    if (mkl->cblas_sgemm && mkl->mkl_cblas_jit_create_sgemm &&
        mkl->mkl_jit_get_sgemm_ptr && mkl->mkl_jit_destroy) {
      if (sage2_cpu_vendor_type == SAGE2_CPU_VENDOR_TYPE_INTEL) {
        fprintf(stderr, "sage2_sgemm is using MKL kernel.\n");
      } else {
        fprintf(stderr,
                "sage2_sgemm is using MKL kernel, MKL_DEBUG_CPU_TYPE=%s.\n",
                MKL_DEBUG_CPU_TYPE);
      }
      sgemm = mkl->cblas_sgemm;
      sgemm_jit_init = &sage2_sgemm_jit_init_mkl;
      sgemm_jit_get = &sage2_sgemm_jit_get_mkl;
      sgemm_jit_uninit = &sage2_sgemm_jit_uninit_mkl;
      return;
    }
  }

  fprintf(stderr, "sage2_sgemm is using default kernel.\n");
  mkl = NULL;
  sgemm = &sage2_sgemm_default;
  sgemm_jit_init = &sage2_sgemm_jit_init_default;
  sgemm_jit_get = &sage2_sgemm_jit_get_default;
  sgemm_jit_uninit = &sage2_sgemm_jit_uninit_default;
}

void sage2_sgemm(int layout, int transX, int transY, int m, int n, int k,
                 float alpha, const float* X, int ldX, const float* Y, int ldY,
                 float beta, float* Z, int ldZ) {
  sgemm(layout, transX, transY, m, n, k, alpha, X, ldX, Y, ldY, beta, Z, ldZ);
}

void* sage2_sgemm_jit_init(int layout, int transX, int transY, int m, int n,
                           int k, float alpha, int ldX, int ldY, float beta,
                           int ldZ) {
  return sgemm_jit_init(layout, transX, transY, m, n, k, alpha, ldX, ldY, beta,
                        ldZ);
}

sage2_sgemm_t sage2_sgemm_jit_get(void* jit) { return sgemm_jit_get(jit); }

void sage2_sgemm_jit_uninit(void* jit) { sgemm_jit_uninit(jit); }
