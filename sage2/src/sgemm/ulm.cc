// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//
// Reference:
// http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html
// https://github.com/michael-lehn/ulmBLAS/blob/bench-eigen/src/level3/dgemm_nn.c
//

#include "sgemm/internal_sgemm.h"

PRIVATE_C_FUNC void sage2_sgemm_ulm_pack_X_Xc_1(int mc, int kc, const float* X,
                                                float* packed_X, int X_inc_row,
                                                int X_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_pack_X_Xc_2(int mc, int kc, const float* X,
                                                float* packed_X, int X_inc_row,
                                                int X_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_pack_X_Xc_4(int mc, int kc, const float* X,
                                                float* packed_X, int X_inc_row,
                                                int X_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_pack_X_Xc_8(int mc, int kc, const float* X,
                                                float* packed_X, int X_inc_row,
                                                int X_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_pack_Y_Yc_8(int kc, int nc, const float* Y,
                                                float* packed_Y, int Y_inc_row,
                                                int Y_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_pack_Y_Yc_16(int kc, int nc, const float* Y,
                                                 float* packed_Y, int Y_inc_row,
                                                 int Y_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_pack_Y_Yc_32(int kc, int nc, const float* Y,
                                                 float* packed_Y, int Y_inc_row,
                                                 int Y_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_1x16(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_a1_Zrm_1x16(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_1x32(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_a1_Zrm_1x32(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_2x16(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_a1_Zrm_2x16(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_2x32(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_a1_Zrm_2x32(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_4x16(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_a1_Zrm_4x16(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_8x8(int kc, float alpha,
                                                     const float* X,
                                                     const float* Y, float beta,
                                                     float* Z, int Z_inc_row,
                                                     int Z_inc_col);
PRIVATE_C_FUNC void sage2_sgemm_ulm_micro_kernel_a1_Zrm_8x8(
    int kc, float alpha, const float* X, const float* Y, float beta, float* Z,
    int Z_inc_row, int Z_inc_col);

namespace sage2 {
namespace {

/************************************************************************/
/* PackX */
/************************************************************************/
template <int MR>
class PackX {
 public:
  static void pack_X(int mc, int kc, const float* X, float* packed_X,
                     int X_inc_row, int X_inc_col) noexcept {
    int mp = mc / MR;
    int _mr = mc % MR;
    const float* _X = X;
    int i, ii, l;
    for (i = 0; i < mp; ++i) {
      _X = X;
      for (l = 0; l < kc; ++l) {
        for (ii = 0; ii < MR; ++ii) {
          packed_X[ii] = _X[ii * X_inc_row];
        }
        packed_X += MR;
        _X += X_inc_col;
      }
      X += MR * X_inc_row;
    }
    if (_mr > 0) {
      for (l = 0; l < kc; ++l) {
        for (i = 0; i < _mr; ++i) {
          packed_X[i] = X[i * X_inc_row];
        }
        for (; i < MR; ++i) {
          packed_X[i] = 0;
        }
        packed_X += MR;
        X += X_inc_col;
      }
    }
  }

  static void pack_X_Xc(int mc, int kc, const float* X, float* packed_X,
                        int X_inc_row, int X_inc_col) noexcept {
    pack_X(mc, kc, X, packed_X, X_inc_row, X_inc_col);
  }
};

template <>
inline void PackX<1>::pack_X(int mc, int kc, const float* X, float* packed_X,
                             int X_inc_row, int X_inc_col) noexcept {
  const float* _X = X;
  int i, l;
  for (i = 0; i < mc; ++i) {
    _X = X;
    for (l = 0; l < kc; ++l) {
      *packed_X = *_X;
      packed_X += 1;
      _X += X_inc_col;
    }
    X += X_inc_row;
  }
}

template <>
inline void PackX<1>::pack_X_Xc(int mc, int kc, const float* X, float* packed_X,
                                int X_inc_row, int X_inc_col) noexcept {
  sage2_sgemm_ulm_pack_X_Xc_1(mc, kc, X, packed_X, X_inc_row, X_inc_col);
}

template <>
inline void PackX<2>::pack_X_Xc(int mc, int kc, const float* X, float* packed_X,
                                int X_inc_row, int X_inc_col) noexcept {
  sage2_sgemm_ulm_pack_X_Xc_2(mc, kc, X, packed_X, X_inc_row, X_inc_col);
}

template <>
inline void PackX<4>::pack_X_Xc(int mc, int kc, const float* X, float* packed_X,
                                int X_inc_row, int X_inc_col) noexcept {
  sage2_sgemm_ulm_pack_X_Xc_4(mc, kc, X, packed_X, X_inc_row, X_inc_col);
}

template <>
inline void PackX<8>::pack_X_Xc(int mc, int kc, const float* X, float* packed_X,
                                int X_inc_row, int X_inc_col) noexcept {
  sage2_sgemm_ulm_pack_X_Xc_8(mc, kc, X, packed_X, X_inc_row, X_inc_col);
}

/************************************************************************/
/* PackY */
/************************************************************************/
template <int NR>
class PackY {
 public:
  static void pack_Y(int kc, int nc, const float* Y, float* packed_Y,
                     int Y_inc_row, int Y_inc_col) noexcept {
    int np = nc / NR;
    int _nr = nc % NR;
    const float* _Y = Y;
    int j, jj, l;
    for (j = 0; j < np; ++j) {
      _Y = Y;
      for (l = 0; l < kc; ++l) {
        for (jj = 0; jj < NR; ++jj) {
          packed_Y[jj] = _Y[jj * Y_inc_col];
        }
        packed_Y += NR;
        _Y += Y_inc_row;
      }
      Y += NR * Y_inc_col;
    }
    if (_nr > 0) {
      for (l = 0; l < kc; ++l) {
        for (j = 0; j < _nr; ++j) {
          packed_Y[j] = Y[j * Y_inc_col];
        }
        for (; j < NR; ++j) {
          packed_Y[j] = 0;
        }
        packed_Y += NR;
        Y += Y_inc_row;
      }
    }
  }

  static void pack_Y_Yc(int kc, int nc, const float* Y, float* packed_Y,
                        int Y_inc_row, int Y_inc_col) noexcept {
    pack_Y(kc, nc, Y, packed_Y, Y_inc_row, Y_inc_col);
  }
};

template <>
inline void PackY<1>::pack_Y(int kc, int nc, const float* Y, float* packed_Y,
                             int Y_inc_row, int Y_inc_col) noexcept {
  const float* _Y = Y;
  int j, l;
  for (j = 0; j < nc; ++j) {
    _Y = Y;
    for (l = 0; l < kc; ++l) {
      *packed_Y = *_Y;
      packed_Y += 1;
      _Y += Y_inc_row;
    }
    Y += Y_inc_col;
  }
}

template <>
inline void PackY<8>::pack_Y_Yc(int kc, int nc, const float* Y, float* packed_Y,
                                int Y_inc_row, int Y_inc_col) noexcept {
  sage2_sgemm_ulm_pack_Y_Yc_8(kc, nc, Y, packed_Y, Y_inc_row, Y_inc_col);
}

template <>
inline void PackY<16>::pack_Y_Yc(int kc, int nc, const float* Y,
                                 float* packed_Y, int Y_inc_row,
                                 int Y_inc_col) noexcept {
  sage2_sgemm_ulm_pack_Y_Yc_16(kc, nc, Y, packed_Y, Y_inc_row, Y_inc_col);
}

template <>
inline void PackY<32>::pack_Y_Yc(int kc, int nc, const float* Y,
                                 float* packed_Y, int Y_inc_row,
                                 int Y_inc_col) noexcept {
  sage2_sgemm_ulm_pack_Y_Yc_32(kc, nc, Y, packed_Y, Y_inc_row, Y_inc_col);
}

/************************************************************************/
/* MicroKernel */
/************************************************************************/
template <int MR, int NR>
class MicroKernel {
 public:
  static void micro_kernel(int kc, float alpha, const float* X, const float* Y,
                           float beta, float* Z, int Z_inc_row,
                           int Z_inc_col) noexcept {
    float XY[MR * NR] = {0};
    int i, j, l;

    for (l = 0; l < kc; ++l) {
      for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
          XY[i + j * MR] += X[i] * Y[j];
        }
      }
      X += MR;
      Y += NR;
    }

    if (beta == 0) {
      for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
          Z[i * Z_inc_row + j * Z_inc_col] = 0;
        }
      }
    } else if (beta == 1) {
    } else {
      for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
          Z[i * Z_inc_row + j * Z_inc_col] *= beta;
        }
      }
    }

    if (alpha == 1) {
      for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
          Z[i * Z_inc_row + j * Z_inc_col] += XY[i + j * MR];
        }
      }
    } else {
      for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
          Z[i * Z_inc_row + j * Z_inc_col] += alpha * XY[i + j * MR];
        }
      }
    }
  }

  static void micro_kernel_a1_Zrm(int kc, float alpha, const float* X,
                                  const float* Y, float beta, float* Z,
                                  int Z_inc_row, int Z_inc_col) noexcept {
    micro_kernel(kc, alpha, X, Y, beta, Z, Z_inc_row, Z_inc_col);
  }
};

template <>
inline void MicroKernel<1, 16>::micro_kernel(int kc, float alpha,
                                             const float* X, const float* Y,
                                             float beta, float* Z,
                                             int Z_inc_row,
                                             int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_1x16(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                    Z_inc_col);
}

template <>
inline void MicroKernel<1, 16>::micro_kernel_a1_Zrm(int kc, float alpha,
                                                    const float* X,
                                                    const float* Y, float beta,
                                                    float* Z, int Z_inc_row,
                                                    int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_a1_Zrm_1x16(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                           Z_inc_col);
}

template <>
inline void MicroKernel<1, 32>::micro_kernel(int kc, float alpha,
                                             const float* X, const float* Y,
                                             float beta, float* Z,
                                             int Z_inc_row,
                                             int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_1x32(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                    Z_inc_col);
}

template <>
inline void MicroKernel<1, 32>::micro_kernel_a1_Zrm(int kc, float alpha,
                                                    const float* X,
                                                    const float* Y, float beta,
                                                    float* Z, int Z_inc_row,
                                                    int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_a1_Zrm_1x32(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                           Z_inc_col);
}

template <>
inline void MicroKernel<2, 16>::micro_kernel(int kc, float alpha,
                                             const float* X, const float* Y,
                                             float beta, float* Z,
                                             int Z_inc_row,
                                             int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_2x16(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                    Z_inc_col);
}

template <>
inline void MicroKernel<2, 16>::micro_kernel_a1_Zrm(int kc, float alpha,
                                                    const float* X,
                                                    const float* Y, float beta,
                                                    float* Z, int Z_inc_row,
                                                    int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_a1_Zrm_2x16(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                           Z_inc_col);
}

template <>
inline void MicroKernel<2, 32>::micro_kernel(int kc, float alpha,
                                             const float* X, const float* Y,
                                             float beta, float* Z,
                                             int Z_inc_row,
                                             int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_2x32(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                    Z_inc_col);
}

template <>
inline void MicroKernel<2, 32>::micro_kernel_a1_Zrm(int kc, float alpha,
                                                    const float* X,
                                                    const float* Y, float beta,
                                                    float* Z, int Z_inc_row,
                                                    int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_a1_Zrm_2x32(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                           Z_inc_col);
}

template <>
inline void MicroKernel<4, 16>::micro_kernel(int kc, float alpha,
                                             const float* X, const float* Y,
                                             float beta, float* Z,
                                             int Z_inc_row,
                                             int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_4x16(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                    Z_inc_col);
}

template <>
inline void MicroKernel<4, 16>::micro_kernel_a1_Zrm(int kc, float alpha,
                                                    const float* X,
                                                    const float* Y, float beta,
                                                    float* Z, int Z_inc_row,
                                                    int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_a1_Zrm_4x16(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                           Z_inc_col);
}

template <>
inline void MicroKernel<8, 8>::micro_kernel(int kc, float alpha, const float* X,
                                            const float* Y, float beta,
                                            float* Z, int Z_inc_row,
                                            int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_8x8(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                   Z_inc_col);
}

template <>
inline void MicroKernel<8, 8>::micro_kernel_a1_Zrm(int kc, float alpha,
                                                   const float* X,
                                                   const float* Y, float beta,
                                                   float* Z, int Z_inc_row,
                                                   int Z_inc_col) noexcept {
  sage2_sgemm_ulm_micro_kernel_a1_Zrm_8x8(kc, alpha, X, Y, beta, Z, Z_inc_row,
                                          Z_inc_col);
}

/************************************************************************/
/* SGEMM */
/************************************************************************/
template <int MC, int KC, int NC, int MR, int NR, int ULM_METHOD>
class SGEMM {
 public:
  using pack_x_t = PackX<MR>;
  using pack_y_t = PackY<NR>;
  using micro_kernel_t = MicroKernel<MR, NR>;

 public:
  struct Free {
    void operator()(void* ptr) const noexcept { free(ptr); }
  };
  using mem_tls_t = std::unique_ptr<void, Free>;
  static thread_local mem_tls_t mem_tls;

  static void* get_mem_tls() noexcept {
    if (!mem_tls) {
      mem_tls.reset(malloc((MC * KC + KC * NC + MR * NR) * sizeof(float)));
    }
    return mem_tls.get();
  }

  static void free_mem_tls(void* ptr) noexcept { (void)ptr; }

 public:
  static void* get_mem() noexcept {
    return malloc((MC * KC + KC * NC + MR * NR) * sizeof(float));
  }

  static void free_mem(void* ptr) noexcept { free(ptr); }

 public:
  static void scal_Z(sage2_sgemm_ctx* ctx, int mr, int nr, float beta,
                     float* Z) noexcept {
    int Z_inc_row = ctx->Z_inc_row;
    int Z_inc_col = ctx->Z_inc_col;
    int i, j;
    if (beta == 0) {
      for (j = 0; j < nr; ++j) {
        for (i = 0; i < mr; ++i) {
          Z[i * Z_inc_row + j * Z_inc_col] = 0;
        }
      }
    } else if (beta == 1) {
    } else {
      for (j = 0; j < nr; ++j) {
        for (i = 0; i < mr; ++i) {
          Z[i * Z_inc_row + j * Z_inc_col] *= beta;
        }
      }
    }
  }

  static void acc_packed_Z(sage2_sgemm_ctx* ctx, int mr, int nr,
                           float* Z) noexcept {
    int Z_inc_row = ctx->Z_inc_row;
    int Z_inc_col = ctx->Z_inc_col;
    const float* packed_Z = ctx->ulm.packed_Z;
    int i, j;
    for (j = 0; j < nr; ++j) {
      for (i = 0; i < mr; ++i) {
        Z[i * Z_inc_row + j * Z_inc_col] += packed_Z[i + j * MR];
      }
    }
  }

  static void macro_kernel(sage2_sgemm_ctx* ctx, int mc, int nc, int kc,
                           float beta, float* Z) noexcept {
    float alpha = ctx->alpha;
    int Z_inc_row = ctx->Z_inc_row;
    int Z_inc_col = ctx->Z_inc_col;
    float* packed_X = ctx->ulm.packed_X;
    float* packed_Y = ctx->ulm.packed_Y;
    float* packed_Z = ctx->ulm.packed_Z;
    int mp = (mc + MR - 1) / MR;
    int np = (nc + NR - 1) / NR;
    int _mr = mc % MR;
    int _nr = nc % NR;
    int mr;
    int nr;
    int i, j;
    for (j = 0; j < np; ++j) {
      nr = (j != np - 1 || _nr == 0) ? NR : _nr;
      for (i = 0; i < mp; ++i) {
        mr = (i != mp - 1 || _mr == 0) ? MR : _mr;
        if (mr == MR && nr == NR) {
          ctx->ulm.micro_kernel(kc, alpha, &packed_X[i * kc * MR],
                                &packed_Y[j * kc * NR], beta,
                                &Z[i * MR * Z_inc_row + j * NR * Z_inc_col],
                                Z_inc_row, Z_inc_col);
        } else {
          micro_kernel_t::micro_kernel(kc, alpha, &packed_X[i * kc * MR],
                                       &packed_Y[j * kc * NR], 0, packed_Z, 1,
                                       MR);
          scal_Z(ctx, mr, nr, beta,
                 &Z[i * MR * Z_inc_row + j * NR * Z_inc_col]);
          acc_packed_Z(ctx, mr, nr,
                       &Z[i * MR * Z_inc_row + j * NR * Z_inc_col]);
        }
      }
    }
  }

  static void entry(sage2_sgemm_ctx* ctx, const float* X, const float* Y,
                    float* Z) noexcept {
    float beta = ctx->beta;
    int mc, nc, kc;
    int i, j, l;
    float _beta;
    for (j = 0; j < ctx->ulm.nb; ++j) {
      nc = (j != ctx->ulm.nb - 1 || ctx->ulm.nc == 0) ? NC : ctx->ulm.nc;
      for (l = 0; l < ctx->ulm.kb; ++l) {
        kc = (l != ctx->ulm.kb - 1 || ctx->ulm.kc == 0) ? KC : ctx->ulm.kc;
        _beta = (l == 0) ? beta : 1;
        ctx->ulm.pack_Y(kc, nc,
                        &Y[l * KC * ctx->Y_inc_row + j * NC * ctx->Y_inc_col],
                        ctx->ulm.packed_Y, ctx->Y_inc_row, ctx->Y_inc_col);
        for (i = 0; i < ctx->ulm.mb; ++i) {
          mc = (i != ctx->ulm.mb - 1 || ctx->ulm.mc == 0) ? MC : ctx->ulm.mc;
          ctx->ulm.pack_X(mc, kc,
                          &X[i * MC * ctx->X_inc_row + l * KC * ctx->X_inc_col],
                          ctx->ulm.packed_X, ctx->X_inc_row, ctx->X_inc_col);
          macro_kernel(ctx, mc, nc, kc, _beta,
                       &Z[i * MC * ctx->Z_inc_row + j * NC * ctx->Z_inc_col]);
        }
      }
    }
  }

 public:
  static int init(sage2_sgemm_ctx* ctx) noexcept {
    int flags = ctx->flags;
    int m = ctx->m;
    int n = ctx->n;
    int k = ctx->k;

    ctx->func = &entry;
    ctx->method = METHOD_ULM;

    ctx->ulm.mb = (m + MC - 1) / MC;  // ceil(m / MC)
    ctx->ulm.nb = (n + NC - 1) / NC;  // ceil(n / NC)
    ctx->ulm.kb = (k + KC - 1) / KC;  // ceil(k / KC)
    ctx->ulm.mc = m % MC;
    ctx->ulm.nc = n % NC;
    ctx->ulm.kc = k % KC;

    if (flags & FLAGS_USE_TLS) {
      ctx->ulm.packed_X = (float*)get_mem_tls();
    } else {
      ctx->ulm.packed_X = (float*)get_mem();
    }
    if (ctx->ulm.packed_X == nullptr) {
      fprintf(stderr, "Failed to alloc memory.\n");
      return -1;
    }
    ctx->ulm.packed_Y = ctx->ulm.packed_X + MC * KC;
    ctx->ulm.packed_Z = ctx->ulm.packed_Y + KC * NC;

    if (ctx->X_inc_row == 1) {
      // X is continuous for packing.
      ctx->ulm.pack_X = &pack_x_t::pack_X_Xc;
    } else {
      ctx->ulm.pack_X = &pack_x_t::pack_X;
    }

    if (ctx->Y_inc_col == 1) {
      // Y is continuous for packing.
      ctx->ulm.pack_Y = &pack_y_t::pack_Y_Yc;
    } else {
      ctx->ulm.pack_Y = &pack_y_t::pack_Y;
    }

    if (ctx->alpha == 1 && ctx->Z_row_major) {
      ctx->ulm.micro_kernel = &micro_kernel_t::micro_kernel_a1_Zrm;
    } else {
      ctx->ulm.micro_kernel = &micro_kernel_t::micro_kernel;
    }

    ctx->ulm.method = ULM_METHOD;
    return 0;
  }

  static void uninit(sage2_sgemm_ctx* ctx) noexcept {
    if (ctx->flags & FLAGS_USE_TLS) {
      free_mem_tls(ctx->ulm.packed_X);
    } else {
      free_mem(ctx->ulm.packed_X);
    }
  }
};

template <int MC, int KC, int NC, int MR, int NR, int ULM_METHOD>
thread_local typename SGEMM<MC, KC, NC, MR, NR, ULM_METHOD>::mem_tls_t
    SGEMM<MC, KC, NC, MR, NR, ULM_METHOD>::mem_tls;

enum {
  ULM_METHOD_4x16 = 1,
  ULM_METHOD_8x8,
};

using ulm_4x16_t = SGEMM<384, 384, 4096, 4, 16, ULM_METHOD_4x16>;
using ulm_8x8_t = SGEMM<384, 384, 4096, 8, 8, ULM_METHOD_8x8>;

}  // namespace
}  // namespace sage2

int sage2_sgemm_ulm_init(sage2_sgemm_ctx* ctx) {
  if (ctx->n <= 8) {
    return sage2::ulm_8x8_t::init(ctx);
  }
  return sage2::ulm_4x16_t::init(ctx);
}

void sage2_sgemm_ulm_uninit(sage2_sgemm_ctx* ctx) {
  switch (ctx->ulm.method) {
    case sage2::ULM_METHOD_4x16:
      sage2::ulm_4x16_t::uninit(ctx);
      break;
    case sage2::ULM_METHOD_8x8:
      sage2::ulm_8x8_t::uninit(ctx);
      break;
  }
}
