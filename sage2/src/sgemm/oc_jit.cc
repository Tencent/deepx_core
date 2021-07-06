// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

namespace sage2 {
namespace {

template <typename T>
inline T max(T a, T b) {
  return (a > b) ? a : b;
}

/************************************************************************/
/* OCZrmJit */
/************************************************************************/
class OCZrmJit : public XbyakCodeGenerator {
 private:
  const Reg64& CTX_PTR = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& Y_PTR = rdx;
  const Reg64& Z_PTR = rcx;
  const Reg64& J_REG = r8;
  const Reg64& M_REG = r9;
  const Reg64& N_REG = r10;
  const Ymm& XI_REG_YMM = ymm0;
  const Xmm& XI_REG_XMM = xmm0;
  int load_alpha_ = 0;
  int load_beta_ = 0;
  int alpha_index_ = 0;
  int beta_index_ = 0;
  int free_ymm_index_ = 0;

 private:
  const Ymm& alpha_ymm() const { return ymm(alpha_index_); }
  const Ymm& beta_ymm() const { return ymm(beta_index_); }
  const Xmm& beta_xmm() const { return xmm(beta_index_); }

 private:
  void EmitLoadY8x_v1(int n, int r_index, int offset) {
    for (int i = 0; i < n; ++i) {
      vmovups(ymm(r_index + i), ptr[Y_PTR + offset + i * 32]);
    }
  }

  void EmitLoadY4_v1(int r_index, int offset) {
    vmovups(xmm(r_index), ptr[Y_PTR + offset]);
  }

  void EmitKernel8x_v1(sage2_sgemm_ctx* ctx, int n, int r1_index, int r2_index,
                       int offset) {
    if (ctx->beta == 0) {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r2_index + i), XI_REG_YMM, ymm(r1_index + i));
      }
    } else if (ctx->beta == 1) {
      for (int i = 0; i < n; ++i) {
        vmovups(ymm(r2_index + i), ptr[Z_PTR + offset + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r2_index + i), XI_REG_YMM, ymm(r1_index + i));
      }
    } else {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r2_index + i), beta_ymm(), ptr[Z_PTR + offset + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r2_index + i), XI_REG_YMM, ymm(r1_index + i));
      }
    }
    for (int i = 0; i < n; ++i) {
      vmovups(ptr[Z_PTR + offset + i * 32], ymm(r2_index + i));
    }
  }

  void EmitKernel4_v1(sage2_sgemm_ctx* ctx, int r1_index, int r2_index,
                      int offset) {
    if (ctx->beta == 0) {
      vmulps(xmm(r2_index), XI_REG_XMM, xmm(r1_index));
    } else if (ctx->beta == 1) {
      vmovups(xmm(r2_index), ptr[Z_PTR + offset]);
      vfmadd231ps(xmm(r2_index), XI_REG_XMM, xmm(r1_index));
    } else {
      vmulps(xmm(r2_index), beta_xmm(), ptr[Z_PTR + offset]);
      vfmadd231ps(xmm(r2_index), XI_REG_XMM, xmm(r1_index));
    }
    vmovups(ptr[Z_PTR + offset], xmm(r2_index));
  }

  void EmitKernel1_v1(sage2_sgemm_ctx* ctx, int r1_index, int r2_index,
                      int offset) {
    vmovss(xmm(r1_index), ptr[Y_PTR + offset]);
    if (ctx->beta == 0) {
      vmulss(xmm(r2_index), XI_REG_XMM, xmm(r1_index));
    } else if (ctx->beta == 1) {
      vmovss(xmm(r2_index), ptr[Z_PTR + offset]);
      vfmadd231ss(xmm(r2_index), XI_REG_XMM, xmm(r1_index));
    } else {
      vmulss(xmm(r2_index), beta_xmm(), ptr[Z_PTR + offset]);
      vfmadd231ss(xmm(r2_index), XI_REG_XMM, xmm(r1_index));
    }
    vmovss(ptr[Z_PTR + offset], xmm(r2_index));
  }

  void EmitKernel8x_v2(sage2_sgemm_ctx* ctx, int n, int r_index) {
    if (ctx->beta == 0) {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r_index + i), XI_REG_YMM, ptr[Y_PTR + J_REG * 4 + i * 32]);
      }
    } else if (ctx->beta == 1) {
      for (int i = 0; i < n; ++i) {
        vmovups(ymm(r_index + i), ptr[Z_PTR + J_REG * 4 + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r_index + i), XI_REG_YMM,
                    ptr[Y_PTR + J_REG * 4 + i * 32]);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r_index + i), beta_ymm(), ptr[Z_PTR + J_REG * 4 + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r_index + i), XI_REG_YMM,
                    ptr[Y_PTR + J_REG * 4 + i * 32]);
      }
    }
    for (int i = 0; i < n; ++i) {
      vmovups(ptr[Z_PTR + J_REG * 4 + i * 32], ymm(r_index + i));
    }
  }

  void EmitKernel4_v2(sage2_sgemm_ctx* ctx, int r_index) {
    if (ctx->beta == 0) {
      vmulps(xmm(r_index), XI_REG_XMM, ptr[Y_PTR + J_REG * 4]);
    } else if (ctx->beta == 1) {
      vmovups(xmm(r_index), ptr[Z_PTR + J_REG * 4]);
      vfmadd231ps(xmm(r_index), XI_REG_XMM, ptr[Y_PTR + J_REG * 4]);
    } else {
      vmulps(xmm(r_index), beta_xmm(), ptr[Z_PTR + J_REG * 4]);
      vfmadd231ps(xmm(r_index), XI_REG_XMM, ptr[Y_PTR + J_REG * 4]);
    }
    vmovups(ptr[Z_PTR + J_REG * 4], xmm(r_index));
  }

  void EmitKernel1x_v2(sage2_sgemm_ctx* ctx, int n, int r_index) {
    if (ctx->beta == 0) {
      for (int i = 0; i < n; ++i) {
        vmulss(xmm(r_index + i), XI_REG_XMM, ptr[Y_PTR + J_REG * 4 + i * 4]);
      }
    } else if (ctx->beta == 1) {
      for (int i = 0; i < n; ++i) {
        vmovss(xmm(r_index + i), ptr[Z_PTR + J_REG * 4 + i * 4]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ss(xmm(r_index + i), XI_REG_XMM,
                    ptr[Y_PTR + J_REG * 4 + i * 4]);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        vmulss(xmm(r_index + i), beta_xmm(), ptr[Z_PTR + J_REG * 4 + i * 4]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ss(xmm(r_index + i), XI_REG_XMM,
                    ptr[Y_PTR + J_REG * 4 + i * 4]);
      }
    }
    for (int i = 0; i < n; ++i) {
      vmovss(ptr[Z_PTR + J_REG * 4 + i * 4], xmm(r_index + i));
    }
  }

 private:
  bool TryCompile_v1(sage2_sgemm_ctx* ctx) {
    int m = ctx->m;
    int n = ctx->n;
    int Z_inc_row = ctx->Z_inc_row;
    int q = n / 8;
    int r = n % 8;
    int r4 = r & 4;
    int r3 = r & 3;
    int tmp_ymm_index = free_ymm_index_ + q + (r4 ? 1 : 0);
    int offset;

    int need_ymm = 1 + load_alpha_ + load_beta_;
    need_ymm += q + (r4 ? 1 : 0);  // y
    need_ymm += max(q, 2);         // tmp
    if (need_ymm > 16) {
      return false;
    }

    offset = 0;
    if (q) {
      EmitLoadY8x_v1(q, free_ymm_index_, offset);
      offset += q * 32;
    }

    if (r4) {
      EmitLoadY4_v1(free_ymm_index_ + q, offset);
    }

    offset = 0;
    mov(M_REG, m);
    L(".1");
    vbroadcastss(XI_REG_YMM, ptr[X_PTR]);
    if (ctx->alpha != 1) {
      vmulps(XI_REG_YMM, XI_REG_YMM, alpha_ymm());
    }

    if (q) {
      EmitKernel8x_v1(ctx, q, free_ymm_index_, tmp_ymm_index, offset);
      offset += q * 32;
    }

    if (r4) {
      EmitKernel4_v1(ctx, free_ymm_index_ + q, tmp_ymm_index, offset);
      offset += 16;
    }

    if (r3) {
      for (int i = 0; i < r3; ++i) {
        EmitKernel1_v1(ctx, tmp_ymm_index, tmp_ymm_index + 1, offset + i * 4);
      }
    }

    add(X_PTR, 4);
    add(Z_PTR, Z_inc_row * 4);
    sub(M_REG, 1);
    jne(".1", T_NEAR);
    return true;
  }

  void Compile_v2(sage2_sgemm_ctx* ctx) {
    int m = ctx->m;
    int n = ctx->n;
    int Z_inc_row = ctx->Z_inc_row;
    int nn;

    mov(M_REG, m);

    L(".1");
    vbroadcastss(XI_REG_YMM, ptr[X_PTR]);
    if (ctx->alpha != 1) {
      vmulps(XI_REG_YMM, XI_REG_YMM, alpha_ymm());
    }

    xor_(J_REG, J_REG);

    nn = n & -64;
    if (nn) {
      mov(N_REG, nn);
      L(".10");
      EmitKernel8x_v2(ctx, 8, free_ymm_index_);
      add(J_REG, 64);
      sub(N_REG, 64);
      jne(".10", T_NEAR);
    }

    nn = (n - nn) & -8;
    if (nn) {
      EmitKernel8x_v2(ctx, nn / 8, free_ymm_index_);
      add(J_REG, nn);
    }

    nn = n & 4;
    if (nn) {
      EmitKernel4_v2(ctx, free_ymm_index_);
      add(J_REG, 4);
    }

    nn = n & 3;
    if (nn) {
      EmitKernel1x_v2(ctx, nn, free_ymm_index_);
    }

    add(X_PTR, 4);
    add(Z_PTR, Z_inc_row * 4);
    sub(M_REG, 1);
    jne(".1", T_NEAR);
  }

 public:
  void Compile(sage2_sgemm_ctx* ctx) {
    free_ymm_index_ = 1;
    if (ctx->alpha != 1) {
      load_alpha_ = 1;
      alpha_index_ = free_ymm_index_++;
      vbroadcastss(ymm(alpha_index_), ptr[CTX_PTR + ALPHA_OFFSET]);
    } else {
      load_alpha_ = 0;
    }

    if (ctx->beta == 0) {
      load_beta_ = 0;
    } else if (ctx->beta == 1) {
      load_beta_ = 0;
    } else {
      load_beta_ = 1;
      beta_index_ = free_ymm_index_++;
      vbroadcastss(ymm(beta_index_), ptr[CTX_PTR + BETA_OFFSET]);
    }

    if (!TryCompile_v1(ctx)) {
      Compile_v2(ctx);
    }

    vzeroupper();
    ret();

    ctx->func = getCode<_sage2_sgemm_t>();
    ctx->method_data = this;
  }
};

/************************************************************************/
/* OCZcmJit */
/************************************************************************/
class OCZcmJit : public XbyakCodeGenerator {
 private:
  const Reg64& CTX_PTR = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& Y_PTR = rdx;
  const Reg64& Z_PTR = rcx;
  const Reg64& I_REG = r8;
  const Reg64& M_REG = r9;
  const Reg64& N_REG = r10;
  const Ymm& YI_REG_YMM = ymm0;
  const Xmm& YI_REG_XMM = xmm0;
  int load_alpha_ = 0;
  int load_beta_ = 0;
  int alpha_index_ = 0;
  int beta_index_ = 0;
  int free_ymm_index_ = 0;

 private:
  const Ymm& alpha_ymm() const { return ymm(alpha_index_); }
  const Ymm& beta_ymm() const { return ymm(beta_index_); }
  const Xmm& beta_xmm() const { return xmm(beta_index_); }

 private:
  void EmitLoadY8x_v1(int n, int r_index, int offset) {
    for (int i = 0; i < n; ++i) {
      vmovups(ymm(r_index + i), ptr[X_PTR + offset + i * 32]);
    }
  }

  void EmitLoadY4_v1(int r_index, int offset) {
    vmovups(xmm(r_index), ptr[X_PTR + offset]);
  }

  void EmitKernel8x_v1(sage2_sgemm_ctx* ctx, int n, int r1_index, int r2_index,
                       int offset) {
    if (ctx->beta == 0) {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r2_index + i), YI_REG_YMM, ymm(r1_index + i));
      }
    } else if (ctx->beta == 1) {
      for (int i = 0; i < n; ++i) {
        vmovups(ymm(r2_index + i), ptr[Z_PTR + offset + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r2_index + i), YI_REG_YMM, ymm(r1_index + i));
      }
    } else {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r2_index + i), beta_ymm(), ptr[Z_PTR + offset + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r2_index + i), YI_REG_YMM, ymm(r1_index + i));
      }
    }
    for (int i = 0; i < n; ++i) {
      vmovups(ptr[Z_PTR + offset + i * 32], ymm(r2_index + i));
    }
  }

  void EmitKernel4_v1(sage2_sgemm_ctx* ctx, int r1_index, int r2_index,
                      int offset) {
    if (ctx->beta == 0) {
      vmulps(xmm(r2_index), YI_REG_XMM, xmm(r1_index));
    } else if (ctx->beta == 1) {
      vmovups(xmm(r2_index), ptr[Z_PTR + offset]);
      vfmadd231ps(xmm(r2_index), YI_REG_XMM, xmm(r1_index));
    } else {
      vmulps(xmm(r2_index), beta_xmm(), ptr[Z_PTR + offset]);
      vfmadd231ps(xmm(r2_index), YI_REG_XMM, xmm(r1_index));
    }
    vmovups(ptr[Z_PTR + offset], xmm(r2_index));
  }

  void EmitKernel1_v1(sage2_sgemm_ctx* ctx, int r1_index, int r2_index,
                      int offset) {
    vmovss(xmm(r1_index), ptr[X_PTR + offset]);
    if (ctx->beta == 0) {
      vmulss(xmm(r2_index), YI_REG_XMM, xmm(r1_index));
    } else if (ctx->beta == 1) {
      vmovss(xmm(r2_index), ptr[Z_PTR + offset]);
      vfmadd231ss(xmm(r2_index), YI_REG_XMM, xmm(r1_index));
    } else {
      vmulss(xmm(r2_index), beta_xmm(), ptr[Z_PTR + offset]);
      vfmadd231ss(xmm(r2_index), YI_REG_XMM, xmm(r1_index));
    }
    vmovss(ptr[Z_PTR + offset], xmm(r2_index));
  }

  void EmitKernel8x_v2(sage2_sgemm_ctx* ctx, int n, int r_index) {
    if (ctx->beta == 0) {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r_index + i), YI_REG_YMM, ptr[X_PTR + I_REG * 4 + i * 32]);
      }
    } else if (ctx->beta == 1) {
      for (int i = 0; i < n; ++i) {
        vmovups(ymm(r_index + i), ptr[Z_PTR + I_REG * 4 + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r_index + i), YI_REG_YMM,
                    ptr[X_PTR + I_REG * 4 + i * 32]);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        vmulps(ymm(r_index + i), beta_ymm(), ptr[Z_PTR + I_REG * 4 + i * 32]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ps(ymm(r_index + i), YI_REG_YMM,
                    ptr[X_PTR + I_REG * 4 + i * 32]);
      }
    }
    for (int i = 0; i < n; ++i) {
      vmovups(ptr[Z_PTR + I_REG * 4 + i * 32], ymm(r_index + i));
    }
  }

  void EmitKernel4_v2(sage2_sgemm_ctx* ctx, int r_index) {
    if (ctx->beta == 0) {
      vmulps(xmm(r_index), YI_REG_XMM, ptr[X_PTR + I_REG * 4]);
    } else if (ctx->beta == 1) {
      vmovups(xmm(r_index), ptr[Z_PTR + I_REG * 4]);
      vfmadd231ps(xmm(r_index), YI_REG_XMM, ptr[X_PTR + I_REG * 4]);
    } else {
      vmulps(xmm(r_index), beta_xmm(), ptr[Z_PTR + I_REG * 4]);
      vfmadd231ps(xmm(r_index), YI_REG_XMM, ptr[X_PTR + I_REG * 4]);
    }
    vmovups(ptr[Z_PTR + I_REG * 4], xmm(r_index));
  }

  void EmitKernel1x_v2(sage2_sgemm_ctx* ctx, int n, int r_index) {
    if (ctx->beta == 0) {
      for (int i = 0; i < n; ++i) {
        vmulss(xmm(r_index + i), YI_REG_XMM, ptr[X_PTR + I_REG * 4 + i * 4]);
      }
    } else if (ctx->beta == 1) {
      for (int i = 0; i < n; ++i) {
        vmovss(xmm(r_index + i), ptr[Z_PTR + I_REG * 4 + i * 4]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ss(xmm(r_index + i), YI_REG_XMM,
                    ptr[X_PTR + I_REG * 4 + i * 4]);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        vmulss(xmm(r_index + i), beta_xmm(), ptr[Z_PTR + I_REG * 4 + i * 4]);
      }
      for (int i = 0; i < n; ++i) {
        vfmadd231ss(xmm(r_index + i), YI_REG_XMM,
                    ptr[X_PTR + I_REG * 4 + i * 4]);
      }
    }
    for (int i = 0; i < n; ++i) {
      vmovss(ptr[Z_PTR + I_REG * 4 + i * 4], xmm(r_index + i));
    }
  }

 private:
  bool TryCompile_v1(sage2_sgemm_ctx* ctx) {
    int m = ctx->m;
    int n = ctx->n;
    int Z_inc_col = ctx->Z_inc_col;
    int q = m / 8;
    int r = m % 8;
    int r4 = r & 4;
    int r3 = r & 3;
    int tmp_ymm_index = free_ymm_index_ + q + (r4 ? 1 : 0);
    int offset;

    int need_ymm = 1 + load_alpha_ + load_beta_;
    need_ymm += q + (r4 ? 1 : 0);  // y
    need_ymm += max(q, 2);         // tmp
    if (need_ymm > 16) {
      return false;
    }

    offset = 0;
    if (q) {
      EmitLoadY8x_v1(q, free_ymm_index_, offset);
      offset += q * 32;
    }

    if (r4) {
      EmitLoadY4_v1(free_ymm_index_ + q, offset);
    }

    offset = 0;
    mov(N_REG, n);
    L(".1");
    vbroadcastss(YI_REG_YMM, ptr[Y_PTR]);
    if (ctx->alpha != 1) {
      vmulps(YI_REG_YMM, YI_REG_YMM, alpha_ymm());
    }

    if (q) {
      EmitKernel8x_v1(ctx, q, free_ymm_index_, tmp_ymm_index, offset);
      offset += q * 32;
    }

    if (r4) {
      EmitKernel4_v1(ctx, free_ymm_index_ + q, tmp_ymm_index, offset);
      offset += 16;
    }

    if (r3) {
      for (int i = 0; i < r3; ++i) {
        EmitKernel1_v1(ctx, tmp_ymm_index, tmp_ymm_index + 1, offset + i * 4);
      }
    }

    add(Y_PTR, 4);
    add(Z_PTR, Z_inc_col * 4);
    sub(N_REG, 1);
    jne(".1", T_NEAR);
    return true;
  }

  void Compile_v2(sage2_sgemm_ctx* ctx) {
    int m = ctx->m;
    int n = ctx->n;
    int Z_inc_col = ctx->Z_inc_col;
    int mm;

    mov(N_REG, n);

    L(".1");
    vbroadcastss(YI_REG_YMM, ptr[Y_PTR]);
    if (ctx->alpha != 1) {
      vmulps(YI_REG_YMM, YI_REG_YMM, alpha_ymm());
    }

    xor_(I_REG, I_REG);

    mm = m & -64;
    if (mm) {
      mov(M_REG, mm);
      L(".10");
      EmitKernel8x_v2(ctx, 8, free_ymm_index_);
      add(I_REG, 64);
      sub(M_REG, 64);
      jne(".10", T_NEAR);
    }

    mm = (m - mm) & -8;
    if (mm) {
      EmitKernel8x_v2(ctx, mm / 8, free_ymm_index_);
      add(I_REG, mm);
    }

    mm = m & 4;
    if (mm) {
      EmitKernel4_v2(ctx, free_ymm_index_);
      add(I_REG, 4);
    }

    mm = m & 3;
    if (mm) {
      EmitKernel1x_v2(ctx, mm, free_ymm_index_);
    }

    add(Y_PTR, 4);
    add(Z_PTR, Z_inc_col * 4);
    sub(N_REG, 1);
    jne(".1", T_NEAR);
  }

 public:
  void Compile(sage2_sgemm_ctx* ctx) {
    free_ymm_index_ = 1;
    if (ctx->alpha != 1) {
      load_alpha_ = 1;
      alpha_index_ = free_ymm_index_++;
      vbroadcastss(ymm(alpha_index_), ptr[CTX_PTR + ALPHA_OFFSET]);
    } else {
      load_alpha_ = 0;
    }

    if (ctx->beta == 0) {
      load_beta_ = 0;
    } else if (ctx->beta == 1) {
      load_beta_ = 0;
    } else {
      load_beta_ = 1;
      beta_index_ = free_ymm_index_++;
      vbroadcastss(ymm(beta_index_), ptr[CTX_PTR + BETA_OFFSET]);
    }

    if (!TryCompile_v1(ctx)) {
      Compile_v2(ctx);
    }

    vzeroupper();
    ret();

    ctx->func = getCode<_sage2_sgemm_t>();
    ctx->method_data = this;
  }
};

}  // namespace
}  // namespace sage2

int sage2_sgemm_oc_Zrm_jit_init(sage2_sgemm_ctx* ctx) {
  auto* jit = new (std::nothrow) sage2::OCZrmJit;
  if (jit == nullptr) {
    fprintf(stderr, "Failed to alloc memory.\n");
    return -1;
  }
  jit->Compile(ctx);
  ctx->method = METHOD_OC_ZRM_JIT;
  return 0;
}

void sage2_sgemm_oc_Zrm_jit_uninit(sage2_sgemm_ctx* ctx) {
  auto* jit = (sage2::OCZrmJit*)ctx->method_data;
  delete jit;
}

int sage2_sgemm_oc_Zcm_jit_init(sage2_sgemm_ctx* ctx) {
  auto* jit = new (std::nothrow) sage2::OCZcmJit;
  if (jit == nullptr) {
    fprintf(stderr, "Failed to alloc memory.\n");
    return -1;
  }
  jit->Compile(ctx);
  ctx->method = METHOD_OC_ZCM_JIT;
  return 0;
}

void sage2_sgemm_oc_Zcm_jit_uninit(sage2_sgemm_ctx* ctx) {
  auto* jit = (sage2::OCZcmJit*)ctx->method_data;
  delete jit;
}
