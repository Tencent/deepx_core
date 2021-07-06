// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

namespace sage2 {
namespace {

/************************************************************************/
/* ICJit */
/************************************************************************/
class ICJit : public XbyakCodeGenerator {
 private:
  const Reg64& CTX_PTR = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& Y_PTR = rdx;
  const Reg64& Z_PTR = rcx;
  const Reg64& I_REG = r8;
  const Reg64& K_REG = r9;

 private:
  void CompileDot16() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    vmulps(ymm0, ymm0, ptr[Y_PTR]);
    vmulps(ymm1, ymm1, ptr[Y_PTR + 32]);

    vaddps(ymm0, ymm1, ymm0);
    vextractf128(xmm1, ymm0, 1);
    vaddps(xmm0, xmm1, xmm0);
    vhaddps(xmm0, xmm0, xmm1);
    vhaddps(xmm0, xmm0, xmm1);
  }

  void CompileDotFallback(int k) {
    int kk;
    int offset = 0;
    int need_vextractf128 = 0;
    int need_vhaddps = 0;

    vxorps(ymm0, ymm0, ymm0);

    kk = k & -16;
    if (kk) {
      vxorps(ymm1, ymm1, ymm1);
      xor_(I_REG, I_REG);
      mov(K_REG, kk);
      L(".1");
      vmovups(ymm2, ptr[X_PTR + I_REG * 4]);
      vmovups(ymm3, ptr[X_PTR + I_REG * 4 + 32]);
      vfmadd231ps(ymm0, ymm2, ptr[Y_PTR + I_REG * 4]);
      vfmadd231ps(ymm1, ymm3, ptr[Y_PTR + I_REG * 4 + 32]);
      add(I_REG, 16);
      sub(K_REG, 16);
      jne(".1", T_NEAR);
      vaddps(ymm0, ymm1, ymm0);
      offset += kk * 4;
      need_vextractf128 = 1;
      need_vhaddps = 1;
    }

    kk = k & 8;
    if (kk) {
      vmovups(ymm1, ptr[X_PTR + offset]);
      vfmadd231ps(ymm0, ymm1, ptr[Y_PTR + offset]);
      offset += 8 * 4;
      need_vextractf128 = 1;
      need_vhaddps = 1;
    }

    if (need_vextractf128) {
      vextractf128(xmm1, ymm0, 1);
      vaddps(xmm0, xmm1, xmm0);
    }

    kk = k & 4;
    if (kk) {
      vmovups(xmm1, ptr[X_PTR + offset]);
      vfmadd231ps(xmm0, xmm1, ptr[Y_PTR + offset]);
      offset += 4 * 4;
      need_vhaddps = 1;
    }

    if (need_vhaddps) {
      vhaddps(xmm0, xmm0, xmm1);
      vhaddps(xmm0, xmm0, xmm1);
    }

    kk = k & 3;
    switch (kk) {
      case 3:
        vmovss(xmm1, ptr[X_PTR + offset]);
        vmovss(xmm2, ptr[X_PTR + offset + 4]);
        vmovss(xmm3, ptr[X_PTR + offset + 8]);
        vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + offset]);
        vfmadd231ss(xmm0, xmm2, ptr[Y_PTR + offset + 4]);
        vfmadd231ss(xmm0, xmm3, ptr[Y_PTR + offset + 8]);
        break;
      case 2:
        vmovss(xmm1, ptr[X_PTR + offset]);
        vmovss(xmm2, ptr[X_PTR + offset + 4]);
        vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + offset]);
        vfmadd231ss(xmm0, xmm2, ptr[Y_PTR + offset + 4]);
        break;
      case 1:
        vmovss(xmm1, ptr[X_PTR + offset]);
        vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + offset]);
        break;
    }
  }

 public:
  void Compile(sage2_sgemm_ctx* ctx) {
    int k = ctx->k;
    if (k == 16) {
      CompileDot16();
    } else {
      CompileDotFallback(k);
    }

    if (ctx->alpha != 1) {
      vmulss(xmm0, xmm0, ptr[CTX_PTR + ALPHA_OFFSET]);
    }

    if (ctx->beta == 0) {
    } else if (ctx->beta == 1) {
      vaddss(xmm0, xmm0, ptr[Z_PTR]);
    } else {
      vmovss(xmm1, ptr[CTX_PTR + BETA_OFFSET]);
      vfmadd231ss(xmm0, xmm1, ptr[Z_PTR]);
    }

    vmovss(ptr[Z_PTR], xmm0);

    vzeroupper();
    ret();

    ctx->func = getCode<_sage2_sgemm_t>();
    ctx->method_data = this;
  }
};

}  // namespace
}  // namespace sage2

int sage2_sgemm_ic_jit_init(sage2_sgemm_ctx* ctx) {
  auto* jit = new (std::nothrow) sage2::ICJit;
  if (jit == nullptr) {
    fprintf(stderr, "Failed to alloc memory.\n");
    return -1;
  }
  jit->Compile(ctx);
  ctx->method = METHOD_IC_JIT;
  return 0;
}

void sage2_sgemm_ic_jit_uninit(sage2_sgemm_ctx* ctx) {
  auto* jit = (sage2::ICJit*)ctx->method_data;
  delete jit;
}
