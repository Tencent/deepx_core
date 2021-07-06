// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

namespace sage2 {
namespace {

/************************************************************************/
/* A0B0ZcJit */
/************************************************************************/
class A0B0ZcJit : public XbyakCodeGenerator {
 private:
  const Reg64& Z_PTR = rcx;
  const Reg64& I_REG = r8;

 private:
  void Compile(int n) {
    int nn;
    int offset = 0;

    nn = n & -64;
    if (nn) {
      mov(I_REG, nn);
      neg(I_REG);
      add(Z_PTR, nn * 4);
      L(".1");
      vmovups(ptr[Z_PTR + I_REG * 4], ymm0);
      vmovups(ptr[Z_PTR + I_REG * 4 + 32], ymm0);
      vmovups(ptr[Z_PTR + I_REG * 4 + 64], ymm0);
      vmovups(ptr[Z_PTR + I_REG * 4 + 96], ymm0);
      vmovups(ptr[Z_PTR + I_REG * 4 + 128], ymm0);
      vmovups(ptr[Z_PTR + I_REG * 4 + 160], ymm0);
      vmovups(ptr[Z_PTR + I_REG * 4 + 192], ymm0);
      vmovups(ptr[Z_PTR + I_REG * 4 + 224], ymm0);
      add(I_REG, 64);
      jne(".1", T_NEAR);
    }

    nn = n & 32;
    if (nn) {
      vmovups(ptr[Z_PTR + offset], ymm0);
      vmovups(ptr[Z_PTR + offset + 32], ymm0);
      vmovups(ptr[Z_PTR + offset + 64], ymm0);
      vmovups(ptr[Z_PTR + offset + 96], ymm0);
      offset += 32 * 4;
    }

    nn = n & 16;
    if (nn) {
      vmovups(ptr[Z_PTR + offset], ymm0);
      vmovups(ptr[Z_PTR + offset + 32], ymm0);
      offset += 16 * 4;
    }

    nn = n & 8;
    if (nn) {
      vmovups(ptr[Z_PTR + offset], ymm0);
      offset += 8 * 4;
    }

    nn = n & 4;
    if (nn) {
      vmovups(ptr[Z_PTR + offset], xmm0);
      offset += 4 * 4;
    }

    nn = n & 3;
    switch (nn) {
      case 3:
        vmovss(ptr[Z_PTR + offset], xmm0);
        vmovss(ptr[Z_PTR + offset + 4], xmm0);
        vmovss(ptr[Z_PTR + offset + 8], xmm0);
        break;
      case 2:
        vmovss(ptr[Z_PTR + offset], xmm0);
        vmovss(ptr[Z_PTR + offset + 4], xmm0);
        break;
      case 1:
        vmovss(ptr[Z_PTR + offset], xmm0);
        break;
    }
  }

 public:
  void Compile(sage2_sgemm_ctx* ctx) {
    vxorps(ymm0, ymm0, ymm0);
    Compile(ctx->m * ctx->n);
    vzeroupper();
    ret();

    ctx->func = getCode<_sage2_sgemm_t>();
    ctx->method_data = this;
  }
};

/************************************************************************/
/* A0ZcJit */
/************************************************************************/
class A0ZcJit : public XbyakCodeGenerator {
 private:
  const Reg64& CTX_PTR = rdi;
  const Reg64& Z_PTR = rcx;
  const Reg64& I_REG = r8;

 private:
  void Compile(int n) {
    int nn;
    int offset = 0;

    nn = n & -32;
    if (nn) {
      mov(I_REG, nn);
      neg(I_REG);
      add(Z_PTR, nn * 4);
      L(".1");
      vmulps(ymm1, ymm0, ptr[Z_PTR + I_REG * 4]);
      vmulps(ymm2, ymm0, ptr[Z_PTR + I_REG * 4 + 32]);
      vmulps(ymm3, ymm0, ptr[Z_PTR + I_REG * 4 + 64]);
      vmulps(ymm4, ymm0, ptr[Z_PTR + I_REG * 4 + 96]);
      vmovups(ptr[Z_PTR + I_REG * 4], ymm1);
      vmovups(ptr[Z_PTR + I_REG * 4 + 32], ymm2);
      vmovups(ptr[Z_PTR + I_REG * 4 + 64], ymm3);
      vmovups(ptr[Z_PTR + I_REG * 4 + 96], ymm4);
      add(I_REG, 32);
      jne(".1", T_NEAR);
    }

    nn = n & 16;
    if (nn) {
      vmulps(ymm1, ymm0, ptr[Z_PTR + offset]);
      vmulps(ymm2, ymm0, ptr[Z_PTR + offset + 32]);
      vmovups(ptr[Z_PTR + offset], ymm1);
      vmovups(ptr[Z_PTR + offset + 32], ymm2);
      offset += 16 * 4;
    }

    nn = n & 8;
    if (nn) {
      vmulps(ymm1, ymm0, ptr[Z_PTR + offset]);
      vmovups(ptr[Z_PTR + offset], ymm1);
      offset += 8 * 4;
    }

    nn = n & 4;
    if (nn) {
      vmulps(xmm1, xmm0, ptr[Z_PTR + offset]);
      vmovups(ptr[Z_PTR + offset], xmm1);
      offset += 4 * 4;
    }

    nn = n & 3;
    switch (nn) {
      case 3:
        vmulss(xmm1, xmm0, ptr[Z_PTR + offset]);
        vmulss(xmm2, xmm0, ptr[Z_PTR + offset + 4]);
        vmulss(xmm3, xmm0, ptr[Z_PTR + offset + 8]);
        vmovss(ptr[Z_PTR + offset], xmm1);
        vmovss(ptr[Z_PTR + offset + 4], xmm2);
        vmovss(ptr[Z_PTR + offset + 8], xmm3);
        break;
      case 2:
        vmulss(xmm1, xmm0, ptr[Z_PTR + offset]);
        vmulss(xmm2, xmm0, ptr[Z_PTR + offset + 4]);
        vmovss(ptr[Z_PTR + offset], xmm1);
        vmovss(ptr[Z_PTR + offset + 4], xmm2);
        break;
      case 1:
        vmulss(xmm1, xmm0, ptr[Z_PTR + offset]);
        vmovss(ptr[Z_PTR + offset], xmm1);
        break;
    }
  }

 public:
  void Compile(sage2_sgemm_ctx* ctx) {
    vbroadcastss(ymm0, ptr[CTX_PTR + BETA_OFFSET]);
    Compile(ctx->m * ctx->n);
    vzeroupper();
    ret();

    ctx->func = getCode<_sage2_sgemm_t>();
    ctx->method_data = this;
  }
};

}  // namespace
}  // namespace sage2

int sage2_sgemm_a0_b0_Zc_jit_init(sage2_sgemm_ctx* ctx) {
  auto* jit = new (std::nothrow) sage2::A0B0ZcJit;
  if (jit == nullptr) {
    fprintf(stderr, "Failed to alloc memory.\n");
    return -1;
  }
  jit->Compile(ctx);
  ctx->method = METHOD_A0_B0_ZCONT_JIT;
  return 0;
}

void sage2_sgemm_a0_b0_Zc_jit_uninit(sage2_sgemm_ctx* ctx) {
  auto* jit = (sage2::A0B0ZcJit*)ctx->method_data;
  delete jit;
}

int sage2_sgemm_a0_Zc_jit_init(sage2_sgemm_ctx* ctx) {
  auto* jit = new (std::nothrow) sage2::A0ZcJit;
  if (jit == nullptr) {
    fprintf(stderr, "Failed to alloc memory.\n");
    return -1;
  }
  jit->Compile(ctx);
  ctx->method = METHOD_A0_ZCONT_JIT;
  return 0;
}

void sage2_sgemm_a0_Zc_jit_uninit(sage2_sgemm_ctx* ctx) {
  auto* jit = (sage2::A0ZcJit*)ctx->method_data;
  delete jit;
}
