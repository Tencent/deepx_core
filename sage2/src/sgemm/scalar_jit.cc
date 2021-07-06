// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include "sgemm/internal_sgemm.h"

namespace sage2 {
namespace {

class ScalarJit : public XbyakCodeGenerator {
 private:
  const Reg64& CTX_PTR = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& Y_PTR = rdx;
  const Reg64& Z_PTR = rcx;

 public:
  void Compile(sage2_sgemm_ctx* ctx) {
    vmovss(xmm0, ptr[X_PTR]);
    vmulss(xmm0, xmm0, ptr[Y_PTR]);

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

int sage2_sgemm_scalar_jit_init(sage2_sgemm_ctx* ctx) {
  auto* jit = new (std::nothrow) sage2::ScalarJit;
  if (jit == nullptr) {
    fprintf(stderr, "Failed to alloc memory.\n");
    return -1;
  }
  jit->Compile(ctx);
  ctx->method = METHOD_SCALAR_JIT;
  return 0;
}

void sage2_sgemm_scalar_jit_uninit(sage2_sgemm_ctx* ctx) {
  auto* jit = (sage2::ScalarJit*)ctx->method_data;
  delete jit;
}
