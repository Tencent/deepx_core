// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include <memory>
#include "internal_macro.h"
#include "xbyak_wrapper.h"

namespace sage2 {
namespace {

using axpby_t = void (*)(uint64_t n, float alpha, const float* x, float beta,
                         float* y);

class AxpbyJitBase : public XbyakCodeGenerator {
 protected:
  const Reg64& N_REG = rdi;
  const Xmm& ALPHA_REG = xmm0;
  const Reg64& X_PTR = rsi;
  const Xmm& BETA_REG = xmm1;
  const Reg64& Y_PTR = rdx;
  const Reg64& I_REG = r8;
  const Reg64& M_REG = r9;

 protected:
  axpby_t func_ = nullptr;

 public:
  axpby_t func() const noexcept { return func_; }
};

class AxpbyJit1 : public AxpbyJitBase {
 private:
  void Compile1() {
    vmulss(xmm0, xmm14, ptr[X_PTR]);
    vfmadd231ss(xmm0, xmm15, ptr[Y_PTR]);
    vmovss(ptr[Y_PTR], xmm0);
  }

  void Compile2() {
    vmulss(xmm0, xmm14, ptr[X_PTR]);
    vmulss(xmm1, xmm14, ptr[X_PTR + 4]);
    vfmadd231ss(xmm0, xmm15, ptr[Y_PTR]);
    vfmadd231ss(xmm1, xmm15, ptr[Y_PTR + 4]);
    vmovss(ptr[Y_PTR], xmm0);
    vmovss(ptr[Y_PTR + 4], xmm1);
  }

  void Compile3() {
    vmulss(xmm0, xmm14, ptr[X_PTR]);
    vmulss(xmm1, xmm14, ptr[X_PTR + 4]);
    vmulss(xmm2, xmm14, ptr[X_PTR + 8]);
    vfmadd231ss(xmm0, xmm15, ptr[Y_PTR]);
    vfmadd231ss(xmm1, xmm15, ptr[Y_PTR + 4]);
    vfmadd231ss(xmm2, xmm15, ptr[Y_PTR + 8]);
    vmovss(ptr[Y_PTR], xmm0);
    vmovss(ptr[Y_PTR + 4], xmm1);
    vmovss(ptr[Y_PTR + 8], xmm2);
  }

  void Compile4() {
    vmulps(xmm0, xmm14, ptr[X_PTR]);
    vfmadd231ps(xmm0, xmm15, ptr[Y_PTR]);
    vmovups(ptr[Y_PTR], xmm0);
  }

  void Compile5() {
    vmulps(xmm0, xmm14, ptr[X_PTR]);
    vmulss(xmm1, xmm14, ptr[X_PTR + 16]);
    vfmadd231ps(xmm0, xmm15, ptr[Y_PTR]);
    vfmadd231ss(xmm1, xmm15, ptr[Y_PTR + 16]);
    vmovups(ptr[Y_PTR], xmm0);
    vmovss(ptr[Y_PTR + 16], xmm1);
  }

  void Compile6() {
    vmulps(xmm0, xmm14, ptr[X_PTR]);
    vmulss(xmm1, xmm14, ptr[X_PTR + 16]);
    vmulss(xmm2, xmm14, ptr[X_PTR + 20]);
    vfmadd231ps(xmm0, xmm15, ptr[Y_PTR]);
    vfmadd231ss(xmm1, xmm15, ptr[Y_PTR + 16]);
    vfmadd231ss(xmm2, xmm15, ptr[Y_PTR + 20]);
    vmovups(ptr[Y_PTR], xmm0);
    vmovss(ptr[Y_PTR + 16], xmm1);
    vmovss(ptr[Y_PTR + 20], xmm2);
  }

  void Compile7() {
    vmulps(xmm0, xmm14, ptr[X_PTR]);
    vmulss(xmm1, xmm14, ptr[X_PTR + 16]);
    vmulss(xmm2, xmm14, ptr[X_PTR + 20]);
    vmulss(xmm3, xmm14, ptr[X_PTR + 24]);
    vfmadd231ps(xmm0, xmm15, ptr[Y_PTR]);
    vfmadd231ss(xmm1, xmm15, ptr[Y_PTR + 16]);
    vfmadd231ss(xmm2, xmm15, ptr[Y_PTR + 20]);
    vfmadd231ss(xmm3, xmm15, ptr[Y_PTR + 24]);
    vmovups(ptr[Y_PTR], xmm0);
    vmovss(ptr[Y_PTR + 16], xmm1);
    vmovss(ptr[Y_PTR + 20], xmm2);
    vmovss(ptr[Y_PTR + 24], xmm3);
  }

  void Compile8() {
    vmulps(ymm0, ymm14, ptr[X_PTR]);
    vfmadd231ps(ymm0, ymm15, ptr[Y_PTR]);
    vmovups(ptr[Y_PTR], ymm0);
  }

  void Compile16() {
    vmulps(ymm0, ymm14, ptr[X_PTR]);
    vmulps(ymm1, ymm14, ptr[X_PTR + 32]);
    vfmadd231ps(ymm0, ymm15, ptr[Y_PTR]);
    vfmadd231ps(ymm1, ymm15, ptr[Y_PTR + 32]);
    vmovups(ptr[Y_PTR], ymm0);
    vmovups(ptr[Y_PTR + 32], ymm1);
  }

  void Compile32() {
    vmulps(ymm0, ymm14, ptr[X_PTR]);
    vmulps(ymm1, ymm14, ptr[X_PTR + 32]);
    vmulps(ymm2, ymm14, ptr[X_PTR + 64]);
    vmulps(ymm3, ymm14, ptr[X_PTR + 96]);
    vfmadd231ps(ymm0, ymm15, ptr[Y_PTR]);
    vfmadd231ps(ymm1, ymm15, ptr[Y_PTR + 32]);
    vfmadd231ps(ymm2, ymm15, ptr[Y_PTR + 64]);
    vfmadd231ps(ymm3, ymm15, ptr[Y_PTR + 96]);
    vmovups(ptr[Y_PTR], ymm0);
    vmovups(ptr[Y_PTR + 32], ymm1);
    vmovups(ptr[Y_PTR + 64], ymm2);
    vmovups(ptr[Y_PTR + 96], ymm3);
  }

  void Compile64() {
    vmulps(ymm0, ymm14, ptr[X_PTR]);
    vmulps(ymm1, ymm14, ptr[X_PTR + 32]);
    vmulps(ymm2, ymm14, ptr[X_PTR + 64]);
    vmulps(ymm3, ymm14, ptr[X_PTR + 96]);
    vmulps(ymm4, ymm14, ptr[X_PTR + 128]);
    vmulps(ymm5, ymm14, ptr[X_PTR + 160]);
    vmulps(ymm6, ymm14, ptr[X_PTR + 192]);
    vmulps(ymm7, ymm14, ptr[X_PTR + 224]);
    vfmadd231ps(ymm0, ymm15, ptr[Y_PTR]);
    vfmadd231ps(ymm1, ymm15, ptr[Y_PTR + 32]);
    vfmadd231ps(ymm2, ymm15, ptr[Y_PTR + 64]);
    vfmadd231ps(ymm3, ymm15, ptr[Y_PTR + 96]);
    vfmadd231ps(ymm4, ymm15, ptr[Y_PTR + 128]);
    vfmadd231ps(ymm5, ymm15, ptr[Y_PTR + 160]);
    vfmadd231ps(ymm6, ymm15, ptr[Y_PTR + 192]);
    vfmadd231ps(ymm7, ymm15, ptr[Y_PTR + 224]);
    vmovups(ptr[Y_PTR], ymm0);
    vmovups(ptr[Y_PTR + 32], ymm1);
    vmovups(ptr[Y_PTR + 64], ymm2);
    vmovups(ptr[Y_PTR + 96], ymm3);
    vmovups(ptr[Y_PTR + 128], ymm4);
    vmovups(ptr[Y_PTR + 160], ymm5);
    vmovups(ptr[Y_PTR + 192], ymm6);
    vmovups(ptr[Y_PTR + 224], ymm7);
  }

  void CompileFallback(uint64_t n) {
    uint64_t nn;
    uint64_t offset = 0;

    nn = n & -32;
    if (nn) {
      xor_(I_REG, I_REG);
      mov(M_REG, nn);
      L(".1");
      vmulps(ymm0, ymm14, ptr[X_PTR + I_REG * 4]);
      vmulps(ymm1, ymm14, ptr[X_PTR + I_REG * 4 + 32]);
      vmulps(ymm2, ymm14, ptr[X_PTR + I_REG * 4 + 64]);
      vmulps(ymm3, ymm14, ptr[X_PTR + I_REG * 4 + 96]);
      vfmadd231ps(ymm0, ymm15, ptr[Y_PTR + I_REG * 4]);
      vfmadd231ps(ymm1, ymm15, ptr[Y_PTR + I_REG * 4 + 32]);
      vfmadd231ps(ymm2, ymm15, ptr[Y_PTR + I_REG * 4 + 64]);
      vfmadd231ps(ymm3, ymm15, ptr[Y_PTR + I_REG * 4 + 96]);
      vmovups(ptr[Y_PTR + I_REG * 4], ymm0);
      vmovups(ptr[Y_PTR + I_REG * 4 + 32], ymm1);
      vmovups(ptr[Y_PTR + I_REG * 4 + 64], ymm2);
      vmovups(ptr[Y_PTR + I_REG * 4 + 96], ymm3);
      sub(I_REG, -32);
      sub(M_REG, 32);
      jne(".1", T_NEAR);
      offset += nn * 4;
    }

    nn = n & 16;
    if (nn) {
      vmulps(ymm0, ymm14, ptr[X_PTR + offset]);
      vmulps(ymm1, ymm14, ptr[X_PTR + offset + 32]);
      vfmadd231ps(ymm0, ymm15, ptr[Y_PTR + offset]);
      vfmadd231ps(ymm1, ymm15, ptr[Y_PTR + offset + 32]);
      vmovups(ptr[Y_PTR + offset], ymm0);
      vmovups(ptr[Y_PTR + offset + 32], ymm1);
      offset += 16 * 4;
    }

    nn = n & 8;
    if (nn) {
      vmulps(ymm0, ymm14, ptr[X_PTR + offset]);
      vfmadd231ps(ymm0, ymm15, ptr[Y_PTR + offset]);
      vmovups(ptr[Y_PTR + offset], ymm0);
      offset += 8 * 4;
    }

    nn = n & 4;
    if (nn) {
      vmulps(xmm0, xmm14, ptr[X_PTR + offset]);
      vfmadd231ps(xmm0, xmm15, ptr[Y_PTR + offset]);
      vmovups(ptr[Y_PTR + offset], xmm0);
      offset += 4 * 4;
    }

    nn = n & 3;
    switch (nn) {
      case 3:
        vmulss(xmm0, xmm14, ptr[X_PTR + offset]);
        vmulss(xmm1, xmm14, ptr[X_PTR + offset + 4]);
        vmulss(xmm2, xmm14, ptr[X_PTR + offset + 8]);
        vfmadd231ss(xmm0, xmm15, ptr[Y_PTR + offset]);
        vfmadd231ss(xmm1, xmm15, ptr[Y_PTR + offset + 4]);
        vfmadd231ss(xmm2, xmm15, ptr[Y_PTR + offset + 8]);
        vmovss(ptr[Y_PTR + offset], xmm0);
        vmovss(ptr[Y_PTR + offset + 4], xmm1);
        vmovss(ptr[Y_PTR + offset + 8], xmm2);
        break;
      case 2:
        vmulss(xmm0, xmm14, ptr[X_PTR + offset]);
        vmulss(xmm1, xmm14, ptr[X_PTR + offset + 4]);
        vfmadd231ss(xmm0, xmm15, ptr[Y_PTR + offset]);
        vfmadd231ss(xmm1, xmm15, ptr[Y_PTR + offset + 4]);
        vmovss(ptr[Y_PTR + offset], xmm0);
        vmovss(ptr[Y_PTR + offset + 4], xmm1);
        break;
      case 1:
        vmulss(xmm0, xmm14, ptr[X_PTR + offset]);
        vfmadd231ss(xmm0, xmm15, ptr[Y_PTR + offset]);
        vmovss(ptr[Y_PTR + offset], xmm0);
        break;
    }
  }

 public:
  void Compile(uint64_t n) {
    vmovss(ptr[rsp - 4], ALPHA_REG);
    vbroadcastss(ymm14, ptr[rsp - 4]);
    vmovss(ptr[rsp - 4], BETA_REG);
    vbroadcastss(ymm15, ptr[rsp - 4]);
    switch (n) {
      case 1:
        Compile1();
        break;
      case 2:
        Compile2();
        break;
      case 3:
        Compile3();
        break;
      case 4:
        Compile4();
        break;
      case 5:
        Compile5();
        break;
      case 6:
        Compile6();
        break;
      case 7:
        Compile7();
        break;
      case 8:
        Compile8();
        break;
      case 16:
        Compile16();
        break;
      case 32:
        Compile32();
        break;
      case 64:
        Compile64();
        break;
      default:
        CompileFallback(n);
        break;
    }
    vzeroupper();
    ret();
    func_ = getCode<axpby_t>();
  }
};

class AxpbyJit2 : public AxpbyJitBase {
 public:
  void Compile() {
    vmovss(ptr[rsp - 4], ALPHA_REG);
    vbroadcastss(ymm14, ptr[rsp - 4]);
    vmovss(ptr[rsp - 4], BETA_REG);
    vbroadcastss(ymm15, ptr[rsp - 4]);

    xor_(I_REG, I_REG);
    mov(M_REG, N_REG);
    and_(M_REG, -32);
    je(".2", T_NEAR);

    L(".1");
    vmulps(ymm0, ymm14, ptr[X_PTR + I_REG * 4]);
    vmulps(ymm1, ymm14, ptr[X_PTR + I_REG * 4 + 32]);
    vmulps(ymm2, ymm14, ptr[X_PTR + I_REG * 4 + 64]);
    vmulps(ymm3, ymm14, ptr[X_PTR + I_REG * 4 + 96]);
    vfmadd231ps(ymm0, ymm15, ptr[Y_PTR + I_REG * 4]);
    vfmadd231ps(ymm1, ymm15, ptr[Y_PTR + I_REG * 4 + 32]);
    vfmadd231ps(ymm2, ymm15, ptr[Y_PTR + I_REG * 4 + 64]);
    vfmadd231ps(ymm3, ymm15, ptr[Y_PTR + I_REG * 4 + 96]);
    vmovups(ptr[Y_PTR + I_REG * 4], ymm0);
    vmovups(ptr[Y_PTR + I_REG * 4 + 32], ymm1);
    vmovups(ptr[Y_PTR + I_REG * 4 + 64], ymm2);
    vmovups(ptr[Y_PTR + I_REG * 4 + 96], ymm3);
    sub(I_REG, -32);
    sub(M_REG, 32);
    jne(".1", T_NEAR);

    L(".2");
    mov(M_REG, N_REG);
    and_(M_REG, 16);
    je(".3", T_NEAR);
    vmulps(ymm0, ymm14, ptr[X_PTR + I_REG * 4]);
    vmulps(ymm1, ymm14, ptr[X_PTR + I_REG * 4 + 32]);
    vfmadd231ps(ymm0, ymm15, ptr[Y_PTR + I_REG * 4]);
    vfmadd231ps(ymm1, ymm15, ptr[Y_PTR + I_REG * 4 + 32]);
    vmovups(ptr[Y_PTR + I_REG * 4], ymm0);
    vmovups(ptr[Y_PTR + I_REG * 4 + 32], ymm1);
    sub(I_REG, -16);

    L(".3");
    mov(M_REG, N_REG);
    and_(M_REG, 8);
    je(".10", T_NEAR);
    vmulps(ymm0, ymm14, ptr[X_PTR + I_REG * 4]);
    vfmadd231ps(ymm0, ymm15, ptr[Y_PTR + I_REG * 4]);
    vmovups(ptr[Y_PTR + I_REG * 4], ymm0);
    sub(I_REG, -8);

    L(".10");
    mov(M_REG, N_REG);
    and_(M_REG, 7);
    je(".12", T_NEAR);

    L(".11");
    vmulss(xmm0, xmm14, ptr[X_PTR + I_REG * 4]);
    vfmadd231ss(xmm0, xmm15, ptr[Y_PTR + I_REG * 4]);
    vmovss(ptr[Y_PTR + I_REG * 4], xmm0);
    sub(I_REG, -1);
    sub(M_REG, 1);
    jne(".11", T_NEAR);

    L(".12");
    vzeroupper();
    ret();
    func_ = getCode<axpby_t>();
  }
};

constexpr uint64_t JIT_MAX_N = 256;
std::unique_ptr<AxpbyJit1> jit1[JIT_MAX_N + 1];
axpby_t jit1_func[JIT_MAX_N + 1];
std::unique_ptr<AxpbyJit2> jit2;
axpby_t jit2_func;

ATTR_CTOR(110) void init() {
  for (uint64_t i = 0; i < JIT_MAX_N + 1; ++i) {
    jit1[i].reset(new AxpbyJit1);
    jit1[i]->Compile(i);
    jit1_func[i] = jit1[i]->func();
  }
  jit2.reset(new AxpbyJit2);
  jit2->Compile();
  jit2_func = jit2->func();
}

}  // namespace
}  // namespace sage2

void sage2_axpby_ps(uint64_t n, float alpha, const float* x, float beta,
                    float* y) {
  if (n <= sage2::JIT_MAX_N) {
    sage2::jit1_func[n](n, alpha, x, beta, y);
    return;
  }
  sage2::jit2_func(n, alpha, x, beta, y);
}
