// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include <memory>
#include "internal_macro.h"
#include "xbyak_wrapper.h"

namespace sage2 {
namespace {

using dot_t = float (*)(uint64_t n, const float* x, const float* y);

class DotJitBase : public XbyakCodeGenerator {
 protected:
  const Reg64& N_REG = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& Y_PTR = rdx;
  const Reg64& I_REG = r8;
  const Reg64& M_REG = r9;

 protected:
  dot_t func_ = nullptr;

 public:
  dot_t func() const noexcept { return func_; }
};

class DotJit1 : public DotJitBase {
 private:
  void Compile1() {
    vmovss(xmm0, ptr[X_PTR]);
    vmulss(xmm0, xmm0, ptr[Y_PTR]);
  }

  void Compile2() {
    vmovss(xmm0, ptr[X_PTR]);
    vmovss(xmm1, ptr[X_PTR + 4]);
    vmulss(xmm0, xmm0, ptr[Y_PTR]);
    vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + 4]);
  }

  void Compile3() {
    vmovss(xmm0, ptr[X_PTR]);
    vmovss(xmm1, ptr[X_PTR + 4]);
    vmovss(xmm2, ptr[X_PTR + 8]);
    vmulss(xmm0, xmm0, ptr[Y_PTR]);
    vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + 4]);
    vfmadd231ss(xmm0, xmm2, ptr[Y_PTR + 8]);
  }

  void Compile4() {
    vmovups(xmm0, ptr[X_PTR]);
    vmulps(xmm0, xmm0, ptr[Y_PTR]);
    SumXmm0();
  }

  void Compile5() {
    vmovups(xmm0, ptr[X_PTR]);
    vmovss(xmm1, ptr[X_PTR + 16]);
    vmulps(xmm0, xmm0, ptr[Y_PTR]);
    SumXmm0();
    vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + 16]);
  }

  void Compile6() {
    vmovups(xmm0, ptr[X_PTR]);
    vmovss(xmm1, ptr[X_PTR + 16]);
    vmovss(xmm2, ptr[X_PTR + 20]);
    vmulps(xmm0, xmm0, ptr[Y_PTR]);
    SumXmm0();
    vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + 16]);
    vfmadd231ss(xmm0, xmm2, ptr[Y_PTR + 20]);
  }

  void Compile7() {
    vmovups(xmm0, ptr[X_PTR]);
    vmovss(xmm1, ptr[X_PTR + 16]);
    vmovss(xmm2, ptr[X_PTR + 20]);
    vmovss(xmm3, ptr[X_PTR + 24]);
    vmulps(xmm0, xmm0, ptr[Y_PTR]);
    SumXmm0();
    vfmadd231ss(xmm0, xmm1, ptr[Y_PTR + 16]);
    vfmadd231ss(xmm0, xmm2, ptr[Y_PTR + 20]);
    vfmadd231ss(xmm0, xmm3, ptr[Y_PTR + 24]);
  }

  void Compile8() {
    vmovups(ymm0, ptr[X_PTR]);
    vmulps(ymm0, ymm0, ptr[Y_PTR]);
    SumYmm0();
  }

  void Compile16() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    vmulps(ymm0, ymm0, ptr[Y_PTR]);
    vfmadd231ps(ymm0, ymm1, ptr[Y_PTR + 32]);
    SumYmm0();
  }

  void Compile32() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    vmovups(ymm2, ptr[X_PTR + 64]);
    vmovups(ymm3, ptr[X_PTR + 96]);
    vmulps(ymm0, ymm0, ptr[Y_PTR]);
    vmulps(ymm1, ymm1, ptr[Y_PTR + 32]);
    vfmadd231ps(ymm0, ymm2, ptr[Y_PTR + 64]);
    vfmadd231ps(ymm1, ymm3, ptr[Y_PTR + 96]);
    vaddps(ymm0, ymm0, ymm1);
    SumYmm0();
  }

  void Compile64() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    vmovups(ymm2, ptr[X_PTR + 64]);
    vmovups(ymm3, ptr[X_PTR + 96]);
    vmovups(ymm4, ptr[X_PTR + 128]);
    vmovups(ymm5, ptr[X_PTR + 160]);
    vmovups(ymm6, ptr[X_PTR + 192]);
    vmovups(ymm7, ptr[X_PTR + 224]);
    vmulps(ymm0, ymm0, ptr[Y_PTR]);
    vmulps(ymm1, ymm1, ptr[Y_PTR + 32]);
    vmulps(ymm2, ymm2, ptr[Y_PTR + 64]);
    vmulps(ymm3, ymm3, ptr[Y_PTR + 96]);
    vfmadd231ps(ymm0, ymm4, ptr[Y_PTR + 128]);
    vfmadd231ps(ymm1, ymm5, ptr[Y_PTR + 160]);
    vfmadd231ps(ymm2, ymm6, ptr[Y_PTR + 192]);
    vfmadd231ps(ymm3, ymm7, ptr[Y_PTR + 224]);
    vaddps(ymm0, ymm0, ymm1);
    vaddps(ymm2, ymm2, ymm3);
    vaddps(ymm0, ymm0, ymm2);
    SumYmm0();
  }

  void CompileFallback(uint64_t n) {
    uint64_t nn;
    uint64_t offset = 0;

    if (n >= 16) {
      vxorps(ymm0, ymm0, ymm0);
      vxorps(ymm1, ymm1, ymm1);
    } else {
      vxorps(ymm0, ymm0, ymm0);
    }

    nn = n & -32;
    if (nn) {
      vxorps(ymm2, ymm2, ymm2);
      vxorps(ymm3, ymm3, ymm3);
      xor_(I_REG, I_REG);
      mov(M_REG, nn);
      L(".1");
      vmovups(ymm4, ptr[X_PTR + I_REG * 4]);
      vmovups(ymm5, ptr[X_PTR + I_REG * 4 + 32]);
      vmovups(ymm6, ptr[X_PTR + I_REG * 4 + 64]);
      vmovups(ymm7, ptr[X_PTR + I_REG * 4 + 96]);
      vfmadd231ps(ymm0, ymm4, ptr[Y_PTR + I_REG * 4]);
      vfmadd231ps(ymm1, ymm5, ptr[Y_PTR + I_REG * 4 + 32]);
      vfmadd231ps(ymm2, ymm6, ptr[Y_PTR + I_REG * 4 + 64]);
      vfmadd231ps(ymm3, ymm7, ptr[Y_PTR + I_REG * 4 + 96]);
      sub(I_REG, -32);
      sub(M_REG, 32);
      jne(".1", T_NEAR);
      vaddps(ymm0, ymm0, ymm2);
      vaddps(ymm1, ymm1, ymm3);
      offset += nn * 4;
    }

    nn = n & 16;
    if (nn) {
      vmovups(ymm4, ptr[X_PTR + offset]);
      vmovups(ymm5, ptr[X_PTR + offset + 32]);
      vfmadd231ps(ymm0, ymm4, ptr[Y_PTR + offset]);
      vfmadd231ps(ymm1, ymm5, ptr[Y_PTR + offset + 32]);
      offset += 16 * 4;
    }

    if (n >= 16) {
      vaddps(ymm0, ymm0, ymm1);
    }

    nn = n & 8;
    if (nn) {
      vmovups(ymm4, ptr[X_PTR + offset]);
      vfmadd231ps(ymm0, ymm4, ptr[Y_PTR + offset]);
      offset += 8 * 4;
    }

    if (n >= 8) {
      vextractf128(xmm1, ymm0, 1);
      vaddps(xmm0, xmm0, xmm1);
    }

    nn = n & 4;
    if (nn) {
      vmovups(xmm1, ptr[X_PTR + offset]);
      vfmadd231ps(xmm0, xmm1, ptr[Y_PTR + offset]);
      offset += 4 * 4;
    }

    if (n >= 4) {
      SumXmm0();
    }

    nn = n & 3;
    switch (nn) {
      case 3:
        vmovss(xmm4, ptr[X_PTR + offset]);
        vmovss(xmm5, ptr[X_PTR + offset + 4]);
        vmovss(xmm6, ptr[X_PTR + offset + 8]);
        vfmadd231ss(xmm0, xmm4, ptr[Y_PTR + offset]);
        vfmadd231ss(xmm0, xmm5, ptr[Y_PTR + offset + 4]);
        vfmadd231ss(xmm0, xmm6, ptr[Y_PTR + offset + 8]);
        break;
      case 2:
        vmovss(xmm4, ptr[X_PTR + offset]);
        vmovss(xmm5, ptr[X_PTR + offset + 4]);
        vfmadd231ss(xmm0, xmm4, ptr[Y_PTR + offset]);
        vfmadd231ss(xmm0, xmm5, ptr[Y_PTR + offset + 4]);
        break;
      case 1:
        vmovss(xmm4, ptr[X_PTR + offset]);
        vfmadd231ss(xmm0, xmm4, ptr[Y_PTR + offset]);
        break;
    }
  }

 public:
  void Compile(uint64_t n) {
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
    func_ = getCode<dot_t>();
  }
};

class DotJit2 : public DotJitBase {
 public:
  void Compile() {
    vxorps(ymm0, ymm0, ymm0);
    vxorps(ymm1, ymm1, ymm1);

    xor_(I_REG, I_REG);
    mov(M_REG, N_REG);
    and_(M_REG, -32);
    je(".2", T_NEAR);

    vxorps(ymm2, ymm2, ymm2);
    vxorps(ymm3, ymm3, ymm3);
    L(".1");
    vmovups(ymm4, ptr[X_PTR + I_REG * 4]);
    vmovups(ymm5, ptr[X_PTR + I_REG * 4 + 32]);
    vmovups(ymm6, ptr[X_PTR + I_REG * 4 + 64]);
    vmovups(ymm7, ptr[X_PTR + I_REG * 4 + 96]);
    vfmadd231ps(ymm0, ymm4, ptr[Y_PTR + I_REG * 4]);
    vfmadd231ps(ymm1, ymm5, ptr[Y_PTR + I_REG * 4 + 32]);
    vfmadd231ps(ymm2, ymm6, ptr[Y_PTR + I_REG * 4 + 64]);
    vfmadd231ps(ymm3, ymm7, ptr[Y_PTR + I_REG * 4 + 96]);
    sub(I_REG, -32);
    sub(M_REG, 32);
    jne(".1", T_NEAR);
    vaddps(ymm0, ymm0, ymm2);
    vaddps(ymm1, ymm1, ymm3);

    L(".2");
    mov(M_REG, N_REG);
    and_(M_REG, 16);
    je(".3", T_NEAR);
    vmovups(ymm4, ptr[X_PTR + I_REG * 4]);
    vmovups(ymm5, ptr[X_PTR + I_REG * 4 + 32]);
    vfmadd231ps(ymm0, ymm4, ptr[Y_PTR + I_REG * 4]);
    vfmadd231ps(ymm1, ymm5, ptr[Y_PTR + I_REG * 4 + 32]);
    sub(I_REG, -16);

    L(".3");
    mov(M_REG, N_REG);
    and_(M_REG, 8);
    je(".10", T_NEAR);
    vmovups(ymm4, ptr[X_PTR + I_REG * 4]);
    vfmadd231ps(ymm0, ymm4, ptr[Y_PTR + I_REG * 4]);
    sub(I_REG, -8);

    L(".10");
    vaddps(ymm0, ymm0, ymm1);
    SumYmm0();
    mov(M_REG, N_REG);
    and_(M_REG, 7);
    je(".12", T_NEAR);

    L(".11");
    vmovss(xmm4, ptr[X_PTR + I_REG * 4]);
    vfmadd231ss(xmm0, xmm4, ptr[Y_PTR + I_REG * 4]);
    sub(I_REG, -1);
    sub(M_REG, 1);
    jne(".11", T_NEAR);

    L(".12");
    vzeroupper();
    ret();
    func_ = getCode<dot_t>();
  }
};

constexpr uint64_t JIT_MAX_N = 256;
std::unique_ptr<DotJit1> jit1[JIT_MAX_N + 1];
dot_t jit1_func[JIT_MAX_N + 1];
std::unique_ptr<DotJit2> jit2;
dot_t jit2_func;

ATTR_CTOR(110) void init() {
  for (uint64_t i = 0; i < JIT_MAX_N + 1; ++i) {
    jit1[i].reset(new DotJit1);
    jit1[i]->Compile(i);
    jit1_func[i] = jit1[i]->func();
  }
  jit2.reset(new DotJit2);
  jit2->Compile();
  jit2_func = jit2->func();
}

}  // namespace
}  // namespace sage2

float sage2_dot_ps(uint64_t n, const float* x, const float* y) {
  if (n <= sage2::JIT_MAX_N) {
    return sage2::jit1_func[n](n, x, y);
  }
  return sage2::jit2_func(n, x, y);
}
