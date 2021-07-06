// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include <memory>
#include "internal_macro.h"
#include "xbyak_wrapper.h"

namespace sage2 {
namespace {

using sum_t = float (*)(uint64_t n, const float* x);

class SumJitBase : public XbyakCodeGenerator {
 protected:
  const Reg64& N_REG = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& I_REG = r8;
  const Reg64& M_REG = r9;

 protected:
  sum_t func_ = nullptr;

 public:
  sum_t func() const noexcept { return func_; }
};

class SumJit1 : public SumJitBase {
 private:
  void Compile1() { vmovss(xmm0, ptr[X_PTR]); }

  void Compile2() {
    vmovss(xmm0, ptr[X_PTR]);
    vaddss(xmm0, ptr[X_PTR + 4]);
  }

  void Compile3() {
    vmovss(xmm0, ptr[X_PTR]);
    vaddss(xmm0, ptr[X_PTR + 4]);
    vaddss(xmm0, ptr[X_PTR + 8]);
  }

  void Compile4() {
    vmovups(xmm0, ptr[X_PTR]);
    SumXmm0();
  }

  void Compile5() {
    vmovups(xmm0, ptr[X_PTR]);
    SumXmm0();
    vaddss(xmm0, xmm0, ptr[X_PTR + 16]);
  }

  void Compile6() {
    vmovups(xmm0, ptr[X_PTR]);
    SumXmm0();
    vaddss(xmm0, xmm0, ptr[X_PTR + 16]);
    vaddss(xmm0, xmm0, ptr[X_PTR + 20]);
  }

  void Compile7() {
    vmovups(xmm0, ptr[X_PTR]);
    SumXmm0();
    vaddss(xmm0, xmm0, ptr[X_PTR + 16]);
    vaddss(xmm0, xmm0, ptr[X_PTR + 20]);
    vaddss(xmm0, xmm0, ptr[X_PTR + 24]);
  }

  void Compile8() {
    vmovups(ymm0, ptr[X_PTR]);
    SumYmm0();
  }

  void Compile16() {
    vmovups(ymm0, ptr[X_PTR]);
    vaddps(ymm0, ymm0, ptr[X_PTR + 32]);
    SumYmm0();
  }

  void Compile32() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    vaddps(ymm0, ymm0, ptr[X_PTR + 64]);
    vaddps(ymm1, ymm1, ptr[X_PTR + 96]);
    vaddps(ymm0, ymm0, ymm1);
    SumYmm0();
  }

  void Compile64() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    vmovups(ymm2, ptr[X_PTR + 64]);
    vmovups(ymm3, ptr[X_PTR + 96]);
    vaddps(ymm0, ptr[X_PTR + 128]);
    vaddps(ymm1, ptr[X_PTR + 160]);
    vaddps(ymm2, ptr[X_PTR + 192]);
    vaddps(ymm3, ptr[X_PTR + 224]);
    vaddps(ymm0, ymm0, ymm1);
    vaddps(ymm0, ymm0, ymm2);
    vaddps(ymm0, ymm0, ymm3);
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
      vaddps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
      vaddps(ymm1, ymm1, ptr[X_PTR + I_REG * 4 + 32]);
      vaddps(ymm2, ymm2, ptr[X_PTR + I_REG * 4 + 64]);
      vaddps(ymm3, ymm3, ptr[X_PTR + I_REG * 4 + 96]);
      sub(I_REG, -32);
      sub(M_REG, 32);
      jne(".1", T_NEAR);
      vaddps(ymm0, ymm0, ymm2);
      vaddps(ymm1, ymm1, ymm3);
      offset += nn * 4;
    }

    nn = n & 16;
    if (nn) {
      vaddps(ymm0, ymm0, ptr[X_PTR + offset]);
      vaddps(ymm1, ymm1, ptr[X_PTR + offset + 32]);
      offset += 16 * 4;
    }

    if (n >= 16) {
      vaddps(ymm0, ymm0, ymm1);
    }

    nn = n & 8;
    if (nn) {
      vaddps(ymm0, ymm0, ptr[X_PTR + offset]);
      offset += 8 * 4;
    }

    if (n >= 8) {
      vextractf128(xmm1, ymm0, 1);
      vaddps(xmm0, xmm0, xmm1);
    }

    nn = n & 4;
    if (nn) {
      vaddps(xmm0, xmm0, ptr[X_PTR + offset]);
      offset += 4 * 4;
    }

    if (n >= 4) {
      SumXmm0();
    }

    nn = n & 3;
    switch (nn) {
      case 3:
        vaddss(xmm0, xmm0, ptr[X_PTR + offset]);
        vaddss(xmm0, xmm0, ptr[X_PTR + offset + 4]);
        vaddss(xmm0, xmm0, ptr[X_PTR + offset + 8]);
        break;
      case 2:
        vaddss(xmm0, xmm0, ptr[X_PTR + offset]);
        vaddss(xmm0, xmm0, ptr[X_PTR + offset + 4]);
        break;
      case 1:
        vaddss(xmm0, xmm0, ptr[X_PTR + offset]);
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
    func_ = getCode<sum_t>();
  }
};

class SumJit2 : public SumJitBase {
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
    vaddps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
    vaddps(ymm1, ymm1, ptr[X_PTR + I_REG * 4 + 32]);
    vaddps(ymm2, ymm2, ptr[X_PTR + I_REG * 4 + 64]);
    vaddps(ymm3, ymm3, ptr[X_PTR + I_REG * 4 + 96]);
    sub(I_REG, -32);
    sub(M_REG, 32);
    jne(".1", T_NEAR);
    vaddps(ymm0, ymm0, ymm2);
    vaddps(ymm1, ymm1, ymm3);

    L(".2");
    mov(M_REG, N_REG);
    and_(M_REG, 16);
    je(".3", T_NEAR);
    vaddps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
    vaddps(ymm1, ymm1, ptr[X_PTR + I_REG * 4 + 32]);
    sub(I_REG, -16);

    L(".3");
    mov(M_REG, N_REG);
    and_(M_REG, 8);
    je(".10", T_NEAR);
    vaddps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
    sub(I_REG, -8);

    L(".10");
    vaddps(ymm0, ymm0, ymm1);
    SumYmm0();
    mov(M_REG, N_REG);
    and_(M_REG, 7);
    je(".12", T_NEAR);

    L(".11");
    vaddss(xmm0, xmm0, ptr[X_PTR + I_REG * 4]);
    sub(I_REG, -1);
    sub(M_REG, 1);
    jne(".11", T_NEAR);

    L(".12");
    vzeroupper();
    ret();
    func_ = getCode<sum_t>();
  }
};

constexpr uint64_t JIT_MAX_N = 256;
std::unique_ptr<SumJit1> jit1[JIT_MAX_N + 1];
sum_t jit1_func[JIT_MAX_N + 1];
std::unique_ptr<SumJit2> jit2;
sum_t jit2_func;

ATTR_CTOR(110) void init() {
  for (uint64_t i = 0; i < JIT_MAX_N + 1; ++i) {
    jit1[i].reset(new SumJit1);
    jit1[i]->Compile(i);
    jit1_func[i] = jit1[i]->func();
  }
  jit2.reset(new SumJit2);
  jit2->Compile();
  jit2_func = jit2->func();
}

}  // namespace
}  // namespace sage2

float sage2_sum_ps(uint64_t n, const float* x) {
  if (n <= sage2::JIT_MAX_N) {
    return sage2::jit1_func[n](n, x);
  }
  return sage2::jit2_func(n, x);
}
