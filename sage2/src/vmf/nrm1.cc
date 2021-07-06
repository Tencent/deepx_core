// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include <memory>
#include "internal_macro.h"
#include "xbyak_wrapper.h"

namespace sage2 {
namespace {

using nrm1_t = float (*)(uint64_t n, const float* x);
const uint32_t ABS_MASK = 0x7fffffff;

class Nrm1JitBase : public XbyakCodeGenerator {
 protected:
  const Reg64& N_REG = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& I_REG = r8;
  const Reg64& M_REG = r9;

 protected:
  nrm1_t func_ = nullptr;

 public:
  nrm1_t func() const noexcept { return func_; }
};

class Nrm1Jit1 : public Nrm1JitBase {
 private:
  void Compile1() {
    vmovss(xmm0, ptr[X_PTR]);
    vandps(xmm0, xmm0, xmm15);
  }

  void Compile2() {
    vmovss(xmm0, ptr[X_PTR]);
    vmovss(xmm1, ptr[X_PTR + 4]);
    vandps(xmm0, xmm0, xmm15);
    vandps(xmm1, xmm1, xmm15);
    vaddss(xmm0, xmm0, xmm1);
  }

  void Compile3() {
    vmovss(xmm0, ptr[X_PTR]);
    vmovss(xmm1, ptr[X_PTR + 4]);
    vmovss(xmm2, ptr[X_PTR + 8]);
    vandps(xmm0, xmm0, xmm15);
    vandps(xmm1, xmm1, xmm15);
    vandps(xmm2, xmm2, xmm15);
    vaddss(xmm0, xmm0, xmm1);
    vaddss(xmm0, xmm0, xmm2);
  }

  void Compile4() {
    vmovups(xmm0, ptr[X_PTR]);
    vandps(xmm0, xmm0, xmm15);
    SumXmm0();
  }

  void Compile5() {
    vmovups(xmm0, ptr[X_PTR]);
    vandps(xmm0, xmm0, xmm15);
    SumXmm0();
    vmovss(xmm1, ptr[X_PTR + 16]);
    vandps(xmm1, xmm1, xmm15);
    vaddss(xmm0, xmm0, xmm1);
  }

  void Compile6() {
    vmovups(xmm0, ptr[X_PTR]);
    vandps(xmm0, xmm0, xmm15);
    SumXmm0();
    vmovss(xmm1, ptr[X_PTR + 16]);
    vmovss(xmm2, ptr[X_PTR + 20]);
    vandps(xmm1, xmm1, xmm15);
    vandps(xmm2, xmm2, xmm15);
    vaddss(xmm0, xmm0, xmm1);
    vaddss(xmm0, xmm0, xmm2);
  }

  void Compile7() {
    vmovups(xmm0, ptr[X_PTR]);
    vandps(xmm0, xmm0, xmm15);
    SumXmm0();
    vmovss(xmm1, ptr[X_PTR + 16]);
    vmovss(xmm2, ptr[X_PTR + 20]);
    vmovss(xmm3, ptr[X_PTR + 24]);
    vandps(xmm1, xmm1, xmm15);
    vandps(xmm2, xmm2, xmm15);
    vandps(xmm3, xmm3, xmm15);
    vaddss(xmm0, xmm0, xmm1);
    vaddss(xmm0, xmm0, xmm2);
    vaddss(xmm0, xmm0, xmm3);
  }

  void Compile8() {
    vandps(ymm0, ymm15, ptr[X_PTR]);
    SumYmm0();
  }

  void Compile16() {
    vandps(ymm0, ymm15, ptr[X_PTR]);
    vandps(ymm1, ymm15, ptr[X_PTR + 32]);
    vaddps(ymm0, ymm0, ymm1);
    SumYmm0();
  }

  void Compile32() {
    vandps(ymm0, ymm15, ptr[X_PTR]);
    vandps(ymm1, ymm15, ptr[X_PTR + 32]);
    vandps(ymm2, ymm15, ptr[X_PTR + 64]);
    vandps(ymm3, ymm15, ptr[X_PTR + 96]);
    vaddps(ymm0, ymm0, ymm1);
    vaddps(ymm2, ymm2, ymm3);
    vaddps(ymm0, ymm0, ymm2);
    SumYmm0();
  }

  void Compile64() {
    vandps(ymm0, ymm15, ptr[X_PTR]);
    vandps(ymm1, ymm15, ptr[X_PTR + 32]);
    vandps(ymm2, ymm15, ptr[X_PTR + 64]);
    vandps(ymm3, ymm15, ptr[X_PTR + 96]);
    vandps(ymm4, ymm15, ptr[X_PTR + 128]);
    vandps(ymm5, ymm15, ptr[X_PTR + 160]);
    vandps(ymm6, ymm15, ptr[X_PTR + 192]);
    vandps(ymm7, ymm15, ptr[X_PTR + 224]);
    vaddps(ymm0, ymm0, ymm1);
    vaddps(ymm2, ymm2, ymm3);
    vaddps(ymm4, ymm4, ymm5);
    vaddps(ymm6, ymm6, ymm7);
    vaddps(ymm0, ymm0, ymm2);
    vaddps(ymm4, ymm4, ymm6);
    vaddps(ymm0, ymm0, ymm4);
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
      vandps(ymm4, ymm15, ptr[X_PTR + I_REG * 4]);
      vandps(ymm5, ymm15, ptr[X_PTR + I_REG * 4 + 32]);
      vandps(ymm6, ymm15, ptr[X_PTR + I_REG * 4 + 64]);
      vandps(ymm7, ymm15, ptr[X_PTR + I_REG * 4 + 96]);
      vaddps(ymm0, ymm0, ymm4);
      vaddps(ymm1, ymm1, ymm5);
      vaddps(ymm2, ymm2, ymm6);
      vaddps(ymm3, ymm3, ymm7);
      sub(I_REG, -32);
      sub(M_REG, 32);
      jne(".1", T_NEAR);
      vaddps(ymm0, ymm0, ymm2);
      vaddps(ymm1, ymm1, ymm3);
      offset += nn * 4;
    }

    nn = n & 16;
    if (nn) {
      vandps(ymm4, ymm15, ptr[X_PTR + offset]);
      vandps(ymm5, ymm15, ptr[X_PTR + offset + 32]);
      vaddps(ymm0, ymm0, ymm4);
      vaddps(ymm1, ymm1, ymm5);
      offset += 16 * 4;
    }

    if (n >= 16) {
      vaddps(ymm0, ymm0, ymm1);
    }

    nn = n & 8;
    if (nn) {
      vandps(ymm4, ymm15, ptr[X_PTR + offset]);
      vaddps(ymm0, ymm0, ymm4);
      offset += 8 * 4;
    }

    if (n >= 8) {
      vextractf128(xmm1, ymm0, 1);
      vaddps(xmm0, xmm0, xmm1);
    }

    nn = n & 4;
    if (nn) {
      vandps(xmm4, xmm15, ptr[X_PTR + offset]);
      vaddps(xmm0, xmm0, xmm4);
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
        vandps(xmm4, xmm4, xmm15);
        vandps(xmm5, xmm5, xmm15);
        vandps(xmm6, xmm6, xmm15);
        vaddss(xmm0, xmm0, xmm4);
        vaddss(xmm0, xmm0, xmm5);
        vaddss(xmm0, xmm0, xmm6);
        break;
      case 2:
        vmovss(xmm4, ptr[X_PTR + offset]);
        vmovss(xmm5, ptr[X_PTR + offset + 4]);
        vandps(xmm4, xmm4, xmm15);
        vandps(xmm5, xmm5, xmm15);
        vaddss(xmm0, xmm0, xmm4);
        vaddss(xmm0, xmm0, xmm5);
        break;
      case 1:
        vmovss(xmm4, ptr[X_PTR + offset]);
        vandps(xmm4, xmm4, xmm15);
        vaddss(xmm0, xmm0, xmm4);
        break;
    }
  }

 public:
  void Compile(uint64_t n) {
    mov(rax, (uint64_t)&ABS_MASK);
    vbroadcastss(ymm15, ptr[rax]);
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
    func_ = getCode<nrm1_t>();
  }
};

class Nrm1Jit2 : public Nrm1JitBase {
 public:
  void Compile() {
    mov(rax, (uint64_t)&ABS_MASK);
    vbroadcastss(ymm15, ptr[rax]);

    vxorps(ymm0, ymm0, ymm0);
    vxorps(ymm1, ymm1, ymm1);

    xor_(I_REG, I_REG);
    mov(M_REG, N_REG);
    and_(M_REG, -32);
    je(".2", T_NEAR);

    vxorps(ymm2, ymm2, ymm2);
    vxorps(ymm3, ymm3, ymm3);
    L(".1");
    vandps(ymm4, ymm15, ptr[X_PTR + I_REG * 4]);
    vandps(ymm5, ymm15, ptr[X_PTR + I_REG * 4 + 32]);
    vandps(ymm6, ymm15, ptr[X_PTR + I_REG * 4 + 64]);
    vandps(ymm7, ymm15, ptr[X_PTR + I_REG * 4 + 96]);
    vaddps(ymm0, ymm0, ymm4);
    vaddps(ymm1, ymm1, ymm5);
    vaddps(ymm2, ymm2, ymm6);
    vaddps(ymm3, ymm3, ymm7);
    sub(I_REG, -32);
    sub(M_REG, 32);
    jne(".1", T_NEAR);
    vaddps(ymm0, ymm0, ymm2);
    vaddps(ymm1, ymm1, ymm3);

    L(".2");
    mov(M_REG, N_REG);
    and_(M_REG, 16);
    je(".3", T_NEAR);
    vandps(ymm4, ymm15, ptr[X_PTR + I_REG * 4]);
    vandps(ymm5, ymm15, ptr[X_PTR + I_REG * 4 + 32]);
    vaddps(ymm0, ymm0, ymm4);
    vaddps(ymm1, ymm1, ymm5);
    sub(I_REG, -16);

    L(".3");
    mov(M_REG, N_REG);
    and_(M_REG, 8);
    je(".10", T_NEAR);
    vandps(ymm4, ymm15, ptr[X_PTR + I_REG * 4]);
    vaddps(ymm0, ymm0, ymm4);
    sub(I_REG, -8);

    L(".10");
    vaddps(ymm0, ymm0, ymm1);
    SumYmm0();
    mov(M_REG, N_REG);
    and_(M_REG, 7);
    je(".12", T_NEAR);

    L(".11");
    vmovss(xmm4, ptr[X_PTR + I_REG * 4]);
    vandps(xmm4, xmm4, xmm15);
    vaddss(xmm0, xmm0, xmm4);
    sub(I_REG, -1);
    sub(M_REG, 1);
    jne(".11", T_NEAR);

    L(".12");
    vzeroupper();
    ret();
    func_ = getCode<nrm1_t>();
  }
};

constexpr uint64_t JIT_MAX_N = 256;
std::unique_ptr<Nrm1Jit1> jit1[JIT_MAX_N + 1];
nrm1_t jit1_func[JIT_MAX_N + 1];
std::unique_ptr<Nrm1Jit2> jit2;
nrm1_t jit2_func;

ATTR_CTOR(110) void init() {
  for (uint64_t i = 0; i < JIT_MAX_N + 1; ++i) {
    jit1[i].reset(new Nrm1Jit1);
    jit1[i]->Compile(i);
    jit1_func[i] = jit1[i]->func();
  }
  jit2.reset(new Nrm1Jit2);
  jit2->Compile();
  jit2_func = jit2->func();
}

}  // namespace
}  // namespace sage2

float sage2_nrm1_ps(uint64_t n, const float* x) {
  if (n <= sage2::JIT_MAX_N) {
    return sage2::jit1_func[n](n, x);
  }
  return sage2::jit2_func(n, x);
}
