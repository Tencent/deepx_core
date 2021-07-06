// Copyright 2021 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/vmf.h>
#include <memory>
#include "internal_macro.h"
#include "xbyak_wrapper.h"

namespace sage2 {
namespace {

using max_t = float (*)(uint64_t n, const float* x);

class MaxJitBase : public XbyakCodeGenerator {
 protected:
  const Reg64& N_REG = rdi;
  const Reg64& X_PTR = rsi;
  const Reg64& I_REG = r8;
  const Reg64& M_REG = r9;

 protected:
  max_t func_ = nullptr;

 public:
  max_t func() const noexcept { return func_; }
};

class MaxJit1 : public MaxJitBase {
 protected:
  virtual void reduceps(const Xmm& xmm, const Operand& op1,
                        const Operand& op2) {
    vmaxps(xmm, op1, op2);
  }

  virtual void reducess(const Xmm& xmm, const Operand& op1,
                        const Operand& op2) {
    vmaxss(xmm, op1, op2);
  }

 private:
  void ReduceXmm0() {
    vunpckhpd(xmm1, xmm0, xmm0);  // xmm1: x2, x3, x2, x3
    reduceps(xmm0, xmm0, xmm1);   // xmm0: x02, x13, x2, x3
    vmovshdup(xmm1, xmm0);        // xmm1: x13, x13, x3, x3
    reducess(xmm0, xmm0, xmm1);   // xmm0: x0123
  }

  void ReduceYmm0() {
    vextractf128(xmm1, ymm0, 1);  // xmm1: x4, x5, x6, x7
    reduceps(xmm0, xmm0, xmm1);   // xmm0: x04, x15, x26, x37
    vunpckhpd(xmm1, xmm0, xmm0);  // xmm1: x26, x37, x26, x37
    reduceps(xmm0, xmm0, xmm1);   // xmm0: x0246, x1357, x26, x37
    vmovshdup(xmm1, xmm0);        // xmm1: x1357, x1357, x37, x37
    reducess(xmm0, xmm0, xmm1);   // xmm0: x01234567
  }

  void Compile1() { vmovss(xmm0, ptr[X_PTR]); }

  void Compile2() {
    vmovss(xmm0, ptr[X_PTR]);
    reducess(xmm0, ptr[X_PTR + 4], Operand());
  }

  void Compile3() {
    vmovss(xmm0, ptr[X_PTR]);
    reducess(xmm0, ptr[X_PTR + 4], Operand());
    reducess(xmm0, ptr[X_PTR + 8], Operand());
  }

  void Compile4() {
    vmovups(xmm0, ptr[X_PTR]);
    ReduceXmm0();
  }

  void Compile5() {
    vmovups(xmm0, ptr[X_PTR]);
    reduceps(xmm0, xmm0, ptr[X_PTR + 4]);
    ReduceXmm0();
  }

  void Compile6() {
    vmovups(xmm0, ptr[X_PTR]);
    reduceps(xmm0, xmm0, ptr[X_PTR + 8]);
    ReduceXmm0();
  }

  void Compile7() {
    vmovups(xmm0, ptr[X_PTR]);
    reduceps(xmm0, xmm0, ptr[X_PTR + 12]);
    ReduceXmm0();
  }

  void Compile8() {
    vmovups(xmm0, ptr[X_PTR]);
    reduceps(xmm0, xmm0, ptr[X_PTR + 16]);
    ReduceXmm0();
  }

  void Compile16() {
    vmovups(ymm0, ptr[X_PTR]);
    reduceps(ymm0, ymm0, ptr[X_PTR + 32]);
    ReduceYmm0();
  }

  void Compile32() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    reduceps(ymm0, ymm0, ptr[X_PTR + 64]);
    reduceps(ymm1, ymm1, ptr[X_PTR + 96]);
    reduceps(ymm0, ymm0, ymm1);
    ReduceYmm0();
  }

  void Compile64() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovups(ymm1, ptr[X_PTR + 32]);
    vmovups(ymm2, ptr[X_PTR + 64]);
    vmovups(ymm3, ptr[X_PTR + 96]);
    reduceps(ymm0, ymm0, ptr[X_PTR + 128]);
    reduceps(ymm1, ymm1, ptr[X_PTR + 160]);
    reduceps(ymm2, ymm2, ptr[X_PTR + 192]);
    reduceps(ymm3, ymm3, ptr[X_PTR + 224]);
    reduceps(ymm0, ymm0, ymm1);
    reduceps(ymm0, ymm0, ymm2);
    reduceps(ymm0, ymm0, ymm3);
    ReduceYmm0();
  }

  void CompileGT8(uint64_t n) {
    int reduce_ymm1;
    uint64_t nn;
    uint64_t offset = 0;

    if (n >= 16) {
      vmovups(ymm0, ptr[X_PTR]);
      vmovups(ymm1, ptr[X_PTR + 32]);
      sub(X_PTR, -64);
      n -= 16;
      reduce_ymm1 = 1;
    } else {
      vmovups(ymm0, ptr[X_PTR]);
      sub(X_PTR, -32);
      n -= 8;
      reduce_ymm1 = 0;
    }

    nn = n & -32;
    if (nn) {
      xor_(I_REG, I_REG);
      mov(M_REG, nn);
      vmovaps(ymm2, ymm0);
      vmovaps(ymm3, ymm0);
      L(".1");
      reduceps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
      reduceps(ymm1, ymm1, ptr[X_PTR + I_REG * 4 + 32]);
      reduceps(ymm2, ymm2, ptr[X_PTR + I_REG * 4 + 64]);
      reduceps(ymm3, ymm3, ptr[X_PTR + I_REG * 4 + 96]);
      sub(I_REG, -32);
      sub(M_REG, 32);
      jne(".1", T_NEAR);
      reduceps(ymm0, ymm0, ymm2);
      reduceps(ymm0, ymm0, ymm3);
      offset += nn * 4;
    }

    nn = n & 16;
    if (nn) {
      reduceps(ymm0, ymm0, ptr[X_PTR + offset]);
      reduceps(ymm1, ymm1, ptr[X_PTR + offset + 32]);
      offset += 16 * 4;
    }

    nn = n & 8;
    if (nn) {
      reduceps(ymm0, ymm0, ptr[X_PTR + offset]);
      offset += 8 * 4;
    }

    nn = n & 7;
    if (nn) {
      offset -= (32 - nn * 4);
      reduceps(ymm0, ymm0, ptr[X_PTR + offset]);
    }

    if (reduce_ymm1) {
      reduceps(ymm0, ymm0, ymm1);
    }
    ReduceYmm0();
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
        CompileGT8(n);
        break;
    }
    vzeroupper();
    ret();
    func_ = getCode<max_t>();
  }
};

class MinJit1 : public MaxJit1 {
 protected:
  void reduceps(const Xmm& xmm, const Operand& op1,
                const Operand& op2) override {
    vminps(xmm, op1, op2);
  }

  void reducess(const Xmm& xmm, const Operand& op1,
                const Operand& op2) override {
    vminss(xmm, op1, op2);
  }
};

class MaxJit2 : public MaxJitBase {
 protected:
  virtual void reduceps(const Xmm& xmm, const Operand& op1,
                        const Operand& op2) {
    vmaxps(xmm, op1, op2);
  }

  virtual void reducess(const Xmm& xmm, const Operand& op1,
                        const Operand& op2) {
    vmaxss(xmm, op1, op2);
  }

 private:
  void ReduceYmm0() {
    vextractf128(xmm1, ymm0, 1);
    reduceps(xmm0, xmm0, xmm1);
    vunpckhpd(xmm1, xmm0, xmm0);
    reduceps(xmm0, xmm0, xmm1);
    vmovshdup(xmm1, xmm0);
    reducess(xmm0, xmm0, xmm1);
  }

 public:
  void CompileGT8() {
    vmovups(ymm0, ptr[X_PTR]);
    vmovaps(ymm1, ymm0);
    sub(X_PTR, -32);
    sub(N_REG, 8);

    xor_(I_REG, I_REG);
    mov(M_REG, N_REG);
    and_(M_REG, -32);
    je(".2", T_NEAR);

    vmovaps(ymm2, ymm0);
    vmovaps(ymm3, ymm0);
    L(".1");
    reduceps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
    reduceps(ymm1, ymm1, ptr[X_PTR + I_REG * 4 + 32]);
    reduceps(ymm2, ymm2, ptr[X_PTR + I_REG * 4 + 64]);
    reduceps(ymm3, ymm3, ptr[X_PTR + I_REG * 4 + 96]);
    sub(I_REG, -32);
    sub(M_REG, 32);
    jne(".1", T_NEAR);
    reduceps(ymm0, ymm0, ymm2);
    reduceps(ymm0, ymm0, ymm3);

    L(".2");
    mov(M_REG, N_REG);
    and_(M_REG, 16);
    je(".3", T_NEAR);
    reduceps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
    reduceps(ymm1, ymm1, ptr[X_PTR + I_REG * 4 + 32]);
    sub(I_REG, -16);

    L(".3");
    mov(M_REG, N_REG);
    and_(M_REG, 8);
    je(".10", T_NEAR);
    reduceps(ymm0, ymm0, ptr[X_PTR + I_REG * 4]);
    sub(I_REG, -8);

    L(".10");
    mov(M_REG, N_REG);
    and_(M_REG, 7);
    je(".11", T_NEAR);
    add(I_REG, M_REG);
    reduceps(ymm0, ymm0, ptr[X_PTR + I_REG * 4 - 32]);

    L(".11");
    reduceps(ymm0, ymm0, ymm1);
    ReduceYmm0();
    vzeroupper();
    ret();
    func_ = getCode<max_t>();
  }
};

class MinJit2 : public MaxJit2 {
 protected:
  void reduceps(const Xmm& xmm, const Operand& op1,
                const Operand& op2) override {
    vminps(xmm, op1, op2);
  }

  void reducess(const Xmm& xmm, const Operand& op1,
                const Operand& op2) override {
    vminss(xmm, op1, op2);
  }
};

constexpr uint64_t JIT_MAX_N = 256;
std::unique_ptr<MaxJit1> max_jit1[JIT_MAX_N + 1];
max_t max_jit1_func[JIT_MAX_N + 1];
std::unique_ptr<MaxJit2> max_jit2;
max_t max_jit2_func;

std::unique_ptr<MinJit1> min_jit1[JIT_MAX_N + 1];
max_t min_jit1_func[JIT_MAX_N + 1];
std::unique_ptr<MinJit2> min_jit2;
max_t min_jit2_func;

ATTR_CTOR(110) void init() {
  for (uint64_t i = 0; i < JIT_MAX_N + 1; ++i) {
    max_jit1[i].reset(new MaxJit1);
    max_jit1[i]->Compile(i);
    max_jit1_func[i] = max_jit1[i]->func();
  }
  max_jit2.reset(new MaxJit2);
  max_jit2->CompileGT8();
  max_jit2_func = max_jit2->func();

  for (uint64_t i = 0; i < JIT_MAX_N + 1; ++i) {
    min_jit1[i].reset(new MinJit1);
    min_jit1[i]->Compile(i);
    min_jit1_func[i] = min_jit1[i]->func();
  }
  min_jit2.reset(new MinJit2);
  min_jit2->CompileGT8();
  min_jit2_func = min_jit2->func();
}

}  // namespace
}  // namespace sage2

float sage2_max_ps(uint64_t n, const float* x) {
  if (n <= sage2::JIT_MAX_N) {
    return sage2::max_jit1_func[n](n, x);
  }
  return sage2::max_jit2_func(n, x);
}

float sage2_min_ps(uint64_t n, const float* x) {
  if (n <= sage2::JIT_MAX_N) {
    return sage2::min_jit1_func[n](n, x);
  }
  return sage2::min_jit2_func(n, x);
}
