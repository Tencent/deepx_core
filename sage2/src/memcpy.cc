// Copyright 2020 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#include <sage2/memcpy.h>
#include <memory>
#include "internal_macro.h"
#include "xbyak_wrapper.h"

namespace sage2 {
namespace {

using memcpy_t = void* (*)(void* dst, const void* src, uint64_t n);

class MemcpyJitBase : public XbyakCodeGenerator {
 protected:
  const Reg64& DST_PTR = rdi;
  const Reg64& SRC_PTR = rsi;
  const Reg64& N_REG = rdx;
  const Reg64& I_REG = rax;
  const Reg64& M_REG = rcx;

 protected:
  memcpy_t func_ = nullptr;

 public:
  memcpy_t func() const noexcept { return func_; }
};

class MemcpyJit1 : public MemcpyJitBase {
 private:
  template <typename Int>
  const Reg8& r(Int i) const {
    switch (i) {
      case 0:
        return r8b;
      case 1:
        return r9b;
      case 2:
        return r10b;
      default:
        throw Xbyak::Error(Xbyak::ERR_BAD_PARAMETER);
    }
  }

 public:
  void CompileLE256(size_t n) {
    uint64_t nn;
    uint64_t offset;
    uint64_t ymm_offset;
    uint64_t r_offset;

    nn = n;
    offset = 0;
    ymm_offset = 0;
    r_offset = 0;
    for (;;) {
      if (nn >= 32) {
        vmovdqu(ymm(ymm_offset), ptr[SRC_PTR + offset]);
        nn -= 32;
        offset += 32;
        ymm_offset += 1;
      } else if (nn >= 16) {
        vmovdqu(xmm(ymm_offset), ptr[SRC_PTR + offset]);
        nn -= 16;
        offset += 16;
        ymm_offset += 1;
      } else if (nn >= 8) {
        vmovsd(xmm(ymm_offset), ptr[SRC_PTR + offset]);
        nn -= 8;
        offset += 8;
        ymm_offset += 1;
      } else if (nn >= 4) {
        vmovss(xmm(ymm_offset), ptr[SRC_PTR + offset]);
        nn -= 4;
        offset += 4;
        ymm_offset += 1;
      } else if (nn >= 1) {
        mov(r(r_offset), ptr[SRC_PTR + offset]);
        nn -= 1;
        offset += 1;
        r_offset += 1;
      } else {
        break;
      }
    }

    nn = n;
    offset = 0;
    ymm_offset = 0;
    r_offset = 0;
    for (;;) {
      if (nn >= 32) {
        vmovdqu(ptr[DST_PTR + offset], ymm(ymm_offset));
        nn -= 32;
        offset += 32;
        ymm_offset += 1;
      } else if (nn >= 16) {
        vmovdqu(ptr[DST_PTR + offset], xmm(ymm_offset));
        nn -= 16;
        offset += 16;
        ymm_offset += 1;
      } else if (nn >= 8) {
        vmovsd(ptr[DST_PTR + offset], xmm(ymm_offset));
        nn -= 8;
        offset += 8;
        ymm_offset += 1;
      } else if (nn >= 4) {
        vmovss(ptr[DST_PTR + offset], xmm(ymm_offset));
        nn -= 4;
        offset += 4;
        ymm_offset += 1;
      } else if (nn >= 1) {
        mov(ptr[DST_PTR + offset], r(r_offset));
        nn -= 1;
        offset += 1;
        r_offset += 1;
      } else {
        break;
      }
    }

    mov(rax, DST_PTR);
    vzeroupper();
    ret();
    func_ = getCode<memcpy_t>();
  }
};

class MemcpyJit2 : public MemcpyJitBase {
 public:
  void Compile() {
    xor_(I_REG, I_REG);
    mov(M_REG, N_REG);
    and_(M_REG, -128);
    je(".2", T_NEAR);

    L(".1");
    vmovdqu(ymm0, ptr[SRC_PTR + I_REG]);
    vmovdqu(ymm1, ptr[SRC_PTR + I_REG + 32]);
    vmovdqu(ymm2, ptr[SRC_PTR + I_REG + 64]);
    vmovdqu(ymm3, ptr[SRC_PTR + I_REG + 96]);
    vmovdqu(ptr[DST_PTR + I_REG], ymm0);
    vmovdqu(ptr[DST_PTR + I_REG + 32], ymm1);
    vmovdqu(ptr[DST_PTR + I_REG + 64], ymm2);
    vmovdqu(ptr[DST_PTR + I_REG + 96], ymm3);
    sub(I_REG, -128);
    sub(M_REG, 128);
    jne(".1", T_NEAR);

    L(".2");
    mov(M_REG, N_REG);
    and_(M_REG, 64);
    je(".3", T_NEAR);
    vmovdqu(ymm0, ptr[SRC_PTR + I_REG]);
    vmovdqu(ymm1, ptr[SRC_PTR + I_REG + 32]);
    vmovdqu(ptr[DST_PTR + I_REG], ymm0);
    vmovdqu(ptr[DST_PTR + I_REG + 32], ymm1);
    sub(I_REG, -64);

    L(".3");
    mov(M_REG, N_REG);
    and_(M_REG, 32);
    je(".4", T_NEAR);
    vmovdqu(ymm0, ptr[SRC_PTR + I_REG]);
    vmovdqu(ptr[DST_PTR + I_REG], ymm0);
    sub(I_REG, -32);

    L(".4");
    mov(M_REG, N_REG);
    and_(M_REG, 16);
    je(".5", T_NEAR);
    vmovdqu(xmm0, ptr[SRC_PTR + I_REG]);
    vmovdqu(ptr[DST_PTR + I_REG], xmm0);
    sub(I_REG, -16);

    L(".5");
    mov(M_REG, N_REG);
    and_(M_REG, 8);
    je(".10", T_NEAR);
    vmovsd(xmm0, ptr[SRC_PTR + I_REG]);
    vmovsd(ptr[DST_PTR + I_REG], xmm0);
    sub(I_REG, -8);

    L(".10");
    mov(M_REG, N_REG);
    and_(M_REG, 7);
    je(".12", T_NEAR);

    L(".11");
    mov(r8b, ptr[SRC_PTR + I_REG]);
    mov(ptr[DST_PTR + I_REG], r8b);
    sub(I_REG, -1);
    sub(M_REG, 1);
    jne(".11", T_NEAR);

    L(".12");
    mov(rax, DST_PTR);
    vzeroupper();
    ret();
    func_ = getCode<memcpy_t>();
  }
};

constexpr size_t JIT_MAX_N = 256;
std::unique_ptr<MemcpyJit1> jit1[JIT_MAX_N + 1];
memcpy_t jit1_func[JIT_MAX_N + 1];
std::unique_ptr<MemcpyJit2> jit2;
memcpy_t jit2_func;

ATTR_CTOR(110) void init() {
  for (size_t i = 0; i < JIT_MAX_N + 1; ++i) {
    jit1[i].reset(new MemcpyJit1);
    jit1[i]->CompileLE256(i);
    jit1_func[i] = jit1[i]->func();
  }
  jit2.reset(new MemcpyJit2);
  jit2->Compile();
  jit2_func = jit2->func();
}

}  // namespace
}  // namespace sage2

void* sage2_memcpy(void* dst, const void* src, size_t n) {
  if (n <= sage2::JIT_MAX_N) {
    return sage2::jit1_func[n](dst, src, n);
  }
  return sage2::jit2_func(dst, src, n);
}

#if EXPORT_MEMCPY == 1
SAGE2_C_API void* memcpy(void* dst, const void* src, size_t n) {
  if (n <= sage2::JIT_MAX_N) {
    return sage2::jit1_func[n](dst, src, n);
  }
  return sage2::jit2_func(dst, src, n);
}
#endif
