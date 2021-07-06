// Copyright 2019 the deepx authors.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//

#pragma once
// NOTE: rename Xbyak to avoid any potential linkage issues.
#define Xbyak sage2_xbyak
#include <xbyak/xbyak.h>

namespace sage2 {

/************************************************************************/
/* XbyakCodeGenerator */
/************************************************************************/
class XbyakCodeGenerator : public Xbyak::CodeGenerator {
 public:
  using Reg8 = Xbyak::Reg8;
  using Reg16 = Xbyak::Reg16;
  using Reg32 = Xbyak::Reg32;
  using Reg64 = Xbyak::Reg64;
  using Xmm = Xbyak::Xmm;
  using Ymm = Xbyak::Ymm;
  using Zmm = Xbyak::Zmm;
  using Operand = Xbyak::Operand;

 public:
  template <typename Int>
  const Ymm& ymm(Int i) const {
    switch (i) {
      case 0:
        return ymm0;
      case 1:
        return ymm1;
      case 2:
        return ymm2;
      case 3:
        return ymm3;
      case 4:
        return ymm4;
      case 5:
        return ymm5;
      case 6:
        return ymm6;
      case 7:
        return ymm7;
      case 8:
        return ymm8;
      case 9:
        return ymm9;
      case 10:
        return ymm10;
      case 11:
        return ymm11;
      case 12:
        return ymm12;
      case 13:
        return ymm13;
      case 14:
        return ymm14;
      case 15:
        return ymm15;
      default:
        throw Xbyak::Error(Xbyak::ERR_BAD_PARAMETER);
    }
  }

  template <typename Int>
  const Xmm& xmm(Int i) const {
    switch (i) {
      case 0:
        return xmm0;
      case 1:
        return xmm1;
      case 2:
        return xmm2;
      case 3:
        return xmm3;
      case 4:
        return xmm4;
      case 5:
        return xmm5;
      case 6:
        return xmm6;
      case 7:
        return xmm7;
      case 8:
        return xmm8;
      case 9:
        return xmm9;
      case 10:
        return xmm10;
      case 11:
        return xmm11;
      case 12:
        return xmm12;
      case 13:
        return xmm13;
      case 14:
        return xmm14;
      case 15:
        return xmm15;
      default:
        throw Xbyak::Error(Xbyak::ERR_BAD_PARAMETER);
    }
  }

  void SumXmm0() {
    vhaddps(xmm0, xmm0, xmm0);  // xmm0: x01, x23, ...
    vhaddps(xmm0, xmm0, xmm0);  // xmm0: x0123, ...
  }

  void SumYmm0() {
    vextractf128(xmm1, ymm0, 1);  // xmm1: x4, x5, x6, x7
    vaddps(xmm0, xmm0, xmm1);     // xmm0: x04, x15, x26, x37
    vhaddps(xmm0, xmm0, xmm0);    // xmm0: x0145, x2367, ...
    vhaddps(xmm0, xmm0, xmm0);    // xmm0: x01234567, ...
  }
};

}  // namespace sage2
