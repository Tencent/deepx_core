// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include "simp_impl.h"

namespace deepx_core {

/************************************************************************/
/* ArithmeticSimp */
/************************************************************************/
class ArithmeticSimp : public Simp {
 public:
  ArithmeticSimp();
  bool Simplify(SimpItem* item) const override;
};

}  // namespace deepx_core
