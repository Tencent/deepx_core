// Copyright 2020 the deepx authors.
// Author: Shuting Guo (tinkleguo@tencent.com)
//

#pragma once
#include "simp_impl.h"

namespace deepx_core {

/************************************************************************/
/* CSESimp */
/************************************************************************/
// 'CSE' means common subexpression elimination.
class CSESimp : public Simp {
 public:
  CSESimp();
  bool Simplify(SimpItem* item) const override;
};

}  // namespace deepx_core
