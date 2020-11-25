// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include "simp_impl.h"

namespace deepx_core {

/************************************************************************/
/* CFConfig */
/************************************************************************/
// 'CF' means constant folding.
struct CFConfig {
  int use_static_shape = 0;
  int max_constant_bytes = 10 * 1024 * 1024;  // magic number
};

/************************************************************************/
/* CFSimp */
/************************************************************************/
class CFSimp : public Simp {
 private:
  const CFConfig config_;

 public:
  explicit CFSimp(const CFConfig& config);
  bool Simplify(SimpItem* item) const override;
};

}  // namespace deepx_core
