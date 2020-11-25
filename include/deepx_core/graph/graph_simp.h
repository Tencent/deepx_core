// Copyright 2020 the deepx authors.
// Author: Chuan Cheng (chuancheng@tencent.com)
//

#pragma once
#include <deepx_core/graph/graph.h>

namespace deepx_core {

// 'simp' means graph simplifier.
struct SimpConfig {
  int max_iteration = 2;
  int use_static_shape = 0;
};

// Simplify graph.
//
// Return true, 'from' has been simplified to 'to'.
// Return false, 'from' can't be simplified any more.
bool Simplify(const Graph& from, const SimpConfig& config, Graph* to);

}  // namespace deepx_core
